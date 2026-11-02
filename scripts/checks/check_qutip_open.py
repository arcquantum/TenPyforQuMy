"""
scripts/checks/check_qutip_open.py

Cross-checks the exact-M and exact-Q matrices from src/dM.py (applied
per trajectory and averaged over an ensemble) against the exact
Lindblad evolution from QuTiP, at N=8, alpha=3, open 1D, for two
Scharnagl twist axes:

    (theta_k, phi_k) = (0, 0)             -- two-axis case
    (theta_k, phi_k) = (pi/4, pi/4)       -- three-axis case

The QuTiP side computes M by central finite differences on the
density matrix, using the exact Lindblad solver:
    M_ij = (Tr[S_j rho_+] - Tr[S_j rho_-]) / (2 delta)

The MPS side computes M per trajectory with the exact formula
    M_ij^(j) = -2 Im[ <u_i^(j)| S_j |psi_0^(j)> ]
and averages over N_traj trajectories. Q is formed from the
trajectory-averaged moments.

Tolerances: 1e-6 at gamma=0 (deterministic), 1e-3 at gamma>0.
"""

from __future__ import annotations
import importlib.util
import os
import sys
import numpy as np
import scipy.linalg
import qutip


ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))


from src.models import PowerLawOATModel
from src.evolution import initialize_css, apply_collective_rotation
from src.observables import _get_3x3_covariance
from src.tjm import (
    apply_dissipative_half_step,
    evolve_coherent_step,
    maybe_apply_jump,
)


# Import the QuTiP Lindblad worker helpers.
_qtw_path = os.path.join(ROOT_DIR, 'scripts', 'workers', 'open_1d', 'exact_lindblad_worker_1d.py')
_spec = importlib.util.spec_from_file_location(
    'exact_lindblad_worker_1d', _qtw_path,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load module from {_qtw_path}")
_qtw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qtw)


AXES = ['x', 'y', 'z']


def k_axis_from_angles(theta_k, phi_k):
    return np.array([
        np.sin(theta_k) * np.cos(phi_k),
        np.sin(theta_k) * np.sin(phi_k),
        np.cos(theta_k),
    ])


# ---------------------------------------------------------------------
# QuTiP side
# ---------------------------------------------------------------------
def compute_qutip_MQ(N, J, alpha, mu, nu, gamma1, gamma2,
                     theta_k, phi_k, delta=1e-5):
    """Exact M and Q on the QuTiP side using the Lindblad solver."""
    H = _qtw.build_hamiltonian(N, J, alpha)
    S_ops = _qtw.build_collective_spin_ops(N)
    rho_0 = _qtw.build_css_dm(N)
    c_ops = _qtw.build_jump_ops(N, gamma1, gamma2)
    R_k = _qtw.build_R_k(theta_k, phi_k, N)

    t_fwd = mu / 2.0
    t_bwd = (nu - mu) / 2.0

    # Forward evolution, then (without signal) build rho_pre.
    rho_pre = _qtw.lindblad_evolve(rho_0, H, c_ops, t_fwd)

    # Q on the final state at phi=0
    rho_post = _qtw.evolve_bwd_with_k(rho_pre, H, c_ops, t_bwd, R_k)
    S_mean = np.array(
        [np.real(qutip.expect(S_ops[a], rho_post)) for a in AXES]
    )
    Q = np.zeros((3, 3), dtype=complex)
    for i, ai in enumerate(AXES):
        for j, aj in enumerate(AXES):
            corr = qutip.expect(S_ops[ai] * S_ops[aj], rho_post)
            Q[i, j] = corr - S_mean[i] * S_mean[j]
    Q = np.real(0.5 * (Q + Q.conj().T))

    # M via central finite differences on phi about each axis.
    M = np.zeros((3, 3))
    for i, ai in enumerate(AXES):
        S_n = S_ops[ai]
        U_plus = (-1j * (+delta) * S_n).expm()
        U_minus = (-1j * (-delta) * S_n).expm()

        rho_imp_plus = U_plus * rho_pre * U_plus.dag()
        rho_imp_minus = U_minus * rho_pre * U_minus.dag()

        rho_plus = _qtw.evolve_bwd_with_k(rho_imp_plus, H, c_ops,
                                          t_bwd, R_k)
        rho_minus = _qtw.evolve_bwd_with_k(rho_imp_minus, H, c_ops,
                                           t_bwd, R_k)

        for j, aj in enumerate(AXES):
            vp = np.real(qutip.expect(S_ops[aj], rho_plus))
            vm = np.real(qutip.expect(S_ops[aj], rho_minus))
            M[i, j] = (vp - vm) / (2.0 * delta)

    return M, Q


# ---------------------------------------------------------------------
# MPS side: single trajectory
# ---------------------------------------------------------------------
def _mps_trajectory_evolution(psi, model, total_time, theta_k, phi_k,
                              gamma1, gamma2, dt_max, chi_max, rng):
    """
    Evolve one MPS trajectory for a signed total_time with Scharnagl
    twist about k = (theta_k, phi_k).

    At gamma1 = gamma2 = 0 (no dissipation, no jumps) this delegates
    to src.evolution.evolve_mps_time, which batches all time steps
    into a single engine call. That is ~10x faster than calling
    evolve_coherent_step once per step. For gamma > 0 the dissipative
    and jump sub-steps must be interleaved, so we fall back to the
    step-by-step loop.
    """
    if abs(total_time) < 1e-15:
        return

    k = k_axis_from_angles(theta_k, phi_k)
    apply_k_rotation = not np.allclose(k, [0.0, 0.0, 1.0])

    theta = np.arccos(np.clip(k[2], -1.0, 1.0))
    phi = np.arctan2(k[1], k[0])

    from src.evolution import (
        apply_single_site_rotation, evolve_mps_time,
    )

    if apply_k_rotation:
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'z', -phi)
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'y', -theta)

    inner_time = total_time / 2.0

    if gamma1 <= 0.0 and gamma2 <= 0.0:
        # No dissipation: batch all steps in one engine call.
        evolve_mps_time(psi, model, inner_time, chi_max, dt_max)
    else:
        N_steps = max(1, int(np.ceil(abs(inner_time) / dt_max)))
        dt = inner_time / N_steps
        dt_abs = abs(dt)
        for _ in range(N_steps):
            apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
            evolve_coherent_step(psi, model, dt, chi_max)
            apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
            maybe_apply_jump(psi, gamma1, gamma2, dt_abs, rng)

    if apply_k_rotation:
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'y', +theta)
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'z', +phi)


def _build_collective_spin_MPO(model, axis):
    from tenpy.models.model import CouplingModel
    cm = CouplingModel(model.lat)
    cm.add_onsite(1.0, 0, f'S{axis}')
    return cm.calc_H_MPO()


def _apply_collective_spin(psi, model, axis, chi_max):
    mpo = _build_collective_spin_MPO(model, axis)
    out = psi.copy()
    mpo.apply(out, options={
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-14},
    })
    return out


def _trajectory_M(psi_pre, model, nu_minus_mu, theta_k, phi_k,
                  gamma1, gamma2, dt_max, chi_max, rng, delta=1e-5):
    """Exact M for one trajectory."""
    # psi_0 = T_k(psi_pre) for the covariance; u_i = T_k S_i psi_pre.
    psi_0 = psi_pre.copy()
    _mps_trajectory_evolution(psi_0, model, nu_minus_mu,
                              theta_k, phi_k,
                              gamma1, gamma2, dt_max, chi_max, rng)

    u = {}
    for a in AXES:
        ua = _apply_collective_spin(psi_pre, model, a, chi_max)
        _mps_trajectory_evolution(ua, model, nu_minus_mu,
                                  theta_k, phi_k,
                                  gamma1, gamma2, dt_max, chi_max, rng)
        u[a] = ua

    M = np.zeros((3, 3))
    for i, ai in enumerate(AXES):
        for j, aj in enumerate(AXES):
            Sj_psi0 = _apply_collective_spin(psi_0, model, aj, chi_max)
            val = u[ai].overlap(Sj_psi0)
            M[i, j] = -2.0 * complex(val).imag
    return M


def _trajectory_Q(psi_q):
    _, Cov = _get_3x3_covariance(psi_q)
    S_vec = np.array([
        np.real(np.sum(psi_q.expectation_value('Sx'))),
        np.real(np.sum(psi_q.expectation_value('Sy'))),
        np.real(np.sum(psi_q.expectation_value('Sz'))),
    ])
    return S_vec, Cov


def compute_mps_MQ(N, J, alpha, mu, nu, gamma1, gamma2,
                   theta_k, phi_k, chi_max, dt_max,
                   n_traj, seed):
    """Exact M and Q on the MPS side via trajectory averaging."""
    model = PowerLawOATModel({
        'N': N, 'J': J, 'alpha': alpha, 'conserve': None,
        'compression_method': 'exponential',
    })

    rng_master = np.random.default_rng(seed)
    nu_minus_mu = nu - mu

    M_accum = np.zeros((3, 3))
    SScorr_accum = np.zeros((3, 3))
    S_vec_accum = np.zeros(3)

    for t in range(n_traj):
        print(f"    trajectory {t + 1}/{n_traj}", flush=True)
        rng = np.random.default_rng(
            rng_master.integers(0, 2 ** 32 - 1)
        )
        psi_pre = initialize_css(model.lat)
        # The forward twist is always a plain z-twist; only the
        # second twisting acts about k. The (0, 0) arguments below
        # enforce that.
        _mps_trajectory_evolution(psi_pre, model, mu,
                                  0.0, 0.0,
                                  gamma1, gamma2, dt_max, chi_max, rng)

        M_traj = _trajectory_M(psi_pre, model, nu_minus_mu,
                               theta_k, phi_k,
                               gamma1, gamma2, dt_max, chi_max,
                               rng)

        # Final state for Q (without signal)
        rng_q = np.random.default_rng(
            rng_master.integers(0, 2 ** 32 - 1)
        )
        psi_q = psi_pre.copy()
        _mps_trajectory_evolution(psi_q, model, nu_minus_mu,
                                  theta_k, phi_k,
                                  gamma1, gamma2, dt_max, chi_max,
                                  rng_q)
        S_vec, Cov = _trajectory_Q(psi_q)
        SScorr = Cov + np.outer(S_vec, S_vec)

        M_accum += M_traj
        SScorr_accum += SScorr
        S_vec_accum += S_vec

    M = M_accum / n_traj
    SScorr_avg = SScorr_accum / n_traj
    S_mean = S_vec_accum / n_traj
    Q = SScorr_avg - np.outer(S_mean, S_mean)
    Q = 0.5 * (Q + Q.T)
    return M, Q


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_case(N, J, alpha, mu, nu, gamma1, gamma2,
             theta_k, phi_k, chi_max, dt_max,
             n_traj, seed, tol):
    print(f"\n--- gamma=({gamma1:.1e}, {gamma2:.1e}), "
          f"(theta_k, phi_k) = ({theta_k:.4f}, {phi_k:.4f}) ---")

    print("  computing QuTiP exact M, Q ...", flush=True)
    M_q, Q_q = compute_qutip_MQ(N, J, alpha, mu, nu,
                                gamma1, gamma2, theta_k, phi_k)

    print(f"  computing MPS exact M, Q (n_traj={n_traj}) ...",
          flush=True)
    M_m, Q_m = compute_mps_MQ(N, J, alpha, mu, nu,
                              gamma1, gamma2, theta_k, phi_k,
                              chi_max, dt_max, n_traj, seed)

    err_M = float(np.linalg.norm(M_m - M_q, 'fro'))
    err_Q = float(np.linalg.norm(Q_m - Q_q, 'fro'))

    ok_M = err_M < tol
    ok_Q = err_Q < tol

    print(f"  ||dM||_F = {err_M:.3e}  (tol {tol:.0e})  "
          f"[{'PASS' if ok_M else 'FAIL'}]")
    print(f"  ||dQ||_F = {err_Q:.3e}  (tol {tol:.0e})  "
          f"[{'PASS' if ok_Q else 'FAIL'}]")

    return ok_M and ok_Q


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-traj', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-nonzero', action='store_true',
                        help='Only run the deterministic gamma=0 cases.')
    args = parser.parse_args()

    N = 8
    J = 1.0
    alpha = 3.0
    mu = np.pi / 2
    nu = -np.pi / 2
    chi_max = 32
    # dt_max = 0.005 rather than 0.01. The MPS-vs-QuTiP comparison has
    # no shared Taylor truncation to cancel, so the full O(dt^2) error
    # of ExpMPOEvolution shows up in the residual: ~1.7e-6 at dt=0.01.
    # Halving dt reduces it by 4x to ~4.3e-7, safely below 1e-6.
    dt_max = 0.005

    print("=" * 70)
    print("check_qutip_open.py: MPS exact-M/Q vs Lindblad reference")
    print("=" * 70)
    print(f"  N={N}, alpha={alpha}, J={J}, "
          f"mu={mu:.4f}, nu={nu:.4f}")
    print(f"  chi_max={chi_max}, dt_max={dt_max}, "
          f"n_traj={args.n_traj}, seed={args.seed}")

    results = {}

    # ---- deterministic: gamma = 0, tol = 1e-6 ----
    for (theta_k, phi_k) in [(0.0, 0.0), (np.pi / 4, np.pi / 4)]:
        results[(0.0, 0.0, theta_k, phi_k)] = run_case(
            N, J, alpha, mu, nu, 0.0, 0.0,
            theta_k, phi_k, chi_max, dt_max,
            n_traj=1, seed=args.seed, tol=1e-6,
        )

    if args.skip_nonzero:
        pass
    else:
        # ---- noisy: gamma > 0, tol = 1e-3 ----
        for (theta_k, phi_k) in [(0.0, 0.0), (np.pi / 4, np.pi / 4)]:
            results[(1e-3, 1e-3, theta_k, phi_k)] = run_case(
                N, J, alpha, mu, nu, 1e-3, 1e-3,
                theta_k, phi_k, chi_max, dt_max,
                n_traj=args.n_traj, seed=args.seed, tol=1e-3,
            )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    all_pass = True
    for k, ok in results.items():
        g1, g2, tk, pk = k
        label = (f"gamma=({g1:.1e},{g2:.1e}) "
                 f"(theta_k, phi_k)=({tk:.4f},{pk:.4f})")
        print(f"  {label:<55} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n[PASS] check_qutip_open.py")
        sys.exit(0)
    print("\n[FAIL] check_qutip_open.py")
    sys.exit(1)


if __name__ == "__main__":
    main()