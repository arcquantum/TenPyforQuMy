"""
scripts/workers/open_2d/scharnagl_echo_2d_open.py

Scharnagl three-axis echo worker for the 2D open-system cylinder
power-law chain, using the Tensor Jump Method.

Protocol
--------
Identical to tjm_worker.py --mode echo --Lx --Ly, with the second
twisting applied via the k-axis rotation wrapper
evolve_trajectory_with_k.

--method flag
-------------
finite_diff      (default) central finite differences with common
                random numbers between the +delta and -delta
                branches.
exact_derivative analytic per-trajectory M via src.dM.exact_M.
                The Scharnagl worker passes
                    nu_minus_mu = nu - mu
                to exact_M. The four backward evolutions inside
                exact_M (psi_0 and the three u_i) share a single
                random seed so that the per-trajectory analytical
                identity holds for that trajectory.

Output
------
data/scharnagl_pts_2d_open/*.npy with keys
    {Lx, Ly, alpha, J, mu, nu, theta_k, phi_k, gamma1, gamma2,
     n_traj, snr, n_opt, m_opt, S_intermediate, method}
"""

import argparse
import os
import sys
import warnings

import numpy as np
import scipy.linalg

warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

from src.models import PowerLawCylinderModel
from src.evolution import (
    initialize_css,
    apply_collective_rotation,
    apply_single_site_rotation,
)
from src.observables import _get_3x3_covariance
from src.tjm import (
    apply_dissipative_half_step,
    evolve_coherent_step,
    maybe_apply_jump,
)


def k_axis_from_angles(theta_k, phi_k):
    return np.array([
        np.sin(theta_k) * np.cos(phi_k),
        np.sin(theta_k) * np.sin(phi_k),
        np.cos(theta_k),
    ])


def evolve_trajectory_with_k(psi, model, total_time, k_axis,
                             gamma1, gamma2, dt_max, chi_max, rng):
    """
    Evolve one trajectory under T_k. The actual inner coherent
    evolution time is ``total_time / 2.0``.
    """
    if abs(total_time) < 1e-15:
        return

    k = np.asarray(k_axis, dtype=float)
    k = k / np.linalg.norm(k)
    apply_k_rotation = not np.allclose(k, [0.0, 0.0, 1.0])

    theta_k = np.arccos(np.clip(k[2], -1.0, 1.0))
    phi_k = np.arctan2(k[1], k[0])

    if apply_k_rotation:
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'z', -phi_k)
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'y', -theta_k)

    inner_time = total_time / 2.0
    N_steps = int(np.ceil(abs(inner_time) / dt_max))
    if N_steps < 1:
        N_steps = 1
    dt = inner_time / N_steps
    dt_abs = abs(dt)
    for _ in range(N_steps):
        apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
        evolve_coherent_step(psi, model, dt, chi_max)
        apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
        maybe_apply_jump(psi, gamma1, gamma2, dt_abs, rng)

    if apply_k_rotation:
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'y', +theta_k)
        for i in range(psi.L):
            apply_single_site_rotation(psi, i, 'z', +phi_k)


def trajectory_covariance(psi):
    _, Cov = _get_3x3_covariance(psi)
    S_vec = np.array([
        np.real(np.sum(psi.expectation_value('Sx'))),
        np.real(np.sum(psi.expectation_value('Sy'))),
        np.real(np.sum(psi.expectation_value('Sz'))),
    ])
    return S_vec, Cov


def _norm(psi):
    return float(np.sqrt(np.real(psi.overlap(psi))))


def _restore_norm(psi, target_norm):
    current_norm = float(np.sqrt(np.real(psi.overlap(psi))))
    if current_norm > 1e-15 and abs(current_norm - target_norm) > 1e-15:
        psi._B[0] = psi._B[0] * (target_norm / current_norm)


def extract_optimal_axes(M, Q):
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > max(1e-12, 1e-8 * evals.max())
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T
    F = M @ Q_inv_sqrt
    U, sv, Vh = scipy.linalg.svd(F, full_matrices=False)
    snr = float(sv[0])
    n_opt = np.real(U[:, 0])
    m_opt = np.real(Q_inv_sqrt @ Vh[0, :])
    return snr, n_opt, m_opt


def get_M_exact(psi_base, model, chi_max, nu_minus_mu,
                theta_k, phi_k,
                gamma1, gamma2, dt_max, rng):
    from src.dM import exact_M

    if abs(theta_k) < 1e-15 and abs(phi_k) < 1e-15:
        k_axis = None
    else:
        k_axis = k_axis_from_angles(theta_k, phi_k)

    branch_seed = int(rng.integers(0, 2 ** 32 - 1))

    def evolve_fn(psi, strength, k_axis_arg):
        original_norm = _norm(psi)
        branch_rng = np.random.default_rng(branch_seed)
        evolve_trajectory_with_k(psi, model, strength, k_axis_arg,
                                 gamma1, gamma2, dt_max, chi_max,
                                 branch_rng)
        _restore_norm(psi, original_norm)

    M, _psi_q = exact_M(
        psi_base, model,
        k_axis=k_axis,
        nu_minus_mu=nu_minus_mu,
        chi_max=chi_max,
        dt_max=dt_max,
        evolve_fn=evolve_fn,
    )
    return M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lx", type=int, required=True)
    parser.add_argument("--Ly", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--theta_k", type=float, required=True)
    parser.add_argument("--phi_k", type=float, required=True)
    parser.add_argument("--chi_max", type=int, default=32)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--n_traj", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method",
                        choices=['finite_diff', 'exact_derivative'],
                        default='exact_derivative')
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()


    # --- (mu1, mu2) translation (migration) ---
    if getattr(args, "mu1", None) is not None and args.mu is None:
        _N = getattr(args, "N", None) or ((getattr(args, "Lx", 0) or 0) * (getattr(args, "Ly", 0) or 0))
        args.mu = _N * args.mu1 / 2.0
        args.nu = _N * (args.mu1 - args.mu2) / 2.0
        args.convention = "mu1mu2"
    elif args.mu is not None and getattr(args, "mu1", None) is None:
        args.convention = "internal"
    else:
        raise SystemExit("Provide either --mu --nu or --mu1 --mu2")
    args.mu_suffix = (f"_mu1{args.mu1:.4f}_mu2{args.mu2:.4f}"
                      if args.convention == "mu1mu2"
                      else f"_mu{args.mu:.4f}_nu{args.nu:.4f}")
    # --- End (mu1, mu2) translation ---
    if not (0.0 <= args.theta_k <= np.pi):
        raise SystemExit(f"--theta_k must be in [0, pi]; "
                         f"got {args.theta_k}.")

    if args.task_id is not None:
        fname = f"scharnagl_pt_{args.task_id}.npy"
    else:
        fname = (f"scharnagl_pt_Lx{args.Lx}_Ly{args.Ly}"
                 f"_alpha{args.alpha}"
                 f"{args.mu_suffix}"
                 f"_tk{args.theta_k:.4f}_pk{args.phi_k:.4f}"
                 f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
                 f"_ntraj{args.n_traj}.npy")

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'scharnagl_pts_2d_open')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, fname)

    if os.path.exists(out_file):
        print(f"Already completed: {out_file}", flush=True)
        return

    model = PowerLawCylinderModel({
        'Lx': args.Lx, 'Ly': args.Ly,
        'J': args.J, 'alpha': args.alpha, 'conserve': None,
    })

    if args.method == 'exact_derivative':
        from src.dM import exact_M  # noqa: F401

    rng_master = np.random.default_rng(args.seed)
    nu_minus_mu = args.nu - args.mu
    k_axis = k_axis_from_angles(args.theta_k, args.phi_k)
    k_z = np.array([0.0, 0.0, 1.0])

    M_accum = np.zeros((3, 3))
    SScorr_accum = np.zeros((3, 3))
    S_vec_accum = np.zeros(3)
    qfi_avg_accum = 0.0
    S_int_accum = 0.0
    S_int_count = 0

    n_traj = args.n_traj
    for t in range(n_traj):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        step_size = max(1, n_traj // 10)
        if t == 0 or (t + 1) % step_size == 0 or (t + 1) == n_traj:
            print(f"  trajectory {t + 1}/{n_traj}", flush=True)

        psi_base = initialize_css(model.lat)
        evolve_trajectory_with_k(
            psi_base, model, args.mu, k_z,
            args.gamma1, args.gamma2, args.dt_max,
            args.chi_max, rng,
        )
        try:
            S_int_accum += float(max(psi_base.entanglement_entropy()))
            S_int_count += 1
        except Exception:
            pass

        if args.method == 'exact_derivative':
            M_traj = get_M_exact(
                psi_base, model, args.chi_max, nu_minus_mu,
                args.theta_k, args.phi_k,
                args.gamma1, args.gamma2, args.dt_max, rng,
            )
        else:
            delta = 1e-3
            M_traj = np.zeros((3, 3))
            for i, axis in enumerate(['x', 'y', 'z']):
                branch_seed = int(rng.integers(0, 2**32 - 1))
                for sign in (+1, -1):
                    branch_rng = np.random.default_rng(branch_seed)
                    psi_copy = psi_base.copy()
                    apply_collective_rotation(psi_copy, axis, sign * delta)
                    evolve_trajectory_with_k(
                        psi_copy, model, nu_minus_mu, k_axis,
                        args.gamma1, args.gamma2, args.dt_max,
                        args.chi_max, branch_rng,
                    )
                    means = [
                        np.real(np.sum(psi_copy.expectation_value(a)))
                        for a in ['Sx', 'Sy', 'Sz']
                    ]
                    for j in range(3):
                        M_traj[i, j] += sign * means[j] / (2.0 * delta)

        psi_q = psi_base.copy()
        evolve_trajectory_with_k(
            psi_q, model, nu_minus_mu, k_axis,
            args.gamma1, args.gamma2, args.dt_max,
            args.chi_max, rng,
        )
        S_vec, Cov = trajectory_covariance(psi_q)
        SScorr = Cov + np.outer(S_vec, S_vec)

        M_accum += M_traj
        SScorr_accum += SScorr
        S_vec_accum += S_vec
        qfi_avg_accum += 4.0 * float(np.max(np.linalg.eigvalsh(Cov)))

    M = M_accum / n_traj
    SScorr_avg = SScorr_accum / n_traj
    S_mean = S_vec_accum / n_traj
    Q = SScorr_avg - np.outer(S_mean, S_mean)
    Q = 0.5 * (Q + Q.T)
    qfi_var = 4.0 * float(np.max(np.linalg.eigvalsh(Q)))
    qfi_avg = float(qfi_avg_accum / n_traj)
    S_intermediate = (S_int_accum / S_int_count
                      if S_int_count > 0 else float('nan'))

    snr, n_opt, m_opt = extract_optimal_axes(M, Q)

    np.save(out_file, {
        'Lx': args.Lx, 'Ly': args.Ly,
        'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu, 'nu': args.nu,
        'theta_k': args.theta_k, 'phi_k': args.phi_k,
        'gamma1': args.gamma1, 'gamma2': args.gamma2,
        'n_traj': n_traj,
        'snr': snr,
        'qfi_var': qfi_var, 'qfi_avg': qfi_avg,
        'n_opt': n_opt, 'm_opt': m_opt,
        'S_intermediate': S_intermediate,
        'method': args.method,
    })
    print(f"Scharnagl 2D open SNR = {snr:.6f} "
          f"(Lx={args.Lx}, Ly={args.Ly}, alpha={args.alpha}, "
          f"mu={args.mu:.4f}, nu={args.nu:.4f}, "
          f"theta_k={args.theta_k:.4f}, phi_k={args.phi_k:.4f}, "
          f"g1={args.gamma1:.4e}, g2={args.gamma2:.4e}, "
          f"n_traj={n_traj}, method={args.method})", flush=True)


if __name__ == "__main__":
    main()