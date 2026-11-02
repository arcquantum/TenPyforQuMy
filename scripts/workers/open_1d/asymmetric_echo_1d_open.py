"""
scripts/workers/open_1d/asymmetric_echo_1d_open.py

Open-system asymmetric echo worker for the 1D power-law chain, using
the Tensor Jump Method of the documentation.

Protocol
--------
   |psi(phi)> = e^{+i H t_-} e^{-i phi S_n} e^{-i H t_+} |+>^N

with t_plus, t_minus >= 0. The backward evolution is time-reversed;
the dissipator and jumps always run forward in real time.

Convention
----------
Code-native mapping to (mu, nu):
   t_plus  = mu / 2
   t_minus = (mu - nu) / 2
No additional chi factor is applied.

--method flag
-------------
finite_diff      (default) central finite differences with common
                random numbers between the +delta and -delta
                branches.
exact_derivative analytic per-trajectory M via src.dM.exact_M.
                The asymmetric worker passes
                    nu_minus_mu = -2 * t_minus
                to exact_M. The four backward evolutions inside
                exact_M (psi_0 and the three u_i) share a single
                random seed so that the per-trajectory analytical
                identity
                    M_ij = -2 Im[<u_i|S_j|psi_0>]
                holds for that trajectory.

Output
------
A .npy file in data/echo_pts_1d_open_asym/ with keys
   {N, alpha, J, t_plus, t_minus, gamma1, gamma2, n_traj, snr,
    method}
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


from src.models import PowerLawOATModel
from src.evolution import initialize_css, apply_collective_rotation
from src.observables import _get_3x3_covariance
from src.tjm import (
    apply_dissipative_half_step,
    evolve_coherent_step,
    maybe_apply_jump,
)


def evolve_trajectory(psi, model, total_time, gamma1, gamma2,
                      dt_max, chi_max, rng):
    """Evolve one trajectory for a signed total_time."""
    if abs(total_time) < 1e-15:
        return
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt = total_time / N_steps
    dt_abs = abs(dt)
    for _ in range(N_steps):
        apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
        evolve_coherent_step(psi, model, dt, chi_max)
        apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
        maybe_apply_jump(psi, gamma1, gamma2, dt_abs, rng)


def trajectory_covariance(psi):
    _, Cov = _get_3x3_covariance(psi)
    S_vec = np.array([
        np.real(np.sum(psi.expectation_value('Sx'))),
        np.real(np.sum(psi.expectation_value('Sy'))),
        np.real(np.sum(psi.expectation_value('Sz'))),
    ])
    return S_vec, Cov


def _norm(psi):
    """Return ||psi|| as a float (via the raw inner product)."""
    return float(np.sqrt(np.real(psi.overlap(psi))))


def _restore_norm(psi, target_norm):
    """
    Rescale the MPS so that ||psi|| == target_norm.

    The dissipative half-steps in apply_dissipative_half_step call
    apply_local_op(..., renormalize=True), which forces the state
    norm back to 1. The exact-M formula, however, requires the
    auxiliary states |u_i> = T_2 S_i |psi_pre> to keep their natural
    norms ||S_i psi_pre||, so we rescale back here.

    The rescaling is implemented by multiplying the first-site
    tensor, which is guaranteed to exist in every TeNPy version.
    psi.overlap(psi) is used to measure the current state norm;
    this bypasses any version-specific norm bookkeeping.
    """
    current_norm = float(np.sqrt(np.real(psi.overlap(psi))))
    if current_norm > 1e-15 and abs(current_norm - target_norm) > 1e-15:
        psi._B[0] = psi._B[0] * (target_norm / current_norm)


def get_M_exact(psi_base, model, chi_max, t_minus,
                gamma1, gamma2, dt_max, rng):
    """
    Exact per-trajectory M via src.dM.exact_M.

    The backward-evolution unitary is e^{+i H t_-}, equivalent to
    the symmetric-echo parameter nu - mu = -2 t_-. The four
    evolutions inside exact_M share the branch_seed so that the
    per-trajectory analytical identity holds for that trajectory.
    The natural norm ||S_i psi_pre|| is captured before evolution
    and restored afterwards, because the dissipative half-steps
    would otherwise renormalize the auxiliary u_i states to 1.
    """
    from src.dM import exact_M

    branch_seed = int(rng.integers(0, 2 ** 32 - 1))

    def evolve_fn(psi, strength, k_axis):
        original_norm = _norm(psi)
        branch_rng = np.random.default_rng(branch_seed)
        actual_time = strength / 2.0
        evolve_trajectory(psi, model, actual_time,
                          gamma1, gamma2, dt_max, chi_max, branch_rng)
        _restore_norm(psi, original_norm)

    M, _psi_q = exact_M(
        psi_base, model,
        k_axis=None,
        nu_minus_mu=-2.0 * t_minus,
        chi_max=chi_max,
        dt_max=dt_max,
        evolve_fn=evolve_fn,
    )
    return M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--t_plus", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--t_minus", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--chi_max", type=int, default=32)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--n_traj", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method",
                        choices=['finite_diff', 'exact_derivative'],
                        default='exact_derivative',
                        help="Method for the signal-response matrix M.")
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()


    # --- (mu1, mu2) translation (migration) ---
    if getattr(args, "mu1", None) is not None and args.t_plus is None:
        _N = getattr(args, "N", None) or ((getattr(args, "Lx", 0) or 0) * (getattr(args, "Ly", 0) or 0))
        args.t_plus = _N * args.mu1 / 4.0
        args.t_minus = _N * args.mu2 / 4.0
        args.convention = "mu1mu2"
    elif args.t_plus is not None and getattr(args, "mu1", None) is None:
        args.convention = "internal"
    else:
        raise SystemExit("Provide either --t_plus --t_minus or --mu1 --mu2")
    args.mu_suffix = (f"_mu1{args.mu1:.4f}_mu2{args.mu2:.4f}"
                      if args.convention == "mu1mu2"
                      else f"_tplus{args.t_plus:.4f}_tminus{args.t_minus:.4f}")
    # --- End (mu1, mu2) translation ---
    assert args.t_plus >= 0.0, "t_plus must be a non-negative duration"
    assert args.t_minus >= 0.0, "t_minus must be a non-negative duration"

    if args.task_id is not None:
        fname = f"asym_pt_{args.task_id}.npy"
    else:
        fname = (f"asym_pt_N{args.N}_alpha{args.alpha}"
                 f"{args.mu_suffix}"
                 f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
                 f"_ntraj{args.n_traj}.npy")

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'echo_pts_1d_open_asym')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, fname)

    if os.path.exists(out_file):
        print(f"Already completed: {out_file}", flush=True)
        return

    model = PowerLawOATModel({
        'N': args.N, 'J': args.J, 'alpha': args.alpha,
        'conserve': None, 'compression_method': 'exponential',
    })

    # If exact_derivative is requested, verify that src.dM is
    # importable now rather than after many hours of trajectory
    # evolution.
    if args.method == 'exact_derivative':
        from src.dM import exact_M  # noqa: F401

    rng_master = np.random.default_rng(args.seed)
    M_accum = np.zeros((3, 3))
    SScorr_accum = np.zeros((3, 3))
    S_vec_accum = np.zeros(3)
    qfi_avg_accum = 0.0

    for t in range(args.n_traj):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        step_size = max(1, args.n_traj // 10)
        if t == 0 or (t + 1) % step_size == 0 or (t + 1) == args.n_traj:
            print(f"  trajectory {t + 1}/{args.n_traj}", flush=True)

        psi_base = initialize_css(model.lat)
        evolve_trajectory(psi_base, model, +args.t_plus,
                          args.gamma1, args.gamma2,
                          args.dt_max, args.chi_max, rng)

        # ---- M via the chosen method ----
        if args.method == 'exact_derivative':
            M_traj = get_M_exact(psi_base, model, args.chi_max,
                                 args.t_minus,
                                 args.gamma1, args.gamma2,
                                 args.dt_max, rng)
        else:
            delta = 1e-3
            M_traj = np.zeros((3, 3))
            for i, axis in enumerate(['x', 'y', 'z']):
                branch_seed = int(rng.integers(0, 2**32 - 1))
                for sign in (+1, -1):
                    branch_rng = np.random.default_rng(branch_seed)
                    psi_copy = psi_base.copy()
                    apply_collective_rotation(psi_copy, axis, sign * delta)
                    evolve_trajectory(psi_copy, model, -args.t_minus,
                                      args.gamma1, args.gamma2,
                                      args.dt_max, args.chi_max,
                                      branch_rng)
                    means = [np.real(np.sum(psi_copy.expectation_value(a)))
                             for a in ['Sx', 'Sy', 'Sz']]
                    for j in range(3):
                        M_traj[i, j] += sign * means[j] / (2.0 * delta)

        # ---- Final state for covariance ----
        psi_q = psi_base.copy()
        evolve_trajectory(psi_q, model, -args.t_minus,
                          args.gamma1, args.gamma2,
                          args.dt_max, args.chi_max, rng)
        S_vec, Cov = trajectory_covariance(psi_q)
        SScorr = Cov + np.outer(S_vec, S_vec)

        M_accum += M_traj
        SScorr_accum += SScorr
        S_vec_accum += S_vec
        qfi_avg_accum += 4.0 * float(np.max(np.linalg.eigvalsh(Cov)))

    n_total = args.n_traj
    M = M_accum / n_total
    SScorr_avg = SScorr_accum / n_total
    S_mean = S_vec_accum / n_total
    Q = SScorr_avg - np.outer(S_mean, S_mean)
    Q = 0.5 * (Q + Q.T)
    qfi_var = 4.0 * float(np.max(np.linalg.eigvalsh(Q)))
    qfi_avg = float(qfi_avg_accum / n_total)

    evals, evecs = np.linalg.eigh(Q)
    mask = evals > max(1e-12, 1e-8 * evals.max())
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T

    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    snr = float(np.max(sv))

    np.save(out_file, {
        'N': args.N, 'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 't_plus': args.t_plus, 't_minus': args.t_minus,
        'gamma1': args.gamma1, 'gamma2': args.gamma2,
        'n_traj': n_total, 'snr': snr,
        'qfi_var': qfi_var, 'qfi_avg': qfi_avg,
        'method': args.method,
    })
    print(f"Asymmetric 1D open SNR = {snr:.6f} "
          f"(N={args.N}, alpha={args.alpha}, "
          f"t_plus={args.t_plus:.4f}, t_minus={args.t_minus:.4f}, "
          f"g1={args.gamma1:.4e}, g2={args.gamma2:.4e}, "
          f"n_traj={n_total}, method={args.method})", flush=True)


if __name__ == "__main__":
    main()