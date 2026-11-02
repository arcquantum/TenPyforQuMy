"""
scripts/workers/shared/tjm_worker.py

Tensor Jump Method worker for the 1D (and 2D cylinder) open-system
power-law echo protocol.

Two modes:
 * --mode echo (default): runs the echo protocol for a single
   (mu, nu) point and outputs the SNR.
 * --mode forward: runs forward-only evolution from the CSS and
   records the trajectory-averaged xi_R^2, S_vN, F_Q, <S_x> at a
   sequence of time points.

MPI parallelization: each MPI rank runs --n_traj_per_rank
trajectories with a distinct seed and averages locally; rank 0
gathers and combines. If mpi4py is not available, runs serially.

--method flag (echo mode only):
 * finite_diff      (default) central finite differences with
                    common random numbers between the +delta and
                    -delta branches.
 * exact_derivative analytic per-trajectory M via
                    src.dM.exact_M. The forward twist is a plain
                    z-twist; the second twisting is also z-aligned
                    (nu_minus_mu = nu - mu). The four backward
                    evolutions inside exact_M (psi_0 and the three
                    u_i) share a single random seed so that the
                    per-trajectory identity
                        M_ij = -2 Im[<u_i| S_j |psi_0>]
                    holds for that trajectory, up to the
                    jump-induced decoherence analysed in the
                    documentation.
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

from src.models import PowerLawOATModel, PowerLawCylinderModel
from src.evolution import initialize_css, apply_collective_rotation
from src.observables import _get_3x3_covariance
from src.tjm import (
    apply_dissipative_half_step,
    evolve_coherent_step,
    maybe_apply_jump,
)
from src.dM import exact_M


try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1


# ----------------------------------------------------------------------
# Trajectory evolution helpers
# ----------------------------------------------------------------------
def evolve_one_step(psi, model, gamma1, gamma2, dt, dt_abs, chi_max, rng):
    """
    One Strang-split step: D[dt_abs/2] U[dt] D[dt_abs/2] + jump.
    """
    apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
    evolve_coherent_step(psi, model, dt, chi_max)
    apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
    applied, _ = maybe_apply_jump(psi, gamma1, gamma2, dt_abs, rng)


def evolve_trajectory(psi, model, total_time, gamma1, gamma2, dt_max,
                      chi_max, rng):
    """Evolve one trajectory for a signed total_time."""
    if abs(total_time) < 1e-15:
        return
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt = total_time / N_steps  # signed
    dt_abs = abs(dt)
    for _ in range(N_steps):
        evolve_one_step(psi, model, gamma1, gamma2, dt, dt_abs,
                        chi_max, rng)


def trajectory_covariance(psi):
    """Return S_vec (3,) and Cov (3,3) for one trajectory."""
    _, Cov = _get_3x3_covariance(psi)
    S_vec = np.array([
        np.real(np.sum(psi.expectation_value('Sx'))),
        np.real(np.sum(psi.expectation_value('Sy'))),
        np.real(np.sum(psi.expectation_value('Sz'))),
    ])
    return S_vec, Cov


def trajectory_density_matrix(psi, L):
    """Full 2^L x 2^L density matrix of an L-site MPS (small L only)."""
    psi.canonical_form()
    theta = psi.get_theta(0, n=L)
    arr = theta.to_ndarray()
    while arr.ndim > L and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr.reshape(2 ** L)
    return np.outer(arr, arr.conj())


def _norm(psi):
    """Return ||psi|| as a float."""
    return float(np.sqrt(np.real(psi.overlap(psi))))


def _restore_norm(psi, target_norm):
    """
    Rescale the MPS so that ||psi|| == target_norm. Needed because
    the dissipative half-steps renormalize the state to 1, whereas
    the exact-M formula requires the auxiliary states
    |u_i> = T_2 S_i |psi_pre> to keep their natural norms
    ||S_i psi_pre||.
    """
    current_norm = float(np.sqrt(np.real(psi.overlap(psi))))
    if current_norm > 1e-15 and abs(current_norm - target_norm) > 1e-15:
        psi._B[0] = psi._B[0] * (target_norm / current_norm)


def get_M_exact(psi_base, model, chi_max, nu_minus_mu,
                gamma1, gamma2, dt_max, rng):
    """
    Exact per-trajectory M via src.dM.exact_M.

    Symmetric-echo convention: nu_minus_mu = nu - mu (no factor of
    2 for the current worker; exact_M divides by 2 internally).
    """
    branch_seed = int(rng.integers(0, 2 ** 32 - 1))

    def evolve_fn(psi, strength, k_axis):
        original_norm = _norm(psi)
        branch_rng = np.random.default_rng(branch_seed)
        actual_time = strength / 2.0
        evolve_trajectory(psi, model, actual_time,
                          gamma1, gamma2, dt_max, chi_max,
                          branch_rng)
        _restore_norm(psi, original_norm)

    M, _psi_q = exact_M(
        psi_base, model,
        k_axis=None,
        nu_minus_mu=nu_minus_mu,
        chi_max=chi_max,
        dt_max=dt_max,
        evolve_fn=evolve_fn,
    )
    return M


# ----------------------------------------------------------------------
# Echo mode
# ----------------------------------------------------------------------
def run_echo(args):
    """Echo protocol: single (mu, nu) point, output SNR."""
    rng_master = np.random.default_rng(args.seed + RANK * 1_000_003)

    M_accum = np.zeros((3, 3))
    SScorr_accum = np.zeros((3, 3))
    S_vec_accum = np.zeros(3)
    qfi_avg_accum = 0.0
    S_int_accum = 0.0
    S_int_count = 0

    if args.Lx is not None:
        model = PowerLawCylinderModel({
            'Lx': args.Lx, 'Ly': args.Ly,
            'J': args.J, 'alpha': args.alpha, 'conserve': None,
        })
    else:
        model = PowerLawOATModel({
            'N': args.N, 'J': args.J, 'alpha': args.alpha,
            'conserve': None,
            'compression_method': 'exponential',
        })

    n_local = args.n_traj_per_rank
    for t in range(n_local):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        step_size = max(1, n_local // 10)
        if t == 0 or (t + 1) % step_size == 0 or (t + 1) == n_local:
            print(f"  [rank {RANK}] trajectory {t + 1}/{n_local}",
                  flush=True)

        # Build trajectory state after the initial forward twisting.
        psi_base = initialize_css(model.lat)
        evolve_trajectory(psi_base, model, args.mu / 2.0,
                          args.gamma1, args.gamma2, args.dt_max,
                          args.chi_max, rng)
        try:
            S_int_accum += float(max(psi_base.entanglement_entropy()))
            S_int_count += 1
        except Exception:
            pass

        nu_minus_mu = args.nu - args.mu

        # ---- M via the chosen method ----
        if args.method == 'exact_derivative':
            M_traj = get_M_exact(psi_base, model, args.chi_max,
                                 nu_minus_mu,
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
                    apply_collective_rotation(psi_copy, axis,
                                              sign * delta)
                    evolve_trajectory(psi_copy, model,
                                      nu_minus_mu / 2.0,
                                      args.gamma1, args.gamma2,
                                      args.dt_max, args.chi_max,
                                      branch_rng)
                    means = [
                        np.real(np.sum(psi_copy.expectation_value(a)))
                        for a in ['Sx', 'Sy', 'Sz']
                    ]
                    for j in range(3):
                        M_traj[i, j] += sign * means[j] / (2.0 * delta)

        # ---- Final state for covariance ----
        psi_q = psi_base.copy()
        evolve_trajectory(psi_q, model, nu_minus_mu / 2.0,
                          args.gamma1, args.gamma2, args.dt_max,
                          args.chi_max, rng)
        S_vec, Cov = trajectory_covariance(psi_q)
        SScorr = Cov + np.outer(S_vec, S_vec)

        M_accum += M_traj
        SScorr_accum += SScorr
        S_vec_accum += S_vec
        qfi_avg_accum += 4.0 * float(np.max(np.linalg.eigvalsh(Cov)))

    # Send to rank 0
    if SIZE > 1:
        assert COMM is not None  # guaranteed by SIZE > 1 above
        M_all = COMM.reduce(M_accum, op=MPI.SUM, root=0)
        SScorr_all = COMM.reduce(SScorr_accum, op=MPI.SUM, root=0)
        S_all = COMM.reduce(S_vec_accum, op=MPI.SUM, root=0)
        qfi_avg_all = COMM.reduce(qfi_avg_accum, op=MPI.SUM, root=0)
    else:
        M_all, SScorr_all, S_all = M_accum, SScorr_accum, S_vec_accum
        qfi_avg_all = qfi_avg_accum

    if RANK == 0:
        n_total = args.n_traj_per_rank * SIZE
        M = M_all / n_total
        SScorr_avg = SScorr_all / n_total
        S_mean = S_all / n_total
        S_intermediate = (S_int_accum / S_int_count
                          if S_int_count > 0 else float('nan'))

        Q = SScorr_avg - np.outer(S_mean, S_mean)
        Q = 0.5 * (Q + Q.T)

        qfi_var = 4.0 * float(np.max(np.linalg.eigvalsh(Q)))
        qfi_avg = float(qfi_avg_all / n_total)

        if os.environ.get('TJM_VERBOSE', '0') == '1':
            print("=" * 60, flush=True)
            print("DIAGNOSTIC M:", flush=True)
            for row in M:
                print("  " + " ".join(f"{x:+.6e}" for x in row),
                      flush=True)
            print("DIAGNOSTIC Q:", flush=True)
            for row in Q:
                print("  " + " ".join(f"{x:+.6e}" for x in row),
                      flush=True)
            _evals_dbg = np.linalg.eigvalsh(Q)
            print(f"DIAGNOSTIC Q_evals: "
                  f"{_evals_dbg[0]:+.4e}, "
                  f"{_evals_dbg[1]:+.4e}, "
                  f"{_evals_dbg[2]:+.4e}", flush=True)
            print(f"DIAGNOSTIC S_mean: "
                  f"({S_mean[0]:+.4f}, "
                  f"{S_mean[1]:+.4f}, "
                  f"{S_mean[2]:+.4f})", flush=True)
            print("=" * 60, flush=True)

        evals, evecs = np.linalg.eigh(Q)
        mask = evals > max(1e-12, 1e-8 * evals.max())
        inv_sqrt = np.zeros_like(evals)
        inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
        Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T

        # SNR and optimal axes from SVD of M Q^{-1/2}
        F = M @ Q_inv_sqrt
        U, sv, Vh = scipy.linalg.svd(F, full_matrices=False)
        snr = float(sv[0])
        n_opt = np.real(U[:, 0])
        m_opt = np.real(Q_inv_sqrt @ Vh[0, :])

        print(f"TJM echo SNR = {snr:.6f} "
              f"(N_traj total = {n_total}, "
              f"gamma1 = {args.gamma1:.4e}, "
              f"gamma2 = {args.gamma2:.4e}, "
              f"method = {args.method})", flush=True)

        DATA_DIR = os.path.join(ROOT_DIR, 'data', 'tjm_pts')
        os.makedirs(DATA_DIR, exist_ok=True)
        if args.Lx is not None:
            fname = (f"tjm_Lx{args.Lx}_Ly{args.Ly}"
                     f"_chi{args.chi_max}_alpha{args.alpha}"
                     f"{args.mu_suffix}"
                     f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
                     f"_ntraj{args.n_traj_per_rank * SIZE}.npy")
            meta = {'Lx': args.Lx, 'Ly': args.Ly}
        else:
            fname = (f"tjm_N{args.N}"
                     f"_chi{args.chi_max}_alpha{args.alpha}"
                     f"{args.mu_suffix}"
                     f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
                     f"_ntraj{args.n_traj_per_rank * SIZE}.npy")
            meta = {'N': args.N}

        out_file = os.path.join(DATA_DIR, fname)
        np.save(out_file, {
            **meta,
            'alpha': args.alpha,
            'J': args.J,
            'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu,
            'nu': args.nu,
            'gamma1': args.gamma1,
            'gamma2': args.gamma2,
            'snr': snr,
            'qfi_var': qfi_var,
            'qfi_avg': qfi_avg,
            'n_opt': n_opt,
            'm_opt': m_opt,
            'S_intermediate': S_intermediate,
            'method': args.method,
        })
        print(f"Saved to {out_file}", flush=True)


# ----------------------------------------------------------------------
# Forward mode (unchanged; exact-M not applicable to time series)
# ----------------------------------------------------------------------
def run_forward(args):
    """
    Forward-only evolution from the CSS. Records trajectory-averaged
    observables at a sequence of time points.
    """
    L = (args.Lx * args.Ly) if args.Lx is not None else args.N

    dt_check = args.t_max / args.n_checkpoints
    time_grid = np.array([k * dt_check
                          for k in range(args.n_checkpoints + 1)])

    xi2_arr = np.zeros(len(time_grid))
    svn_arr = np.zeros(len(time_grid))
    fq_arr = np.zeros(len(time_grid))
    sx_arr = np.zeros(len(time_grid))

    rng_master = np.random.default_rng(args.seed + RANK * 1_000_003)
    n_local = args.n_traj_per_rank

    if args.Lx is not None:
        model = PowerLawCylinderModel({
            'Lx': args.Lx, 'Ly': args.Ly,
            'J': args.J, 'alpha': args.alpha, 'conserve': None,
        })
    else:
        model = PowerLawOATModel({
            'N': args.N, 'J': args.J, 'alpha': args.alpha,
            'conserve': None,
            'compression_method': 'exponential',
        })

    for t in range(n_local):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        step_size = max(1, n_local // 10)
        if t == 0 or (t + 1) % step_size == 0 or (t + 1) == n_local:
            print(f"  [rank {RANK}] trajectory {t + 1}/{n_local}",
                  flush=True)

        psi = initialize_css(model.lat)

        S_vec, Cov = trajectory_covariance(psi)
        evals = np.linalg.eigvalsh(Cov)
        fq_arr[0] += 4.0 * np.max(evals)
        n_spin = np.sum(S_vec**2)
        if n_spin > 1e-12:
            n_vec = S_vec / np.sqrt(n_spin)
            P_perp = np.eye(3) - np.outer(n_vec, n_vec)
            Cov_perp = P_perp @ Cov @ P_perp
            evals_perp = np.sort(np.linalg.eigvalsh(Cov_perp))
            v_minus = evals_perp[1]
            xi2_arr[0] += L * v_minus / n_spin
        sx_arr[0] += S_vec[0]
        try:
            ents = psi.entanglement_entropy()
            svn_arr[0] += ents[(L - 1) // 2]
        except Exception:
            svn_arr[0] += 0.0

        for k in range(1, len(time_grid)):
            dt_step = time_grid[k] - time_grid[k - 1]
            N_steps = max(1, int(np.ceil(dt_step / args.dt_max)))
            sub_dt = dt_step / N_steps
            for _ in range(N_steps):
                apply_dissipative_half_step(psi, args.gamma1,
                                            sub_dt / 2.0)
                evolve_coherent_step(psi, model, sub_dt, args.chi_max)
                apply_dissipative_half_step(psi, args.gamma1,
                                            sub_dt / 2.0)
                maybe_apply_jump(psi, args.gamma1, args.gamma2,
                                 sub_dt, rng)

            S_vec, Cov = trajectory_covariance(psi)
            evals = np.linalg.eigvalsh(Cov)
            fq_arr[k] += 4.0 * np.max(evals)
            n_spin = np.sum(S_vec**2)
            if n_spin > 1e-12:
                n_vec = S_vec / np.sqrt(n_spin)
                P_perp = np.eye(3) - np.outer(n_vec, n_vec)
                Cov_perp = P_perp @ Cov @ P_perp
                evals_perp = np.sort(np.linalg.eigvalsh(Cov_perp))
                v_minus = evals_perp[1]
                xi2_arr[k] += L * v_minus / n_spin
            sx_arr[k] += S_vec[0]
            try:
                ents = psi.entanglement_entropy()
                svn_arr[k] += ents[(L - 1) // 2]
            except Exception:
                pass

    xi2_local = xi2_arr
    svn_local = svn_arr
    fq_local = fq_arr
    sx_local = sx_arr

    if SIZE > 1:
        assert COMM is not None  # guaranteed by SIZE > 1 above
        xi2_all = COMM.reduce(xi2_local, op=MPI.SUM, root=0)
        svn_all = COMM.reduce(svn_local, op=MPI.SUM, root=0)
        fq_all = COMM.reduce(fq_local, op=MPI.SUM, root=0)
        sx_all = COMM.reduce(sx_local, op=MPI.SUM, root=0)
    else:
        xi2_all, svn_all, fq_all, sx_all = (
            xi2_local, svn_local, fq_local, sx_local
        )

    if RANK == 0:
        n_total = n_local * SIZE
        xi2_avg = xi2_all / n_total
        svn_avg = svn_all / n_total
        fq_avg = fq_all / n_total
        sx_avg = sx_all / n_total

        DATA_DIR = os.path.join(ROOT_DIR, 'data', 'tjm_forward')
        os.makedirs(DATA_DIR, exist_ok=True)
        if args.Lx is not None:
            fname = (f"tjm_forward_Lx{args.Lx}_Ly{args.Ly}"
                     f"_chi{args.chi_max}_alpha{args.alpha}"
                     f"_tmax{args.t_max}"
                     f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
                     f"_ntraj{n_total}.npy")
            meta = {'Lx': args.Lx, 'Ly': args.Ly,
                    'geometry': 'cylinder'}
        else:
            fname = (f"tjm_forward_N{args.N}"
                     f"_chi{args.chi_max}_alpha{args.alpha}"
                     f"_tmax{args.t_max}"
                     f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
                     f"_ntraj{n_total}.npy")
            meta = {'N': args.N, 'geometry': 'chain'}

        out_file = os.path.join(DATA_DIR, fname)
        np.save(out_file, {
            **meta,
            'alpha': args.alpha,
            'J': args.J,
            'gamma1': args.gamma1,
            'gamma2': args.gamma2,
            't_max': args.t_max,
            'time': time_grid,
            'xi_R2': xi2_avg,
            'entropy': svn_avg,
            'qfi': fq_avg,
            'Sx': sx_avg,
        })
        print(f"Saved to {out_file}", flush=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Tensor Jump Method worker."
    )
    parser.add_argument("--mode", choices=['echo', 'forward'],
                        default='echo')
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--Lx", type=int, default=None)
    parser.add_argument("--Ly", type=int, default=None)
    parser.add_argument("--chi_max", type=int, default=32)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--t_max", type=float, default=0.0)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--n_traj_per_rank", type=int, default=10)
    parser.add_argument("--n_checkpoints", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method",
                        choices=['finite_diff', 'exact_derivative'],
                        default='exact_derivative',
                        help="Method for M in echo mode.")
    args = parser.parse_args()


    # --- (mu1, mu2) translation (migration) ---
    if args.mode == 'forward':
        # Forward mode does not use twist arguments. Keep defaults
        # so the rest of the pipeline sees numeric mu/nu.
        if args.mu is None:
            args.mu = 0.0
        if args.nu is None:
            args.nu = 0.0
        args.convention = "forward"
        args.mu_suffix = ""
    elif getattr(args, "mu1", None) is not None and args.mu is None:
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
    if args.Lx is None and args.N is None:
        parser.error("Provide either --N (1D chain) or "
                     "--Lx/--Ly (cylinder).")
    if args.Lx is not None and args.Ly is None:
        parser.error("--Lx requires --Ly.")
    if args.mode == 'echo' and (args.mu == 0.0 and args.nu == 0.0):
        pass  # allowed: coherent-state limit

    if RANK == 0:
        print(f"TJM worker: mode={args.mode}, "
              f"N={args.N}, Lx={args.Lx}, Ly={args.Ly}, "
              f"chi={args.chi_max}, alpha={args.alpha}, "
              f"gamma1={args.gamma1:.4e}, "
              f"gamma2={args.gamma2:.4e}, "
              f"n_traj_per_rank={args.n_traj_per_rank}, "
              f"method={args.method}, "
              f"MPI size={SIZE}", flush=True)

    if args.mode == 'echo':
        run_echo(args)
    else:
        run_forward(args)


if __name__ == "__main__":
    main()