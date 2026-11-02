"""
scripts/workers/closed_1d/powerlaw_echo_worker.py

Computes one full echo SNR point for the power-law Ising model.

The echo sequence is:
    T_{mu/2}  ->  R_{n}(±delta)  ->  T_{(nu-mu)/2}

where T_t = exp(-i t sum_{i<j} (J/r_{ij}^{alpha}) sigma_z^i sigma_z^j / N)

The script is intended for SLURM array or loop submission.

The time-evolution helper ``evolve_mps_time`` is imported from
src/evolution.py; this file does not define its own copy.

--method flag
-------------
finite_diff      (default) central finite differences.
exact_derivative analytic M via src.dM.exact_M with the default
                (unitary) backward evolution. The exact path also
                returns psi_q directly, so it skips the separate
                backward-evolution step that the finite-difference
                path needs. n_opt and m_opt are the optimal signal
                and measurement axes extracted from the SVD of
                M Q^{-1/2}.
"""

import numpy as np
import scipy.linalg
import os
import sys
import warnings
import argparse
import h5py
from tenpy.tools import hdf5_io


warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)


from src.models import PowerLawOATModel
from src.evolution import (
    initialize_css,
    apply_collective_rotation,
    evolve_mps_time,
)
from src.observables import _get_3x3_covariance, compute_qfi


def get_Q_mps(psi):
    """Return 3x3 covariance matrix."""
    _, Cov = _get_3x3_covariance(psi)
    return Cov


def get_M_mps(psi_mu, model, chi_max, nu_minus_mu, dt_max=0.05, delta=1e-3):
    """Compute signal response matrix M via central finite differences."""
    M = np.zeros((3, 3))
    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']

    for i, axis in enumerate(axes):
        # forward phase imprint
        psi_plus = psi_mu.copy()
        apply_collective_rotation(psi_plus, axis, delta)
        evolve_mps_time(psi_plus, model, nu_minus_mu / 2.0, chi_max, dt_max)
        means_plus = [np.real(np.sum(psi_plus.expectation_value(a))) for a in op_names]

        # backward phase imprint
        psi_minus = psi_mu.copy()
        apply_collective_rotation(psi_minus, axis, -delta)
        evolve_mps_time(psi_minus, model, nu_minus_mu / 2.0, chi_max, dt_max)
        means_minus = [np.real(np.sum(psi_minus.expectation_value(a))) for a in op_names]

        for j in range(3):
            M[i, j] = (means_plus[j] - means_minus[j]) / (2.0 * delta)

    return M


def save_checkpoint(checkpoint_file, data_dict):
    tmp_file = checkpoint_file + ".tmp"
    with h5py.File(tmp_file, 'w') as f:
        hdf5_io.save_to_hdf5(f, data_dict)
    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with h5py.File(checkpoint_file, 'r') as f:
            return hdf5_io.load_from_hdf5(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--chi_max", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--method",
                        choices=['finite_diff', 'exact_derivative'],
                        default='exact_derivative',
                        help="Method for the signal-response matrix M.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override the output directory (default: "
                             "data/powerlaw_echo_pts/). Used by validation "
                             "scripts to avoid colliding with production.")
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
    # output and checkpoint paths
    if args.output_dir:
        DATA_DIR = args.output_dir
    else:
        DATA_DIR = os.path.join(ROOT_DIR, 'data', 'powerlaw_echo_pts')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(
        DATA_DIR,
        f"plaw_echo_N{args.N}_chi{args.chi_max}_alpha{args.alpha}"
        f"{args.mu_suffix}.npy"
    )
    checkpoint_file = out_file + ".checkpoint.h5"

    if os.path.exists(out_file):
        try:
            _d = np.load(out_file, allow_pickle=True).item()
            _has_sint = ('S_intermediate' in _d and
                         np.isfinite(float(_d.get('S_intermediate',
                                                    float('nan')))))
        except Exception:
            _has_sint = False
        if _has_sint:
            print(f"Already completed: {out_file}", flush=True)
            return
        print(f"Recomputing (missing S_intermediate): {out_file}",
              flush=True)
        os.remove(out_file)

    chk = load_checkpoint(checkpoint_file)

    model = PowerLawOATModel({
        'N': args.N, 'J': args.J, 'alpha': args.alpha,
        'conserve': None,
    })

    # 1. Base state after initial twisting T_{mu/2}
    if 'psi_base' in chk:
        psi_base = chk['psi_base']
        S_intermediate = chk.get('S_intermediate', float('nan'))
        print(f"Loaded psi_base from checkpoint. "
              f"S_intermediate = {S_intermediate:.4f}", flush=True)
    else:
        psi_base = initialize_css(model.lat)
        evolve_mps_time(psi_base, model, args.mu / 2.0,
                        args.chi_max, args.dt_max)
        try:
            S_intermediate = float(max(psi_base.entanglement_entropy()))
        except Exception:
            S_intermediate = float('nan')
        save_checkpoint(checkpoint_file,
                        {'psi_base': psi_base,
                         'S_intermediate': S_intermediate})
        print(f"Saved psi_base. S_intermediate = {S_intermediate:.4f}",
              flush=True)

    # 2. Signal response matrix M
    nu_minus_mu = args.nu - args.mu
    if 'M' in chk:
        M = chk['M']
        print("Loaded M from checkpoint.", flush=True)
    else:
        if args.method == 'exact_derivative':
            from src.dM import exact_M
            M, psi_q_from_M = exact_M(
                psi_base, model,
                k_axis=None,
                nu_minus_mu=nu_minus_mu,
                chi_max=args.chi_max,
                dt_max=args.dt_max,
            )
            # Save psi_q as well, since it comes for free from exact_M.
            save_checkpoint(checkpoint_file,
                            {'psi_base': psi_base,
                             'S_intermediate': S_intermediate,
                             'M': M, 'psi_q': psi_q_from_M})
        else:
            M = get_M_mps(psi_base, model, args.chi_max,
                          nu_minus_mu, args.dt_max)
            save_checkpoint(checkpoint_file,
                            {'psi_base': psi_base,
                             'S_intermediate': S_intermediate,
                             'M': M})
        print(f"Saved M (method={args.method}).", flush=True)

    # 3. Final state for covariance Q
    if 'psi_q' in chk:
        psi_q = chk['psi_q']
        print("Loaded psi_q from checkpoint.", flush=True)
    else:
        psi_q = psi_base.copy()
        evolve_mps_time(psi_q, model, nu_minus_mu / 2.0,
                        args.chi_max, args.dt_max)
        save_checkpoint(checkpoint_file,
                        {'psi_base': psi_base,
                         'S_intermediate': S_intermediate,
                         'M': M, 'psi_q': psi_q})
        print("Saved psi_q.", flush=True)

    Q = get_Q_mps(psi_q)

    # 4. SNR and optimal axes via SVD of M Q^{-1/2}
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > max(1e-12, 1e-8 * evals.max())
    inv_sqrt_evals = np.zeros_like(evals)
    inv_sqrt_evals[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt_evals) @ evecs.T

    F_matrix = M @ Q_inv_sqrt
    U, sv, Vh = scipy.linalg.svd(F_matrix, full_matrices=False)
    snr = float(sv[0])
    n_opt = np.real(U[:, 0])
    m_opt = np.real(Q_inv_sqrt @ Vh[0, :])

    # QFI of psi_q with respect to the best generator direction.
    # Upper bound on any measurement's sensitivity; SNR^2 <= qfi.
    qfi = float(compute_qfi(psi_q))

    # 5. Save result
    np.save(out_file, {
        'N': args.N,
        'chi_max': args.chi_max,
        'alpha': args.alpha,
        'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu,
        'nu': args.nu,
        'snr': snr,
        'qfi': qfi,
        'n_opt': n_opt,
        'm_opt': m_opt,
        'S_intermediate': S_intermediate,
        'method': args.method,
    })

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Power-law echo SNR = {snr:.6f} for N={args.N}, chi={args.chi_max}, "
          f"alpha={args.alpha}, mu={args.mu:.4f}, nu={args.nu:.4f}, "
          f"method={args.method}", flush=True)


if __name__ == "__main__":
    main()