"""
scripts/workers/closed_1d/scharnagl_echo_1d_closed.py

Scharnagl three-axis echo worker for the 1D closed-system power-law
Ising chain.

Protocol
--------
   |psi(phi)> = T_k((nu-mu)/2) R_n(phi) T_z(mu/2) |CSS>

with T_z(mu) = exp(-i mu/2 H_plaw) and
T_k(mu) = R_k T_z(mu) R_k^dagger applied via apply_twist_about_k.
The signal axis n and measurement axis m are optimized analytically
by SVD of M Q^{-1/2}.

CLI
---
Same as powerlaw_echo_worker.py plus:
  --theta_k   polar angle of the twisting axis k in [0, pi]
  --phi_k     azimuthal angle of the twisting axis k in [0, 2 pi)
  --method    {finite_diff, exact_derivative}  (default finite_diff)
  --task_id   optional integer for SLURM array tasks

Output
------
data/scharnagl_pts_1d_closed/*.npy with keys
    {N, alpha, J, mu, nu, theta_k, phi_k, snr, n_opt, m_opt,
     S_intermediate, method}
"""

import argparse
import os
import sys
import warnings

import numpy as np
import scipy.linalg
import h5py

warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

from tenpy.tools import hdf5_io

from src.models import PowerLawOATModel
from src.evolution import (
    initialize_css,
    apply_collective_rotation,
    evolve_mps_time,
    apply_twist_about_k,
)
from src.observables import _get_3x3_covariance, compute_qfi


def k_axis_from_angles(theta_k, phi_k):
    """Unit vector k = (sin θ cos φ, sin θ sin φ, cos θ)."""
    return np.array([
        np.sin(theta_k) * np.cos(phi_k),
        np.sin(theta_k) * np.sin(phi_k),
        np.cos(theta_k),
    ])


def get_Q_mps(psi):
    _, Cov = _get_3x3_covariance(psi)
    return Cov


def get_M_mps(psi_base, model, chi_max, nu_minus_mu,
              theta_k, phi_k, dt_max=0.05, delta=1e-3):
    """
    Signal-response matrix M_ij = ∂<S_j>/∂φ_i |_{φ=0} via central
    finite differences. The backward evolution applies
    apply_twist_about_k with the given (theta_k, phi_k).
    """
    k_axis = k_axis_from_angles(theta_k, phi_k)
    M = np.zeros((3, 3))
    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']
    for i, axis in enumerate(axes):
        for sign in (+1, -1):
            psi_copy = psi_base.copy()
            apply_collective_rotation(psi_copy, axis, sign * delta)
            apply_twist_about_k(psi_copy, model, nu_minus_mu, k_axis,
                                chi_max, dt_max)
            means = [np.real(np.sum(psi_copy.expectation_value(a)))
                     for a in op_names]
            for j in range(3):
                M[i, j] += sign * means[j] / (2.0 * delta)
    return M


def get_M_exact(psi_base, model, chi_max, nu_minus_mu,
                theta_k, phi_k, dt_max=0.05):
    """
    Analytic M via src.dM.exact_M (the documentation). Passes the k-axis
    as a (3,) unit vector; the inner twist argument is nu - mu, and
    exact_M divides by 2 internally, matching the code convention.
    """
    try:
        from src.dM import exact_M
    except ImportError as exc:
        raise SystemExit(
            "--method exact_derivative requires src/dM.py, which is "
            "scheduled for the documentation and is not yet available. "
            "Use --method finite_diff for now."
        ) from exc
    k_axis = k_axis_from_angles(theta_k, phi_k)
    M, _psi_q = exact_M(
        psi_base, model,
        k_axis=k_axis,
        nu_minus_mu=nu_minus_mu,
        chi_max=chi_max,
        dt_max=dt_max,
    )
    return M


def extract_optimal_axes(M, Q):
    """
    Return (snr, n_opt, m_opt) from the SVD of M Q^{-1/2} with the
    mask-based null-eigenvalue handling used by all echo workers.
    """
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


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with h5py.File(tmp, 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            return hdf5_io.load_from_hdf5(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--theta_k", type=float, required=True)
    parser.add_argument("--phi_k", type=float, required=True)
    parser.add_argument("--chi_max", type=int, default=64)
    parser.add_argument("--dt_max", type=float, default=0.05)
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
        raise SystemExit(f"--theta_k must be in [0, pi]; got "
                         f"{args.theta_k}.")

    if args.task_id is not None:
        fname = f"scharnagl_pt_{args.task_id}.npy"
    else:
        fname = (f"scharnagl_pt_N{args.N}_alpha{args.alpha}"
                 f"{args.mu_suffix}"
                 f"_tk{args.theta_k:.4f}_pk{args.phi_k:.4f}.npy")

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'scharnagl_pts_1d_closed')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, fname)
    chk_file = out_file + ".checkpoint.h5"

    if os.path.exists(out_file):
        print(f"Already completed: {out_file}", flush=True)
        return

    chk = load_checkpoint(chk_file)

    model = PowerLawOATModel({
        'N': args.N, 'J': args.J, 'alpha': args.alpha,
        'conserve': None, 'compression_method': 'exponential',
    })

    # Stage 1: forward z-twist.
    if 'psi_base' in chk:
        psi_base = chk['psi_base']
        S_intermediate = chk.get('S_intermediate', float('nan'))
        print(f"Loaded psi_base. S_intermediate = {S_intermediate:.4f}",
              flush=True)
    else:
        psi_base = initialize_css(model.lat)
        evolve_mps_time(psi_base, model, args.mu / 2.0,
                        args.chi_max, args.dt_max)
        try:
            S_intermediate = float(max(psi_base.entanglement_entropy()))
        except Exception:
            S_intermediate = float('nan')
        save_checkpoint(chk_file,
                        {'psi_base': psi_base,
                         'S_intermediate': S_intermediate})
        print(f"Saved psi_base. S_intermediate = {S_intermediate:.4f}",
              flush=True)

    # Stage 2: signal-response matrix M.
    nu_minus_mu = args.nu - args.mu
    if 'M' in chk:
        M = chk['M']
        print("Loaded M from checkpoint.", flush=True)
    else:
        if args.method == 'exact_derivative':
            M = get_M_exact(psi_base, model, args.chi_max,
                            nu_minus_mu, args.theta_k, args.phi_k,
                            args.dt_max)
        else:
            M = get_M_mps(psi_base, model, args.chi_max,
                          nu_minus_mu, args.theta_k, args.phi_k,
                          args.dt_max)
        save_checkpoint(chk_file, {'psi_base': psi_base,
                                   'S_intermediate': S_intermediate,
                                   'M': M})
        print("Saved M.", flush=True)

    # Stage 3: final state for covariance Q.
    if 'psi_q' in chk:
        psi_q = chk['psi_q']
        print("Loaded psi_q from checkpoint.", flush=True)
    else:
        k_axis = k_axis_from_angles(args.theta_k, args.phi_k)
        psi_q = psi_base.copy()
        apply_twist_about_k(psi_q, model, nu_minus_mu, k_axis,
                            args.chi_max, args.dt_max)
        save_checkpoint(chk_file,
                        {'psi_base': psi_base,
                         'S_intermediate': S_intermediate,
                         'M': M, 'psi_q': psi_q})
        print("Saved psi_q.", flush=True)

    Q = get_Q_mps(psi_q)
    snr, n_opt, m_opt = extract_optimal_axes(M, Q)

    # QFI of the final state with respect to the best generator.
    qfi = float(compute_qfi(psi_q))


    np.save(out_file, {
        'N': args.N, 'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu, 'nu': args.nu,
        'theta_k': args.theta_k, 'phi_k': args.phi_k,
        'snr': snr,
        'qfi': qfi,
        'qfi': qfi,
        'n_opt': n_opt, 'm_opt': m_opt,
        'S_intermediate': S_intermediate,
        'method': args.method,
    })
    if os.path.exists(chk_file):
        os.remove(chk_file)
    print(f"Scharnagl 1D closed SNR = {snr:.6f} "
          f"(N={args.N}, alpha={args.alpha}, "
          f"mu={args.mu:.4f}, nu={args.nu:.4f}, "
          f"theta_k={args.theta_k:.4f}, phi_k={args.phi_k:.4f})",
          flush=True)


if __name__ == "__main__":
    main()