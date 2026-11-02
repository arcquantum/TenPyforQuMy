"""
scripts/workers/open_1d/simple_mpo_worker.py

Naive vectorized-MPO worker (failure baseline) for the 1D power-law
open-system echo protocol. See `src/simple_mpo.py` for the physics
and the expected failure mode.
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

from src.simple_mpo import (
    SimpleMPOModel,
    initialize_vectorized_css,
    evolve_mpo,
    apply_phase_imprint,
    contract_mps_to_rho,
    collective_spin_operator,
    diagnose_rho,
)


def compute_M_Q(psi_base, model, chi_mpo, nu_minus_mu, dt_max,
                delta=1e-3):
    """
    Run the echo sequence for each axis and phase-imprint sign, extract
    the signal-response matrix M and the covariance Q, and return the
    final state at φ=0 for failure diagnostics.
    """
    L = model.L
    axes = ['x', 'y', 'z']

    M = np.zeros((3, 3))
    for i, axis in enumerate(axes):
        for sign in (+1, -1):
            psi_copy = psi_base.copy()
            apply_phase_imprint(psi_copy, axis, sign * delta)
            evolve_mpo(psi_copy, model, nu_minus_mu / 2.0, chi_mpo, dt_max)
            rho = contract_mps_to_rho(psi_copy, L)
            for j, axis_j in enumerate(axes):
                S_j = collective_spin_operator(axis_j, L)
                exp_S_j = float(np.real(np.trace(S_j @ rho)))
                M[i, j] += sign * exp_S_j / (2.0 * delta)

    # Final state at φ = 0 for the covariance matrix
    psi_q = psi_base.copy()
    evolve_mpo(psi_q, model, nu_minus_mu / 2.0, chi_mpo, dt_max)
    rho_q = contract_mps_to_rho(psi_q, L)

    Q = np.zeros((3, 3), dtype=complex)
    for i, axis_i in enumerate(axes):
        S_i = collective_spin_operator(axis_i, L)
        for j, axis_j in enumerate(axes):
            S_j = collective_spin_operator(axis_j, L)
            Sym = (S_i @ S_j + S_j @ S_i) / 2.0
            Q[i, j] = (np.trace(Sym @ rho_q)
                       - np.trace(S_i @ rho_q) * np.trace(S_j @ rho_q))
    Q = np.real(Q)

    return M, Q, rho_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--chi_mpo", type=int, default=32)
    parser.add_argument("--dt_max", type=float, default=0.05)
    args = parser.parse_args()

    # --- (mu1, mu2) translation (migration) ---
    if getattr(args, "mu1", None) is not None and args.mu is None:
        _N = args.L
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
    print(f"Initializing simple MPO worker: L={args.L}, "
          f"alpha={args.alpha}, J={args.J}, "
          f"mu={args.mu:.6f}, nu={args.nu:.6f}, "
          f"gamma1={args.gamma1:.4e}, gamma2={args.gamma2:.4e}, "
          f"chi_mpo={args.chi_mpo}", flush=True)

    model = SimpleMPOModel({
        'L': args.L,
        'J': args.J,
        'alpha': args.alpha,
        'gamma1': args.gamma1,
        'gamma2': args.gamma2,
    })

    psi = initialize_vectorized_css(model)
    evolve_mpo(psi, model, args.mu / 2.0, args.chi_mpo, args.dt_max)

    nu_minus_mu = args.nu - args.mu
    M, Q, rho_q = compute_M_Q(
        psi, model, args.chi_mpo, nu_minus_mu, args.dt_max
    )

    trace = float(np.real(np.trace(rho_q)))
    min_eig, purity = diagnose_rho(rho_q)

    # SNR via SVD of M Q^{-1/2}
    try:
        evals, evecs = np.linalg.eigh(Q)
        mask = evals > 1e-12
        inv_sqrt = np.zeros_like(evals)
        inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
        Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T
        sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
        snr = float(np.max(sv))
    except Exception as exc:
        print(f"[simple_mpo_worker] SNR extraction failed: {exc}", flush=True)
        snr = float('nan')

    print(f"Simple MPO: trace={trace:.8f}, min_eig={min_eig:.4e}, "
          f"purity={purity:.8f}, SNR={snr:.6f}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'simple_mpo')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(
        DATA_DIR,
        f"simple_mpo_L{args.L}_alpha{args.alpha}"
        f"{args.mu_suffix}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
        f"_chi{args.chi_mpo}.npy"
    )
    np.save(out_file, {
        'L': args.L,
        'alpha': args.alpha,
        'J': args.J,
        'mu1': getattr(args, 'mu1', None),
        'mu2': getattr(args, 'mu2', None),
        'mu': args.mu,
        'nu': args.nu,
        'gamma1': args.gamma1,
        'gamma2': args.gamma2,
        'chi_mpo': args.chi_mpo,
        'trace': trace,
        'snr': snr,
        'min_eig': min_eig,
        'purity': purity,
    })
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()