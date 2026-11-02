#!/usr/bin/env python
"""
Diagnose a single-point exact_M failure at the TN level.

Runs the exact_M path step by step, printing the chi, norm, and
any traceback at each stage. Used to identify whether the failure
is in apply_collective_spin, evolve_mps_time (through
apply_twist_about_k), the M overlap, or the Q extraction.

    python scripts/checks/diag_exact_M.py
    python scripts/checks/diag_exact_M.py --N 8 --alpha 1 --chi 4 --mu 0 --nu 0
"""
import argparse
import sys
import traceback

import numpy as np
import scipy.linalg

sys.path.insert(0, '.')

from src.models import PowerLawOATModel
from src.evolution import initialize_css, evolve_mps_time
from src.dM import exact_M, exact_Q, apply_collective_spin, _build_collective_spin_MPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--chi', type=int, default=4)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--nu', type=float, default=0.0)
    parser.add_argument('--J', type=float, default=1.0)
    args = parser.parse_args()

    print(f"Diagnostic: N={args.N}, alpha={args.alpha}, chi={args.chi}, "
          f"dt={args.dt}, mu={args.mu}, nu={args.nu}")
    print("=" * 60)

    # [1] model
    print("\n[1] Build model")
    model = PowerLawOATModel({
        'N': args.N, 'J': args.J, 'alpha': args.alpha,
        'conserve': None, 'compression_method': 'exponential',
    })
    print(f"    OK")

    # [2] CSS
    print("\n[2] Initialize CSS")
    psi_base = initialize_css(model.lat)
    print(f"    chi = {psi_base.chi}, norm = {psi_base.norm}")

    # [3] forward twist
    print(f"\n[3] Forward twist for t = {args.mu/2.0:.4f}")
    try:
        evolve_mps_time(psi_base, model, args.mu / 2.0, args.chi, args.dt)
        print(f"    OK. chi = {psi_base.chi}, norm = {psi_base.norm}")
    except Exception:
        print(f"    FAILED:")
        traceback.print_exc()
        sys.exit(1)

    # [4] collective spin MPO build
    print("\n[4] Build collective spin MPO for each axis")
    for axis in ['x', 'y', 'z']:
        try:
            mpo = _build_collective_spin_MPO(model, axis)
            print(f"    S_{axis}: D_MPO = {max(mpo.chi)}")
        except Exception:
            print(f"    S_{axis} FAILED:")
            traceback.print_exc()
            sys.exit(1)

    # [5] apply S_x, S_y, S_z to psi_base
    print("\n[5] Apply S_x, S_y, S_z to psi_base")
    for axis in ['x', 'y', 'z']:
        try:
            u = apply_collective_spin(psi_base, model, axis, chi_max=args.chi)
            nrm = float(np.sqrt(np.real(u.overlap(u))))
            print(f"    S_{axis}|psi_base>: chi = {u.chi}, norm = {nrm:.6e}")
        except Exception:
            print(f"    S_{axis}|psi_base> FAILED:")
            traceback.print_exc()
            sys.exit(1)

    # [6] exact_M
    print(f"\n[6] exact_M (nu - mu = {args.nu - args.mu:.4f})")
    try:
        M, psi_q = exact_M(
            psi_base, model, k_axis=None,
            nu_minus_mu=args.nu - args.mu,
            chi_max=args.chi, dt_max=args.dt,
        )
        print(f"    M =")
        print(M)
        print(f"    psi_q: chi = {psi_q.chi}, norm = {psi_q.norm}")
    except Exception:
        print(f"    FAILED:")
        traceback.print_exc()
        sys.exit(1)

    # [7] exact_Q
    print("\n[7] exact_Q")
    try:
        Q = exact_Q(psi_q)
        print(f"    Q =")
        print(Q)
        print(f"    Q eigenvalues = {np.linalg.eigvalsh(Q)}")
    except Exception:
        print(f"    FAILED:")
        traceback.print_exc()
        sys.exit(1)

    # [8] SNR
    print("\n[8] SNR from SVD of M Q^{-1/2}")
    try:
        evals, evecs = np.linalg.eigh(Q)
        mask = evals > 1e-12
        inv_sqrt = np.zeros_like(evals)
        inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
        Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T
        sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
        print(f"    SNR = {sv[0]:.10f}")
    except Exception:
        print(f"    FAILED:")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All steps succeeded.")


if __name__ == '__main__':
    main()