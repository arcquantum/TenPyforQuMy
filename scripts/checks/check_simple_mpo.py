"""
scripts/checks/check_simple_mpo.py

Failure-baseline check for the naive vectorized MPO method.

The method is expected to fail: local SVD truncation at a finite
chi_MPO breaks positivity of the reconstructed density matrix. The failure signatures are

  (a) minimum eigenvalue of the symmetrised rho < 0,
  (b) trace Tr(rho) ≠ 1,
  (c) purity Tr(rho^2) > 1 after renormalisation.

In practice, signature (b) is the dominant one at small chi_MPO: the
MPO truncation discards part of the trace-preservation term
−γ/2 · ρ, so the renormalised state acquires a large purity excess.
Signature (a) is also present because the discarded weight is not
distributed symmetrically across the spectrum. Both are documented
in the failure-baseline docstring of the simple MPO module.

The check runs the worker at (L, alpha, mu, nu, gamma1, gamma2,
chi_MPO) = (8, 1.0, pi/2, -pi/2, 0.0, 0.1, 8) and asserts that at
least one of the two failure signatures is present. A small chi_MPO
is chosen deliberately so that the truncation is aggressive and the
failure is unambiguous.
"""

import os
import subprocess
import sys

import numpy as np

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

WORKER = os.path.join(ROOT_DIR, 'scripts', 'workers', 'open_1d', 'simple_mpo_worker.py')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'simple_mpo')

L = 8
ALPHA = 1.0
MU = np.pi / 2
NU = -np.pi / 2
GAMMA1 = 0.0
GAMMA2 = 0.1
CHI_MPO = 8

OUT_FILE = os.path.join(
    DATA_DIR,
    f"simple_mpo_L{L}_alpha{ALPHA}"
    f"_mu{MU:.4f}_nu{NU:.4f}"
    f"_g1{GAMMA1:.4e}_g2{GAMMA2:.4e}"
    f"_chi{CHI_MPO}.npy"
)


def main():
    print("=" * 70)
    print(f"Failure-baseline check: simple MPO at L={L}, alpha={ALPHA}")
    print(f"  mu={MU:.4f}, nu={NU:.4f}, "
          f"gamma1={GAMMA1}, gamma2={GAMMA2}, chi_mpo={CHI_MPO}")
    print("=" * 70)

    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    cmd = [
        sys.executable, WORKER,
        "--L", str(L),
        "--alpha", str(ALPHA),
        "--mu", f"{MU:.6f}",
        "--nu", f"{NU:.6f}",
        "--gamma1", str(GAMMA1),
        "--gamma2", str(GAMMA2),
        "--chi_mpo", str(CHI_MPO),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    data = np.load(OUT_FILE, allow_pickle=True).item()
    trace = data['trace']
    min_eig = data['min_eig']
    purity = data['purity']
    snr = data['snr']

    print()
    print(f"  trace          = {trace:.8f}")
    print(f"  min eigenvalue = {min_eig:+.6e}")
    print(f"  purity         = {purity:.8f}")
    print(f"  SNR            = {snr:.6f}")

    # Numerical-noise thresholds. The failure signatures are
    # qualitative (order 1e-3 or larger in practice), so tolerances
    # of order 1e-6 are safe.
    NEG_TOL = -1e-6
    PURITY_TOL = 1.0 + 1e-6
    TRACE_TOL = 1.0 - 1e-6

    # Purity on the unnormalised rho is misleading when trace
    # preservation is broken. Report the purity of the renormalised
    # rho as well, since that is the quantity the naive method
    # should have preserved.
    if abs(trace) > 1e-12:
        purity_renorm = purity / (trace * trace)
    else:
        purity_renorm = float('nan')

    sig_eig = min_eig < NEG_TOL
    sig_trace = abs(trace - 1.0) > 1e-6
    sig_purity = purity_renorm > PURITY_TOL

    print()
    print(f"  Failure signature (min_eig < {NEG_TOL:+.0e}):    "
          f"{'YES' if sig_eig else 'no'}")
    print(f"  Failure signature (|trace-1| > 1e-6):   "
          f"{'YES' if sig_trace else 'no'}")
    print(f"  Failure signature (purity_renorm > 1):  "
          f"{'YES' if sig_purity else 'no'}   "
          f"(purity_renorm = {purity_renorm:.6f})")

    if sig_eig or sig_trace or sig_purity:
        print()
        print("[PASS] Failure signature detected as expected.")
        sys.exit(0)
    else:
        print()
        print("[FAIL] The method did not exhibit the expected failure.")
        print("       Consider lowering chi_mpo further, increasing "
              "gamma2, or extending the evolution time.")
        sys.exit(1)


if __name__ == "__main__":
    main()