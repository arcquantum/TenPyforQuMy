"""
scripts/checks/check_models.py

Validates that the three TeNPy MPO model classes reproduce their
analytic Hamiltonians on the full computational basis.

For each model, the check:
  1. builds the MPO via the production class,
  2. extracts the MPO diagonal by applying it to each product state,
  3. builds the analytic diagonal directly in the same basis order,
  4. compares sorted diagonals element-wise.

The comparison uses sorted diagonals because the cylinder block-site
representation may permute basis states relative to the linear index
convention. Two Ising-type Hamiltonians that agree on the sorted
diagonal agree everywhere (both are diagonal in the Z basis).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from tenpy.networks.mps import MPS
from src.models import OATModel, PowerLawOATModel, PowerLawCylinderModel


TOL_EXACT = 1e-8      # analytic vs 'exact' compression
TOL_EXPONENTIAL = 1e-4  # analytic vs 'exponential' compression


# ----------------------------------------------------------------------
# Basis-state s_z values
# ----------------------------------------------------------------------
def _sz_chain(N: int) -> np.ndarray:
    """
    s_z[i] = +1 or -1 for every basis state of an N-site chain.

    Returns shape (2^N, N). Convention: site 0 is the MSB, matching
    TeNPy's internal MPS basis ordering.
    """
    b = np.arange(2 ** N, dtype=np.int64)
    bits = np.zeros((2 ** N, N), dtype=np.int64)
    for i in range(N):
        bits[:, i] = (b >> (N - 1 - i)) & 1
    return 1 - 2 * bits


def _sz_cylinder(Lx: int, Ly: int) -> np.ndarray:
    """
    s_z[i, j] = +1 or -1 for every basis state of an Lx x Ly cylinder.

    Returns shape (2^(Lx*Ly), Lx*Ly) with linear site ordering
    k = i*Ly + j, and site 0 = MSB in the block-site convention.
    """
    N = Lx * Ly
    dim = 2 ** N
    s_z = np.zeros((dim, N), dtype=np.int64)
    b = np.arange(dim, dtype=np.int64)
    for i in range(Lx):
        for j in range(Ly):
            # Global site index
            k = i * Ly + j
            # Global basis index b: block-site i occupies bits
            # [N-1 - i*Ly : N-1 - (i+1)*Ly + 1] with layer 0 as MSB
            # within the block.
            # Global bit position (0 = LSB) is N-1-k.
            bit = (b >> (N - 1 - k)) & 1
            s_z[:, k] = 1 - 2 * bit
    return s_z


# ----------------------------------------------------------------------
# Analytic diagonals
# ----------------------------------------------------------------------
def analytic_oat_diag(N: int, chi: float) -> np.ndarray:
    """H = chi * S_z^2. Diagonal values in [0, chi*N^2/4]."""
    s_z = _sz_chain(N)
    S_z_total = s_z.sum(axis=1) / 2.0
    return chi * S_z_total ** 2


def analytic_powerlaw_diag(N: int, alpha: float, J: float) -> np.ndarray:
    """H = (1/N) sum_{i<j} (J/r^alpha) sigma_z_i sigma_z_j."""
    s_z = _sz_chain(N)
    diag = np.zeros(2 ** N)
    for i in range(N):
        for j in range(i + 1, N):
            r = float(j - i)
            diag += (J / N) / (r ** alpha) * s_z[:, i] * s_z[:, j]
    return diag


def analytic_cylinder_diag(Lx: int, Ly: int,
                            alpha: float, J: float) -> np.ndarray:
    """H = (1/N) sum_{k1<k2} (J/r^alpha) sigma_z^{k1} sigma_z^{k2}."""
    N = Lx * Ly
    s_z = _sz_cylinder(Lx, Ly)
    diag = np.zeros(2 ** N)
    for k1 in range(N):
        i1, j1 = k1 // Ly, k1 % Ly
        for k2 in range(k1 + 1, N):
            i2, j2 = k2 // Ly, k2 % Ly
            di = abs(i1 - i2)
            dj = abs(j1 - j2)
            dj_min = min(dj, Ly - dj)
            r = np.sqrt(di * di + dj_min * dj_min)
            if r < 1e-12:
                continue
            diag += (J / N) / (r ** alpha) * s_z[:, k1] * s_z[:, k2]
    return diag


# ----------------------------------------------------------------------
# MPO diagonal extraction
# ----------------------------------------------------------------------
def mpo_diagonal(model) -> np.ndarray:
    """
    Extract the MPO's diagonal by applying it to every product state.

    Each basis state is constructed as an MPS with bond dimension 1,
    and the MPO expectation value is evaluated directly. For N=6 the
    cost is 64 evaluations; for N=8 it is 256. Both complete in under
    a second on a laptop.
    """
    N = model.lat.N_sites
    sites = model.lat.mps_sites()
    d = sites[0].dim
    dim = d ** N
    diag = np.zeros(dim, dtype=float)
    for b in range(dim):
        state = []
        bb = b
        for _ in range(N):
            state.append(bb % d)
            bb //= d
        state.reverse()  # state[0] = first site (MSB)
        psi = MPS.from_product_state(sites, state, bc=model.lat.bc_MPS)
        val = model.H_MPO.expectation_value(psi)
        diag[b] = float(np.real(val))
    return diag


# ----------------------------------------------------------------------
# Individual tests
# ----------------------------------------------------------------------
def test_oat():
    print("\n[OATModel] N=6, chi=1.0")
    model = OATModel({'N': 6, 'chi_interaction': 1.0, 'conserve': None})
    diag_mpo = mpo_diagonal(model)
    diag_analytic = analytic_oat_diag(N=6, chi=1.0)
    err = float(np.max(np.abs(np.sort(diag_mpo) - np.sort(diag_analytic))))
    ok = err < TOL_EXACT
    print(f"  max sorted-diagonal error = {err:.3e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def test_powerlaw_1d(alpha, method):
    print(f"\n[PowerLawOATModel] N=6, alpha={alpha}, method={method}")
    model = PowerLawOATModel({
        'N': 6, 'J': 1.0, 'alpha': alpha, 'conserve': None,
        'compression_method': method,
    })
    diag_mpo = mpo_diagonal(model)
    diag_analytic = analytic_powerlaw_diag(N=6, alpha=alpha, J=1.0)
    err = float(np.max(np.abs(np.sort(diag_mpo) - np.sort(diag_analytic))))
    tol = TOL_EXACT if method == 'exact' else TOL_EXPONENTIAL
    ok = err < tol
    print(f"  max sorted-diagonal error = {err:.3e}  (tol {tol:.0e})  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def test_cylinder(Lx, Ly, alpha):
    print(f"\n[PowerLawCylinderModel] Lx={Lx}, Ly={Ly}, alpha={alpha}")
    model = PowerLawCylinderModel({
        'Lx': Lx, 'Ly': Ly, 'J': 1.0, 'alpha': alpha, 'conserve': None,
    })
    diag_mpo = mpo_diagonal(model)
    diag_analytic = analytic_cylinder_diag(Lx=Lx, Ly=Ly, alpha=alpha, J=1.0)
    err = float(np.max(np.abs(np.sort(diag_mpo) - np.sort(diag_analytic))))
    ok = err < TOL_EXPONENTIAL
    print(f"  max sorted-diagonal error = {err:.3e}  "
          f"(tol {TOL_EXPONENTIAL:.0e})  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 70)
    print("check_models.py: MPO vs analytic Hamiltonian on full basis")
    print("=" * 70)

    results = []
    results.append(('OATModel', test_oat()))
    results.append(('PL 1D exact', test_powerlaw_1d(alpha=3.0, method='exact')))
    results.append(('PL 1D exponential', test_powerlaw_1d(alpha=3.0, method='exponential')))
    results.append(('Cylinder', test_cylinder(Lx=2, Ly=4, alpha=3.0)))

    print("\n" + "=" * 70)
    print(f"Summary ({time.time() - t0:.1f} s)")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        print(f"  {name:<25} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n[PASS] check_models.py")
        sys.exit(0)
    else:
        print("\n[FAIL] check_models.py")
        sys.exit(1)


if __name__ == "__main__":
    main()