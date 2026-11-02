"""
scripts/checks/check_observables.py

Compares the observables module against exact diagonalization on a
small entangled state.

Tests:
  1. _get_3x3_covariance matches ED to < 1e-8
  2. compute_squeezing matches ED to < 1e-8
  3. compute_qfi matches ED to < 1e-8
  4. compute_entanglement_entropy matches ED to < 1e-8
  5. compute_qfi_mixed_bounds, if present, returns (F_Q, F_Q) on
     pure-state input and satisfies F_Q^(var) <= F_Q^(avg) on a
     mixture; skips cleanly if the function does not exist.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from src.models import OATModel
from src.evolution import initialize_css
from src.observables import (
    _get_3x3_covariance, compute_squeezing, compute_qfi,
    compute_entanglement_entropy,
)

try:
    from src.observables import compute_qfi_mixed_bounds
    HAVE_MIXED_QFI = True
except ImportError:
    HAVE_MIXED_QFI = False


TOL = 1e-8
N = 6


# ----------------------------------------------------------------------
# Test state
# ----------------------------------------------------------------------
def build_entangled_state(model, t_total=0.1, dt=0.02, chi_max=32):
    """OAT-evolved CSS: small entanglement, nonzero squeezing."""
    psi = initialize_css(model.lat)
    N_steps = int(np.ceil(t_total / dt))
    mpo_params = {
        'dt': t_total / N_steps,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12},
    }
    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = N_steps
    engine.run()
    return psi


# ----------------------------------------------------------------------
# Dense ED reference
# ----------------------------------------------------------------------
def build_collective_S(axis, N):
    pauli = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
    }[axis]
    dim = 2 ** N
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[i] = 0.5 * pauli
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        S += term
    return S


def mps_to_vector(psi, N):
    theta = psi.get_theta(0, n=N)
    arr = theta.to_ndarray()
    while arr.ndim > N and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.reshape(2 ** N)


def ed_observables(psi, N):
    vec = mps_to_vector(psi, N)
    S_ops = [build_collective_S(a, N) for a in 'xyz']

    S_vec = np.array([np.real(vec.conj() @ S @ vec) for S in S_ops])

    Cov = np.zeros((3, 3), dtype=complex)
    for i, Si in enumerate(S_ops):
        for j, Sj in enumerate(S_ops):
            corr = vec.conj() @ (Si @ Sj) @ vec
            Cov[i, j] = corr - S_vec[i] * S_vec[j]
    Cov = 0.5 * (Cov + Cov.conj().T)
    Cov = np.real(Cov)

    # Squeezing
    n_spin_sq = np.sum(S_vec ** 2)
    n_vec = S_vec / np.sqrt(n_spin_sq)
    P_perp = np.eye(3) - np.outer(n_vec, n_vec)
    Cov_perp = P_perp @ Cov @ P_perp
    evals = np.sort(np.linalg.eigvalsh(Cov_perp))
    xi2 = N * evals[1] / n_spin_sq

    # QFI
    qfi = 4.0 * np.max(np.linalg.eigvalsh(Cov))

    # Half-chain entanglement entropy
    half = N // 2
    psi_mat = vec.reshape(2 ** half, 2 ** (N - half))
    s = np.linalg.svd(psi_mat, compute_uv=False)
    probs = s ** 2
    probs = probs[probs > 1e-15]
    ent = -np.sum(probs * np.log(probs))

    return S_vec, Cov, xi2, qfi, ent


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_observables_vs_ed(model):
    print(f"\n[Observables vs ED] N={N}")
    psi = build_entangled_state(model)

    S_vec_ed, Cov_ed, xi2_ed, qfi_ed, ent_ed = ed_observables(psi, N)

    S_vec_mps, Cov_mps = _get_3x3_covariance(psi)
    xi2_mps = compute_squeezing(psi)
    qfi_mps = compute_qfi(psi)
    ent_mps = compute_entanglement_entropy(psi)

    err_cov = float(np.max(np.abs(Cov_mps - Cov_ed)))
    err_xi2 = abs(xi2_mps - xi2_ed)
    err_qfi = abs(qfi_mps - qfi_ed)
    err_ent = abs(ent_mps - ent_ed)

    print(f"  |Δ Cov|_max = {err_cov:.3e}  [{'PASS' if err_cov < TOL else 'FAIL'}]")
    print(f"  |Δ ξ_R²|    = {err_xi2:.3e}  [{'PASS' if err_xi2 < TOL else 'FAIL'}]")
    print(f"  |Δ F_Q|     = {err_qfi:.3e}  [{'PASS' if err_qfi < TOL else 'FAIL'}]")
    print(f"  |Δ S_vN|    = {err_ent:.3e}  [{'PASS' if err_ent < TOL else 'FAIL'}]")

    return (err_cov < TOL and err_xi2 < TOL
            and err_qfi < TOL and err_ent < TOL)


def test_mixed_qfi(model):
    if not HAVE_MIXED_QFI:
        print("\n[compute_qfi_mixed_bounds] SKIPPED (function not in src/observables.py)")
        return None

    print("\n[compute_qfi_mixed_bounds]")
    psi1 = build_entangled_state(model, t_total=0.1)
    psi2 = build_entangled_state(model, t_total=0.2)

    # Pure-state input: diagnostic and upper bound should both equal
    # the pure-state QFI
    def _stats(psi):
        S_vec, Cov = _get_3x3_covariance(psi)
        return Cov, S_vec

    cov1, sv1 = _stats(psi1)
    cov2, sv2 = _stats(psi2)

    F_var, F_avg = compute_qfi_mixed_bounds([cov1], [sv1])
    qfi_pure = compute_qfi(psi1)
    err_pure = max(abs(F_var - qfi_pure), abs(F_avg - qfi_pure))
    ok_pure = err_pure < TOL
    print(f"  pure-state input: |Δ| = {err_pure:.3e}  "
          f"[{'PASS' if ok_pure else 'FAIL'}]")

    F_var2, F_avg2 = compute_qfi_mixed_bounds([cov1, cov2], [sv1, sv2])
    ok_order = F_var2 <= F_avg2 + 1e-10
    print(f"  mixture ordering: F_var = {F_var2:.6f}, "
          f"F_avg = {F_avg2:.6f}  "
          f"[{'PASS' if ok_order else 'FAIL'}]")

    return ok_pure and ok_order


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 70)
    print("check_observables.py: MPS observables vs exact diagonalization")
    print("=" * 70)

    model = OATModel({'N': N, 'chi_interaction': 1.0, 'conserve': None})

    results = []
    results.append(('Observables vs ED', test_observables_vs_ed(model)))
    results.append(('Mixed-state QFI', test_mixed_qfi(model)))

    print("\n" + "=" * 70)
    print(f"Summary ({time.time() - t0:.1f} s)")
    print("=" * 70)
    all_pass = True
    for name, result in results:
        if result is None:
            print(f"  {name:<25} SKIPPED")
        else:
            print(f"  {name:<25} {'PASS' if result else 'FAIL'}")
            if not result:
                all_pass = False

    if all_pass:
        print("\n[PASS] check_observables.py")
        sys.exit(0)
    else:
        print("\n[FAIL] check_observables.py")
        sys.exit(1)


if __name__ == "__main__":
    main()