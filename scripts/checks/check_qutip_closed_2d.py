"""
scripts/checks/check_qutip_closed_2d.py

Two-stage validation of the exact 2D cylinder echo worker.

Stage 1 (alpha = 0, independent of any TN worker)
-------------------------------------------------
Build the 2D Hamiltonian at Lx=2, Ly=4 (so N=8) and J=1.

Verify:
  (a) The minimum eigenvalue of the full 2D Hamiltonian equals the
      analytical collective-spin ground-state energy
          E_min = -J/2              (even N),
          E_min = J/(2N) - J/2      (odd N)
      to absolute error < 1e-8.
  (b) The multiset of eigenvalues of H restricted to the maximal-spin
      sector S = N/2 equals the Dicke spectrum
          E(m) = (2J/N) m^2 - J/2,   m ∈ {-N/2, ..., N/2}
      to absolute error < 1e-8.
  (c) Bonus sanity check: the coherent-state SNR at (mu, nu) = (0, 0)
      equals sqrt(N).

The projection to the maximal-spin sector is performed by finding the
eigenvectors of S^2 = Sx^2 + Sy^2 + Sz^2 with eigenvalue
S(S+1) = (N/2)(N/2+1).

Stage 2 (alpha = 3, TN comparison)
----------------------------------
Run the exact worker and the TN cylinder worker
(`scripts/powerlaw_cylinder_worker.py`, created in the documentation) at
Lx=2, Ly=4, alpha=3, mu=pi/2, nu=-pi/2, chi_max=32. Compare the two
SNRs to relative error < 1e-4.

If the TN worker file does not yet exist, Stage 2 is deferred with a
clear message and is not counted as a failure. The exact worker's SNR
is still printed so that a manual comparison can be made once the TN
worker becomes available.
"""

from __future__ import annotations

import os
import sys
import subprocess

import numpy as np
import qutip


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
for p in (ROOT_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from exact_echo_worker_2d import (           # noqa: E402
    build_hamiltonian_2d,
    build_collective_spin_ops,
    build_css,
    build_pauli_at,
    _expm_neg_i_t_H,
    compute_M,
    compute_Q,
    compute_snr,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
Lx_TEST = 2
Ly_TEST = 4
J_TEST = 1.0
TOL_STAGE1 = 1e-8
TOL_STAGE2 = 1e-4
CHI_MAX_STAGE2 = 32
MU_TEST = np.pi / 2
NU_TEST = -np.pi / 2
ALPHA_STAGE2 = 3.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def symmetric_subspace_projector(N: int) -> np.ndarray:
    """
    Return a (2^N, N+1) array whose columns form an orthonormal basis
    for the maximal-spin sector S = N/2 of the collective spin algebra.
    """
    sx, sy, sz = qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()
    Sx = qutip.qzero([2] * N)
    Sy = qutip.qzero([2] * N)
    Sz = qutip.qzero([2] * N)
    for i in range(N):
        Sx = Sx + build_pauli_at(i, sx, N)
        Sy = Sy + build_pauli_at(i, sy, N)
        Sz = Sz + build_pauli_at(i, sz, N)
    Sx, Sy, Sz = Sx / 2.0, Sy / 2.0, Sz / 2.0
    S2 = Sx * Sx + Sy * Sy + Sz * Sz

    evals, evecs = np.linalg.eigh(S2.full())
    target = (N / 2.0) * (N / 2.0 + 1.0)
    mask = np.isclose(evals, target, atol=1e-10)
    return evecs[:, mask]


# ----------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------
def stage1_alpha0() -> bool:
    print("\nStage 1: alpha=0 analytical validation")
    N = Lx_TEST * Ly_TEST
    print(f"  Geometry: Lx={Lx_TEST}, Ly={Ly_TEST}, N={N}")

    H = build_hamiltonian_2d(Lx_TEST, Ly_TEST, J_TEST, alpha=0.0)
    H_dense = H.full()

    # --- (a) minimum eigenvalue ---
    evals = np.linalg.eigvalsh(H_dense)
    E_min_num = float(evals[0])
    if N % 2 == 0:
        E_min_ana = -J_TEST / 2.0
    else:
        E_min_ana = J_TEST / (2.0 * N) - J_TEST / 2.0

    err_min = abs(E_min_num - E_min_ana)
    pass_min = err_min < TOL_STAGE1
    print(f"  (a) Min eigenvalue numerical  = {E_min_num:+.12f}")
    print(f"      Min eigenvalue analytical = {E_min_ana:+.12f}")
    print(f"      Absolute error            = {err_min:.3e}   "
          f"[{'PASS' if pass_min else 'FAIL'}]")

    # --- (b) symmetric-subspace spectrum vs Dicke ---
    P = symmetric_subspace_projector(N)
    H_sym = P.conj().T @ H_dense @ P
    evals_sym = np.sort(np.linalg.eigvalsh(H_sym))

    m_vals = np.arange(-N / 2.0, N / 2.0 + 1.0)
    dicke_evals = np.sort((2.0 * J_TEST / N) * m_vals ** 2 - J_TEST / 2.0)

    err_spec = float(np.max(np.abs(evals_sym - dicke_evals)))
    pass_spec = err_spec < TOL_STAGE1
    print(f"  (b) Symmetric subspace: {N + 1} eigenvalues")
    print(f"      Max abs error vs Dicke    = {err_spec:.3e}   "
          f"[{'PASS' if pass_spec else 'FAIL'}]")

    # --- (c) coherent-state SNR at mu = nu = 0 ---
    #
    # The finite-difference step for this sub-check is delta=1e-4 rather
    # than the worker's default 1e-3. The central-difference error scales
    # as delta^2, and at delta=1e-3 the SNR error is ~4.7e-7 (driven by
    # the ~delta^2/6 relative error of the sin(delta)/delta approximation
    # to d<Sz>/dphi), which exceeds the 1e-8 tolerance the codebase assigns
    # to this sub-check. At delta=1e-4 the error drops to ~4.7e-9, well
    # below tolerance. The worker keeps delta=1e-3 for consistency with
    # the stated finite-difference convention; the smaller step
    # is used only for this isolated analytical benchmark, where round-off
    # is negligible (O(1e-15) against a signal of O(1e-4)).
    S_ops = build_collective_spin_ops(N)
    psi_0 = build_css(N)

    M = compute_M(S_ops, psi_0, qutip.qeye(H.dims[0]), delta=1e-4)
    Q = compute_Q(S_ops, psi_0)
    snr = compute_snr(M, Q)
    snr_expected = float(np.sqrt(N))
    err_snr = abs(snr - snr_expected)
    pass_snr = err_snr < TOL_STAGE1
    print(f"  (c) Coherent-state SNR        = {snr:.10f}")
    print(f"      Expected sqrt(N)          = {snr_expected:.10f}")
    print(f"      Absolute error            = {err_snr:.3e}   "
          f"[{'PASS' if pass_snr else 'FAIL'}]")

    stage_pass = pass_min and pass_spec and pass_snr
    print(f"  Stage 1: {'PASS' if stage_pass else 'FAIL'}")
    return stage_pass


# ----------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------
def stage2_alpha3() -> bool | None:
    print("\nStage 2: alpha=3 vs TN cylinder worker")

    tn_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_2d', 'powerlaw_cylinder_worker.py')
    exact_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_2d', 'exact_echo_worker_2d.py')

    # Always run the exact worker, so the SNR is printed even if the
    # TN comparison cannot yet be done.
    cmd_exact = [
        sys.executable, exact_worker,
        "--Lx", str(Lx_TEST), "--Ly", str(Ly_TEST),
        "--alpha", str(ALPHA_STAGE2), "--J", str(J_TEST),
        "--mu", f"{MU_TEST:.6f}", "--nu", f"{NU_TEST:.6f}",
    ]
    print(f"  Running: {' '.join(cmd_exact)}")
    subprocess.run(cmd_exact, check=True)

    exact_file = os.path.join(
        ROOT_DIR, 'data', 'exact_2d_closed',
        f"exact_echo_2d_Lx{Lx_TEST}_Ly{Ly_TEST}"
        f"_alpha{ALPHA_STAGE2}_J{J_TEST}"
        f"_mu{MU_TEST:.4f}_nu{NU_TEST:.4f}.npy"
    )
    exact_data = np.load(exact_file, allow_pickle=True).item()
    snr_exact = exact_data['snr']
    print(f"  Exact SNR (Lx={Lx_TEST}, Ly={Ly_TEST}, alpha={ALPHA_STAGE2}) "
          f"= {snr_exact:.8f}")

    if not os.path.exists(tn_worker):
        print("  powerlaw_cylinder_worker.py not yet implemented.")
        print("  Scheduled for the documentation. Deferring the TN comparison.")
        return None

    # TN worker exists: run it and compare.
    cmd_tn = [
        sys.executable, tn_worker,
        "--Lx", str(Lx_TEST), "--Ly", str(Ly_TEST),
        "--alpha", str(ALPHA_STAGE2), "--J", str(J_TEST),
        "--mu", f"{MU_TEST:.6f}", "--nu", f"{NU_TEST:.6f}",
        "--chi_max", str(CHI_MAX_STAGE2),
    ]
    print(f"  Running: {' '.join(cmd_tn)}")
    subprocess.run(cmd_tn, check=True)

    tn_file = os.path.join(
        ROOT_DIR, 'data', 'powerlaw_cylinder_pts',
        f"plaw_cyl_Lx{Lx_TEST}_Ly{Ly_TEST}"
        f"_chi{CHI_MAX_STAGE2}_alpha{ALPHA_STAGE2}"
        f"_mu{MU_TEST:.4f}_nu{NU_TEST:.4f}.npy"
    )
    tn_data = np.load(tn_file, allow_pickle=True).item()
    snr_tn = tn_data['snr']

    denom = max(abs(snr_exact), 1e-30)
    rel_err = abs(snr_exact - snr_tn) / denom
    print(f"  Exact SNR = {snr_exact:.8f}")
    print(f"  TN SNR    = {snr_tn:.8f}")
    print(f"  Relative error = {rel_err:.3e}  (tol = {TOL_STAGE2})")

    stage_pass = rel_err < TOL_STAGE2
    print(f"  Stage 2: {'PASS' if stage_pass else 'FAIL'}")
    return stage_pass


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Check 2D (closed): exact-diagonalisation validation")
    print("=" * 70)

    stage1_pass = stage1_alpha0()
    stage2_pass = stage2_alpha3()

    print("\nSummary:")
    print(f"  Stage 1: {'PASS' if stage1_pass else 'FAIL'}")
    if stage2_pass is None:
        print("  Stage 2: DEFERRED (TN worker not yet implemented)")
    else:
        print(f"  Stage 2: {'PASS' if stage2_pass else 'FAIL'}")

    all_pass = stage1_pass and (stage2_pass is not False)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()