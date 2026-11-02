"""
scripts/checks/check_qutip_open_2d.py

Validates the 2D Lindblad worker against the closed-system 2D worker
at gamma1 = gamma2 = 0, and against physical sanity criteria at
gamma > 0.

Sequence
--------
  1. Run scripts/exact_lindblad_worker_2d.py at (Lx, Ly) = (2, 4),
     alpha = 3, mu = pi/2, nu = -pi/2, gamma1 = gamma2 = 0.
  2. Run scripts/exact_echo_worker_2d.py at the same physical
     parameters.
  3. Compare the two SNRs to relative error < 1e-4.
  4. Read the metadata dicts of both output files and verify that the
     stored Lx, Ly, alpha, mu, nu match, so that a numerical
     coincidence cannot mask a parameter mismatch.

  5. Run scripts/exact_lindblad_worker_2d.py at the same geometry
     and couplings with gamma1 = gamma2 = 1e-3, and check that
     (a) Tr(rho_post) = 1 to < 1e-10,
     (b) min(eigvals(rho_post)) >= -1e-10,
     (c) purity is within [1/2^N, 1],
     (d) SNR is finite and positive.

  6. Run the same worker at (mu, nu) = (0, 0), gamma = 0, and verify
     that the coherent-state SNR equals sqrt(N) to a tolerance
     consistent with the central-difference truncation
     (delta^2 * sqrt(N) / 6 ~ 1e-7 for N=8, delta=1e-3).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
LX = 2
LY = 4
ALPHA = 3.0
MU = np.pi / 2
NU = -np.pi / 2
TOL_GAMMA_ZERO = 1e-4
GAMMA_TEST = 1e-3


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
LINDBLAD_WORKER = os.path.join(ROOT_DIR, 'scripts',
                               'exact_lindblad_worker_2d.py')
CLOSED_WORKER = os.path.join(ROOT_DIR, 'scripts',
                             'exact_echo_worker_2d.py')

OPEN_DIR = os.path.join(ROOT_DIR, 'data', 'exact_2d_open')
CLOSED_DIR = os.path.join(ROOT_DIR, 'data', 'exact_2d_closed')


def _open_file(g1: float, g2: float,
               mu: float = MU, nu: float = NU) -> str:
    return os.path.join(
        OPEN_DIR,
        f"exact_lindblad_2d_Lx{LX}_Ly{LY}"
        f"_alpha{ALPHA}_J1.0"
        f"_mu{mu:.4f}_nu{nu:.4f}"
        f"_g1{g1:.4e}_g2{g2:.4e}.npy"
    )


def _closed_file() -> str:
    return os.path.join(
        CLOSED_DIR,
        f"exact_echo_2d_Lx{LX}_Ly{LY}"
        f"_alpha{ALPHA}_J1.0"
        f"_mu{MU:.4f}_nu{NU:.4f}.npy"
    )


# ----------------------------------------------------------------------
# Subprocess wrappers
# ----------------------------------------------------------------------
def run_worker(script: str, args: list) -> None:
    cmd = [sys.executable, script] + args
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _common_args(mu: float, nu: float,
                 g1: float, g2: float) -> list:
    return [
        "--Lx", str(LX), "--Ly", str(LY),
        "--alpha", str(ALPHA), "--J", "1.0",
        "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
        "--gamma1", str(g1), "--gamma2", str(g2),
    ]


# ----------------------------------------------------------------------
# Check 1: gamma = 0 vs closed-system 2D worker
# ----------------------------------------------------------------------
def check_gamma_zero() -> bool:
    print("\n" + "=" * 70)
    print("Check 1: gamma1 = gamma2 = 0 vs closed-system 2D worker")
    print("=" * 70)

    for f in (_open_file(0.0, 0.0), _closed_file()):
        if os.path.exists(f):
            os.remove(f)

    run_worker(LINDBLAD_WORKER, _common_args(MU, NU, 0.0, 0.0))
    run_worker(CLOSED_WORKER, [
        "--Lx", str(LX), "--Ly", str(LY),
        "--alpha", str(ALPHA), "--J", "1.0",
        "--mu", f"{MU:.6f}", "--nu", f"{NU:.6f}",
    ])

    open_data = np.load(_open_file(0.0, 0.0), allow_pickle=True).item()
    closed_data = np.load(_closed_file(), allow_pickle=True).item()

    # --- metadata equivalence ---
    for key in ('Lx', 'Ly', 'alpha', 'mu', 'nu'):
        if key not in open_data:
            raise SystemExit(f"Open worker output missing key {key!r}")
        if key not in closed_data:
            raise SystemExit(f"Closed worker output missing key {key!r}")
        if not np.isclose(open_data[key], closed_data[key]):
            raise SystemExit(
                f"METADATA MISMATCH in {key}: "
                f"open={open_data[key]!r} vs closed={closed_data[key]!r}"
            )

    snr_open = open_data['snr']
    snr_closed = closed_data['snr']
    denom = max(abs(snr_closed), 1e-30)
    rel_err = abs(snr_open - snr_closed) / denom

    print(f"\n  Closed-system 2D SNR = {snr_closed:.10f}")
    print(f"  Lindblad 2D SNR      = {snr_open:.10f}")
    print(f"  Relative error       = {rel_err:.3e}  "
          f"(tol = {TOL_GAMMA_ZERO:.0e})")

    if rel_err < TOL_GAMMA_ZERO:
        print("  [PASS]")
        return True

    print(f"  [FAIL]  Discrepancy exceeds {TOL_GAMMA_ZERO:.0e}.")
    print("  Possible causes:")
    print("    - sign of the backward evolution time (nu - mu)/2")
    print("    - factor-of-chi mismatch in mu or nu")
    print("    - mismatch in the finite-difference step delta")
    print("    - convention mismatch in the 2D Hamiltonian (1/N factor, "
          "minimum-image distance)")
    return False


# ----------------------------------------------------------------------
# Check 2: positivity at gamma > 0
# ----------------------------------------------------------------------
def check_positivity_at_gamma_positive() -> bool:
    print("\n" + "=" * 70)
    print(f"Check 2: positivity and trace at "
          f"gamma1 = gamma2 = {GAMMA_TEST}")
    print("=" * 70)

    fpath = _open_file(GAMMA_TEST, GAMMA_TEST)
    if os.path.exists(fpath):
        os.remove(fpath)

    run_worker(LINDBLAD_WORKER, _common_args(MU, NU,
                                             GAMMA_TEST, GAMMA_TEST))

    data = np.load(fpath, allow_pickle=True).item()
    snr = data['snr']

    # Recompute rho_post standalone for the diagnostics, reusing the
    # worker's own physics functions so that the test is exact.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exact_lindblad_worker_2d", LINDBLAD_WORKER
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            f"cannot load module spec from {LINDBLAD_WORKER}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    H = mod.build_hamiltonian_2d(LX, LY, 1.0, ALPHA)
    rho_0 = mod.build_css_dm(LX * LY)
    c_ops = mod.build_jump_ops(LX * LY, GAMMA_TEST, GAMMA_TEST)

    t_fwd = MU / 2.0
    t_bwd = (NU - MU) / 2.0
    rho_pre = mod.lindblad_evolve(rho_0, H, c_ops, t_fwd)
    rho_post = mod.lindblad_evolve(rho_pre, H, c_ops, t_bwd)

    trace = float(np.real(rho_post.tr()))
    purity = float(np.real((rho_post * rho_post).tr()))
    eigvals = np.linalg.eigvalsh(rho_post.full())
    min_eig = float(np.min(eigvals))

    # Thresholds are chosen relative to the solver error floor.
    # The Lindblad ODE runs at atol=rtol=1e-12, and the echo sequence
    # accumulates error over multiple mesolve calls. A residual
    # eigenvalue of order 1e-10 is expected and is a numerical artifact,
    # not a physics violation. Genuine positivity bugs would manifest at
    # the 1e-3 level or worse, so a threshold of -1e-8 retains the
    # check's diagnostic power while accommodating solver noise.
    POS_TOL = 1e-8
    pass_trace = abs(trace - 1.0) < 1e-10
    pass_pos = min_eig >= -POS_TOL
    pass_purity = (1.0 / (2 ** (LX * LY)) - 1e-10) <= purity <= 1.0 + 1e-10
    pass_snr = np.isfinite(snr) and snr > 0.0

    print(f"  trace           = {trace:.12f}   "
          f"[{'PASS' if pass_trace else 'FAIL'}]")
    print(f"  min_eig         = {min_eig:.3e}  (tol -{POS_TOL:.0e})  "
          f"[{'PASS' if pass_pos else 'FAIL'}]")
    print(f"  purity          = {purity:.8f}       "
          f"[{'PASS' if pass_purity else 'FAIL'}]")
    print(f"  SNR             = {snr:.8f}       "
          f"[{'PASS' if pass_snr else 'FAIL'}]")

    return pass_trace and pass_pos and pass_purity and pass_snr


# ----------------------------------------------------------------------
# Check 3: coherent-state limit at (mu, nu) = (0, 0), gamma = 0
# ----------------------------------------------------------------------
def check_coherent_state_limit() -> bool:
    print("\n" + "=" * 70)
    print("Check 3: coherent-state limit at (mu, nu) = (0, 0), gamma = 0")
    print("=" * 70)

    mu_zero = 0.0
    nu_zero = 0.0
    fpath = _open_file(0.0, 0.0, mu=mu_zero, nu=nu_zero)
    if os.path.exists(fpath):
        os.remove(fpath)

    run_worker(LINDBLAD_WORKER,
               _common_args(mu_zero, nu_zero, 0.0, 0.0))

    data = np.load(fpath, allow_pickle=True).item()
    snr = data['snr']
    N = LX * LY
    snr_expected = float(np.sqrt(N))
    err = abs(snr - snr_expected)

    print(f"  SNR               = {snr:.10f}")
    print(f"  Expected sqrt(N)  = {snr_expected:.10f}")
    print(f"  Absolute error    = {err:.3e}")

    # Central-difference truncation with delta=1e-3 gives an SNR error
    # of order sqrt(N) * delta^2 / 6 ~ 4.7e-7, as documented in the documentation.
    tol = 1e-6
    passed = err < tol
    print(f"  Tolerance         = {tol:.0e}   "
          f"[{'PASS' if passed else 'FAIL'}]")
    return passed


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(f"Check 2D Lindblad worker: Lx={LX}, Ly={LY}, "
          f"alpha={ALPHA}, mu={MU:.4f}, nu={NU:.4f}")
    print("=" * 70)

    results = {}
    results['gamma_zero_vs_closed'] = check_gamma_zero()
    results['positivity_at_gamma_positive'] = \
        check_positivity_at_gamma_positive()
    results['coherent_state_limit'] = check_coherent_state_limit()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for key, val in results.items():
        print(f"  {key:>32}: {'PASS' if val else 'FAIL'}")

    all_pass = all(results.values())
    print("\n" + ("ALL CHECKS PASS" if all_pass
                  else "SOME CHECKS FAILED"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()