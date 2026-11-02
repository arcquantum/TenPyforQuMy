"""
scripts/checks/check_banuls.py

Validates the Bañuls worker at γ1 = γ2 = 0 against the closed-system
worker of the documentation. At zero dissipation the Bañuls worker should
reduce to closed-system evolution, and the SNR should agree with the
closed-system worker to relative error < 10^{-3}.

This is the required sanity check  of the simulation
plan. It is intentionally a thin wrapper: at γ=0, the Bañuls
disentangling step is expected to find no useful factorization (the
state is pure and its MPS representation is already optimal), so the
algorithm should reproduce the closed-system SNR essentially exactly.
"""

import os
import subprocess
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

N = 8
ALPHA = 1.0
MU = np.pi / 2
NU = -np.pi / 2
CHI_MAX = 32
TOL = 1e-3

BANULS_WORKER = os.path.join(ROOT_DIR, 'scripts', 'workers', 'open_1d', 'banuls_worker.py')
CLOSED_WORKER = os.path.join(ROOT_DIR, 'scripts', 'workers', 'closed_1d', 'powerlaw_echo_worker.py')

BANULS_DIR = os.path.join(ROOT_DIR, 'data', 'banuls_pts')
CLOSED_DIR = os.path.join(ROOT_DIR, 'data', 'powerlaw_echo_pts')

BANULS_FILE = os.path.join(
    BANULS_DIR,
    f"banuls_N{N}_chi{CHI_MAX}_alpha{ALPHA}"
    f"_mu{MU:.4f}_nu{NU:.4f}"
    f"_g10.0000e+00_g20.0000e+00_eta0.05.npy"
)
CLOSED_FILE = os.path.join(
    CLOSED_DIR,
    f"plaw_echo_N{N}_chi{CHI_MAX}_alpha{ALPHA}"
    f"_mu{MU:.4f}_nu{NU:.4f}.npy"
)


def run(script, args):
    cmd = [sys.executable, script] + args
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    print("=" * 70)
    print(f"Check Bañuls γ=0 vs closed-system, N={N}, alpha={ALPHA}, "
          f"mu={MU:.4f}, nu={NU:.4f}, chi_max={CHI_MAX}")
    print("=" * 70)

    for f in (BANULS_FILE, CLOSED_FILE):
        if os.path.exists(f):
            os.remove(f)

    run(BANULS_WORKER, [
        "--N", str(N),
        "--chi_max", str(CHI_MAX),
        "--alpha", str(ALPHA),
        "--mu", f"{MU:.6f}",
        "--nu", f"{NU:.6f}",
        "--gamma1", "0.0",
        "--gamma2", "0.0",
    ])

    run(CLOSED_WORKER, [
        "--N", str(N),
        "--chi_max", str(CHI_MAX),
        "--alpha", str(ALPHA),
        "--mu", f"{MU:.6f}",
        "--nu", f"{NU:.6f}",
    ])

    banuls_data = np.load(BANULS_FILE, allow_pickle=True).item()
    closed_data = np.load(CLOSED_FILE, allow_pickle=True).item()

    snr_banuls = banuls_data['snr']
    snr_closed = closed_data['snr']

    for key in ('N', 'alpha', 'mu', 'nu'):
        if not np.isclose(banuls_data[key], closed_data[key]):
            raise SystemExit(
                f"METADATA MISMATCH in {key}: "
                f"banuls={banuls_data[key]!r} vs closed={closed_data[key]!r}"
            )

    denom = max(abs(snr_closed), 1e-30)
    rel_err = abs(snr_banuls - snr_closed) / denom

    print()
    print(f"  Closed-system SNR = {snr_closed:.10f}")
    print(f"  Bañuls SNR        = {snr_banuls:.10f}")
    print(f"  Relative error    = {rel_err:.3e}  (tol = {TOL:.0e})")

    if rel_err < TOL:
        print()
        print(f"[PASS] Bañuls reproduces the closed-system SNR to {TOL}.")
        sys.exit(0)

    print()
    print(f"[FAIL] Discrepancy exceeds {TOL:.0e}.")
    print("  Expected cause: at γ=0 the Bañuls worker should reduce to")
    print("  closed-system evolution. If the disagreement is at the")
    print("  few-percent level, the disentangling step is "
          "corrupting the state.")
    sys.exit(1)


if __name__ == "__main__":
    main()