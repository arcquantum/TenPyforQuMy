"""
scripts/checks/check_banuls_open.py

Compares the Bañuls worker against the exact Lindblad worker of
the documentation at gamma1 = gamma2 = 0.01, N=8, alpha=1.

Expected: the Bañuls worker as currently implemented applies only
the no-jump dissipative contraction without the stochastic jumps,
so its trace is not preserved. The SNR will not match Lindblad.
This check documents the failure and motivates the MCWF extension.
"""

import os
import sys
import numpy as np
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

N = 8
ALPHA = 1.0
MU = np.pi / 2
NU = -np.pi / 2
GAMMA1 = 0.01
GAMMA2 = 0.01
DISENTANGLE_EVERY = 2
CHI_MAX = 32

BANULS_WORKER = os.path.join(ROOT_DIR, 'scripts', 'workers', 'open_1d', 'banuls_worker.py')
LINDBLAD_FILE = os.path.join(
    ROOT_DIR, 'data', 'exact_1d_open',
    f"exact_lindblad_1d_N{N}_alpha{ALPHA}_J1.0"
    f"_mu{MU:.4f}_nu{NU:.4f}"
    f"_g1{GAMMA1:.4e}_g2{GAMMA2:.4e}.npy"
)


def main():
    if not os.path.exists(LINDBLAD_FILE):
        print(f"[FAIL] Reference file missing: {LINDBLAD_FILE}")
        print("       Run check_qutip_open.py first.")
        sys.exit(1)

    lindblad = np.load(LINDBLAD_FILE, allow_pickle=True).item()
    snr_lindblad = lindblad['snr']
    print(f"Exact Lindblad SNR = {snr_lindblad:.6f}")

    cmd = [
        sys.executable, BANULS_WORKER,
        "--N", str(N), "--chi_max", str(CHI_MAX),
        "--alpha", str(ALPHA),
        "--mu", f"{MU:.6f}", "--nu", f"{NU:.6f}",
        "--gamma1", str(GAMMA1), "--gamma2", str(GAMMA2),
        "--disentangle_every", str(DISENTANGLE_EVERY),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    banuls_file = os.path.join(
        ROOT_DIR, 'data', 'banuls_pts',
        f"banuls_N{N}_chi{CHI_MAX}_alpha{ALPHA}"
        f"_mu{MU:.4f}_nu{NU:.4f}"
        f"_g1{GAMMA1:.4e}_g2{GAMMA2:.4e}"
        f"_eta0.05.npy"
    )
    banuls = np.load(banuls_file, allow_pickle=True).item()
    snr_banuls = banuls['snr']

    print(f"Bañuls SNR         = {snr_banuls:.6f}")
    rel_err = abs(snr_banuls - snr_lindblad) / abs(snr_lindblad)
    print(f"Relative error     = {rel_err:.3e}")

    # We do not assert a tolerance here. This check is diagnostic:
    # we record the observed discrepancy and print it for the log.
    print()
    print("Note: this check is diagnostic, not pass/fail. The Bañuls")
    print("worker currently lacks the MCWF stochastic jumps, so a")
    print("mismatch is expected. The observed value quantifies the")
    print("error introduced by omitting the jumps.")


if __name__ == "__main__":
    main()