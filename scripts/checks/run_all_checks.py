"""
scripts/checks/run_all_checks.py

Master runner. Scans the checks/ directory at runtime and executes
each check script in sequence.

Design:
  - Discovers scripts at runtime; no hard-coded list. Scripts added
    later (check_scharnagl.py, check_exact_M.py, check_powerlaw_M.py,
    check_selector_*.py, preflight.py) are picked up automatically.
  - Skips legacy batch scripts and diagnostic utilities.
  - Passes default arguments to scripts that require them.
  - Returns 0 if all checks pass or are skipped, 1 otherwise.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time


CHECKS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CHECKS_DIR, '..', '..'))


# Scripts excluded from the run. Utilities and self.
EXCLUDE = {
    'run_all_checks.py',
    'run_updated_sanity.py',
}


# Legacy batch scripts to skip (superseded by numbered checks).
LEGACY_PATTERNS = [
    re.compile(r'^sanity_check_batch'),
    re.compile(r'^sanity_check_powerlaw_echo'),
    re.compile(r'^test_block_decomposition'),
]


# Diagnostic scripts that print but do not pass/fail.
DIAGNOSTIC = {
    'check_banuls_open.py',
}


# Scripts that require CLI arguments. Provide defaults here.
DEFAULT_ARGS = {
    'check_asymmetric_echo.py': ['--N', '8'],
}


# Per-script wall time limit (seconds). Sized for the slowest
# expected check: preflight.py at ~45 min serial once it lands.
# Individual checks still print their own summary line and will
# normally complete well under this ceiling.
TIMEOUT = 7200


def discover_checks():
    scripts = []
    for f in sorted(os.listdir(CHECKS_DIR)):
        if not f.endswith('.py'):
            continue
        if f in EXCLUDE or f in DIAGNOSTIC:
            continue
        if any(p.match(f) for p in LEGACY_PATTERNS):
            continue
        scripts.append(f)
    return scripts


def run_one(script_name):
    script_path = os.path.join(CHECKS_DIR, script_name)
    cmd = [sys.executable, script_path] + DEFAULT_ARGS.get(script_name, [])

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'

    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=ROOT_DIR, env=env,
                                timeout=TIMEOUT)
        elapsed = time.time() - t0
        print(f"[{script_name}] exit code {result.returncode} "
              f"in {elapsed:.1f} s")
        return result.returncode == 0, elapsed
    except subprocess.TimeoutExpired:
        print(f"[{script_name}] TIMEOUT after {TIMEOUT} s")
        return False, TIMEOUT
    except Exception as exc:
        print(f"[{script_name}] EXCEPTION: {exc}")
        return False, 0.0


def main():
    scripts = discover_checks()
    print(f"Discovered {len(scripts)} check scripts in {CHECKS_DIR}:")
    for s in scripts:
        print(f"  {s}")

    results = {}
    times = {}
    for s in scripts:
        ok, elapsed = run_one(s)
        results[s] = ok
        times[s] = elapsed

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    n_pass = 0
    n_fail = 0
    for s in scripts:
        status = "PASS" if results[s] else "FAIL"
        print(f"  {s:<50} {status:>5}  ({times[s]:.1f} s)")
        if results[s]:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed")

    if n_fail == 0:
        print("\n[PASS] run_all_checks.py")
        sys.exit(0)
    else:
        print("\n[FAIL] run_all_checks.py")
        sys.exit(1)


if __name__ == "__main__":
    main()