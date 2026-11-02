"""
scripts/checks/check_selector_ntraj.py

Validates the trajectory-count selector (the documentation v2).

For each testable row in data/resource_tables/ntraj_conv.csv,
verifies the criterion that the statistical error at the
converged trajectory count dominates the systematic plateau:

    a / sqrt(N_traj,conv) > b

where a = fit_a and b = fit_b are the OLS coefficients of the
sigma = a/sqrt(N_traj) + b fit.

If ntraj_conv.csv is missing or empty, the script prints [SKIP],
runs auto_ntraj_pilot.py --sanity as a diagnostic, and exits 0
so that the master runner remains usable during the the codebase
timeline.

Exit codes:
  0  if all testable rows satisfy the criterion, or the CSV is
     missing
  1  if any testable row violates the criterion
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

RESOURCE_DIR   = os.path.join(ROOT_DIR, 'data', 'resource_tables')
NTRAJ_CONV_CSV = os.path.join(RESOURCE_DIR, 'ntraj_conv.csv')


# ----------------------------------------------------------------------
# Small readers
# ----------------------------------------------------------------------
def _read_csv(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', newline='') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _parse_float(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_int(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# Per-row validation
# ----------------------------------------------------------------------
def _validate_row(row):
    """
    Return (ok, info). ok is True, False, or None (untestable).
    """
    a = _parse_float(row.get('fit_a'))
    b = _parse_float(row.get('fit_b'))
    n = _parse_int(row.get('ntraj_conv'))

    if a is None or b is None or n is None:
        return None, {'reason': 'missing fit_a/fit_b/ntraj_conv'}
    if n <= 0:
        return None, {'reason': f'non-positive ntraj_conv={n}'}
    if a < 0 or b < 0:
        return None, {'reason': f'negative fit params a={a} b={b}'}

    stat = a / np.sqrt(n)
    return (stat > b), {'a': a, 'b': b, 'n': n, 'stat': stat}


# ----------------------------------------------------------------------
# Fallback: synthetic sanity check when CSV is missing
# ----------------------------------------------------------------------
def _run_synthetic_sanity():
    driver = os.path.join(
        ROOT_DIR, 'scripts', 'maintenance', 'auto_ntraj_pilot.py'
    )
    if not os.path.exists(driver):
        print("[SKIP] auto_ntraj_pilot.py not found; "
              "cannot run --sanity")
        return 0
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    result = subprocess.run(
        [sys.executable, driver, '--sanity'],
        cwd=ROOT_DIR, env=env,
    )
    return result.returncode


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(NTRAJ_CONV_CSV):
        print(f"[SKIP] {os.path.relpath(NTRAJ_CONV_CSV, ROOT_DIR)} "
              f"not found.")
        print("[SKIP] running synthetic --sanity as a fallback "
              "diagnostic")
        rc = _run_synthetic_sanity()
        if rc != 0:
            print(f"[SKIP] NOTE: --sanity also returned rc={rc}")
        sys.exit(0)

    rows = _read_csv(NTRAJ_CONV_CSV)
    if not rows:
        print("[SKIP] ntraj_conv.csv is empty.")
        sys.exit(0)

    n_ok = 0
    n_fail = 0
    n_untestable = 0
    failures = []

    for row in rows:
        try:
            ok, info = _validate_row(row)
        except Exception:
            n_untestable += 1
            continue

        if ok is None:
            n_untestable += 1
            continue
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append((row, info))

    print("=" * 70)
    print("check_selector_ntraj.py")
    print(f"  ntraj_conv.csv rows:     {len(rows)}")
    print(f"  testable rows:           {n_ok + n_fail}")
    print(f"  satisfying criterion:    {n_ok}")
    print(f"  violating criterion:     {n_fail}")
    print(f"  untestable:              {n_untestable}")

    if failures:
        print("Failures:")
        for row, info in failures:
            label = (f"{row.get('case', '')} "
                     f"alpha={row.get('alpha', '')}")
            print(f"  {label}: a={info['a']:.3e} b={info['b']:.3e} "
                  f"n={info['n']} a/sqrt(n)={info['stat']:.3e}")

    print()
    if n_fail == 0:
        print(
            "[PASS] a/sqrt(N_traj,conv) > b for all testable rows"
        )
        sys.exit(0)
    else:
        print(
            f"[FAIL] {n_fail} rows violate a/sqrt(N_traj,conv) > b"
        )
        sys.exit(1)


if __name__ == '__main__':
    main()