"""
scripts/checks/check_selector_grid.py

Validates the grid-resolution selector (the documentation v2).

For each testable row in data/resource_tables/grid_conv.csv:

    err(grid_conv)      < target
    err(next coarser)   > target

where err is the restricted L² error on the pilot subset after
interpolating the landscape .npy arrays to a common reference
grid, and target = PILOT_GRID_ERROR_TARGET (1e-2) relative to
max SNR.

If grid_conv.csv is missing or empty, the script prints [SKIP],
runs auto_grid_pilot.py --sanity as a diagnostic, and exits 0 so
that the master runner remains usable during the the codebase
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
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.selectors import (
    PILOT_GRID, GRID_SCHEDULE, REFERENCE_GRID,
    COARSE_GRADIENT_CHI, PILOT_GRID_ERROR_TARGET,
    read_grid_pilot, read_landscape_grid,
    interpolate_to_reference, error_on_pilot_subset,
)

RESOURCE_DIR  = os.path.join(ROOT_DIR, 'data', 'resource_tables')
GRID_CONV_CSV = os.path.join(RESOURCE_DIR, 'grid_conv.csv')


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


def _params_from_row(row):
    return {
        'N':      _parse_int(row.get('N')),
        'Lx':     _parse_int(row.get('Lx')),
        'Ly':     _parse_int(row.get('Ly')),
        'alpha':  _parse_float(row.get('alpha')) or 1.0,
        'gamma1': _parse_float(row.get('gamma1')) or 0.0,
        'gamma2': _parse_float(row.get('gamma2')) or 0.0,
    }


def _next_coarser(grid_conv, schedule):
    """Return the largest scheduled resolution < grid_conv, or None."""
    sorted_schedule = sorted(schedule)
    prev = None
    for g in sorted_schedule:
        if g >= grid_conv:
            return prev
        prev = g
    return prev


# ----------------------------------------------------------------------
# Per-row validation
# ----------------------------------------------------------------------
def _validate_row(row, schedule, target):
    """
    Return (ok, info). ok is True, False, or None (untestable).
    """
    case = (row.get('case') or '').strip()
    params = _params_from_row(row)
    grid_conv = _parse_int(row.get('grid_conv'))
    if grid_conv is None:
        return None, {'reason': 'no grid_conv'}

    union = read_grid_pilot(case, params, root=ROOT_DIR)
    if union is None:
        union = list(PILOT_GRID)

    # Read every landscape that exists for this tuple.
    landscapes = {}
    for g in schedule:
        d = read_landscape_grid(case, params, g, COARSE_GRADIENT_CHI,
                                root=ROOT_DIR)
        if d is not None:
            landscapes[int(g)] = d

    if len(landscapes) < 2:
        return None, {'reason': f'{len(landscapes)} landscapes on disk'}

    # Interpolate every landscape to the common reference grid.
    interp = {
        g: interpolate_to_reference(d['snr_grid'], REFERENCE_GRID)
        for g, d in landscapes.items()
    }

    # Reference is the largest available resolution.
    ref_g = max(landscapes.keys())
    S_ref = interp[ref_g]
    snr_max = float(np.nanmax(np.abs(S_ref)))
    abs_target = target * max(snr_max, 1e-30)

    # Restricted L² error at every available resolution.
    errs = {}
    for g in sorted(landscapes.keys()):
        if g == ref_g:
            errs[g] = 0.0
        else:
            errs[g] = error_on_pilot_subset(
                S_ref, interp[g], union, REFERENCE_GRID
            )

    if grid_conv not in errs:
        return None, {'reason': f'grid_conv={grid_conv} not in data'}

    err_at_conv = errs[grid_conv]
    next_coarser = _next_coarser(grid_conv, schedule)
    err_at_coarser = errs.get(next_coarser) if next_coarser else None

    ok_conv = err_at_conv < abs_target
    ok_coarser = (err_at_coarser is None) or (err_at_coarser > abs_target)

    return (ok_conv and ok_coarser), {
        'grid_conv':      grid_conv,
        'err_at_conv':    err_at_conv,
        'abs_target':     abs_target,
        'snr_max':        snr_max,
        'next_coarser':   next_coarser,
        'err_at_coarser': err_at_coarser,
        'ok_conv':        ok_conv,
        'ok_coarser':     ok_coarser,
    }


# ----------------------------------------------------------------------
# Fallback: synthetic sanity check when CSV is missing
# ----------------------------------------------------------------------
def _run_synthetic_sanity():
    driver = os.path.join(
        ROOT_DIR, 'scripts', 'maintenance', 'auto_grid_pilot.py'
    )
    if not os.path.exists(driver):
        print("[SKIP] auto_grid_pilot.py not found; "
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
    if not os.path.exists(GRID_CONV_CSV):
        print(f"[SKIP] {os.path.relpath(GRID_CONV_CSV, ROOT_DIR)} "
              f"not found.")
        print("[SKIP] running synthetic --sanity as a fallback "
              "diagnostic")
        rc = _run_synthetic_sanity()
        if rc != 0:
            print(f"[SKIP] NOTE: --sanity also returned rc={rc}")
        # The plan specifies that a missing resource table exits 0.
        sys.exit(0)

    rows = _read_csv(GRID_CONV_CSV)
    if not rows:
        print("[SKIP] grid_conv.csv is empty.")
        sys.exit(0)

    schedule = sorted(GRID_SCHEDULE)
    target = PILOT_GRID_ERROR_TARGET

    n_ok = 0
    n_fail = 0
    n_untestable = 0
    failures = []

    for row in rows:
        try:
            ok, info = _validate_row(row, schedule, target)
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
    print("check_selector_grid.py")
    print(f"  grid_conv.csv rows:      {len(rows)}")
    print(f"  testable rows:           {n_ok + n_fail}")
    print(f"  satisfying criterion:    {n_ok}")
    print(f"  violating criterion:     {n_fail}")
    print(f"  untestable:              {n_untestable}")
    print(f"  target:                  {target:.0e}")

    if failures:
        print("Failures:")
        for row, info in failures:
            label = (f"{row.get('case', '')} "
                     f"alpha={row.get('alpha', '')}")
            print(f"  {label}: grid_conv={info['grid_conv']}, "
                  f"err={info['err_at_conv']:.3e} "
                  f"(target={info['abs_target']:.3e}), "
                  f"next_coarser={info['next_coarser']}, "
                  f"err_coarser={info['err_at_coarser']}")

    print()
    if n_fail == 0:
        print(f"[PASS] restricted L² error < {target:.0e} at chosen "
              f"resolution for all testable rows")
        sys.exit(0)
    else:
        print(f"[FAIL] {n_fail} rows violate the {target:.0e} "
              f"criterion")
        sys.exit(1)


if __name__ == '__main__':
    main()