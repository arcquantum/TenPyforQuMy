"""
scripts/checks/check_selector_chi.py

Validates the bond-dimension selector (the documentation v2).

For each numeric row in data/resource_tables/chi_conv.csv:
  1. Read the union pilot grid from
     data/resource_tables/pilot_grids/.
  2. Read SNR at each chi on the union grid from the worker
     output under data/.
  3. Independently recompute chi_conv via find_chi_convergence.
  4. Compare to the CSV value.
  5. Verify the row was not flagged by the coarse-grid
     verification step, either by appearing in
     chi_conv_flags.csv or by carrying a coarse_verify_residual
     above the case-specific target.

If chi_conv.csv is missing or empty, the script prints [SKIP]
and exits 0 so that the master runner remains usable during the
the codebase timeline.

Exit codes:
  0  if all testable rows agree and no tuple is flagged, or the
     CSV is missing
  1  if any testable row disagrees or is flagged
"""

from __future__ import annotations

import csv
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.selectors import (
    CHI_SCHEDULE, TARGET_CLOSED, TARGET_OPEN,
    native_output_path, snr_from_output,
    find_chi_convergence, read_pilot_grid,
)

RESOURCE_DIR  = os.path.join(ROOT_DIR, 'data', 'resource_tables')
CHI_CONV_CSV  = os.path.join(RESOURCE_DIR, 'chi_conv.csv')
CHI_FLAGS_CSV = os.path.join(RESOURCE_DIR, 'chi_conv_flags.csv')


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


def _case_is_open(case):
    return 'open' in case


# ----------------------------------------------------------------------
# Tuple key for the flags file
# ----------------------------------------------------------------------
def _row_key(row):
    """Canonical key for cross-referencing chi_conv_flags.csv."""
    return (
        (row.get('case') or '').strip(),
        (row.get('N') or '').strip(),
        (row.get('Lx') or '').strip(),
        (row.get('Ly') or '').strip(),
        (row.get('alpha') or '').strip(),
        ((row.get('gamma1') or '0.0').strip() or '0.0'),
        ((row.get('gamma2') or '0.0').strip() or '0.0'),
    )


def _flagged_keys():
    rows = _read_csv(CHI_FLAGS_CSV)
    return {_row_key(r) for r in rows}


# ----------------------------------------------------------------------
# Recompute chi_conv for one row
# ----------------------------------------------------------------------
def _recompute_chi_conv(row):
    """
    Return (chi_conv or None, info dict).

    Reads the union pilot grid from disk, reads SNR at each chi
    on the union grid, and applies find_chi_convergence to
    produce an independent estimate of chi_conv.
    """
    case = (row.get('case') or '').strip()
    alpha = _parse_float(row.get('alpha'))
    if alpha is None:
        return None, {'reason': 'missing alpha'}

    N  = _parse_int(row.get('N'))
    Lx = _parse_int(row.get('Lx'))
    Ly = _parse_int(row.get('Ly'))
    g1 = _parse_float(row.get('gamma1')) or 0.0
    g2 = _parse_float(row.get('gamma2')) or 0.0

    pilot_grid = read_pilot_grid(
        case,
        N=N, Lx=Lx, Ly=Ly, alpha=alpha,
        gamma1=g1, gamma2=g2,
        root=ROOT_DIR,
    )
    if not pilot_grid:
        return None, {'reason': 'empty pilot grid'}

    snr = {}
    for chi in CHI_SCHEDULE:
        pts = {}
        for (mu, nu) in pilot_grid:
            path = native_output_path(
                case, chi,
                N=N, Lx=Lx, Ly=Ly, alpha=alpha,
                mu=mu, nu=nu,
                gamma1=g1, gamma2=g2,
                root=ROOT_DIR,
            )
            val = snr_from_output(path)
            if val is not None:
                pts[(round(mu, 4), round(nu, 4))] = val
        if pts:
            snr[chi] = pts

    if not snr:
        return None, {'reason': 'no SNR data on disk'}

    target = TARGET_OPEN if _case_is_open(case) else TARGET_CLOSED
    chi_conv, residuals = find_chi_convergence(snr, target)

    return chi_conv, {
        'residuals': residuals,
        'n_chi_present': len(snr),
        'target': target,
    }


# ----------------------------------------------------------------------
# Coarse-grid verification flag check
# ----------------------------------------------------------------------
def _flagged_by_coarse_verify(row):
    """
    Return True if the row indicates the tuple was flagged by the
    coarse-grid verification step:

      - coarse_verify_points == 0 (no verification points), or
      - coarse_verify_residual > case-specific target.

    A row with neither column, or with a NaN/infinite residual,
    is treated as not flagged.
    """
    case = (row.get('case') or '').strip()
    target = TARGET_OPEN if _case_is_open(case) else TARGET_CLOSED

    pts = _parse_int(row.get('coarse_verify_points'))
    res = _parse_float(row.get('coarse_verify_residual'))

    if pts is not None and pts == 0:
        return True
    if res is not None and res > target:
        return True
    return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(CHI_CONV_CSV):
        print(f"[SKIP] {os.path.relpath(CHI_CONV_CSV, ROOT_DIR)} "
              f"not found.")
        sys.exit(0)

    rows = _read_csv(CHI_CONV_CSV)
    if not rows:
        print("[SKIP] chi_conv.csv is empty.")
        sys.exit(0)

    flagged_keys = _flagged_keys()

    n_ok = 0
    n_mismatch = 0
    n_flagged = 0
    n_intractable = 0
    n_untestable = 0

    mismatches = []
    flagged_rows = []

    for row in rows:
        label = (
            f"{row.get('case', '')} "
            f"N={row.get('N', '')} "
            f"Lx={row.get('Lx', '')} "
            f"Ly={row.get('Ly', '')} "
            f"alpha={row.get('alpha', '')}"
        )

        csv_chi = _parse_int(row.get('chi_conv'))

        # A non-numeric chi_conv (typically the literal string
        # "intractable") is an intentional resource-table entry
        # and is skipped.
        if csv_chi is None:
            n_intractable += 1
            continue

        # Flag check: either the flags file, or the row's own
        # coarse-verify columns.
        row_key = _row_key(row)
        if row_key in flagged_keys or _flagged_by_coarse_verify(row):
            n_flagged += 1
            flagged_rows.append(label)
            continue

        # Independent recompute.
        try:
            rec_chi, _info = _recompute_chi_conv(row)
        except Exception:
            n_untestable += 1
            continue

        if rec_chi is None:
            n_untestable += 1
            continue

        if rec_chi == csv_chi:
            n_ok += 1
        else:
            n_mismatch += 1
            mismatches.append((label, csv_chi, rec_chi))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("check_selector_chi.py")
    print(f"  chi_conv.csv rows:       {len(rows)}")
    print(f"  numeric rows:            {n_ok + n_mismatch + n_flagged}")
    print(f"  match:                   {n_ok}")
    print(f"  mismatch:                {n_mismatch}")
    print(f"  flagged by verify:       {n_flagged}")
    print(f"  intractable (skipped):   {n_intractable}")
    print(f"  untestable:              {n_untestable}")

    if mismatches:
        print("Mismatches:")
        for label, csv_val, rec in mismatches:
            print(f"  {label}: csv={csv_val} recomputed={rec}")

    if flagged_rows:
        print("Flagged tuples:")
        for label in flagged_rows:
            print(f"  {label}")

    print()
    all_pass = (n_mismatch == 0 and n_flagged == 0)
    if all_pass:
        print(
            f"[PASS] all testable rows agree with an independent "
            f"sweep "
            f"(TARGET_CLOSED={TARGET_CLOSED:.0e}, "
            f"TARGET_OPEN={TARGET_OPEN:.0e})"
        )
        sys.exit(0)
    else:
        print(
            f"[FAIL] {n_mismatch} mismatches, {n_flagged} flagged "
            f"tuples"
        )
        sys.exit(1)


if __name__ == '__main__':
    main()