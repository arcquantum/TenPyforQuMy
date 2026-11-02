#!/usr/bin/env python3
"""
scripts/maintenance/bundle_results.py


the documentation — aggregate per-point .npy outputs into per-tuple bundles
under data/bundles/.


For each case identifier (four symmetric, four asymmetric), the
script:

  1. Resolves the input directory. The plan's preferred name is
     ``data/echo_pts_<case>/``. The current repository uses
     ``data/powerlaw_echo_pts/``, ``data/powerlaw_cylinder_pts/``,
     ``data/tjm_pts/``, and the four ``data/echo_pts_*_asym/``
     directories. The script prefers the naming and falls
     back to the current naming.

  2. Globs every ``*.npy`` in that directory, skipping files whose
     names end in ``.tmp``, ``.bak``, or contain ``checkpoint``.

  3. Loads each file, filters out bundles whose geometry does not
     match the case (needed because 1D open and 2D open share the
     ``tjm_pts/`` directory), and groups by the parameter tuple
     ``(alpha, geometry, gamma1, gamma2)``.

  4. For each group, builds a 2D array of shape ``(n_mu, n_nu)`` or
     ``(n_t_plus, n_t_minus)`` for the asymmetric cases, by matching
     the stored mu/nu values to a sorted unique list with a
     tolerance of 1e-3.

  5. Writes a single bundle ``.npy`` per group to ``data/bundles/``.
     The filename follows the documentation's convention including the
     suffix-omission rule for noise parameters.

  6. Any group whose finite-point fraction falls below
     ``--min-fraction`` (default 0.90) is logged to
     ``data/bundles/missing.txt``.


Scharnagl bundling (the documentation)
------------------------------
The four ``scharnagl_*`` identifiers are accepted by the CLI but
the bundling itself is deferred to the documentation, which is the step
that creates the Scharnagl workers. Passing a Scharnagl case to
this script today prints a skip message and exits cleanly.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
import numpy as np


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BUNDLES_DIR = os.path.join(ROOT_DIR, 'data', 'bundles')
DATA_DIR = os.path.join(ROOT_DIR, 'data')


# ----------------------------------------------------------------------
# Case dispatch
# ----------------------------------------------------------------------
SYMMETRIC_CASES = ('1d_closed', '1d_open', '2d_closed', '2d_open')
ASYM_CASES = ('1d_closed_asym', '1d_open_asym',
              '2d_closed_asym', '2d_open_asym')
SCH_CASES = ('scharnagl_1d_closed', 'scharnagl_1d_open',
             'scharnagl_2d_closed', 'scharnagl_2d_open')


# Per-case source directories, in preference order. The plan's
# preferred name is tried first; the actual repository layout is
# the fallback.
CASE_SOURCES = {
    '1d_closed':      ('echo_pts_1d_closed', 'powerlaw_echo_pts'),
    '1d_open':        ('echo_pts_1d_open',   'tjm_pts'),
    '2d_closed':      ('echo_pts_2d_closed', 'powerlaw_cylinder_pts'),
    '2d_open':        ('echo_pts_2d_open',   'tjm_pts'),
    '1d_closed_asym': ('echo_pts_1d_closed_asym',),
    '1d_open_asym':   ('echo_pts_1d_open_asym',),
    '2d_closed_asym': ('echo_pts_2d_closed_asym',),
    '2d_open_asym':   ('echo_pts_2d_open_asym',),
}


def _is_2d(case: str) -> bool:
    return case.startswith('2d')


def _is_open(case: str) -> bool:
    return '_open' in case


def _is_asymmetric(case: str) -> bool:
    return case in ASYM_CASES


# ----------------------------------------------------------------------
# Input discovery
# ----------------------------------------------------------------------
def _resolve_input_dir(case: str):
    """Return the first existing source directory for the case."""
    for name in CASE_SOURCES.get(case, ()):
        p = os.path.join(DATA_DIR, name)
        if os.path.isdir(p):
            return p
    return None


def _discover_files(case: str):
    """Return the list of candidate per-point .npy paths."""
    input_dir = _resolve_input_dir(case)
    if input_dir is None:
        return []
    out = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.npy'):
            continue
        if fname.endswith('.tmp') or fname.endswith('.bak'):
            continue
        if 'checkpoint' in fname:
            continue
        out.append(os.path.join(input_dir, fname))
    return out


def _load_point(path: str):
    """Load a per-point .npy, returning the dict or None."""
    try:
        d = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        print(f"[warn] cannot load {os.path.basename(path)}: {exc}")
        return None
    return d if isinstance(d, dict) else None


def _matches_case(case: str, bundle: dict) -> bool:
    """Check whether the bundle's geometry matches the case."""
    has_N = bundle.get('N') is not None
    has_L = (bundle.get('Lx') is not None and bundle.get('Ly') is not None)
    if _is_2d(case):
        return has_L
    return has_N and not has_L

# ----------------------------------------------------------------------
# Resource-table lookup for chi selection
# ----------------------------------------------------------------------
def _read_chi_conv():
    """
    Read data/resource_tables/chi_conv.csv into a dict keyed by
    (case, geometry, alpha), where geometry is an int N for 1D or
    a tuple (Lx, Ly) for 2D. Returns an empty dict if the file is
    absent or unreadable.
    """
    import csv as _csv
    path = os.path.join(DATA_DIR, 'resource_tables', 'chi_conv.csv')
    if not os.path.exists(path):
        return {}
    out = {}
    try:
        with open(path, 'r', newline='') as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                case = (row.get('case') or '').strip()
                if not case:
                    continue
                try:
                    alpha = round(float(row.get('alpha', 1.0)), 6)
                except (TypeError, ValueError):
                    continue
                N_str = (row.get('N') or '').strip()
                Lx_str = (row.get('Lx') or '').strip()
                Ly_str = (row.get('Ly') or '').strip()
                if N_str:
                    geom = int(float(N_str))
                elif Lx_str and Ly_str:
                    geom = (int(float(Lx_str)), int(float(Ly_str)))
                else:
                    continue
                chi_str = (row.get('chi_prod') or '').strip()
                if not chi_str:
                    chi_str = (row.get('chi_conv') or '').strip()
                if not chi_str:
                    continue
                try:
                    chi = int(float(chi_str))
                except (TypeError, ValueError):
                    continue
                out[(case, geom, alpha)] = chi
    except Exception as exc:
        print(f"[warn] could not read chi_conv.csv: {exc}")
    return out


_CHI_CONV_CACHE = None


def _chi_conv_table():
    """Lazily load and cache the chi_conv table."""
    global _CHI_CONV_CACHE
    if _CHI_CONV_CACHE is None:
        _CHI_CONV_CACHE = _read_chi_conv()
    return _CHI_CONV_CACHE


def _select_chi_for_group(case, points, key, selection):
    """
    Filter a group's points to a single chi value.

    selection == 'production': use chi_prod from chi_conv.csv if the
                               value is present in the data; fall
                               back to the maximum available chi.
    selection == 'max':        always use the maximum available chi.

    Returns (filtered_points, chosen_chi, source_label). chosen_chi
    is None if the group has no chi information at all.
    """
    if not points:
        return points, None, 'empty'

    available = sorted({int(p['chi_max'])
                        for p in points
                        if p.get('chi_max') is not None})
    if not available:
        return points, None, 'no-chi-key'

    if _is_2d(case):
        alpha, Lx, Ly, _, _ = key
        lookup_key = (case, (Lx, Ly), alpha)
    else:
        alpha, N, _, _ = key
        lookup_key = (case, N, alpha)

    if selection == 'production':
        target = _chi_conv_table().get(lookup_key)
        if target is not None and target in available:
            chosen = target
            source = 'production'
        else:
            chosen = max(available)
            source = 'max-fallback'
    else:  # selection == 'max'
        chosen = max(available)
        source = 'max'

    filtered = [p for p in points
                if int(p.get('chi_max', -1)) == chosen]
    return filtered, chosen, source


# ----------------------------------------------------------------------
# Grouping and grid construction
# ----------------------------------------------------------------------
def _group_key(case: str, bundle: dict):
    """Parameter tuple used to group points into one bundle."""
    alpha = round(float(bundle.get('alpha', 1.0)), 6)
    g1 = round(float(bundle.get('gamma1', 0.0) or 0.0), 6)
    g2 = round(float(bundle.get('gamma2', 0.0) or 0.0), 6)
    if _is_2d(case):
        return (alpha, int(bundle['Lx']), int(bundle['Ly']), g1, g2)
    return (alpha, int(bundle['N']), g1, g2)


def _unique_with_tol(values, tol: float) -> np.ndarray:
    """
    Unique values in sorted order, merged to within ``tol``.
    Uses a linear pass over the sorted array; the input is assumed
    to be a 1-D sequence of floats.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return arr
    keep = [arr[0]]
    for v in arr[1:]:
        if abs(v - keep[-1]) > tol:
            keep.append(v)
    return np.asarray(keep)


def _build_grid(points, asymmetric: bool, tol: float = 1e-3):
    """
    Build a 2D SNR grid from a list of per-point dicts.

    Returns (x_vals, y_vals, grid). Missing entries are NaN.
    """
    if asymmetric:
        x_key, y_key = 't_plus', 't_minus'
    else:
        # Migrated workers include mu1/mu2 in every per-point dict.
        # Fall back to legacy mu/nu for pre-migration files.
        sample = points[0] if points else {}
        if 'mu1' in sample and 'mu2' in sample:
            x_key, y_key = 'mu1', 'mu2'
        else:
            x_key, y_key = 'mu', 'nu'

    xs = [float(p[x_key]) for p in points]
    ys = [float(p[y_key]) for p in points]
    x_vals = _unique_with_tol(xs, tol)
    y_vals = _unique_with_tol(ys, tol)

    grid = np.full((x_vals.size, y_vals.size), np.nan)
    for p in points:
        x = float(p[x_key])
        y = float(p[y_key])
        i = int(np.argmin(np.abs(x_vals - x)))
        j = int(np.argmin(np.abs(y_vals - y)))
        if np.isfinite(grid[i, j]):
            continue  # duplicate grid point; keep the first
        grid[i, j] = float(p['snr'])

    return x_vals, y_vals, grid


# ----------------------------------------------------------------------
# Bundle filename
# ----------------------------------------------------------------------
def _bundle_filename(case: str, key) -> str:
    """Compute the bundle filename for a group key."""
    if _is_2d(case):
        alpha, Lx, Ly, g1, g2 = key
        base = f'{case}_alpha{alpha}_Lx{Lx}_Ly{Ly}'
    else:
        alpha, N, g1, g2 = key
        base = f'{case}_alpha{alpha}_N{N}'

    if _is_open(case):
        if g1 > 0.0 and g2 > 0.0:
            base += f'_g1{g1}_g2{g2}'
        elif g1 > 0.0:
            base += f'_g1{g1}'
        elif g2 > 0.0:
            base += f'_g2{g2}'
        # else: both zero, omit both suffixes
    return base + '.npy'


# ----------------------------------------------------------------------
# Bundle writer
# ----------------------------------------------------------------------
def _write_bundle(case, key, points, asymmetric,
                  chi_used=None, chi_source=None):
    """
    Write a single bundle. Returns (path, n_finite, n_expected).
    """
    x_vals, y_vals, grid = _build_grid(points, asymmetric)
    n_finite = int(np.isfinite(grid).sum())
    n_expected = int(grid.size)

    ref = points[0]
    meta = {
        'case':  case,
        'alpha': float(ref.get('alpha', 1.0)),
        'J':     float(ref.get('J', 1.0)),
    }
    if _is_2d(case):
        meta['Lx'] = int(ref['Lx'])
        meta['Ly'] = int(ref['Ly'])
    else:
        meta['N'] = int(ref['N'])
    if _is_open(case):
        meta['gamma1'] = float(ref.get('gamma1', 0.0) or 0.0)
        meta['gamma2'] = float(ref.get('gamma2', 0.0) or 0.0)
        if 'n_traj' in ref:
            meta['n_traj'] = int(ref['n_traj'])
    if chi_used is not None:
        meta['chi_used'] = int(chi_used)
        meta['chi_source'] = str(chi_source)
    if 'chi_max' in ref:
        meta['chi_max'] = int(ref['chi_max'])
    if asymmetric and 'method' in ref:
        meta['method'] = str(ref['method'])

    bundle = dict(meta)
    if asymmetric:
        bundle['t_plus'] = x_vals
        bundle['t_minus'] = y_vals
    else:
        if 'mu1' in ref:
            bundle['mu1_vals'] = x_vals
            bundle['mu2_vals'] = y_vals
        else:
            bundle['mu_vals'] = x_vals
            bundle['nu_vals'] = y_vals
    bundle['snr'] = grid

    os.makedirs(BUNDLES_DIR, exist_ok=True)
    out_path = os.path.join(BUNDLES_DIR, _bundle_filename(case, key))
    np.save(out_path, bundle)
    return out_path, n_finite, n_expected


def _log_missing(entries):
    """Append entries to data/bundles/missing.txt."""
    os.makedirs(BUNDLES_DIR, exist_ok=True)
    path = os.path.join(BUNDLES_DIR, 'missing.txt')
    with open(path, 'a') as fh:
        for entry in entries:
            fh.write(entry + '\n')


# ----------------------------------------------------------------------
# Per-case driver
# ----------------------------------------------------------------------
def bundle_case(case: str, min_fraction: float = 0.90,
                chi_selection: str = 'production') -> None:
    if case in SCH_CASES:
        print(f"[skip] {case}: Scharnagl bundling is deferred to "
              f"the documentation.")
        return

    asymmetric = _is_asymmetric(case)
    input_dir = _resolve_input_dir(case)
    if input_dir is None:
        print(f"[skip] {case}: no input directory found "
              f"(searched {CASE_SOURCES.get(case)}).")
        return

    files = _discover_files(case)
    if not files:
        print(f"[skip] {case}: no per-point files found in "
              f"{os.path.basename(input_dir)}.")
        return

    print(f"[{case}] {len(files)} candidate files in "
          f"{os.path.basename(input_dir)}")

    groups = defaultdict(list)
    n_skipped_geom = 0
    n_skipped_load = 0
    n_skipped_snr = 0
    for path in files:
        d = _load_point(path)
        if d is None:
            n_skipped_load += 1
            continue
        if not _matches_case(case, d):
            n_skipped_geom += 1
            continue
        if 'snr' not in d:
            n_skipped_snr += 1
            continue
        groups[_group_key(case, d)].append(d)

    print(f"[{case}] {len(groups)} groups; skipped "
          f"{n_skipped_load} (load), "
          f"{n_skipped_geom} (geometry), "
          f"{n_skipped_snr} (no snr)")

    missing_entries = []
    n_written = 0
    for key in sorted(groups.keys()):
        points, chosen_chi, chi_source = _select_chi_for_group(
            case, groups[key], key, chi_selection
        )
        if not points:
            print(f"[{case}] group {key}: no points after chi "
                  f"selection, skipping.")
            continue

        try:
            out_path, n_finite, n_expected = _write_bundle(
                case, key, points, asymmetric,
                chi_used=chosen_chi, chi_source=chi_source,
            )
        except Exception as exc:
            print(f"[{case}] failed to bundle {key}: {exc}")
            continue
        n_written += 1
        frac = n_finite / max(n_expected, 1)
        flag = ''
        if frac < min_fraction:
            flag = f'  [LOW {frac:.1%}]'
            missing_entries.append(
                f'{os.path.basename(out_path)}: '
                f'{n_finite}/{n_expected} ({frac:.1%})'
            )
        print(f"[{case}] {os.path.basename(out_path)}: "
              f'{n_finite}/{n_expected} ({frac:.1%}), '
              f'chi={chosen_chi} ({chi_source}){flag}')

    if missing_entries:
        _log_missing(missing_entries)
        print(f"[{case}] logged {len(missing_entries)} "
              f"missing-input entries")
    print(f"[{case}] wrote {n_written} bundles")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle per-point .npy files into data/bundles/."
    )
    parser.add_argument(
        '--case',
        choices=list(SYMMETRIC_CASES + ASYM_CASES + SCH_CASES),
        default=None,
        help='Bundle only this case. If omitted, bundle all '
             'symmetric and asymmetric cases.',
    )
    parser.add_argument(
        '--min-fraction', type=float, default=0.90,
        help='Fraction of grid points required before logging a '
             'missing-input warning (default: 0.90).',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Bundle all symmetric and asymmetric cases '
             '(default behaviour when --case is omitted).',
    )
    parser.add_argument(
        '--chi-selection',
        choices=['production', 'max'],
        default='production',
        help='How to select the chi value when multiple chi files '
             'exist for the same tuple. "production" reads chi_prod '
             'from chi_conv.csv with fallback to max. "max" always '
             'uses the maximum available chi.',
    )

    args = parser.parse_args()

    if args.case is not None:
        cases = [args.case]
    else:
        cases = list(SYMMETRIC_CASES + ASYM_CASES)

    print("=" * 72)
    print("the documentation — bundle utility")
    print("=" * 72)
    print(f"Output directory: {BUNDLES_DIR}")

    for case in cases:
        try:
            bundle_case(
                case,
                min_fraction=args.min_fraction,
                chi_selection=args.chi_selection,
            )
        except Exception as exc:
            print(f"[error] {case}: {exc}")


if __name__ == '__main__':
    main()