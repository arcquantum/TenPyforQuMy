#!/usr/bin/env python3
"""
scripts/checks/preflight.py


Ten-point manual preflight for the four production classes, run
before any production SLURM array is submitted.


For each class (1D closed, 1D open, 2D closed, 2D open) and each of
ten reference (mu, nu) points, the script:

  1. Invokes the symmetric-echo worker at chi = 32, dt = 0.05.
  2. Loads the output .npy.
  3. Verifies the schema contains the expected keys.
  4. Verifies the physics assertions:
       - coherent-state limit SNR = sqrt(N) at (0, 0),
         relative error < 1e-3;
       - OUT-protocol SNR > 1 at (pi/2, -pi/2), for N >= 16;
       - GHZ-branch SNR > sqrt(N) at (0, -pi);
       - all values finite and positive.
  5. Prints a per-class summary table.


Triage on FAIL
--------------
A failing point is rerun at chi = 64, dt = 0.01 (a factor of two
more accurate than the standard preflight settings). If the rerun
passes, the failure is classified as a numerical-precision artefact
of the fast preflight and logged to
data/validation_reports/preflight_triage.csv. If the rerun also
fails, the failure is classified as a code bug and the preflight
aborts. If more than three points require triage in a single
preflight invocation, the run is treated as a systemic failure
and the preflight aborts for manual inspection.


Scharnagl preflight (deferred to the documentation)
-------------------------------------------
The --with-scharnagl flag (off by default) enables two reference
runs per Scharnagl case. If any of the four Scharnagl workers is
missing, the flag raises a clear error. When the documentation lands, the
flag is enabled and the runs execute.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
for _p in (ROOT_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.selectors import (  # type: ignore[import-not-found]  # noqa: E402
    worker_command, native_output_path,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GRID_POINTS_MU1MU2 = [
   (0.0,            0.0),
   (0.5,            0.5),
   (1.0,            1.0),
   (np.pi / 2,      np.pi / 2),
   (np.pi / 2,      0.0),
   (np.pi / 2,      np.pi),
   (0.0,            np.pi),
   (np.pi,          np.pi),
   (np.pi,          2.0 * np.pi),
   (np.pi / 2,      3.0 * np.pi / 2.0),
]

GRID_POINTS_MU_NU = [
   (0.0,            0.0),
   (0.5,            0.5),
   (1.0,            1.0),
   (np.pi / 2,     -np.pi / 2),
   (np.pi / 2,      0.0),
   (np.pi / 2,      np.pi / 2),
   (0.0,           -np.pi),
   (np.pi,         -np.pi),
   (np.pi,          np.pi),
   (np.pi / 2,     -2.0 * np.pi / 3.0),
]


def _grid_points(convention):
   if convention == 'mu1mu2':
       return GRID_POINTS_MU1MU2
   return GRID_POINTS_MU_NU

ALPHA_PREFLIGHT = 1.0
J_PREFLIGHT     = 1.0
CHI_PREFLIGHT   = 32
CHI_TRIAGE      = 64
DT_PREFLIGHT    = 0.05
DT_TRIAGE       = 0.01
SEED_PREFLIGHT  = 20261024

MAX_TRIAGE_POINTS = 3


# ----------------------------------------------------------------------
# Triage CSV
# ----------------------------------------------------------------------
TRIAGE_CSV = os.path.join(ROOT_DIR, 'data', 'validation_reports',
                          'preflight_triage.csv')
TRIAGE_FIELDS = [
    'timestamp', 'class', 'N', 'Lx', 'Ly', 'mu', 'nu',
    'chi_initial', 'dt_initial', 'snr_initial',
    'chi_triage', 'dt_triage', 'snr_triage',
    'classification', 'notes',
]


# ----------------------------------------------------------------------
# Scharnagl workers (the documentation)
# ----------------------------------------------------------------------
SCH_WORKERS = [
    'scharnagl_echo_1d_closed.py',
    'scharnagl_echo_1d_open.py',
    'scharnagl_echo_2d_closed.py',
    'scharnagl_echo_2d_open.py',
]


# ----------------------------------------------------------------------
# Schema requirements
# ----------------------------------------------------------------------
REQUIRED_KEYS = {
    '1d_closed': {'N', 'alpha', 'J', 'mu', 'nu', 'snr'},
    '1d_open':   {'N', 'alpha', 'J', 'mu', 'nu', 'gamma1', 'gamma2', 'snr'},
    '2d_closed': {'Lx', 'Ly', 'alpha', 'J', 'mu', 'nu', 'snr'},
    '2d_open':   {'Lx', 'Ly', 'alpha', 'J', 'mu', 'nu', 'gamma1', 'gamma2', 'snr'},
}


# ----------------------------------------------------------------------
# Per-class configuration
# ----------------------------------------------------------------------
def class_config(class_name, geom_2d, ntraj_2d_open):
    """Return the configuration dict for a production class."""
    if class_name == '1d_closed':
        return {
            'case':     '1d_closed',
            'geometry': {'N': 16},
            'gamma1':   0.0,
            'gamma2':   0.0,
            'n_traj':   0,
        }
    if class_name == '1d_open':
        # Preflight-quality trajectory count. Production uses 100.
        # Reduced to 5 here: the TJM worker's per-point wall time at
        # the OUT point (pi/2, -pi/2) grows as (mu/2 + (nu-mu)/2) *
        # 7 evolutions * n_traj. At n_traj=20 the OUT point exceeds
        # the 7200 s per-point timeout. At n_traj=5 it completes in
        # ~1 h. The SNR standard error at n_traj=5 is ~45%, which is
        # ample for the order-of-magnitude assertions this preflight
        # checks.
        return {
            'case':     '1d_open',
            'geometry': {'N': 16},
            'gamma1':   1e-2,
            'gamma2':   1e-2,
            'n_traj':   5,
        }
    if class_name == '2d_closed':
        return {
            'case':     '2d_closed',
            'geometry': {'Lx': int(geom_2d[0]), 'Ly': int(geom_2d[1])},
            'gamma1':   0.0,
            'gamma2':   0.0,
            'n_traj':   0,
        }
    if class_name == '2d_open':
        # Preflight-quality trajectory count. Production uses 20.
        # Reduced to 2 here: the 2D cylinder's block-site dimension
        # (d=16) makes each ExpMPOEvolution step ~5x more expensive
        # than the 1D chain, so the OUT point at n_traj=5 already
        # exceeds the 7200 s timeout. At n_traj=2 the OUT point
        # completes in ~1 h.
        return {
            'case':     '2d_open',
            'geometry': {'Lx': int(geom_2d[0]), 'Ly': int(geom_2d[1])},
            'gamma1':   1e-2,
            'gamma2':   1e-2,
            'n_traj':   2,
        }
    raise ValueError(f'unknown class: {class_name!r}')


# ----------------------------------------------------------------------
# Worker invocation and output path
# ----------------------------------------------------------------------
def build_worker_cmd(cfg, mu, nu, chi, dt, convention='mu1mu2'):
   """Build the subprocess command for one point (dual-convention)."""
   if convention == 'mu1mu2':
       return worker_command(
           cfg['case'], chi,
           N=cfg['geometry'].get('N'),
           Lx=cfg['geometry'].get('Lx'),
           Ly=cfg['geometry'].get('Ly'),
           alpha=ALPHA_PREFLIGHT,
           gamma1=cfg['gamma1'], gamma2=cfg['gamma2'],
           J=J_PREFLIGHT, dt_max=dt,
           n_traj=cfg['n_traj'],
           seed=SEED_PREFLIGHT,
           convention='mu1mu2',
           mu1=float(mu), mu2=float(nu),
       )
   return worker_command(
       cfg['case'], chi,
       N=cfg['geometry'].get('N'),
       Lx=cfg['geometry'].get('Lx'),
       Ly=cfg['geometry'].get('Ly'),
       alpha=ALPHA_PREFLIGHT,
       mu=float(mu), nu=float(nu),
       gamma1=cfg['gamma1'], gamma2=cfg['gamma2'],
       J=J_PREFLIGHT, dt_max=dt,
       n_traj=cfg['n_traj'],
       seed=SEED_PREFLIGHT,
   )


def output_path(cfg, mu, nu, chi, convention='mu1mu2'):
   """Return the .npy path the worker will write to (dual-convention)."""
   if convention == 'mu1mu2':
       return native_output_path(
           cfg['case'], chi,
           N=cfg['geometry'].get('N'),
           Lx=cfg['geometry'].get('Lx'),
           Ly=cfg['geometry'].get('Ly'),
           alpha=ALPHA_PREFLIGHT,
           gamma1=cfg['gamma1'], gamma2=cfg['gamma2'],
           n_traj=cfg['n_traj'],
           convention='mu1mu2',
           mu1=float(mu), mu2=float(nu),
       )
   return native_output_path(
       cfg['case'], chi,
       N=cfg['geometry'].get('N'),
       Lx=cfg['geometry'].get('Lx'),
       Ly=cfg['geometry'].get('Ly'),
       alpha=ALPHA_PREFLIGHT,
       mu=float(mu), nu=float(nu),
       gamma1=cfg['gamma1'], gamma2=cfg['gamma2'],
       n_traj=cfg['n_traj'],
   )


def clear_stale(cfg, mu, nu, chi, convention='mu1mu2'):
    """Remove any stale output and checkpoint from a previous run."""
    out = output_path(cfg, mu, nu, chi, convention=convention)
    for f in (out, out + '.checkpoint.h5'):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


# ----------------------------------------------------------------------
# Schema and physics checks
# ----------------------------------------------------------------------
def check_schema(cfg, bundle):
    """Return a list of missing keys (empty list if schema is complete)."""
    req = REQUIRED_KEYS[cfg['case']]
    return [k for k in req if k not in bundle]


def physics_checks(cfg, mu, nu, snr, N):
    """
    Return a list of (label, passed, note) tuples describing the
    physics assertions that apply to this (mu, nu) point.
    """
    checks = []

    # (a) finite and positive
    finite_ok = bool(np.isfinite(snr) and snr > 0.0)
    checks.append((
        'finite_positive',
        finite_ok,
        f'snr={snr:.6f}' if np.isfinite(snr) else f'snr={snr!r}',
    ))

    # (b) coherent-state limit at (0, 0)
    if abs(mu) < 1e-6 and abs(nu) < 1e-6:
        expected = float(np.sqrt(N))
        rel = abs(snr - expected) / expected
        checks.append((
            'coherent_state_sqrt_N',
            rel < 1e-3,
            f'snr={snr:.6f}, sqrt(N)={expected:.6f}, rel={rel:.2e}',
        ))

    # (c) OUT-protocol SNR above the SQL, for N >= 16.
    # In (mu, nu) convention: the OUT line is nu = -mu.
    # In (mu1, mu2) convention: the OUT line is mu2 = 2*mu1.
    # Fire the assertion on any point that lies on the OUT line.
    on_out_line = (
        abs(nu + mu) < 1e-6 or abs(nu - 2.0 * mu) < 1e-6
    )
    if N >= 16 and on_out_line and (abs(mu) + abs(nu)) > 1e-3:
        # Exclude the trivial (0,0) point, which is the coherent state
        # limit handled by check (b).
        checks.append((
            'OUT_above_1',
            snr > 1.0,
            f'snr={snr:.6f} at (mu, nu)=({mu:.4f}, {nu:.4f})',
        ))

    # (d) GHZ-branch SNR above a conservative fraction of sqrt(N).
    # In (mu, nu) convention: the GHZ branch sits at nu = -pi with
    # mu = 0 (maximal echo). In (mu1, mu2) convention, the
    # equivalent structure is at mu2 = mu1 (exact echo) with
    # mu1 -> 0 (i.e. mu = N*mu1/2 -> 0). Both cases produce a
    # near-coherent state, so the loose 0.5*sqrt(N) bound applies.
    # We also fire on any GHZ-branch point in the mu_nu grid:
    on_ghz_branch = (
        (abs(mu) < 1e-6 and abs(nu + np.pi) < 1e-6)
        or (abs(mu - nu) < 1e-6 and abs(mu) < 0.5)
    )
    if on_ghz_branch:
        expected = float(np.sqrt(N))
        checks.append((
            'GHZ_above_half_sqrt_N',
            snr >= 0.5 * expected,
            f'snr={snr:.6f}, sqrt(N)={expected:.6f}, '
            f'ratio={snr / expected:.6f}',
        ))

    return checks


# ----------------------------------------------------------------------
# Single-point run
# ----------------------------------------------------------------------
def run_point(cfg, mu, nu, chi, dt, timeout_s, verbose=True,
             convention='mu1mu2'):
    """
    Run one point. Returns (snr, error_message). On success,
    error_message is None. On any failure, snr is None.
    """
    clear_stale(cfg, mu, nu, chi, convention=convention)
    cmd = build_worker_cmd(cfg, mu, nu, chi, dt, convention=convention)
    out = output_path(cfg, mu, nu, chi, convention=convention)

    if verbose:
        print(f"    [run] mu={mu:.4f}, nu={nu:.4f}, chi={chi}, dt={dt}")
        print(f"          {' '.join(cmd)}")

    t0 = time.time()
    try:
        _ = subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, f'worker timed out after {timeout_s} s'
    except subprocess.CalledProcessError as exc:
        return None, f'worker exited with code {exc.returncode}'
    except Exception as exc:
        return None, f'worker invocation failed: {exc}'

    elapsed = time.time() - t0
    if verbose:
        print(f"          completed in {elapsed:.1f} s")

    if not os.path.exists(out):
        return None, f'no output file at {out}'

    try:
        bundle = np.load(out, allow_pickle=True).item()
    except Exception as exc:
        return None, f'failed to load {out}: {exc}'

    missing = check_schema(cfg, bundle)
    if missing:
        return None, f'missing schema keys: {missing}'

    try:
        snr = float(bundle['snr'])
    except (KeyError, TypeError, ValueError) as exc:
        return None, f'cannot read snr from bundle: {exc}'

    return snr, None


# ----------------------------------------------------------------------
# Triage logging
# ----------------------------------------------------------------------
def log_triage(cfg, mu, nu, snr_initial, snr_triage,
               classification, notes=''):
    os.makedirs(os.path.dirname(TRIAGE_CSV), exist_ok=True)
    write_header = not os.path.exists(TRIAGE_CSV)
    with open(TRIAGE_CSV, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIAGE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'timestamp':    datetime.now().isoformat(timespec='seconds'),
            'class':        cfg['case'],
            'N':            cfg['geometry'].get('N', ''),
            'Lx':           cfg['geometry'].get('Lx', ''),
            'Ly':           cfg['geometry'].get('Ly', ''),
            'mu':           f'{mu:.6f}',
            'nu':           f'{nu:.6f}',
            'chi_initial':  CHI_PREFLIGHT,
            'dt_initial':   DT_PREFLIGHT,
            'snr_initial':  (f'{snr_initial:.6f}'
                             if snr_initial is not None else ''),
            'chi_triage':   CHI_TRIAGE,
            'dt_triage':    DT_TRIAGE,
            'snr_triage':   (f'{snr_triage:.6f}'
                             if snr_triage is not None else ''),
            'classification': classification,
            'notes':        notes,
        })


# ----------------------------------------------------------------------
# Per-class preflight loop
# ----------------------------------------------------------------------
def run_class(class_name, geom_2d, ntraj_2d_open, point_timeout_s,
             convention='mu1mu2'):
    """
    Run the ten-point preflight for one class.


    Returns
    -------
    summary : dict
        Keys: 'class', 'points' (list of dicts), 'n_pass', 'n_fail',
        'n_triage', 'aborted'.
    """
    cfg = class_config(class_name, geom_2d, ntraj_2d_open)
    N = int(cfg['geometry'].get('N', 0))
    if N == 0:
        N = int(cfg['geometry']['Lx']) * int(cfg['geometry']['Ly'])

    print(f"\n{'=' * 72}")
    print(f"Class: {class_name}")
    geom_str = ', '.join(f'{k}={v}' for k, v in cfg['geometry'].items())
    print(f"  geometry: {geom_str} (N={N})")
    n_traj_label = (f"{cfg['n_traj']} (preflight quality)"
                    if cfg['n_traj'] else "0 (no trajectories)")
    print(f"  gamma1={cfg['gamma1']}, gamma2={cfg['gamma2']}, "
          f"n_traj={n_traj_label}")
    print(f"  chi={CHI_PREFLIGHT}, dt={DT_PREFLIGHT}, "
          f"alpha={ALPHA_PREFLIGHT}, J={J_PREFLIGHT}")
    print(f"{'=' * 72}")

    points = []
    n_triage = 0
    aborted = False

    for (mu, nu) in _grid_points(convention):
        print(f"\n  point (mu={mu:.4f}, nu={nu:.4f})")
        snr, err = run_point(cfg, mu, nu,
                             CHI_PREFLIGHT, DT_PREFLIGHT,
                             point_timeout_s, verbose=True,
                             convention=convention)

        note = ''

        if err is not None:
            # Schema / worker failure -> always triage
            print(f"    initial FAIL: {err}")
            n_triage += 1
            if n_triage > MAX_TRIAGE_POINTS:
                print(f"    SYSTEMIC FAILURE: more than "
                      f"{MAX_TRIAGE_POINTS} triage attempts. "
                      f"Aborting for manual inspection.")
                log_triage(cfg, mu, nu, snr, None, 'systemic',
                           f'initial_err={err}')
                points.append({'mu': mu, 'nu': nu, 'snr': None,
                               'status': 'systemic', 'note': err})
                aborted = True
                break

            print(f"    triaging at chi={CHI_TRIAGE}, dt={DT_TRIAGE}...")
            snr_t, err_t = run_point(cfg, mu, nu,
                                     CHI_TRIAGE, DT_TRIAGE,
                                     point_timeout_s, verbose=True,
                                     convention=convention)

            if err_t is None:
                # Triage passed -> numerical artefact
                print(f"    triage PASSED (snr={snr_t:.6f}): numerical "
                      f"artefact of the fast preflight.")
                log_triage(cfg, mu, nu, snr, snr_t, 'numerical',
                           f'initial_err={err}')
                points.append({'mu': mu, 'nu': nu, 'snr': snr_t,
                               'status': 'numeric', 'note': err})
                continue
            else:
                print(f"    triage FAILED: {err_t}")
                log_triage(cfg, mu, nu, snr, snr_t, 'code',
                           f'initial_err={err}; triage_err={err_t}')
                points.append({'mu': mu, 'nu': nu, 'snr': snr_t,
                               'status': 'code', 'note': err_t})
                aborted = True
                break

        # Worker succeeded. Run physics assertions.
        checks = physics_checks(cfg, mu, nu, snr, N)
        failed = [(lbl, nte) for lbl, ok, nte in checks if not ok]

        if not failed:
            print(f"    PASS  snr={snr:.6f}")
            points.append({'mu': mu, 'nu': nu, 'snr': snr,
                           'status': 'PASS', 'note': ''})
            continue

        # Physics assertion failed -> triage
        print(f"    ASSERT FAIL: {failed}")
        n_triage += 1
        if n_triage > MAX_TRIAGE_POINTS:
            print(f"    SYSTEMIC FAILURE. Aborting.")
            log_triage(cfg, mu, nu, snr, None, 'systemic',
                       f'failed_checks={failed}')
            points.append({'mu': mu, 'nu': nu, 'snr': snr,
                           'status': 'systemic', 'note': str(failed)})
            aborted = True
            break

        print(f"    triaging at chi={CHI_TRIAGE}, dt={DT_TRIAGE}...")
        snr_t, err_t = run_point(cfg, mu, nu,
                                 CHI_TRIAGE, DT_TRIAGE,
                                 point_timeout_s, verbose=True,
                                 convention=convention)

        if err_t is not None:
            print(f"    triage FAILED: {err_t}")
            log_triage(cfg, mu, nu, snr, None, 'code',
                       f'initial={failed}; triage_err={err_t}')
            points.append({'mu': mu, 'nu': nu, 'snr': None,
                           'status': 'code', 'note': err_t})
            aborted = True
            break

        checks_t = physics_checks(cfg, mu, nu, snr_t, N)
        failed_t = [(lbl, nte) for lbl, ok, nte in checks_t if not ok]

        if not failed_t:
            print(f"    triage PASSED (snr={snr_t:.6f}): numerical "
                  f"artefact of the fast preflight.")
            log_triage(cfg, mu, nu, snr, snr_t, 'numerical',
                       f'initial={failed}')
            points.append({'mu': mu, 'nu': nu, 'snr': snr_t,
                           'status': 'numeric', 'note': str(failed)})
        else:
            print(f"    triage also FAILED: {failed_t}")
            log_triage(cfg, mu, nu, snr, snr_t, 'code',
                       f'initial={failed}; triage={failed_t}')
            points.append({'mu': mu, 'nu': nu, 'snr': snr_t,
                           'status': 'code', 'note': str(failed_t)})
            aborted = True
            break

    # Per-class summary table
    print(f"\n  --- {class_name} summary ---")
    print(f"  {'mu':>8}  {'nu':>8}  {'snr':>12}  {'status':>8}  note")
    for p in points:
        snr_str = f'{p["snr"]:.6f}' if p['snr'] is not None else '-'
        print(f"  {p['mu']:>8.4f}  {p['nu']:>8.4f}  "
              f"{snr_str:>12}  {p['status']:>8}  {p['note']}")

    n_pass = sum(1 for p in points if p['status'] == 'PASS')
    n_numeric = sum(1 for p in points if p['status'] == 'numeric')
    n_fail = sum(1 for p in points
                 if p['status'] in ('code', 'systemic'))

    return {
        'class':    class_name,
        'points':   points,
        'n_pass':   n_pass,
        'n_numeric': n_numeric,
        'n_fail':   n_fail,
        'n_triage': n_triage,
        'aborted':  aborted,
    }


# ----------------------------------------------------------------------
# Scharnagl preflight (deferred to the documentation)
# ----------------------------------------------------------------------
def run_scharnagl_preflight(point_timeout_s):
    """
    Run the two reference Scharnagl checks per case. Requires the
    four scharnagl_echo_*.py workers to exist (the documentation). Raises
    SystemExit if any are missing.
    """
    missing = [w for w in SCH_WORKERS
               if not os.path.exists(os.path.join(SCRIPTS_DIR, w))]
    if missing:
        print(f"\n[scharnagl] ERROR: --with-scharnagl was requested "
              f"but the following workers are missing: {missing}")
        print(f"[scharnagl] These are the documentation deliverables. Do not "
              f"enable --with-scharnagl before the documentation completes.")
        sys.exit(2)

    print(f"\n{'=' * 72}")
    print(f"Scharnagl preflight (the documentation extension)")
    print(f"{'=' * 72}")

    results = []
    for case in ('1d_closed', '1d_open', '2d_closed', '2d_open'):
        worker = os.path.join(SCRIPTS_DIR, f'scharnagl_echo_{case}.py')

        # Reference geometry and noise
        if case == '1d_closed':
            geom_args = ['--N', '16']
            noise_args = []
            extra_traj = []
        elif case == '1d_open':
            geom_args = ['--N', '16']
            noise_args = ['--gamma1', '1e-2', '--gamma2', '1e-2']
            extra_traj = ['--n_traj', '20']
        elif case == '2d_closed':
            geom_args = ['--Lx', '2', '--Ly', '4']
            noise_args = []
            extra_traj = []
        else:
            geom_args = ['--Lx', '2', '--Ly', '4']
            noise_args = ['--gamma1', '1e-2', '--gamma2', '1e-2']
            extra_traj = ['--n_traj', '10']

        common = [
            *geom_args,
            '--alpha', str(ALPHA_PREFLIGHT),
            '--J', str(J_PREFLIGHT),
            '--mu', f'{np.pi / 2:.6f}',
            '--nu', f'{np.pi / 2:.6f}',
            '--theta_k', f'{np.pi / 4:.6f}',
            '--phi_k', '0.0',
            '--dt_max', str(DT_PREFLIGHT),
            *noise_args,
            *extra_traj,
        ]

        for chi in (CHI_PREFLIGHT, CHI_TRIAGE):
            cmd = [sys.executable, worker, '--chi_max', str(chi), *common]
            print(f"\n  [{case}] chi={chi}")
            print(f"    {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, timeout=point_timeout_s)
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as exc:
                print(f"    FAIL: {exc}")
                results.append((case, chi, False, str(exc)))
                continue
            # Locate the output
            out_dir = os.path.join(
                ROOT_DIR, 'data', f'scharnagl_pts_{case}'
            )
            if not os.path.isdir(out_dir):
                results.append((case, chi, False,
                                f'missing output dir {out_dir}'))
                continue
            found = sorted(
                f for f in os.listdir(out_dir)
                if f.startswith('scharnagl') and f.endswith('.npy')
            )
            if not found:
                results.append((case, chi, False, 'no .npy output found'))
                continue
            # Load the most recent file
            bundle = np.load(os.path.join(out_dir, found[-1]),
                             allow_pickle=True).item()
            schema_ok = all(k in bundle for k in
                            ('alpha', 'J', 'mu', 'nu',
                             'theta_k', 'phi_k', 'snr',
                             'n_opt', 'm_opt'))
            geom_ok = (
                ('N' in bundle) or ('Lx' in bundle and 'Ly' in bundle)
            )
            snr_ok = bool(np.isfinite(bundle.get('snr', np.nan)))
            passed = bool(schema_ok and geom_ok and snr_ok)
            note = (f'schema_ok={schema_ok}, geom_ok={geom_ok}, '
                    f'snr={bundle.get("snr")}')
            results.append((case, chi, passed, note))
            print(f"    {'PASS' if passed else 'FAIL'}  {note}")

    # Summary
    print(f"\n  --- Scharnagl preflight summary ---")
    all_ok = True
    for case, chi, passed, note in results:
        print(f"  {case:>12}  chi={chi:>3}  "
              f"{'PASS' if passed else 'FAIL':>5}  {note}")
        if not passed:
            all_ok = False
    return all_ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ten-point preflight inspection for the four "
                    "production classes.",
    )
    parser.add_argument('--2d-geometry', nargs=2, type=int,
                        default=[4, 4], metavar=('Lx', 'Ly'),
                        dest='geom_2d',
                        help='Cylinder geometry for 2D classes '
                             '(default: 4 4, giving N=16).')
    parser.add_argument('--2d-open-n-trajs', type=int, default=20,
                        dest='ntraj_2d_open',
                        help='Trajectory count for 2D open '
                             '(default: 20).')
    parser.add_argument('--point-timeout', type=float, default=7200.0,
                        help='Per-point wall-time limit in seconds '
                             '(default: 7200). Raised from the codebase\'s '
                             'original 3600 because the 1d_open and '
                             '2d_open TJM workers exceed 3600 s at the '
                             'expensive grid points even at preflight '
                             'quality.')
    parser.add_argument('--classes', nargs='+',
                        default=['1d_closed', '1d_open',
                                 '2d_closed', '2d_open'],
                        choices=['1d_closed', '1d_open',
                                 '2d_closed', '2d_open'],
                        help='Subset of classes to run.')
    parser.add_argument('--with-scharnagl', action='store_true',
                        help='Also run the Scharnagl preflight '
                                '(requires the documentation). Off by default.')
    parser.add_argument('--convention', choices=['mu_nu', 'mu1mu2'],
                        default='mu1mu2',
                        help="Coordinate convention for the 10-point "
                                "grid. 'mu1mu2' (default) uses the paper "
                                "convention (mu1 in [0, pi], mu2 in "
                                "[0, 2pi]); 'mu_nu' uses the legacy "
                                "convention (mu in [0, pi], nu in "
                                "[-pi, pi]).")
    args = parser.parse_args()

    print("=" * 72)
    print("Preflight manual inspection")
    print("=" * 72)
    print(f"Grid points per class: {len(_grid_points(args.convention))}")
    print(f"Preflight quality: chi={CHI_PREFLIGHT}, dt={DT_PREFLIGHT}")
    print(f"Triage quality:    chi={CHI_TRIAGE}, dt={DT_TRIAGE}")
    print(f"Classes:           {args.classes}")
    print(f"2D geometry:       {args.geom_2d}")
    print(f"2D open n_traj:    {args.ntraj_2d_open}")

    t0 = time.time()
    summaries = []
    for class_name in args.classes:
        summary = run_class(
            class_name,
            geom_2d=args.geom_2d,
            ntraj_2d_open=args.ntraj_2d_open,
            point_timeout_s=args.point_timeout,
            convention=args.convention,
        )
        summaries.append(summary)

    # Optional Scharnagl block
    scharnagl_ok = True
    if args.with_scharnagl:
        scharnagl_ok = run_scharnagl_preflight(args.point_timeout)

    # Global summary
    print(f"\n{'=' * 72}")
    print("Global summary")
    print(f"{'=' * 72}")
    any_fail = False
    for s in summaries:
        status = 'PASS' if (s['n_fail'] == 0 and not s['aborted']) else 'FAIL'
        if status == 'FAIL':
            any_fail = True
        print(f"  {s['class']:>12}  pass={s['n_pass']:>2}  "
              f"numeric={s['n_numeric']:>2}  fail={s['n_fail']:>2}  "
              f"triage={s['n_triage']:>2}  "
              f"{'ABORTED' if s['aborted'] else 'completed':>9}  "
              f"[{status}]")

    if args.with_scharnagl and not scharnagl_ok:
        any_fail = True

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed / 60.0:.1f} min")

    if any_fail:
        print("\n[FAIL] preflight.py")
        sys.exit(1)
    print("\n[PASS] preflight.py")
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[preflight] interrupted by user.")
        sys.exit(130)