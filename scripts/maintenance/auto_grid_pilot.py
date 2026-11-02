"""
scripts/maintenance/auto_grid_pilot.py

Automatic grid-resolution selector for the production runs of
Phases 1-4 (v2: sharp-feature-guided pilot grid).

Modes
-----
 --submit              Two-stage submission per tuple:
                         Stage A (single array): run the coarse 20x20
                             scan at chi=32 on the cluster. When it
                             completes, identify the sharpest gradient
                             points and write the union grid.
                         Stage B (four arrays, depends on Stage A):
                             run the four landscape resolutions
                             20, 40, 60, 80 at the same chi=32.

 --aggregate           Read the aggregated landscapes, apply the
                       choose_grid_size criterion, write
                       data/resource_tables/grid_conv.csv.

 --sanity              Run the sanity check against synthetic data.

 --run-grid-task IDX GRID CHI TASK_ID
                       Internal: run one grid point in the SLURM array.

 --aggregate-tuple IDX GRID
                       Internal: aggregate one (tuple, grid) into a
                       landscape .npy file.

This driver is a diagnostic. It is not part of the production pipeline.
The production grid sizes are defined by GRID_POLICY in
submit_production.py. Run this driver manually if a specific tuple's
resolution is uncertain and you want the sharp-point data to decide.

Convention
----------
All coordinates are in the Vineesha convention (mu1, mu2) with
mu1 in [0, pi] and mu2 in [0, 2*pi]. The CONVENTION constant below
controls this. All calls to worker_command, native_output_path,
grid_pilot_filename, landscape_grid_filename, and the pilot data
readers pass convention=CONVENTION.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, ROOT_DIR)

from src.selectors import (
    PILOT_GRID, PILOT_GRID_MU1MU2, GRID_SCHEDULE, REFERENCE_GRID,
    COARSE_GRADIENT_CHI, COARSE_GRADIENT_GRID, SHARP_POINTS_TOP_K,
    GRID_PILOT_CAP, PILOT_GRID_ERROR_TARGET,
    N_TRAJ_GRID_PILOT,
    worker_command, native_output_path, snr_from_output,
    grid_pilot_filename, read_grid_pilot, write_grid_pilot,
    landscape_grid_filename, read_landscape_grid, write_landscape_grid,
    interpolate_to_reference, error_on_pilot_subset,
    identify_sharpest_landscape_points, build_grid_pilot,
    choose_grid_size,
)

RESOURCE_DIR = os.path.join(ROOT_DIR, 'data', 'resource_tables')

# Convention used for all pilot runs.
CONVENTION = "mu1mu2"


# ---------------------------------------------------------------------
# Production tuple list. Same 44 tuples as auto_chi_pilot.py so both
# pilots cover the same parameter space.
# ---------------------------------------------------------------------
PRODUCTION_TUPLES = (
    # ---- 1D closed: 4 the codebase test + 30 the codebase production = 34 ----
    [
        {'case': '1d_closed', 'N': 4, 'alpha': 3.0},
        {'case': '1d_closed', 'N': 4, 'alpha': 6.0},
        {'case': '1d_closed', 'N': 8, 'alpha': 3.0},
        {'case': '1d_closed', 'N': 8, 'alpha': 6.0},
    ]
    + [
        {'case': '1d_closed', 'N': N, 'alpha': a}
        for N in [16, 32, 64, 128, 160, 200]
        for a in [1.0, 2.0, 3.0, 4.0, 6.0]
    ]
) + [
    # ---- 1D open (5 anchors) ----
    {'case': '1d_open', 'N': 16, 'alpha': 0.0, 'gamma1': 0.0, 'gamma2': 0.1},
    {'case': '1d_open', 'N': 16, 'alpha': 1.0, 'gamma1': 0.0, 'gamma2': 0.1},
    {'case': '1d_open', 'N': 16, 'alpha': 3.0, 'gamma1': 0.0, 'gamma2': 0.1},
    {'case': '1d_open', 'N': 32, 'alpha': 0.0, 'gamma1': 0.0, 'gamma2': 0.1},
    {'case': '1d_open', 'N': 32, 'alpha': 3.0, 'gamma1': 0.0, 'gamma2': 0.1},
    # ---- 2D closed (4 anchors) ----
    {'case': '2d_closed', 'Lx': 4, 'Ly': 4, 'alpha': 1.0},
    {'case': '2d_closed', 'Lx': 4, 'Ly': 4, 'alpha': 3.0},
    {'case': '2d_closed', 'Lx': 4, 'Ly': 4, 'alpha': 6.0},
    {'case': '2d_closed', 'Lx': 12, 'Ly': 4, 'alpha': 3.0},
    # ---- 2D open (1 anchor) ----
    {'case': '2d_open', 'Lx': 4, 'Ly': 4, 'alpha': 3.0,
     'gamma1': 0.0, 'gamma2': 0.1},
]


# ---------------------------------------------------------------------
# Task-ID -> (x, y) mapping, convention-aware
# ---------------------------------------------------------------------
def task_id_to_grid_point(task_id, grid_size, convention="mu_nu"):
    """
    Map a SLURM array task_id to a grid point.

    convention='mu_nu':   (mu in [0, pi], nu in [-pi, pi])
    convention='mu1mu2':  (mu1 in [0, pi], mu2 in [0, 2*pi])

    Row-major: task_id = i * grid_size + j
        i = task_id // grid_size   (first-axis index)
        j = task_id %  grid_size   (second-axis index)
    """
    i = int(task_id) // int(grid_size)
    j = int(task_id) %  int(grid_size)
    x_vals = np.linspace(0.0, np.pi, grid_size)
    if convention == "mu1mu2":
        y_vals = np.linspace(0.0, 2.0 * np.pi, grid_size)
    else:
        y_vals = np.linspace(-np.pi, np.pi, grid_size)
    return float(x_vals[i]), float(y_vals[j])


# ---------------------------------------------------------------------
# Resource assignment
# ---------------------------------------------------------------------
def _resources(case, params, grid_size):
    N  = params.get('N')
    Lx = params.get('Lx')
    Ly = params.get('Ly')

    if case == '1d_closed':
        base_mem = 128 if (N is not None and N >= 128) else 32
    elif case == '1d_open':
        base_mem = 32
    elif case == '2d_closed':
        n_sites = (Lx or 0) * (Ly or 0)
        base_mem = 224 if n_sites >= 48 else 128
    elif case == '2d_open':
        base_mem = 128
    else:
        raise ValueError(f"Unknown case: {case!r}")

    if grid_size <= 20:
        walltime = '12:00:00'
    elif grid_size <= 40:
        walltime = '24:00:00'
    else:
        walltime = '48:00:00'

    return base_mem, walltime


# ---------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------
def _write_stage_a_script(tuple_idx, td):
    case = td['case']
    g = COARSE_GRADIENT_GRID
    chi = COARSE_GRADIENT_CHI
    mem, walltime = _resources(case, td, g)

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_grid_pilot_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'auto_grid_pilot')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = f"gridA_{case}_t{tuple_idx:02d}"
    script_path = os.path.join(jobdir, f"job_{tag}.sh")

    n_tasks = g * g
    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}G
#SBATCH --time={walltime}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --output={logdir}/{tag}_%A_%a.log
#SBATCH --error={logdir}/{tag}_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {ROOT_DIR}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/maintenance/auto_grid_pilot.py \\
    --run-grid-task {tuple_idx} {g} {chi} ${{SLURM_ARRAY_TASK_ID}}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


def _write_stage_b_script(tuple_idx, td, grid_size, dependency_jobid=None):
    case = td['case']
    chi = COARSE_GRADIENT_CHI
    mem, walltime = _resources(case, td, grid_size)

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_grid_pilot_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'auto_grid_pilot')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = f"gridB_{case}_t{tuple_idx:02d}_g{grid_size}"
    script_path = os.path.join(jobdir, f"job_{tag}.sh")

    n_tasks = grid_size * grid_size
    dep_line = (f"#SBATCH --dependency=afterok:{dependency_jobid}\n"
                if dependency_jobid is not None else "")

    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}G
#SBATCH --time={walltime}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --output={logdir}/{tag}_%A_%a.log
#SBATCH --error={logdir}/{tag}_%A_%a.err
{dep_line}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {ROOT_DIR}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/maintenance/auto_grid_pilot.py \\
    --run-grid-task {tuple_idx} {grid_size} {chi} ${{SLURM_ARRAY_TASK_ID}}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


# ---------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------
def _submit_stage_a(tuple_idx, td, dry_run=False):
    script = _write_stage_a_script(tuple_idx, td)
    if dry_run:
        print(f"[dry-run] stage A: {script}")
        return None
    result = subprocess.run(
        ['sbatch', '--parsable', script],
        capture_output=True, text=True, check=True,
    )
    jobid = result.stdout.strip()
    print(f"[submit] stage A job {jobid}: {script}")
    return jobid


def _submit_stage_b(tuple_idx, td, grid_size, dep_jobid, dry_run=False):
    script = _write_stage_b_script(tuple_idx, td, grid_size, dep_jobid)
    if dry_run:
        print(f"[dry-run] stage B g={grid_size}: {script}")
        return None
    result = subprocess.run(
        ['sbatch', '--parsable', script],
        capture_output=True, text=True, check=True,
    )
    jobid = result.stdout.strip()
    print(f"[submit] stage B g={grid_size} job {jobid}: {script}")
    return jobid


def submit_all(dry_run=False, cases=None):
    """
    Two-stage submission per tuple:
      1. Stage A (coarse 20x20 pre-scan) for the tuple.
      2. Stage B (four resolution arrays) depending on Stage A.

    If ``cases`` is a list of strings, only tuples whose 'case' field
    is in the list are submitted.
    """
    for idx, td in enumerate(PRODUCTION_TUPLES):
        if cases is not None and td['case'] not in cases:
            continue
        stage_a_jobid = _submit_stage_a(idx, td, dry_run=dry_run)

        for g in GRID_SCHEDULE:
            if g == COARSE_GRADIENT_GRID:
                continue
            _submit_stage_b(idx, td, g, stage_a_jobid, dry_run=dry_run)


# ---------------------------------------------------------------------
# Stage A / Stage B task execution
# ---------------------------------------------------------------------
def run_grid_task(tuple_idx, grid_size, chi_max, task_id):
    td = PRODUCTION_TUPLES[tuple_idx]
    case = td['case']

    a, b = task_id_to_grid_point(task_id, grid_size, CONVENTION)

    if CONVENTION == "mu1mu2":
        out_path = native_output_path(
            case, chi_max, convention="mu1mu2", mu1=a, mu2=b,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
            n_traj=N_TRAJ_GRID_PILOT,
        )
    else:
        out_path = native_output_path(
            case, chi_max, mu=a, nu=b,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
            n_traj=N_TRAJ_GRID_PILOT,
        )
    if os.path.exists(out_path):
        print(f"[skip] {out_path}", flush=True)
        return

    if CONVENTION == "mu1mu2":
        cmd = worker_command(
            case, chi_max, convention="mu1mu2", mu1=a, mu2=b,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
            n_traj=N_TRAJ_GRID_PILOT,
        )
    else:
        cmd = worker_command(
            case, chi_max, mu=a, nu=b,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
            n_traj=N_TRAJ_GRID_PILOT,
        )
    print(f"[run] task_id={task_id} a={a:.4f} b={b:.4f} "
          f"convention={CONVENTION}", flush=True)
    print(f"      {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------
# Landscape aggregation
# ---------------------------------------------------------------------
def aggregate_tuple_landscape(tuple_idx, grid_size):
    td = PRODUCTION_TUPLES[tuple_idx]
    case = td['case']
    chi_max = COARSE_GRADIENT_CHI

    if CONVENTION == "mu1mu2":
        a_vals = np.linspace(0.0, np.pi, grid_size)
        b_vals = np.linspace(0.0, 2.0 * np.pi, grid_size)
    else:
        a_vals = np.linspace(0.0, np.pi, grid_size)
        b_vals = np.linspace(-np.pi, np.pi, grid_size)
    snr_grid = np.full((grid_size, grid_size), np.nan)

    n_missing = 0
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            if CONVENTION == "mu1mu2":
                path = native_output_path(
                    case, chi_max, convention="mu1mu2",
                    mu1=float(a), mu2=float(b),
                    N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
                    alpha=td['alpha'],
                    gamma1=td.get('gamma1', 0.0),
                    gamma2=td.get('gamma2', 0.0),
                    n_traj=N_TRAJ_GRID_PILOT,
                )
            else:
                path = native_output_path(
                    case, chi_max, mu=float(a), nu=float(b),
                    N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
                    alpha=td['alpha'],
                    gamma1=td.get('gamma1', 0.0),
                    gamma2=td.get('gamma2', 0.0),
                    n_traj=N_TRAJ_GRID_PILOT,
                )
            val = snr_from_output(path)
            if val is None:
                n_missing += 1
            else:
                snr_grid[i, j] = val

    params = {
        'N': td.get('N'), 'Lx': td.get('Lx'), 'Ly': td.get('Ly'),
        'alpha': td['alpha'],
        'gamma1': td.get('gamma1', 0.0),
        'gamma2': td.get('gamma2', 0.0),
    }
    path = write_landscape_grid(case, params, grid_size, chi_max, snr_grid,
                                convention=CONVENTION)
    print(f"[aggregate] tuple={tuple_idx} grid={grid_size} "
          f"missing={n_missing}/{grid_size ** 2} -> {path}", flush=True)
    return path


# ---------------------------------------------------------------------
# Full aggregation and choose_grid_size
# ---------------------------------------------------------------------
def aggregate(cases=None):
    """
    Aggregate all landscapes for all tuples, build the union grid,
    run choose_grid_size, and write grid_conv.csv.
    """
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    output_csv = os.path.join(RESOURCE_DIR, 'grid_conv.csv')

    fieldnames = [
        'case', 'N', 'Lx', 'Ly', 'alpha', 'gamma1', 'gamma2',
        'grid_conv', 'grid_prod', 'max_pilot_error', 'coarse_snr_max',
    ]

    rows = []

    for idx, td in enumerate(PRODUCTION_TUPLES):
        if cases is not None and td['case'] not in cases:
            continue
        case = td['case']
        params = {
            'N': td.get('N'), 'Lx': td.get('Lx'), 'Ly': td.get('Ly'),
            'alpha': td['alpha'],
            'gamma1': td.get('gamma1', 0.0),
            'gamma2': td.get('gamma2', 0.0),
        }

        # Aggregate every scheduled resolution into a landscape .npy.
        for g in GRID_SCHEDULE:
            try:
                aggregate_tuple_landscape(idx, g)
            except Exception as exc:
                print(f"[aggregate] tuple={idx} grid={g} "
                      f"failed: {exc}", flush=True)

        # Build the union grid.
        try:
            union = build_grid_pilot(case, params, convention=CONVENTION)
            write_grid_pilot(case, params, union, convention=CONVENTION)
        except RuntimeError as exc:
            print(f"[union] {case} t{idx}: {exc}", flush=True)
            fallback = (list(PILOT_GRID_MU1MU2) if CONVENTION == "mu1mu2"
                        else list(PILOT_GRID))
            write_grid_pilot(case, params, fallback, convention=CONVENTION)

        # Choose the grid resolution.
        try:
            grid_conv, max_err, snr_max = choose_grid_size(
                case, params,
                grid_schedule=GRID_SCHEDULE,
                ref_grid_size=REFERENCE_GRID,
                target=PILOT_GRID_ERROR_TARGET,
                chi_max=COARSE_GRADIENT_CHI,
                convention=CONVENTION,
            )
        except Exception as exc:
            print(f"[choose_grid_size] {case} t{idx} failed: {exc}",
                  flush=True)
            grid_conv, max_err, snr_max = None, float('nan'), float('nan')

        grid_prod = None
        if grid_conv is not None:
            grid_prod = grid_conv
            for g in GRID_SCHEDULE:
                if g >= grid_conv:
                    grid_prod = g
                    break

        row = {
            'case': case,
            'N': td.get('N', ''),
            'Lx': td.get('Lx', ''),
            'Ly': td.get('Ly', ''),
            'alpha': td['alpha'],
            'gamma1': td.get('gamma1', 0.0),
            'gamma2': td.get('gamma2', 0.0),
            'grid_conv': grid_conv if grid_conv is not None else '',
            'grid_prod': grid_prod if grid_prod is not None else '',
            'max_pilot_error': (f"{max_err:.4e}"
                                if np.isfinite(max_err) else ''),
            'coarse_snr_max': (f"{snr_max:.4e}"
                               if np.isfinite(snr_max) else ''),
        }
        rows.append(row)
        print(f"[row] {case} N={td.get('N')} "
              f"Lx={td.get('Lx')} Ly={td.get('Ly')} "
              f"alpha={td['alpha']} -> "
              f"grid_conv={grid_conv} grid_prod={grid_prod} "
              f"(max_err={max_err:.3e}, "
              f"snr_max={snr_max:.3e})", flush=True)

    if rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote {output_csv} ({len(rows)} rows)")
    else:
        print("[warn] no rows written; nothing to aggregate.")


# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------
def sanity():
    """
    Synthetic sanity check for the grid-resolution machinery.
    """
    import tempfile
    import shutil

    print("=" * 70)
    print(f"Sanity check: grid-resolution selector (convention={CONVENTION})")
    print("=" * 70)

    tmp_root = tempfile.mkdtemp(prefix='grid_pilot_sanity_')
    try:
        case = '1d_closed'
        params = {'N': 16, 'alpha': 3.0, 'gamma1': 0.0, 'gamma2': 0.0}

        # Synthetic landscape at grid=20 with a sharp Gaussian peak
        # at (x, y) = (0.3, 2.9) in mu1mu2 convention.
        g = COARSE_GRADIENT_GRID
        if CONVENTION == "mu1mu2":
            x_vals = np.linspace(0.0, np.pi, g)
            y_vals = np.linspace(0.0, 2.0 * np.pi, g)
        else:
            x_vals = np.linspace(0.0, np.pi, g)
            y_vals = np.linspace(-np.pi, np.pi, g)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        peak = 5.0 * np.exp(-((X - 0.3) ** 2 + (Y - 2.9) ** 2) / 0.01)
        snr_grid = 1.0 + peak + 0.1 * np.sin(X) * np.cos(Y)
        write_landscape_grid(case, params, g, COARSE_GRADIENT_CHI,
                             snr_grid, root=tmp_root, convention=CONVENTION)

        # (a) sharp-point identification
        sharp = identify_sharpest_landscape_points(
            case, params, root=tmp_root, convention=CONVENTION,
        )
        print(f"(a) sharp points: {sharp[:5]}{'...' if len(sharp) > 5 else ''}")
        near_peak = [pt for pt in sharp
                     if abs(pt[0] - 0.3) < 0.6 and abs(pt[1] - 2.9) < 0.6]
        ok_a = len(near_peak) > 0
        print(f"    peak found: {'PASS' if ok_a else 'FAIL'}")

        # (b) union grid
        union = build_grid_pilot(case, params, root=tmp_root,
                                 convention=CONVENTION)
        print(f"(b) union grid ({len(union)} points): {union[:3]}...")
        ok_b = 3 <= len(union) <= GRID_PILOT_CAP
        print(f"    size within [3, {GRID_PILOT_CAP}]: "
              f"{'PASS' if ok_b else 'FAIL'}")

        # (c) resolution selection on finer landscapes
        for g_test in [40, 60, 80]:
            if CONVENTION == "mu1mu2":
                x_t = np.linspace(0.0, np.pi, g_test)
                y_t = np.linspace(0.0, 2.0 * np.pi, g_test)
            else:
                x_t = np.linspace(0.0, np.pi, g_test)
                y_t = np.linspace(-np.pi, np.pi, g_test)
            X_t, Y_t = np.meshgrid(x_t, y_t, indexing='ij')
            peak_t = 5.0 * np.exp(-((X_t - 0.3) ** 2
                                    + (Y_t - 2.9) ** 2) / 0.01)
            snr_t = 1.0 + peak_t + 0.1 * np.sin(X_t) * np.cos(Y_t)
            write_landscape_grid(case, params, g_test,
                                 COARSE_GRADIENT_CHI, snr_t,
                                 root=tmp_root, convention=CONVENTION)

        grid_conv, max_err, snr_max = choose_grid_size(
            case, params,
            grid_schedule=GRID_SCHEDULE,
            ref_grid_size=REFERENCE_GRID,
            target=PILOT_GRID_ERROR_TARGET,
            chi_max=COARSE_GRADIENT_CHI,
            root=tmp_root, convention=CONVENTION,
        )
        print(f"(c) grid_conv={grid_conv}, max_err={max_err:.3e}, "
              f"snr_max={snr_max:.3e}")
        ok_c = grid_conv is not None and grid_conv in GRID_SCHEDULE
        print(f"    returned a scheduled resolution: "
              f"{'PASS' if ok_c else 'FAIL'}")

        all_pass = ok_a and ok_b and ok_c
        print(f"\n{'[PASS] sanity check' if all_pass else '[FAIL] sanity check'}")
        return all_pass
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--cases', nargs='+', default=None,
                        help='Restrict to these case identifiers '
                             '(e.g. --cases 1d_closed).')
    parser.add_argument('--run-grid-task', nargs=4, default=None,
                        metavar=('TUPLE_IDX', 'GRID', 'CHI', 'TASK_ID'))
    parser.add_argument('--aggregate-tuple', nargs=2, default=None,
                        metavar=('TUPLE_IDX', 'GRID'))
    args = parser.parse_args()

    if args.run_grid_task is not None:
        idx, g, chi, tid = args.run_grid_task
        run_grid_task(int(idx), int(g), int(chi), int(tid))
        return

    if args.aggregate_tuple is not None:
        idx, g = args.aggregate_tuple
        aggregate_tuple_landscape(int(idx), int(g))
        return

    if args.sanity:
        ok = sanity()
        sys.exit(0 if ok else 1)

    did_something = False
    if args.submit:
        submit_all(dry_run=args.dry_run, cases=args.cases)
        did_something = True
    if args.aggregate:
        aggregate(cases=args.cases)
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == '__main__':
    main()