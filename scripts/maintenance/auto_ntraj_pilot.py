#!/usr/bin/env python3
"""
scripts/maintenance/auto_ntraj_pilot.py
Automatic trajectory-count selector for open-system production.

v2: variance-guided pilot, migrated to mu1mu2 convention.

Workflow
--------
--submit              Two-stage submission per tuple:
                        Stage A: coarse variance pre-scan on a
                                20x20 (mu1, mu2) grid at chi=32
                                with 5 batches x 20 trajectories.
                        Stage B: (chi, n_traj) grid on the union
                                grid with 5 seeds per (chi, n_traj).

--aggregate           Apply converge_ntraj_with_chi per tuple and
                      write data/resource_tables/ntraj_conv.csv.

--sanity              Offline machinery test in mu1mu2 convention.

--run-batch-a IDX TASK_ID   Internal
--run-batch-b IDX TASK_ID   Internal
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
    PILOT_GRID_MU1MU2,
    N_BATCHES_VARIANCE, N_TRAJ_PER_BATCH_VARIANCE,
    VARIANCE_POINTS_TOP_K, VARIANCE_PILOT_CAP, SEED_BATCH_BASE,
    CHI_LIST_NTRAJ, NTRAJ_LIST, N_SEEDS_NTRAJ, TARGET_STD_DEFAULT,
    batch_output_path, read_batch_snr,
    identify_highest_variance_points, build_traj_pilot,
    converge_ntraj_with_chi,
    write_pilot_grid, write_grid_pilot, read_grid_pilot,
    worker_command, snr_from_output, _grid_axes,
)

RESOURCE_DIR = os.path.join(ROOT_DIR, 'data', 'resource_tables')

# Convention used for all pilot runs.
CONVENTION = "mu1mu2"


# ----------------------------------------------------------------------
# Production tuple list (the codebase test tuples + the codebase/4 production)
# ----------------------------------------------------------------------
PRODUCTION_TUPLES = (
    # the codebase test tuples (N=4, 8 x alpha=3, 6 x 1d, 2d open)
    [
        {'case': '1d_open', 'N': 4, 'alpha': a,
         'gamma1': 1e-2, 'gamma2': 1e-2}
        for a in (3.0, 6.0)
    ] + [
        {'case': '1d_open', 'N': 8, 'alpha': a,
         'gamma1': 1e-2, 'gamma2': 1e-2}
        for a in (3.0, 6.0)
    ] + [
        {'case': '2d_open', 'Lx': 2, 'Ly': 2, 'alpha': a,
         'gamma1': 1e-2, 'gamma2': 1e-2}
        for a in (3.0, 6.0)
    ] + [
        {'case': '2d_open', 'Lx': 2, 'Ly': 4, 'alpha': a,
         'gamma1': 1e-2, 'gamma2': 1e-2}
        for a in (3.0, 6.0)
    ]
    # the codebase production (1d_open flagship)
    + [
        {'case': '1d_open', 'N': 16, 'alpha': a,
         'gamma1': 0.0, 'gamma2': 0.1}
        for a in (0.0, 1.0, 3.0)
    ]
    + [
        {'case': '1d_open', 'N': 16, 'alpha': 3.0,
         'gamma1': 0.0, 'gamma2': 0.5},
        {'case': '1d_open', 'N': 16, 'alpha': 3.0,
         'gamma1': 1e-2, 'gamma2': 1e-2},
        {'case': '1d_open', 'N': 16, 'alpha': 1.0,
         'gamma1': 0.0, 'gamma2': 0.5},
        {'case': '1d_open', 'N': 16, 'alpha': 1.0,
         'gamma1': 0.0, 'gamma2': 1.0},
    ]
    # the codebase production (1d_open exploratory)
    + [
        {'case': '1d_open', 'N': 32, 'alpha': a,
         'gamma1': 0.0, 'gamma2': 0.1}
        for a in (0.0, 1.0, 3.0)
    ]
    + [
        {'case': '1d_open', 'N': 16, 'alpha': a,
         'gamma1': 0.0, 'gamma2': 0.1}
        for a in (2.0, 4.0, 6.0)
    ]
    + [
        {'case': '1d_open', 'N': 64, 'alpha': a,
         'gamma1': 0.0, 'gamma2': 0.1}
        for a in (0.0, 3.0)
    ]
    # the codebase production (2d_open flagship)
    + [
        {'case': '2d_open', 'Lx': 3, 'Ly': 4, 'alpha': 3.0,
         'gamma1': 0.0, 'gamma2': g2}
        for g2 in (1e-2, 0.1, 0.5)
    ]
    + [
        {'case': '2d_open', 'Lx': 3, 'Ly': 4, 'alpha': 0.0,
         'gamma1': 0.0, 'gamma2': 0.1},
    ]
)


# ----------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------
def _resources(case, params):
    N = params.get('N')
    Lx, Ly = params.get('Lx'), params.get('Ly')
    if case == '1d_open':
        if N is not None and N >= 32:
            return 32, '24:00:00'
        return 32, '12:00:00'
    if case == '2d_open':
        n_sites = (Lx or 0) * (Ly or 0)
        return (128, '36:00:00') if n_sites >= 32 else (64, '24:00:00')
    raise ValueError(f"Unknown case: {case!r}")


# ----------------------------------------------------------------------
# SLURM script generation
# ----------------------------------------------------------------------
def _write_stage_a_script(tuple_idx, td):
    case = td['case']
    mem, walltime = _resources(case, td)
    grid = 20
    n_tasks = grid * grid * N_BATCHES_VARIANCE

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_ntraj_pilot_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'auto_ntraj_pilot')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = f"ntrajA_{case}_t{tuple_idx:02d}"
    script_path = os.path.join(jobdir, f"job_{tag}.sh")
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

python scripts/maintenance/auto_ntraj_pilot.py \\
   --run-batch-a {tuple_idx} ${{SLURM_ARRAY_TASK_ID}}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


def _write_stage_b_script(tuple_idx, td, dependency_jobid=None):
    case = td['case']
    mem, walltime = _resources(case, td)

    params = {
        'N': td.get('N'), 'Lx': td.get('Lx'), 'Ly': td.get('Ly'),
        'alpha': td['alpha'],
        'gamma1': td.get('gamma1', 0.0), 'gamma2': td.get('gamma2', 0.0),
    }
    union = read_grid_pilot(case, params, convention=CONVENTION) \
            or list(PILOT_GRID_MU1MU2)
    n_points = len(union)
    n_tasks = (n_points * len(CHI_LIST_NTRAJ)
               * len(NTRAJ_LIST) * N_SEEDS_NTRAJ)

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_ntraj_pilot_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'auto_ntraj_pilot')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = f"ntrajB_{case}_t{tuple_idx:02d}"
    script_path = os.path.join(jobdir, f"job_{tag}.sh")
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

python scripts/maintenance/auto_ntraj_pilot.py \\
   --run-batch-b {tuple_idx} ${{SLURM_ARRAY_TASK_ID}}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


# ----------------------------------------------------------------------
# Task ID decoding (convention-aware)
# ----------------------------------------------------------------------
def _decode_stage_a_task(task_id, grid):
    """task_id = (i_mu1 * grid + j_mu2) * n_batches + batch."""
    n_batches = N_BATCHES_VARIANCE
    batch_idx = task_id % n_batches
    point_idx = task_id // n_batches
    i = point_idx // grid
    j = point_idx % grid
    a_vals, b_vals = _grid_axes(grid, convention=CONVENTION)
    return (float(a_vals[i]), float(b_vals[j]), int(batch_idx))


def _decode_stage_b_task(task_id, n_points):
    n_chi = len(CHI_LIST_NTRAJ)
    n_ntraj = len(NTRAJ_LIST)
    n_seeds = N_SEEDS_NTRAJ
    seed_idx = task_id % n_seeds
    rest = task_id // n_seeds
    ntraj_idx = rest % n_ntraj
    rest = rest // n_ntraj
    chi_idx = rest % n_chi
    point_idx = rest // n_chi
    return (point_idx, chi_idx, ntraj_idx, seed_idx)


# ----------------------------------------------------------------------
# Stage task execution
# ----------------------------------------------------------------------
def run_batch_a(tuple_idx, task_id):
    td = PRODUCTION_TUPLES[tuple_idx]
    case = td['case']
    mu1, mu2, batch_idx = _decode_stage_a_task(task_id, grid=20)
    seed = SEED_BATCH_BASE + batch_idx
    chi = 32
    n_traj = N_TRAJ_PER_BATCH_VARIANCE

    params = {
        'N': td.get('N'), 'Lx': td.get('Lx'), 'Ly': td.get('Ly'),
        'alpha': td['alpha'],
        'gamma1': td.get('gamma1', 0.0), 'gamma2': td.get('gamma2', 0.0),
    }

    out_path = batch_output_path(case, params, chi,
                                 n_traj=n_traj, seed=seed,
                                 convention=CONVENTION,
                                 mu1=mu1, mu2=mu2)
    if os.path.exists(out_path):
        print(f"[skip] {out_path}", flush=True)
        return

    cmd = worker_command(
        case, chi,
        N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
        alpha=td['alpha'],
        gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
        n_traj=n_traj, seed=seed,
        convention=CONVENTION, mu1=mu1, mu2=mu2,
    )
    print(f"[run-a] tuple={tuple_idx} batch={batch_idx} "
          f"mu1={mu1:.4f} mu2={mu2:.4f} chi={chi} "
          f"n_traj={n_traj} seed={seed}", flush=True)
    subprocess.run(cmd, check=True)


def run_batch_b(tuple_idx, task_id):
    td = PRODUCTION_TUPLES[tuple_idx]
    case = td['case']

    params = {
        'N': td.get('N'), 'Lx': td.get('Lx'), 'Ly': td.get('Ly'),
        'alpha': td['alpha'],
        'gamma1': td.get('gamma1', 0.0), 'gamma2': td.get('gamma2', 0.0),
    }
    union = read_grid_pilot(case, params, convention=CONVENTION) \
            or list(PILOT_GRID_MU1MU2)

    point_idx, chi_idx, ntraj_idx, seed_idx = _decode_stage_b_task(
        task_id, len(union)
    )
    mu1, mu2 = union[point_idx]
    chi = CHI_LIST_NTRAJ[chi_idx]
    n_traj = NTRAJ_LIST[ntraj_idx]
    seed = SEED_BATCH_BASE + seed_idx

    out_path = batch_output_path(case, params, chi,
                                 n_traj=n_traj, seed=seed,
                                 convention=CONVENTION,
                                 mu1=mu1, mu2=mu2)
    if os.path.exists(out_path):
        print(f"[skip] {out_path}", flush=True)
        return

    cmd = worker_command(
        case, chi,
        N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
        alpha=td['alpha'],
        gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
        n_traj=n_traj, seed=seed,
        convention=CONVENTION, mu1=mu1, mu2=mu2,
    )
    print(f"[run-b] tuple={tuple_idx} point={point_idx} "
          f"mu1={mu1:.4f} mu2={mu2:.4f} chi={chi} "
          f"n_traj={n_traj} seed={seed}", flush=True)
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------
def _submit(script):
    r = subprocess.run(['sbatch', '--parsable', script],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def submit_all(dry_run=False, cases=None):
    for idx, td in enumerate(PRODUCTION_TUPLES):
        if cases and td['case'] not in cases:
            continue
        sa = _write_stage_a_script(idx, td)
        if dry_run:
            print(f"[dry-run] stage A: {sa}")
            sb = _write_stage_b_script(idx, td)
            print(f"[dry-run] stage B (no dep): {sb}")
            continue

        sa_jobid = _submit(sa)
        print(f"[submit] stage A job {sa_jobid}: {sa}")

        sb = _write_stage_b_script(idx, td, dependency_jobid=sa_jobid)
        sb_jobid = _submit(sb)
        print(f"[submit] stage B job {sb_jobid} (dep {sa_jobid}): {sb}")


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def aggregate():
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    output_csv = os.path.join(RESOURCE_DIR, 'ntraj_conv.csv')
    fieldnames = [
        'case', 'N', 'Lx', 'Ly', 'alpha', 'gamma1', 'gamma2',
        'chi_floor', 'ntraj_conv', 'ntraj_prod',
        'fit_a', 'fit_b',
    ]

    rows = []
    for idx, td in enumerate(PRODUCTION_TUPLES):
        case = td['case']
        params = {
            'N': td.get('N'), 'Lx': td.get('Lx'), 'Ly': td.get('Ly'),
            'alpha': td['alpha'],
            'gamma1': td.get('gamma1', 0.0), 'gamma2': td.get('gamma2', 0.0),
        }

        try:
            union = build_traj_pilot(case, params, convention=CONVENTION)
            write_pilot_grid(case, union,
                             N=td.get('N'), Lx=td.get('Lx'),
                             Ly=td.get('Ly'), alpha=td['alpha'],
                             gamma1=td.get('gamma1', 0.0),
                             gamma2=td.get('gamma2', 0.0),
                             convention=CONVENTION)
            write_grid_pilot(case, params, union, convention=CONVENTION)
        except Exception as exc:
            print(f"[union] {case} t{idx}: {exc}", flush=True)
            union = list(PILOT_GRID_MU1MU2)

        try:
            result = converge_ntraj_with_chi(
                case, params,
                target_std=TARGET_STD_DEFAULT,
                union_grid=union,
                convention=CONVENTION,
            )
        except Exception as exc:
            print(f"[converge] {case} t{idx} failed: {exc}", flush=True)
            result = {'chi_floor': None, 'ntraj_conv': None, 'per_chi': {}}

        chi_floor = result.get('chi_floor')
        ntraj_conv = result.get('ntraj_conv')
        ntraj_prod = None
        if ntraj_conv is not None:
            for n in NTRAJ_LIST:
                if n >= 1.5 * ntraj_conv:
                    ntraj_prod = n
                    break
            if ntraj_prod is None:
                ntraj_prod = NTRAJ_LIST[-1]

        fit_a = result['per_chi'].get(chi_floor, {}).get('a', float('nan'))
        fit_b = result['per_chi'].get(chi_floor, {}).get('b', float('nan'))

        row = {
            'case': case,
            'N': td.get('N', ''), 'Lx': td.get('Lx', ''),
            'Ly': td.get('Ly', ''),
            'alpha': td['alpha'],
            'gamma1': td.get('gamma1', 0.0),
            'gamma2': td.get('gamma2', 0.0),
            'chi_floor': chi_floor if chi_floor is not None else '',
            'ntraj_conv': ntraj_conv if ntraj_conv is not None else '',
            'ntraj_prod': ntraj_prod if ntraj_prod is not None else '',
            'fit_a': f"{fit_a:.4e}" if np.isfinite(fit_a) else '',
            'fit_b': f"{fit_b:.4e}" if np.isfinite(fit_b) else '',
        }
        rows.append(row)
        print(f"[row] {case} N={td.get('N')} Lx={td.get('Lx')} "
              f"Ly={td.get('Ly')} alpha={td['alpha']} "
              f"g1={td.get('gamma1', 0.0)} "
              f"g2={td.get('gamma2', 0.0)} -> "
              f"chi_floor={chi_floor} ntraj_conv={ntraj_conv} "
              f"ntraj_prod={ntraj_prod}", flush=True)

    if rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote {output_csv} ({len(rows)} rows)")
    else:
        print("[warn] no rows written.")


# ----------------------------------------------------------------------
# Sanity (offline)
# ----------------------------------------------------------------------
def sanity():
    print('=' * 70)
    print('Sanity: trajectory-count selector (mu1mu2 convention)')
    print('=' * 70)

    # 1. PILOT_GRID_MU1MU2 in range
    for m1, m2 in PILOT_GRID_MU1MU2:
        assert 0 <= m1 <= np.pi + 1e-9, f"pilot mu1={m1}"
        assert 0 <= m2 <= 2 * np.pi + 1e-9, f"pilot mu2={m2}"
    print(f'(a) PILOT_GRID_MU1MU2 {len(PILOT_GRID_MU1MU2)} points '
          f'in range: PASS')

    # 2. build_traj_pilot with monkeypatched variance detector
    import src.selectors as _sel
    orig = _sel.identify_highest_variance_points
    _sel.identify_highest_variance_points = \
        lambda *a, **k: [(0.3, 1.5), (0.5, 2.5), (1.0, 3.0)]
    try:
        params = {'N': 16, 'alpha': 0.0,
                  'gamma1': 0.0, 'gamma2': 0.1}
        union = build_traj_pilot('1d_open', params,
                                 convention=CONVENTION)
    finally:
        _sel.identify_highest_variance_points = orig
    for m1, m2 in union:
        assert 0 <= m1 <= np.pi + 1e-9, f"union mu1={m1}"
        assert 0 <= m2 <= 2 * np.pi + 1e-9, f"union mu2={m2}"
    print(f'(b) build_traj_pilot {len(union)} points in range: PASS')

    # 3. _fit_sigma_vs_ntraj round-trip
    from src.selectors import _fit_sigma_vs_ntraj
    a_true, b_true = 0.5, 0.01
    ntraj = [50, 100, 200, 500, 1000]
    sigmas = [a_true / np.sqrt(n) + b_true for n in ntraj]
    a_fit, b_fit = _fit_sigma_vs_ntraj(ntraj, sigmas)
    assert abs(a_fit - a_true) < 1e-6
    assert abs(b_fit - b_true) < 1e-6
    print(f'(c) fit recovery a={a_fit:.6f} b={b_fit:.6f}: PASS')

    # 4. Batch path suffix
    p = batch_output_path('1d_open',
                          {'N': 16, 'alpha': 1.0,
                           'gamma1': 0.0, 'gamma2': 0.1},
                          32, n_traj=20, seed=20261024,
                          convention=CONVENTION,
                          mu1=0.5, mu2=1.5)
    base = os.path.basename(p)
    assert 'mu10.5000_mu21.5000' in base, base
    assert '_nu' not in base, base
    print(f'(d) batch_output_path suffix: PASS  ({base})')

    # 5. Tuple list coverage
    cases = {td['case'] for td in PRODUCTION_TUPLES}
    assert '1d_open' in cases and '2d_open' in cases
    print(f'(e) PRODUCTION_TUPLES {len(PRODUCTION_TUPLES)} tuples, '
          f'cases={sorted(cases)}: PASS')

    print('\n[PASS] sanity check')
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--cases', nargs='+',
                        choices=['1d_open', '2d_open'],
                        default=None)
    parser.add_argument('--run-batch-a', nargs=2, default=None,
                        metavar=('TUPLE_IDX', 'TASK_ID'))
    parser.add_argument('--run-batch-b', nargs=2, default=None,
                        metavar=('TUPLE_IDX', 'TASK_ID'))
    args = parser.parse_args()

    if args.run_batch_a is not None:
        idx, tid = args.run_batch_a
        run_batch_a(int(idx), int(tid))
        return
    if args.run_batch_b is not None:
        idx, tid = args.run_batch_b
        run_batch_b(int(idx), int(tid))
        return

    if args.sanity:
        sys.exit(0 if sanity() else 1)

    did_something = False
    if args.submit:
        submit_all(dry_run=args.dry_run, cases=args.cases)
        did_something = True
    if args.aggregate:
        aggregate()
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == '__main__':
    main()