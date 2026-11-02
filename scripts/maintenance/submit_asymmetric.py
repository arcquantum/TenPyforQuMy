#!/usr/bin/env python3
"""
scripts/maintenance/submit_asymmetric.py

SLURM driver for asymmetric-echo production runs (the codebase.3c,
2.5b, 3.4, 4.7).

Grid:   t_plus  in linspace(0, pi,   grid_size)
        t_minus in linspace(0, 2*pi, grid_size)
Row-major task_id -> coordinate mapping, identical to
submit_production.py.

Worker mapping
--------------
  1d_closed  -> scripts/workers/closed_1d/asymmetric_echo_1d_closed.py
  1d_open    -> scripts/workers/open_1d/asymmetric_echo_1d_open.py
  2d_closed  -> scripts/workers/closed_2d/asymmetric_echo_2d_closed.py
  2d_open    -> scripts/workers/open_2d/asymmetric_echo_2d_open.py

Each worker writes to data/echo_pts_<case>_asym/asym_pt_<task_id>.npy
when --task_id is passed. bundle_results.py --case <case>_asym
reconstructs the bundle.

Tuple source
------------
Reads submit_production.PHASE_TUPLES filtered by case, so tuple
lists stay in one place.

CLI
---
  python3 scripts/maintenance/submit_asymmetric.py --case 1d_closed --submit
  python3 scripts/maintenance/submit_asymmetric.py --case 1d_closed --dry-run
  python3 scripts/maintenance/submit_asymmetric.py --case 1d_closed \
      --run-task TUPLE_IDX CHI GRID TASK_ID
  python3 scripts/maintenance/submit_asymmetric.py --case 1d_closed \
      --run-task TUPLE_IDX CHI GRID TASK_ID --convention mu1mu2
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime

import numpy as np


ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.selectors import (  # noqa: E402
    CHI_SCHEDULE,
    find_chi_convergence,
    read_pilot_grid,
)


RESOURCE_DIR = os.path.join(ROOT_DIR, 'data', 'resource_tables')
JOB_IDS_CSV = os.path.join(RESOURCE_DIR, 'job_ids_asym.csv')
MAX_ARRAY_TASKS = 1001
DEFAULT_WALLTIME = '14-00:00:00'
MAX_MEM_GB = 224


ASYM_WORKERS = {
    '1d_closed': 'asymmetric_echo_1d_closed.py',
    '1d_open':   'asymmetric_echo_1d_open.py',
    '2d_closed': 'asymmetric_echo_2d_closed.py',
    '2d_open':   'asymmetric_echo_2d_open.py',
}


# ----------------------------------------------------------------------
# Tuple source (imports from submit_production so lists stay in sync)
# ----------------------------------------------------------------------
def _load_phase_tuples():
    sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts', 'maintenance'))
    import submit_production as sp
    return sp


# ----------------------------------------------------------------------
# Resource lookups (delegate to submit_production)
# ----------------------------------------------------------------------
def _resources_for(sp, case, td):
    chi_conv = sp.read_chi_conv()
    chi_prod = sp.chi_prod_for(td, chi_conv)
    grid_size = sp.grid_size_for(case, td)

    n_traj = 0
    if 'open' in case:
        n_traj = sp.ntraj_for(case, td)

    mem_gb = sp.compute_mem_gb(chi_prod)
    return chi_prod, grid_size, n_traj, mem_gb


# ----------------------------------------------------------------------
# Task-ID <-> (t_plus, t_minus) mapping
# ----------------------------------------------------------------------
def _task_id_to_coords(task_id, grid_size):
    i = task_id // grid_size
    j = task_id % grid_size
    tp = np.linspace(0.0, np.pi, grid_size)
    tm = np.linspace(0.0, 2.0 * np.pi, grid_size)
    return float(tp[i]), float(tm[j])


# ----------------------------------------------------------------------
# SLURM script
# ----------------------------------------------------------------------
def _write_array_script(case, tuple_idx, chi_prod, grid_size,
                        chunk_start, chunk_end, mem_gb, walltime):
    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_asym_production_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'asym_production')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = (f"asym_{case}_t{tuple_idx:03d}_chi{chi_prod}"
           f"_g{grid_size}_c{chunk_start}")
    script_path = os.path.join(jobdir, f"job_{tag}.sh")
    chunk_size = chunk_end - chunk_start + 1

    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem_gb}G
#SBATCH --time={walltime}
#SBATCH --array=0-{chunk_size - 1}
#SBATCH --output={logdir}/{tag}_%A_%a.log
#SBATCH --error={logdir}/{tag}_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {ROOT_DIR}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/maintenance/submit_asymmetric.py \\
   --case {case} --run-task {tuple_idx} {chi_prod} {grid_size} \\
   $(({chunk_start} + SLURM_ARRAY_TASK_ID))
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


def _chunk_arrays(n_tasks):
    if n_tasks <= 0:
        return
    if n_tasks <= MAX_ARRAY_TASKS:
        yield (0, n_tasks - 1)
        return
    start = 0
    while start < n_tasks:
        end = min(start + MAX_ARRAY_TASKS - 1, n_tasks - 1)
        yield (start, end)
        start = end + 1


def _append_job_id(record):
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    fields = ['timestamp', 'case', 'tuple_idx', 'N', 'Lx', 'Ly',
              'alpha', 'gamma1', 'gamma2', 'chi_prod', 'grid_size',
              'n_traj', 'mem_gb', 'walltime', 'chunk_idx',
              'chunk_start', 'chunk_end', 'job_id', 'script_path']
    write_header = not os.path.exists(JOB_IDS_CSV)
    with open(JOB_IDS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------
def submit(case, dry_run=False):
    sp = _load_phase_tuples()
    tuples = [td for td in sp.PHASE_TUPLES_BY_CASE.get(case, [])
              if td['case'] == case]
    if not tuples:
        print(f"[submit] no {case} tuples in PHASE_TUPLES; nothing to do",
              flush=True)
        return

    print(f"[submit] case={case}: {len(tuples)} tuples", flush=True)
    n_submitted = 0

    for tuple_idx, td in enumerate(tuples):
        try:
            chi_prod, grid_size, n_traj, mem_gb = _resources_for(
                sp, case, td
            )
        except KeyError as exc:
            print(f"[skip] tuple {tuple_idx}: {exc}", flush=True)
            continue

        n_tasks = grid_size * grid_size
        for chunk_idx, (cs, ce) in enumerate(_chunk_arrays(n_tasks)):
            script = _write_array_script(
                case, tuple_idx, chi_prod, grid_size,
                cs, ce, mem_gb, DEFAULT_WALLTIME,
            )
            if dry_run:
                print(f"[dry-run] {case} tuple {tuple_idx} "
                      f"chi={chi_prod} grid={grid_size} "
                      f"chunk {chunk_idx} [{cs}..{ce}]: {script}",
                      flush=True)
                continue

            try:
                result = subprocess.run(
                    ['sbatch', '--parsable', script],
                    capture_output=True, text=True, check=True,
                )
                job_id = result.stdout.strip()
            except subprocess.CalledProcessError as exc:
                print(f"[error] sbatch failed: {exc.stderr.strip()}",
                      flush=True)
                continue

            _append_job_id({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'case': case, 'tuple_idx': tuple_idx,
                'N': td.get('N', ''), 'Lx': td.get('Lx', ''),
                'Ly': td.get('Ly', ''),
                'alpha': td['alpha'],
                'gamma1': td.get('gamma1', 0.0),
                'gamma2': td.get('gamma2', 0.0),
                'chi_prod': chi_prod, 'grid_size': grid_size,
                'n_traj': n_traj, 'mem_gb': mem_gb,
                'walltime': DEFAULT_WALLTIME,
                'chunk_idx': chunk_idx,
                'chunk_start': cs, 'chunk_end': ce,
                'job_id': job_id, 'script_path': script,
            })
            print(f"[submit] {case} tuple {tuple_idx} chunk {chunk_idx} "
                  f"-> {job_id}", flush=True)
            n_submitted += 1

    print(f"[submit] done: {n_submitted} arrays submitted", flush=True)


# ----------------------------------------------------------------------
# Task dispatcher
# ----------------------------------------------------------------------
def run_task(case, tuple_idx, chi_prod, grid_size, task_id):
    sp = _load_phase_tuples()
    tuples = [td for td in sp.PHASE_TUPLES_BY_CASE.get(case, [])
              if td['case'] == case]
    if not (0 <= tuple_idx < len(tuples)):
        raise IndexError(f"tuple_idx={tuple_idx} out of range "
                         f"(case has {len(tuples)} tuples)")
    td = tuples[tuple_idx]

    t_plus, t_minus = _task_id_to_coords(task_id, grid_size)

    n_traj = 0
    if 'open' in case:
        n_traj = sp.ntraj_for(case, td)

    ASYM_SUBDIRS = {'1d_closed': 'closed_1d', '1d_open': 'open_1d',
                    '2d_closed': 'closed_2d', '2d_open': 'open_2d'}
    worker = os.path.join(ROOT_DIR, 'scripts', 'workers',
                          ASYM_SUBDIRS[case], ASYM_WORKERS[case])
    cmd = [sys.executable, worker,
           '--alpha', str(td['alpha']),
           '--t_plus', f'{t_plus:.6f}',
           '--t_minus', f'{t_minus:.6f}',
           '--chi_max', str(chi_prod),
           '--task_id', str(task_id)]

    if case == '1d_closed':
        cmd += ['--N', str(td['N']), '--J', str(td.get('J', 1.0))]
    elif case == '2d_closed':
        cmd += ['--Lx', str(td['Lx']), '--Ly', str(td['Ly']),
                '--J', str(td.get('J', 1.0))]
    else:
        cmd += ['--gamma1', str(td.get('gamma1', 0.0)),
                '--gamma2', str(td.get('gamma2', 0.0)),
                '--n_traj', str(n_traj), '--seed', '42']
        if case == '1d_open':
            cmd += ['--N', str(td['N']), '--J', str(td.get('J', 1.0))]
        else:
            cmd += ['--Lx', str(td['Lx']), '--Ly', str(td['Ly']),
                    '--J', str(td.get('J', 1.0))]

    print(f"[run-task] case={case} tuple={tuple_idx} "
          f"chi={chi_prod} grid={grid_size} task={task_id} -> "
          f"t_plus={t_plus:.4f} t_minus={t_minus:.4f}", flush=True)
    print(f"[run-task] cmd: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True,
                        choices=list(ASYM_WORKERS.keys()))
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-task', nargs=4, default=None,
                        metavar=('TUPLE_IDX', 'CHI', 'GRID', 'TASK_ID'))
    args = parser.parse_args()

    if args.run_task is not None:
        tuple_idx, chi, grid, task_id = map(int, args.run_task)
        run_task(args.case, tuple_idx, chi, grid, task_id)
        return

    if args.submit:
        submit(args.case, dry_run=args.dry_run)
        return

    parser.print_help()


if __name__ == '__main__':
    main()