#!/usr/bin/env python3
"""
scripts/maintenance/run_asym_batch.py
Single-node driver for the documentation(c) asymmetric echo, 1D closed.

ITP relaunch safety (see handoff 3 Section 9):
  - --n_jobs capped at HARD_CAP_N_JOBS = 16 (default 8)
  - PYTHONPATH points at /scratch-local/c705144/pylibs
  - Process reniced to nice=19
  - Lock file prevents two instances on same node
  - Outputs staged on /scratch-local, rsynced to /scratch on completion
  - Use --no-staging to disable staging, --force-unlock to override a
    stale lock

Usage:
    python3 scripts/maintenance/run_asym_batch.py START END --n_jobs 8
    python3 scripts/maintenance/run_asym_batch.py START END --n_jobs 8 --dry-run
"""
import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


GRID = 40
DT_MAX = 0.05
METHOD = 'exact_derivative'
N_VALUES = [16, 32, 64, 128, 160, 200]
ALPHA_VALUES = [1.0, 2.0, 3.0, 4.0, 6.0]
TP_VALS = np.linspace(0.0, np.pi, GRID)
TM_VALS = np.linspace(0.0, 2.0 * np.pi, GRID)
REPO = '/scratch/c705144/metrology_thesis'
OUT_DIR = f'{REPO}/data/echo_pts_1d_closed_asym'
LOG_DIR = f'{REPO}/logs'

# ITP relaunch safety constants (see handoff 3 Section 9).
LOCAL_STAGING_DIR = '/scratch-local/c705144/asym_staging'
PYLIBS_PATH = '/scratch-local/c705144/pylibs'
NICE_LEVEL = 19
LOCK_FILE = '/scratch-local/c705144/asym_batch.lock'
HARD_CAP_N_JOBS = 16


def chi_for(N, alpha):
    if N == 16 and alpha in (1.0, 3.0, 6.0):
        return 32
    if N == 64 and alpha == 6.0:
        return 32
    if N == 200:
        return 128
    return 64


def run_one(gid):
    stride = GRID * GRID
    tuple_idx = gid // stride
    point_idx = gid % stride
    tp_idx = point_idx // GRID
    tm_idx = point_idx % GRID
    n_idx = tuple_idx // len(ALPHA_VALUES)
    a_idx = tuple_idx % len(ALPHA_VALUES)
    N = N_VALUES[n_idx]
    alpha = ALPHA_VALUES[a_idx]
    chi = chi_for(N, alpha)
    tp = float(TP_VALS[tp_idx])
    tm = float(TM_VALS[tm_idx])

    out = f'{OUT_DIR}/asym_pt_{gid}.npy'
    # Note: if staging is enabled, OUT_DIR is a symlink to LOCAL_STAGING_DIR,
    # so this existence check resolves through the symlink. Good.
    if os.path.exists(out):
        return (gid, 'skip', 0)

    cmd = [
        sys.executable, 'scripts/workers/closed_1d/asymmetric_echo_1d_closed.py',
        '--N', str(N), '--alpha', str(alpha),
        '--t_plus', f'{tp:.6f}', '--t_minus', f'{tm:.6f}',
        '--chi_max', str(chi), '--dt_max', str(DT_MAX),
        '--method', METHOD, '--task_id', str(gid),
    ]
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    # Local package cache (avoids NFS cold-start storms).
    env['PYTHONPATH'] = PYLIBS_PATH + ':' + env.get('PYTHONPATH', '')

    log_path = f'{LOG_DIR}/asym_gid{gid}.log'
    try:
        with open(log_path, 'w') as f:
            r = subprocess.run(cmd, cwd=REPO, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
        return (gid, 'ok' if r.returncode == 0 else 'fail', r.returncode)
    except Exception as exc:
        with open(log_path, 'a') as f:
            f.write(f'\n[driver exception] {exc}\n')
        return (gid, 'fail', -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('start', type=int, help='start gid (inclusive)')
    p.add_argument('end', type=int, help='end gid (exclusive)')
    p.add_argument('--n_jobs', type=int, default=8,
                   help='Parallel workers on this node. Safe defaults: '
                        '8 for 16-core nodes (graviton, dopey, fermion, '
                        'grunt, noise, qbit, quantum), 16 for 64-core nodes '
                        '(yedi, vortex, panic). Capped at HARD_CAP_N_JOBS.')
    p.add_argument('--no-staging', action='store_true',
                   help='Write directly to /scratch instead of '
                        '/scratch-local staging. Only on low-contention runs.')
    p.add_argument('--force-unlock', action='store_true',
                   help='Remove a stale lock file and proceed.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print config and exit without running workers.')
    args = p.parse_args()

    # Cap n_jobs at HARD_CAP_N_JOBS.
    if args.n_jobs > HARD_CAP_N_JOBS:
        print(f'[cap] --n_jobs {args.n_jobs} exceeds safety ceiling '
              f'{HARD_CAP_N_JOBS}; capping to {HARD_CAP_N_JOBS}.',
              flush=True)
        args.n_jobs = HARD_CAP_N_JOBS

    if args.dry_run:
        print(f'[dry-run] start={args.start} end={args.end} '
              f'n_jobs={args.n_jobs} '
              f'staging={"off" if args.no_staging else "on"}',
              flush=True)
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # -- 1. Lock file (prevents two instances on same node).
    try:
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)
    except FileExistsError:
        if args.force_unlock:
            print(f'[lock] --force-unlock given, removing stale lock.',
                  flush=True)
            try:
                os.remove(LOCK_FILE)
            except OSError:
                pass
            lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(lock_fd, str(os.getpid()).encode())
            os.close(lock_fd)
        else:
            try:
                with open(LOCK_FILE, 'r') as f:
                    existing_pid = f.read().strip()
            except Exception:
                existing_pid = '?'
            print(f'[abort] Lock file exists at {LOCK_FILE} '
                  f'(pid={existing_pid}). Another run_asym_batch.py '
                  f'is on this node. Use --force-unlock if stale.',
                  flush=True)
            return 1

    # -- 2. Nice (good cluster citizen).
    try:
        os.nice(NICE_LEVEL)
        print(f'[nice] process priority set to nice={NICE_LEVEL}',
              flush=True)
    except Exception as exc:
        print(f'[nice] WARNING: could not set nice={NICE_LEVEL}: {exc}',
              flush=True)

    # -- 3. Staging setup.
    staging_dir = None
    if args.no_staging:
        print(f'[staging] disabled, writing directly to {OUT_DIR}',
              flush=True)
    else:
        staging_dir = LOCAL_STAGING_DIR
        os.makedirs(staging_dir, exist_ok=True)
        # Backup any existing real OUT_DIR.
        if os.path.isdir(OUT_DIR) and not os.path.islink(OUT_DIR):
            backup = OUT_DIR + f'.bak.{int(time.time())}'
            os.rename(OUT_DIR, backup)
            print(f'[staging] backed up {OUT_DIR} -> {backup}',
                  flush=True)
        # Remove any stale symlink.
        if os.path.islink(OUT_DIR):
            os.remove(OUT_DIR)
        # Create symlink OUT_DIR -> staging_dir.
        os.symlink(staging_dir, OUT_DIR)
        print(f'[staging] symlinked {OUT_DIR} -> {staging_dir}',
              flush=True)

    gids = list(range(args.start, args.end))
    print(f'[start] gids {args.start}..{args.end - 1} '
          f'({len(gids)} tasks, {args.n_jobs} parallel, '
          f'staging={"off" if args.no_staging else "on"})', flush=True)

    n_ok = n_skip = n_fail = 0
    try:
        with cf.ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            for i, (gid, status, rc) in enumerate(
                    pool.map(run_one, gids, chunksize=1), start=1):
                if status == 'ok':
                    n_ok += 1
                elif status == 'skip':
                    n_skip += 1
                else:
                    n_fail += 1
                if i % 50 == 0 or i == len(gids):
                    print(f'[progress] {i}/{len(gids)} '
                          f'ok={n_ok} skip={n_skip} fail={n_fail}',
                          flush=True)
    finally:
        # -- 4. Restore real OUT_DIR and rsync staging -> OUT_DIR.
        if staging_dir is not None:
            if os.path.islink(OUT_DIR):
                os.remove(OUT_DIR)
            os.makedirs(OUT_DIR, exist_ok=True)
            print(f'[rsync] {staging_dir} -> {OUT_DIR}', flush=True)
            subprocess.run(
                ['rsync', '-a', staging_dir + '/', OUT_DIR + '/'],
                check=False,
            )
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

    print(f'[done] ok={n_ok} skip={n_skip} fail={n_fail}', flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())