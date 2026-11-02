#!/usr/bin/env python3
"""
scripts/maintenance/submit_scharnagl_stage1.py
Stage 1 Scharnagl three-axis scan driver.

Convention: mu1, mu2 (Vineesha). Full range by default:
  mu1 in [0, pi], mu2 in [0, 2 pi], theta_k in [0, pi], phi_k in [0, 2 pi).

Range control
-------------
--ghz-window K (default: off = full range)
  Restrict mu1 to [2*(pi - K*4/sqrt(N))/N, 2*pi/N] (the GHZ
  working window with K-sigma margin). mu2 stays full range.
  Used for production at large N once the codebase has validated
  that the peak is inside the K=2 window.

Grid
----
10 mu1 x 10 mu2 x 5 theta_k x 4 phi_k = 2000 tasks per (case, N, alpha),
split into 2 SLURM arrays of 1000 tasks each.
Task ID decoding is row-major: ((mu1_idx * 10 + mu2_idx) * 5 + theta_idx) * 4 + phi_idx.

Reads PRODUCTION_TUPLES from submit_production.PHASE_TUPLES_BY_CASE.
Filters by --cases / --N / --alpha.

Usage
-----
  python3 scripts/maintenance/submit_scharnagl_stage1.py \
      --submit [--dry-run] \
      [--cases 1d_closed 2d_closed] \
      [--N 16 32] \
      [--alpha 3.0 6.0] \
      [--ghz-window 2]

  # Internal
  python3 scripts/maintenance/submit_scharnagl_stage1.py \
      --run-task CASE TUPLE_IDX CHUNK TASK_ID

  # Sanity (offline, no cluster)
  python3 scripts/maintenance/submit_scharnagl_stage1.py --sanity
"""
import argparse
import os
import subprocess
import sys
import numpy as np

REPO = os.environ.get('METROLOGY_REPO',
                      os.path.expanduser('~/metrology_thesis'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

JOB_DIR = os.path.join(REPO, 'scripts/maintenance/_scharnagl_jobs')
LOG_DIR = os.path.join(REPO, 'logs/scharnagl_stage1')


# ----------------------------------------------------------------------
# Convention and grid configuration
# ----------------------------------------------------------------------
CONVENTION = "mu1mu2"

THETA_VALS = [i * np.pi / 4.0 for i in range(5)]           # 0, pi/4, ..., pi
PHI_VALS   = [i * np.pi / 2.0 for i in range(4)]           # 0, pi/2, pi, 3pi/2

GRID_SIZE = 10
MU1_VALS = np.linspace(0.0, np.pi, GRID_SIZE)
MU2_VALS = np.linspace(0.0, 2.0 * np.pi, GRID_SIZE)


def _mu_grids(N, ghz_window=None):
    """Return (MU1_VALS, MU2_VALS) for the given N (or N = Lx*Ly).

    Full range: mu1 in linspace(0, pi, GRID_SIZE),
                mu2 in linspace(0, 2*pi, GRID_SIZE).

    --ghz-window K:
      mu1 in linspace(2*(pi - K*4/sqrt(N))/N, 2*pi/N, GRID_SIZE)
      mu2 in linspace(0, 2*pi, GRID_SIZE)
    """
    if ghz_window is None:
        return MU1_VALS.copy(), MU2_VALS.copy()
    lo = 2.0 * (np.pi - ghz_window * 4.0 / np.sqrt(N)) / N
    hi = 2.0 * np.pi / N
    if lo >= hi:
        raise ValueError(
            f"--ghz-window {ghz_window} gives empty window at N={N}: "
            f"mu1 in [{lo:.4f}, {hi:.4f}]"
        )
    mu1 = np.linspace(lo, hi, GRID_SIZE)
    return mu1, MU2_VALS.copy()


def chi_for(N, alpha):
    """Chi based on CHI_FLOOR_POLICY, delegated to submit_production."""
    try:
        sys.path.insert(0, os.path.join(REPO, 'scripts', 'maintenance'))
        import submit_production as sp
        td = {'case': '1d_closed', 'N': N, 'alpha': alpha}
        if N == 24:  # (Lx, Ly) = (4, 6) case
            td = {'case': '2d_closed', 'Lx': 4, 'Ly': 6, 'alpha': alpha}
        return sp.chi_prod_for(td, sp.read_chi_conv())
    except Exception:
        return 32 if N <= 16 else 64


# ----------------------------------------------------------------------
# Tuple filtering
# ----------------------------------------------------------------------
def _load_tuples(cases, N_filter, alpha_filter):
    """Return list of (case, N_or_LxLy, alpha) tuples to process."""
    sys.path.insert(0, os.path.join(REPO, 'scripts', 'maintenance'))
    import submit_production as sp

    all_tuples = []
    for case in ('1d_closed', '1d_open', '2d_closed', '2d_open'):
        if cases and case not in cases:
            continue
        for td in sp.PHASE_TUPLES_BY_CASE.get(case, []):
            if td['case'] != case:
                continue
            if alpha_filter and round(float(td['alpha']), 6) not in [
                    round(a, 6) for a in alpha_filter]:
                continue
            if 'N' in td:
                if N_filter and int(td['N']) not in N_filter:
                    continue
                geometry = int(td['N'])
            else:
                # 2D: filter on Lx * Ly == requested N if N_filter given
                if N_filter:
                    n_sites = int(td['Lx']) * int(td['Ly'])
                    if n_sites not in N_filter:
                        continue
                geometry = (int(td['Lx']), int(td['Ly']))
            all_tuples.append((case, geometry, td, float(td['alpha'])))
    return all_tuples


# ----------------------------------------------------------------------
# Task ID <-> coordinates
# ----------------------------------------------------------------------
def _decode_task_id(task_id, N, ghz_window):
    """task_id -> (mu1_idx, mu2_idx, theta_idx, phi_idx)."""
    idx = task_id
    phi_idx = idx % len(PHI_VALS); idx //= len(PHI_VALS)
    theta_idx = idx % len(THETA_VALS); idx //= len(THETA_VALS)
    mu2_idx = idx % GRID_SIZE; idx //= GRID_SIZE
    mu1_idx = idx
    if mu1_idx >= GRID_SIZE:
        raise IndexError(f"task_id {task_id} exceeds "
                         f"{GRID_SIZE**2 * len(THETA_VALS) * len(PHI_VALS)}")
    mu1_vals, mu2_vals = _mu_grids(N, ghz_window)
    return (mu1_idx, mu2_idx, theta_idx, phi_idx,
            float(mu1_vals[mu1_idx]),
            float(mu2_vals[mu2_idx]),
            float(THETA_VALS[theta_idx]),
            float(PHI_VALS[phi_idx]))


# ----------------------------------------------------------------------
# Output path and worker CLI
# ----------------------------------------------------------------------
SCH_SUBDIRS = {'1d_closed': 'closed_1d', '1d_open': 'open_1d',
               '2d_closed': 'closed_2d', '2d_open': 'open_2d'}


def _worker_script(case):
    subdir = SCH_SUBDIRS.get(case)
    if subdir is None:
        raise ValueError(f'Unknown case: {case!r}')
    return f'scripts/workers/{subdir}/scharnagl_echo_{case}.py'


def _output_path(case, geometry, alpha, mu1, mu2, theta, phi,
                 ghz_window):
    suffix = '_ghz' if ghz_window is not None else ''
    tag = (f'scharnagl_pt_N{geometry}_alpha{alpha}'
           f'_mu1{mu1:.4f}_mu2{mu2:.4f}'
           f'_tk{theta:.4f}_pk{phi:.4f}{suffix}.npy'
           if not isinstance(geometry, tuple) else
           f'scharnagl_pt_Lx{geometry[0]}_Ly{geometry[1]}_alpha{alpha}'
           f'_mu1{mu1:.4f}_mu2{mu2:.4f}'
           f'_tk{theta:.4f}_pk{phi:.4f}{suffix}.npy')
    return os.path.join(REPO, f'data/scharnagl_pts_{case}', tag)


def _build_worker_cmd(case, geometry, td, alpha, chi, mu1, mu2,
                      theta, phi, task_id):
    cmd = [sys.executable, _worker_script(case),
           '--alpha', str(alpha),
           '--J', str(td.get('J', 1.0)),
           '--mu1', f'{mu1:.6f}',
           '--mu2', f'{mu2:.6f}',
           '--theta_k', f'{theta:.6f}',
           '--phi_k', f'{phi:.6f}',
           '--chi_max', str(chi),
           '--dt_max', '0.05',
           '--method', 'exact_derivative']
    if isinstance(geometry, tuple):
        cmd += ['--Lx', str(geometry[0]), '--Ly', str(geometry[1])]
    else:
        cmd += ['--N', str(geometry)]
    if case.endswith('_open'):
        n_traj = td.get('n_traj', 100)
        cmd += ['--gamma1', str(td.get('gamma1', 0.0)),
                '--gamma2', str(td.get('gamma2', 0.0)),
                '--n_traj', str(n_traj),
                '--seed', '42']
    return cmd


# ----------------------------------------------------------------------
# SLURM script generation
# ----------------------------------------------------------------------
def _n_tasks():
    return GRID_SIZE * GRID_SIZE * len(THETA_VALS) * len(PHI_VALS)


def _write_array_script(case, tuple_tag, geometry, alpha,
                        chi, chunk_start, chunk_end, mem_gb,
                        walltime, ghz_window):
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = f'_ghz{ghz_window}' if ghz_window is not None else ''
    tag = (f'sch_{case}_{tuple_tag}_a{alpha}{suffix}'
           f'_c{chunk_start}')
    path = os.path.join(JOB_DIR, f'job_{tag}.sh')
    chunk_size = chunk_end - chunk_start + 1
    ghz_arg = '' if ghz_window is None else f'--ghz-window {ghz_window}'
    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem_gb}G
#SBATCH --time={walltime}
#SBATCH --array=0-{chunk_size - 1}
#SBATCH --output={LOG_DIR}/{tag}_%A_%a.log
#SBATCH --error={LOG_DIR}/{tag}_%A_%a.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {REPO}
python scripts/maintenance/submit_scharnagl_stage1.py \\
   --run-task {case} {tuple_tag} {chunk_start} \\
   $(({chunk_start} + SLURM_ARRAY_TASK_ID)) \\
   {ghz_arg}
"""
    with open(path, 'w') as f:
        f.write(body)
    os.chmod(path, 0o755)
    return path


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------
def submit_all(dry_run=False, cases=None, N_filter=None,
               alpha_filter=None, ghz_window=None):
    tuples = _load_tuples(cases, N_filter, alpha_filter)
    if not tuples:
        print(f"[submit] no tuples match filter "
              f"(cases={cases}, N={N_filter}, alpha={alpha_filter})",
              flush=True)
        return

    print(f"[submit] {len(tuples)} (case, geometry, alpha) tuples",
          flush=True)
    total_tasks = 0
    for tuple_idx, (case, geometry, td, alpha) in enumerate(tuples):
        N_int = (int(geometry) if not isinstance(geometry, tuple)
                 else int(geometry[0]) * int(geometry[1]))
        chi = chi_for(N_int, alpha)
        mem_gb = min(64 + 4 * chi, 224)
        n = _n_tasks()
        # Split into 1000-task chunks
        chunks = []
        start = 0
        while start < n:
            end = min(start + 999, n - 1)
            chunks.append((start, end))
            start = end + 1

        tuple_tag = (f't{tuple_idx:03d}')
        for chunk_idx, (cs, ce) in enumerate(chunks):
            script = _write_array_script(
                case, tuple_tag, geometry, alpha, chi,
                cs, ce, mem_gb, '14-00:00:00', ghz_window,
            )
            chunk_size = ce - cs + 1
            total_tasks += chunk_size
            if dry_run:
                print(f"[dry-run] {case} {geometry} a={alpha} "
                      f"chi={chi} chunk {chunk_idx} "
                      f"[{cs}..{ce}] {chunk_size} tasks: {script}",
                      flush=True)
                continue
            try:
                r = subprocess.run(['sbatch', '--parsable', script],
                                   capture_output=True, text=True,
                                   check=True)
                print(f"[submit] {case} {geometry} a={alpha} "
                      f"chunk {chunk_idx} -> {r.stdout.strip()}",
                      flush=True)
            except subprocess.CalledProcessError as exc:
                print(f"[error] {case} {geometry} a={alpha} "
                      f"chunk {chunk_idx}: {exc.stderr.strip()}",
                      flush=True)

    print(f"[submit] total: {total_tasks} tasks across "
          f"{len(tuples)} (case, geometry, alpha) tuples", flush=True)


# ----------------------------------------------------------------------
# Task dispatcher
# ----------------------------------------------------------------------
def run_task(case, tuple_tag, chunk_start, task_id,
             ghz_window=None):
    tuples = _load_tuples([case], None, None)
    tuple_idx = int(tuple_tag[1:])
    if tuple_idx >= len(tuples):
        raise IndexError(f"tuple_idx {tuple_idx} out of range")

    _, geometry, td, alpha = tuples[tuple_idx]
    N_int = (int(geometry) if not isinstance(geometry, tuple)
             else int(geometry[0]) * int(geometry[1]))

    if task_id >= _n_tasks():
        raise IndexError(f"task_id {task_id} out of range")

    (mu1_idx, mu2_idx, theta_idx, phi_idx,
     mu1, mu2, theta, phi) = _decode_task_id(
        task_id, N_int, ghz_window,
    )
    out = _output_path(case, geometry, alpha,
                       mu1, mu2, theta, phi, ghz_window)
    if os.path.exists(out):
        print(f'[skip] exists: {out}', flush=True)
        return

    chi = chi_for(N_int, alpha)
    cmd = _build_worker_cmd(case, geometry, td, alpha, chi,
                            mu1, mu2, theta, phi, task_id)
    print(f'[run] task_id={task_id} -> mu1={mu1:.4f} mu2={mu2:.4f} '
          f'theta={theta:.4f} phi={phi:.4f}', flush=True)
    print(f'       {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


# ----------------------------------------------------------------------
# Sanity (offline)
# ----------------------------------------------------------------------
def sanity():
    print('=' * 70)
    print('Sanity: Scharnagl Stage-1 driver (mu1mu2 convention)')
    print('=' * 70)

    # 1. Full-range grids
    for N in (4, 8, 16, 200):
        mu1, mu2 = _mu_grids(N, None)
        assert mu1[0] == 0.0 and abs(mu1[-1] - np.pi) < 1e-9, \
            f'N={N}: mu1 endpoints [{mu1[0]}, {mu1[-1]}]'
        assert mu2[0] == 0.0 and abs(mu2[-1] - 2 * np.pi) < 1e-9, \
            f'N={N}: mu2 endpoints [{mu2[0]}, {mu2[-1]}]'
    print(f'(a) full-range grids: PASS  '
          f'(mu1 [0, pi], mu2 [0, 2pi], all N)')

    # 2. GHZ-window grids
    for N in (16, 32, 200):
        mu1, mu2 = _mu_grids(N, ghz_window=2)
        lo = 2 * (np.pi - 2 * 4.0 / np.sqrt(N)) / N
        hi = 2 * np.pi / N
        assert abs(mu1[0] - lo) < 1e-9, f'N={N}: mu1[0]={mu1[0]} vs {lo}'
        assert abs(mu1[-1] - hi) < 1e-9, f'N={N}: mu1[-1]={mu1[-1]} vs {hi}'
    print(f'(b) --ghz-window 2 grids: PASS  '
          f'(mu1 in [lo, hi], full mu2)')

    # 3. Task ID round-trip
    N = 16
    for task_id in (0, 1999, 1000):
        (i1, i2, it, ip, mu1, mu2, theta, phi) = _decode_task_id(
            task_id, N, None,
        )
        assert 0 <= i1 < GRID_SIZE
        assert 0 <= i2 < GRID_SIZE
        assert 0 <= it < len(THETA_VALS)
        assert 0 <= ip < len(PHI_VALS)
        assert 0 <= mu1 <= np.pi + 1e-9
        assert 0 <= mu2 <= 2 * np.pi + 1e-9
    # round-trip: re-encode and match
    assert _decode_task_id(0, N, None)[:4] == (0, 0, 0, 0)
    assert _decode_task_id(1999, N, None)[:4] == (9, 9, 4, 3)
    print(f'(c) task_id round-trip 0..1999: PASS')

    print('\n[PASS] sanity check')
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--cases', nargs='+',
                        choices=['1d_closed', '1d_open',
                                 '2d_closed', '2d_open'],
                        default=None)
    parser.add_argument('--N', nargs='+', type=int, default=None,
                        help='Filter on N (or Lx*Ly for 2D)')
    parser.add_argument('--alpha', nargs='+', type=float, default=None)
    parser.add_argument('--ghz-window', type=float, default=None,
                        help='GHZ window margin (default: full range)')
    parser.add_argument('--run-task', nargs=4, default=None,
                        metavar=('CASE', 'TUPLE_TAG', 'CHUNK_START',
                                 'TASK_ID'))
    args = parser.parse_args()

    if args.sanity:
        sys.exit(0 if sanity() else 1)

    if args.run_task is not None:
        case, tuple_tag, chunk_start, task_id = args.run_task
        run_task(case, tuple_tag, int(chunk_start), int(task_id),
                 ghz_window=args.ghz_window)
        return

    if args.submit:
        submit_all(dry_run=args.dry_run, cases=args.cases,
                   N_filter=args.N, alpha_filter=args.alpha,
                   ghz_window=args.ghz_window)
        return

    parser.print_help()


if __name__ == '__main__':
    main()