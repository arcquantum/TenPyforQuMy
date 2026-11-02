"""
scripts/maintenance/auto_chi_pilot.py

Automatic bond-dimension selector for the production runs of
Phases 1-4 (v2: entanglement-guided pilot grid).

Modes
-----
  --submit              Two-stage submission per tuple:
                          Stage A (single task): coarse 20x20 scan at
                              chi=8, identifies the top-k highest-
                              entanglement points, writes the union
                              pilot grid to data/resource_tables/
                              pilot_grids/.
                          Stage B (six-chi array, afterok Stage A):
                              runs the worker on the union grid.

  --aggregate           Read the pilot output, apply the convergence
                        criterion, verify chi_conv on the coarse grid,
                        and write data/resource_tables/chi_conv.csv.
                        Tuples whose coarse-grid verification fails are
                        appended to chi_conv_flags.csv.

  --skip-verify         With --aggregate, skip the coarse-grid
                        verification step.

  --sanity              Run the sanity check against existing data.

  --run-stage-a IDX     Internal: run Stage A for tuple IDX.
  --run-tuple IDX       Internal: run one (tuple, chi) array task.
  --run-chi-idx J       With --run-tuple, identifies which chi value
                        the current array task handles.

Production tuple list
---------------------
PRODUCTION_TUPLES is a representative subset: 19 anchor tuples across
the four cases. Extending the list is a one-line edit.
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
    PILOT_GRID, PILOT_GRID_MU1MU2, CHI_SCHEDULE, TARGET_CLOSED, TARGET_OPEN,
    N_TRAJ_PILOT, N_TRAJ_SCAN, SEED_PILOT,
    ENTANGLEMENT_SCAN_CHI, ENTANGLEMENT_SCAN_GRID,
    ENTANGLEMENT_SCAN_TOP_K, COARSE_VERIFY_TOL,
    worker_command, native_output_path, snr_from_output,
    snr_by_chi, find_chi_convergence, chi_production_from_conv,
    build_pilot_grid, write_pilot_grid, read_pilot_grid,
    verify_chi_on_coarse_grid,
)

RESOURCE_DIR = os.path.join(ROOT_DIR, 'data', 'resource_tables')

# Convention used for all pilot runs (Stage 1 migration).
CONVENTION = "mu1mu2"


# ----------------------------------------------------------------------
# Production tuple list
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Resource assignment
# ----------------------------------------------------------------------
def _resources(case, N=None, Lx=None, Ly=None, alpha=None):
    if case == '1d_closed':
        mem = 128 if (N is not None and N >= 128) else 32
        return mem, '08:00:00'
    if case == '2d_closed':
        n_sites = (Lx or 0) * (Ly or 0)
        mem = 224 if n_sites >= 48 else 128
        return mem, '08:00:00'
    if case == '1d_open':
        return 32, '12:00:00'
    if case == '2d_open':
        return 128, '12:00:00'
    raise ValueError(f"Unknown case: {case!r}")


# ----------------------------------------------------------------------
# SLURM scripts
# ----------------------------------------------------------------------
def _write_stage_a_script(tuple_idx, td):
    """SLURM script for the coarse entanglement scan (one task)."""
    case = td['case']
    mem, _ = _resources(case, N=td.get('N'),
                        Lx=td.get('Lx'), Ly=td.get('Ly'),
                        alpha=td['alpha'])

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_chi_pilot_jobs')
    os.makedirs(jobdir, exist_ok=True)
    logdir = os.path.join(ROOT_DIR, 'logs', 'auto_chi_pilot')
    os.makedirs(logdir, exist_ok=True)

    tag = f"chiA_{case}_t{tuple_idx:02d}"
    script_path = os.path.join(jobdir, f"job_{tag}.sh")

    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}G
#SBATCH --time=12:00:00
#SBATCH --output={logdir}/{tag}_%j.log
#SBATCH --error={logdir}/{tag}_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {ROOT_DIR}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/maintenance/auto_chi_pilot.py --run-stage-a {tuple_idx}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


def _write_stage_b_script(tuple_idx, td, dependency_jobid=None):
    """SLURM array script for the six-chi pilot on the union grid."""
    case = td['case']
    mem, walltime = _resources(case, N=td.get('N'),
                               Lx=td.get('Lx'), Ly=td.get('Ly'),
                               alpha=td['alpha'])

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_chi_pilot_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'auto_chi_pilot')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = f"chi_{case}_t{tuple_idx:02d}"
    script_path = os.path.join(jobdir, f"job_{tag}.sh")

    n_tasks = len(CHI_SCHEDULE)
    dep_line = (f"#SBATCH --dependency=afterok:{dependency_jobid}\n"
                if dependency_jobid is not None else "")

    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}G
#SBATCH --time={walltime}
#SBATCH --output={logdir}/{tag}_%A_%a.log
#SBATCH --error={logdir}/{tag}_%A_%a.err
#SBATCH --array=0-{n_tasks - 1}
{dep_line}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {ROOT_DIR}

python scripts/maintenance/auto_chi_pilot.py \\
    --run-tuple {tuple_idx} \\
    --run-chi-idx ${{SLURM_ARRAY_TASK_ID}}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------
def submit_all(dry_run=False, cases=None):
    """
    Two-stage submission per tuple:
      1. sbatch Stage A; capture the job ID.
      2. sbatch Stage B with --dependency=afterok:<Stage A job ID>.

    If ``cases`` is a list of strings, only tuples whose 'case' field
    is in the list are submitted.
    """
    for idx, td in enumerate(PRODUCTION_TUPLES):
        if cases is not None and td['case'] not in cases:
            continue
        stage_a = _write_stage_a_script(idx, td)

        if dry_run:
            print(f"[dry-run] stage A: {stage_a}")
            stage_b = _write_stage_b_script(idx, td)
            print(f"[dry-run] stage B (no dep): {stage_b}")
            continue

        result = subprocess.run(
            ['sbatch', '--parsable', stage_a],
            capture_output=True, text=True, check=True,
        )
        stage_a_jobid = result.stdout.strip()
        print(f"[submit] stage A job {stage_a_jobid}: {stage_a}")

        stage_b = _write_stage_b_script(idx, td, dependency_jobid=stage_a_jobid)
        subprocess.run(['sbatch', stage_b], check=True)
        print(f"[submit] stage B array (depends on {stage_a_jobid}): "
              f"{stage_b}")


# ----------------------------------------------------------------------
# Stage A: coarse entanglement scan
# ----------------------------------------------------------------------
def run_stage_a(tuple_idx):
    """
    Run the coarse entanglement scan for one tuple, then write the
    union grid to disk.
    """
    td = PRODUCTION_TUPLES[tuple_idx]
    case = td['case']
    gamma1 = td.get('gamma1', 0.0)
    gamma2 = td.get('gamma2', 0.0)

    print(f"[stage A] tuple_idx={tuple_idx}, case={case}, "
          f"alpha={td['alpha']}", flush=True)

    params = {
        'N': td.get('N'),
        'Lx': td.get('Lx'),
        'Ly': td.get('Ly'),
        'alpha': td['alpha'],
        'gamma1': gamma1,
        'gamma2': gamma2,
    }

    top = build_pilot_grid(case, params, convention=CONVENTION)

    grid_file = write_pilot_grid(
        case, top,
        N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
        alpha=td['alpha'], gamma1=gamma1, gamma2=gamma2,
        convention=CONVENTION,
    )
    print(f"[stage A] union grid ({len(top)} points) -> {grid_file}",
          flush=True)
    for pt in top:
        print(f"    ({pt[0]:.4f}, {pt[1]:.4f})", flush=True)


# ----------------------------------------------------------------------
# Stage B: six-chi array task
# ----------------------------------------------------------------------
def run_one_task(tuple_idx, chi_idx):
    """
    Run the pilot for one (tuple, chi) pair on the union grid.
    """
    td = PRODUCTION_TUPLES[tuple_idx]
    case = td['case']
    chi = CHI_SCHEDULE[chi_idx]
    gamma1 = td.get('gamma1', 0.0)
    gamma2 = td.get('gamma2', 0.0)

    union_grid = read_pilot_grid(
        case, N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
        alpha=td['alpha'], gamma1=gamma1, gamma2=gamma2,
        convention=CONVENTION,
    )

    print(f"[stage B] tuple_idx={tuple_idx}, chi_idx={chi_idx}, "
          f"chi={chi}, case={case}, grid_size={len(union_grid)}",
          flush=True)

    for (mu1, mu2) in union_grid:
        out_path = native_output_path(
            case, chi,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=gamma1, gamma2=gamma2,
            convention=CONVENTION, mu1=mu1, mu2=mu2,
        )
        if os.path.exists(out_path):
            print(f"[skip] {out_path}", flush=True)
            continue

        cmd = worker_command(
            case, chi_max=chi,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=gamma1, gamma2=gamma2,
            convention=CONVENTION, mu1=mu1, mu2=mu2,
        )
        print(f"[run] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    print(f"[stage B done] tuple_idx={tuple_idx}, chi={chi}", flush=True)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def aggregate(skip_verify=False):
    """
    Read the pilot output, apply the convergence criterion, and write
    chi_conv.csv. If skip_verify is False, also call
    verify_chi_on_coarse_grid and write chi_conv_flags.csv for any
    tuples whose verification fails.
    """
    output_csv = os.path.join(RESOURCE_DIR, 'chi_conv.csv')
    flags_csv = os.path.join(RESOURCE_DIR, 'chi_conv_flags.csv')
    os.makedirs(RESOURCE_DIR, exist_ok=True)

    fieldnames = ['case', 'N', 'Lx', 'Ly', 'alpha',
                  'gamma1', 'gamma2', 'protocol',
                  'chi_conv', 'chi_prod',
                  'pilot_grid_size',
                  'coarse_verify_residual',
                  'coarse_verify_points',
                  'max_residual']
    flag_fields = ['case', 'N', 'Lx', 'Ly', 'alpha',
                   'gamma1', 'gamma2', 'mu1', 'mu2',
                   'snr_diff', 'round']

    rows = []
    flags = []

    for td in PRODUCTION_TUPLES:
        case = td['case']
        is_open = case.endswith('_open')
        target = TARGET_OPEN if is_open else TARGET_CLOSED
        gamma1 = td.get('gamma1', 0.0)
        gamma2 = td.get('gamma2', 0.0)

        union_grid = read_pilot_grid(
            case, N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'], gamma1=gamma1, gamma2=gamma2,
            convention=CONVENTION,
        )
        grid_size = len(union_grid)

        snr = snr_by_chi(
            case,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=gamma1, gamma2=gamma2,
            pilot_grid=union_grid,
            convention=CONVENTION,
        )
        if not snr:
            print(f"[skip] no SNR data for {case} "
                  f"N={td.get('N')} alpha={td['alpha']}")
            continue

        chi_conv, residuals = find_chi_convergence(snr, target)
        chi_prod = chi_production_from_conv(chi_conv)
        max_res = max(residuals.values()) if residuals else float('nan')

        coarse_res = float('nan')
        coarse_pts = 0
        coarse_worst = None

        if not skip_verify and chi_conv is not None:
            params = {
                'N': td.get('N'),
                'Lx': td.get('Lx'),
                'Ly': td.get('Ly'),
                'alpha': td['alpha'],
                'gamma1': gamma1,
                'gamma2': gamma2,
            }
            try:
                coarse_res, coarse_worst, coarse_pts = \
                    verify_chi_on_coarse_grid(
                        case, params,
                        chi_conv=chi_conv,
                        chi_double=min(chi_conv * 2, CHI_SCHEDULE[-1]),
                        convention=CONVENTION,
                    )
            except Exception as exc:
                print(f"[verify] {case} N={td.get('N')} "
                      f"alpha={td['alpha']}: verification failed "
                      f"({exc})")
                coarse_res = float('nan')

            if coarse_pts == 0 or not np.isfinite(coarse_res):
                flags.append({
                    'case': case,
                    'N': td.get('N', ''),
                    'Lx': td.get('Lx', ''),
                    'Ly': td.get('Ly', ''),
                    'alpha': td['alpha'],
                    'gamma1': gamma1,
                    'gamma2': gamma2,
                    'mu1': '',
                    'mu2': '',
                    'snr_diff': 'no_verify_points',
                    'round': 1,
                })
            elif coarse_res > target:
                flags.append({
                    'case': case,
                    'N': td.get('N', ''),
                    'Lx': td.get('Lx', ''),
                    'Ly': td.get('Ly', ''),
                    'alpha': td['alpha'],
                    'gamma1': gamma1,
                    'gamma2': gamma2,
                    'mu1': coarse_worst[0] if coarse_worst else '',
                    'mu2': coarse_worst[1] if coarse_worst else '',
                    'snr_diff': f"{coarse_res:.4e}",
                    'round': 1,
                })

        row = {
            'case': case,
            'N': td.get('N', ''),
            'Lx': td.get('Lx', ''),
            'Ly': td.get('Ly', ''),
            'alpha': td['alpha'],
            'gamma1': gamma1,
            'gamma2': gamma2,
            'protocol': 'echo',
            'chi_conv': chi_conv if chi_conv is not None else '',
            'chi_prod': chi_prod if chi_prod is not None else '',
            'pilot_grid_size': grid_size,
            'coarse_verify_residual': (f"{coarse_res:.4e}"
                                       if np.isfinite(coarse_res) else ''),
            'coarse_verify_points': coarse_pts,
            'max_residual': f"{max_res:.4e}",
        }
        rows.append(row)
        print(f"[row] {case} N={td.get('N')} Lx={td.get('Lx')} "
              f"Ly={td.get('Ly')} alpha={td['alpha']} -> "
              f"chi_conv={chi_conv} chi_prod={chi_prod} "
              f"grid={grid_size} "
              f"(max_res={max_res:.3e}, target={target:.1e}, "
              f"coarse_verify={coarse_res:.3e} over {coarse_pts} pts)")

    if rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote {output_csv} ({len(rows)} rows)")
    else:
        print("[warn] no rows written; nothing to aggregate.")

    if flags:
        with open(flags_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flag_fields)
            writer.writeheader()
            for r in flags:
                writer.writerow(r)
        print(f"Wrote {flags_csv} ({len(flags)} flagged tuples)")


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------
# (function defined below)

def sanity():
    """
    Sanity check against existing pilot data in mu1mu2 convention.
    Reads three protocol reference points (exact-echo, squeezing, OUT)
    at N=32, alpha=1 and reports chi_conv.
    """
    case = '1d_closed'
    N = 32
    alpha = 1.0

    available_grid = [
        (np.pi / 2,  np.pi / 2),   # exact echo
        (np.pi / 2,  0.0),          # squeezing
        (np.pi / 2,  np.pi),        # OUT
    ]

    print(f"Sanity check: N={N}, alpha={alpha}, case={case}, "
          f"convention={CONVENTION}")
    print(f"Using {len(available_grid)} reference points.")
    print()

    snr = {}
    for chi in CHI_SCHEDULE:
        pts = {}
        for (mu1, mu2) in available_grid:
            path = native_output_path(
                case, chi, N=N, alpha=alpha,
                convention=CONVENTION, mu1=mu1, mu2=mu2,
            )
            val = snr_from_output(path)
            if val is not None:
                pts[(round(mu1, 4), round(mu2, 4))] = val
        if pts:
            snr[chi] = pts
            print(f"  chi={chi:>3}: {len(pts)} pilot points, "
                  f"snr values = "
                  f"{[f'{v:.4f}' for v in pts.values()]}")
        else:
            print(f"  chi={chi:>3}: no data")

    print()
    chi_conv, residuals = find_chi_convergence(snr, TARGET_CLOSED)
    print("Residuals (relative SNR difference):")
    for (chi_lo, chi_hi), res in sorted(residuals.items()):
        marker = "  <-- converged" if res < TARGET_CLOSED else ""
        print(f"  ({chi_lo:>3}, {chi_hi:>3}): {res:.4e}{marker}")

    print()
    print(f"Inferred chi_conv  = {chi_conv}")
    print(f"Inferred chi_prod  = {chi_production_from_conv(chi_conv)}")
    print(f"Target residual    = {TARGET_CLOSED:.1e}")

    if chi_conv is not None:
        print(f"\n[PASS] chi_conv={chi_conv} inferred correctly.")
    else:
        print(f"\n[INFO] no convergence yet (expected — data not produced).")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--skip-verify', action='store_true',
                        help='With --aggregate, skip the coarse-grid '
                             'verification step.')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-stage-a', type=int, default=None,
                        help='Internal: tuple index for Stage A.')
    parser.add_argument('--run-tuple', type=int, default=None,
                        help='Internal: tuple index for Stage B.')
    parser.add_argument('--run-chi-idx', type=int, default=None,
                        help='Internal: chi index for Stage B.')
    parser.add_argument('--cases', nargs='+', default=None,
                        help='Submit only these case identifiers '
                             '(e.g. --cases 1d_closed). If omitted, '
                             'submit all tuples.')
    args = parser.parse_args()

    if args.run_stage_a is not None:
        run_stage_a(args.run_stage_a)
        return

    if args.run_tuple is not None and args.run_chi_idx is not None:
        run_one_task(args.run_tuple, args.run_chi_idx)
        return

    if args.sanity:
        sanity()
        return

    did_something = False
    if args.submit:
        submit_all(dry_run=args.dry_run, cases=args.cases)
        did_something = True
    if args.aggregate:
        aggregate(skip_verify=args.skip_verify)
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == '__main__':
    main()