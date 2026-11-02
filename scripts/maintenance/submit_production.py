"""scripts/maintenance/submit_production.py"""

from __future__ import annotations

import argparse
import csv
import json
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

from src.selectors import worker_command  # noqa: E402

# Policy for landscape grid resolution.
# Rationale: (i) N=16 α=0 has a sharp GHZ peak near μ=0 and needs
# 60; (ii) N=16 α≥1 and N ≥ 32 closed resolve at 40; (iii) all N=200
# closed tuples are documented as intractable below grid=40 anyway;
# (iv) open-system landscapes are smoothed by noise and resolve at
# 40 at N=16 and 20 at N ≥ 32; (v) for 2D cylinders the effective
# 1D length is Lx, so the same reasoning applies with Lx in place of
# N. Small Lx (2, 3) resolves at 20 in both closed and open; larger
# Lx resolves at 40 closed and 20 open. Ly=6 landscapes are treated
# the same as Ly=4 at the same Lx. This is derived from the physics
# of the OAT baseline (Numerical Results §2–§5), not from a pilot scan.
GRID_POLICY = {
    # ---- 1D closed ----
    ('1d_closed', 16, 1.0): 40,
    ('1d_closed', 16, 2.0): 40,
    ('1d_closed', 16, 3.0): 40,
    ('1d_closed', 16, 4.0): 40,
    ('1d_closed', 16, 6.0): 40,
    ('1d_closed', 32, '*'): 40,
    ('1d_closed', 64, '*'): 40,
    ('1d_closed', 128, '*'): 40,
    ('1d_closed', 160, '*'): 40,
    ('1d_closed', 200, '*'): 40,
    # ---- 1D open ----
    ('1d_open', 16, '*'): 40,
    ('1d_open', 32, '*'): 20,
    ('1d_open', 64, '*'): 20,
    # ---- 2D closed (the codebase geometries) ----
    ('2d_closed', (2, 4),  '*'): 20,
    ('2d_closed', (3, 4),  '*'): 20,
    ('2d_closed', (4, 4),  '*'): 40,
    ('2d_closed', (6, 4),  '*'): 40,
    ('2d_closed', (8, 4),  '*'): 40,
    ('2d_closed', (12, 4), '*'): 40,
    ('2d_closed', (16, 4), '*'): 40,
    ('2d_closed', (3, 6),  '*'): 20,
    ('2d_closed', (4, 6),  '*'): 40,
    # ---- 2D open (the codebase geometries) ----
    ('2d_open', (2, 4),  '*'): 20,
    ('2d_open', (3, 4),  '*'): 40,
    ('2d_open', (4, 4),  '*'): 40,
    ('2d_open', (6, 4),  '*'): 20,
    ('2d_open', (8, 4),  '*'): 20,
    ('2d_open', (3, 6),  '*'): 20,
}


def grid_size_for(case, params):
    """
    Look up the production grid resolution for a tuple.

    Fallback order:
      1. (case, geometry, alpha)   exact alpha
      2. (case, geometry, '*')     wildcard alpha

    Raises KeyError with a diagnostic message if neither matches.

    Closed systems (Cases A and C of the codebase) are defined only for
    alpha >= 1. The all-to-all limit alpha = 0 is excluded because
    the MPS representation encounters the entanglement wall at
    N >= 32. A closed-system tuple with alpha = 0 is therefore
    treated as a caller error and rejected before the wildcard
    lookup is attempted.
    """
    N = params.get('N')
    Lx = params.get('Lx')
    Ly = params.get('Ly')
    alpha = float(params.get('alpha', 1.0))

    # Guard: reject alpha = 0 for closed systems.
    if case in ('1d_closed', '2d_closed') and abs(alpha) < 1e-12:
        raise KeyError(
            f"GRID_POLICY: alpha = 0 is not a defined closed-system "
            f"production tuple (case={case!r}). Cases A and C of the "
            f"plan restrict closed systems to alpha in "
            f"{{1, 2, 3, 4, 6}}; alpha = 0 is an open-system-only limit "
            f"(Cases B and D)."
        )

    if Lx is not None and Ly is not None:
        geometry = (int(Lx), int(Ly))
    elif N is not None:
        geometry = int(N)
    else:
        raise ValueError(
            "grid_size_for: params must specify either N or (Lx, Ly)"
        )

    for key in [(case, geometry, alpha), (case, geometry, '*')]:
        if key in GRID_POLICY:
            return GRID_POLICY[key]

    raise KeyError(
        f"No GRID_POLICY entry for case={case!r}, "
        f"geometry={geometry!r}, alpha={alpha!r}. "
        f"Add the tuple to GRID_POLICY in submit_production.py."
    )

# Policy for production trajectory count.
# Rationale: N_traj is set by wall-time budget, not by a precision
# target. The statistical error on the SNR scales as a/sqrt(N_traj)
# with a ~ O(1) per trajectory, so at N_traj = 200 the statistical
# error is ~7% -- adequate for the qualitative physics questions the
# landscapes are designed to answer (does OUT survive at gamma2=0.1,
# does GHZ collapse, how does chi_floor scale with Lx). Chasing 1%
# precision would require N_traj >= 1000, which exceeds the wall-time
# budget of every the codebase and the codebase landscape by a factor of 5.
# This policy is derived from the Numerical Results document
# (Sections 6-8), not from a pilot scan.
N_TRAJ_POLICY = {
    # Flagship and extended tuples: 200 trajectories.
    ('1d_open', 16, '*'): 200,
    ('2d_open', (4, 4), '*'): 200,
    # Exploratory tuples: 100 trajectories.
    ('1d_open', 32, '*'): 100,
    ('1d_open', 64, '*'): 100,
    ('2d_open', (2, 4), '*'): 100,
    ('2d_open', (3, 4), '*'): 100,
    ('2d_open', (6, 4), '*'): 100,
    ('2d_open', (8, 4), '*'): 100,
    ('2d_open', (3, 6), '*'): 100,
}


def ntraj_for(case, params):
    """
    Look up the production trajectory count for an open-system tuple.

    Fallback order:
      1. (case, geometry, alpha)   exact alpha
      2. (case, geometry, '*')     wildcard alpha

    Raises KeyError for closed-system cases; those do not use
    trajectories and the caller should not invoke this lookup.
    """
    if 'closed' in case:
        raise KeyError(
            f"ntraj_for: case={case!r} is closed-system and has no "
            f"trajectory count."
        )

    N = params.get('N')
    Lx = params.get('Lx')
    Ly = params.get('Ly')
    alpha = float(params.get('alpha', 1.0))

    if Lx is not None and Ly is not None:
        geometry = (int(Lx), int(Ly))
    elif N is not None:
        geometry = int(N)
    else:
        raise ValueError(
            "ntraj_for: params must specify either N or (Lx, Ly)"
        )

    for key in [(case, geometry, alpha), (case, geometry, '*')]:
        if key in N_TRAJ_POLICY:
            return N_TRAJ_POLICY[key]

    raise KeyError(
        f"No N_TRAJ_POLICY entry for case={case!r}, "
        f"geometry={geometry!r}, alpha={alpha!r}. "
        f"Add the tuple to N_TRAJ_POLICY in submit_production.py."
    )

# Chi floor for production runs.
#
# Rationale: the the documentation v2 selector ranks pilot points by the
# half-chain entropy S_int(mu) of the INTERMEDIATE state
# T_z(mu/2)|CSS>, evaluated right after the forward twist. S_int is
# monotonic in mu and peaks near mu = pi. This ranking is correct for
# the forward OAT sweep, where the intermediate state IS the state the
# MPS must represent. It is incorrect for the echo landscape, where
# the chi requirement is set by the entanglement of the FINAL state
# after the backward twist T_z((nu-mu)/2). On the OUT line (nu = -mu),
# the final-state entanglement is maximal at mu ~ pi/2, not at pi.
# The resulting union grids sample only the top sliver of the OUT
# plateau (mu >= 2.2 for N=64), where the echo is easiest to
# represent, and the pilot's chi_conv therefore underestimates the
# requirement.
#
# This floor overrides the pilot for the affected tuples, using the
# forward OAT baseline (Numerical Results, Sec. 2-5) as the source of
# truth: at N >= 64 the baseline already needed chi >= 64 for the same
# (N, alpha) tuples; the echo is somewhat easier, but not by a factor
# of 4. A corrected selector would need to rank by the final-state
# entanglement on the OUT line rather than by S_int(mu); that is a
# the documentation revision, deferred to a future iteration.
CHI_FLOOR_POLICY = {
    # ---- 1D closed ----
    # Safety bump to 32 for the four tuples whose pilot chi_prod
    # was 16 (chi_conv=8 from the coarse pilot). The other tuples
    # are already at 64 or 128, which is 8-32x the validated
    # chi_conv=8 , so they need no bump.
    ('1d_closed', 16, 1.0): 32,
    ('1d_closed', 16, 3.0): 32,
    ('1d_closed', 16, 6.0): 32,
    ('1d_closed', 64, 6.0): 32,
    # Existing floors, kept unchanged.
    ('1d_closed', 64, 1.0): 64,
    ('1d_closed', 64, 2.0): 64,
    ('1d_closed', 64, 3.0): 64,
    ('1d_closed', 128, '*'): 64,
    ('1d_closed', 160, '*'): 64,
    ('1d_closed', 200, '*'): 128,
    # ---- 2D closed, 2D open: unchanged ----
    ('2d_closed', (2, 4),  '*'): 16,
    ('2d_closed', (3, 4),  '*'): 32,
    ('2d_closed', (4, 4),  '*'): 32,
    ('2d_closed', (6, 4),  '*'): 32,
    ('2d_closed', (8, 4),  '*'): 32,
    ('2d_closed', (12, 4), '*'): 64,
    ('2d_closed', (16, 4), '*'): 64,
    ('2d_closed', (3, 6),  '*'): 32,
    ('2d_closed', (4, 6),  '*'): 64,
    ('2d_open', (2, 4), '*'): 16,
    ('2d_open', (3, 4), '*'): 32,
    ('2d_open', (4, 4), '*'): 32,
    ('2d_open', (6, 4), '*'): 32,
    ('2d_open', (8, 4), '*'): 32,
    ('2d_open', (3, 6), '*'): 32,

    # ---- 2D closed (the codebase geometries) ----
    # The effective 1D length is Lx, so the floor follows the same
    # pattern as the 1D chain: Lx >= 12 needs 64, Lx in [4, 8] needs 32,
    # Lx <= 3 needs 16. Ly = 6 geometries get one step higher because
    # the block-site dimension 2^6 = 64 carries more internal
    # entanglement than Ly = 4 (block dimension 16).
    ('2d_closed', (2, 4),  '*'): 16,
    ('2d_closed', (3, 4),  '*'): 32,
    ('2d_closed', (4, 4),  '*'): 32,
    ('2d_closed', (6, 4),  '*'): 32,
    ('2d_closed', (8, 4),  '*'): 32,
    ('2d_closed', (12, 4), '*'): 64,
    ('2d_closed', (16, 4), '*'): 64,
    ('2d_closed', (3, 6),  '*'): 32,
    ('2d_closed', (4, 6),  '*'): 64,
    # ---- 2D open (the codebase geometries) ----
    ('2d_open', (2, 4), '*'): 16,
    ('2d_open', (3, 4), '*'): 32,
    ('2d_open', (4, 4), '*'): 32,
    ('2d_open', (6, 4), '*'): 32,
    ('2d_open', (8, 4), '*'): 32,
    ('2d_open', (3, 6), '*'): 32,
}


def _chi_floor_for(case, td):
    """
    Return the chi floor for a tuple, or 0 if no floor is set.
    """
    geometry = (int(td['Lx']), int(td['Ly'])) if 'Lx' in td else int(td['N'])
    alpha = round(float(td['alpha']), 6)
    for key in [(case, geometry, alpha), (case, geometry, '*')]:
        if key in CHI_FLOOR_POLICY:
            return CHI_FLOOR_POLICY[key]
    return 0

# =====================================================================
# the documentation: production submission driver
# =====================================================================
#
# The driver reads
#   - chi_conv.csv        (the documentation)  for chi_prod per tuple
#   - GRID_POLICY         (the documentation)  for grid_size per tuple
#   - N_TRAJ_POLICY       (the documentation)  for n_traj per open tuple
#   - tjm_only.csv        (optional)   to skip tuples escalated to TJM
#   - intractable.csv     (optional)   to skip documented failures
#
# and emits, per (phase, tuple), one or more chunked SLURM arrays.
# Each array task invokes this same script with --run-task, which
# computes the (mu, nu) coordinates from the task ID and dispatches
# to the appropriate worker.

RESOURCE_TABLES_DIR = os.path.join(ROOT_DIR, 'data', 'resource_tables')
JOB_IDS_CSV = os.path.join(RESOURCE_TABLES_DIR, 'job_ids.csv')
RETRY_STATE_JSON = os.path.join(RESOURCE_TABLES_DIR, 'resubmit_state.json')
INTTRACTABLE_CSV = os.path.join(RESOURCE_TABLES_DIR, 'intractable.csv')
TJM_ONLY_CSV = os.path.join(RESOURCE_TABLES_DIR, 'tjm_only.csv')


# Regulus reports MaxArraySize = 1001.
MAX_ARRAY_TASKS = 1001
# All production phases use the cluster maximum wall time.
DEFAULT_WALLTIME = '14-00:00:00'
# Memory escalation step for auto_resubmit.
MEM_ESCALATION_GB = 32
MAX_MEM_GB = 224
MAX_RETRIES = 4


# ---------------------------------------------------------------------
# Phase-tagged production tuple lists
# ---------------------------------------------------------------------
def _phase1_tuples():
    """
    the codebase: 1D closed systems.

    Cases A of the codebase. alpha = 0 is excluded (guard in GRID_POLICY).
    """
    Ns = [16, 32, 64, 128, 160, 200]
    alphas = [1.0, 2.0, 3.0, 4.0, 6.0]
    return [
        {'case': '1d_closed', 'N': N, 'alpha': a, 'phase': 1}
        for N in Ns for a in alphas
    ]


def _phase2_tuples():
    """
    the codebase: 1D open systems.

    Case B of the codebase. Flagship + extended + exploratory tuples.
    """
    tuples = []
    # Flagship landscapes (full 40x40, n_traj = 200)
    for g2 in [0.01, 0.1, 0.5]:
        tuples.append({'case': '1d_open', 'N': 16, 'alpha': 3.0,
                       'gamma1': 0.0, 'gamma2': g2, 'phase': 2})
    tuples.append({'case': '1d_open', 'N': 16, 'alpha': 1.0,
                   'gamma1': 0.0, 'gamma2': 0.1, 'phase': 2})
    tuples.append({'case': '1d_open', 'N': 16, 'alpha': 0.0,
                   'gamma1': 0.0, 'gamma2': 0.1, 'phase': 2})
    # Extended landscapes (full 40x40, n_traj = 200)
    for g1 in [0.01, 0.1, 1.0]:
        tuples.append({'case': '1d_open', 'N': 16, 'alpha': 3.0,
                       'gamma1': g1, 'gamma2': 0.0, 'phase': 2})
    tuples.append({'case': '1d_open', 'N': 16, 'alpha': 3.0,
                   'gamma1': 0.01, 'gamma2': 0.01, 'phase': 2})
    for g2 in [0.5, 1.0]:
        tuples.append({'case': '1d_open', 'N': 16, 'alpha': 1.0,
                       'gamma1': 0.0, 'gamma2': g2, 'phase': 2})
    # Exploratory landscapes (20x20, n_traj = 100)
    for a in [0.0, 1.0, 3.0]:
        tuples.append({'case': '1d_open', 'N': 32, 'alpha': a,
                       'gamma1': 0.0, 'gamma2': 0.1, 'phase': 2})
    for a in [2.0, 4.0, 6.0]:
        tuples.append({'case': '1d_open', 'N': 16, 'alpha': a,
                       'gamma1': 0.0, 'gamma2': 0.1, 'phase': 2})
    for a in [0.0, 3.0]:
        tuples.append({'case': '1d_open', 'N': 64, 'alpha': a,
                       'gamma1': 0.0, 'gamma2': 0.1, 'phase': 2})
    return tuples


def _phase3_tuples():
    """
    the codebase: 2D closed systems (cylinder).
    """
    geometries = [(2, 4), (3, 4), (4, 4), (6, 4), (8, 4),
                  (12, 4), (16, 4), (3, 6), (4, 6)]
    alphas = [1.0, 2.0, 3.0, 4.0, 6.0]
    return [
        {'case': '2d_closed', 'Lx': Lx, 'Ly': Ly, 'alpha': a, 'phase': 3}
        for (Lx, Ly) in geometries for a in alphas
    ]


def _phase4_tuples():
    """
    the codebase: 2D open systems (cylinder).
    """
    geometries = [(2, 4), (3, 4), (4, 4), (6, 4), (8, 4), (3, 6)]
    alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    tuples = []
    for (Lx, Ly) in geometries:
        for a in alphas:
            tuples.append({
                'case': '2d_open', 'Lx': Lx, 'Ly': Ly, 'alpha': a,
                'gamma1': 0.0, 'gamma2': 0.1, 'phase': 4,
            })
    return tuples


PHASE_TUPLES = {
    1: _phase1_tuples(),
    2: _phase2_tuples(),
    3: _phase3_tuples(),
    4: _phase4_tuples(),
}

PHASE_TUPLES_BY_CASE = {
    '1d_closed': [t for t in PHASE_TUPLES[1] if t['case'] == '1d_closed'],
    '1d_open':   [t for t in PHASE_TUPLES[2] if t['case'] == '1d_open'],
    '2d_closed': [t for t in PHASE_TUPLES[3] if t['case'] == '2d_closed'],
    '2d_open':   [t for t in PHASE_TUPLES[4] if t['case'] == '2d_open'],
}


# ---------------------------------------------------------------------
# Resource-table readers
# ---------------------------------------------------------------------
def _tuple_key(td):
    """
    Canonical tuple key for chi_conv.csv lookup.
    """
    return (
        td['case'],
        int(td['N']) if 'N' in td else None,
        int(td['Lx']) if 'Lx' in td else None,
        int(td['Ly']) if 'Ly' in td else None,
        round(float(td['alpha']), 6),
        round(float(td.get('gamma1', 0.0)), 6),
        round(float(td.get('gamma2', 0.0)), 6),
    )


def read_chi_conv():
    """
    Read chi_conv.csv into a dict keyed by tuple_key. Returns an empty
    dict if the file does not exist or is unreadable.
    """
    if not os.path.exists(os.path.join(RESOURCE_TABLES_DIR, 'chi_conv.csv')):
        return {}
    path = os.path.join(RESOURCE_TABLES_DIR, 'chi_conv.csv')
    out = {}
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row['case'],
                    int(row['N']) if row.get('N') else None,
                    int(row['Lx']) if row.get('Lx') else None,
                    int(row['Ly']) if row.get('Ly') else None,
                    round(float(row['alpha']), 6),
                    round(float(row.get('gamma1', 0.0) or 0.0), 6),
                    round(float(row.get('gamma2', 0.0) or 0.0), 6),
                )
                chi_prod_str = (row.get('chi_prod') or '').strip()
                chi_conv_str = (row.get('chi_conv') or '').strip()
                if chi_prod_str:
                    out[key] = int(float(chi_prod_str))
                elif chi_conv_str:
                    # Fall back to chi_conv * 1.5 rounded up.
                    cc = int(float(chi_conv_str))
                    out[key] = int(np.ceil(1.5 * cc))
    except Exception as exc:
        print(f"[read_chi_conv] WARNING: {exc}", flush=True)
    return out


def chi_prod_for(td, chi_conv):
    """
    Look up chi_prod for a tuple. Falls back to 64 if missing, and
    applies the CHI_FLOOR_POLICY as an additional lower bound.
    """
    key = _tuple_key(td)
    base = chi_conv[key] if key in chi_conv else 64
    floor = _chi_floor_for(td['case'], td)
    return max(base, floor)


def read_tjm_only():
    """
    Read tjm_only.csv into a set of tuple keys. Returns an empty set
    if the file does not exist.
    """
    if not os.path.exists(TJM_ONLY_CSV):
        return set()
    out = set()
    try:
        with open(TJM_ONLY_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row['case'],
                    int(row['N']) if row.get('N') else None,
                    int(row['Lx']) if row.get('Lx') else None,
                    int(row['Ly']) if row.get('Ly') else None,
                    round(float(row['alpha']), 6),
                    round(float(row.get('gamma1', 0.0) or 0.0), 6),
                    round(float(row.get('gamma2', 0.0) or 0.0), 6),
                )
                out.add(key)
    except Exception as exc:
        print(f"[read_tjm_only] WARNING: {exc}", flush=True)
    return out


def read_intractable():
    """
    Read intractable.csv into a set of tuple keys. Returns an empty set
    if the file does not exist, has no usable header, or cannot be
    parsed. Missing columns on any row are treated as a warning, not an
    error; the row is skipped.
    """
    if not os.path.exists(INTTRACTABLE_CSV):
        return set()
    out = set()
    try:
        with open(INTTRACTABLE_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            if 'case' not in header or 'alpha' not in header:
                print(f"[read_intractable] WARNING: file exists but "
                      f"header is {header!r}; ignoring.", flush=True)
                return set()
            for row in reader:
                try:
                    key = (
                        row['case'],
                        int(row['N']) if row.get('N') else None,
                        int(row['Lx']) if row.get('Lx') else None,
                        int(row['Ly']) if row.get('Ly') else None,
                        round(float(row['alpha']), 6),
                        round(float(row.get('gamma1', 0.0) or 0.0), 6),
                        round(float(row.get('gamma2', 0.0) or 0.0), 6),
                    )
                except (KeyError, ValueError, TypeError) as row_exc:
                    print(f"[read_intractable] skipping malformed row: "
                          f"{row_exc}", flush=True)
                    continue
                out.add(key)
    except Exception as exc:
        print(f"[read_intractable] WARNING: {exc}", flush=True)
    return out


# ---------------------------------------------------------------------
# Memory and walltime computation
# ---------------------------------------------------------------------
def compute_mem_gb(chi_prod):
    """
    Memory formula from the codebase: min(64 + 4 * chi_prod, 224) GB.
    """
    return int(min(64 + 4 * int(chi_prod), MAX_MEM_GB))


# ---------------------------------------------------------------------
# Array chunking
# ---------------------------------------------------------------------
def _chunk_arrays(n_tasks):
    """
    Yield (start, end) pairs of array indices, each spanning at most
    MAX_ARRAY_TASKS tasks.

    Regulus reports MaxArraySize = 1001, so a chunk of size 1001 is
    the largest single --array=0-(N-1) that is accepted.
    """
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


# ---------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------
def _write_array_script(phase, tuple_idx, chi_prod, grid_size,
                       chunk_start, chunk_end, mem_gb, walltime,
                       convention='mu_nu'):
    """
    Write one SLURM array script for a contiguous chunk of array
    indices. Returns the path to the script.
    """
    td = PHASE_TUPLES[phase][tuple_idx]
    case = td['case']

    jobdir = os.path.join(ROOT_DIR, 'scripts', 'maintenance',
                          '_production_jobs')
    logdir = os.path.join(ROOT_DIR, 'logs', 'production')
    os.makedirs(jobdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tag = (f"prod_p{phase}_t{tuple_idx:03d}_chi{chi_prod}"
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

python scripts/maintenance/submit_production.py \\
   --run-task {phase} {tuple_idx} {chi_prod} {grid_size} \\
    $(({chunk_start} + SLURM_ARRAY_TASK_ID)) \\
   --convention {convention}
"""
    with open(script_path, 'w') as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path


def _submit(script, extra_args=None):
    """
    Submit one SLURM script. Returns the job ID as a string, or None
    on failure.
    """
    cmd = ['sbatch', '--parsable']
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(script)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"[submit] sbatch failed: {exc.stderr.strip()}", flush=True)
        return None


# ---------------------------------------------------------------------
# Job ID log
# ---------------------------------------------------------------------
def _append_job_id(record):
    """
    Append one row to job_ids.csv, creating the file with a header if
    it does not exist.
    """
    os.makedirs(RESOURCE_TABLES_DIR, exist_ok=True)
    fieldnames = [
        'timestamp', 'phase', 'case', 'N', 'Lx', 'Ly', 'alpha',
        'gamma1', 'gamma2', 'chi_prod', 'grid_size', 'n_traj',
        'mem_gb', 'walltime', 'chunk_idx', 'chunk_start', 'chunk_end',
        'job_id', 'script_path',
    ]
    write_header = not os.path.exists(JOB_IDS_CSV)
    with open(JOB_IDS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ---------------------------------------------------------------------
# Submission driver
# ---------------------------------------------------------------------
def submit_phase(phase, dry_run=False, convention='mu_nu'):
    """
    Enumerate the tuples in PHASE_TUPLES[phase], look up chi_prod and
    grid_size (and n_traj for open systems), compute the memory
    request, and submit one or more chunked SLURM arrays per tuple.
    """
    if phase not in PHASE_TUPLES:
        raise ValueError(f"Unknown phase: {phase!r}")

    chi_conv = read_chi_conv()
    tjm_only = read_tjm_only()
    intractable = read_intractable()

    tuples = PHASE_TUPLES[phase]
    print(f"[submit] phase {phase}: {len(tuples)} tuples in list",
          flush=True)

    n_submitted = 0
    n_skipped = 0

    for idx, td in enumerate(tuples):
        case = td['case']
        key = _tuple_key(td)

        if key in tjm_only:
            print(f"[skip] tuple {idx} in tjm_only.csv: {td}", flush=True)
            n_skipped += 1
            continue
        if key in intractable:
            print(f"[skip] tuple {idx} in intractable.csv: {td}",
                  flush=True)
            n_skipped += 1
            continue

        # Guard: closed-system tuples must not have alpha = 0.
        if 'closed' in case and abs(float(td['alpha'])) < 1e-12:
            print(f"[skip] closed-system tuple {idx} has alpha=0: {td}",
                  flush=True)
            n_skipped += 1
            continue

        try:
            chi_prod = chi_prod_for(td, chi_conv)
        except KeyError as exc:
            print(f"[skip] tuple {idx}: {exc}", flush=True)
            n_skipped += 1
            continue

        try:
            grid_size = grid_size_for(case, td)
        except KeyError as exc:
            print(f"[skip] tuple {idx}: {exc}", flush=True)
            n_skipped += 1
            continue

        # Open-system trajectory count; closed-system has no such parameter.
        n_traj = 0
        if 'open' in case:
            try:
                n_traj = ntraj_for(case, td)
            except KeyError as exc:
                print(f"[skip] tuple {idx}: {exc}", flush=True)
                n_skipped += 1
                continue

        mem_gb = compute_mem_gb(chi_prod)
        walltime = DEFAULT_WALLTIME

        n_tasks = grid_size * grid_size
        chunks = list(_chunk_arrays(n_tasks))

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            script = _write_array_script(
                phase, idx, chi_prod, grid_size,
                chunk_start, chunk_end, mem_gb, walltime,
                convention=convention,
            )
            chunk_size = chunk_end - chunk_start + 1

            if dry_run:
                print(f"[dry-run] phase {phase}, tuple {idx} "
                      f"({case}, chi={chi_prod}, grid={grid_size}, "
                      f"n_traj={n_traj}), chunk {chunk_idx} "
                      f"[{chunk_start}..{chunk_end}], {chunk_size} tasks: "
                      f"{script}", flush=True)
                continue

            job_id = _submit(script)
            if job_id is None:
                print(f"[error] sbatch failed for tuple {idx} "
                      f"chunk {chunk_idx}", flush=True)
                continue

            record = {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'phase': phase,
                'case': case,
                'N': td.get('N', ''),
                'Lx': td.get('Lx', ''),
                'Ly': td.get('Ly', ''),
                'alpha': td['alpha'],
                'gamma1': td.get('gamma1', 0.0),
                'gamma2': td.get('gamma2', 0.0),
                'chi_prod': chi_prod,
                'grid_size': grid_size,
                'n_traj': n_traj,
                'mem_gb': mem_gb,
                'walltime': walltime,
                'chunk_idx': chunk_idx,
                'chunk_start': chunk_start,
                'chunk_end': chunk_end,
                'job_id': job_id,
                'script_path': script,
            }
            _append_job_id(record)
            print(f"[submit] phase {phase} tuple {idx} "
                  f"chunk {chunk_idx} -> job {job_id} "
                  f"({chunk_size} tasks, mem {mem_gb}G, "
                  f"walltime {walltime})", flush=True)
            n_submitted += 1

    print(f"[submit] done. {n_submitted} arrays submitted, "
          f"{n_skipped} tuples skipped.", flush=True)


# ---------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------
def run_task(phase, tuple_idx, chi_prod, grid_size, task_id,
             convention='mu_nu'):
    """
    Dispatcher for one array task. Computes the (mu, nu) coordinates
    from task_id and grid_size, then invokes the appropriate worker
    via worker_command.
    """
    if phase not in PHASE_TUPLES:
        raise ValueError(f"Unknown phase: {phase!r}")
    tuples = PHASE_TUPLES[phase]
    if not (0 <= tuple_idx < len(tuples)):
        raise IndexError(
            f"tuple_idx={tuple_idx} out of range for phase {phase} "
            f"(list has {len(tuples)} tuples)"
        )
    td = tuples[tuple_idx]

    if task_id < 0 or task_id >= grid_size * grid_size:
        raise IndexError(
            f"task_id={task_id} out of range for grid_size={grid_size}"
        )

    # Row-major (first-axis-major) mapping.
    i = task_id // grid_size
    j = task_id % grid_size


    case = td['case']
    n_traj = 0
    if 'open' in case:
        n_traj = ntraj_for(case, td)


    if convention == 'mu1mu2':
        a_vals = np.linspace(0.0, np.pi, grid_size)
        b_vals = np.linspace(0.0, 2.0 * np.pi, grid_size)
        a = float(a_vals[i])
        b = float(b_vals[j])
        cmd = worker_command(
            case, chi_prod,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'],
            gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
            n_traj=n_traj,
            convention='mu1mu2', mu1=a, mu2=b,
        )
        coord_label = f"mu1={a:.4f} mu2={b:.4f}"
    elif convention == 'mu_nu':
        a_vals = np.linspace(0.0, np.pi, grid_size)
        b_vals = np.linspace(-np.pi, np.pi, grid_size)
        a = float(a_vals[i])
        b = float(b_vals[j])
        cmd = worker_command(
            case, chi_prod,
            N=td.get('N'), Lx=td.get('Lx'), Ly=td.get('Ly'),
            alpha=td['alpha'], mu=a, nu=b,
            gamma1=td.get('gamma1', 0.0), gamma2=td.get('gamma2', 0.0),
            n_traj=n_traj,
        )
        coord_label = f"mu={a:.4f} nu={b:.4f}"
    else:
        raise ValueError(f"Unknown convention: {convention!r}")


    print(f"[run-task] phase={phase} tuple_idx={tuple_idx} "
          f"case={case} chi={chi_prod} grid={grid_size} "
          f"task_id={task_id} -> {coord_label} "
          f"n_traj={n_traj} convention={convention}", flush=True)
    print(f"[run-task] cmd: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Production submission driver (the documentation).",
    )
    parser.add_argument('--submit', type=int, choices=[1, 2, 3, 4],
                        default=None,
                        help='Submit production arrays for the given phase.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the array scripts without submitting.')
    parser.add_argument('--run-task', nargs=5, default=None,
                       metavar=('PHASE', 'TUPLE_IDX', 'CHI',
                                'GRID_SIZE', 'TASK_ID'),
                       help='Internal dispatcher for one array task.')
    parser.add_argument('--convention', choices=['mu_nu', 'mu1mu2'],
                       default='mu_nu',
                       help="Coordinate convention for --submit and "
                            "--run-task. 'mu_nu' preserves legacy "
                            "behavior; 'mu1mu2' uses the paper convention "
                            "with mu1 in [0, pi] and mu2 in [0, 2 pi].")
    args = parser.parse_args()


    if args.run_task is not None:
        phase_s, tuple_idx_s, chi_s, grid_s, task_s = args.run_task
        run_task(int(phase_s), int(tuple_idx_s),
                int(chi_s), int(grid_s), int(task_s),
                convention=args.convention)
        return


    if args.submit is not None:
        submit_phase(args.submit, dry_run=args.dry_run,
                     convention=args.convention)
        return

    parser.print_help()


if __name__ == '__main__':
    main()