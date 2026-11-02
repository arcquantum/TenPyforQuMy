"""
scripts/maintenance/auto_resubmit.py


SLURM watcher that reads job_ids.csv, polls sacct for terminated
jobs, and resubmits any that failed with an OOM or SIGKILL status
using an increased --mem. Tracks retry counts in a JSON state file
and appends to intractable.csv once a job exceeds MAX_RETRIES.

Usage
-----
    python scripts/maintenance/auto_resubmit.py --poll
    python scripts/maintenance/auto_resubmit.py --poll --once
    python scripts/maintenance/auto_resubmit.py --status
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, ROOT_DIR)

RESOURCE_TABLES_DIR = os.path.join(ROOT_DIR, 'data', 'resource_tables')
JOB_IDS_CSV = os.path.join(RESOURCE_TABLES_DIR, 'job_ids.csv')
RETRY_STATE_JSON = os.path.join(RESOURCE_TABLES_DIR, 'resubmit_state.json')
INTTRACTABLE_CSV = os.path.join(RESOURCE_TABLES_DIR, 'intractable.csv')


MEM_ESCALATION_GB = 32
MAX_MEM_GB = 224
MAX_RETRIES = 4
POLL_INTERVAL_S = 300


# ---------------------------------------------------------------------
# Retry-state bookkeeping
# ---------------------------------------------------------------------
def _load_state():
    if not os.path.exists(RETRY_STATE_JSON):
        return {}
    try:
        with open(RETRY_STATE_JSON, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    os.makedirs(RESOURCE_TABLES_DIR, exist_ok=True)
    tmp = RETRY_STATE_JSON + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RETRY_STATE_JSON)


# ---------------------------------------------------------------------
# SLURM query
# ---------------------------------------------------------------------
def _sacct(job_id):
    """
    Query sacct for a single job. Returns a dict with keys
    {'state': str, 'exit_code': str} or None on failure.
    """
    try:
        result = subprocess.run(
            ['sacct', '-j', str(job_id), '--noheader',
             '--parsable2', '--format=State,ExitCode'],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[sacct] failed for {job_id}: {exc.stderr.strip()}",
              flush=True)
        return None

    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if not lines:
        return None

    # First line is the parent job; subsequent lines are steps.
    state, exit_code = (lines[0].split('|') + ['', ''])[:2]
    return {'state': state.strip(), 'exit_code': exit_code.strip()}


def _is_failure(state, exit_code):
    """
    Return True if the job's state indicates an OOM or SIGKILL
    failure that warrants a resubmission.
    """
    if state in ('OUT_OF_MEMORY', 'OOM'):
        return True
    if exit_code in ('9:0', '9:15', '9:9'):
        return True
    return False


# ---------------------------------------------------------------------
# Intractable log
# ---------------------------------------------------------------------
def _append_intractable(record, reason):
    os.makedirs(RESOURCE_TABLES_DIR, exist_ok=True)
    fieldnames = [
        'timestamp', 'phase', 'case', 'N', 'Lx', 'Ly', 'alpha',
        'gamma1', 'gamma2', 'chi_prod', 'grid_size', 'n_traj',
        'job_id', 'reason',
    ]
    write_header = not os.path.exists(INTTRACTABLE_CSV)
    with open(INTTRACTABLE_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'phase': record.get('phase', ''),
            'case': record.get('case', ''),
            'N': record.get('N', ''),
            'Lx': record.get('Lx', ''),
            'Ly': record.get('Ly', ''),
            'alpha': record.get('alpha', ''),
            'gamma1': record.get('gamma1', ''),
            'gamma2': record.get('gamma2', ''),
            'chi_prod': record.get('chi_prod', ''),
            'grid_size': record.get('grid_size', ''),
            'n_traj': record.get('n_traj', ''),
            'job_id': record.get('job_id', ''),
            'reason': reason,
        })


# ---------------------------------------------------------------------
# Read job_ids.csv
# ---------------------------------------------------------------------
def _read_job_ids():
    if not os.path.exists(JOB_IDS_CSV):
        return []
    with open(JOB_IDS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------
# Resubmission
# ---------------------------------------------------------------------
def _resubmit_with_mem(record, new_mem_gb):
    """
    Rewrite the original SLURM script with the new memory request,
    then submit it. Returns the new job ID or None.
    """
    script = record.get('script_path', '')
    if not script or not os.path.exists(script):
        print(f"[resubmit] script not found: {script}", flush=True)
        return None

    with open(script, 'r') as f:
        body = f.read()

    # Replace the #SBATCH --mem line.
    new_lines = []
    replaced = False
    for line in body.splitlines():
        if line.startswith('#SBATCH --mem='):
            new_lines.append(f'#SBATCH --mem={new_mem_gb}G')
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        print(f"[resubmit] no #SBATCH --mem line found in {script}",
              flush=True)
        return None

    with open(script, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

    try:
        result = subprocess.run(
            ['sbatch', '--parsable', script],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"[resubmit] sbatch failed: {exc.stderr.strip()}",
              flush=True)
        return None


def _resubmit(record, state):
    """
    Handle the resubmission of a single failed job.
    """
    job_id = str(record['job_id'])
    prev = state.get(job_id, {'retries': 0, 'last_mem_gb': int(record['mem_gb'])})
    retries = int(prev.get('retries', 0)) + 1

    if retries > MAX_RETRIES:
        print(f"[resubmit] job {job_id}: exceeded {MAX_RETRIES} "
              f"retries, marking intractable", flush=True)
        _append_intractable(record, 'max_retries_exceeded')
        state[job_id] = {'retries': retries,
                         'last_mem_gb': int(prev.get('last_mem_gb', 0)),
                         'status': 'intractable'}
        return

    new_mem = min(int(prev.get('last_mem_gb', int(record['mem_gb'])))
                  + MEM_ESCALATION_GB, MAX_MEM_GB)

    new_job = _resubmit_with_mem(record, new_mem)
    if new_job is None:
        return

    state[job_id] = {
        'retries': retries,
        'last_mem_gb': new_mem,
        'resubmitted_as': new_job,
        'status': 'resubmitted',
    }
    print(f"[resubmit] job {job_id} -> {new_job} "
          f"(retry {retries}, mem {new_mem}G)", flush=True)


# ---------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------
def _poll_once():
    records = _read_job_ids()
    state = _load_state()

    for record in records:
        job_id = str(record.get('job_id', '')).strip()
        if not job_id:
            continue

        # Skip jobs that are already known-terminal in the state file.
        prev = state.get(job_id, {})
        if prev.get('status') in ('intractable',):
            continue
        if prev.get('status') == 'resubmitted':
            # Only re-examine the replacement.
            new_job = str(prev.get('resubmitted_as', ''))
            if new_job:
                result = _sacct(new_job)
                if result and _is_failure(result['state'],
                                          result['exit_code']):
                    _resubmit(record, state)
            continue

        result = _sacct(job_id)
        if result is None:
            continue

        state_val = result['state']
        exit_code = result['exit_code']

        # Still running? Skip.
        if state_val in ('RUNNING', 'PENDING', 'CONFIGURING',
                         'COMPLETING', 'SUSPENDED'):
            continue

        # Completed normally? Mark as done.
        if state_val == 'COMPLETED' and exit_code in ('0:0', ''):
            state[job_id] = {'status': 'completed', 'retries': 0}
            continue

        # Any other terminal state is a failure.
        if _is_failure(state_val, exit_code):
            _resubmit(record, state)
        else:
            state[job_id] = {
                'status': 'failed',
                'state': state_val,
                'exit_code': exit_code,
                'retries': 0,
            }

    _save_state(state)


def _status():
    records = _read_job_ids()
    state = _load_state()
    print(f"Job ID log: {JOB_IDS_CSV}")
    print(f"Retry state: {RETRY_STATE_JSON}")
    print(f"Total arrays: {len(records)}")
    counts = {}
    for job_id, info in state.items():
        s = info.get('status', 'unknown')
        counts[s] = counts.get(s, 0) + 1
    for s, n in sorted(counts.items()):
        print(f"  {s:>15}: {n}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()

    if args.status:
        _status()
        return

    if args.poll:
        if args.once:
            _poll_once()
        else:
            while True:
                print(f"[poll] {datetime.now().isoformat(timespec='seconds')}",
                      flush=True)
                _poll_once()
                time.sleep(POLL_INTERVAL_S)
        return

    parser.print_help()


if __name__ == '__main__':
    main()