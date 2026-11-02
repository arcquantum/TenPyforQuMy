#!/usr/bin/env python3
"""
scripts/checks/test_production_ranges.py

Seven-layer structural test for the migrated production pipeline.

Verifies that the (mu1, mu2) convention is correctly threaded through
every code path that produces worker CLI arguments, output filenames,
SLURM scripts, forward-sweep data, or Scharnagl task coordinates, and
that both conventions give the same physics at a reference point.

Layers
------
1. Config invariants
   PILOT_GRID_MU1MU2 lies in [0, pi] x [0, 2pi].
   auto_chi_pilot, auto_grid_pilot, auto_ntraj_pilot carry
   CONVENTION = "mu1mu2".
   (auto_ntraj_pilot is SKIP until Stage 5e.3 lands.)

2. worker_command dispatch
   convention='mu1mu2' emits --mu1/--mu2, not --mu/--nu.
   convention='mu_nu'  emits --mu/--nu, not --mu1/--mu2.
   Missing arguments raise ValueError.

3. native_output_path suffix
   convention='mu1mu2' -> "_mu1{...}_mu2{...}"
   convention='mu_nu'  -> "_mu{...}_nu{...}"

4. submit_production
   Part A: run_task task_id -> (mu1, mu2) or (mu, nu) mapping,
           for 9 task IDs spanning the 40x40 grid, both conventions.
           Bogus convention raises ValueError.
   Part B: submit_phase writes SLURM scripts whose --run-task bodies
           carry --convention {chosen convention}. PHASE_TUPLES and
           subprocess.run are patched; job_ids.csv and _production_jobs
           are byte-restored afterward.

5. submit_scharnagl_stage1
   MU1_VALS in [0, pi], MU2_VALS in [0, 2pi].
   run_task dispatches to --mu1/--mu2 (not --mu/--nu).
   (SKIP until Stage 5e.2.)

6. Forward sweep + asymmetric SLURM driver
   Part A: forward plotters use "mu1 = 4.0 * 1.0 * t / N" and xlabel
           r"$\\mu_1$".
   Part B: main_powerlaw.py at N=8, alpha=3, t_max=0.1 produces a
           .npy with correct schema and x-axis normalization
           mu1 = 4 J t_max / N. Files cleaned up afterward.
   Part C: asymmetric SLURM driver exists.
           (SKIP until Stage 5f.)

7. Cross-convention physics
   powerlaw_echo_worker.py at N=8, alpha=3, chi=8:
     (mu1, mu2) = (pi/2, pi/2)  in new convention
     (mu, nu)   = (2pi, 0)      in old convention
   produce equal SNR to 1e-5.

Exit code 0 if no FAIL / ERROR. SKIP does not fail the test.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ======================================================================
# Helpers
# ======================================================================
def _import_by_path(name: str, *candidates: str):
    """Import a module by file path; return None on any failure."""
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            continue
    return None


def _import_pilot(name: str):
    return _import_by_path(
        name,
        os.path.join(ROOT, 'scripts', 'maintenance', name + '.py'),
        os.path.join(ROOT, 'scripts', name + '.py'),
    )


class _FakeCompleted:
    returncode = 0
    stdout = '12345\n'
    stderr = ''


# ======================================================================
# Layer 1 -- config invariants
# ======================================================================
def layer_1_config():
    from src.selectors import PILOT_GRID_MU1MU2

    for m1, m2 in PILOT_GRID_MU1MU2:
        if not (-1e-9 <= m1 <= np.pi + 1e-9):
            return ("FAIL", f"PILOT_GRID_MU1MU2 mu1={m1!r} out of [0,pi]")
        if not (-1e-9 <= m2 <= 2.0 * np.pi + 1e-9):
            return ("FAIL", f"PILOT_GRID_MU1MU2 mu2={m2!r} out of [0,2pi]")

    fail_notes = []
    pending_notes = []

    for name in ('auto_chi_pilot', 'auto_grid_pilot'):
        mod = _import_pilot(name)
        if mod is None:
            fail_notes.append(f'{name}: not importable')
            continue
        conv = getattr(mod, 'CONVENTION', None)
        if conv != 'mu1mu2':
            fail_notes.append(f'{name}: CONVENTION={conv!r}')

    mod = _import_pilot('auto_ntraj_pilot')
    if mod is None:
        pending_notes.append('auto_ntraj_pilot: not importable (Stage 5e.3)')
    else:
        conv = getattr(mod, 'CONVENTION', None)
        if conv != 'mu1mu2':
            pending_notes.append(
                f'auto_ntraj_pilot: CONVENTION={conv!r} (Stage 5e.3)'
            )

    if fail_notes:
        return ("FAIL", "; ".join(fail_notes))
    if pending_notes:
        return ("SKIP", "; ".join(pending_notes))
    return ("PASS", "PILOT_GRID_MU1MU2 in range; all pilots CONVENTION='mu1mu2'")


# ======================================================================
# Layer 2 -- worker_command dispatch
# ======================================================================
def layer_2_worker_command():
    from src.selectors import worker_command

    cmd = worker_command('1d_closed', 32, N=16, alpha=1.0,
                         convention='mu1mu2', mu1=0.5, mu2=1.5)
    if '--mu1' not in cmd or '--mu2' not in cmd:
        return ("FAIL", f"mu1mu2 missing --mu1/--mu2: {cmd}")
    if '--mu' in cmd or '--nu' in cmd:
        return ("FAIL", f"mu1mu2 unexpectedly has --mu/--nu: {cmd}")
    if cmd[cmd.index('--mu1') + 1] != '0.500000':
        return ("FAIL", f"mu1mu2 mu1 arg not 0.500000: {cmd}")
    if cmd[cmd.index('--mu2') + 1] != '1.500000':
        return ("FAIL", f"mu1mu2 mu2 arg not 1.500000: {cmd}")

    cmd = worker_command('1d_closed', 32, N=16, alpha=1.0,
                         convention='mu_nu', mu=0.5, nu=-1.5)
    if '--mu' not in cmd or '--nu' not in cmd:
        return ("FAIL", f"mu_nu missing --mu/--nu: {cmd}")
    if '--mu1' in cmd or '--mu2' in cmd:
        return ("FAIL", f"mu_nu unexpectedly has --mu1/--mu2: {cmd}")
    if cmd[cmd.index('--nu') + 1] != '-1.500000':
        return ("FAIL", f"mu_nu nu arg not -1.500000: {cmd}")

    try:
        worker_command('1d_closed', 32, N=16, alpha=1.0,
                       convention='mu1mu2')
        return ("FAIL", "no error on missing mu1/mu2")
    except ValueError:
        pass
    try:
        worker_command('1d_closed', 32, N=16, alpha=1.0,
                       convention='mu_nu')
        return ("FAIL", "no error on missing mu/nu")
    except ValueError:
        pass

    return ("PASS", "worker_command dispatches both conventions correctly")


# ======================================================================
# Layer 3 -- native_output_path suffix
# ======================================================================
def layer_3_native_output_path():
    from src.selectors import native_output_path

    p = native_output_path('1d_closed', 32, N=16, alpha=1.0,
                           convention='mu1mu2', mu1=0.5, mu2=1.5)
    base = os.path.basename(p)
    if '_mu10.5000_mu21.5000' not in base:
        return ("FAIL", f"mu1mu2 filename suffix missing: {base}")
    if '_nu' in base:
        return ("FAIL", f"mu1mu2 filename contains _nu: {base}")

    p = native_output_path('1d_closed', 32, N=16, alpha=1.0,
                           convention='mu_nu', mu=0.5, nu=-1.5)
    base = os.path.basename(p)
    if '_mu0.5000_nu-1.5000' not in base:
        return ("FAIL", f"mu_nu filename suffix missing: {base}")
    if '_mu1' in base or '_mu2' in base:
        return ("FAIL", f"mu_nu filename contains mu1/mu2: {base}")

    return ("PASS", "native_output_path suffixes correct for both conventions")


# ======================================================================
# Layer 4 -- submit_production (run_task + submit_phase)
# ======================================================================
def layer_4_submit_production():
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'maintenance'))
    import submit_production as sp

    # --------------------------------------------------------------
    # Part A -- run_task task_id -> coordinates (both conventions)
    # --------------------------------------------------------------
    GRID = 40
    mu1_vals = np.linspace(0.0, np.pi, GRID)
    mu2_vals = np.linspace(0.0, 2.0 * np.pi, GRID)
    mu_vals  = np.linspace(0.0, np.pi, GRID)
    nu_vals  = np.linspace(-np.pi, np.pi, GRID)

    cases = [
        (0,    'mu1mu2', mu1_vals[0],  mu2_vals[0]),
        (39,   'mu1mu2', mu1_vals[0],  mu2_vals[39]),
        (40,   'mu1mu2', mu1_vals[1],  mu2_vals[0]),
        (1560, 'mu1mu2', mu1_vals[39], mu2_vals[0]),
        (1599, 'mu1mu2', mu1_vals[39], mu2_vals[39]),
        (0,    'mu_nu',  mu_vals[0],   nu_vals[0]),
        (39,   'mu_nu',  mu_vals[0],   nu_vals[39]),
        (1560, 'mu_nu',  mu_vals[39],  nu_vals[0]),
        (1599, 'mu_nu',  mu_vals[39],  nu_vals[39]),
    ]

    captured = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return _FakeCompleted()

    with patch('subprocess.run', fake_run):
        for task_id, conv, _, _ in cases:
            sp.run_task(1, 0, 64, GRID, task_id, convention=conv)

    if len(captured) != len(cases):
        return ("FAIL",
                f"Part A: captured {len(captured)} calls, expected "
                f"{len(cases)}")
    for (task_id, conv, e_first, e_second), cmd in zip(cases, captured):
        if conv == 'mu1mu2':
            if '--mu1' not in cmd or '--mu2' not in cmd:
                return ("FAIL",
                        f"Part A task_id={task_id} {conv}: missing "
                        f"--mu1/--mu2")
            got_first  = float(cmd[cmd.index('--mu1') + 1])
            got_second = float(cmd[cmd.index('--mu2') + 1])
        else:
            if '--mu' not in cmd or '--nu' not in cmd:
                return ("FAIL",
                        f"Part A task_id={task_id} {conv}: missing "
                        f"--mu/--nu")
            got_first  = float(cmd[cmd.index('--mu') + 1])
            got_second = float(cmd[cmd.index('--nu') + 1])
        if abs(got_first - e_first) > 1e-5:
            return ("FAIL",
                    f"Part A task_id={task_id} {conv} first axis "
                    f"got {got_first:.6f} expected {e_first:.6f}")
        if abs(got_second - e_second) > 1e-5:
            return ("FAIL",
                    f"Part A task_id={task_id} {conv} second axis "
                    f"got {got_second:.6f} expected {e_second:.6f}")

    try:
        with patch('subprocess.run', fake_run):
            sp.run_task(1, 0, 64, GRID, 0, convention='bogus')
        return ("FAIL", "Part A: no error on bogus convention")
    except ValueError:
        pass

    # --------------------------------------------------------------
    # Part B -- submit_phase SLURM script body includes --convention
    # --------------------------------------------------------------
    job_dir = os.path.join(ROOT, 'scripts', 'maintenance',
                           '_production_jobs')
    os.makedirs(job_dir, exist_ok=True)
    job_ids_csv = os.path.join(sp.RESOURCE_TABLES_DIR, 'job_ids.csv')

    # Back up pre-existing state, restore afterward
    pre_files = {}
    for f in glob.glob(os.path.join(job_dir, 'job_prod_p1_t000_*.sh')):
        with open(f, 'r') as fh:
            pre_files[f] = fh.read()
    pre_job_ids = None
    if os.path.exists(job_ids_csv):
        with open(job_ids_csv, 'rb') as fh:
            pre_job_ids = fh.read()

    original_tuples = sp.PHASE_TUPLES
    sp.PHASE_TUPLES = {
        1: [{'case': '1d_closed', 'N': 16, 'alpha': 1.0, 'phase': 1}]
    }

    part_b_error = None
    try:
        for conv in ('mu1mu2', 'mu_nu'):
            # clean slate for this convention
            for f in glob.glob(os.path.join(
                    job_dir, 'job_prod_p1_t000_*.sh')):
                try: os.remove(f)
                except OSError: pass

            try:
                with patch('subprocess.run', fake_run):
                    sp.submit_phase(1, dry_run=False, convention=conv)
            except Exception as exc:
                part_b_error = (
                    f"Part B: submit_phase(convention={conv!r}) raised "
                    f"{type(exc).__name__}: {exc}"
                )
                break

            new_files = glob.glob(os.path.join(
                job_dir, 'job_prod_p1_t000_*.sh'))
            if not new_files:
                part_b_error = (
                    f"Part B: submit_phase(convention={conv!r}) created "
                    f"no scripts"
                )
                break

            marker = f'--convention {conv}'
            for f in new_files:
                with open(f, 'r') as fh:
                    body = fh.read()
                if '--run-task' not in body:
                    part_b_error = (
                        f"Part B: {os.path.basename(f)} has no "
                        f"--run-task line"
                    )
                    break
                if marker not in body:
                    part_b_error = (
                        f"Part B: {os.path.basename(f)} missing "
                        f"{marker!r}"
                    )
                    break
            if part_b_error:
                break
    finally:
        sp.PHASE_TUPLES = original_tuples
        # wipe anything the test created
        for f in glob.glob(os.path.join(job_dir, 'job_prod_p1_t000_*.sh')):
            try: os.remove(f)
            except OSError: pass
        # restore pre-existing scripts
        for f, content in pre_files.items():
            with open(f, 'w') as fh:
                fh.write(content)
        # restore job_ids.csv
        if pre_job_ids is not None:
            with open(job_ids_csv, 'wb') as fh:
                fh.write(pre_job_ids)
        elif os.path.exists(job_ids_csv):
            try: os.remove(job_ids_csv)
            except OSError: pass

    if part_b_error:
        return ("FAIL", part_b_error)

    return ("PASS",
            "Part A: 9 run_task task_id cases correct; "
            "Part B: submit_phase SLURM scripts carry --convention")


# ======================================================================
# Layer 5 -- submit_scharnagl_stage1
# ======================================================================
def layer_5_submit_scharnagl():
    mod = _import_pilot('submit_scharnagl_stage1')
    if mod is None:
        return ("SKIP", "submit_scharnagl_stage1 not importable")

    if not hasattr(mod, 'MU1_VALS'):
        return ("SKIP",
                "MU1_VALS not present (pending Stage 5e.2 migration)")

    MU1_VALS = np.asarray(getattr(mod, 'MU1_VALS'), dtype=float)
    MU2_VALS = getattr(mod, 'MU2_VALS', None)
    if MU2_VALS is None:
        return ("FAIL", "MU1_VALS present but MU2_VALS missing")
    MU2_VALS = np.asarray(MU2_VALS, dtype=float)

    if MU1_VALS.size == 0 or MU2_VALS.size == 0:
        return ("FAIL", "MU1_VALS or MU2_VALS is empty")

    if MU1_VALS.min() < -1e-9 or MU1_VALS.max() > np.pi + 1e-9:
        return ("FAIL",
                f"MU1_VALS [{MU1_VALS.min():.4f}, {MU1_VALS.max():.4f}] "
                f"outside [0, {np.pi:.4f}]")
    if MU2_VALS.min() < -1e-9 or MU2_VALS.max() > 2.0 * np.pi + 1e-9:
        return ("FAIL",
                f"MU2_VALS [{MU2_VALS.min():.4f}, {MU2_VALS.max():.4f}] "
                f"outside [0, {2*np.pi:.4f}]")

    run_task = getattr(mod, 'run_task', None)
    if run_task is None:
        return ("FAIL", "MU1_VALS present but run_task function missing")

    # Clear any cached output for the test task so run_task proceeds
    # past its "file exists" check to subprocess.run.
    _test_out = os.path.join(
        ROOT, 'data', 'scharnagl_pts_1d_closed',
        'scharnagl_pt_N16_alpha1.0_mu10.0000_mu20.0000_tk0.0000_pk0.0000.npy',
    )
    if os.path.exists(_test_out):
        try:
            os.remove(_test_out)
        except OSError:
            pass

    captured = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return _FakeCompleted()

    with patch('subprocess.run', fake_run):
        try:
            run_task('1d_closed', 't000', 0, 0)
        except Exception as exc:
            return ("FAIL",
                    f"run_task('1d_closed', 't000', 0, 0) raised "
                    f"{type(exc).__name__}: {exc}")

    if not captured:
        return ("FAIL", "run_task did not invoke subprocess.run")
    cmd = captured[0]
    if '--mu1' in cmd and '--mu2' in cmd:
        return ("PASS",
                f"MU1_VALS range OK; run_task dispatches "
                f"--mu1/--mu2")
    if '--mu' in cmd and '--nu' in cmd:
        return ("FAIL",
                "run_task emits --mu/--nu despite MU1_VALS being "
                "present (migration incomplete)")
    return ("FAIL",
            f"run_task command has no mu arguments: {cmd}")


# ======================================================================
# Layer 6 -- forward sweep + asymmetric SLURM driver
# ======================================================================
def layer_6_forward_sweep():
    # --------------------------------------------------------------
    # Part A -- forward plotter formula
    # --------------------------------------------------------------
    mod1 = _import_by_path(
        'plot_forward_spectroscopy',
        os.path.join(ROOT, 'notebooks', 'plot_forward_spectroscopy.py'),
    )
    mod2 = _import_by_path(
        'plot_forward_spectroscopy_2d',
        os.path.join(ROOT, 'notebooks', 'plot_forward_spectroscopy_2d.py'),
    )
    if mod1 is None or mod2 is None:
        return ("SKIP",
                "Part A: forward plotters not importable")

    src1 = inspect.getsource(mod1)
    src2 = inspect.getsource(mod2)
    marker = 'mu1 = 4.0 * 1.0 * t / N'
    xlabel_marker = r'$\mu_1$'
    for name, src in [('plot_forward_spectroscopy', src1),
                      ('plot_forward_spectroscopy_2d', src2)]:
        if marker not in src:
            return ("FAIL", f"Part A: {name} missing '{marker}'")
        if xlabel_marker not in src:
            return ("FAIL",
                    f"Part A: {name} xlabel does not contain '$\\mu_1$'")

    # --------------------------------------------------------------
    # Part B -- main_powerlaw.py actual run
    # --------------------------------------------------------------
    worker = os.path.join(ROOT, 'scripts', 'main_powerlaw.py')
    if not os.path.exists(worker):
        return ("FAIL", "Part B: main_powerlaw.py not found")

    N = 8
    alpha = 3.0
    J = 1.0
    t_max = 0.1
    dt = 0.05
    chi = 8

    out_npy = os.path.join(ROOT, 'data',
                           f'powerlaw_N{N}_chi{chi}_alpha{alpha}.npy')
    chk_h5 = os.path.join(ROOT, 'data',
                          f'checkpoint_powerlaw_N{N}_chi{chi}_alpha{alpha}.h5')

    for p in (out_npy, chk_h5):
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    cmd = [sys.executable, worker,
           '--N', str(N), '--chi_max', str(chi),
           '--alpha', str(alpha), '--J', str(J),
           '--t_max', str(t_max), '--dt', str(dt)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return ("FAIL", "Part B: main_powerlaw.py timed out")
    finally:
        pass

    if r.returncode != 0:
        for p in (out_npy, chk_h5):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        return ("FAIL",
                f"Part B: main_powerlaw.py rc={r.returncode}: "
                f"{r.stderr[-200:].strip()}")

    if not os.path.exists(out_npy):
        return ("FAIL",
                f"Part B: output {os.path.basename(out_npy)} missing")

    try:
        d = np.load(out_npy, allow_pickle=True).item()
    except Exception as exc:
        for p in (out_npy, chk_h5):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        return ("FAIL", f"Part B: failed to load output: {exc}")

    schema_fail = None
    meta = d.get('metadata', {})
    if meta.get('N') != N:
        schema_fail = f"metadata N={meta.get('N')} expected {N}"
    elif abs(float(meta.get('alpha', -1.0)) - alpha) > 1e-9:
        schema_fail = f"metadata alpha={meta.get('alpha')} expected {alpha}"
    elif abs(float(meta.get('J', -1.0)) - J) > 1e-9:
        schema_fail = f"metadata J={meta.get('J')} expected {J}"

    if schema_fail is None:
        for key in ('time', 'squeezing', 'qfi', 'entropy', 'trunc_error'):
            if key not in d:
                schema_fail = f"missing top-level '{key}'"
                break

    if schema_fail is None:
        time = np.asarray(d['time'])
        if time.size < 2:
            schema_fail = f"time array has {time.size} points, expected >= 2"
        elif abs(time[0]) > 1e-9:
            schema_fail = f"time[0]={time[0]} expected 0"
        elif abs(time[-1] - t_max) > dt:
            schema_fail = (f"time[-1]={time[-1]} expected "
                           f"~{t_max} (dt={dt})")

    if schema_fail is None:
        time = np.asarray(d['time'])
        for name in ('squeezing', 'qfi', 'entropy', 'trunc_error'):
            arr = np.asarray(d[name])
            if arr.size != time.size:
                schema_fail = (f"'{name}' length {arr.size} != "
                               f"time {time.size}")
                break

    if schema_fail is None:
        mu1_endpoint = 4.0 * J * time[-1] / N
        mu1_expected = 4.0 * J * t_max / N
        if abs(mu1_endpoint - mu1_expected) > 1e-6:
            schema_fail = (f"mu1 endpoint {mu1_endpoint:.6f} != "
                           f"expected {mu1_expected:.6f}")

    for p in (out_npy, chk_h5):
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    if schema_fail is not None:
        return ("FAIL", f"Part B: {schema_fail}")

    # --------------------------------------------------------------
    # Part C -- asymmetric SLURM driver presence
    # --------------------------------------------------------------
    asym_driver = None
    for candidate in ('submit_asymmetric.py', 'submit_asym.py',
                      'submit_asymmetric_echo.py'):
        p = os.path.join(ROOT, 'scripts', 'maintenance', candidate)
        if os.path.exists(p):
            asym_driver = p
            break

    if asym_driver is None:
        return ("SKIP",
                "Parts A+B PASS; Part C: asymmetric SLURM driver "
                "pending Stage 5f")
    return ("PASS",
            f"Parts A+B PASS; Part C: asymmetric driver at "
            f"{os.path.basename(asym_driver)}")


# ======================================================================
# Layer 7 -- cross-convention physics
# ======================================================================
def layer_7_cross_convention():
    from src.selectors import worker_command

    N = 8
    alpha = 3.0
    chi = 8
    dt = 0.05

    with tempfile.TemporaryDirectory() as tmp_new, \
         tempfile.TemporaryDirectory() as tmp_old:

        cmd_new = worker_command('1d_closed', chi, N=N, alpha=alpha,
                                 convention='mu1mu2',
                                 mu1=np.pi / 2, mu2=np.pi / 2,
                                 dt_max=dt)
        cmd_new += ['--output_dir', tmp_new]

        cmd_old = worker_command('1d_closed', chi, N=N, alpha=alpha,
                                 convention='mu_nu',
                                 mu=2.0 * np.pi, nu=0.0,
                                 dt_max=dt)
        cmd_old += ['--output_dir', tmp_old]

        r_new = subprocess.run(cmd_new, capture_output=True,
                               text=True, timeout=180)
        if r_new.returncode != 0:
            return ("FAIL",
                    f"mu1mu2 worker failed rc={r_new.returncode}: "
                    f"{r_new.stderr[-200:].strip()}")
        r_old = subprocess.run(cmd_old, capture_output=True,
                               text=True, timeout=180)
        if r_old.returncode != 0:
            return ("FAIL",
                    f"mu_nu worker failed rc={r_old.returncode}: "
                    f"{r_old.stderr[-200:].strip()}")

        files_new = glob.glob(os.path.join(tmp_new, '*.npy'))
        files_old = glob.glob(os.path.join(tmp_old, '*.npy'))
        if not files_new:
            return ("FAIL", "mu1mu2 worker produced no .npy output")
        if not files_old:
            return ("FAIL", "mu_nu worker produced no .npy output")

        snr_new = float(np.load(files_new[0], allow_pickle=True).item()['snr'])
        snr_old = float(np.load(files_old[0], allow_pickle=True).item()['snr'])

    rel = abs(snr_new - snr_old) / max(abs(snr_new), 1e-12)
    if rel > 1e-5:
        return ("FAIL",
                f"SNR mismatch: mu1mu2={snr_new:.8f} vs "
                f"mu_nu={snr_old:.8f} (rel={rel:.2e})")
    return ("PASS",
            f"cross-convention SNR mu1mu2={snr_new:.8f} == "
            f"mu_nu={snr_old:.8f} (rel={rel:.2e})")


# ======================================================================
# Main
# ======================================================================
LAYERS = [
    ("1: config invariants",           layer_1_config),
    ("2: worker_command dispatch",     layer_2_worker_command),
    ("3: native_output_path suffix",   layer_3_native_output_path),
    ("4: submit_production",           layer_4_submit_production),
    ("5: submit_scharnagl_stage1",     layer_5_submit_scharnagl),
    ("6: forward sweep + asym driver", layer_6_forward_sweep),
    ("7: cross-convention physics",    layer_7_cross_convention),
]


def main():
    parser = argparse.ArgumentParser(
        description="Structural test: pipeline dispatches to "
                    "(mu1, mu2) correctly.",
    )
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip Layer 7 (runs two workers, ~15s).')
    parser.add_argument('--stop-on-fail', action='store_true',
                        help='Halt at the first FAIL/ERROR.')
    args = parser.parse_args()

    print("=" * 72)
    print("test_production_ranges.py -- 7-layer structural test")
    print("=" * 72)

    n_pass = n_fail = n_skip = n_err = 0
    for name, fn in LAYERS:
        if args.skip_slow and name.startswith('7:'):
            print(f"[ SKIP] {name}: --skip-slow")
            n_skip += 1
            continue
        try:
            status, detail = fn()
        except Exception as exc:
            status, detail = "ERROR", f"{type(exc).__name__}: {exc}"

        print(f"[{status:>5}] {name}: {detail}")

        if status == "PASS":
            n_pass += 1
        elif status == "FAIL":
            n_fail += 1
            if args.stop_on_fail:
                break
        elif status == "SKIP":
            n_skip += 1
        else:
            n_err += 1
            if args.stop_on_fail:
                break

    print()
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, "
          f"{n_skip} SKIP, {n_err} ERROR")
    sys.exit(0 if (n_fail == 0 and n_err == 0) else 1)


if __name__ == '__main__':
    main()