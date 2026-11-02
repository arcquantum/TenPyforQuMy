#!/usr/bin/env python
"""
scripts/checks/check_qfi_all_workers.py
End-to-end QFI sanity for all 11 production workers.

Each worker is invoked fresh at (mu=0, nu=0) for symmetric cases, or
(t_plus=t_minus=pi/4) for asymmetric cases. After the worker runs,
the check reads only files matching the check's own parameters in
the worker's default directory. No cross-worker contamination.
"""
import glob
import os
import subprocess
import sys
import numpy as np

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

WORKERS = [
    ('1d_closed', 'powerlaw_echo_worker.py',
     ['--N', '8', '--chi_max', '16', '--alpha', '1.0'], False, False),
    ('2d_closed', 'powerlaw_cylinder_worker.py',
     ['--Lx', '2', '--Ly', '4', '--chi_max', '16', '--alpha', '1.0'],
     False, False),
    ('1d_closed_asym', 'asymmetric_echo_1d_closed.py',
     ['--N', '8', '--chi_max', '16', '--alpha', '1.0'], False, True),
    ('2d_closed_asym', 'asymmetric_echo_2d_closed.py',
     ['--Lx', '2', '--Ly', '4', '--chi_max', '16', '--alpha', '1.0'],
     False, True),
    ('1d_closed_sch', 'scharnagl_echo_1d_closed.py',
     ['--N', '8', '--chi_max', '16', '--alpha', '1.0',
      '--theta_k', '0.0', '--phi_k', '0.0'], False, False),
    ('2d_closed_sch', 'scharnagl_echo_2d_closed.py',
     ['--Lx', '2', '--Ly', '4', '--chi_max', '16', '--alpha', '1.0',
      '--theta_k', '0.0', '--phi_k', '0.0'], False, False),
    ('1d_open', 'tjm_worker.py',
     ['--mode', 'echo', '--N', '8', '--chi_max', '16',
      '--alpha', '1.0', '--gamma1', '1e-3', '--gamma2', '1e-3',
      '--n_traj_per_rank', '10'], True, False),
    ('1d_open_asym', 'asymmetric_echo_1d_open.py',
     ['--N', '8', '--chi_max', '16', '--alpha', '1.0',
      '--gamma1', '1e-3', '--gamma2', '1e-3', '--n_traj', '10'],
     True, True),
    ('2d_open_asym', 'asymmetric_echo_2d_open.py',
     ['--Lx', '2', '--Ly', '4', '--chi_max', '16', '--alpha', '1.0',
      '--gamma1', '1e-3', '--gamma2', '1e-3', '--n_traj', '10'],
     True, True),
    ('1d_open_sch', 'scharnagl_echo_1d_open.py',
     ['--N', '8', '--chi_max', '16', '--alpha', '1.0',
      '--theta_k', '0.0', '--phi_k', '0.0',
      '--gamma1', '1e-3', '--gamma2', '1e-3', '--n_traj', '10'],
     True, False),
    ('2d_open_sch', 'scharnagl_echo_2d_open.py',
     ['--Lx', '2', '--Ly', '4', '--chi_max', '16', '--alpha', '1.0',
      '--theta_k', '0.0', '--phi_k', '0.0',
      '--gamma1', '1e-3', '--gamma2', '1e-3', '--n_traj', '10'],
     True, False),
]

SCRIPT_TO_DIR = {
    'powerlaw_echo_worker.py': 'data/powerlaw_echo_pts',
    'powerlaw_cylinder_worker.py': 'data/powerlaw_cylinder_pts',
    'asymmetric_echo_1d_closed.py': 'data/echo_pts_1d_closed_asym',
    'asymmetric_echo_2d_closed.py': 'data/echo_pts_2d_closed_asym',
    'scharnagl_echo_1d_closed.py': 'data/scharnagl_pts_1d_closed',
    'scharnagl_echo_2d_closed.py': 'data/scharnagl_pts_2d_closed',
    'tjm_worker.py': 'data/tjm_pts',
    'asymmetric_echo_1d_open.py': 'data/echo_pts_1d_open_asym',
    'asymmetric_echo_2d_open.py': 'data/echo_pts_2d_open_asym',
    'scharnagl_echo_1d_open.py': 'data/scharnagl_pts_1d_open',
    'scharnagl_echo_2d_open.py': 'data/scharnagl_pts_2d_open',
}


SCRIPT_TO_SUBDIR = {
    'powerlaw_echo_worker.py': 'closed_1d',
    'asymmetric_echo_1d_closed.py': 'closed_1d',
    'scharnagl_echo_1d_closed.py': 'closed_1d',
    'exact_echo_worker_1d.py': 'closed_1d',
    'powerlaw_cylinder_worker.py': 'closed_2d',
    'asymmetric_echo_2d_closed.py': 'closed_2d',
    'scharnagl_echo_2d_closed.py': 'closed_2d',
    'exact_echo_worker_2d.py': 'closed_2d',
    'asymmetric_echo_1d_open.py': 'open_1d',
    'scharnagl_echo_1d_open.py': 'open_1d',
    'exact_lindblad_worker_1d.py': 'open_1d',
    'banuls_worker.py': 'open_1d',
    'simple_mpo_worker.py': 'open_1d',
    'asymmetric_echo_2d_open.py': 'open_2d',
    'scharnagl_echo_2d_open.py': 'open_2d',
    'exact_lindblad_worker_2d.py': 'open_2d',
    'tjm_worker.py': 'shared',
    'main_powerlaw.py': 'shared',
    'main_fss.py': 'shared',
    'echo_worker.py': 'shared',
}


def _params_dict(extra_args):
    return dict(zip(extra_args[::2], extra_args[1::2]))


def _matches_check_params(content, N, Lx, Ly, chi, alpha):
    if N is not None and content.get('N') != int(N):
        return False
    if Lx is not None and content.get('Lx') != int(Lx):
        return False
    if Ly is not None and content.get('Ly') != int(Ly):
        return False
    if (content.get('chi_max') is not None
            and int(content.get('chi_max')) != int(chi)):
        return False
    if (content.get('alpha') is not None
            and abs(float(content.get('alpha')) - float(alpha)) > 1e-6):
        return False
    return True


def _delete_check_output(script, extra_args, uses_tplus):
    params = _params_dict(extra_args)
    N = params.get('--N')
    Lx = params.get('--Lx')
    Ly = params.get('--Ly')
    chi = params.get('--chi_max')
    alpha = params.get('--alpha')
    d = os.path.join(ROOT_DIR, SCRIPT_TO_DIR[script])
    if not os.path.isdir(d):
        return
    if uses_tplus:
        glob_pattern = '*tplus0.7854_tminus0.7854*.npy'
    else:
        glob_pattern = '*mu0.0000_nu0.0000*.npy'
    for f in glob.glob(os.path.join(d, glob_pattern)):
        try:
            content = np.load(f, allow_pickle=True).item()
        except Exception:
            continue
        if _matches_check_params(content, N, Lx, Ly, chi, alpha):
            try:
                os.remove(f)
            except OSError:
                pass


def main():
    n_pass = n_fail = 0
    for name, script, extra_args, is_open, uses_tplus in WORKERS:
        print(f"\n=== {name} ===")
        if not os.path.exists(os.path.join(ROOT_DIR, 'scripts', 'workers', SCRIPT_TO_SUBDIR[script], script)):
            print("  SKIP (missing)"); continue

        _delete_check_output(script, extra_args, uses_tplus)

        cmd = [sys.executable,
               os.path.join(ROOT_DIR, 'scripts', 'workers', SCRIPT_TO_SUBDIR[script], script),
               *extra_args]
        if uses_tplus:
            cmd += ['--t_plus', '0.785398', '--t_minus', '0.785398']
        else:
            cmd += ['--mu', '0.000000', '--nu', '0.000000']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL (crashed: {r.stderr[-250:]})")
            n_fail += 1; continue

        params = _params_dict(extra_args)
        N_ck = params.get('--N')
        Lx_ck = params.get('--Lx')
        Ly_ck = params.get('--Ly')
        chi_ck = params.get('--chi_max')
        alpha_ck = params.get('--alpha')

        worker_dir = os.path.join(ROOT_DIR, SCRIPT_TO_DIR[script])
        if uses_tplus:
            pattern = '*tplus0.7854_tminus0.7854*.npy'
        else:
            pattern = '*mu0.0000_nu0.0000*.npy'
        filtered = []
        for f in glob.glob(os.path.join(worker_dir, pattern)):
            try:
                content = np.load(f, allow_pickle=True).item()
            except Exception:
                continue
            if _matches_check_params(content, N_ck, Lx_ck, Ly_ck,
                                     chi_ck, alpha_ck):
                filtered.append(f)
        if not filtered:
            print(f"  FAIL (no output in {SCRIPT_TO_DIR[script]})")
            n_fail += 1; continue
        fpath = max(filtered, key=os.path.getmtime)
        d = np.load(fpath, allow_pickle=True).item()
        # Bug 1b guard: reject stale pre-QFI files. If the loaded
        # file lacks the expected QFI key, treat as FAIL rather than
        # letting the missing-key error propagate.
        expected_key = 'qfi_avg' if is_open else 'qfi'
        if expected_key not in d:
            print(f"  FAIL (stale file {os.path.basename(fpath)}: "
                  f"missing '{expected_key}'; delete it and re-run)")
            n_fail += 1
            continue


        N = d.get('N') or (int(d.get('Lx', 0)) * int(d.get('Ly', 0)))
        snr = float(d['snr'])

        if is_open:
            if 'qfi_var' not in d or 'qfi_avg' not in d:
                print(f"  FAIL (missing qfi_var/qfi_avg in "
                      f"{os.path.basename(fpath)})")
                n_fail += 1; continue
            qv, qa = float(d['qfi_var']), float(d['qfi_avg'])
            ok = (abs(qa - N) < 0.1 * N
                  and (uses_tplus or snr ** 2 <= qa + 1e-6)
                  and qv >= 0 and qa >= 0)
            print(f"  N={N} snr={snr:.4f} qv={qv:.4f} qa={qa:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")
            n_pass += ok; n_fail += (not ok)
        else:
            if 'qfi' not in d:
                print(f"  FAIL (missing qfi in "
                      f"{os.path.basename(fpath)})")
                n_fail += 1; continue
            qf = float(d['qfi'])
            ok = (abs(qf - N) < 0.1 * N
                  and (uses_tplus or snr ** 2 <= qf + 1e-6)
                  and qf >= 0)
            print(f"  N={N} snr={snr:.4f} qf={qf:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")
            n_pass += ok; n_fail += (not ok)

    print(f"\n{'=' * 50}\nSummary: {n_pass} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
