"""
scripts/checks/check_asymmetric_echo.py

Validates the four asymmetric echo workers against the
corresponding symmetric echo workers at the exact-echo point.

By the code-native mapping
    t_plus  = mu / 2
    t_minus = (mu - nu) / 2
the pair (t_plus = t_minus = pi/4) corresponds to the symmetric
worker evaluated at (mu, nu) = (pi/2, 0). This is the exact-echo
protocol.

--method
--------
finite_diff      (default) the production path; the check uses the
                 same default in both workers, and the two must
                 agree to machine precision because they use the
                 same seed and n_traj.
exact_derivative optional passthrough. Requires src/dM.py
                 (the documentation). The asymmetric worker's exact-M path
                 passes nu_minus_mu = -2 * t_minus, matching the
                 sign convention documented in the documentation.

The check runs:
  1D closed : asymmetric_echo_1d_closed.py  vs powerlaw_echo_worker.py
  1D open   : asymmetric_echo_1d_open.py    vs tjm_worker.py --mode echo
  2D closed : asymmetric_echo_2d_closed.py  vs powerlaw_cylinder_worker.py
  2D open   : asymmetric_echo_2d_open.py    vs tjm_worker.py --mode echo
              (with --Lx --Ly)

Tolerances: 1e-4 for closed and gamma=0; 1e-3 for open at
(gamma1, gamma2) = (1e-3, 1e-3). For the open cases the same
--seed and --n_traj are passed to both workers so that their
random-number sequences match and the difference is not dominated
by Monte Carlo noise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
for p in (ROOT_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def run(cmd, label):
    print(f"\n[run] {label}")
    print("     " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_snr(path):
    d = np.load(path, allow_pickle=True).item()
    return float(d['snr'])


def compare(snr_asym, snr_sym, tol, label):
    rel_err = abs(snr_asym - snr_sym) / max(abs(snr_sym), 1e-30)
    status = "PASS" if rel_err < tol else "FAIL"
    print(f"\n[{label}]")
    print(f"  asymmetric SNR = {snr_asym:.10f}")
    print(f"  symmetric SNR  = {snr_sym:.10f}")
    print(f"  relative error = {rel_err:.3e}  (tol {tol:.0e})")
    print(f"  -> {status}")
    return rel_err < tol


def check_1d_closed(args):
    asym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_1d', 'asymmetric_echo_1d_closed.py')
    sym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_1d', 'powerlaw_echo_worker.py')

    mu_sym = 2.0 * args.t_plus
    nu_sym = mu_sym - 2.0 * args.t_minus

    asym_out = os.path.join(
        ROOT_DIR, 'data', 'echo_pts_1d_closed_asym',
        f"asym_pt_N{args.N}_alpha{args.alpha}"
        f"_tplus{args.t_plus:.4f}_tminus{args.t_minus:.4f}.npy",
    )
    sym_out = os.path.join(
        ROOT_DIR, 'data', 'powerlaw_echo_pts',
        f"plaw_echo_N{args.N}_chi{args.chi_max}_alpha{args.alpha}"
        f"_mu{mu_sym:.4f}_nu{nu_sym:.4f}.npy",
    )
    for f in (asym_out, sym_out):
        if os.path.exists(f):
            os.remove(f)

    run([sys.executable, asym_worker,
         "--N", str(args.N), "--alpha", str(args.alpha),
         "--t_plus", f"{args.t_plus:.6f}",
         "--t_minus", f"{args.t_minus:.6f}",
         "--chi_max", str(args.chi_max),
         "--dt_max", str(args.dt_max),
         "--method", args.method],
        "asymmetric 1D closed")
    run([sys.executable, sym_worker,
         "--N", str(args.N), "--chi_max", str(args.chi_max),
         "--alpha", str(args.alpha),
         "--mu", f"{mu_sym:.6f}", "--nu", f"{nu_sym:.6f}",
         "--dt_max", str(args.dt_max)],
        "symmetric 1D closed")

    return compare(load_snr(asym_out), load_snr(sym_out), 1e-4,
                   "1D closed: asym vs sym")


def check_1d_open(args):
    asym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'open_1d', 'asymmetric_echo_1d_open.py')
    sym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'shared', 'tjm_worker.py')

    mu_sym = 2.0 * args.t_plus
    nu_sym = mu_sym - 2.0 * args.t_minus

    asym_out = os.path.join(
        ROOT_DIR, 'data', 'echo_pts_1d_open_asym',
        f"asym_pt_N{args.N}_alpha{args.alpha}"
        f"_tplus{args.t_plus:.4f}_tminus{args.t_minus:.4f}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
        f"_ntraj{args.n_traj}.npy",
    )
    sym_out = os.path.join(
        ROOT_DIR, 'data', 'tjm_pts',
        f"tjm_N{args.N}_chi{args.chi_max}_alpha{args.alpha}"
        f"_mu{mu_sym:.4f}_nu{nu_sym:.4f}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
        f"_ntraj{args.n_traj}.npy",
    )
    for f in (asym_out, sym_out):
        if os.path.exists(f):
            os.remove(f)

    run([sys.executable, asym_worker,
         "--N", str(args.N), "--alpha", str(args.alpha),
         "--t_plus", f"{args.t_plus:.6f}",
         "--t_minus", f"{args.t_minus:.6f}",
         "--chi_max", str(args.chi_max),
         "--dt_max", str(args.dt_max),
         "--gamma1", str(args.gamma1),
         "--gamma2", str(args.gamma2),
         "--n_traj", str(args.n_traj),
         "--seed", str(args.seed),
         "--method", args.method],
        "asymmetric 1D open")
    run([sys.executable, sym_worker,
         "--mode", "echo",
         "--N", str(args.N), "--chi_max", str(args.chi_max),
         "--alpha", str(args.alpha),
         "--mu", f"{mu_sym:.6f}", "--nu", f"{nu_sym:.6f}",
         "--gamma1", str(args.gamma1),
         "--gamma2", str(args.gamma2),
         "--n_traj_per_rank", str(args.n_traj),
         "--seed", str(args.seed),
         "--dt_max", str(args.dt_max)],
        "symmetric 1D open (TJM)")

    tol = 1e-3 if (args.gamma1 > 0 or args.gamma2 > 0) else 1e-4
    return compare(load_snr(asym_out), load_snr(sym_out), tol,
                   "1D open: asym vs sym")


def check_2d_closed(args):
    asym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_2d', 'asymmetric_echo_2d_closed.py')
    sym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_2d', 'powerlaw_cylinder_worker.py')

    mu_sym = 2.0 * args.t_plus
    nu_sym = mu_sym - 2.0 * args.t_minus

    asym_out = os.path.join(
        ROOT_DIR, 'data', 'echo_pts_2d_closed_asym',
        f"asym_pt_Lx{args.Lx}_Ly{args.Ly}_alpha{args.alpha}"
        f"_tplus{args.t_plus:.4f}_tminus{args.t_minus:.4f}.npy",
    )
    sym_out = os.path.join(
        ROOT_DIR, 'data', 'powerlaw_cylinder_pts',
        f"plaw_cyl_Lx{args.Lx}_Ly{args.Ly}"
        f"_chi{args.chi_max}_alpha{args.alpha}"
        f"_mu{mu_sym:.4f}_nu{nu_sym:.4f}.npy",
    )
    for f in (asym_out, sym_out):
        if os.path.exists(f):
            os.remove(f)

    run([sys.executable, asym_worker,
         "--Lx", str(args.Lx), "--Ly", str(args.Ly),
         "--alpha", str(args.alpha),
         "--t_plus", f"{args.t_plus:.6f}",
         "--t_minus", f"{args.t_minus:.6f}",
         "--chi_max", str(args.chi_max),
         "--dt_max", str(args.dt_max),
         "--method", args.method],
        "asymmetric 2D closed")
    run([sys.executable, sym_worker,
         "--Lx", str(args.Lx), "--Ly", str(args.Ly),
         "--chi_max", str(args.chi_max),
         "--alpha", str(args.alpha),
         "--mu", f"{mu_sym:.6f}", "--nu", f"{nu_sym:.6f}",
         "--dt_max", str(args.dt_max)],
        "symmetric 2D closed")

    return compare(load_snr(asym_out), load_snr(sym_out), 1e-4,
                   "2D closed: asym vs sym")


def check_2d_open(args):
    asym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'open_2d', 'asymmetric_echo_2d_open.py')
    sym_worker = os.path.join(SCRIPTS_DIR, 'workers', 'shared', 'tjm_worker.py')

    mu_sym = 2.0 * args.t_plus
    nu_sym = mu_sym - 2.0 * args.t_minus

    asym_out = os.path.join(
        ROOT_DIR, 'data', 'echo_pts_2d_open_asym',
        f"asym_pt_Lx{args.Lx}_Ly{args.Ly}_alpha{args.alpha}"
        f"_tplus{args.t_plus:.4f}_tminus{args.t_minus:.4f}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
        f"_ntraj{args.n_traj}.npy",
    )
    sym_out = os.path.join(
        ROOT_DIR, 'data', 'tjm_pts',
        f"tjm_Lx{args.Lx}_Ly{args.Ly}_chi{args.chi_max}"
        f"_alpha{args.alpha}"
        f"_mu{mu_sym:.4f}_nu{nu_sym:.4f}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
        f"_ntraj{args.n_traj}.npy",
    )
    for f in (asym_out, sym_out):
        if os.path.exists(f):
            os.remove(f)

    run([sys.executable, asym_worker,
         "--Lx", str(args.Lx), "--Ly", str(args.Ly),
         "--alpha", str(args.alpha),
         "--t_plus", f"{args.t_plus:.6f}",
         "--t_minus", f"{args.t_minus:.6f}",
         "--chi_max", str(args.chi_max),
         "--dt_max", str(args.dt_max),
         "--gamma1", str(args.gamma1),
         "--gamma2", str(args.gamma2),
         "--n_traj", str(args.n_traj),
         "--seed", str(args.seed),
         "--method", args.method],
        "asymmetric 2D open")
    run([sys.executable, sym_worker,
         "--mode", "echo",
         "--Lx", str(args.Lx), "--Ly", str(args.Ly),
         "--chi_max", str(args.chi_max),
         "--alpha", str(args.alpha),
         "--mu", f"{mu_sym:.6f}", "--nu", f"{nu_sym:.6f}",
         "--gamma1", str(args.gamma1),
         "--gamma2", str(args.gamma2),
         "--n_traj_per_rank", str(args.n_traj),
         "--seed", str(args.seed),
         "--dt_max", str(args.dt_max)],
        "symmetric 2D open (TJM)")

    tol = 1e-3 if (args.gamma1 > 0 or args.gamma2 > 0) else 1e-4
    return compare(load_snr(asym_out), load_snr(sym_out), tol,
                   "2D open: asym vs sym")


def main():
    parser = argparse.ArgumentParser()
    # Geometry
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--Lx", type=int, default=None)
    parser.add_argument("--Ly", type=int, default=None)
    # Physics
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--t_plus", type=float, default=np.pi / 4)
    parser.add_argument("--t_minus", type=float, default=np.pi / 4)
    # Case
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    # Numerical
    parser.add_argument("--chi_max", type=int, default=16)
    parser.add_argument("--n_traj", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--method",
                        choices=['finite_diff', 'exact_derivative'],
                        default='finite_diff',
                        help="Pass-through to the asymmetric workers. "
                             "'finite_diff' is the default.")
    args = parser.parse_args()

    have_N = args.N is not None
    have_L = (args.Lx is not None) or (args.Ly is not None)

    if have_N and have_L:
        raise SystemExit("Provide either --N or --Lx/--Ly, not both.")
    if not have_N and not have_L:
        raise SystemExit("Provide --N (1D) or --Lx --Ly (2D).")
    if have_L and (args.Lx is None or args.Ly is None):
        raise SystemExit("Both --Lx and --Ly must be provided for 2D.")

    is_2d = have_L

    if not is_2d:
        if args.open:
            ok = check_1d_open(args)
        else:
            ok = check_1d_closed(args)
    else:
        if args.open:
            ok = check_2d_open(args)
        else:
            ok = check_2d_closed(args)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()