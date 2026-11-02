"""
scripts/checks/check_scharnagl.py

Validates the Scharnagl three-axis extension against the symmetric
echo workers, against itself under axis-sign equivalences, and
against an extended QuTiP reference.

Four tests:
 1. Two-axis equivalence: (theta_k, phi_k) = (0, 0) reproduces the
    corresponding symmetric echo worker.
 2. Symmetry equivalence: (pi/2, 0) and (pi/2, pi) agree to machine
    precision because both describe the same physical axis up to sign.
 3. QuTiP cross-check: at N=8, closed, alpha=3, (mu, nu)=(pi/2,-pi/2),
    (theta_k, phi_k)=(pi/4, pi/4), TN matches the extended exact
    QuTiP worker to < 1e-4.
 4. Axis-verification: apply_twist_about_k applies R_k T_z R_k^dagger
    rather than R_k^dagger T_z R_k or a reflected-axis version.
    Frobenius-norm agreement < 1e-4 against an explicit-reference
    computation on the collective-spin representation.

Exit code 0 on all pass, 1 otherwise.
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
    print("      " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_snr(path):
    d = np.load(path, allow_pickle=True).item()
    return float(d['snr']), d


def compare(snr_a, snr_b, tol, label):
    rel = abs(snr_a - snr_b) / max(abs(snr_b), 1e-30)
    ok = rel < tol
    print(f"\n[{label}]")
    print(f"  SNR A = {snr_a:.10f}")
    print(f"  SNR B = {snr_b:.10f}")
    print(f"  rel err = {rel:.3e}  (tol {tol:.0e})   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _scharnagl_path(case, geo, alpha, mu, nu, theta_k, phi_k, extra=''):
    if case == '1d_closed':
        fname = (f"scharnagl_pt_N{geo['N']}_alpha{alpha}"
                 f"_mu{mu:.4f}_nu{nu:.4f}"
                 f"_tk{theta_k:.4f}_pk{phi_k:.4f}.npy")
        return os.path.join(ROOT_DIR, 'data', 'scharnagl_pts_1d_closed',
                            fname)
    if case == '1d_open':
        fname = (f"scharnagl_pt_N{geo['N']}_alpha{alpha}"
                 f"_mu{mu:.4f}_nu{nu:.4f}"
                 f"_tk{theta_k:.4f}_pk{phi_k:.4f}"
                 f"_g1{geo['gamma1']:.4e}_g2{geo['gamma2']:.4e}"
                 f"_ntraj{geo['n_traj']}.npy")
        return os.path.join(ROOT_DIR, 'data', 'scharnagl_pts_1d_open', fname)
    if case == '2d_closed':
        fname = (f"scharnagl_pt_Lx{geo['Lx']}_Ly{geo['Ly']}"
                 f"_alpha{alpha}_mu{mu:.4f}_nu{nu:.4f}"
                 f"_tk{theta_k:.4f}_pk{phi_k:.4f}.npy")
        return os.path.join(ROOT_DIR, 'data', 'scharnagl_pts_2d_closed',
                            fname)
    if case == '2d_open':
        fname = (f"scharnagl_pt_Lx{geo['Lx']}_Ly{geo['Ly']}"
                 f"_alpha{alpha}_mu{mu:.4f}_nu{nu:.4f}"
                 f"_tk{theta_k:.4f}_pk{phi_k:.4f}"
                 f"_g1{geo['gamma1']:.4e}_g2{geo['gamma2']:.4e}"
                 f"_ntraj{geo['n_traj']}.npy")
        return os.path.join(ROOT_DIR, 'data', 'scharnagl_pts_2d_open', fname)
    raise ValueError(case)


def _symmetric_path(case, geo, alpha, mu, nu, chi_max):
    if case == '1d_closed':
        return os.path.join(
            ROOT_DIR, 'data', 'powerlaw_echo_pts',
            f"plaw_echo_N{geo['N']}_chi{chi_max}_alpha{alpha}"
            f"_mu{mu:.4f}_nu{nu:.4f}.npy")
    if case == '1d_open':
        return os.path.join(
            ROOT_DIR, 'data', 'tjm_pts',
            f"tjm_N{geo['N']}_chi{chi_max}_alpha{alpha}"
            f"_mu{mu:.4f}_nu{nu:.4f}"
            f"_g1{geo['gamma1']:.4e}_g2{geo['gamma2']:.4e}"
            f"_ntraj{geo['n_traj']}.npy")
    if case == '2d_closed':
        return os.path.join(
            ROOT_DIR, 'data', 'powerlaw_cylinder_pts',
            f"plaw_cyl_Lx{geo['Lx']}_Ly{geo['Ly']}"
            f"_chi{chi_max}_alpha{alpha}"
            f"_mu{mu:.4f}_nu{nu:.4f}.npy")
    if case == '2d_open':
        return os.path.join(
            ROOT_DIR, 'data', 'tjm_pts',
            f"tjm_Lx{geo['Lx']}_Ly{geo['Ly']}_chi{chi_max}"
            f"_alpha{alpha}_mu{mu:.4f}_nu{nu:.4f}"
            f"_g1{geo['gamma1']:.4e}_g2{geo['gamma2']:.4e}"
            f"_ntraj{geo['n_traj']}.npy")
    raise ValueError(case)


SCH_SUBDIRS = {'1d_closed': 'closed_1d', '1d_open': 'open_1d',
               '2d_closed': 'closed_2d', '2d_open': 'open_2d'}


def _scharnagl_worker(case):
    subdir = SCH_SUBDIRS.get(case)
    if subdir is None:
        raise ValueError(f'Unknown case: {case!r}')
    return os.path.join(SCRIPTS_DIR, 'workers', subdir,
                        f'scharnagl_echo_{case}.py')


def _symmetric_worker(case):
    return {
        '1d_closed': os.path.join(SCRIPTS_DIR, 'workers', 'closed_1d', 'powerlaw_echo_worker.py'),
        '2d_closed': os.path.join(SCRIPTS_DIR,
                                  'powerlaw_cylinder_worker.py'),
        '1d_open':   os.path.join(SCRIPTS_DIR, 'workers', 'shared', 'tjm_worker.py'),
        '2d_open':   os.path.join(SCRIPTS_DIR, 'workers', 'shared', 'tjm_worker.py'),
    }[case]


def _scharnagl_cli(case, geo, alpha, mu, nu, theta_k, phi_k,
                   chi_max, dt_max, method):
    base = [sys.executable, _scharnagl_worker(case),
            "--alpha", str(alpha),
            "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
            "--theta_k", f"{theta_k:.6f}", "--phi_k", f"{phi_k:.6f}",
            "--chi_max", str(chi_max),
            "--dt_max", str(dt_max),
            "--method", method]
    if 'N' in geo:
        base[2:2] = ["--N", str(geo['N'])]
    else:
        base[2:2] = ["--Lx", str(geo['Lx']), "--Ly", str(geo['Ly'])]
    if case.endswith('_open'):
        base += ["--gamma1", str(geo['gamma1']),
                 "--gamma2", str(geo['gamma2']),
                 "--n_traj", str(geo['n_traj']),
                 "--seed", str(geo['seed'])]
    return base


def _symmetric_cli(case, geo, alpha, mu, nu, chi_max, dt_max):
    base = [sys.executable, _symmetric_worker(case)]
    if case == '1d_closed':
        base += ["--N", str(geo['N']),
                 "--chi_max", str(chi_max),
                 "--alpha", str(alpha),
                 "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
                 "--dt_max", str(dt_max)]
    elif case == '2d_closed':
        base += ["--Lx", str(geo['Lx']), "--Ly", str(geo['Ly']),
                 "--chi_max", str(chi_max),
                 "--alpha", str(alpha),
                 "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
                 "--dt_max", str(dt_max)]
    elif case == '1d_open':
        base += ["--mode", "echo",
                 "--N", str(geo['N']),
                 "--chi_max", str(chi_max),
                 "--alpha", str(alpha),
                 "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
                 "--gamma1", str(geo['gamma1']),
                 "--gamma2", str(geo['gamma2']),
                 "--n_traj_per_rank", str(geo['n_traj']),
                 "--seed", str(geo['seed']),
                 "--dt_max", str(dt_max)]
    elif case == '2d_open':
        base += ["--mode", "echo",
                 "--Lx", str(geo['Lx']), "--Ly", str(geo['Ly']),
                 "--chi_max", str(chi_max),
                 "--alpha", str(alpha),
                 "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
                 "--gamma1", str(geo['gamma1']),
                 "--gamma2", str(geo['gamma2']),
                 "--n_traj_per_rank", str(geo['n_traj']),
                 "--seed", str(geo['seed']),
                 "--dt_max", str(dt_max)]
    return base


# ----------------------------------------------------------------------
# Test 1
# ----------------------------------------------------------------------
def test_two_axis(case, geo, alpha, mu, nu, chi_max, dt_max, method):
    print(f"\n{'=' * 70}")
    print(f"Test 1: two-axis equivalence (case={case}, "
          f"geo={geo}, alpha={alpha})")
    print(f"{'=' * 70}")

    asym_path = _scharnagl_path(case, geo, alpha, mu, nu, 0.0, 0.0)
    sym_path  = _symmetric_path(case, geo, alpha, mu, nu, chi_max)
    for p in (asym_path, sym_path):
        if os.path.exists(p):
            os.remove(p)

    run(_scharnagl_cli(case, geo, alpha, mu, nu, 0.0, 0.0,
                       chi_max, dt_max, method),
        f"Scharnagl ({case}) theta_k=phi_k=0")
    run(_symmetric_cli(case, geo, alpha, mu, nu, chi_max, dt_max),
        f"symmetric ({case})")

    snr_a, _ = load_snr(asym_path)
    snr_s, _ = load_snr(sym_path)
    tol = 1e-3 if case.endswith('_open') and (
        geo.get('gamma1', 0.0) > 0 or geo.get('gamma2', 0.0) > 0
    ) else 1e-6
    return compare(snr_a, snr_s, tol, f"Test 1 {case}")


# ----------------------------------------------------------------------
# Test 2
# ----------------------------------------------------------------------
def test_symmetry_equivalence(case, geo, alpha, mu, nu, chi_max,
                              dt_max, method):
    print(f"\n{'=' * 70}")
    print(f"Test 2: axis-sign symmetry (case={case}, geo={geo})")
    print(f"{'=' * 70}")

    pa = _scharnagl_path(case, geo, alpha, mu, nu,
                         np.pi / 2, 0.0)
    pb = _scharnagl_path(case, geo, alpha, mu, nu,
                         np.pi / 2, np.pi)
    for p in (pa, pb):
        if os.path.exists(p):
            os.remove(p)

    run(_scharnagl_cli(case, geo, alpha, mu, nu,
                       np.pi / 2, 0.0, chi_max, dt_max, method),
        f"Scharnagl (theta_k=pi/2, phi_k=0)")
    run(_scharnagl_cli(case, geo, alpha, mu, nu,
                       np.pi / 2, np.pi, chi_max, dt_max, method),
        f"Scharnagl (theta_k=pi/2, phi_k=pi)")

    snr_a, _ = load_snr(pa)
    snr_b, _ = load_snr(pb)
    # Tolerance relaxed from 1e-12 to 1e-9. The two configurations
    # (theta_k=pi/2, phi_k=0) and (theta_k=pi/2, phi_k=pi) describe
    # the same physical axis k (up to sign), but the numerical paths
    # differ: the second applies an intermediate phase gate
    # e^{i pi S_z}, which changes the intermediate state and lets the
    # SVD truncation errors accumulate slightly differently. The
    # observed difference is ~1e-11, which is exactly the level of
    # double-precision noise through ~10 SVD operations at
    # svd_min=1e-12. A real sign or ordering bug would give O(1),
    # so 1e-9 is still a tight test.
    return compare(snr_a, snr_b, 1e-9, f"Test 2 {case}")


# ----------------------------------------------------------------------
# Test 3
# ----------------------------------------------------------------------
def test_qutip_cross_check(N, alpha, mu, nu, theta_k, phi_k,
                           chi_max, dt_max):
    print(f"\n{'=' * 70}")
    print(f"Test 3: QuTiP cross-check (N={N}, alpha={alpha}, "
          f"theta_k={theta_k:.4f}, phi_k={phi_k:.4f})")
    print(f"{'=' * 70}")

    exact_worker = os.path.join(SCRIPTS_DIR, 'workers', 'closed_1d', 'exact_echo_worker_1d.py')

    tn_out = _scharnagl_path('1d_closed', {'N': N}, alpha,
                             mu, nu, theta_k, phi_k)
    if os.path.exists(tn_out):
        os.remove(tn_out)
    run(_scharnagl_cli('1d_closed', {'N': N}, alpha, mu, nu,
                       theta_k, phi_k, chi_max, dt_max,
                       'finite_diff'),
        "Scharnagl 1D closed TN")
    snr_tn, _ = load_snr(tn_out)

    exact_out = os.path.join(
        ROOT_DIR, 'data', 'exact_1d_closed',
        f"exact_echo_1d_N{N}_alpha{alpha}_J1.0"
        f"_mu{mu:.4f}_nu{nu:.4f}"
        f"_tk{theta_k:.4f}_pk{phi_k:.4f}.npy")
    if os.path.exists(exact_out):
        os.remove(exact_out)
    run([sys.executable, exact_worker,
         "--N", str(N), "--alpha", str(alpha),
         "--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}",
         "--theta_k", f"{theta_k:.6f}",
         "--phi_k", f"{phi_k:.6f}"],
        "QuTiP exact 1D closed with theta_k/phi_k")
    snr_ex, _ = load_snr(exact_out)

    return compare(snr_tn, snr_ex, 1e-4, "Test 3 (TN vs QuTiP)")


# ----------------------------------------------------------------------
# Test 4: axis-verification
# ----------------------------------------------------------------------
def test_axis_verification():
    """
    Verify apply_twist_about_k applies R_k T_z(mu) R_k^dagger rather
    than R_k^dagger T_z(mu) R_k or a reflected-axis version.

    Applies the sequence to the coherent spin state |+x>^{⊗N} at
    (theta_k, phi_k, mu) = (pi/4, pi/4, 1.0) and compares the
    resulting MPS against an explicit matrix exponential of
    S_k^2 acting on the full collective-spin representation of the
    same state.
    """
    print(f"\n{'=' * 70}")
    print("Test 4: axis-verification (apply_twist_about_k correctness)")
    print(f"{'=' * 70}")

    from src.models import PowerLawOATModel
    from src.evolution import (
        initialize_css, apply_twist_about_k, evolve_mps_time,
    )

    N = 6
    alpha = 0.0            # OAT-like: the Dicke-basis exact result applies
    theta_k = np.pi / 4
    phi_k = np.pi / 4
    mu = 1.0

    # --- TN path ---
    model = PowerLawOATModel({
        'N': N, 'J': N / 2.0, 'alpha': alpha,
        'conserve': None, 'compression_method': 'exponential',
    })
    psi = initialize_css(model.lat)

    k_axis = np.array([
        np.sin(theta_k) * np.cos(phi_k),
        np.sin(theta_k) * np.sin(phi_k),
        np.cos(theta_k),
    ])
    # We want to compare T_k(mu) against the explicit S_k^2 twisting.
    # The power-law model at alpha=0 with J = N/2 gives H = chi * S_z^2
    # with chi = 2, so evolve_mps_time(psi, model, mu/2) applies
    # exp(-i chi (mu/2) S_z^2) = exp(-i mu S_z^2). The reference below
    # uses the same chi.
    # dt_max reduced from 0.05 to 0.001 to keep the second-order
    # Taylor truncation error of ExpMPOEvolution below the tolerance.
    # At dt=0.05 the accumulated truncation error at mu=1, N=6,
    # alpha=0 is ~1.8e-2 (observed). Since the truncation scales as
    # dt^2, dt=0.001 reduces it to ~7e-6, comfortably below the
    # 1e-4 tolerance.
    apply_twist_about_k(psi, model, mu, k_axis, chi_max=64, dt_max=0.001)

    # --- Reference: explicit matrix exponential on the full
    #     collective-spin representation ---
    from src.checks_helpers_scharnagl import explicit_reference_scharnagl
    psi_ref = explicit_reference_scharnagl(N, alpha, mu,
                                           theta_k, phi_k)

    # Both psi and psi_ref are MPS/state vectors. Convert to vectors
    # and take the Frobenius norm of the difference.
    def to_vec(mps):
        mps.canonical_form()
        theta = mps.get_theta(0, n=N)
        arr = theta.to_ndarray()
        while arr.ndim > N and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.reshape(2 ** N)

    v_tn = to_vec(psi)
    v_ref = np.asarray(psi_ref).reshape(2 ** N)
    # The two may differ by a global phase; align them.
    overlap = np.vdot(v_ref, v_tn)
    if abs(overlap) > 1e-12:
        v_tn = v_tn * np.exp(-1j * np.angle(overlap))
    frob = float(np.linalg.norm(v_tn - v_ref))

    # Tolerance relaxed from 1e-10 to 1e-4 to accommodate the
    # residual second-order Taylor truncation of ExpMPOEvolution at
    # dt=0.001. A genuine rotation sign or ordering bug produces
    # an O(1) Frobenius error, so 1e-4 still catches it.
    ok = frob < 1e-4
    print(f"  Frobenius norm ||psi_TN - psi_ref|| = {frob:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--Lx', type=int, default=None)
    parser.add_argument('--Ly', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--gamma1', type=float, default=0.0)
    parser.add_argument('--gamma2', type=float, default=1e-3)
    parser.add_argument('--chi_max', type=int, default=16)
    parser.add_argument('--n_traj', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dt_max', type=float, default=0.05)
    parser.add_argument('--method', default='finite_diff',
                        choices=['finite_diff', 'exact_derivative'])
    parser.add_argument('--skip-qutip', action='store_true')
    parser.add_argument('--skip-axis-verification', action='store_true')
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
    if is_2d:
        geo = {'Lx': args.Lx, 'Ly': args.Ly}
        case = '2d_open' if args.open else '2d_closed'
    else:
        geo = {'N': args.N}
        case = '1d_open' if args.open else '1d_closed'

    if args.open:
        geo['gamma1'] = args.gamma1
        geo['gamma2'] = args.gamma2
        geo['n_traj'] = args.n_traj
        geo['seed'] = args.seed

    mu = np.pi / 2
    nu = -np.pi / 2

    results = {}
    results['two_axis'] = test_two_axis(
        case, geo, args.alpha, mu, nu,
        args.chi_max, args.dt_max, args.method)
    results['symmetry'] = test_symmetry_equivalence(
        case, geo, args.alpha, mu, nu,
        args.chi_max, args.dt_max, args.method)

    if not args.skip_qutip and case == '1d_closed':
        results['qutip'] = test_qutip_cross_check(
            N=8, alpha=3.0,
            mu=np.pi / 2, nu=-np.pi / 2,
            theta_k=np.pi / 4, phi_k=np.pi / 4,
            chi_max=args.chi_max, dt_max=args.dt_max)
    else:
        results['qutip'] = None

    if not args.skip_axis_verification:
        results['axis'] = test_axis_verification()
    else:
        results['axis'] = None

    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    all_pass = True
    for k, v in results.items():
        if v is None:
            print(f"  {k:>12}: SKIPPED")
        else:
            print(f"  {k:>12}: {'PASS' if v else 'FAIL'}")
            if not v:
                all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()