"""
scripts/checks/check_qutip_closed.py

Cross-checks the exact-M and exact-Q matrices from src/dM.py against
an independent QuTiP implementation, at N=8, alpha=3, closed 1D, for
two Scharnagl twist axes:

    (theta_k, phi_k) = (0, 0)             -- two-axis case
    (theta_k, phi_k) = (pi/4, pi/4)       -- three-axis case

The QuTiP side computes M and Q from the analytical formula
    M_ij = -2 Im[ <u_i| S_j |psi_0> ]
    Q_ij = Re[ <psi_0| S_i S_j |psi_0> ] - <S_i><S_j>
where the states are built with exact matrix exponentials.

Tolerances: Frobenius norm < 1e-6 for both M and Q, at both
(theta_k, phi_k) points.
"""

from __future__ import annotations
import importlib.util
import os
import sys
import numpy as np
import scipy.linalg
import qutip


ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))


from src.models import PowerLawOATModel
from src.evolution import initialize_css, evolve_mps_time
from src.dM import exact_M, exact_Q


# Import the QuTiP worker helpers. Do NOT execute its main().
_qtw_path = os.path.join(ROOT_DIR, 'scripts', 'workers', 'closed_1d', 'exact_echo_worker_1d.py')
_spec = importlib.util.spec_from_file_location(
    'exact_echo_worker_1d', _qtw_path,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load module from {_qtw_path}")
_qtw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qtw)


AXES = ['x', 'y', 'z']


def k_axis_from_angles(theta_k, phi_k):
    return np.array([
        np.sin(theta_k) * np.cos(phi_k),
        np.sin(theta_k) * np.sin(phi_k),
        np.cos(theta_k),
    ])


def compute_qutip_MQ(N, J, alpha, mu, nu, theta_k, phi_k):
    """Exact M and Q on the QuTiP side."""
    H = _qtw.build_hamiltonian(N, J, alpha)
    S_ops = _qtw.build_collective_spin_ops(N)
    psi_0 = _qtw.build_css(N)
    R_k = _qtw.build_R_k(theta_k, phi_k, N)

    t_fwd = mu / 2.0
    t_bwd = (nu - mu) / 2.0
    U_fwd = _qtw._evolve(t_fwd, H)
    U_bwd = R_k * _qtw._evolve(t_bwd, H) * R_k.dag()

    psi_pre = U_fwd * psi_0
    psi_q = U_bwd * psi_pre

    # M_ij = -2 Im[ <u_i| S_j |psi_0> ].
    # Qobj.overlap(other) returns <self|other> as a Python complex in
    # both QuTiP 4.x and 5.x, so no .full()[0, 0] indexing is needed.
    u = {a: U_bwd * (S_ops[a] * psi_pre) for a in AXES}
    M = np.zeros((3, 3))
    for i, ai in enumerate(AXES):
        for j, aj in enumerate(AXES):
            ket = S_ops[aj] * psi_q
            val = complex(u[ai].overlap(ket))
            M[i, j] = -2.0 * val.imag

    # Q_ij = Re[ <psi_0| S_i S_j |psi_0> ] - <S_i><S_j>.
    # qutip.expect(op, psi) returns Tr[op |psi><psi|] as a complex in
    # both versions.
    S_mean = np.array(
        [np.real(qutip.expect(S_ops[a], psi_q)) for a in AXES]
    )
    Q = np.zeros((3, 3), dtype=complex)
    for i, ai in enumerate(AXES):
        for j, aj in enumerate(AXES):
            corr = complex(qutip.expect(S_ops[ai] * S_ops[aj], psi_q))
            Q[i, j] = corr - S_mean[i] * S_mean[j]
    Q = np.real(0.5 * (Q + Q.conj().T))
    return M, Q


def compute_mps_MQ(N, J, alpha, mu, nu, theta_k, phi_k,
                   chi_max, dt_max):
    """Exact M and Q on the MPS side via src.dM."""
    model = PowerLawOATModel({
        'N': N, 'J': J, 'alpha': alpha, 'conserve': None,
        'compression_method': 'exponential',
    })
    psi_pre = initialize_css(model.lat)
    evolve_mps_time(psi_pre, model, mu / 2.0, chi_max, dt_max)

    if abs(theta_k) < 1e-15 and abs(phi_k) < 1e-15:
        k_axis = None
    else:
        k_axis = k_axis_from_angles(theta_k, phi_k)

    M, psi_q = exact_M(psi_pre, model, k_axis, nu - mu,
                       chi_max, dt_max)
    Q = exact_Q(psi_q)
    return M, Q


def run_case(N, J, alpha, mu, nu, theta_k, phi_k,
             chi_max, dt_max, tol):
    print(f"\n--- (theta_k, phi_k) = ({theta_k:.4f}, {phi_k:.4f}) ---")

    print("  computing QuTiP exact M, Q ...", flush=True)
    M_q, Q_q = compute_qutip_MQ(N, J, alpha, mu, nu, theta_k, phi_k)

    print("  computing MPS exact M, Q ...", flush=True)
    M_m, Q_m = compute_mps_MQ(N, J, alpha, mu, nu,
                               theta_k, phi_k, chi_max, dt_max)

    err_M = float(np.linalg.norm(M_m - M_q, 'fro'))
    err_Q = float(np.linalg.norm(Q_m - Q_q, 'fro'))

    ok_M = err_M < tol
    ok_Q = err_Q < tol

    print(f"  ||dM||_F = {err_M:.3e}  (tol {tol:.0e})  "
          f"[{'PASS' if ok_M else 'FAIL'}]")
    print(f"  ||dQ||_F = {err_Q:.3e}  (tol {tol:.0e})  "
          f"[{'PASS' if ok_Q else 'FAIL'}]")

    return ok_M and ok_Q


def main():
    N = 8
    J = 1.0
    alpha = 3.0
    mu = np.pi / 2
    nu = -np.pi / 2
    chi_max = 32
    # dt_max = 0.005 rather than 0.01. See check_qutip_open.py for the
    # rationale: the residual is dominated by the O(dt^2) Taylor error
    # of ExpMPOEvolution, which does not cancel against the exact QuTiP
    # reference. Halving dt reduces the residual by 4x to ~4e-7.
    dt_max = 0.005
    tol = 1e-6

    print("=" * 70)
    print("check_qutip_closed.py: MPS exact-M/Q vs QuTiP reference")
    print("=" * 70)
    print(f"  N={N}, alpha={alpha}, J={J}, "
          f"mu={mu:.4f}, nu={nu:.4f}")
    print(f"  chi_max={chi_max}, dt_max={dt_max}, tol={tol:.0e}")

    ok1 = run_case(N, J, alpha, mu, nu, 0.0, 0.0,
                   chi_max, dt_max, tol)
    ok2 = run_case(N, J, alpha, mu, nu, np.pi / 4, np.pi / 4,
                   chi_max, dt_max, tol)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  (0, 0)         : {'PASS' if ok1 else 'FAIL'}")
    print(f"  (pi/4, pi/4)   : {'PASS' if ok2 else 'FAIL'}")

    if ok1 and ok2:
        print("\n[PASS] check_qutip_closed.py")
        sys.exit(0)
    print("\n[FAIL] check_qutip_closed.py")
    sys.exit(1)


if __name__ == "__main__":
    main()