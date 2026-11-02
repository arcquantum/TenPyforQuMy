"""
scripts/workers/closed_1d/exact_echo_worker_1d.py

Exact-diagonalisation reference worker for the 1D closed-system
power-law echo protocol, extended to the Scharnagl three-axis family.

Protocol
--------
   |psi(phi)> = T_k((nu-mu)/2) R_n(phi) T_z(mu/2) |CSS>

with T_z(t) = e^{-i t H} and T_k(t) = R_k T_z(t) R_k^dagger. When
(theta_k, phi_k) = (0, 0) this reduces to the two-axis protocol of
the symmetric echo workers.

The --dt argument is accepted for CLI compatibility with the TN
workers but ignored; the exact worker uses closed-form matrix
exponentials.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.linalg
import qutip


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# ----------------------------------------------------------------------
# Physics construction
# ----------------------------------------------------------------------
def build_pauli_at(i: int, op: qutip.Qobj, N: int) -> qutip.Qobj:
    """Return I ⊗ ... ⊗ op_i ⊗ ... ⊗ I on N qubits."""
    I = qutip.qeye(2)
    ops = [I] * N
    ops[i] = op
    return qutip.tensor(ops)


def build_hamiltonian(N: int, J: float, alpha: float) -> qutip.Qobj:
    """Build H = (1/N) Σ_{i<j} (J / |i-j|^α) σ_z^i σ_z^j."""
    sz = qutip.sigmaz()
    sz_list = [build_pauli_at(i, sz, N) for i in range(N)]

    H = qutip.qzero([2] * N)
    for i in range(N):
        for j in range(i + 1, N):
            r = float(j - i)
            coupling = J / (N * (r ** alpha))
            H = H + coupling * (sz_list[i] * sz_list[j])
    return H


def build_collective_spin_ops(N: int) -> dict:
    """Return the collective spin operators S_x, S_y, S_z."""
    def collective(op: qutip.Qobj) -> qutip.Qobj:
        total = qutip.qzero([2] * N)
        for i in range(N):
            total = total + build_pauli_at(i, op, N)
        return total / 2.0

    return {
        'x': collective(qutip.sigmax()),
        'y': collective(qutip.sigmay()),
        'z': collective(qutip.sigmaz()),
    }


def build_css(N: int) -> qutip.Qobj:
    """Build |+>^{⊗N}."""
    plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    return qutip.tensor([plus] * N)


def build_R_k(theta_k: float, phi_k: float, N: int) -> qutip.Qobj:
    """
    Many-body rotation R_k = Π_i e^{-i φ_k σ_z^i / 2}
                                e^{-i θ_k σ_y^i / 2}.

    As a matrix, this is Z @ Y applied left-to-right, i.e. the σ_y
    part acts on a ket first, then the σ_z part. Both are collective
    (product) rotations.
    """
    if abs(theta_k) < 1e-15 and abs(phi_k) < 1e-15:
        return qutip.qeye([2] * N)
    sz = qutip.sigmaz()
    sy = qutip.sigmay()
    U_z = (-1j * phi_k * sz / 2).expm()
    U_y = (-1j * theta_k * sy / 2).expm()
    U_single = U_z * U_y
    return qutip.tensor([U_single] * N)


def _expm(A: qutip.Qobj) -> qutip.Qobj:
    try:
        return A.expm()
    except AttributeError:
        return qutip.Qobj(scipy.linalg.expm(A.full()), dims=A.dims)


def _evolve(t: float, H: qutip.Qobj) -> qutip.Qobj:
    """U(t) = exp(-i t H), identity when |t| < 1e-15."""
    if abs(t) < 1e-15:
        return qutip.qeye(H.dims[0])
    return _expm(-1j * t * H)


# ----------------------------------------------------------------------
# Signal and covariance
# ----------------------------------------------------------------------
def compute_M(S_ops: dict,
              psi_pre: qutip.Qobj,
              U_bwd: qutip.Qobj,
              delta: float = 1e-3) -> np.ndarray:
    """
    M_ij = ∂_φ ⟨S_j⟩ |_{φ=0} via central finite differences.
    U_bwd already contains the Scharnagl R_k wrap; this function
    does not need to know about it.
    """
    axes = ['x', 'y', 'z']
    M = np.zeros((3, 3))

    for a_idx, a in enumerate(axes):
        S_n = S_ops[a]
        U_phase_plus = _expm(-1j * (+delta) * S_n)
        U_phase_minus = _expm(-1j * (-delta) * S_n)
        psi_plus = U_bwd * (U_phase_plus * psi_pre)
        psi_minus = U_bwd * (U_phase_minus * psi_pre)
        for j_idx, j in enumerate(axes):
            S_j = S_ops[j]
            val_plus = np.real(qutip.expect(S_j, psi_plus))
            val_minus = np.real(qutip.expect(S_j, psi_minus))
            M[a_idx, j_idx] = (val_plus - val_minus) / (2.0 * delta)
    return M


def compute_Q(S_ops: dict, psi_q: qutip.Qobj) -> np.ndarray:
    axes = ['x', 'y', 'z']
    S_vec = np.array([np.real(qutip.expect(S_ops[a], psi_q)) for a in axes])
    Cov = np.zeros((3, 3), dtype=complex)
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            corr = qutip.expect(S_ops[a] * S_ops[b], psi_q)
            Cov[i, j] = corr - S_vec[i] * S_vec[j]
    return np.real(0.5 * (Cov + Cov.conj().T))


def compute_snr(M: np.ndarray, Q: np.ndarray) -> float:
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T
    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    return float(np.max(sv))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--theta_k", type=float, default=0.0)
    parser.add_argument("--phi_k", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Accepted for CLI compatibility; ignored.")
    args = parser.parse_args()


    # --- (mu1, mu2) translation (migration) ---
    if getattr(args, "mu1", None) is not None and args.mu is None:
        _N = getattr(args, "N", None) or ((getattr(args, "Lx", 0) or 0) * (getattr(args, "Ly", 0) or 0))
        args.mu = _N * args.mu1 / 2.0
        args.nu = _N * (args.mu1 - args.mu2) / 2.0
        args.convention = "mu1mu2"
    elif args.mu is not None and getattr(args, "mu1", None) is None:
        args.convention = "internal"
    else:
        raise SystemExit("Provide either --mu --nu or --mu1 --mu2")
    args.mu_suffix = (f"_mu1{args.mu1:.4f}_mu2{args.mu2:.4f}"
                      if args.convention == "mu1mu2"
                      else f"_mu{args.mu:.4f}_nu{args.nu:.4f}")
    # --- End (mu1, mu2) translation ---
    if args.N > 12:
        print(f"[exact_echo_worker_1d] Warning: N={args.N} requires a "
              f"2^{args.N} Hilbert space.", flush=True)

    print(f"Initializing exact echo worker: N={args.N}, alpha={args.alpha}, "
          f"J={args.J}, mu={args.mu:.6f}, nu={args.nu:.6f}, "
          f"theta_k={args.theta_k:.6f}, phi_k={args.phi_k:.6f}",
          flush=True)

    H = build_hamiltonian(args.N, args.J, args.alpha)
    S_ops = build_collective_spin_ops(args.N)
    psi_0 = build_css(args.N)

    t_fwd = args.mu / 2.0
    t_bwd = (args.nu - args.mu) / 2.0

    R_k = build_R_k(args.theta_k, args.phi_k, args.N)

    U_fwd = _evolve(t_fwd, H)
    # Backward evolution is R_k T_z(t_bwd) R_k^dagger.
    U_bwd = R_k * _evolve(t_bwd, H) * R_k.dag()

    psi_pre = U_fwd * psi_0
    psi_q = U_bwd * psi_pre

    M = compute_M(S_ops, psi_pre, U_bwd, delta=1e-3)
    Q = compute_Q(S_ops, psi_q)
    snr = compute_snr(M, Q)

    print(f"Exact echo SNR = {snr:.8f}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'exact_1d_closed')
    os.makedirs(DATA_DIR, exist_ok=True)

    suffix = ''
    if abs(args.theta_k) > 1e-15 or abs(args.phi_k) > 1e-15:
        suffix = (f"_tk{args.theta_k:.4f}_pk{args.phi_k:.4f}")

    out_file = os.path.join(
        DATA_DIR,
        f"exact_echo_1d_N{args.N}_alpha{args.alpha}_J{args.J}"
        f"{args.mu_suffix}{suffix}.npy"
    )

    np.save(out_file, {
        'N': args.N, 'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu, 'nu': args.nu,
        'theta_k': args.theta_k, 'phi_k': args.phi_k,
        'snr': snr,
    })
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()