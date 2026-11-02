"""
scripts/workers/closed_2d/exact_echo_worker_2d.py

Exact-diagonalisation reference worker for the 2D cylinder
closed-system power-law echo protocol, extended to the Scharnagl
three-axis family.
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


def site_index(k: int, Ly: int) -> tuple[int, int]:
    return (k // Ly, k % Ly)


def min_image_distance(k1: int, k2: int, Lx: int, Ly: int) -> float:
    i1, j1 = site_index(k1, Ly)
    i2, j2 = site_index(k2, Ly)
    di = abs(i1 - i2)
    dj_abs = abs(j1 - j2)
    dj_min = min(dj_abs, Ly - dj_abs)
    return float(np.sqrt(di * di + dj_min * dj_min))


def coupling(k1: int, k2: int, Lx: int, Ly: int,
             J: float, alpha: float) -> float:
    N = Lx * Ly
    r = min_image_distance(k1, k2, Lx, Ly)
    if abs(alpha) < 1e-12:
        return J / N
    return J / (N * (r ** alpha))


def build_pauli_at(i: int, op: qutip.Qobj, N: int) -> qutip.Qobj:
    I = qutip.qeye(2)
    ops = [I] * N
    ops[i] = op
    return qutip.tensor(ops)


def build_hamiltonian_2d(Lx: int, Ly: int, J: float,
                         alpha: float) -> qutip.Qobj:
    N = Lx * Ly
    sz = qutip.sigmaz()
    sz_list = [build_pauli_at(i, sz, N) for i in range(N)]
    H = qutip.qzero([2] * N)
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            c = coupling(k1, k2, Lx, Ly, J, alpha)
            H = H + c * (sz_list[k1] * sz_list[k2])
    return H


def build_collective_spin_ops(N: int) -> dict:
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
    plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    return qutip.tensor([plus] * N)


def build_R_k(theta_k: float, phi_k: float, N: int) -> qutip.Qobj:
    if abs(theta_k) < 1e-15 and abs(phi_k) < 1e-15:
        return qutip.qeye([2] * N)
    sz = qutip.sigmaz()
    sy = qutip.sigmay()
    U_z = (-1j * phi_k * sz / 2).expm()
    U_y = (-1j * theta_k * sy / 2).expm()
    return qutip.tensor([U_z * U_y] * N)


def _expm_neg_i_t_H(t: float, H: qutip.Qobj) -> qutip.Qobj:
    if abs(t) < 1e-15:
        return qutip.qeye(H.dims[0])
    return (-1j * t * H).expm()


def compute_M(S_ops, psi_pre, U_bwd, delta=1e-3):
    axes = ['x', 'y', 'z']
    M = np.zeros((3, 3))
    for a_idx, a in enumerate(axes):
        S_n = S_ops[a]
        U_plus = (-1j * (+delta) * S_n).expm()
        U_minus = (-1j * (-delta) * S_n).expm()
        psi_plus = U_bwd * (U_plus * psi_pre)
        psi_minus = U_bwd * (U_minus * psi_pre)
        for j_idx, j in enumerate(axes):
            S_j = S_ops[j]
            val_plus = np.real(qutip.expect(S_j, psi_plus))
            val_minus = np.real(qutip.expect(S_j, psi_minus))
            M[a_idx, j_idx] = (val_plus - val_minus) / (2.0 * delta)
    return M


def compute_Q(S_ops, psi_q):
    axes = ['x', 'y', 'z']
    S_vec = np.array([np.real(qutip.expect(S_ops[a], psi_q)) for a in axes])
    Cov = np.zeros((3, 3), dtype=complex)
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            Cov[i, j] = qutip.expect(S_ops[a] * S_ops[b], psi_q) \
                        - S_vec[i] * S_vec[j]
    return np.real(0.5 * (Cov + Cov.conj().T))


def compute_snr(M, Q):
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T
    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    return float(np.max(sv))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lx", type=int, required=True)
    parser.add_argument("--Ly", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--theta_k", type=float, default=0.0)
    parser.add_argument("--phi_k", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.02)
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
    N = args.Lx * args.Ly
    if N > 12:
        print(f"[exact_echo_worker_2d] Warning: N={N} requires a "
              f"2^{N} Hilbert space.", flush=True)

    print(f"Initializing exact 2D echo worker: Lx={args.Lx}, Ly={args.Ly}, "
          f"N={N}, alpha={args.alpha}, mu={args.mu:.6f}, "
          f"nu={args.nu:.6f}, theta_k={args.theta_k:.6f}, "
          f"phi_k={args.phi_k:.6f}", flush=True)

    H = build_hamiltonian_2d(args.Lx, args.Ly, args.J, args.alpha)
    S_ops = build_collective_spin_ops(N)
    psi_0 = build_css(N)

    t_fwd = args.mu / 2.0
    t_bwd = (args.nu - args.mu) / 2.0
    R_k = build_R_k(args.theta_k, args.phi_k, N)

    U_fwd = _expm_neg_i_t_H(t_fwd, H)
    U_bwd = R_k * _expm_neg_i_t_H(t_bwd, H) * R_k.dag()

    psi_pre = U_fwd * psi_0
    psi_q = U_bwd * psi_pre

    M = compute_M(S_ops, psi_pre, U_bwd, delta=1e-3)
    Q = compute_Q(S_ops, psi_q)
    snr = compute_snr(M, Q)
    print(f"Exact 2D echo SNR = {snr:.8f}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'exact_2d_closed')
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = ''
    if abs(args.theta_k) > 1e-15 or abs(args.phi_k) > 1e-15:
        suffix = f"_tk{args.theta_k:.4f}_pk{args.phi_k:.4f}"
    out_file = os.path.join(
        DATA_DIR,
        f"exact_echo_2d_Lx{args.Lx}_Ly{args.Ly}"
        f"_alpha{args.alpha}_J{args.J}"
        f"{args.mu_suffix}{suffix}.npy"
    )
    np.save(out_file, {
        'Lx': args.Lx, 'Ly': args.Ly, 'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu, 'nu': args.nu,
        'theta_k': args.theta_k, 'phi_k': args.phi_k,
        'snr': snr,
    })
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()