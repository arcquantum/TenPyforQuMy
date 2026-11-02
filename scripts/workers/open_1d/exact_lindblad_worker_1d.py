"""
scripts/workers/open_1d/exact_lindblad_worker_1d.py

Exact Lindblad reference worker for the 1D open-system power-law
echo protocol, extended to the Scharnagl three-axis family.
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


def build_pauli_at(i: int, op: qutip.Qobj, N: int) -> qutip.Qobj:
    I = qutip.qeye(2)
    ops = [I] * N
    ops[i] = op
    return qutip.tensor(ops)


def build_hamiltonian(N: int, J: float, alpha: float) -> qutip.Qobj:
    sz = qutip.sigmaz()
    sz_list = [build_pauli_at(i, sz, N) for i in range(N)]
    H = qutip.qzero([2] * N)
    for i in range(N):
        for j in range(i + 1, N):
            r = float(j - i)
            H = H + (J / (N * (r ** alpha))) * (sz_list[i] * sz_list[j])
    return H


def build_collective_spin_ops(N: int) -> dict:
    def collective(op):
        total = qutip.qzero([2] * N)
        for i in range(N):
            total = total + build_pauli_at(i, op, N)
        return total / 2.0
    return {
        'x': collective(qutip.sigmax()),
        'y': collective(qutip.sigmay()),
        'z': collective(qutip.sigmaz()),
    }


def build_css_dm(N: int) -> qutip.Qobj:
    plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    psi = qutip.tensor([plus] * N)
    return psi * psi.dag()


def build_jump_ops(N: int, gamma1: float, gamma2: float) -> list:
    c_ops = []
    if gamma1 > 0.0:
        sm = qutip.sigmam()
        for i in range(N):
            c_ops.append(np.sqrt(gamma1) * build_pauli_at(i, sm, N))
    if gamma2 > 0.0:
        sz = qutip.sigmaz()
        for i in range(N):
            c_ops.append(np.sqrt(gamma2 / 2.0) * build_pauli_at(i, sz, N))
    return c_ops


def build_R_k(theta_k: float, phi_k: float, N: int) -> qutip.Qobj:
    if abs(theta_k) < 1e-15 and abs(phi_k) < 1e-15:
        return qutip.qeye([2] * N)
    sz = qutip.sigmaz()
    sy = qutip.sigmay()
    U_z = (-1j * phi_k * sz / 2).expm()
    U_y = (-1j * theta_k * sy / 2).expm()
    return qutip.tensor([U_z * U_y] * N)


def lindblad_evolve(rho, H, c_ops, tau):
    """
    Evolve rho for a signed coherent time tau.
    tau > 0 : forward Lindblad.
    tau < 0 : coherent reversal under -H for |tau|; dissipator
              still runs forward in time.
    """
    if abs(tau) < 1e-15:
        return rho
    H_eff = H if tau > 0 else -H
    duration = abs(tau)
    if not c_ops:
        U = (-1j * H_eff * duration).expm()
        return U * rho * U.dag()
    result = qutip.mesolve(
        H_eff, rho, [0, duration], c_ops=c_ops,
        options={'atol': 1e-12, 'rtol': 1e-12},
    )
    return result.states[-1]


def evolve_bwd_with_k(rho, H, c_ops, t_bwd, R_k):
    """
    Backward evolution wrapped by the Scharnagl rotation:

        rho  ->  R_k  Lindblad( R_k^dagger rho R_k , t_bwd )  R_k^dagger
    """
    if R_k is None:
        return lindblad_evolve(rho, H, c_ops, t_bwd)
    rho_rot = R_k.dag() * rho * R_k
    rho_ev = lindblad_evolve(rho_rot, H, c_ops, t_bwd)
    return R_k * rho_ev * R_k.dag()


def compute_M(rho_pre, H, c_ops, t_bwd, S_ops, R_k, delta=1e-3):
    axes = ['x', 'y', 'z']
    M = np.zeros((3, 3))
    for i, a in enumerate(axes):
        S_n = S_ops[a]
        U_plus = (-1j * (+delta) * S_n).expm()
        U_minus = (-1j * (-delta) * S_n).expm()
        rho_imp_plus = U_plus * rho_pre * U_plus.dag()
        rho_imp_minus = U_minus * rho_pre * U_minus.dag()
        rho_plus = evolve_bwd_with_k(rho_imp_plus, H, c_ops, t_bwd, R_k)
        rho_minus = evolve_bwd_with_k(rho_imp_minus, H, c_ops, t_bwd, R_k)
        for j, b in enumerate(axes):
            S_j = S_ops[b]
            val_plus = np.real(qutip.expect(S_j, rho_plus))
            val_minus = np.real(qutip.expect(S_j, rho_minus))
            M[i, j] = (val_plus - val_minus) / (2.0 * delta)
    return M


def compute_Q(rho_post, S_ops):
    axes = ['x', 'y', 'z']
    S_vec = np.array(
        [np.real(qutip.expect(S_ops[a], rho_post)) for a in axes]
    )
    Q = np.zeros((3, 3), dtype=complex)
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            corr = qutip.expect(S_ops[a] * S_ops[b], rho_post)
            Q[i, j] = corr - S_vec[i] * S_vec[j]
    return np.real(0.5 * (Q + Q.conj().T))


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
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--theta_k", type=float, default=0.0)
    parser.add_argument("--phi_k", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
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
    N = args.N
    if N > 8:
        print(f"[exact_lindblad_worker_1d] Warning: N={N} requires a "
              f"2^{N} Hilbert space and 2N jump operators.", flush=True)

    print(f"Initializing exact Lindblad worker: N={N}, alpha={args.alpha}, "
          f"mu={args.mu:.6f}, nu={args.nu:.6f}, "
          f"theta_k={args.theta_k:.6f}, phi_k={args.phi_k:.6f}, "
          f"gamma1={args.gamma1:.4e}, gamma2={args.gamma2:.4e}",
          flush=True)

    H = build_hamiltonian(N, args.J, args.alpha)
    S_ops = build_collective_spin_ops(N)
    rho_0 = build_css_dm(N)
    c_ops = build_jump_ops(N, args.gamma1, args.gamma2)
    R_k = build_R_k(args.theta_k, args.phi_k, N)

    t_fwd = args.mu / 2.0
    t_bwd = (args.nu - args.mu) / 2.0

    rho_pre = lindblad_evolve(rho_0, H, c_ops, t_fwd)
    # Backward evolution wraps Lindblad in R_k^dagger / R_k.
    rho_post = evolve_bwd_with_k(rho_pre, H, c_ops, t_bwd, R_k)

    M = compute_M(rho_pre, H, c_ops, t_bwd, S_ops, R_k, delta=1e-3)
    Q = compute_Q(rho_post, S_ops)
    snr = compute_snr(M, Q)

    trace = float(np.real(rho_post.tr()))
    purity = float(np.real((rho_post * rho_post).tr()))
    eigvals = np.linalg.eigvalsh(rho_post.full())
    min_eig = float(np.min(eigvals))
    print(f"rho_post: trace={trace:.12f}, purity={purity:.6f}, "
          f"min_eig={min_eig:.3e}", flush=True)
    print(f"Exact Lindblad SNR = {snr:.8f}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'exact_1d_open')
    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = ''
    if abs(args.theta_k) > 1e-15 or abs(args.phi_k) > 1e-15:
        suffix = f"_tk{args.theta_k:.4f}_pk{args.phi_k:.4f}"
    out_file = os.path.join(
        DATA_DIR,
        f"exact_lindblad_1d_N{N}_alpha{args.alpha}_J{args.J}"
        f"{args.mu_suffix}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}{suffix}.npy"
    )
    np.save(out_file, {
        'N': N, 'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 'mu': args.mu, 'nu': args.nu,
        'theta_k': args.theta_k, 'phi_k': args.phi_k,
        'gamma1': args.gamma1, 'gamma2': args.gamma2,
        'snr': snr,
    })
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()