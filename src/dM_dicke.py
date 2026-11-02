"""
src/dM_dicke.py

Dicke-basis analytical formulas for the three-axis Scharnagl
protocol. Exact for OAT and for the alpha = 0 power-law Ising
Hamiltonian, whose twisting generator preserves the fully
symmetric subspace.

Reference: exact_M_Q_derivation.pdf, Sections 3-5.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg
from scipy.special import comb


def _dicke_operators(N):
    """
    Return (Sx, Sy, Sz) as (N+1)x(N+1) matrices in the Dicke basis
    |S, m> with S = N/2 and m in {-S, ..., S}.
    """
    S = N / 2.0
    dim = N + 1
    m_vals = np.arange(-S, S + 1)

    Sz = np.diag(m_vals).astype(complex)

    Sp = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        m = m_vals[i]
        Sp[i + 1, i] = np.sqrt(S * (S + 1) - m * (m + 1))
    Sm = Sp.conj().T

    Sx = 0.5 * (Sp + Sm)
    Sy = (Sp - Sm) / (2j)
    return Sx, Sy, Sz


def _css_dicke(N):
    """Coherent spin state |+x>^{tensor N} in the Dicke basis."""
    S = N / 2.0
    m_vals = np.arange(-S, S + 1)
    coeffs = np.array([
        2.0 ** (-S) * np.sqrt(comb(2 * S, S + m)) for m in m_vals
    ], dtype=complex)
    return coeffs


def _apply_T_z_dicke(vec, mu):
    """Apply T_z(mu) = e^{-i mu S_z^2 / 2} to a Dicke vector."""
    N = len(vec) - 1
    S = N / 2.0
    m_vals = np.arange(-S, S + 1)
    return vec * np.exp(-1j * mu * m_vals**2 / 2.0)


def _apply_T_k_dicke(vec, mu, theta_k, phi_k):
    """Apply T_k(mu) = R_k T_z(mu) R_k^dag to a Dicke vector."""
    if abs(theta_k) < 1e-15 and abs(phi_k) < 1e-15:
        return _apply_T_z_dicke(vec, mu)
    N = len(vec) - 1
    _, Sy, Sz = _dicke_operators(N)
    R_k = (scipy.linalg.expm(-1j * phi_k * Sz)
           @ scipy.linalg.expm(-1j * theta_k * Sy))
    R_k_dag = R_k.conj().T
    vec1 = R_k_dag @ vec
    vec2 = _apply_T_z_dicke(vec1, mu)
    return R_k @ vec2


def M_analytical(N, mu, nu, theta_k=0.0, phi_k=0.0):
    """Exact M matrix in the Dicke basis."""
    css = _css_dicke(N)
    Sx, Sy, Sz = _dicke_operators(N)
    ops = {'x': Sx, 'y': Sy, 'z': Sz}

    psi_pre = _apply_T_z_dicke(css, mu)
    psi_0 = _apply_T_k_dicke(psi_pre, nu - mu, theta_k, phi_k)

    u = {
        a: _apply_T_k_dicke(ops[a] @ psi_pre, nu - mu, theta_k, phi_k)
        for a in ops
    }

    axes = ['x', 'y', 'z']
    M = np.zeros((3, 3))
    for i, ai in enumerate(axes):
        for j, aj in enumerate(axes):
            val = np.vdot(u[ai], ops[aj] @ psi_0)
            # Eq. (25) of exact_M_Q_derivation.pdf.
            # Reduces to the commutator form
            #     M_ij = i <[S_i(mu), S_j(nu)]>
            # when theta_k = phi_k = 0 (Section 4.6 of the
            # derivation). The Dicke-basis evaluation here is exact
            # only for the OAT and alpha = 0 cases; see Sections 7-8
            # of the derivation document.
            M[i, j] = -2.0 * val.imag
    return M


def Q_analytical(N, mu, nu, theta_k=0.0, phi_k=0.0):
    """Exact Q matrix in the Dicke basis."""
    css = _css_dicke(N)
    Sx, Sy, Sz = _dicke_operators(N)
    ops = {'x': Sx, 'y': Sy, 'z': Sz}

    psi_pre = _apply_T_z_dicke(css, mu)
    psi_0 = _apply_T_k_dicke(psi_pre, nu - mu, theta_k, phi_k)

    axes = ['x', 'y', 'z']
    S_mean = np.array([np.real(np.vdot(psi_0, ops[a] @ psi_0))
                       for a in axes])

    Q = np.zeros((3, 3), dtype=complex)
    for i, ai in enumerate(axes):
        for j, aj in enumerate(axes):
            corr = np.vdot(psi_0, ops[ai] @ ops[aj] @ psi_0)
            Q[i, j] = corr - S_mean[i] * S_mean[j]
    return np.real(0.5 * (Q + Q.conj().T))


def snr_analytical(N, mu, nu, theta_k=0.0, phi_k=0.0):
    """Exact SNR of the three-axis protocol, sigma_max(M Q^{-1/2})."""
    M = M_analytical(N, mu, nu, theta_k, phi_k)
    Q = Q_analytical(N, mu, nu, theta_k, phi_k)

    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T

    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    return float(np.max(sv))


def snr_analytical_2axis(N, mu, nu):
    """Two-axis limit (theta_k = phi_k = 0)."""
    return snr_analytical(N, mu, nu, 0.0, 0.0)