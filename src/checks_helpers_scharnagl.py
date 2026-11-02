"""
src/checks_helpers_scharnagl.py

Reference implementation of the Scharnagl three-axis twist applied
to the coherent spin state, used by Test 4 of check_scharnagl.py.

Uses dense vectors on the 2^N Hilbert space. Only tractable for
N <= 6.

Conventions
-----------
The TeNPy SpinHalfSite basis is (|up>, |down>) with S_z = +1/2 on
|up> and -1/2 on |down>. The MPS index ordering has site 0 as the
most-significant bit, which matches the tensor-product ordering of
qutip.tensor and of the np.kron chain used here.

The Hamiltonian is the power-law Ising Hamiltonian of the production
model,

    H = (1/N) sum_{i<j} (J / |i-j|^alpha) sigma_z^i sigma_z^j,

with J = N/2 by default so that at alpha = 0 the result is exactly
H = S_z^2 - N/4. The evolution uses the code convention
T_z(t) = exp(-i t H) with t = mu/2, matching src/evolution.py.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg


def _pauli_at(i: int, op: np.ndarray, N: int) -> np.ndarray:
    """I ⊗ ... ⊗ op_i ⊗ ... ⊗ I on N qubits (site 0 leftmost)."""
    I = np.eye(2, dtype=complex)
    mats = [I] * N
    mats[i] = op
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def explicit_reference_scharnagl(N: int,
                                 alpha: float,
                                 mu: float,
                                 theta_k: float,
                                 phi_k: float,
                                 J: float | None = None) -> np.ndarray:
    """
    Return the dense vector

        T_k(mu) |+x>^{⊗N}

    where T_k(mu) = R_k exp(-i (mu/2) H) R_k^dagger, R_k is the
    many-body rotation that maps ẑ to k, and H is the power-law
    Ising Hamiltonian with coupling J (default J = N/2).

    Parameters
    ----------
    N : int
        Number of spins.
    alpha : float
        Power-law exponent.
    mu : float
        Twisting strength, in the codebase convention where the
        actual evolution time passed to the engine is mu/2.
    theta_k, phi_k : float
        Polar and azimuthal angles of the twisting axis.
    J : float or None
        Coupling. Default N/2.

    Returns
    -------
    psi : ndarray, shape (2**N,), complex
        The reference state, in the standard tensor-product
        ordering with site 0 as the most-significant bit.
    """
    if J is None:
        J = N / 2.0

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    dim = 2 ** N

    # --- Hamiltonian H = (1/N) sum_{i<j} (J / r^alpha) sigma_z^i sigma_z^j ---
    H = np.zeros((dim, dim), dtype=complex)
    sz_list = [_pauli_at(i, sz, N) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            r = float(j - i)
            H = H + (J / (N * (r ** alpha))) * (sz_list[i] @ sz_list[j])

    # --- CSS |+x>^{⊗N} ---
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = plus
    for _ in range(N - 1):
        psi = np.kron(psi, plus)

    # --- R_k = tensor product of U_single over all sites ---
    # Single-site unitary: U_single = U_z @ U_y, i.e. U_y acts on a
    # ket first, then U_z. This matches R_k = e^{-i phi S_z} e^{-i theta S_y}.
    U_z = scipy.linalg.expm(-1j * phi_k * sz / 2.0)
    U_y = scipy.linalg.expm(-1j * theta_k * sy / 2.0)
    U_single = U_z @ U_y

    R_k = np.eye(dim, dtype=complex)
    for i in range(N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[i] = U_single
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        # Accumulate left-to-right; different-site operators commute,
        # so the order of accumulation does not matter.
        R_k = term @ R_k

    # --- evolution: R_k exp(-i (mu/2) H) R_k^dagger |+x>^{⊗N} ---
    U_evolve = scipy.linalg.expm(-1j * (mu / 2.0) * H)
    psi_final = R_k @ U_evolve @ R_k.conj().T @ psi
    return psi_final