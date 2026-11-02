"""
src/simple_mpo.py

Naive vectorized-MPO method for the 1D power-law open-system echo
protocol, based on the Choi-Jamiołkowski isomorphism.

Physics
-------
The density matrix rho on a chain of L spin-1/2 sites is mapped to a
vector |rho>> on the doubled Hilbert space, with local dimension
d^2 = 4. The vectorization convention is

    vec(|i><j|) = |i> ⊗ |j>          (column-stacking)

so that the identity

    vec(A rho B) = (A ⊗ B^T) |rho>>

holds for any operators A, B on the physical Hilbert space. Under this
map, the Lindblad master equation becomes a Schrödinger-like equation

    d/dt |rho>> = L |rho>>

with the Liouvillian superoperator

    L = -i (H ⊗ I - I ⊗ H^T)
        + Σ_m γ_m [ L_m ⊗ L_m^* - ½ (L_m† L_m ⊗ I
                                      + I ⊗ (L_m† L_m)^*) ].

TeNPy's ``ExpMPOEvolution`` computes ``exp(-i H_MPO Δt)`` for an MPO
``H_MPO``. To obtain the vectorized Lindblad evolution
``exp(L Δt)``, the MPO we build must be ``H_MPO = i L``. The
implemented MPO coefficients below are therefore all multiplied by
a factor of ``i`` relative to the bare Liouvillian.

Failure mode
------------
The Liouvillian is not Hermitian, and the MPO truncation performed
by ``ExpMPOEvolution`` uses local SVDs that minimize the Frobenius
norm on the vectorized state. This is not the same as preserving
the positivity of the reconstructed density matrix. The method is
designated in the codebase documentation as a failure baseline; its purpose
is to demonstrate that the naive vectorized-MPO approach breaks
positivity under local SVD truncation.

Failure signatures
------------------
  * purity = Tr(rho^2) > 1 (physical: ≤ 1)
  * minimum eigenvalue of the symmetrised rho < 0 (physical: ≥ 0)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import Site
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution


# ----------------------------------------------------------------------
# Doubled site
# ----------------------------------------------------------------------
def _build_doubled_site():
    """
    Build a TeNPy Site of dimension d^2 = 4 for the vectorized density
    matrix. The local basis is ordered as

        |00>, |01>, |10>, |11>

    where the left factor is the physical ("ket") side and the right
    factor is the conjugate ("bra") side.

    Operators:
        SzL, SxL, SyL   --  S_α ⊗ I
        SzR, SxR, SyR   --  I ⊗ S_α
        smL_smR         --  σ_- ⊗ σ_-
        proj1L          --  |1><1| ⊗ I
        proj1R          --  I ⊗ |1><1|
        szL_szR         --  σ_z ⊗ σ_z
    """
    d = 4
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)     # |0><1| = lowering
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
    id2 = np.eye(2, dtype=complex)

    def kron(a, b):
        return np.kron(a, b)

    raw_ops = {
        'SzL': 0.5 * kron(sz, id2),
        'SxL': 0.5 * kron(sx, id2),
        'SyL': 0.5 * kron(sy, id2),
        'SzR': 0.5 * kron(id2, sz),
        'SxR': 0.5 * kron(id2, sx),
        'SyR': 0.5 * kron(id2, sy),
        'smL_smR': kron(sm, sm),
        'proj1L': kron(proj1, id2),
        'proj1R': kron(id2, proj1),
        'szL_szR': kron(sz, sz),
    }
    leg = npc.LegCharge.from_trivial(d)
    state_labels = ['00', '01', '10', '11']
    site = Site(leg, state_labels, **raw_ops)
    return site


# ----------------------------------------------------------------------
# Liouvillian MPO
# ----------------------------------------------------------------------
class SimpleMPOModel(CouplingModel, MPOModel):
    """
    Vectorized Liouvillian superoperator for the 1D power-law
    open-system echo protocol.

    The MPO encodes ``H_MPO = i L``, so that ``ExpMPOEvolution``
    computes the vectorized Lindblad evolution ``exp(L Δt)``. All
    coefficients below are therefore multiplied by ``i`` relative
    to the bare Liouvillian.
    """

    def __init__(self, model_params):
        self.L = model_params.get('L', 8)
        self.J = model_params.get('J', 1.0)
        self.alpha = model_params.get('alpha', 1.0)
        self.gamma1 = model_params.get('gamma1', 0.0)
        self.gamma2 = model_params.get('gamma2', 0.0)
        self.N = self.L

        site = _build_doubled_site()
        lattice = Chain(L=self.L, site=site, bc='open')
        CouplingModel.__init__(self, lattice)
        self.init_terms()
        H_MPO = self.calc_H_MPO()
        MPOModel.__init__(self, lattice, H_MPO)

        print(f"[SimpleMPOModel] L={self.L}, alpha={self.alpha}, "
              f"J={self.J}, gamma1={self.gamma1:.4e}, "
              f"gamma2={self.gamma2:.4e}, "
              f"D_MPO_max={max(H_MPO.chi)}", flush=True)

    def init_terms(self):
        L = self.L
        base = (4.0 * self.J) / self.N  # 4J/N because S_z = σ_z / 2

        # --- Coherent part of H_MPO = i L ---
        # i * (-i H ⊗ I) = +H ⊗ I  →  strength +J_ij on (SzL, SzL)
        # i * (+i I ⊗ H^T) = -I ⊗ H^T  →  strength -J_ij on (SzR, SzR)
        for dx in range(1, L):
            J_ij = base / (dx ** self.alpha)
            self.add_coupling(+J_ij, 0, 'SzL', 0, 'SzL', dx)
            self.add_coupling(-J_ij, 0, 'SzR', 0, 'SzR', dx)

        # --- Dissipative part of H_MPO = i L, T1 channel ---
        # i * γ1 σ_- ρ σ_-†  →  strength +i γ1 on smL_smR
        # i * (-γ1/2) {|1><1|, ρ}  →  strength -i γ1/2 on proj1L and proj1R
        if self.gamma1 > 0.0:
            self.add_onsite(+0.5j * self.gamma1 * 2.0, 0, 'smL_smR')
            self.add_onsite(-0.5j * self.gamma1, 0, 'proj1L')
            self.add_onsite(-0.5j * self.gamma1, 0, 'proj1R')

        # --- Dissipative part of H_MPO = i L, T2 channel ---
        # i * (γ2/2) σ_z ρ σ_z  →  strength +i γ2/2 on szL_szR
        # i * (-γ2/2) ρ        →  strength -i γ2/2 on Id
        if self.gamma2 > 0.0:
            self.add_onsite(+0.5j * self.gamma2, 0, 'szL_szR')
            self.add_onsite(-0.5j * self.gamma2, 0, 'Id')


# ----------------------------------------------------------------------
# State initialization
# ----------------------------------------------------------------------
def initialize_vectorized_css(model):
    """
    Build the vectorized Coherent Spin State on the doubled lattice:

        |ρ_0>> = |+>^{⊗L} ⊗ |+>^{⊗L}

    which at each doubled site is the uniform superposition
    (|00> + |01> + |10> + |11>) / 2.
    """
    L = model.L
    sites = model.lat.mps_sites()
    local_vec = np.ones(4, dtype=complex) / 2.0
    product_state_list = [local_vec] * L
    psi = MPS.from_product_state(
        sites, product_state_list, bc='finite', dtype=complex
    )
    return psi


# ----------------------------------------------------------------------
# Evolution
# ----------------------------------------------------------------------
def evolve_mpo(psi, model, total_time, chi_mpo, dt_max=0.05):
    """
    Evolve the vectorized state under the Liouvillian MPO for a signed
    total_time. Negative times are handled by ``ExpMPOEvolution`` via a
    negative ``dt``.
    """
    if abs(total_time) < 1e-15:
        return
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt_step = total_time / N_steps

    mpo_params = {
        'dt': dt_step,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_mpo, 'svd_min': 1e-12},
    }
    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = N_steps
    engine.run()


# ----------------------------------------------------------------------
# Phase imprint on physical side
# ----------------------------------------------------------------------
def apply_phase_imprint(psi, axis, phi):
    """
    Apply the phase imprint e^{-i φ S_n} ρ e^{+i φ S_n} on the physical
    side. In the doubled space this is a product of single-site operators
        U_L ⊗ U_R = e^{-i φ S_n} ⊗ e^{+i φ S_n^*}.

    The operator is wrapped into an ``npc.Array`` with the same trivial
    leg as the doubled site, because TeNPy's ``apply_local_op`` expects
    a conserved array rather than a plain numpy matrix.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli = {'x': sx, 'y': sy, 'z': sz}[axis]
    S_n_local = 0.5 * pauli

    U_L = scipy.linalg.expm(-1j * phi * S_n_local)
    U_R = scipy.linalg.expm(+1j * phi * S_n_local.conj().T)
    U_doubled = np.kron(U_L, U_R)

    # Wrap into an npc.Array with the site's leg, matching the pattern
    # the Site constructor uses internally for the operators it stores.
    site = psi.sites[0]
    leg = site.leg
    U_npc = npc.Array.from_ndarray(
        U_doubled, [leg, leg.conj()], labels=['p', 'p*']
    )

    for i in range(psi.L):
        psi.apply_local_op(i, U_npc, unitary=True)


# ----------------------------------------------------------------------
# Reconstruction of the density matrix
# ----------------------------------------------------------------------
def contract_mps_to_rho(psi, L):
    """
    Contract the doubled MPS to a full density matrix of shape
    (2^L, 2^L). Only feasible for small L (L ≤ 10 in practice).
    """
    psi.canonical_form()
    theta = psi.get_theta(0, n=L)
    arr = theta.to_ndarray()

    # Squeeze trailing boundary bond dimensions
    while arr.ndim > L and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # Reshape each local dim 4 → (2, 2) and reorder so that all
    # physical indices come first, conjugate indices after.
    arr = arr.reshape([2, 2] * L)
    perm = list(range(0, 2 * L, 2)) + list(range(1, 2 * L, 2))
    arr = np.transpose(arr, perm)

    return arr.reshape(2 ** L, 2 ** L)


# ----------------------------------------------------------------------
# Collective spin operators on the physical Hilbert space
# ----------------------------------------------------------------------
def collective_spin_operator(axis, L):
    """Return the collective S_α = Σ_i S_α^(i) on the physical space."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli = {'x': sx, 'y': sy, 'z': sz}[axis]

    dim = 2 ** L
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(L):
        ops = [np.eye(2, dtype=complex)] * L
        ops[i] = 0.5 * pauli
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        S += term
    return S


# ----------------------------------------------------------------------
# Failure diagnostics
# ----------------------------------------------------------------------
def diagnose_rho(rho):
    """
    Return (min_eig, purity) of the symmetrised rho.

    A physical density matrix has min_eig >= 0 and 1/2^L ≤ purity ≤ 1.
    Values outside this range are the failure signatures of the naive
    vectorized MPO method.
    """
    rho_sym = (rho + rho.conj().T) / 2.0
    evals = np.linalg.eigvalsh(rho_sym)
    min_eig = float(np.min(evals))
    purity = float(np.real(np.trace(rho_sym @ rho_sym)))
    return min_eig, purity