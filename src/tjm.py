"""
src/tjm.py

Tensor Jump Method (TJM) for the 1D open-system power-law echo
protocol. Following Sander et al. (2025).

Algorithm
---------
The Lindblad master equation is unravelled into a trajectory
ensemble in which each trajectory evolves under the non-Hermitian
effective Hamiltonian

    H_eff = H_0 - (i/2) Σ_m γ_m L_m† L_m

for a short time δt, followed by a stochastic quantum jump chosen
from the set {L_m} with probability proportional to

    δp_m = δt γ_m ⟨ψ|L_m† L_m|ψ⟩.

The full non-unitary evolution is approximated by Strang splitting,

    exp(-i H_eff δt) ≈
        exp(-i H_D δt/2) exp(-i H_0 δt) exp(-i H_D δt/2)
        + O(δt^3),

where H_D = -(i/2) Σ_m γ_m L_m† L_m is the dissipative Hamiltonian.
Because the jump operators are single-site, the dissipative
contraction factorizes exactly into a product of local diagonal
operators, without truncation.

Coherent step
-------------
For the unitary part exp(-i H_0 δt), the coherent power-law
Hamiltonian is applied with TeNPy's ExpMPOEvolution at fixed
maximum bond dimension chi_max. This matches the closed-system
worker's evolution and ensures the γ = 0 limit is reproduced
exactly.

Stochastic jump
---------------
After each full step the norm loss δp of the pre-jump state is
compared against a uniform random draw ε ∈ [0, 1). If ε < δp, a
jump is applied. For the T1 channel L1 = √γ1 σ_- with jump
probability ∝ dt γ1 Σ_i p_up(i); for the T2 channel
L2 = √(γ2/2) σ_z with jump probability dt γ2/2 per site. The total
δp is the sum of the two channels, and the jump type is chosen
according to their relative weights.

Scope
-----
This implementation covers the 1D chain. For the 2D cylinder the
same trajectory machinery is used with the block-site model of
the documentation, and the same jump process acts on the block physical
index. The 2D extension is documented in the worker.

Notation
--------
|up⟩ = |0⟩ = |m_s = 0⟩ (excited state for T1)
|down⟩ = |1⟩ = |m_s = -1⟩ (ground state for T1)
The T1 jump operator is sigmam() = |down⟩⟨up|, which takes an
excited spin to its ground state.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution


# ----------------------------------------------------------------------
# Dissipative contraction
# ----------------------------------------------------------------------
def apply_dissipative_half_step(psi, gamma1, dt_half):
    """
    Apply the exact single-site dissipative half-step contraction

        D[dt_half] = Π_i diag( 1, exp(-dt_half * γ1 / 2) )

    on every site. The T2 contribution is a global scalar
    exp(-dt_half * N * γ2 / 2), which is dropped because it does not
    affect the physical state after renormalization.

    Dimension-aware: for a standard spin-1/2 MPS (d=2 per site), the
    operator is diag(1, d1) as before. For a block-site MPS (d = 2^Ly
    per site), each of the Ly layers carries an independent factor
    diag(1, d1), so the block-site operator is diagonal with entries

        D_block[b] = d1 ** popcount(b),    b = 0, ..., 2^Ly - 1.

    Parameters
    ----------
    psi : MPS
    gamma1 : float
        T1 rate.
    dt_half : float
        Half time step (positive).
    """
    if gamma1 <= 0.0 or dt_half <= 0.0:
        return

    site = psi.sites[0]
    d = site.dim
    d1 = float(np.exp(-dt_half * gamma1 / 2.0))

    if d == 2:
        diag_vals = np.array([1.0, d1], dtype=complex)
    else:
        # Block site with d = 2^Ly. Each physical layer contributes an
        # independent diag(1, d1) factor, so the block-site diagonal
        # value is d1 raised to the number of bits set in the basis
        # state index.
        diag_vals = np.array(
            [d1 ** bin(b).count("1") for b in range(d)],
            dtype=complex,
        )

    D_site = npc.Array.from_ndarray(
        np.diag(diag_vals), [site.leg, site.leg.conj()],
        labels=['p', 'p*'],
    )
    for i in range(psi.L):
        psi.apply_local_op(i, D_site, unitary=False, renormalize=True)


# ----------------------------------------------------------------------
# Coherent evolution
# ----------------------------------------------------------------------
def evolve_coherent_step(psi, model, dt, chi_max):
    """
    Apply exp(-i H_0 dt) to the MPS with ExpMPOEvolution.
    """
    if abs(dt) < 1e-15:
        return
    mpo_params = {
        'dt': dt,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12},
    }
    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = 1
    engine.run()


# ----------------------------------------------------------------------
# Down/up populations
# ----------------------------------------------------------------------
def _physical_site_populations(psi):
    """
    Return (p_up, p_down) as flat 1-D arrays over physical sites.

    For a standard spin-1/2 MPS (d=2 per MPS site), the arrays have
    length L_mps. For a block-site MPS with Ly = log2(d) layers per
    site, the arrays have length L_mps * Ly, with ordering

        physical index = i_block * Ly + j_layer.

    The per-layer population is obtained by evaluating the registered
    ``Sz_j`` operator on the block site. This assumes the block site
    was constructed by ``_build_block_site`` in ``src/models.py``,
    which registers ``Sz_j`` for each layer j.
    """
    site = psi.sites[0]
    d = site.dim
    L_mps = psi.L

    if d == 2:
        sz_exp = np.real(psi.expectation_value('Sz'))
        sz_sigma = 2.0 * sz_exp
        p_down = np.clip(0.5 * (1.0 - sz_sigma), 0.0, 1.0)
        p_up = 1.0 - p_down
        return p_up, p_down

    Ly = int(round(np.log2(d)))
    if 2 ** Ly != d:
        raise ValueError(f"Site dimension {d} is not a power of 2")
    if Ly < 1:
        raise ValueError(f"Site dimension {d} is degenerate")

    p_up_2d = np.zeros((L_mps, Ly))
    for j in range(Ly):
        try:
            sz_j = np.real(psi.expectation_value(f'Sz_{j}'))
        except KeyError as exc:
            raise KeyError(
                f"Block site does not define 'Sz_{j}'. "
                f"was it built with _build_block_site?"
            ) from exc
        sz_sigma = 2.0 * sz_j
        p_down_j = np.clip(0.5 * (1.0 - sz_sigma), 0.0, 1.0)
        p_up_2d[:, j] = 1.0 - p_down_j

    p_up = p_up_2d.flatten()
    p_down = (1.0 - p_up_2d).flatten()
    return p_up, p_down


def _layer_operator(op_2x2, j_layer, Ly):
    """Block-site operator: op_2x2 on layer j_layer, identity elsewhere."""
    id2 = np.eye(2, dtype=complex)
    mats = [id2] * Ly
    mats[j_layer] = op_2x2
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _apply_physical_jump(psi, physical_idx, op_2x2):
    """
    Apply a single-physical-site operator on the given physical index.

    For d=2 MPS: physical_idx is the MPS site index.
    For block-site MPS: physical_idx = i_block * Ly + j_layer.
    """
    site = psi.sites[0]
    d = site.dim
    L_mps = psi.L

    if d == 2:
        i_block = physical_idx
        op_np = op_2x2
    else:
        Ly = int(round(np.log2(d)))
        i_block = physical_idx // Ly
        j_layer = physical_idx % Ly
        op_np = _layer_operator(op_2x2, j_layer, Ly)

    if not (0 <= i_block < L_mps):
        raise IndexError(
            f"physical_idx={physical_idx} maps to MPS site {i_block} "
            f"out of range [0, {L_mps})"
        )

    op = npc.Array.from_ndarray(
        op_np, [site.leg, site.leg.conj()], labels=['p', 'p*']
    )
    psi.apply_local_op(i_block, op, unitary=False, renormalize=True)


def site_populations(psi):
    """
    Return arrays p_up[i], p_down[i] of per-physical-site
    probabilities.

    Deprecated. Kept for backward compatibility with code that
    imports this name. The returned arrays have length
    ``L_mps * Ly`` where ``Ly = log2(d)`` is the number of physical
    layers per MPS site. For a spin-1/2 chain (d=2), this reduces to
    the standard length-``L`` array.

    New code should call ``_physical_site_populations`` directly.
    """
    return _physical_site_populations(psi)


def maybe_apply_jump(psi, gamma1, gamma2, dt, rng):
    """
    Sample the stochastic jump process and apply a jump if triggered.

    The total jump probability over dt is

        δp = dt * γ1 * Σ_i p_up(i)   +   dt * (γ2 / 2) * N_phys

    where the sum runs over physical sites and N_phys is the number of
    physical sites (equal to L_mps for a spin-1/2 chain, and
    L_mps * Ly for a cylinder). A uniform ε ∈ [0, 1) is drawn; if
    ε < δp, the jump type is chosen with probability proportional to
    the two contributions.

    T1 jump: choose a physical site i with probability
             p_up(i) / Σ_j p_up(j), apply σ_- on that site.
    T2 jump: choose a physical site uniformly, apply σ_z on that site.

    Both jump operators are applied as block-site operators when the
    MPS uses the cylinder block-site construction.
    """
    if gamma1 <= 0.0 and gamma2 <= 0.0:
        return False, None

    p_up, _ = _physical_site_populations(psi)
    total_up = float(p_up.sum())
    n_phys = len(p_up)

    delta_p_T1 = dt * gamma1 * total_up
    delta_p_T2 = dt * (gamma2 / 2.0) * n_phys
    delta_p = delta_p_T1 + delta_p_T2

    if delta_p <= 0.0:
        return False, None
    if delta_p > 1.0:
        delta_p = 1.0

    if rng.random() >= delta_p:
        return False, None

    prob_T1 = delta_p_T1 / max(delta_p_T1 + delta_p_T2, 1e-30)
    if rng.random() < prob_T1 and total_up > 0.0:
        site_weights = p_up / total_up
        physical_idx = int(rng.choice(n_phys, p=site_weights))
        sm_2x2 = np.array([[0, 0], [1, 0]], dtype=complex)  # |down><up|
        _apply_physical_jump(psi, physical_idx, sm_2x2)
        return True, 'T1'
    else:
        physical_idx = int(rng.integers(n_phys))
        sz_2x2 = np.array([[1, 0], [0, -1]], dtype=complex)
        _apply_physical_jump(psi, physical_idx, sz_2x2)
        return True, 'T2'
    

# ----------------------------------------------------------------------
# One full trajectory step
# ----------------------------------------------------------------------
def tjm_step(psi, model, gamma1, gamma2, dt, chi_max, rng):
    """
    Apply one full Strang-split TJM time step of size dt.

    Sequence:
        D[dt/2]  ->  U[dt]  ->  D[dt/2]  ->  stochastic jump

    The D[dt/2] at the end of step j and the D[dt/2] at the start of
    step j+1 combine to D[dt], giving the correct total dissipation
    per time step.
    """
    apply_dissipative_half_step(psi, gamma1, dt / 2.0)
    evolve_coherent_step(psi, model, dt, chi_max)
    apply_dissipative_half_step(psi, gamma1, dt / 2.0)
    applied, _ = maybe_apply_jump(psi, gamma1, gamma2, dt, rng)
    return applied


# ----------------------------------------------------------------------
# Full-trajectory evolution
# ----------------------------------------------------------------------
def evolve_trajectory(psi, model, total_time, gamma1, gamma2, dt_max,
                      chi_max, rng, measurement_times=None):
    """
    Evolve one trajectory for a signed total_time.

    Parameters
    ----------
    psi : MPS
        Initial state (modified in place).
    model : MPOModel
    total_time : float
        Signed total evolution time.
    gamma1, gamma2 : float
    dt_max : float
    chi_max : int
    rng : numpy.random.Generator
    measurement_times : iterable of float or None
        If given, positions are recorded as the trajectory passes
        through each time. The first entry must be 0.0.

    Returns
    -------
    (psi, jump_count)
        The evolved MPS and the number of jumps applied.
    """
    if abs(total_time) < 1e-15:
        return psi, 0
    dt_sign = np.sign(total_time)
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt = dt_sign * abs(total_time) / N_steps

    # Dissipative contraction and jumps always run forward in real time,
    # so we use abs(dt) for the dissipative and jump sub-steps even when
    # dt is negative (backward coherent evolution). This matches the
    # convention of the closed-system worker, where dt < 0 encodes
    # coherent time reversal only.
    dt_abs = abs(dt)

    jump_count = 0
    for step in range(N_steps):
        apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
        evolve_coherent_step(psi, model, dt, chi_max)
        apply_dissipative_half_step(psi, gamma1, dt_abs / 2.0)
        applied, _ = maybe_apply_jump(psi, gamma1, gamma2, dt_abs, rng)
        if applied:
            jump_count += 1
    return psi, jump_count