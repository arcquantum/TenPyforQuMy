"""
src/dM.py

Exact signal-response matrix M and covariance matrix Q.

    M_ij = -2 Im[ <u_i| S_j |psi_0> ]
    Q_ij = Re[ <psi_0| S_i S_j |psi_0> ] - <S_i><S_j>

with
    |psi_0> = T_k(nu-mu) T_z(mu) |CSS>
    |u_i>   = T_k(nu-mu) S_i T_z(mu) |CSS>

where T_z(mu) = e^{-i mu H / 2} and T_k(x) = R_k T_z(x) R_k^dag.

The formula for M is exact for a single pure state. In the open-
system workers it is applied per trajectory and averaged over the
ensemble.

Valid in the full Hilbert space for any alpha >= 0. The Dicke-basis
evaluation in src/dM_dicke.py is additionally available for OAT
and alpha = 0, but the general MPS formula here covers alpha > 0
as well.

The optional ``evolve_fn`` argument on exact_M lets callers supply a
custom backward-evolution function. The default is the deterministic
closed-system path (apply_twist_about_k). Open-system workers pass
a trajectory-specific function that:
  * applies dissipative contraction and quantum jumps in addition
    to the coherent step,
  * uses the same random seed across the four evolutions inside
    exact_M, so that the per-trajectory analytical identity holds
    for that trajectory,
and averages M over the trajectory ensemble.
"""

from __future__ import annotations
import numpy as np


def single_site_generator(axis):
    """Return the 2x2 matrix sigma_axis / 2."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli = {'x': sx, 'y': sy, 'z': sz}[axis]
    return 0.5 * pauli


def _build_collective_spin_MPO(model, axis):
    """
    Build an MPO for S_axis = sum_k S_axis^(k) on the same lattice
    as `model`. Uses the site operator registered under 'Sx', 'Sy',
    or 'Sz' (both the standard SpinHalfSite and the cylinder block
    site define these names).
    """
    from tenpy.models.model import CouplingModel
    lat = model.lat
    cm = CouplingModel(lat)
    # u = 0 with one site per unit cell adds the operator at every
    # site. This is the same pattern used by OATModel.init_terms.
    cm.add_onsite(1.0, 0, f'S{axis}')
    return cm.calc_H_MPO()


def apply_collective_spin(psi, model, axis, chi_max=128):
    """
    Apply S_axis = sum_k S_axis^(k) to psi, returning a new MPS.


    TeNPy's MPO.apply modifies its argument in place and returns a
    TruncationError, not the modified MPS. We therefore copy the
    input first, apply the MPO to the copy, discard the returned
    truncation error, and return the modified copy. The caller's
    state is preserved.


    The S_alpha MPO depends only on the model's lattice and site
    (not on the state or the bond dimension), so it is cached on the
    model instance. Rebuilding it on every call was dominating the
    runtime of exact_M in the validation sweep: the S_alpha MPO was
    being reconstructed dozens of times per grid point. The cache
    lives on the model object so it is garbage-collected with the
    model, avoiding any id() recycling hazard from a module-level
    cache.
    """
    cache = getattr(model, '_collective_spin_mpo_cache', None)
    if cache is None:
        cache = {}
        model._collective_spin_mpo_cache = cache
    if axis not in cache:
        cache[axis] = _build_collective_spin_MPO(model, axis)
    mpo = cache[axis]
    result = psi.copy()
    mpo.apply(
        result,
        options={
            'compression_method': 'SVD',
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-14},
        },
    )
    return result


def exact_M(psi_pre, model, k_axis, nu_minus_mu, chi_max, dt_max,
            evolve_fn=None):
    """
    Exact signal matrix M.

    Parameters
    ----------
    psi_pre : MPS
        State T_1 |CSS> right after the first twisting.
    model : MPOModel
    k_axis : array-like, shape (3,) or None
        Axis of the second twisting. None or (0,0,1) reduces to
        the two-axis case.
    nu_minus_mu : float
        Signed strength of the second twisting (Schulte convention
        T_z(x) = exp(-i x H / 2)).
    chi_max, dt_max : controls for the MPS evolution.
    evolve_fn : callable or None
        Custom backward-evolution function with signature
            evolve_fn(psi, strength, k_axis)  ->  None
        that evolves ``psi`` in place for coherent time
        ``strength / 2`` about the given twisting axis. The
        default (None) uses ``apply_twist_about_k``, which is the
        deterministic closed-system evolution. Open-system workers
        pass a trajectory-specific function that combines the
        coherent step with dissipative contraction and jumps and
        uses the same random seed across the four calls inside
        ``exact_M``.

    Returns
    -------
    M : ndarray, shape (3, 3)
    psi_0 : MPS
        State T_2 T_1 |CSS>, for the Q computation.
    """
    from src.evolution import apply_twist_about_k

    if evolve_fn is None:
        def _default_evolve(psi, strength, k_axis_):
            apply_twist_about_k(psi, model, strength, k_axis_,
                                chi_max, dt_max)
        resolved_evolve = _default_evolve
    else:
        resolved_evolve = evolve_fn

    # psi_0 = T_2 psi_pre
    psi_0 = psi_pre.copy()
    resolved_evolve(psi_0, nu_minus_mu, k_axis)

    axes = ['x', 'y', 'z']

    # u_i = T_2 S_i psi_pre
    u_states = {}
    for axis in axes:
        u = apply_collective_spin(psi_pre, model, axis, chi_max)
        resolved_evolve(u, nu_minus_mu, k_axis)
        u_states[axis] = u

    # S_j|psi_0> depends only on j, so precompute all three once.
    # The earlier version recomputed S_j|psi_0> inside the (i,j)
    # loop, giving 9 MPO applications where only 3 are needed.
    Sj_psi0 = {}
    for aj in axes:
        Sj_psi0[aj] = apply_collective_spin(psi_0, model, aj, chi_max)

    # M_ij = -2 Im[ <u_i| S_j |psi_0> ]
    M = np.zeros((3, 3))
    for i, ai in enumerate(axes):
        u_i = u_states[ai]
        for j, aj in enumerate(axes):
            val = u_i.overlap(Sj_psi0[aj])
            # Eq. (25) of exact_M_Q_derivation.pdf.
            # Equivalent to the commutator form
            #     M_ij = i <[S_i(mu), S_j(nu)]>
            # in the two-axis limit theta_k = phi_k = 0 (Section 4.6
            # of the derivation document). For general k the
            # commutator form does not extend, because T_k(nu-mu)
            # and T_z(mu) do not commute. The state-based form above
            # is therefore the primary one.
            M[i, j] = -2.0 * complex(val).imag
    return M, psi_0


def exact_Q(psi_0):
    """Exact covariance matrix Q on psi_0."""
    from src.observables import _get_3x3_covariance
    _, Cov = _get_3x3_covariance(psi_0)
    return Cov


def validate_against_finite_diff(psi_pre, model, k_axis,
                                 nu_minus_mu, chi_max, dt_max,
                                 delta=1e-5):
    """
    Compute M_exact and M_fd and return the Frobenius norm of the
    difference. The default delta is small enough that the O(delta^2)
    truncation error of the central difference is negligible against
    the achievable tolerance.
    """
    from src.evolution import apply_twist_about_k, apply_collective_rotation

    M_exact, _ = exact_M(psi_pre, model, k_axis, nu_minus_mu,
                         chi_max, dt_max)

    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']
    M_fd = np.zeros((3, 3))
    for i, axis in enumerate(axes):
        for sign in (+1, -1):
            psi_copy = psi_pre.copy()
            apply_collective_rotation(psi_copy, axis, sign * delta)
            apply_twist_about_k(psi_copy, model, nu_minus_mu, k_axis,
                                chi_max, dt_max)
            means = [np.real(np.sum(psi_copy.expectation_value(a)))
                     for a in op_names]
            for j in range(3):
                M_fd[i, j] += sign * means[j] / (2.0 * delta)

    return float(np.linalg.norm(M_exact - M_fd, 'fro'))