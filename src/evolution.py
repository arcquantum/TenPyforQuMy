"""
src/evolution.py

Contains the state initialization and time-evolution wrappers 
for the tensor network metrology simulations.
"""

import numpy as np
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution

def initialize_css(lattice):
    """
    Initializes a Coherent Spin State (CSS) polarized along the x-axis.

    For a chain of standard spin-1/2 sites (local dimension 2), this is
    the product state |+>^{⊗N} with |+> = (|0> + |1>)/sqrt(2).

    For the 2D cylinder, the MPS has one site per axial position, and
    each site is a block site of local dimension d = 2^Ly representing
    the full circumference. The corresponding CSS is the uniform
    superposition over all d block-site basis states, with amplitude
    1/sqrt(d), which is exactly |+>^{⊗Ly} on the block.

    The function detects the local dimension from the lattice's site
    and builds the appropriate vector for each MPS tensor.
    """
    N = lattice.N_sites

    # Detect local dimension from the first MPS site. All sites of a
    # TeNPy lattice share the same Site object by construction.
    sites = lattice.mps_sites()
    d = sites[0].dim

    # Uniform superposition over all d basis states.
    # For d = 2 this reduces to (1, 1)/sqrt(2), reproducing the
    # standard spin-1/2 CSS. For d = 2^Ly this is the block-site CSS.
    plus_x_state = np.ones(d, dtype=np.complex128) / np.sqrt(float(d))
    product_state_list = [plus_x_state] * N

    psi = MPS.from_product_state(
        sites,
        product_state_list,
        bc=lattice.bc_MPS,
        dtype=np.complex128
    )

    return psi

def run_evolution_step(engine, N_steps=1):
    """
    Evolves the MPO engine forward in time.
    Relies entirely on the 'dt' parameter bound to the engine at initialization.
    """
    # Instruct the engine on how many steps to take in this specific run cycle
    engine.options['N_steps'] = N_steps
    
    # Modern TeNPy API: engine.run() automatically advances time by N_steps * dt
    engine.run()
    
    if hasattr(engine, 'trunc_err') and engine.trunc_err is not None:
        return engine.trunc_err.eps
    return 0.0

def _get_site_op(site, name):
    """
    Retrieve an on-site operator by name.

    Uses ``site.get_op(name)``, which is TeNPy's canonical accessor
    and works uniformly for both built-in site classes (SpinHalfSite,
    SpinSite, BosonSite, ...) and custom-built sites (such as the
    cylinder block site created by ``_build_block_site``).

    Attribute access (``site.Sx``) is intentionally not used here:
    TeNPy only auto-binds operators as Python attributes for a subset
    of built-in site classes, and custom sites do not expose them as
    attributes at all. The ``site.ops`` dict is also not used, because
    it is not a public interface in the installed TeNPy version.
    """
    try:
        return site.get_op(name)
    except (AttributeError, KeyError) as exc:
        available = getattr(site, 'opnames', None)
        raise KeyError(
            f"Site {site!r} does not define an operator named {name!r}. "
            f"Available operators: {available}"
        ) from exc


def apply_collective_rotation(psi, axis, angle):
    """
    Applies a collective rotation e^{-i * angle * S_axis} to the MPS.

    The operator is looked up via ``site.ops`` (see ``_get_site_op``),
    so this function works on both standard spin-1/2 chains and on
    the block-site cylinder construction. On the cylinder, ``Sx``,
    ``Sy``, ``Sz`` are the collective spins of a single block (i.e.,
    the summed spin over all Ly layers of the circumference); applying
    the rotation to every block therefore rotates the cylinder
    uniformly.
    """
    site = psi.sites[0]

    if isinstance(axis, str):
        if axis in ('x', 'y', 'z'):
            op = _get_site_op(site, f'S{axis}')
        else:
            raise ValueError("String axis must be 'x', 'y', or 'z'")
    else:
        n = np.array(axis, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("Rotation axis vector cannot be zero.")
        n = n / norm
        op = (n[0] * _get_site_op(site, 'Sx')
              + n[1] * _get_site_op(site, 'Sy')
              + n[2] * _get_site_op(site, 'Sz'))

    U = npc.expm(-1j * angle * op)

    for i in range(psi.L):
        psi.apply_local_op(i, U, unitary=True)


# ----------------------------------------------------------------------
# Additional helpers
# ----------------------------------------------------------------------

def evolve_mps_time(psi, model, total_time, chi_max, dt_max=0.05):
    """
    Evolve the MPS forward (positive total_time) or backward
    (negative total_time) using ExpMPOEvolution at the specified
    bond dimension.

    This is the consolidated version of the helper that was previously
    duplicated in every echo worker. All workers now import this
    definition, so a single implementation governs every MPS
    time-evolution call in the codebase.

    Parameters
    ----------
    psi : MPS
        State to evolve (in place).
    model : MPOModel
        Model whose Hamiltonian MPO generates the evolution.
    total_time : float
        Signed total evolution time. Negative times implement
        coherent time reversal.
    chi_max : int
        Maximum MPS bond dimension for SVD compression.
    dt_max : float
        Maximum time step. The evolution is split into
        ``ceil(|total_time| / dt_max)`` equal sub-steps.
    """
    if abs(total_time) < 1e-12:
        return
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt_step = total_time / N_steps  # preserves sign
    mpo_params = {
        'dt': dt_step,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12},
    }
    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = N_steps
    engine.run()

def apply_single_site_rotation(psi, site_idx, axis, angle):
    """
    Apply e^{-i angle S_axis} on a single MPS site.

    Parameters
    ----------
    psi : MPS
    site_idx : int
    axis : str, one of 'x', 'y', 'z'
    angle : float
    """
    site = psi.sites[site_idx]
    op = _get_site_op(site, f'S{axis}')
    U = npc.expm(-1j * angle * op)
    psi.apply_local_op(site_idx, U, unitary=True)

def apply_twist_about_k(psi, model, mu, k_axis, chi_max, dt_max=0.05):
    """
    Apply T_k(mu) = R_k T_z(mu) R_k^dagger to the MPS.

    Parameters
    ----------
    psi : MPS
    model : MPOModel (z-aligned power-law or cylinder)
    mu : float
        Twisting strength parameter. The inner z-twist is applied
        for evolution time ``mu / 2.0``, consistent with the
        codebase convention that the CLI parameters ``mu`` and
        ``nu`` are twice the evolution times passed to
        ``evolve_mps_time``. The symmetric workers pass
        ``(nu - mu) / 2.0`` to ``evolve_mps_time``; here we pass
        the full ``nu - mu`` and divide by 2 inside.
    k_axis : array-like, shape (3,) or None
        Unit vector. If None or numerically equal to (0, 0, 1),
        falls back to the plain z-twist.
    chi_max, dt_max : controls for the inner z-twist.
    """
    k = None if k_axis is None else np.asarray(k_axis, dtype=float)
    if k is None or np.allclose(k, [0.0, 0.0, 1.0]):
        evolve_mps_time(psi, model, mu / 2.0, chi_max, dt_max)
        return
    k = k / np.linalg.norm(k)

    # The rotation that maps z to k is
    #     R_k = e^{-i phi_k S_z} e^{-i theta_k S_y} ,
    # which in the operator-product sense is applied to a state as
    # e^{-i theta_k S_y} first, then e^{-i phi_k S_z}. The inverse is
    # R_k^dagger = e^{+i theta_k S_y} e^{+i phi_k S_z}.
    theta_k = np.arccos(np.clip(k[2], -1.0, 1.0))
    phi_k = np.arctan2(k[1], k[0])

    # Apply R_k^dagger: e^{+i phi_k S_z} first, then e^{+i theta_k S_y}.
    for i in range(psi.L):
        apply_single_site_rotation(psi, i, 'z', -phi_k)
    for i in range(psi.L):
        apply_single_site_rotation(psi, i, 'y', -theta_k)

    # Twist along z in the rotated frame, for time mu / 2.
    evolve_mps_time(psi, model, mu / 2.0, chi_max, dt_max)

    # Apply R_k: e^{-i theta_k S_y} first, then e^{-i phi_k S_z}.
    for i in range(psi.L):
        apply_single_site_rotation(psi, i, 'y', +theta_k)
    for i in range(psi.L):
        apply_single_site_rotation(psi, i, 'z', +phi_k)