"""
src/models.py

Contains the physical Hamiltonian architectures for the metrology simulations.
Strictly relies on Object-Oriented inheritance from TeNPy's CouplingModel and MPOModel.
"""

import inspect

import numpy as np
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain


def _call_exp_decay_coupling(model, strength, op_name, decay, op_name_j=None):
    """
    Call TeNPy's ``add_exponentially_decaying_coupling`` with whichever
    signature the installed version provides.

    Parameters
    ----------
    strength, decay : float
        Coupling strength and per-step decay factor.
    op_name : str
        Name of the operator at the left site.
    op_name_j : str or None
        Name of the operator at the right site. If None, defaults to
        op_name (symmetric coupling). Cylinder models pass distinct
        operator names for the two layers.

    TeNPy's API has changed across versions. Known variants:
      (1) ``(strength, lambda_, op_i, op_j, op_string=None, plus_hc=False,
           subsites=None, subsites_start=None)``
          — current TeNPy (>=0.9).
      (2) ``(strength, op1, op2, decay, shift1=0, shift2=0, ...)``
          — older TeNPy.
      (3) ``(strength, op, decay, shift=0, ...)``
          — single symmetric operator.

    We introspect the bound method's signature and dispatch accordingly.
    Raises a clear RuntimeError if the signature cannot be recognised.
    """
    if op_name_j is None:
        op_name_j = op_name

    try:
        sig = inspect.signature(model.add_exponentially_decaying_coupling)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot introspect add_exponentially_decaying_coupling: {exc}"
        )
    params = set(sig.parameters.keys())

    # Variant 1: current TeNPy (>=0.9)
    if {'strength', 'lambda_', 'op_i', 'op_j'}.issubset(params):
        return model.add_exponentially_decaying_coupling(
            strength=strength,
            lambda_=decay,
            op_i=op_name,
            op_j=op_name_j,
        )

    # Variant 2: older TeNPy
    if {'strength', 'op1', 'op2', 'decay'}.issubset(params):
        kwargs = {'strength': strength, 'op1': op_name, 'op2': op_name_j,
                  'decay': decay}
        if 'shift1' in params:
            kwargs['shift1'] = 0
        if 'shift2' in params:
            kwargs['shift2'] = 0
        return model.add_exponentially_decaying_coupling(**kwargs)

    # Variant 3: single symmetric operator. If op_name_j differs from
    # op_name, this variant cannot represent the term.
    if {'strength', 'op', 'decay'}.issubset(params):
        if op_name_j != op_name:
            raise RuntimeError(
                f"The installed TeNPy only supports the single-operator "
                f"form of add_exponentially_decaying_coupling, which "
                f"cannot represent an asymmetric coupling with "
                f"op_i={op_name!r} and op_j={op_name_j!r}."
            )
        kwargs = {'strength': strength, 'op': op_name, 'decay': decay}
        if 'shift' in params:
            kwargs['shift'] = 0
        return model.add_exponentially_decaying_coupling(**kwargs)

    raise RuntimeError(
        f"Unrecognised signature of add_exponentially_decaying_coupling. "
        f"Found parameters: {sorted(params)}. "
        f"Please report this to the developer."
    )


def _build_block_site(Ly):
    """
    Build a TeNPy Site of local dimension 2^Ly representing a full
    circumference of a cylinder with Ly layers.

    The block-site index is the binary string s_0 s_1 ... s_{Ly-1},
    corresponding to the tensor product of Ly spin-1/2 degrees of
    freedom. Operators defined on the site:

      Id, Sx_{j}, Sy_{j}, Sz_{j}            for j in [0, Ly),
      Sz_{j1}_{j2}  for j1 < j2             composite operators.

    The composite operators are used for the intra-block pair terms.
    All operators are Pauli-halves (S = sigma / 2), matching the
    convention of the 1D models.

    Construction follows the standard TeNPy pattern used by built-in
    site classes: the leg is passed as a single-element positional
    argument, and the operator dictionary is passed as plain numpy
    arrays. TeNPy's Site constructor wraps each numpy array into an
    npc.Array with the appropriate legs internally.
    """
    import numpy as np
    import tenpy.linalg.np_conserved as npc
    from tenpy.networks.site import Site

    d = 2 ** Ly
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)

    def _kron_at(op, j):
        """Kronecker product with `op` at layer j, identity elsewhere."""
        mats = [id2] * Ly
        mats[j] = op
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result

    # Operator naming convention on the block site:
    #
    #   Sx, Sy, Sz          -- collective spin of the whole block:
    #                          Sx = sum_j 0.5 * sigma_x^{j}   (and similarly
    #                          for y, z). These are the names the rest of
    #                          the codebase expects on a TeNPy Site, so
    #                          apply_collective_rotation(psi, 'x', ...) and
    #                          _get_3x3_covariance(psi) work on the block
    #                          site without modification. Summing over
    #                          all Lx blocks then gives the total
    #                          collective spin of the cylinder.
    #
    #   Sx_j, Sy_j, Sz_j    -- single-layer operators, layer j.
    #
    #   Sz_{j1}_{j2}        -- composite two-layer operators used by the
    #                          intra-block Ising coupling.
    #
    # TeNPy's Site constructor already registers 'Id' internally before
    # processing the **ops kwargs, so we must not pass it here.
    raw_ops = {}

    # Single-layer operators (needed for constructing the composites).
    for j in range(Ly):
        raw_ops[f'Sx_{j}'] = 0.5 * _kron_at(sx, j)
        raw_ops[f'Sy_{j}'] = 0.5 * _kron_at(sy, j)
        raw_ops[f'Sz_{j}'] = 0.5 * _kron_at(sz, j)

    # Composite two-layer operators for intra-block coupling.
    for j1 in range(Ly):
        for j2 in range(j1 + 1, Ly):
            raw_ops[f'Sz_{j1}_{j2}'] = raw_ops[f'Sz_{j1}'] @ raw_ops[f'Sz_{j2}']

    # Collective (block-summed) spin operators under the canonical
    # names Sx, Sy, Sz, so that the existing rotation and observable
    # code paths work unchanged on the cylinder.
    raw_ops['Sx'] = sum(raw_ops[f'Sx_{j}'] for j in range(Ly))
    raw_ops['Sy'] = sum(raw_ops[f'Sy_{j}'] for j in range(Ly))
    raw_ops['Sz'] = sum(raw_ops[f'Sz_{j}'] for j in range(Ly))

    leg = npc.LegCharge.from_trivial(d)
    state_labels = [format(i, f'0{Ly}b') for i in range(d)]

    # Pass the leg as a single LegCharge and the operators via **kwargs
    # so that TeNPy's Site.__init__(leg, state_labels, **ops) receives
    # each numpy array as an individual keyword argument named after
    # the operator. Site.add_op wraps each numpy array into an
    # npc.Array internally, which is the pattern used by every built-in
    # site class (e.g. SpinHalfSite).
    site = Site(leg, state_labels, **raw_ops)
    return site


class OATModel(CouplingModel, MPOModel):
    """
    One-Axis Twisting (OAT) Hamiltonian class.

    Simulates the interaction H = chi * S_z^2 on a 1D chain of N spin-1/2
    particles. Includes both the cross-terms and the on-site constant energy
    shift to ensure exact global phase matching during state-fidelity
    calculations.
    """

    def __init__(self, model_params):
        self.N = model_params.get('N', 32)
        self.chi_interaction = model_params.get('chi_interaction', 1.0)

        # Explicitly enforce no symmetry conservation.
        # The initial Coherent Spin State |+x> breaks U(1) Sz symmetry.
        conserve = model_params.get('conserve', None)
        if conserve is not None:
            raise ValueError("OAT dynamics on CSS require conserve=None to allow "
                             "superposition across Sz sectors.")

        site = SpinHalfSite(conserve=conserve)
        lattice = Chain(L=self.N, site=site, bc='open')
        assert self.N == lattice.N_sites, "Model N does not match lattice length."

        CouplingModel.__init__(self, lattice)
        self.init_terms()

        H_MPO = self.calc_H_MPO()
        MPOModel.__init__(self, lattice, H_MPO)

    def init_terms(self):
        """
        Maps the physical H = chi * S_z^2 operator into the tensor network
        topology.
        """
        # 1. Cross terms: 2 * chi * sum_{i<j} Sz_i Sz_j
        cross_term_strength = 2.0 * self.chi_interaction
        for dx in range(1, self.N):
            self.add_coupling(cross_term_strength, 0, 'Sz', 0, 'Sz', dx)

        # 2. On-site diagonal: chi * sum_i (Sz_i)^2 = chi * (1/4) * I
        onsite_strength = 0.25 * self.chi_interaction
        self.add_onsite(onsite_strength, 0, 'Id')


class PowerLawOATModel(CouplingModel, MPOModel):
    """
    Power-Law One-Axis Twisting (OAT) Hamiltonian.

        H = (1/N) * sum_{i<j} (J / r_{ij}^alpha) sigma_z^i sigma_z^j

    Supports two compression strategies for the algebraic interaction tail:

    - ``'exact'``: one ``add_coupling`` call per distance, giving an MPO
      bond dimension that scales as O(N). Only tractable for small N.
    - ``'exponential'`` (default): fit the tail 1/r^alpha to a sum of
      exponentials via ``src.tails.fit_power_law`` and register each via
      TeNPy's ``add_exponentially_decaying_coupling``. The MPO bond
      dimension becomes O(n_exp), independent of N.
    """

    def __init__(self, model_params):
        self.N = model_params.get('N', 32)
        self.J = model_params.get('J', 1.0)
        self.alpha = model_params.get('alpha', 0.0)
        self.compression_method = model_params.get('compression_method',
                                                   'exponential')

        if self.compression_method not in ('exact', 'exponential'):
            raise ValueError(
                f"compression_method must be 'exact' or 'exponential', "
                f"got {self.compression_method!r}"
            )

        conserve = model_params.get('conserve', None)
        if conserve is not None:
            raise ValueError("PowerLaw dynamics on CSS require conserve=None.")

        site = SpinHalfSite(conserve=conserve)
        lattice = Chain(L=self.N, site=site, bc='open')
        assert self.N == lattice.N_sites, "Model N does not match lattice length."

        CouplingModel.__init__(self, lattice)
        self.init_terms()

        H_MPO = self.calc_H_MPO()
        MPOModel.__init__(self, lattice, H_MPO)

        # --- diagnostic log ---
        print(f"[PowerLawOATModel] N={self.N}, alpha={self.alpha}, "
              f"J={self.J}, compression={self.compression_method}, "
              f"D_MPO_max={max(H_MPO.chi)}", flush=True)

    def init_terms(self):
        """
        Populate the CouplingModel with distance-dependent interactions
        matching the exact 1/N normalization requirement.
        """
        # TeNPy's 'Sz' is (1/2) sigma_z, so Sz*Sz = (1/4) sigma_z*sigma_z.
        # To map to J * sigma_z * sigma_z, multiply by 4.
        base_coupling = (4.0 * self.J) / self.N

        if self.compression_method == 'exact':
            for dx in range(1, self.N):
                r_ij_term = 1.0 / (dx ** self.alpha)
                coupling_strength = base_coupling * r_ij_term
                self.add_coupling(coupling_strength, 0, 'Sz', 0, 'Sz', dx)
            return

        # ---------------- exponential compression ----------------
        from src.tails import fit_power_law

        fits = fit_power_law(self.alpha, r_max=self.N - 1, n_exp=8)

        if len(fits) == 0:
            raise RuntimeError(
                f"Exponential fit for alpha={self.alpha}, N={self.N} "
                f"returned no terms. Consider compression_method='exact'."
            )

        print(f"[PowerLawOATModel] exponential fit for alpha={self.alpha}, "
              f"N={self.N}: {len(fits)} term(s):", flush=True)
        for i, (c_k, lam_k) in enumerate(fits):
            print(f"    term {i}:  c_k = {c_k:+.6e},  "
                  f"lambda_k = {lam_k:.10f}", flush=True)

        for c_k, lam_k in fits:
            _call_exp_decay_coupling(
                self,
                strength=base_coupling * c_k,
                op_name='Sz',
                decay=lam_k,
            )


class PowerLawCylinderModel(CouplingModel, MPOModel):
    """
    2D cylinder power-law Ising model.

        H = (1/N) sum_{k1<k2} (J / r(k1, k2)^alpha) sigma_z^{k1} sigma_z^{k2}

    Lattice: Lx (axial, open) x Ly (azimuthal, periodic). Sites are
    indexed linearly by k = i * Ly + j with i in [0, Lx),
    j in [0, Ly). The block-site construction encodes the full
    circumference in a single TeNPy Site of local dimension 2^Ly, so
    the MPS has Lx tensors and the couplings are:

      - Intra-block (within a single super-site): pairs (j1, j2) with
        minimum-image distance dy_min, added as on-site terms with
        strength (4J/N) / dy_min^alpha.
      - Inter-block (between super-sites at axial distance dx >= 1):
        for each layer pair (j1, j2), the axial dependence
        1/r(dx; dy_min)^alpha is fit to a sum of exponentials in dx,
        and each exponential is registered with
        `add_exponentially_decaying_coupling`.

    Valid for Ly <= 6. For Ly >= 7 the block-site dimension exceeds 128
    and the construction becomes intractable.

    See the class docstring for the geometric derivation
    and the justification for the per-dy_min fit.
    """

    def __init__(self, model_params):
        self.Lx = model_params.get('Lx', 4)
        self.Ly = model_params.get('Ly', 4)
        self.J = model_params.get('J', 1.0)
        self.alpha = model_params.get('alpha', 1.0)
        self.N = self.Lx * self.Ly

        conserve = model_params.get('conserve', None)
        if conserve is not None:
            raise ValueError(
                "PowerLawCylinderModel requires conserve=None; the initial "
                "Coherent Spin State breaks U(1) and Z2 symmetry."
            )

        if self.Ly > 6:
            raise ValueError(
                f"PowerLawCylinderModel supports Ly <= 6; got Ly={self.Ly}. "
                f"The block-site dimension would be 2^{self.Ly} = "
                f"{2 ** self.Ly}."
            )

        site = _build_block_site(self.Ly)
        lattice = Chain(L=self.Lx, site=site, bc='open')
        CouplingModel.__init__(self, lattice)
        self.init_terms()
        H_MPO = self.calc_H_MPO()
        MPOModel.__init__(self, lattice, H_MPO)

        print(f"[PowerLawCylinderModel] Lx={self.Lx}, Ly={self.Ly}, "
              f"N={self.N}, alpha={self.alpha}, J={self.J}, "
              f"D_MPO_max={max(H_MPO.chi)}", flush=True)

    def init_terms(self):
        base_coupling = (4.0 * self.J) / self.N

        # --- Intra-block couplings: same block, layers (j1, j2) ---
        for j1 in range(self.Ly):
            for j2 in range(j1 + 1, self.Ly):
                dy_abs = j2 - j1
                dy_min = min(dy_abs, self.Ly - dy_abs)
                r = float(dy_min)
                coupling = base_coupling / (r ** self.alpha)
                op_name = f'Sz_{j1}_{j2}'
                self.add_onsite(coupling, 0, op_name)

        # --- Inter-block couplings: layers (j1, j2) in adjacent blocks ---
        # The axial fit is cached per dy_min value. Each pair (j1, j2)
        # contributes n_exp(dy_min) MPO wires. Total bond dimension
        # contribution from inter-block is L_y^2 * n_exp_avg.
        fits_per_dy = {}
        for j1 in range(self.Ly):
            for j2 in range(self.Ly):
                dy_abs = abs(j1 - j2)
                dy_min = min(dy_abs, self.Ly - dy_abs)
                if dy_min not in fits_per_dy:
                    fits_per_dy[dy_min] = self._fit_axial_tail(dy_min)
                for c_k, lam_k in fits_per_dy[dy_min]:
                    _call_exp_decay_coupling(
                        self,
                        strength=base_coupling * c_k,
                        op_name=f'Sz_{j1}',
                        op_name_j=f'Sz_{j2}',
                        decay=lam_k,
                    )

    def _fit_axial_tail(self, dy_min):
        """
        Fit 1/(dx^2 + dy_min^2)^(alpha/2) to a sum of exponentials in dx
        over dx in [1, Lx-1]. Reuses the nonlinear least-squares fitter
        from src.tails.

        Returns
        -------
        fits : list of (float, float)
            (c_k, lambda_k) tuples.
        """
        import numpy as np
        from src.tails import _nonlinear_fit

        Lx = self.Lx
        if Lx < 2:
            return []

        dx_vals = np.arange(1, Lx, dtype=float)
        y_vals = 1.0 / ((dx_vals ** 2 + float(dy_min) ** 2)
                        ** (self.alpha / 2.0))

        # For Lx = 2 there is a single axial distance dx = 1.
        # A single exponential is exact: c * lambda = y_1 with any
        # choice of lambda. Use lambda = 0.5 for numerical stability.
        if Lx == 2:
            return [(float(y_vals[0] / 0.5), 0.5)]

        # For longer chains, do a multistart fit with increasing n_exp.
        n_max = min(8, Lx - 1)
        best_fits = None
        best_residual = np.inf
        for n_try in range(2, n_max + 1):
            for seed in range(2):
                lambdas, coeffs = _nonlinear_fit(
                    y_vals, dx_vals, n_try, seed=seed
                )
                if lambdas is None or coeffs is None or len(lambdas) == 0:
                    continue
                y_fit = np.array([
                    float(sum(c * (lam ** dx)
                              for c, lam in zip(coeffs, lambdas)))
                    for dx in dx_vals
                ])
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.abs(y_vals - y_fit) / np.abs(y_vals)
                rel = np.nan_to_num(rel, nan=0.0, posinf=np.inf)
                max_rel = float(np.max(rel))
                if max_rel < best_residual:
                    best_residual = max_rel
                    best_fits = [(float(c), float(l))
                                 for c, l in zip(coeffs, lambdas)]
                if max_rel < 1e-3:
                    return best_fits

        if best_fits is None:
            print(f"[PowerLawCylinderModel] Warning: axial fit for "
                  f"dy_min={dy_min}, alpha={self.alpha}, Lx={self.Lx} "
                  f"returned no valid terms.", flush=True)
            return []
        return best_fits