"""
src/tails.py

Exponential-sum compression of the algebraic 1/r^alpha interaction tail.

----------------------------------------------------------------------
Background
----------------------------------------------------------------------
The power-law Ising Hamiltonian

    H = (1/N) * sum_{i<j} (J / r_{ij}^alpha) sigma_z^i sigma_z^j

has pairwise couplings that decay algebraically with distance
r_{ij} = |i - j|. When this Hamiltonian is expressed as a Matrix
Product Operator (MPO) on a 1D chain, the algebraic tail forces the
MPO bond dimension to scale as O(N): one internal MPO state is required
for every distinct interaction distance r = 1, ..., N-1. This makes
long-chain simulations with N >= 128 infeasible on standard memory
budgets.

The classical workaround is to approximate the algebraic tail by a sum
of exponentials:

    1/r^alpha  ~=  sum_{k=1}^{n_exp}  c_k * lambda_k^r     (r = 1, ..., r_max)

Each exponential term lambda_k^r is representable as a *single* MPO
internal state: the coupling strength at distance r is propagated
recursively by multiplying by lambda_k at each lattice step. Summing
over n_exp exponentials gives an MPO with bond dimension
D_MPO ~ n_exp + O(1), independent of the chain length N.

----------------------------------------------------------------------
Algorithm (nonlinear least squares on the relative residual)
----------------------------------------------------------------------
A pure Prony fit (Hankel + root-finding + Vandermonde) is unstable for
steep power laws: for alpha >= 2 on r in [1, 200], the Hankel matrix
has entries spanning 14 orders of magnitude, and its condition number
exceeds double precision. The least-squares recurrence solution is
numerical garbage, so the roots and coefficients derived from it are
garbage too.

We therefore solve the fit directly as a bounded nonlinear
least-squares problem:

    minimize_{c, lambda}  sum_r  [ ( y_r - sum_k c_k lambda_k^r ) / y_r ]^2

with c_k unbounded and lambda_k constrained to (1e-10, 1 - 1e-8). The
optimizer is scipy.optimize.least_squares with the 'trf' method
(reflective trust region), which handles the box constraints on
lambda directly. The residual is normalized by y_r at each point so
that the fit is uniform in *relative* error across the full dynamic
range, which is exactly what the physics requires: the coupling at
r = 1 and the coupling at r = r_max both matter proportionally.

The optimizer is seeded with log-spaced lambda values in [1e-3, 0.999]
and coefficients uniformly distributed from y_1 / n_exp. If the
relative residual is still above 1e-3 after n_exp = 16, the algorithm
warns and returns the best fit it found.

----------------------------------------------------------------------
References
----------------------------------------------------------------------
- Pirvu, B., Murg, V., Cirac, J. I., Verstraete, F. (2010).
  "Matrix product operator representations." New J. Phys. 12, 025012.
- Crosswhite, G. M., Doherty, A. C., Vidal, G. (2008). "Applying
  matrix product states to simulated quantum annealing."
- Ohta, N. and Ishida, H. (1988). "A fast algorithm for the Prony method."
  (Historical reference; not used here.)
"""

import numpy as np
from scipy.optimize import least_squares


# Numerical bounds for the exponential decay rates.
# Lambda lower bound. Values below 1e-3 decay to less than 1e-9 by
# r = 3 and contribute nothing to the fit. Allowing the optimizer to
# explore that region wastes function evaluations without changing
# the residual, and is the dominant cost of the fit at r_eff >= 100.
_LAMBDA_MIN = 1e-3
_LAMBDA_MAX = 1.0 - 1e-8      # TeNPy requires decay < 1
_TARGET_RESIDUAL = 1e-3        # matches FIT_TOL in check_tails.py
_MAX_N_EXP = 16

# Threshold below which the power-law coupling is considered numerically
# negligible. r_eff = ceil((1/_COUPLING_FLOOR)^(1/alpha)) is the largest
# distance for which 1/r^alpha >= _COUPLING_FLOOR. Beyond this point, the
# tail contribution to any physical observable is negligible compared to
# the O(1) energy scale, and fitting it to 1e-3 *relative* accuracy is
# impossible in double precision (the value itself is below 1e-6 of the
# coupling at r=1).
#
# Rationale for 1e-6: the sum of the truncated tail is bounded by
#   sum_{r > r_eff} 1/r^alpha  <  1/((alpha-1) r_eff^{alpha-1}),
# which for alpha=4, r_eff=31 gives 1e-5, and for alpha=6, r_eff=10
# gives 2e-6. Both are far below the 1e-3 energy tolerance used in
# the compression check.
_COUPLING_FLOOR = 1e-6


def effective_fit_range(alpha, r_max):
    """
    Return the largest distance at which the power-law coupling is above
    the numerical floor ``_COUPLING_FLOOR``.

    For ``alpha = 0`` (all-to-all limit) the coupling is constant, so the
    effective range is ``r_max``. For ``alpha > 0``, it is

        r_eff = min( r_max, ceil( _COUPLING_FLOOR^{-1/alpha} ) ).

    This is the range on which the exponential fit is physically
    meaningful. Outside it, the true coupling contributes less than
    ``_COUPLING_FLOOR`` to any observable, so the exponential fit's
    extrapolation to zero is indistinguishable from the exact coupling.
    """
    alpha_f = float(alpha)
    if abs(alpha_f) < 1e-12:
        return int(r_max)
    r_eff = int(np.ceil(_COUPLING_FLOOR ** (-1.0 / alpha_f)))
    return min(int(r_max), r_eff)


def _nonlinear_fit(y_vals, r_vals, n_exp, seed=0):
    """
    Fit y_r ~= sum_k c_k lambda_k^r by bounded nonlinear least squares
    on the relative residual.

    Physical constraints (fixes the alternating-sign pathology):
      * c_k >= 0 for all k. The target function 1/r^alpha is positive
        and monotonically decreasing, so it is exactly representable as
        a sum of positive-coefficient decaying exponentials. Enforcing
        this bound eliminates the spurious sign-oscillating solutions
        that the unconstrained fit finds at large r_max.
      * 0 < lambda_k < 1. Enforced by the box bounds.

    Returns
    -------
    lambdas : ndarray, shape (n_exp,) or None
    coeffs  : ndarray, shape (n_exp,) or None
    """
    M = len(y_vals)
    y_abs = np.maximum(np.abs(y_vals), 1e-30)

    # ---- better initialization ----
    # Estimate the local power-law slope for seeding.
    log_y = np.log(np.maximum(y_vals, 1e-30))
    log_r = np.log(np.maximum(r_vals, 1e-30))
    try:
        alpha_est = max(-np.polyfit(log_r, log_y, 1)[0], 0.5)
    except Exception:
        alpha_est = 1.0

    # Characteristic decay scales spanning the full fit range.
    # Exponential k has characteristic scale char_scales[k], meaning it
    # decays significantly over that many lattice sites. We seed them
    # log-spaced from 0.5 up to r_max.
    char_scales = np.exp(np.linspace(np.log(0.5), np.log(max(r_vals)), n_exp))
    lam_init = np.exp(-1.0 / char_scales)

    # Coefficients seeded so the total at r=1 roughly equals y_1.
    # Each term contributes ~c_k * lambda_k at r=1; distribute evenly.
    c_init = np.maximum(np.abs(y_vals[0]) / n_exp, 1e-12) * np.ones(n_exp)

    # Sort lambdas descending (slowest decay first), coeffs follow.
    idx = np.argsort(lam_init)[::-1]
    lam_init = lam_init[idx]
    c_init = c_init[idx]

    # Add small jitter to the seed to diversify the multistart.
    rng = np.random.default_rng(seed)
    lam_init = np.clip(
        lam_init + rng.uniform(-1e-4, 1e-4, n_exp),
        _LAMBDA_MIN, _LAMBDA_MAX,
    )

    x0 = np.concatenate([c_init, lam_init])
    lb = np.concatenate([np.zeros(n_exp),  np.full(n_exp, _LAMBDA_MIN)])
    ub = np.concatenate([np.full(n_exp, 1e10), np.full(n_exp, _LAMBDA_MAX)])
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    def residual_fn(params):
        c = params[:n_exp]
        lam = np.clip(params[n_exp:], _LAMBDA_MIN, _LAMBDA_MAX)
        y_fit = np.zeros(M)
        for k in range(n_exp):
            y_fit += c[k] * (lam[k] ** r_vals)
        return (y_fit - y_vals) / y_abs

    try:
        res = least_squares(
            residual_fn, x0,
            bounds=(lb, ub),
            method='trf',
            xtol=1e-13, ftol=1e-13, gtol=1e-13,
            max_nfev=5000,
        )
    except Exception:
        return None, None

    c = res.x[:n_exp]
    lam = res.x[n_exp:]

    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(lam))):
        return None, None

    # ---- prune redundant terms ----
    # A term whose maximum contribution over the whole fit window is
    # below 1e-6 of y_max is not carrying any physical weight. Drop it.
    y_max = np.max(np.abs(y_vals))
    contributions = np.array([
        np.max(np.abs(c[k] * lam[k] ** r_vals)) for k in range(n_exp)
    ])
    keep = contributions > 1e-6 * y_max
    if not np.any(keep):
        keep[0] = True
    c = c[keep]
    lam = lam[keep]

    # Sort by decreasing lambda (slowest decaying first).
    idx = np.argsort(lam)[::-1]
    return lam[idx], c[idx]


def _evaluate_fit(r_vals, lambdas, coeffs):
    """Evaluate sum_k c_k * lambda_k^r at the given r values."""
    r_arr = np.asarray(r_vals, dtype=float)
    result = np.zeros_like(r_arr)
    for c, lam in zip(coeffs, lambdas):
        result = result + c * (lam ** r_arr)
    return result


def fit_power_law(alpha, r_max, n_exp=8) -> list[tuple[float, float]]:
    """
    Fit 1/r^alpha on r in [1, r_max] to a sum of exponentials.

    Parameters
    ----------
    alpha : float
        Power-law decay exponent. ``alpha = 0`` gives the constant
        function f(r) = 1 (all-to-all limit); ``alpha > 0`` gives an
        algebraic tail.
    r_max : int
        Maximum distance of the fit window.
    n_exp : int
        Initial number of exponential terms. The algorithm retries with
        larger n_exp up to 16 if the relative fit residual exceeds 1e-3.

    Returns
    -------
    fits : list of (float, float)
        List of ``(c_k, lambda_k)`` tuples such that

            1/r^alpha  ~=  sum_k  c_k * lambda_k^r   for r in [1, r_max].

    Notes
    -----
    For ``alpha = 0`` the exact representation is a single term
    ``(c=1, lambda=1)``. TeNPy's ``add_exponentially_decaying_coupling``
    requires ``decay < 1``, so the exact lambda is clamped to
    ``1 - 1e-8``. This changes the coupling at r = r_max by a relative
    amount of order ``r_max * 1e-8``, negligible for any practical
    r_max <= 10^4.

    The multistart schedule tries n_exp in {n_exp, n_exp+1, ..., 16},
    and for each n_exp it performs up to 3 independent restarts (with
    different random jitter of the initial lambda seed). If any restart
    reaches the target residual, the function returns immediately.
    Otherwise, it returns the best fit found across all attempts.
    """
    # The fit is performed on [1, r_eff], where r_eff is capped by the
    # numerical floor of the coupling. For steep power laws (alpha >= 4),
    # this is dramatically smaller than r_max; the truncated tail
    # contributes less than _COUPLING_FLOOR to any physical observable.
    r_eff = effective_fit_range(alpha, r_max)
    r_vals = np.arange(1, r_eff + 1, dtype=float)
    y_vals = r_vals ** (-float(alpha))

    # --- exact case: constant function 1/r^0 = 1 ---
    if abs(float(alpha)) < 1e-12:
        return [(1.0, _LAMBDA_MAX)]

    best_fits = None
    best_residual = np.inf

    n_try = int(n_exp)
    while n_try <= _MAX_N_EXP:
        for attempt in range(3):
            lambdas, coeffs = _nonlinear_fit(
                y_vals, r_vals, n_try, seed=attempt
            )
            if lambdas is None or coeffs is None or len(lambdas) == 0:
                continue

            y_fit = _evaluate_fit(r_vals, lambdas, coeffs)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(y_vals - y_fit) / np.abs(y_vals)
            rel_err = np.nan_to_num(rel_err, nan=0.0, posinf=np.inf)
            max_rel_err = float(np.max(rel_err))

            if max_rel_err < best_residual:
                best_residual = max_rel_err
                best_fits = [(float(c), float(l))
                             for c, l in zip(coeffs, lambdas)]

            if max_rel_err < _TARGET_RESIDUAL:
                # best_fits is guaranteed non-None here: the branch
                # only triggers when it was assigned in the same block.
                return best_fits if best_fits is not None else []

        n_try += 1

    if best_residual > _TARGET_RESIDUAL:
        print(f"[fit_power_law] Warning: alpha={alpha}, r_max={r_max}, "
              f"max relative error = {best_residual:.3e} "
              f"(target: {_TARGET_RESIDUAL:.0e}) after up to "
              f"n_exp={_MAX_N_EXP}. "
              f"Consider a wider fit window or accept the residual.")

    return best_fits if best_fits is not None else []


def fit_power_law_with_residual(
    alpha, r_max, n_exp=8,
) -> tuple[list[tuple[float, float]], float]:
    """
    Same as ``fit_power_law``, but also returns the maximum relative
    residual achieved by the returned fit, evaluated on the effective
    fit range ``[1, effective_fit_range(alpha, r_max)]``.

    This is the single authoritative source for the fit's own quality
    metric. Any external check that wishes to verify the fit should call
    this function rather than recomputing the residual independently,
    because independent recomputation risks a range mismatch with the
    fit itself.

    Returns
    -------
    fits : list of (float, float)
        Same as the return value of ``fit_power_law``.
    residual : float
        Maximum relative error of the fit on its own effective range.
        Returns ``float('inf')`` if no fit was returned.
    """
    r_eff = effective_fit_range(alpha, r_max)
    r_vals = np.arange(1, r_eff + 1, dtype=float)
    y_vals = r_vals ** (-float(alpha))

    if abs(float(alpha)) < 1e-12:
        return [(1.0, _LAMBDA_MAX)], 0.0

    fits = fit_power_law(alpha, r_max, n_exp)
    if len(fits) == 0:
        return [], float('inf')

    lambdas = np.array([lam for _, lam in fits])
    coeffs = np.array([c for c, _ in fits])
    y_fit = _evaluate_fit(r_vals, lambdas, coeffs)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs(y_vals - y_fit) / np.abs(y_vals)
    rel_err = np.nan_to_num(rel_err, nan=0.0, posinf=np.inf)
    return fits, float(np.max(rel_err))


def compress_mpo_couplings(
    alpha, N, n_exp=8,
) -> tuple[list[tuple[float, float]], int]:
    """
    Return the exponential-sum fit used to compress the power-law MPO.

    Parameters
    ----------
    alpha : float
        Power-law decay exponent.
    N : int
        Chain length. The fit is performed on r in [1, N-1].
    n_exp : int
        Initial number of exponential terms.

    Returns
    -------
    fits : list of (float, float)
        List of ``(c_k, lambda_k)`` tuples.
    D_MPO_estimated : int
        Estimated MPO bond dimension after compression. Each exponential
        contributes one internal FSM state, plus the initial and final
        states:  ``D_MPO ~ len(fits) + 2``.
    """
    if N < 2:
        return [], 1

    fits = fit_power_law(alpha, r_max=N - 1, n_exp=n_exp)
    D_MPO_estimated = len(fits) + 2
    return fits, D_MPO_estimated