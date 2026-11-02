"""
scripts/checks/validate_1d_closed.py

the documentation of the codebase: validate the 1D closed-system TN worker against
exact diagonalization for alpha in {1, 2, 3, 4, 6} and N in {8, 12, 16}.

For each (alpha, N, chi) and each (mu, nu) on a grid, the script:

  - computes the TN SNR, M matrix and Q matrix via the production
    src.dM exact-derivative path (the default since the documentation);
  - computes the exact SNR, M matrix and Q matrix via a vectorized
    exact reference (equivalent to scripts/exact_echo_worker_1d.py,
    memory O(N * 2^N) instead of O(2^{2N}));
  - computes the state fidelity |<psi_TN|psi_exact>|^2 at three
    reference points (0,0), (pi/2,0), (pi/2,-pi/2);
  - records the four error metrics.

The script then:

  - writes the per-point table to data/validation_reports/<tag>1d_closed.csv
    (append mode, one row per grid point);
  - determines chi_conv per (alpha, N) from the four tolerances, writes
    to <tag>1d_closed_chi_conv.csv;
  - runs dt independence (0.02, 0.05, 0.10) at N=16, alpha=1, chi=64;
  - runs svd_min independence (1e-10, 1e-12, 1e-14) at N=16, alpha=1, chi=64;
  - verifies [S_x, S_y] = i S_z on a TN state.

Usage
-----
    # Quick sanity (~1-2 min)
    python scripts/checks/validate_1d_closed.py \
        --N 8 --alpha 1 --chi 4 --grid-size 5 \
        --skip-independence --skip-algebra --output-tag quick_

    # Small but nontrivial (~15-30 min)
    python scripts/checks/validate_1d_closed.py \
        --N 8 --alpha 1 3 --chi 4 16 64 --grid-size 10 \
        --skip-independence --output-tag N8_test_

    # Full sweep (hours, use SLURM or nohup)
    python scripts/checks/validate_1d_closed.py --grid-size 20

    # Split by N (recommended for SLURM)
    python scripts/checks/validate_1d_closed.py --N 8  --output-tag N8_
    python scripts/checks/validate_1d_closed.py --N 12 --output-tag N12_
    python scripts/checks/validate_1d_closed.py --N 16 --output-tag N16_
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import warnings

import numpy as np
import scipy.linalg

warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from src.models import PowerLawOATModel
from src.evolution import initialize_css, evolve_mps_time
from src.observables import _get_3x3_covariance
from src.dM import exact_M, exact_Q, apply_collective_spin
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
J = 1.0
DEFAULT_ALPHAS = [1.0, 2.0, 3.0, 4.0, 6.0]
DEFAULT_NS = [8, 12, 16]
DEFAULT_CHIS = [4, 8, 16, 32, 64]
DEFAULT_GRID = 20
DEFAULT_DT = 0.02

# Three reference points for the fidelity comparison: coherent
# state, exact echo, and OUT. Chosen because they exercise the
# three distinct protocols of the echo family.
REFERENCE_POINTS = [
    (0.0, 0.0),
    (np.pi / 2.0, 0.0),
    (np.pi / 2.0, -np.pi / 2.0),
]

# Convergence tolerances  of the codebase documentation.
TOL_EPS = 1e-3
TOL_FIDELITY = 1e-6
TOL_MATRIX = 1e-3

# Independence-check parameters.
DT_VALUES = [0.02, 0.05, 0.10]
SVDMIN_VALUES = [1e-10, 1e-12, 1e-14]
INDEP_N = 16
INDEP_ALPHA = 1.0
INDEP_CHI = 64
INDEP_GRID = 5

# Sanity-check parameters for the QuTiP cross-check.
QUTIP_N = 8
QUTIP_ALPHA = 3.0


# ----------------------------------------------------------------------
# Exact vectorized computation
# ----------------------------------------------------------------------
def build_H_diagonal(N, J, alpha):
    """
    Diagonal of H = (1/N) sum_{i<j} (J / |i-j|^alpha) sigma_z^i sigma_z^j.

    The Hamiltonian is diagonal in the computational basis, so we
    represent it as a length-2^N diagonal vector. Site 0 is the
    most-significant bit, matching TeNPy's MPS convention.

    Returns
    -------
    H_diag : ndarray, shape (2^N,), float
    sz     : ndarray, shape (2^N, N), int8
        sz[b, i] = +1 if bit i of b is 0, -1 if 1. This is the
        eigenvalue of sigma_z^i on the basis state b (up to the
        overall sign of sigma_z that we fix as +1 for |0>).
    """
    dim = 2 ** N
    b_arr = np.arange(dim, dtype=np.int64)
    sz = np.zeros((dim, N), dtype=np.int8)
    for i in range(N):
        bit = (b_arr >> (N - 1 - i)) & 1
        sz[:, i] = 1 - 2 * bit.astype(np.int8)

    H_diag = np.zeros(dim, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            r = j - i
            H_diag += (J / (N * (r ** alpha))) * sz[:, i].astype(float) * sz[:, j].astype(float)

    return H_diag, sz


def build_CSS_vector(N):
    """|+x>^{tensor N} = (1/sqrt(2^N)) * sum_b |b>."""
    return np.ones(2 ** N, dtype=complex) / np.sqrt(2 ** N)


def apply_Sz(v, sz, N):
    """S_z v where S_z = (1/2) sum_i sigma_z^i."""
    S_z_diag = 0.5 * sz.sum(axis=1).astype(float)
    return S_z_diag * v


def apply_Sx(v, N):
    """S_x v where S_x = (1/2) sum_i sigma_x^i."""
    out = np.zeros_like(v)
    b_arr = np.arange(2 ** N, dtype=np.int64)
    for i in range(N):
        mask = 1 << (N - 1 - i)
        b_flipped = b_arr ^ mask
        out += v[b_flipped]
    return 0.5 * out


def apply_Sy(v, N):
    """S_y v where S_y = (1/2) sum_i sigma_y^i."""
    out = np.zeros_like(v)
    b_arr = np.arange(2 ** N, dtype=np.int64)
    for i in range(N):
        mask = 1 << (N - 1 - i)
        b_flipped = b_arr ^ mask
        bit_i = (b_arr >> (N - 1 - i)) & 1
        # <0|sigma_y|1> = -i, <1|sigma_y|0> = +i
        coeff = np.where(bit_i == 0, -1j, +1j)
        out += coeff * v[b_flipped]
    return 0.5 * out


def apply_S(v, axis, sz, N):
    if axis == 'z':
        return apply_Sz(v, sz, N)
    if axis == 'x':
        return apply_Sx(v, N)
    if axis == 'y':
        return apply_Sy(v, N)
    raise ValueError(f"Unknown axis {axis!r}")


def apply_exp_phase_imprint(v, axis, delta, sz, N, order=4):
    """
    exp(-i delta S_axis) v via Taylor expansion.

    For delta = 1e-3 and order = 4, the truncation error is
    O(delta^5) ~ 1e-15, far below the finite-difference noise.
    """
    result = v.copy()
    term = v.copy()
    for k in range(1, order + 1):
        term = apply_S(term, axis, sz, N)
        result += ((-1j * delta) ** k / math.factorial(k)) * term
    return result


def exact_snr_and_matrices(N, J, alpha, mu, nu, delta=1e-3):
    """
    Vectorized exact SNR, M, Q, psi_pre, psi_q.

    Equivalent to running scripts/exact_echo_worker_1d.py at
    (N, alpha, J, mu, nu) with k_axis = z, but with memory
    O(N * 2^N) instead of O(2^{2N}).
    """
    H_diag, sz = build_H_diagonal(N, J, alpha)
    psi_0 = build_CSS_vector(N)

    # Forward twist: exp(-i (mu/2) H) psi_0
    psi_pre = np.exp(-1j * (mu / 2.0) * H_diag) * psi_0

    # Backward twist: exp(-i ((nu - mu)/2) H) applied to a vector
    t_bwd = (nu - mu) / 2.0

    def apply_backward(v):
        return np.exp(-1j * t_bwd * H_diag) * v

    psi_q = apply_backward(psi_pre)

    # M via central differences on the phase imprint.
    axes = ['x', 'y', 'z']
    M = np.zeros((3, 3))
    for i, axis in enumerate(axes):
        psi_pre_plus = apply_exp_phase_imprint(psi_pre, axis, +delta, sz, N)
        psi_pre_minus = apply_exp_phase_imprint(psi_pre, axis, -delta, sz, N)
        psi_plus = apply_backward(psi_pre_plus)
        psi_minus = apply_backward(psi_pre_minus)
        for j, axis_j in enumerate(axes):
            val_plus = np.real(np.vdot(psi_plus, apply_S(psi_plus, axis_j, sz, N)))
            val_minus = np.real(np.vdot(psi_minus, apply_S(psi_minus, axis_j, sz, N)))
            M[i, j] = (val_plus - val_minus) / (2.0 * delta)

    # Q on psi_q.
    S_mean = np.array([
        np.real(np.vdot(psi_q, apply_S(psi_q, a, sz, N))) for a in axes
    ])
    Q = np.zeros((3, 3), dtype=complex)
    for i, ai in enumerate(axes):
        S_i_psi = apply_S(psi_q, ai, sz, N)
        for j, aj in enumerate(axes):
            corr = np.vdot(S_i_psi, apply_S(psi_q, aj, sz, N))
            Q[i, j] = corr - S_mean[i] * S_mean[j]
    Q = np.real(0.5 * (Q + Q.conj().T))

    snr = _snr_from_M_Q(M, Q)
    return {
        'snr': snr, 'M': M, 'Q': Q,
        'psi_pre': psi_pre, 'psi_q': psi_q,
        'H_diag': H_diag, 'sz': sz,
    }


# ----------------------------------------------------------------------
# TN computation
# ----------------------------------------------------------------------
def _snr_from_M_Q(M, Q):
    """SNR = sigma_max(M Q^{-1/2}), with mask-based null-eigenvalue
    handling matching the production workers."""
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T
    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    return float(np.max(sv))


def mps_to_vector(mps, N):
    """Contract an MPS to a length-2^N state vector (C-ordering, site 0 MSB)."""
    mps.canonical_form()
    theta = mps.get_theta(0, n=N)
    arr = theta.to_ndarray()
    while arr.ndim > N and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.reshape(2 ** N)


def tn_sweep_full(N, alpha, chi, dt, points, verbose=True):
    """
    Compute TN results at the specific (mu, nu) points given.

    The model and the forward twist depend only on mu, so both are
    cached per mu-row. Points are grouped by mu (rounded to 6
    decimals), so reference points outside the grid are evaluated
    in the same mu-row as their nearest grid points and share the
    forward-twist state.

    Keys in the returned dict are (round(float(mu), 6),
    round(float(nu), 6)), matching the lookup convention in
    compute_and_write_one_combo.
    """
    from collections import defaultdict

    model = PowerLawOATModel({
        'N': N, 'J': J, 'alpha': alpha, 'conserve': None,
        'compression_method': 'exponential',
    })

    by_mu = defaultdict(list)
    for (mu, nu) in points:
        by_mu[round(float(mu), 6)].append(round(float(nu), 6))

    results = {}
    mu_rows = sorted(by_mu.keys())
    n_mu = len(mu_rows)
    for i_mu, mu_r in enumerate(mu_rows):
        if verbose and (i_mu % max(1, n_mu // 5) == 0 or i_mu == n_mu - 1):
            print(f"    mu row {i_mu + 1}/{n_mu} (mu = {mu_r:.4f})",
                  flush=True)

        try:
            psi_base = initialize_css(model.lat)
            evolve_mps_time(psi_base, model, mu_r / 2.0, chi, dt)
        except Exception as exc:
            print(f"    [WARN] forward twist failed at mu={mu_r:.4f}: {exc}",
                  flush=True)
            for nu_r in by_mu[mu_r]:
                results[(mu_r, nu_r)] = {'failed': True,
                                          'error': str(exc)}
            continue

        for nu_r in by_mu[mu_r]:
            try:
                M, psi_q = exact_M(
                    psi_base, model, k_axis=None,
                    nu_minus_mu=nu_r - mu_r,
                    chi_max=chi, dt_max=dt,
                )
                Q = exact_Q(psi_q)
                snr = _snr_from_M_Q(M, Q)
                results[(mu_r, nu_r)] = {
                    'snr': snr, 'M': M, 'Q': Q, 'psi_q': psi_q,
                    'failed': False,
                }
            except Exception as exc:
                print(f"    [FAIL-TN] mu={mu_r:.4f} nu={nu_r:.4f}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                results[(mu_r, nu_r)] = {'failed': True,
                                          'error': str(exc)}

    return results


def evolve_with_svdmin(psi, model, total_time, chi_max, dt_max, svd_min):
    """Same as src.evolution.evolve_mps_time but with parametrized svd_min."""
    if abs(total_time) < 1e-12:
        return
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt_step = total_time / N_steps
    mpo_params = {
        'dt': dt_step,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': svd_min},
    }
    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = N_steps
    engine.run()


def tn_snr_with_svdmin(N, alpha, chi, dt, mu, nu, svd_min):
    """
    Single-point TN SNR with parametrized svd_min. Replicates the
    exact_M path from src.dM but with evolve_mps_time replaced by
    evolve_with_svdmin.
    """
    model = PowerLawOATModel({
        'N': N, 'J': J, 'alpha': alpha, 'conserve': None,
        'compression_method': 'exponential',
    })
    psi_pre = initialize_css(model.lat)
    evolve_with_svdmin(psi_pre, model, mu / 2.0, chi, dt, svd_min)

    nu_minus_mu = nu - mu

    def _evolve(v):
        evolve_with_svdmin(v, model, nu_minus_mu / 2.0, chi, dt, svd_min)

    psi_q = psi_pre.copy()
    _evolve(psi_q)

    u_states = {}
    for axis in ['x', 'y', 'z']:
        u = apply_collective_spin(psi_pre, model, axis, chi)
        _evolve(u)
        u_states[axis] = u

    M = np.zeros((3, 3))
    for i, ai in enumerate(['x', 'y', 'z']):
        for j, aj in enumerate(['x', 'y', 'z']):
            Sj_psi = apply_collective_spin(psi_q, model, aj, chi)
            val = u_states[ai].overlap(Sj_psi)
            M[i, j] = -2.0 * complex(val).imag

    _, Q = _get_3x3_covariance(psi_q)
    return {'snr': _snr_from_M_Q(M, Q)}


# ----------------------------------------------------------------------
# Reference-point utility
# ----------------------------------------------------------------------
def all_evaluation_points(mu_vals, nu_vals):
    """
    Union of the grid points and the three reference points,
    deduplicated to 6 decimals.
    """
    points = set()
    for mu in mu_vals:
        for nu in nu_vals:
            points.add((round(float(mu), 6), round(float(nu), 6)))
    for (m, n) in REFERENCE_POINTS:
        points.add((round(float(m), 6), round(float(n), 6)))
    return sorted(points)


def is_reference_point(mu, nu, tol=1e-4):
    return any(abs(mu - m) < tol and abs(nu - n) < tol
               for (m, n) in REFERENCE_POINTS)


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------
MAIN_CSV_FIELDS = [
    'alpha', 'N', 'chi_max', 'dt', 'mu', 'nu',
    'snr_tn', 'snr_exact', 'eps_rel',
    'one_minus_fidelity', 'M_err', 'Q_err',
]


def compute_and_write_one_combo(
    csv_path, alpha, N, chi, dt, mu_vals, nu_vals,
    write_header, verbose,
):
    """
    Compute one (alpha, N, chi) combination and append rows to csv_path.
    Returns (rows, n_failed).
    """
    print(f"\n[combo] alpha={alpha}, N={N}, chi={chi}, dt={dt}",
          flush=True)
    t0 = time.time()

    points = all_evaluation_points(mu_vals, nu_vals)
    print(f"  {len(points)} grid points "
          f"({len(mu_vals) * len(nu_vals)} from grid + reference points)",
          flush=True)

    # TN sweep — compute all points, including reference points
    # outside the grid, so that tn_results covers the full set.
    tn_results = tn_sweep_full(N, alpha, chi, dt, points,
                                verbose=verbose)

    rows = []
    n_failed = 0
    n_finite_tn = 0

    for (mu, nu) in points:
        # TN
        tn = tn_results.get((round(float(mu), 6), round(float(nu), 6)))
        if tn is None or tn.get('failed'):
            err_msg = (tn.get('error', '<no error message>')
                       if tn is not None else '<no result>')
            print(f"  [FAIL-TN] mu={mu:.4f} nu={nu:.4f}: "
                  f"{err_msg}", flush=True)
            tn_snr = float('nan')
            tn_M = np.full((3, 3), np.nan)
            tn_Q = np.full((3, 3), np.nan)
            tn_failed = True
        else:
            tn_snr = tn['snr']
            tn_M = tn['M']
            tn_Q = tn['Q']
            tn_failed = not np.isfinite(tn_snr)
        if not tn_failed:
            n_finite_tn += 1

        # Exact
        try:
            ex = exact_snr_and_matrices(N, J, alpha, float(mu), float(nu))
            ex_snr = ex['snr']
            ex_M = ex['M']
            ex_Q = ex['Q']
        except Exception as exc:
            print(f"  [WARN] exact failed at mu={mu:.4f} nu={nu:.4f}: "
                  f"{exc}", flush=True)
            ex_snr = float('nan')
            ex_M = np.full((3, 3), np.nan)
            ex_Q = np.full((3, 3), np.nan)

        # Errors
        if np.isfinite(tn_snr) and np.isfinite(ex_snr):
            eps_rel = abs(tn_snr - ex_snr) / max(abs(ex_snr), 1e-30)
        else:
            eps_rel = float('nan')
            n_failed += 1

        if not tn_failed and np.all(np.isfinite(ex_M)):
            M_ref = np.linalg.norm(ex_M, 'fro')
            M_err = np.linalg.norm(tn_M - ex_M, 'fro') / max(M_ref, 1e-12)
        else:
            M_err = float('nan')

        if not tn_failed and np.all(np.isfinite(ex_Q)):
            Q_ref = np.linalg.norm(ex_Q, 'fro')
            Q_err = np.linalg.norm(tn_Q - ex_Q, 'fro') / max(Q_ref, 1e-12)
        else:
            Q_err = float('nan')

        # Fidelity at reference points
        one_minus_fid = float('nan')
        if (is_reference_point(float(mu), float(nu))
                and not tn_failed and tn is not None):
            try:
                v_tn = mps_to_vector(tn['psi_q'], N)
                overlap = abs(np.vdot(v_tn, ex['psi_q'])) ** 2
                one_minus_fid = 1.0 - float(overlap)
            except Exception as exc:
                print(f"  [WARN] fidelity failed at mu={mu:.4f} "
                      f"nu={nu:.4f}: {exc}", flush=True)

        rows.append({
            'alpha': alpha, 'N': N, 'chi_max': chi, 'dt': dt,
            'mu': f"{float(mu):.6f}", 'nu': f"{float(nu):.6f}",
            'snr_tn': f"{tn_snr:.10e}" if np.isfinite(tn_snr) else 'nan',
            'snr_exact': f"{ex_snr:.10e}" if np.isfinite(ex_snr) else 'nan',
            'eps_rel': f"{eps_rel:.6e}" if np.isfinite(eps_rel) else 'nan',
            'one_minus_fidelity': (f"{one_minus_fid:.6e}"
                                    if np.isfinite(one_minus_fid) else 'nan'),
            'M_err': f"{M_err:.6e}" if np.isfinite(M_err) else 'nan',
            'Q_err': f"{Q_err:.6e}" if np.isfinite(Q_err) else 'nan',
        })

    # Write rows
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=MAIN_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    elapsed = time.time() - t0
    print(f"[combo done] alpha={alpha}, N={N}, chi={chi}: "
          f"{n_finite_tn}/{len(points)} finite TN, "
          f"{n_failed} failed, {elapsed:.1f}s",
          flush=True)
    return rows, n_failed


def determine_chi_conv(rows, chis, alphas, Ns, csv_path):
    """
    From the per-point rows, determine the smallest chi per (alpha, N)
    that satisfies all four criteria, and write the summary.
    """
    summary_fields = ['alpha', 'N', 'chi_conv', 'eps_max', 'fid_max',
                       'M_max', 'Q_max', 'status', 'n_finite', 'n_total']

    summary_rows = []
    for alpha in alphas:
        for N in Ns:
            chi_found = None
            chi_stats = {}
            for chi in sorted(chis):
                sub = [r for r in rows
                       if float(r['alpha']) == alpha
                       and int(r['N']) == N
                       and int(r['chi_max']) == chi]
                if not sub:
                    continue
                eps_vals = [_to_float(r['eps_rel']) for r in sub]
                M_vals = [_to_float(r['M_err']) for r in sub]
                Q_vals = [_to_float(r['Q_err']) for r in sub]
                fid_vals = [_to_float(r['one_minus_fidelity']) for r in sub]
                eps_max = _safe_max(eps_vals)
                M_max = _safe_max(M_vals)
                Q_max = _safe_max(Q_vals)
                fid_max = _safe_max(fid_vals) if any(
                    np.isfinite(v) for v in fid_vals) else 0.0

                n_finite = sum(
                    1 for v in eps_vals if np.isfinite(v)
                )
                fraction = n_finite / max(len(sub), 1)

                chi_stats[chi] = {
                    'eps_max': eps_max, 'fid_max': fid_max,
                    'M_max': M_max, 'Q_max': Q_max,
                    'n_finite': n_finite,
                    'n_total': len(sub),
                    'fraction': fraction,
                }

                # Require that at least 95% of the grid points
                # produced a finite SNR. Without this guard, a
                # single successful point trivially satisfies all
                # four tolerances (because missing fidelity values
                # are treated as 0.0), and the selector would
                # misreport convergence on a nearly empty grid.
                MIN_FINITE_FRACTION = 0.95
                if fraction < MIN_FINITE_FRACTION:
                    print(f"  [chi={chi}, alpha={alpha}, N={N}]: "
                          f"only {n_finite}/{len(sub)} finite "
                          f"({fraction*100:.1f}%), skipping (need "
                          f">= {MIN_FINITE_FRACTION*100:.0f}%)",
                          flush=True)
                    continue

                if (np.isfinite(eps_max) and eps_max < TOL_EPS
                        and np.isfinite(fid_max) and fid_max < TOL_FIDELITY
                        and np.isfinite(M_max) and M_max < TOL_MATRIX
                        and np.isfinite(Q_max) and Q_max < TOL_MATRIX):
                    chi_found = chi
                    break

            if chi_found is not None:
                st = chi_stats[chi_found]
                summary_rows.append({
                    'alpha': alpha, 'N': N, 'chi_conv': chi_found,
                    'eps_max': f"{st['eps_max']:.6e}",
                    'fid_max': f"{st['fid_max']:.6e}",
                    'M_max': f"{st['M_max']:.6e}",
                    'Q_max': f"{st['Q_max']:.6e}",
                    'status': 'converged',
                    'n_finite': st.get('n_finite', ''),
                    'n_total': st.get('n_total', ''),
                })
            else:
                # Report the largest chi's statistics as "not converged"
                largest = sorted(chi_stats.keys())[-1] if chi_stats else None
                if largest is not None:
                    st = chi_stats[largest]
                    summary_rows.append({
                        'alpha': alpha, 'N': N, 'chi_conv': 'NOT_CONVERGED',
                        'n_finite': st.get('n_finite', ''),
                        'n_total': st.get('n_total', ''),
                        'eps_max': f"{st['eps_max']:.6e}"
                                    if np.isfinite(st['eps_max']) else 'nan',
                        'fid_max': f"{st['fid_max']:.6e}"
                                    if np.isfinite(st['fid_max']) else 'nan',
                        'M_max': f"{st['M_max']:.6e}"
                                  if np.isfinite(st['M_max']) else 'nan',
                        'Q_max': f"{st['Q_max']:.6e}"
                                  if np.isfinite(st['Q_max']) else 'nan',
                        'status': f'not_converged_at_chi{largest}',
                    })

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"\nWrote {csv_path} ({len(summary_rows)} rows)")
    for r in summary_rows:
        print(f"  alpha={r['alpha']}, N={r['N']}: "
              f"chi_conv={r['chi_conv']}  "
              f"(eps={r['eps_max']}, fid={r['fid_max']}, "
              f"M={r['M_max']}, Q={r['Q_max']})")
    return summary_rows


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def _safe_max(vals):
    finite = [v for v in vals if np.isfinite(v)]
    return max(finite) if finite else float('nan')


# ----------------------------------------------------------------------
# dt and svd_min independence checks
# ----------------------------------------------------------------------
def dt_independence_check(alphas, N, chi, grid_size, output_dir, tag):
    print(f"\n=== dt independence at N={N}, alpha={alphas}, chi={chi} ===",
          flush=True)
    mu_vals = np.linspace(0.0, np.pi, grid_size)
    nu_vals = np.linspace(-np.pi, np.pi, grid_size)

    rows = []
    for alpha in alphas:
        snrs_by_dt = {}
        for dt in DT_VALUES:
            print(f"  alpha={alpha}, dt={dt}", flush=True)
            points_grid = [(float(m), float(n))
                           for m in mu_vals for n in nu_vals]
            tn = tn_sweep_full(N, alpha, chi, dt, points_grid,
                                verbose=False)
            snrs = []
            for mu in mu_vals:
                for nu in nu_vals:
                    r = tn.get((round(float(mu), 6), round(float(nu), 6)))
                    if r and not r.get('failed'):
                        snrs.append(r['snr'])
                    else:
                        snrs.append(float('nan'))
            snrs_by_dt[dt] = np.array(snrs)

        ref = snrs_by_dt[DT_VALUES[0]]
        for dt in DT_VALUES[1:]:
            arr = snrs_by_dt[dt]
            mask = np.isfinite(ref) & np.isfinite(arr) & (np.abs(ref) > 1e-12)
            if mask.any():
                rel = float(np.max(np.abs(arr[mask] - ref[mask])
                                    / np.abs(ref[mask])))
            else:
                rel = float('nan')
            rows.append({
                'N': N, 'alpha': alpha, 'chi_max': chi,
                'dt_ref': DT_VALUES[0], 'dt_test': dt,
                'max_rel_diff': f"{rel:.6e}" if np.isfinite(rel) else 'nan',
            })

    path = os.path.join(output_dir, f'{tag}1d_closed_dt_independence.csv')
    fields = ['N', 'alpha', 'chi_max', 'dt_ref', 'dt_test', 'max_rel_diff']
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {path}")
    return rows


def svdmin_independence_check(alphas, N, chi, grid_size, output_dir, tag):
    print(f"\n=== svd_min independence at N={N}, alpha={alphas}, "
          f"chi={chi} ===", flush=True)
    mu_vals = np.linspace(0.0, np.pi, grid_size)
    nu_vals = np.linspace(-np.pi, np.pi, grid_size)
    dt = DT_VALUES[0]

    rows = []
    for alpha in alphas:
        snrs_by_svd = {}
        for svd_min in SVDMIN_VALUES:
            print(f"  alpha={alpha}, svd_min={svd_min:.1e}", flush=True)
            snrs = []
            for mu in mu_vals:
                for nu in nu_vals:
                    try:
                        r = tn_snr_with_svdmin(N, alpha, chi, dt,
                                                float(mu), float(nu),
                                                svd_min)
                        snrs.append(r['snr'])
                    except Exception as exc:
                        print(f"    [WARN] {exc}", flush=True)
                        snrs.append(float('nan'))
            snrs_by_svd[svd_min] = np.array(snrs)

        ref = snrs_by_svd[SVDMIN_VALUES[0]]
        for svd_min in SVDMIN_VALUES[1:]:
            arr = snrs_by_svd[svd_min]
            mask = np.isfinite(ref) & np.isfinite(arr) & (np.abs(ref) > 1e-12)
            if mask.any():
                rel = float(np.max(np.abs(arr[mask] - ref[mask])
                                    / np.abs(ref[mask])))
            else:
                rel = float('nan')
            rows.append({
                'N': N, 'alpha': alpha, 'chi_max': chi,
                'svd_min_ref': f"{SVDMIN_VALUES[0]:.1e}",
                'svd_min_test': f"{svd_min:.1e}",
                'max_rel_diff': f"{rel:.6e}" if np.isfinite(rel) else 'nan',
            })

    path = os.path.join(output_dir, f'{tag}1d_closed_svdmin_independence.csv')
    fields = ['N', 'alpha', 'chi_max', 'svd_min_ref', 'svd_min_test',
              'max_rel_diff']
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {path}")
    return rows


# ----------------------------------------------------------------------
# Algebra check
# ----------------------------------------------------------------------
def algebra_check(N=16, alpha=1.0, chi=64, dt=0.02, output_dir=None, tag=''):
    """
    Verify [S_a, S_b] = i eps_{abc} S_c on a TN state, for all three
    cyclic permutations.
    """
    print(f"\n=== Algebra check: [S_a, S_b] = i eps_{{abc}} S_c ===",
          flush=True)
    print(f"    N={N}, alpha={alpha}, chi={chi}, dt={dt}")

    model = PowerLawOATModel({
        'N': N, 'J': J, 'alpha': alpha, 'conserve': None,
        'compression_method': 'exponential',
    })
    psi = initialize_css(model.lat)
    evolve_mps_time(psi, model, np.pi / 4.0, chi, dt)

    rels = [
        ('Sx', 'Sy', 'Sz', 1.0),
        ('Sy', 'Sz', 'Sx', 1.0),
        ('Sz', 'Sx', 'Sy', 1.0),
    ]

    rows = []
    for a, b, c, sign in rels:
        corr_ab = np.sum(psi.correlation_function(a, b))
        corr_ba = np.sum(psi.correlation_function(b, a))
        lhs = corr_ab - corr_ba
        rhs = 1j * sign * np.sum(psi.expectation_value(c))
        diff = abs(lhs - rhs)
        # The relative-error scale is max(|rhs|, |lhs|, 1.0), not
        # max(|rhs|, 1e-30). The 1e-30 floor is meaningless for
        # expectation values: when <S_c> = 0 on the state (e.g.
        # <S_z> = 0 on the x-polarised CSS), the true commutator
        # is zero, and a relative error is misleading. Using the
        # absolute collective-spin scale (which is 1.0 as a natural
        # unit here, since <S_x> ~ N/2 for N=16 is O(8), so |lhs|
        # or |rhs| ~ 8 in the non-trivial case) makes the criterion
        # "abs_diff < 1e-10 in collective-spin units" instead of
        # "relative to a vanishing RHS".
        scale = max(abs(rhs), abs(lhs), 1.0)
        rel = diff / scale

        rows.append({
            'N': N, 'alpha': alpha, 'chi_max': chi, 'dt': dt,
            'relation': f'[{a},{b}]',
            'lhs': f"{complex(lhs).real:.6e}+{complex(lhs).imag:.6e}j",
            'rhs': f"{complex(rhs).real:.6e}+{complex(rhs).imag:.6e}j",
            'abs_diff': f"{diff:.6e}",
            'rel_diff': f"{rel:.6e}",
        })
        print(f"  [{a}, {b}] = i {c}: "
              f"|lhs - rhs| = {diff:.3e}, "
              f"rel = {rel:.3e}  "
              f"{'PASS' if rel < 1e-10 else 'FAIL'}")

    if output_dir is not None:
        path = os.path.join(output_dir, f'{tag}1d_closed_algebra.csv')
        fields = ['N', 'alpha', 'chi_max', 'dt', 'relation',
                  'lhs', 'rhs', 'abs_diff', 'rel_diff']
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {path}")

    return rows


# ----------------------------------------------------------------------
# Optional QuTiP cross-check of the vectorized exact reference
# ----------------------------------------------------------------------
def cross_check_qutip(N=QUTIP_N, alpha=QUTIP_ALPHA,
                      points=None, output_dir=None, tag=''):
    """
    Validate the vectorized exact reference against the existing
    QuTiP worker at N=8 (small enough for the QuTiP worker to fit in
    memory). Compares SNRs only, since the QuTiP worker does not
    save M or Q.
    """
    import subprocess

    if points is None:
        points = [(np.pi / 2, -np.pi / 2), (np.pi / 2, 0.0), (0.0, 0.0)]

    print(f"\n=== QuTiP cross-check (N={N}, alpha={alpha}) ===", flush=True)

    qutip_worker = os.path.join(ROOT_DIR, 'scripts',
                                 'exact_echo_worker_1d.py')

    rows = []
    for (mu, nu) in points:
        cmd = [
            sys.executable, qutip_worker,
            '--N', str(N), '--alpha', str(alpha), '--J', str(J),
            '--mu', f'{mu:.6f}', '--nu', f'{nu:.6f}',
        ]
        print(f"  [QuTiP] {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True, timeout=7200)
        except subprocess.TimeoutExpired:
            print(f"    [WARN] QuTiP worker timed out", flush=True)
            continue

        out_path = os.path.join(
            ROOT_DIR, 'data', 'exact_1d_closed',
            f'exact_echo_1d_N{N}_alpha{alpha}_J{J}'
            f'_mu{mu:.4f}_nu{nu:.4f}.npy',
        )
        if not os.path.exists(out_path):
            print(f"    [WARN] QuTiP worker produced no output",
                  flush=True)
            continue
        d = np.load(out_path, allow_pickle=True).item()
        snr_qutip = float(d['snr'])

        try:
            ex = exact_snr_and_matrices(N, J, alpha, mu, nu)
            snr_vec = ex['snr']
        except Exception as exc:
            print(f"    [WARN] vectorized exact failed: {exc}",
                  flush=True)
            continue

        rel = abs(snr_qutip - snr_vec) / max(abs(snr_qutip), 1e-30)
        rows.append({
            'N': N, 'alpha': alpha, 'mu': f'{mu:.4f}', 'nu': f'{nu:.4f}',
            'snr_qutip': f'{snr_qutip:.10e}',
            'snr_vectorized': f'{snr_vec:.10e}',
            'rel_diff': f'{rel:.6e}',
        })
        print(f"    QuTiP={snr_qutip:.10f}, "
              f"vectorized={snr_vec:.10f}, rel={rel:.3e}",
              flush=True)

    if output_dir is not None and rows:
        path = os.path.join(output_dir, f'{tag}1d_closed_qutip_crosscheck.csv')
        fields = ['N', 'alpha', 'mu', 'nu', 'snr_qutip', 'snr_vectorized',
                  'rel_diff']
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {path}")

    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="the codebase the documentation: TN vs exact validation for 1D "
                    "closed systems."
    )
    parser.add_argument('--N', nargs='+', type=int, default=DEFAULT_NS,
                        help=f"N values (default: {DEFAULT_NS})")
    parser.add_argument('--alpha', nargs='+', type=float,
                        default=DEFAULT_ALPHAS,
                        help=f"alpha values (default: {DEFAULT_ALPHAS})")
    parser.add_argument('--chi', nargs='+', type=int, default=DEFAULT_CHIS,
                        help=f"chi values (default: {DEFAULT_CHIS})")
    parser.add_argument('--grid-size', type=int, default=DEFAULT_GRID,
                        help=f"grid points per axis (default: {DEFAULT_GRID})")
    parser.add_argument('--dt', type=float, default=DEFAULT_DT,
                        help=f"time step (default: {DEFAULT_DT})")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ROOT_DIR, 'data',
                                              'validation_reports'),
                        help="directory for output CSVs")
    parser.add_argument('--output-tag', type=str, default='',
                        help="prefix tag for output filenames "
                             "(e.g. 'N8_' for a per-N run)")
    parser.add_argument('--skip-independence', action='store_true',
                        help="skip dt and svd_min independence checks")
    parser.add_argument('--skip-algebra', action='store_true',
                        help="skip the [Sx, Sy] = i Sz check")
    parser.add_argument('--cross-check-qutip', action='store_true',
                        help="also run the QuTiP worker at N=8 to "
                             "validate the vectorized exact reference")
    parser.add_argument('--verbose', action='store_true',
                        help="print progress per grid row")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.output_tag or ''

    print("=" * 72)
    print(f"validate_1d_closed.py  (tag={tag or '(none)'})")
    print("=" * 72)
    print(f"  N       = {args.N}")
    print(f"  alpha   = {args.alpha}")
    print(f"  chi     = {args.chi}")
    print(f"  grid    = {args.grid_size} x {args.grid_size}")
    print(f"  dt      = {args.dt}")
    print(f"  output  = {args.output_dir}")

    mu_vals = np.linspace(0.0, np.pi, args.grid_size)
    nu_vals = np.linspace(-np.pi, np.pi, args.grid_size)

    # ---- Main sweep ----
    csv_path = os.path.join(args.output_dir, f'{tag}1d_closed.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    print(f"\nFresh CSV: {csv_path}")

    t_start = time.time()
    all_rows = []
    write_header = True
    for alpha in args.alpha:
        for N in args.N:
            for chi in args.chi:
                rows, n_failed = compute_and_write_one_combo(
                    csv_path, alpha, N, chi, args.dt,
                    mu_vals, nu_vals, write_header, args.verbose,
                )
                all_rows.extend(rows)
                write_header = False

    elapsed = time.time() - t_start
    print(f"\nTotal main sweep: {len(all_rows)} rows, "
          f"{elapsed/60.0:.1f} min")

    # ---- chi_conv summary ----
    chi_conv_path = os.path.join(args.output_dir,
                                  f'{tag}1d_closed_chi_conv.csv')
    determine_chi_conv(all_rows, args.chi, args.alpha, args.N,
                        chi_conv_path)

    # ---- dt independence ----
    if not args.skip_independence:
        dt_independence_check([INDEP_ALPHA], INDEP_N, INDEP_CHI,
                               INDEP_GRID, args.output_dir, tag)
        svdmin_independence_check([INDEP_ALPHA], INDEP_N, INDEP_CHI,
                                   INDEP_GRID, args.output_dir, tag)

    # ---- algebra check ----
    if not args.skip_algebra:
        algebra_check(N=INDEP_N, alpha=INDEP_ALPHA, chi=INDEP_CHI,
                       dt=args.dt, output_dir=args.output_dir, tag=tag)

    # ---- optional QuTiP cross-check ----
    if args.cross_check_qutip:
        cross_check_qutip(N=QUTIP_N, alpha=QUTIP_ALPHA,
                           output_dir=args.output_dir, tag=tag)

    print("\n" + "=" * 72)
    print("validate_1d_closed.py: DONE")
    print("=" * 72)
    print(f"  {csv_path}")
    print(f"  {chi_conv_path}")
    if not args.skip_independence:
        print(f"  {args.output_dir}/{tag}1d_closed_dt_independence.csv")
        print(f"  {args.output_dir}/{tag}1d_closed_svdmin_independence.csv")
    if not args.skip_algebra:
        print(f"  {args.output_dir}/{tag}1d_closed_algebra.csv")


if __name__ == '__main__':
    main()