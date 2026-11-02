"""
src/observables.py

Contains functions to extract physical observables and 
metrological parameters from the tensor network state.
"""

import numpy as np

def _get_3x3_covariance(mps):
    """
    Helper function to construct the mean spin vector and the 
    symmetric 3x3 quantum covariance matrix.
    """
    S_vec = np.array([
        np.sum(mps.expectation_value('Sx')),
        np.sum(mps.expectation_value('Sy')),
        np.sum(mps.expectation_value('Sz'))
    ])
    
    Cov = np.zeros((3, 3), dtype=complex)
    ops = ['Sx', 'Sy', 'Sz']
    
    for i in range(3):
        for j in range(3):
            # Sum the non-local correlation matrix for collective operators
            corr_ij = np.sum(mps.correlation_function(ops[i], ops[j]))
            Cov[i, j] = corr_ij - (S_vec[i] * S_vec[j])
            
    # Mathematically enforce Hermitian symmetry for the quantum covariance:
    # 1/2 <{S_i, S_j}> - <S_i><S_j>
    Cov = np.real(0.5 * (Cov + Cov.conj().T))
    
    return S_vec, Cov


def compute_squeezing(mps, xi_max=10.0):
    """
    Wineland squeezing parameter xi_R^2.

    Two regime-of-validity guards are applied before returning:
      1. Mean spin must be a non-negligible fraction of the CSS scale
         (|S|^2 > 0.05 (N/2)^2); otherwise the ratio is dominated by
         the collapse of <S> rather than by genuine spin squeezing.
      2. xi_R^2 must be below xi_max (default 10). A value above this
         threshold indicates the state is deeply over-squeezed and the
         Wineland ratio no longer represents a metrological figure of
         merit.
    In either case, NaN is returned, and the downstream plotting
    pipeline treats the point as absent.
    """
    N = mps.L
    S_vec, Cov = _get_3x3_covariance(mps)

    mean_spin_length_sq = np.sum(S_vec**2)
    if mean_spin_length_sq < 1e-12:
        return float('nan')

    # Guard 1: mean spin well above the CSS scale
    if mean_spin_length_sq < 0.05 * (N/2)**2:
        return float('nan')

    n_vec = S_vec / np.sqrt(mean_spin_length_sq)
    P_perp = np.eye(3) - np.outer(n_vec, n_vec)
    Cov_perp = P_perp @ Cov @ P_perp
    evals = np.sort(np.linalg.eigvalsh(Cov_perp))
    V_minus = evals[1]
    xi_R_sq = N * V_minus / mean_spin_length_sq

    # Guard 2: deep over-squeezed regime
    if xi_R_sq > xi_max:
        return float('nan')

    return float(xi_R_sq)


def compute_qfi(mps):
    """
    Computes the Quantum Fisher Information (QFI) maximized over all 
    collective spin directions for a pure state.
    """
    _, Cov = _get_3x3_covariance(mps)
    
    # For a pure state, QFI = 4 * Max(Variance)
    eigenvalues = np.linalg.eigvalsh(Cov)
    max_var = np.max(eigenvalues)
    
    return float(4.0 * max_var)


def compute_entanglement_entropy(mps):
    """
    Returns the half-chain von Neumann bipartite entanglement entropy.

    For even L, the half-chain partition is L/2 | L/2, whose bond index
    is L/2 - 1. For odd L, the closest-to-centre bond is (L-1)/2. Both
    cases are covered by (L - 1) // 2.

    Historical note: the previous implementation used L // 2, which is
    correct for odd L but off by one for even L, picking the (L/2 + 1) |
    (L/2 - 1) partition instead of the (L/2) | (L/2) partition. This was
    detected by the exact-diagonalization cross-check in
    scripts/checks/check_observables.py at N = 6.
    """
    entropies = mps.entanglement_entropy()
    mid_point = (mps.L - 1) // 2
    return float(entropies[mid_point])


def compute_qfi_mixed_bounds(traj_cov_list, traj_S_vec_list=None):
    """
    Trajectory-ensemble QFI bounds.

    Parameters
    ----------
    traj_cov_list : list of (3,3) ndarray
        Per-trajectory 3x3 covariance matrices, one per trajectory.
    traj_S_vec_list : list of (3,) ndarray, optional
        Per-trajectory mean-spin vectors. If provided, qfi_var
        correctly includes the between-trajectory variance. If
        not provided, qfi_var only uses the mean of the within-
        trajectory covariances and undercounts for ensembles where
        the mean spin varies between trajectories.

    Returns
    -------
    qfi_var : float
        4 * lambda_max(covariance of the trajectory-averaged density
        matrix). This is a diagnostic: it equals the true pure-state
        QFI for a single-trajectory input, but for a mixture it is
        NOT a bound on the true mixed-state QFI. In particular, for
        the maximally mixed single-qubit state it is 1 while the
        true QFI is 0.
    qfi_avg : float
        Mean over trajectories of 4 * lambda_max(Cov_j). By
        convexity of the QFI, this is a strict upper bound on the
        true mixed-state QFI.

    Raises
    ------
    ValueError
        If traj_cov_list is empty.
    """
    import numpy as np

    if not traj_cov_list:
        raise ValueError(
            "compute_qfi_mixed_bounds: traj_cov_list must be non-empty"
        )

    covs = np.asarray(traj_cov_list, dtype=complex)
    cov_mean = covs.mean(axis=0)
    cov_mean = np.real(0.5 * (cov_mean + cov_mean.conj().T))

    if traj_S_vec_list is not None:
        S_vecs = np.asarray(traj_S_vec_list, dtype=complex)
        n = len(S_vecs)
        S_mean = S_vecs.mean(axis=0)
        S_cov = np.real(
            np.einsum('ij,ik->jk', S_vecs, S_vecs) / n
            - np.outer(S_mean, S_mean)
        )
        cov_mean = cov_mean + S_cov

    qfi_var = 4.0 * float(np.max(np.linalg.eigvalsh(cov_mean)))
    qfi_avg = float(np.mean([
        4.0 * float(np.max(np.linalg.eigvalsh(
            0.5 * (c + c.conj().T)
        )))
        for c in covs
    ]))
    return qfi_var, qfi_avg