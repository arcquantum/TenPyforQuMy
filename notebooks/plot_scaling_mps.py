"""
notebooks/plot_scaling_mps.py

Evaluates MPS scaling for OAT echo protocols and compares with
the exact analytical results from Schulte et al. (2020).
"""

import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import os
import sys
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import OATModel
from src.evolution import initialize_css, apply_collective_rotation
from src.observables import _get_3x3_covariance
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

# Module-level verbose flag, set from --verbose in main().
# When False, per-call diagnostics are suppressed so the Slurm log
# is not flooded during long scaling runs.
VERBOSE = False

def evolve_mps_time(psi, model, total_time, chi_max, dt_max=0.005):
    """Evolves the MPS forward or backward in time using ExpMPOEvolution."""
    if abs(total_time) < 1e-12:
        return

    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt_step = total_time / N_steps   # preserves the sign

    mpo_params = {
        'dt': dt_step,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12}
    }

    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = N_steps
    engine.run()


def get_Q_mps(psi):
    """Extracts the exact 3x3 quantum covariance matrix."""
    _, Cov = _get_3x3_covariance(psi)
    return Cov


def get_M_mps(psi_mu, model, chi_max, nu_minus_mu, delta=1e-3, dt_max=0.005):
    """Extracts the 3x3 signal response matrix via central finite differences."""
    M = np.zeros((3, 3))
    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']

    for i, axis in enumerate(axes):
        # Forward phase imprint
        psi_plus = psi_mu.copy()
        apply_collective_rotation(psi_plus, axis, delta)
        evolve_mps_time(psi_plus, model, nu_minus_mu / 2.0, chi_max, dt_max)
        means_plus = [np.real(np.sum(psi_plus.expectation_value(a))) for a in op_names]

        # Backward phase imprint
        psi_minus = psi_mu.copy()
        apply_collective_rotation(psi_minus, axis, -delta)
        evolve_mps_time(psi_minus, model, nu_minus_mu / 2.0, chi_max, dt_max)
        means_minus = [np.real(np.sum(psi_minus.expectation_value(a))) for a in op_names]

        for j in range(3):
            M[i, j] = (means_plus[j] - means_minus[j]) / (2.0 * delta)

    return M


def compute_sensitivity_mps(model, chi_max, mu, nu, dt_max=0.002):
    """Evaluates the pure MPS SNR at a given (mu, nu)."""
    psi = initialize_css(model.lat)
    evolve_mps_time(psi, model, mu / 2.0, chi_max, dt_max)

    M = get_M_mps(psi, model, chi_max, nu - mu, dt_max=dt_max)
    evolve_mps_time(psi, model, (nu - mu) / 2.0, chi_max, dt_max)
    Q = get_Q_mps(psi)

    # Correct null‑direction handling
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt_evals = np.zeros_like(evals)
    inv_sqrt_evals[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt_evals) @ evecs.T

    if VERBOSE:
        print(f"  N={model.lat.N_sites} chi={chi_max}: "
              f"Q evals = {np.linalg.eigvalsh(Q)}, "
              f"M norms = {np.linalg.norm(M, axis=1)}")
    return np.max(scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False))


def schulte_analytical_snr(N, mu, nu, sigma):
    """
    Exact analytical mapping of collective dephasing (Schulte 2020, App. A & B).
    Returns SNR normalized by N (Sensitivity Δφ^-2 / N).
    """
    S = N / 2.0
    N_1, N_2 = N - 1, N - 2

    q0 = S * (np.cos(nu/2)**N_1) * np.exp(-sigma * (abs(nu-mu) + abs(mu)) / 4.0)
    q1 = S**2 + S/2.0
    q2 = np.exp(-sigma * (abs(nu-mu) + abs(mu))) * (S/2.0) * (N_1) * (np.cos(nu)**N_2)
    q3 = np.exp(-sigma * (abs(nu-mu) + abs(mu)) / 4.0) * (S/2.0) * (N_1) * (np.cos(nu/2)**N_2) * np.sin(nu/2)
    q4 = S / 2.0

    n1 = np.exp(-sigma * abs(nu-mu) / 4.0) * S * (N_1) * np.sin((mu-nu)/2.0) * (np.cos((mu-nu)/2.0)**N_2)
    n2 = -np.exp(-sigma * (abs(nu-mu)/4.0 + abs(mu))) * S * (N_1) * np.sin((mu-nu)/2.0) * (np.cos((mu+nu)/2.0)**N_2)
    n3 = -np.exp(-sigma * abs(mu) / 4.0) * S * (np.cos(mu/2)**N_1)
    n4 = np.exp(-sigma * (abs(nu-mu) + abs(mu)) / 4.0) * S * (np.cos(nu/2)**N_1)

    Q_xx = 0.5 * (q1 + q2) - q0**2
    Q_yz = np.array([[0.5 * (q1 - q2), q3], [q3, q4]])

    M_xx = 0.5 * (n1 + n2)
    M_yz = np.array([[0.5 * (n1 - n2), n3], [n4, 0.0]])

    snr_xx = 0.0 if Q_xx < 1e-12 else (M_xx**2) / Q_xx

    evals, evecs = np.linalg.eigh(Q_yz)
    mask = evals > 1e-12
    inv_evals = np.zeros_like(evals)
    inv_evals[mask] = 1.0 / evals[mask]
    Q_yz_inv = evecs @ np.diag(inv_evals) @ evecs.T

    snr_yz = np.max(np.linalg.eigvalsh(M_yz @ Q_yz_inv @ M_yz.T))
    return max(snr_xx, snr_yz) / N


def get_optimal_analytical(N, sigma, protocol):
    """
    Returns optimal (mu, nu, snr) for the requested protocol.
    For GHZ, nu is fixed to 0 (exact echo), not mu.
    """
    if protocol == 'OUT':
        return np.pi/2, -np.pi/2, schulte_analytical_snr(N, np.pi/2, -np.pi/2, sigma)

    if protocol == 'Squeezing':
        res = scipy.optimize.minimize(
            lambda x: -schulte_analytical_snr(N, x[0], x[0], sigma),
            x0=[1.0/np.sqrt(N)],
            bounds=[(0.01, 4.0/np.sqrt(N))]
        )
        return res.x[0], res.x[0], -res.fun

    if protocol == 'GHZ':
        # GHZ-like protocols use exact echo: nu = 0.
        # Optimize only over mu near pi.
        mu_min = np.pi - 4.0/np.sqrt(N)
        mu_max = np.pi

        # Coarse grid search to avoid missing the sharp optimum
        grid_size = 300
        mu_grid = np.linspace(mu_min, mu_max, grid_size)
        snr_grid = [schulte_analytical_snr(N, mu, 0.0, sigma) for mu in mu_grid]
        best_mu = mu_grid[int(np.argmax(snr_grid))]

        # Local refinement
        res = scipy.optimize.minimize(
            lambda x: -schulte_analytical_snr(N, x[0], 0.0, sigma),
            x0=[best_mu],
            bounds=[(mu_min, mu_max)]
        )
        return res.x[0], 0.0, -res.fun

    raise ValueError(f"Unknown protocol: {protocol}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MPS scaling logic.")
    parser.add_argument("--chi_max", type=int, default=64, help="Maximum bond dimension")
    parser.add_argument("--max_N", type=int, default=200, help="Maximum N to calculate in the list")
    parser.add_argument("--dt_max", type=float, default=0.005, help="Maximum time step for ExpMPOEvolution")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-N Q eigenvalues and M-matrix norms to stdout")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    N_list = [n for n in [2, 4, 8, 16, 32, 64, 128, 200] if n <= args.max_N]

    snr_mps_out, snr_mps_sq, snr_mps_ghz = [], [], []
    sigmas = [0.0, 0.01, 0.5]
    analytical_data = {sig: {'OUT': [], 'SQ': [], 'GHZ': []} for sig in sigmas}

    print(f"Running MPS & Analytical Benchmark up to N={args.max_N}...", flush=True)
    for N in N_list:
        model = OATModel({'N': N, 'chi_interaction': 1.0, 'conserve': None})

        mu_out, nu_out, _ = get_optimal_analytical(N, 0.0, 'OUT')
        mu_sq, nu_sq, _   = get_optimal_analytical(N, 0.0, 'Squeezing')
        mu_ghz, nu_ghz, _ = get_optimal_analytical(N, 0.0, 'GHZ')

        snr_mps_out.append((compute_sensitivity_mps(model, args.chi_max, mu_out, nu_out, args.dt_max)**2)/N)
        snr_mps_sq.append((compute_sensitivity_mps(model, args.chi_max, mu_sq, nu_sq, args.dt_max)**2)/N)
        snr_mps_ghz.append((compute_sensitivity_mps(model, args.chi_max, mu_ghz, nu_ghz, args.dt_max)**2)/N)

        for sig in sigmas:
            _, _, snr_o = get_optimal_analytical(N, sig, 'OUT')
            _, _, snr_s = get_optimal_analytical(N, sig, 'Squeezing')
            _, _, snr_g = get_optimal_analytical(N, sig, 'GHZ')
            analytical_data[sig]['OUT'].append(snr_o)
            analytical_data[sig]['SQ'].append(snr_s)
            analytical_data[sig]['GHZ'].append(snr_g)

    np.save(os.path.join(DATA_DIR, f"scaling_c{args.chi_max}_maxN{args.max_N}.npy"), {
        'N_list': N_list, 'chi_max': args.chi_max,
        'snr_out_pure': snr_mps_out, 'snr_sq_pure': snr_mps_sq, 'snr_ghz_pure': snr_mps_ghz,
        'analytical': analytical_data
    })

    for sig in sigmas:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(N_list, analytical_data[sig]['OUT'], marker='o', color='red', label='OUT')
        ax.plot(N_list, analytical_data[sig]['SQ'], marker='s', color='blue', linestyle='--', label='Squeezing')
        ax.plot(N_list, analytical_data[sig]['GHZ'], marker='^', color='black', linestyle=':', label='GHZ')

        if sig == 0.0:
            ax.scatter(N_list, snr_mps_out, edgecolor='red', facecolor='none', s=100, linewidths=1.5, zorder=5)
            ax.scatter(N_list, snr_mps_sq, edgecolor='blue', facecolor='none', s=100, linewidths=1.5, zorder=5)
            ax.scatter(N_list, snr_mps_ghz, edgecolor='black', facecolor='none', s=100, linewidths=1.5, zorder=5)
            ax.set_title(rf'MPS vs Analytical Scaling ($\sigma={sig}$, $\chi={args.chi_max}$)')
        else:
            ax.set_title(rf'Analytical Dephasing Scaling ($\sigma={sig}$)')

        N_arr = np.asarray(N_list)
        ax.fill_between(N_arr, N_arr, 100000, color='green', alpha=0.3, label='Heisenberg Limit')
        ax.fill_between(N_arr, 1e-2, 1, color='gray', alpha=0.5, label='sub-SQL')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Particle Number $N$')
        ax.set_ylabel(r'Sensitivity $\Delta\phi^{-2}/N$')
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        if len(N_list) > 1:
            ax.set_xlim(min(N_list), max(N_list))
        ax.set_ylim(0.1, 100000)

        fig.savefig(os.path.join(PLOTS_DIR, f"fig3_scaling_sigma_{sig}_c{args.chi_max}.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)

    print("Scaling plots saved.")


if __name__ == "__main__":
    main()