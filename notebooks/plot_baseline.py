"""
notebooks/plot_baseline.py
Generates the baseline OAT sensitivity plots. 
Contrasts the Gaussian Wineland limit against the ultimate QFI Cramér-Rao bound.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.lines import Line2D
import matplotlib


def _get_cmap(name, copy=False):
    """Stub-safe colormap accessor. Inline because this script predates
    _plot_helpers and importing it would re-run matplotlib.use('Agg')
    after pyplot is already imported."""
    cmap = matplotlib.colormaps[name]  # type: ignore[attr-defined]
    return cmap.copy() if copy else cmap

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "lines.linewidth": 2.0
})

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
    os.makedirs(PLOTS_DIR, exist_ok=True)

    search_pattern = os.path.join(DATA_DIR, "out_N*_chi*_tmax*.npy")
    data_files = glob.glob(search_pattern)
    
    if not data_files:
        print(f"No .npy files found in {DATA_DIR}.")
        return

    # Load and strictly sort data by N, then chi_max to prevent alphabetical scrambling
    parsed_data = []
    for filepath in data_files:
        data = np.load(filepath, allow_pickle=True).item()
        N = data.get('metadata', {}).get('N', 0)
        chi = data.get('metadata', {}).get('chi_max', 0)
        parsed_data.append((N, chi, data))
        
    parsed_data.sort(key=lambda x: (x[0], x[1]))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = _get_cmap('viridis')(np.linspace(0.1, 0.9, len(parsed_data)))
    
    custom_legend_lines = []
    
    for idx, (N, chi, data) in enumerate(parsed_data):
        times = data['time']
        squeezing_vals = data['squeezing']
        qfi_vals = data['qfi']

        # mu = 2 * chi * t. (chi_interaction = 1.0 in models.py)
        mu_vals = 2.0 * times 
        
        # Regime-of-validity mask, matching the guard in src/observables.py.
        # Excludes (i) numerically-near-zero xi_R^2 that would produce
        # spurious 1/xi_R^2 spikes, and (ii) deeply over-squeezed points
        # with xi_R^2 > 10 where the Wineland ratio is off-regime. NaN
        # values written by the guarded compute_squeezing are also
        # excluded automatically because NaN comparisons return False.
        valid_idx = (squeezing_vals > 1e-10) & (squeezing_vals < 10.0)
        mu_valid = mu_vals[valid_idx]
        
        # Extract Sensitivity limits
        wineland_snr = 1.0 / squeezing_vals[valid_idx]
        qfi_snr = qfi_vals[valid_idx] / N # QFI enhancement relative to SQL

        # MATHEMATICAL SANITY CHECK: Cramér-Rao Bound Validation
        # The Wineland sensitivity can never theoretically exceed the QFI.
        if not np.all(qfi_snr >= wineland_snr - 1e-10):
            raise ValueError(f"CRITICAL PHYSICS VIOLATION: Wineland exceeds QFI for N={N}.")

        color = colors[idx]
        
        # Plot Wineland (Gaussian Limit)
        ax.plot(mu_valid, wineland_snr, color=color, linestyle='-', alpha=0.9)
        # Plot QFI (Ultimate Cramér-Rao Bound)
        ax.plot(mu_valid, qfi_snr, color=color, linestyle='--', alpha=0.9)
        
        custom_legend_lines.append(Line2D([0], [0], color=color, lw=2, label=rf'$N={N}, \chi={chi}$'))

    ax.set_xlabel(r'Squeezing Strength $\mu$')
    ax.set_ylabel(r'Sensitivity $\Delta\phi^{-2}/N$')
    ax.set_title(r'OAT Metrological Limits: Gaussian vs. QFI Bound')

    # Fix the Legend Aesthetics (Color = System Size, Style = Metric)
    custom_legend_lines.append(Line2D([0], [0], color='purple', lw=2, linestyle='-', label=r'Wineland $\xi_R^{-2}$'))
    custom_legend_lines.append(Line2D([0], [0], color='purple', lw=2, linestyle='--', label=r'QFI Bound $F_Q/N$'))
    custom_legend_lines.append(Line2D([0], [0], color='black', lw=1, linestyle='-', alpha=0.5, label='SQL'))
    
    ax.axhline(1, color='black', linewidth=1, linestyle='-', alpha=0.5)

    ax.set_yscale('log')
    # Focus on the relevant interaction time
    ax.set_xlim(0, max(mu_valid) if 'mu_valid' in locals() else 2.0)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(handles=custom_legend_lines, loc='upper right', bbox_to_anchor=(1.45, 1.0))

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "oat_baseline_scaling.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Plot mathematically verified and saved: {out_path}")

if __name__ == "__main__":
    main()