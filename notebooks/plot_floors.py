"""
notebooks/plot_floors.py

Updated plotting script for error-floor analysis.
Generates:
  1. numerical_convergence_scaling.pdf    (all N vs chi truncation error)
  2. algorithmic_breakdown_scaling.pdf    (grid of panels, one per N, multiple chi)
  3. contour_error_floor_combined.pdf     (two-panel contour for selected N values)
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def _get_cmap(name, copy=False):
    """Stub-safe colormap accessor. See plot_baseline.py for rationale."""
    cmap = matplotlib.colormaps[name]  # type: ignore[attr-defined]
    return cmap.copy() if copy else cmap

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 1.8
})

def main():
    parser = argparse.ArgumentParser(description="Generate merged error-floor plots.")
    parser.add_argument("--contour_Ns", type=str, default="64,200",
                        help="Comma-separated N values for contour panels, e.g. '64,200'")
    args = parser.parse_args()

    # Parse requested contour N values
    try:
        contour_Ns = [int(x.strip()) for x in args.contour_Ns.split(',')]
    except Exception:
        contour_Ns = [64, 200]

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
    os.makedirs(PLOTS_DIR, exist_ok=True)

    search_pattern = os.path.join(DATA_DIR, "out_N*_chi*_tmax*.npy")
    data_files = glob.glob(search_pattern)

    if not data_files:
        print(f"No .npy files found in {DATA_DIR}.")
        return

    # -------- Load and organise data --------
    convergence_data = {}   # {N: {chi: total_trunc_error}}
    dynamics_data = {}      # {N: {chi: (times, squeezing)}}

    for filepath in data_files:
        data = np.load(filepath, allow_pickle=True).item()
        N = data['metadata']['N']
        chi = data['metadata']['chi_max']
        if N not in convergence_data:
            convergence_data[N] = {}
            dynamics_data[N] = {}
        convergence_data[N][chi] = np.sum(data['trunc_error'])
        dynamics_data[N][chi] = (data['time'], data['squeezing'])

    sorted_Ns = sorted(convergence_data.keys())
    if not sorted_Ns:
        print("No valid N found.")
        return

    # ============================================================
    # FIGURE 1: Cumulative truncation error convergence
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    colors1 = _get_cmap('viridis')(np.linspace(0.1, 0.9, len(sorted_Ns)))
    all_chis = set()

    for idx, N in enumerate(sorted_Ns):
        chi_list = sorted(convergence_data[N].keys())
        error_list = [convergence_data[N][chi] for chi in chi_list]
        all_chis.update(chi_list)
        ax1.plot(chi_list, error_list, marker='o', markersize=6,
                 color=colors1[idx], linestyle='--', label=rf'$N={N}$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Maximum Bond Dimension ($\chi_{\mathrm{max}}$)')
    ax1.set_ylabel(r'Cumulative Truncation Error ($\Sigma \epsilon$)')
    ax1.set_title(r'Numerical Convergence Floor Scaling')
    sorted_all_chis = sorted(list(all_chis))
    ax1.set_xticks(sorted_all_chis)
    ax1.set_xticklabels([str(chi) for chi in sorted_all_chis])
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    fig1.savefig(os.path.join(PLOTS_DIR, "numerical_convergence_scaling.pdf"),
                 format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # ============================================================
    # FIGURE 2: Algorithmic breakdown grid (one panel per N)
    # ============================================================
    # Arrange panels in a grid: 2 columns, as many rows as needed
    n_cols = 2
    n_rows = int(np.ceil(len(sorted_Ns) / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes2 = np.array(axes2).reshape(-1)  # flatten for easy indexing

    for idx, N in enumerate(sorted_Ns):
        ax = axes2[idx]
        available_chis = sorted(dynamics_data[N].keys())
        colors2 = _get_cmap('plasma')(np.linspace(0.1, 0.9, len(available_chis)))

        max_time = 0.0
        for chi, color in zip(available_chis, colors2):
            t_data, sq_data = dynamics_data[N][chi]
            valid = (sq_data > 1e-10) & (sq_data < 1e5)
            if not np.any(valid):
                continue
            sq_dB = 10 * np.log10(sq_data[valid])
            t_valid = t_data[valid]
            max_time = max(max_time, np.max(t_valid))
            ax.plot(t_valid, sq_dB, color=color, label=rf'$\chi={chi}$')

        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5, label='SQL')
        ax.set_xlabel(r'Evolution Time ($\chi t$)')
        ax.set_ylabel(r'Wineland $\xi_R^2$ (dB)')
        ax.set_title(rf'$N={N}$')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, ls='--', alpha=0.5)
        if max_time > 0:
            ax.set_xlim(0, max_time * 1.05)

    # Hide any unused subplots
    for idx in range(len(sorted_Ns), len(axes2)):
        axes2[idx].axis('off')

    fig2.suptitle('Algorithmic Breakdown Floor', y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOTS_DIR, "algorithmic_breakdown_scaling.pdf"),
                 format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # ============================================================
    # FIGURE 3: Combined contour for selected N values
    # ============================================================
    # Ensure requested N values exist; if not, fallback to two largest
    available_for_contour = [N for N in contour_Ns if N in dynamics_data]
    if len(available_for_contour) < 2:
        print(f"Warning: Not enough requested N ({contour_Ns}) available in data. Using two largest N.")
        available_for_contour = sorted_Ns[-2:] if len(sorted_Ns) >= 2 else sorted_Ns

    if len(available_for_contour) < 1:
        print("Skipping contour plot: insufficient N data.")
    else:
        fig3, axes3 = plt.subplots(1, len(available_for_contour), figsize=(7*len(available_for_contour), 6))
        if len(available_for_contour) == 1:
            axes3 = [axes3]

        for ax, N in zip(axes3, available_for_contour):
            chis = sorted(dynamics_data[N].keys())
            if len(chis) < 3:
                print(f"Skipping contour for N={N}: need at least 3 chi values, found {len(chis)}")
                ax.axis('off')
                continue

            X_time, Y_chi, Z_sq = [], [], []
            for chi in chis:
                t_data, sq_data = dynamics_data[N][chi]
                for i in range(0, len(t_data), 3):
                    if sq_data[i] > 1e-10:
                        X_time.append(t_data[i])
                        Y_chi.append(chi)
                        Z_sq.append(10 * np.log10(sq_data[i]))

            X_time = np.array(X_time)
            Y_chi = np.array(Y_chi)
            Z_sq = np.array(Z_sq)

            try:
                c = ax.tricontourf(X_time, Y_chi, Z_sq, levels=30, cmap='viridis_r')
                fig3.colorbar(c, ax=ax, label=r'$\xi_R^2$ (dB)')
            except Exception as e:
                print(f"Contour failed for N={N}: {e}")

            ax.set_xlabel(r'Evolution Time ($\chi t$)')
            ax.set_ylabel(r'$\chi_{\mathrm{max}}$')
            ax.set_title(rf'$N={N}$')
            ax.set_yticks(chis)

        fig3.suptitle('MPS Squeezing Breakdown Contours', y=1.02)
        fig3.tight_layout()
        fig3.savefig(os.path.join(PLOTS_DIR, "contour_error_floor_combined.pdf"),
                     format='pdf', bbox_inches='tight')
        plt.close(fig3)

    print(f"Error Floor Pipeline Complete. PDFs generated in {PLOTS_DIR}/.")

if __name__ == "__main__":
    main()