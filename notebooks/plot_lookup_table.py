#!/usr/bin/env python3
"""
notebooks/plot_lookup_table.py


Renders the experimental lookup table (produced by the documentation) as a PDF.
Reads data/bundles/master_lookup.csv. Writes plots/lookup_table.pdf
and plots/fig_ext_9.pdf (identical content, two filenames).
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    BUNDLES_DIR, setup_style, no_data, read_csv_rows, PLOTS_DIR,
)


LOOKUP_CSV = os.path.join(BUNDLES_DIR, 'master_lookup.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    rows = read_csv_rows(LOOKUP_CSV)
    if not rows:
        no_data(f'no master_lookup.csv at {LOOKUP_CSV} '
                f'(the codebase the documentation not yet complete)')

    # Columns: N, alpha, gamma1, gamma2, geometry, mu_opt, nu_opt,
    #          theta_k_opt, phi_k_opt, k_gain, snr_opt, sensitivity,
    #          chi_prod, n_traj_prod, walltime, method
    preferred = [
        'N', 'Lx', 'Ly', 'alpha', 'gamma1', 'gamma2',
        'mu_opt', 'nu_opt', 'theta_k_opt', 'phi_k_opt',
        'k_gain', 'snr_opt', 'chi_prod',
    ]
    headers = [h for h in preferred if h in rows[0]]
    if not headers:
        headers = list(rows[0].keys())

    n_rows = len(rows)
    n_cols = len(headers)

    fig, ax = plt.subplots(
        figsize=(0.9 * n_cols, 0.35 * (n_rows + 1) + 1.2)
    )
    ax.axis('off')
    cell_text = [[r.get(h, '') for h in headers] for r in rows]
    table = ax.table(
        cellText=cell_text, colLabels=headers,
        loc='center', cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.05)
    ax.set_title('Experimental lookup table', fontsize=14, pad=12)

    out1 = args.outfile or os.path.join(PLOTS_DIR, 'lookup_table.pdf')
    out2 = os.path.join(PLOTS_DIR, 'fig_ext_9.pdf')
    fig.savefig(out1, bbox_inches='tight')
    fig.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] wrote {out1}')
    print(f'[plot] wrote {out2}')


if __name__ == '__main__':
    main()