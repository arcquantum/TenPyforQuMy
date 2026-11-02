#!/usr/bin/env python3
"""
notebooks/plot_validation_errors.py


Heatmap of epsilon(mu, nu) from the validation reports produced by
validate_1d_closed.py, validate_tjm_open.py, etc. If the report
directory is empty (Phases 1-4 not yet complete), exits cleanly.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    VALIDATION_REPORTS_DIR, setup_style, no_data, read_csv_rows,
    float_or_nan, case_label, PLOTS_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    # Convention from the codebase: reports/<case>.csv
    report = os.path.join(VALIDATION_REPORTS_DIR, f'{args.case}.csv')
    rows = read_csv_rows(report)
    if not rows:
        no_data(f'no validation report at {report} '
                f'(the codebase+ not yet complete)')

    # Look for columns mu, nu, epsilon
    mu_list, nu_list, eps_list = [], [], []
    for r in rows:
        mu = float_or_nan(r.get('mu'))
        nu = float_or_nan(r.get('nu'))
        eps = float_or_nan(r.get('epsilon', r.get('eps_max', 'nan')))
        if np.isfinite(mu) and np.isfinite(nu):
            mu_list.append(mu)
            nu_list.append(nu)
            eps_list.append(eps)

    if not mu_list:
        no_data('validation report contains no (mu, nu) rows')

    mu_arr = np.asarray(mu_list)
    nu_arr = np.asarray(nu_list)
    eps_arr = np.asarray(eps_list)
    mu_u = np.unique(mu_arr)
    nu_u = np.unique(nu_arr)

    grid = np.full((len(nu_u), len(mu_u)), np.nan)
    for mu, nu, e in zip(mu_arr, nu_arr, eps_arr):
        i = np.argmin(np.abs(nu_u - nu))
        j = np.argmin(np.abs(mu_u - mu))
        grid[i, j] = e

    fig, ax = plt.subplots(figsize=(7, 5))
    if np.any(np.isfinite(grid)):
        norm = matplotlib.colors.LogNorm(
            vmin=max(np.nanmin(grid), 1e-12),
            vmax=max(np.nanmax(grid), 1e-11),
        )
    else:
        norm = matplotlib.colors.Normalize()
    im = ax.imshow(
        grid, origin='lower', aspect='auto', cmap='magma_r',
        extent=[mu_u[0], mu_u[-1], nu_u[0], nu_u[-1]],
        norm=norm,
    )
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\nu$')
    ax.set_title(f'Validation error $\\epsilon(\\mu,\\nu)$ — {case_label(args.case)}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # type: ignore[arg-type]

    out = args.outfile or os.path.join(
        PLOTS_DIR, f'validation_errors_{args.case}.pdf'
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()