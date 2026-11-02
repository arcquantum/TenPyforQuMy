#!/usr/bin/env python3
"""
notebooks/plot_master_landscape_grid.py


Full landscape grid: rows = alpha, columns = N (or geometry). One
heatmap per (alpha, geometry) tuple, reading from data/bundles/.


Reads all bundles matching <case>_alpha*.npy. For Scharnagl cases,
collapses the last two axes by max over (theta_k, phi_k) unless a
k_slice is given via --k_slice I_THETA I_PHI.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    ALL_CASES, is_asymmetric, is_scharnagl, case_label,
    setup_style, no_data, discover_bundles, parse_bundle_filename,
    load_bundle, bundle_metadata, bundle_axes, bundle_snr_2d,
    geometry_label, PLOTS_DIR,
)


def _panel_key(meta: dict, fallback: dict) -> tuple:
    """(alpha, geometry_tuple) used for panel placement."""
    alpha = meta.get('alpha', fallback.get('alpha'))
    if alpha is None:
        # A bundle with no alpha cannot be placed in the grid. Use 0.0
        # as a sentinel so the panel still renders; the label will read
        # alpha=0.0 and the caller can spot the malformed bundle.
        alpha = 0.0
    if meta.get('Lx') is not None and meta.get('Ly') is not None:
        return (float(alpha), (int(meta['Lx']), int(meta['Ly'])))
    if meta.get('N') is not None:
        return (float(alpha), (int(meta['N']),))
    return (float(alpha), ())


def _collect(case, k_slice):
    paths = discover_bundles(case)
    if not paths:
        return {}
    panels = {}
    for path in paths:
        bundle = load_bundle(path)
        if bundle is None:
            continue
        fallback = parse_bundle_filename(path) or {}
        meta = bundle_metadata(bundle, fallback=fallback)
        try:
            x, y, xlabel, ylabel = bundle_axes(bundle)
            snr = bundle_snr_2d(bundle, k_slice=k_slice)
        except Exception as exc:
            print(f"[skip] {os.path.basename(path)}: {exc}")
            continue
        panels[_panel_key(meta, fallback)] = {
            'x': x, 'y': y, 'snr': snr,
            'xlabel': xlabel, 'ylabel': ylabel,
            'meta': meta, 'path': path,
        }
    return panels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=list(ALL_CASES))
    parser.add_argument('--k_slice', nargs=2, type=int, default=None,
                        metavar=('I_THETA', 'I_PHI'),
                        help='Scharnagl only: fixed (theta_k, phi_k) index.')
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()

    setup_style()
    case = args.case
    panels = _collect(case, tuple(args.k_slice) if args.k_slice else None)
    if not panels:
        no_data(f'no bundles for case={case} in data/bundles/')

    alphas = sorted({k[0] for k in panels})
    geoms  = sorted({k[1] for k in panels})
    n_rows, n_cols = len(alphas), len(geoms)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 3.0 * n_rows),
        squeeze=False, constrained_layout=True,
    )

    for i, a in enumerate(alphas):
        for j, g in enumerate(geoms):
            ax = axes[i, j]
            if (a, g) not in panels:
                ax.set_visible(False)
                continue
            p = panels[(a, g)]
            snr = p['snr']
            vmin = float(np.nanmin(snr))
            vmax = float(np.nanmax(snr))
            im = ax.imshow(
                snr, origin='lower', aspect='auto',
                extent=[p['x'][0], p['x'][-1], p['y'][0], p['y'][-1]],
                cmap='viridis',
                vmin=vmin, vmax=vmax,
            )
            if i == n_rows - 1:
                ax.set_xlabel(p['xlabel'])
            if j == 0:
                ax.set_ylabel(p['ylabel'])
            if i == 0:
                g_lbl = (
                    f'$N$={g[0]}' if len(g) == 1
                    else f'$L_x$={g[0]}, $L_y$={g[1]}'
                )
                ax.set_title(g_lbl, fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Master landscape grid — {case_label(case)}',
                 fontsize=14)

    out = args.outfile or os.path.join(
        PLOTS_DIR, f'master_landscape_{case}.pdf'
    )
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()