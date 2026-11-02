#!/usr/bin/env python3
"""
notebooks/plot_asymmetric_echo.py


Small-multiple grid over (alpha, N or geometry) for one of the four
asymmetric echo case identifiers. Reference curves t_- = t_+,
t_- = 2 t_+, and t_- = 0 are overlaid on each panel.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    ASYM_CASES, is_open, setup_style, no_data, discover_bundles,
    parse_bundle_filename, load_bundle, bundle_metadata, bundle_axes,
    bundle_snr_2d, case_label, PLOTS_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=list(ASYM_CASES))
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    case = args.case
    paths = discover_bundles(case)
    if not paths:
        no_data(f'no bundles for case={case}')

    panels = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        if meta.get('Lx') is not None and meta.get('Ly') is not None:
            geom = (int(meta['Lx']), int(meta['Ly']))
        elif meta.get('N') is not None:
            geom = (int(meta['N']),)
        else:
            continue
        try:
            t_plus, t_minus, xlab, ylab = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        key = (meta.get('alpha'), geom)
        panels[key] = (t_plus, t_minus, snr, xlab, ylab)

    if not panels:
        no_data(f'no usable bundles for case={case}')

    alphas = sorted({k[0] for k in panels})
    geoms = sorted({k[1] for k in panels})
    n_rows, n_cols = len(alphas), len(geoms)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for i, a in enumerate(alphas):
        for j, g in enumerate(geoms):
            ax = axes[i, j]
            if (a, g) not in panels:
                ax.set_visible(False)
                continue
            t_plus, t_minus, snr, xlab, ylab = panels[(a, g)]
            im = ax.imshow(
                snr, origin='lower', aspect='auto', cmap='viridis',
                extent=[t_plus[0], t_plus[-1], t_minus[0], t_minus[-1]],
            )
            # Reference curves
            tp = np.linspace(t_plus[0], t_plus[-1], 100)
            ax.plot(tp, tp, 'w--', lw=1.2, label=r'$t_-=t_+$')
            ax.plot(tp, 2 * tp, 'w-.', lw=1.2, label=r'$t_-=2t_+$')
            ax.plot(tp, np.zeros_like(tp), 'w:', lw=1.2, label=r'$t_-=0$')
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            if i == 0:
                lbl = (f'$N$={g[0]}' if len(g) == 1
                       else f'$L_x$={g[0]}, $L_y$={g[1]}')
                ax.set_title(lbl, fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel(xlab)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Asymmetric echo — {case_label(case)}', fontsize=14)
    fig.tight_layout()

    out = args.outfile or os.path.join(
        PLOTS_DIR, f'asymmetric_echo_{case}.pdf'
    )
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()