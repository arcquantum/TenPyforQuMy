#!/usr/bin/env python3
"""
notebooks/plot_decoherence_smallmultiples.py


2x2 grid, one panel per gamma2 (or gamma1) value, showing the (mu, nu)
landscape at fixed (N, alpha). Case must be an open-system case.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    CASE_1D_OPEN, CASE_2D_OPEN,
    setup_style, no_data, discover_bundles, parse_bundle_filename,
    load_bundle, bundle_metadata, bundle_axes, bundle_snr_2d,
    case_label, PLOTS_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default=CASE_1D_OPEN,
                        choices=[CASE_1D_OPEN, CASE_2D_OPEN])
    parser.add_argument('--N', type=int, default=None,
                        help='Fixed system size (1D cases).')
    parser.add_argument('--Lx', type=int, default=None)
    parser.add_argument('--Ly', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--channel', default='gamma2',
                        choices=['gamma1', 'gamma2'],
                        help='Which gamma to vary across panels.')
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    case = args.case
    paths = discover_bundles(case)
    if not paths:
        no_data(f'no bundles for case={case}')

    # Filter to fixed (N, alpha) and vary (channel)
    panels = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        if meta.get('alpha') != args.alpha:
            continue
        if args.N is not None and meta.get('N') != args.N:
            continue
        if args.Lx is not None and meta.get('Lx') != args.Lx:
            continue
        if args.Ly is not None and meta.get('Ly') != args.Ly:
            continue
        g_val = meta.get(args.channel)
        if g_val is None:
            continue
        try:
            x, y, xlab, ylab = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        panels.setdefault(float(g_val), (x, y, snr, xlab, ylab))

    if not panels:
        no_data(f'no bundles matching the requested fixed parameters')

    gs = sorted(panels.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), squeeze=False)
    axes_flat = axes.flatten()

    for idx, g in enumerate(gs):
        ax = axes_flat[idx]
        x, y, snr, xlab, ylab = panels[g]
        im = ax.imshow(
            snr, origin='lower', aspect='auto', cmap='viridis',
            extent=[x[0], x[-1], y[0], y[-1]],
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(rf'$\gamma_{{{args.channel[-1]}}}={g}$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(gs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    geom = f'$N$={args.N}' if args.N is not None else f'$L_x$={args.Lx}, $L_y$={args.Ly}'
    fig.suptitle(
        f'Decoherence small multiples — {case_label(case)}, '
        f'{geom}, ' + rf'$\alpha={args.alpha}$',
        fontsize=14,
    )
    fig.tight_layout()

    out = args.outfile or os.path.join(
        PLOTS_DIR, f'decoherence_smallmultiples_{case}.pdf'
    )
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()