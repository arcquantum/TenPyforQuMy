#!/usr/bin/env python3
"""
notebooks/plot_frontier_alpha.py


Slices of max_nu SNR^2/N vs mu, one line per alpha, one panel per N.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    ALL_CASES, setup_style, no_data, discover_bundles,
    parse_bundle_filename, load_bundle, bundle_metadata, bundle_axes,
    bundle_snr_2d, max_over_nu, normalised_sensitivity,
    case_label, ALPHA_COLORS, PLOTS_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=list(ALL_CASES))
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    case = args.case
    paths = discover_bundles(case)
    if not paths:
        no_data(f'no bundles for case={case}')

    # Collect: dict[N] = list of (alpha, mu, sens)
    by_N = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        N = meta.get('N')
        if N is None:
            # 2D bundles: use (Lx, Ly) as a composite key
            if meta.get('Lx') is not None and meta.get('Ly') is not None:
                N = (int(meta['Lx']), int(meta['Ly']))
            else:
                continue
        try:
            x, y, _, _ = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        snr_max = max_over_nu(snr, y)
        sens = normalised_sensitivity(snr_max, meta.get('N'))
        by_N.setdefault(N, []).append((meta.get('alpha'), x, sens))

    if not by_N:
        no_data(f'no usable bundles for case={case}')

    keys = sorted(by_N.keys(), key=lambda k: (isinstance(k, tuple), k))
    n = len(keys)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes_flat[idx]
        for alpha, x, sens in sorted(by_N[key], key=lambda t: t[0] or 0.0):
            colour = ALPHA_COLORS.get(float(alpha), '#333333')
            ax.plot(x, sens, color=colour, label=rf'$\alpha={alpha}$')
            ax.set_xlabel(r'$\mu_1$')
            ax.set_xlim(0.0, float(np.pi))
            ax.set_ylabel(r'max$_{\mu_2}$ $\mathrm{SNR}^2/N$')
        if isinstance(key, tuple):
            ax.set_title(f'$L_x$={key[0]}, $L_y$={key[1]}')
        else:
            ax.set_title(f'$N$={key}')
        ax.set_yscale('log')
        ax.grid(True, ls=':', alpha=0.5)
        ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.legend(loc='best', fontsize=8)

    for idx in range(len(keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f'Frontier vs $\\alpha$ — {case_label(case)}', fontsize=14)
    fig.tight_layout()

    out = args.outfile or os.path.join(PLOTS_DIR, f'frontier_alpha_{case}.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()