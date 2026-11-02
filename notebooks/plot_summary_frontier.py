#!/usr/bin/env python3
"""
notebooks/plot_summary_frontier.py


Single-panel summary: max_nu SNR^2/N vs mu, one curve per (alpha,
gamma) chosen from the flagship set of the given case. Invoked once
per case. In the codebase the documentation, the 1D closed and 2D open invocations
produce fig_main_2.pdf and fig_main_6.pdf.
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
    case_label, ALPHA_COLORS, is_open, PLOTS_DIR,
)


FLAGSHIP_ALPHAS = (0.0, 1.0, 3.0, 6.0)
FLAGSHIP_G2 = (0.01, 0.1, 0.5)


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

    # Collect all (alpha, gamma_label, mu, sens)
    curves = []
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        alpha = meta.get('alpha')
        if alpha is None:
            continue
        if float(alpha) not in FLAGSHIP_ALPHAS:
            continue
        if is_open(case):
            g1 = float(meta.get('gamma1') or 0.0)
            g2 = float(meta.get('gamma2') or 0.0)
            if g2 not in FLAGSHIP_G2:
                continue
            if g1 != 0.0:
                continue
            gamma_label = rf'$\gamma_2={g2}$'
        else:
            gamma_label = 'closed'

        try:
            x, y, xlab, ylab = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        curve = max_over_nu(snr, y)
        sens = normalised_sensitivity(curve, meta.get('N'))
        curves.append((float(alpha), gamma_label, x, sens, xlab, ylab))

    if not curves:
        no_data(f'no flagship tuples for case={case}')

    fig, ax = plt.subplots(figsize=(7.5, 5))
    seen = set()
    for alpha, glbl, x, sens, _, _ in sorted(curves, key=lambda t: (t[0], t[1])):
        key = (alpha, glbl)
        if key in seen:
            continue
        seen.add(key)
        colour = ALPHA_COLORS.get(alpha, '#333333')
        ls = {
            'closed': '-',
            r'$\gamma_2=0.01$': '--',
            r'$\gamma_2=0.1$':  '-.',
            r'$\gamma_2=0.5$':  ':',
        }.get(glbl, '-')
        ax.plot(x, sens, color=colour, linestyle=ls,
                label=rf'$\alpha={alpha}, {glbl}$')

    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6,
               label='SQL')
    # Labels reflect the convention of the bundles used
    xlab_used = curves[0][4] if curves else r'$\mu$'
    ylab_used = curves[0][5] if curves else r'$\nu$'
    y_sub = (ylab_used[1:-1]
             if ylab_used.startswith('$') and ylab_used.endswith('$')
             else ylab_used)
    ax.set_xlabel(xlab_used)
    ax.set_ylabel(rf'max$_{{{y_sub}}}$ $\mathrm{{SNR}}^2/N$')
    ax.set_yscale('log')
    ax.set_title(f'Summary frontier — {case_label(case)}')
    ax.legend(fontsize=8, loc='best', ncol=1)
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_xlim(0.0, float(np.pi))
    fig.tight_layout()

    out = args.outfile or os.path.join(
        PLOTS_DIR, f'summary_frontier_{case}.pdf'
    )
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()