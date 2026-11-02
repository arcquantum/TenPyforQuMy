#!/usr/bin/env python3
"""
notebooks/plot_frontier_N.py


Rows = protocol (OUT / Squeeze / GHZ), columns = alpha, one line per N.
Supports --circumference-scaling which switches to the 2D cylinder
circumference scaling summary (the codebase the documentation, fig_ext_6.pdf).
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    ALL_CASES, is_scharnagl, setup_style, no_data,
    discover_bundles, parse_bundle_filename, load_bundle,
    bundle_metadata, bundle_axes, bundle_snr_2d, max_over_nu,
    normalised_sensitivity, case_label, PLOTS_DIR,
)


PROTO_LABELS = ['OUT', 'Squeeze', 'GHZ']


def _frontier_curves(case, protocol):
    """
    Return dict: (alpha) -> list of (N, x_vals, sens_vals).
    protocol in {'OUT','SQ','GHZ'} selects the (mu, nu) sub-family.
    """
    paths = discover_bundles(case)
    out = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        N = meta.get('N')
        if N is None and meta.get('Lx') is not None:
            N = (int(meta['Lx']), int(meta['Ly']))
        try:
            x, y, _, _ = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        # Select protocol.
        # Migrated bundles: (x, y) = (mu1, mu2), mu2 in [0, 2*pi].
        #   OUT (nu = -mu): mu2 = 2*mu1
        #   SQ  (nu = +mu): mu2 = 0
        #   GHZ (nu = 0) : mu2 = mu1, restricted to the maximal-echo
        #                  window mu1 > 2*(pi - 4/sqrt(N))/N
        # Legacy bundles: (x, y) = (mu, nu), nu in [-pi, pi].
        #   OUT: nu = -mu
        #   SQ : nu = +mu
        #   GHZ: nu = 0 with mu > pi - 4/sqrt(N)
        if isinstance(N, tuple):
            N_int = N[0] * N[1]
        else:
            N_int = int(N) if N is not None else 32
        convention = 'mu1mu2' if 'mu1_vals' in b else 'mu_nu'
        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        mask = np.zeros_like(snr, dtype=bool)
        for i, xval in enumerate(x_arr):
            if convention == 'mu1mu2':
                if protocol == 'OUT':
                    target = 2.0 * xval
                elif protocol == 'SQ':
                    target = 0.0
                elif protocol == 'GHZ':
                    m1_window = (2.0 * (np.pi - 4.0 / max(np.sqrt(N_int), 1.0))
                                 / N_int)
                    if xval < m1_window:
                        continue
                    target = xval
                else:
                    continue
            else:
                if protocol == 'OUT':
                    target = -xval
                elif protocol == 'SQ':
                    target = xval
                elif protocol == 'GHZ':
                    if xval < np.pi - 4.0 / max(np.sqrt(N_int), 1.0):
                        continue
                    target = 0.0
                else:
                    continue
            j = int(np.argmin(np.abs(y_arr - target)))
            mask[i, j] = True
        sub = np.where(mask, snr, np.nan)
        if not np.any(np.isfinite(sub)):
            curve = None
        else:
            with np.errstate(invalid='ignore'):
                curve = np.nanmax(sub, axis=1)
        if curve is None:
            continue
        sens = normalised_sensitivity(curve, meta.get('N'))
        out.setdefault(meta.get('alpha'), []).append((N, x, sens))
    return out


def _plot_circumference_scaling(ax, case):
    """Alternate mode: 2D cylinder circumference scaling."""
    paths = discover_bundles(case)
    by_Ly = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        Ly = meta.get('Ly')
        if Ly is None:
            continue
        try:
            x, y, _, _ = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        curve = max_over_nu(snr, y)
        sens = normalised_sensitivity(curve, meta.get('N'))
        by_Ly.setdefault(int(Ly), []).append((meta.get('Lx'), x, sens))
    for Ly in sorted(by_Ly):
        for Lx, x, sens in sorted(by_Ly[Ly], key=lambda t: t[0] or 0):
            ax.plot(x, sens, label=f'$L_x$={Lx}, $L_y$={Ly}')
    ax.set_xlabel(r'$\mu_1$')
    ax.set_xlim(0.0, float(np.pi))
    ax.set_ylabel(r'$\mathrm{SNR}^2/N$')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, ls=':', alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=list(ALL_CASES))
    parser.add_argument('--circumference-scaling', action='store_true')
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    case = args.case

    if args.circumference_scaling:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_circumference_scaling(ax, case)
        fig.suptitle(f'Cylinder circumference scaling — {case_label(case)}',
                     fontsize=14)
        out = args.outfile or os.path.join(
            PLOTS_DIR, f'frontier_N_{case}_circumference.pdf'
        )
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f'[plot] wrote {out}')
        return

    # Standard mode: rows = protocol, columns = alpha
    alphas = set()
    for proto in ('OUT', 'SQ', 'GHZ'):
        for a in _frontier_curves(case, proto).keys():
            alphas.add(a)
    alphas = sorted(a for a in alphas if a is not None)
    if not alphas:
        no_data(f'no usable bundles for case={case}')

    n_rows = 3
    n_cols = len(alphas)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for j, alpha in enumerate(alphas):
        for i, proto in enumerate(('OUT', 'SQ', 'GHZ')):
            ax = axes[i, j]
            curves = _frontier_curves(case, proto).get(alpha, [])
            for N, x, sens in sorted(curves, key=lambda t: (t[0] is None, t[0])):
                label = (
                    f'$N$={N}' if not isinstance(N, tuple)
                    else f'$L_x$={N[0]}, $L_y$={N[1]}'
                )
                ax.plot(x, sens, label=label)
            if i == 0:
                ax.set_title(rf'$\alpha={alpha}$', fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    rf'{PROTO_LABELS[i]}' + '\n' + r'$\mathrm{SNR}^2/N$',
                    fontsize=11,
                )
            if i == n_rows - 1:
                ax.set_xlabel(r'$\mu_1$')
            ax.set_xlim(0.0, float(np.pi))
            ax.set_yscale('log')
            ax.grid(True, ls=':', alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)

    fig.suptitle(f'Frontier vs N — {case_label(case)} (in $\\mu_1, \\mu_2$)',
                 fontsize=14)
    fig.tight_layout()
    out = args.outfile or os.path.join(PLOTS_DIR, f'frontier_N_{case}.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()