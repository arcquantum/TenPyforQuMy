#!/usr/bin/env python3
"""
notebooks/plot_1d_vs_2d.py


Side-by-side landscape comparison: 1D closed vs 2D closed at fixed
N.  Two optional flags:

  --entropy    switch to the half-chain entanglement-entropy
               comparison (the codebase the documentation, fig_ext_7.pdf).
  --with-noise overlay the open-system curves from the 2D open case
               on top of the closed-system 2D landscape (the codebase
               the documentation).
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    CASE_1D_CLOSED, CASE_2D_CLOSED, CASE_2D_OPEN,
    setup_style, no_data, discover_bundles, parse_bundle_filename,
    load_bundle, bundle_metadata, bundle_axes, bundle_snr_2d,
    max_over_nu, normalised_sensitivity, PLOTS_DIR,
)


def _pick(bundles, N=None, Lx=None, Ly=None, alpha=None, g1=None, g2=None):
    """Return the first bundle matching the given filters."""
    for path in bundles:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        if N is not None and meta.get('N') != N:
            continue
        if Lx is not None and meta.get('Lx') != Lx:
            continue
        if Ly is not None and meta.get('Ly') != Ly:
            continue
        if alpha is not None and meta.get('alpha') != alpha:
            continue
        if g1 is not None and meta.get('gamma1') != g1:
            continue
        if g2 is not None and meta.get('gamma2') != g2:
            continue
        return b, meta
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--with-noise', action='store_true')
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    b1 = discover_bundles(CASE_1D_CLOSED)
    b2 = discover_bundles(CASE_2D_CLOSED)
    if not b1 or not b2:
        no_data('1D or 2D closed bundles missing')

    b1_bundle, b1_meta = _pick(b1, N=args.N, alpha=args.alpha)
    b2_bundle, b2_meta = _pick(b2, Lx=args.N // 4, Ly=4, alpha=args.alpha)
    if b1_bundle is None or b2_bundle is None:
        no_data(f'no bundles for N={args.N}, alpha={args.alpha}')
    if b1_meta is None or b2_meta is None:
        no_data(f'missing metadata for N={args.N}, alpha={args.alpha}')
    # Pylance now knows b1_bundle, b2_bundle, b1_meta, b2_meta are all
    # non-None for the rest of main().

    if args.entropy:
        # Entropy comparison: plot max_entropy vs mu in both cases.
        fig, ax = plt.subplots(figsize=(8, 5))
        for tag, b, meta in (
            ('1D closed', b1_bundle, b1_meta),
            ('2D closed', b2_bundle, b2_meta),
        ):
            if 'entropy' not in b:
                continue
            x = np.asarray(b.get('mu_vals', b.get('t_plus', [])))
            ent = np.asarray(b['entropy'])
            if len(x) != len(ent):
                continue
            ax.plot(x, ent, label=tag)
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$S_{vN}$')
        ax.set_title(f'Half-chain entropy — N={args.N}, alpha={args.alpha}')
        ax.legend()
        out = args.outfile or os.path.join(PLOTS_DIR, 'fig_ext_7.pdf')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f'[plot] wrote {out}')
        return

    # Landscape comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for ax, tag, b in zip(axes[0], ['1D closed', '2D closed'],
                          [b1_bundle, b2_bundle]):
        x, y, xlab, ylab = bundle_axes(b)
        snr = bundle_snr_2d(b)
        im = ax.imshow(snr, origin='lower', aspect='auto', cmap='viridis',
                       extent=[x[0], x[-1], y[0], y[-1]])
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(tag)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if args.with_noise:
        b3 = discover_bundles(CASE_2D_OPEN)
        pack3 = _pick(b3, Lx=args.N // 4, Ly=4, alpha=args.alpha,
                      g1=0.0, g2=0.1)
        b3_bundle, b3_meta = pack3
        if b3_bundle is not None and b3_meta is not None:
            x, y, _, _ = bundle_axes(b3_bundle)
            snr = bundle_snr_2d(b3_bundle)
            # Add a third panel: noise-free vs noisy frontier
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            x_c, y_c, _, _ = bundle_axes(b2_bundle)
            s_c = bundle_snr_2d(b2_bundle)
            ax2.plot(x_c, max_over_nu(s_c, y_c) ** 2 / b2_meta['N'],
                     label='2D closed')
            ax2.plot(x,  max_over_nu(snr, y) ** 2 / b3_meta['N'],
                     label=r'2D open, $\gamma_2=0.1$')
            ax2.set_xlabel(r'$\mu$')
            ax2.set_ylabel(r'max$_\nu$ $\mathrm{SNR}^2/N$')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.set_title('Closed vs open 2D cylinder')
            out2 = os.path.join(PLOTS_DIR, 'closed_vs_open_2d.pdf')
            fig2.tight_layout()
            fig2.savefig(out2)
            plt.close(fig2)
            print(f'[plot] wrote {out2}')

    fig.suptitle(f'1D vs 2D — alpha={args.alpha}, N={args.N}')
    out = args.outfile or os.path.join(PLOTS_DIR, 'landscape_1d_vs_2d.pdf')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()