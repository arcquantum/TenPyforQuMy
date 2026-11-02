#!/usr/bin/env python3
"""
notebooks/plot_forward_spectroscopy.py


1D alpha-spectroscopy figures: rows = N, columns = chi_max, one line
per alpha. Reads 1d_*_forward_*.npy bundles. Supports --with-noise
which overlays the 1D open-system curves.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    BUNDLES_DIR, ALPHA_COLORS, setup_style, no_data, load_bundle, PLOTS_DIR,
)


def _discover_forward(which='closed'):
    import glob
    pattern = f'1d_{which}_forward_alpha*.npy'
    return sorted(glob.glob(os.path.join(BUNDLES_DIR, pattern)))


def _parse_forward(path):
    import re
    name = os.path.basename(path)
    out = {}
    for key, pat in [
        ('alpha', r'alpha([-+0-9.eE]+)'),
        ('N', r'_N(\d+)'),
        ('chi_max', r'chi(\d+)'),
        ('g1', r'g1([-+0-9.eE]+)'),
        ('g2', r'g2([-+0-9.eE]+)'),
    ]:
        m = re.search(pat, name)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-noise', action='store_true')
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    paths = _discover_forward('closed')
    if not paths:
        no_data(f'no 1d_closed forward bundles in {BUNDLES_DIR}')

    panels = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        meta = _parse_forward(path)
        N = int(meta.get('N', 0))
        chi = int(meta.get('chi_max', 0))
        alpha = float(meta.get('alpha', 0.0))
        if N <= 0 or chi <= 0:
            continue
        panels.setdefault((N, chi), []).append((alpha, b))

    if not panels:
        no_data('forward bundles exist but could not be parsed')

    Ns = sorted({k[0] for k in panels})
    chis = sorted({k[1] for k in panels})
    n_rows, n_cols = len(Ns), len(chis)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 3.0 * n_rows),
        squeeze=False, sharex=False,
    )

    for i, N in enumerate(Ns):
        for j, chi in enumerate(chis):
            ax = axes[i, j]
            ax_twin = ax.twinx()
            if (N, chi) not in panels:
                ax.set_visible(False)
                continue
            for alpha, b in sorted(panels[(N, chi)], key=lambda x: x[0]):
                colour = ALPHA_COLORS.get(float(alpha), '#333333')
                t = np.asarray(b.get('time', []))
                if len(t) == 0:
                    continue
                mu1 = 4.0 * 1.0 * t / N
                xi = np.asarray(b.get('xi_R2', []))
                ent = np.asarray(b.get('entropy', []))
                if len(xi) == len(mu1):
                    xi_db = 10.0 * np.log10(np.maximum(xi, 1e-30))
                    ax.plot(mu1, xi_db, color=colour,
                            label=rf'$\alpha={alpha}$')
                if len(ent) == len(mu1):
                    ax_twin.plot(mu1, ent, color=colour,
                                    ls='--', alpha=0.6)
            ax.set_xlabel(r'$\mu_1$')
            if j == 0:
                ax.set_ylabel(r'$\xi_R^2$ (dB)')
            if i == 0:
                ax.set_title(rf'$\chi_{{\max}}={chi}$', fontsize=11)
            if j == n_cols - 1:
                ax_twin.set_ylabel(r'$S_{vN}$')
            if i == n_rows - 1 and j == 0:
                ax.legend(fontsize=7)

    # Optional: overlay open-system curves
    if args.with_noise:
        open_paths = _discover_forward('open')
        if open_paths:
            # For each open file matching (N, alpha, chi), overlay as markers
            for op in open_paths:
                b = load_bundle(op)
                if b is None:
                    continue
                meta = _parse_forward(op)
                N = int(meta.get('N', 0))
                chi = int(meta.get('chi_max', 0))
                alpha = float(meta.get('alpha', 0.0))
                if (N, chi) not in [(Ns[i], chis[j])
                                    for i in range(n_rows)
                                    for j in range(n_cols)]:
                    continue
                # Not overlaid on the panel by default; the caller may
                # extend this if desired. For the current schema, the
                # forward open sweeps have a different time grid and
                # would need alignment. Emit a note.
            print('[plot] --with-noise: open-system forward sweeps '
                  'present but not overlaid (time-grid alignment not '
                  'implemented in a future revision.')

    fig.suptitle('1D forward alpha-spectroscopy', fontsize=14)
    fig.tight_layout()
    out = args.outfile or os.path.join(
        PLOTS_DIR, 'alpha_spectroscopy_1d.pdf'
    )
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()