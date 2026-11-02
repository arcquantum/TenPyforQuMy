#!/usr/bin/env python3
"""
notebooks/plot_dephasing_scaling.py


Consolidated dephasing-scaling figure (Schulte Fig. 3 analogue):
one panel per protocol, one curve per sigma in {0, 0.01, 0.5}, MPS
validation markers overlaid at sigma=0 restricted to the one-percent
agreement window.

Absorbs the functionality of plot_scaling_mps.py and
plot_scaling_aggregate.py, which are deprecated by this consolidated
module.

Reads scaling_c*_maxN*.npy from either data/bundles/ (new path) or
data/ (legacy path). Writes plots/dephasing_scaling.pdf and
plots/fig_ext_8.pdf.
"""

from __future__ import annotations
import os, sys, argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
   ROOT_DIR, BUNDLES_DIR, setup_style, no_data, PLOTS_DIR, get_cmap,
)


SIGMAS = [0.0, 0.01, 0.5]
PROTOCOLS = ['OUT', 'SQ', 'GHZ']
PROTO_LABELS = ['OUT', 'Squeezing', 'GHZ']
CHI_MARKERS = {16: 'o', 64: 's', 128: 'D', 256: '^'}


def _find_scaling_files():
    """Return sorted list of scaling_c*_maxN*.npy from bundles/ or data/."""
    files = sorted(glob.glob(os.path.join(BUNDLES_DIR, 'scaling_c*_maxN*.npy')))
    if files:
        return files
    return sorted(glob.glob(os.path.join(ROOT_DIR, 'data', 'scaling_c*_maxN*.npy')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    files = _find_scaling_files()
    if not files:
        no_data('no scaling_c*_maxN*.npy files in bundles/ or data/')

    by_chi = {}
    for f in files:
        try:
            d = np.load(f, allow_pickle=True).item()
        except Exception:
            continue
        chi = d.get('chi_max')
        if chi is not None:
            by_chi[int(chi)] = d

    if not by_chi:
        no_data('scaling files present but contain no chi_max metadata')

    chi_list = sorted(by_chi.keys())
    ref_data = by_chi[chi_list[0]]
    N_list = np.asarray(ref_data['N_list'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (proto, label) in enumerate(zip(PROTOCOLS, PROTO_LABELS)):
        ax = axes[i]
        for sig in SIGMAS:
            if sig not in ref_data['analytical']:
                continue
            analytical = np.asarray(ref_data['analytical'][sig][proto])
            ax.plot(N_list, analytical, label=rf'$\sigma={sig}$')
        if 0.0 in ref_data['analytical']:
            analytical0 = np.asarray(ref_data['analytical'][0.0][proto])
            for chi in chi_list:
                d = by_chi[chi]
                key = {'OUT': 'snr_out_pure',
                       'SQ':  'snr_sq_pure',
                       'GHZ': 'snr_ghz_pure'}[proto]
                if key not in d:
                    continue
                mps = np.asarray(d[key])
                # One-percent agreement threshold
                valid = np.array([
                    abs(m - a) / max(a, 1e-30) < 0.01
                    for m, a in zip(mps, analytical0)
                ])
                if not np.any(valid):
                    continue
                ax.scatter(
                    N_list[valid], mps[valid],
                    marker=CHI_MARKERS.get(chi, 'o'),
                    facecolors='none',
                    edgecolors=get_cmap('tab10')(i),
                    linewidths=1.5, s=60,
                    label=rf'MPS $\chi={chi}$',
                )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$\Delta\phi^{-2}/N$')
        ax.set_title(label)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle('Dephasing scaling (Schulte Fig. 3 analogue)', fontsize=14)
    fig.tight_layout()

    out1 = args.outfile or os.path.join(PLOTS_DIR, 'dephasing_scaling.pdf')
    out2 = os.path.join(PLOTS_DIR, 'fig_ext_8.pdf')
    fig.savefig(out1)
    fig.savefig(out2)
    plt.close(fig)
    print(f'[plot] wrote {out1}')
    print(f'[plot] wrote {out2}')


if __name__ == '__main__':
    main()