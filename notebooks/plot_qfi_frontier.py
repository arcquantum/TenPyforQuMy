#!/usr/bin/env python3
"""
notebooks/plot_qfi_frontier.py
SNR vs QFI frontier comparison.

For each alpha, compute max_nu SNR^2/N vs mu and max_nu QFI/N^2 vs mu
and plot side by side so that SNR^2 <= QFI is visible directly.

CSV fallback: for 1d_closed, N=16 production files predate the QFI
patch, so QFI at N=16 is read from qfi_scaling_1d.csv. Only a single
mu point (pi/2) is available; the SNR panel stays empty at N=16.

    python notebooks/plot_qfi_frontier.py --case 1d_closed
"""
import argparse
import csv
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

CASE_DIRS = {
    '1d_closed':      'data/powerlaw_echo_pts',
    '2d_closed':      'data/powerlaw_cylinder_pts',
    '1d_open':        'data/tjm_pts',
    '2d_open':        'data/tjm_pts',
    '1d_closed_asym': 'data/echo_pts_1d_closed_asym',
    '2d_closed_asym': 'data/echo_pts_2d_closed_asym',
    '1d_open_asym':   'data/echo_pts_1d_open_asym',
    '2d_open_asym':   'data/echo_pts_2d_open_asym',
}

ALPHA_COLORS = {
    0.0: '#000000', 1.0: '#d95f02', 2.0: '#7570b3',
    3.0: '#1b9e77', 4.0: '#66a61e', 6.0: '#e7298a',
}


def _frontier(data_dict, mu_grid):
    """For each mu, pick the finite value that maximizes it. Skip NaN."""
    out = np.full(len(mu_grid), np.nan)
    for (mu, nu), val in data_dict.items():
        if not np.isfinite(val):
            continue
        j = int(np.argmin(np.abs(mu_grid - mu)))
        if not np.isfinite(out[j]) or val > out[j]:
            out[j] = val
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=list(CASE_DIRS))
    parser.add_argument('--N', type=int, default=None,
                        help='Restrict to one N (uses smallest N if unset)')
    parser.add_argument('--outfile', default=None)
    args = parser.parse_args()

    plt.rcParams.update({
        'text.usetex': False, 'mathtext.fontset': 'cm',
        'font.family': 'serif', 'figure.dpi': 300,
    })

    dirpath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        CASE_DIRS[args.case],
    )
    if not os.path.isdir(dirpath):
        print(f"[error] directory missing: {dirpath}")
        sys.exit(1)

    is_open = '_open' in args.case
    qfi_key = 'qfi_avg' if is_open else 'qfi'

    groups = {}

    for path in sorted(glob.glob(os.path.join(dirpath, '*.npy'))):
        try:
            d = np.load(path, allow_pickle=True).item()
        except Exception:
            continue
        if qfi_key not in d or 'snr' not in d:
            continue
        alpha = round(float(d['alpha']), 6)
        if 'N' in d and d['N'] is not None:
            N = int(d['N'])
        elif 'Lx' in d and 'Ly' in d:
            N = int(d['Lx']) * int(d['Ly'])
        else:
            continue
        mu1_val = d.get('mu1')
        mu2_val = d.get('mu2')
        if mu1_val is not None and mu2_val is not None:
            mu1 = float(mu1_val)
            mu2 = float(mu2_val)
        else:
            mu = float(d.get('mu', 0.0))
            nu = float(d.get('nu', 0.0))
            mu1 = 2.0 * mu / N if N > 0 else 0.0
            mu2 = 2.0 * (mu - nu) / N if N > 0 else 0.0
        groups.setdefault((alpha, N), {})[(mu1, mu2)] = {
            'snr': float(d['snr']),
            'qfi': float(d[qfi_key]),
        }

    if not groups:
        print(f"[error] no data in {dirpath}")
        sys.exit(1)

    Ns = sorted({n for (_, n) in groups})
    target_N = args.N if args.N is not None else Ns[0]
    if target_N not in Ns:
        print(f"[error] N={target_N} not in data. Available: {Ns}")
        sys.exit(1)

    mu1_all = sorted({mu1 for (_, n), d in groups.items()
                      if n == target_N for (mu1, _) in d})
    mu1_grid = np.array(mu1_all)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for alpha in sorted({a for (a, n) in groups if n == target_N}):
        d = groups[(alpha, target_N)]
        snr_front = _frontier({k: v['snr'] for k, v in d.items()},
                              mu1_grid)
        qfi_front = _frontier({k: v['qfi'] for k, v in d.items()},
                              mu1_grid)
        c = ALPHA_COLORS.get(alpha, '#333')
        axes[0].plot(mu1_grid, snr_front**2 / target_N, '-', color=c,
                     label=fr'$\alpha={alpha}$')
        axes[1].plot(mu1_grid, qfi_front / target_N**2, '-', color=c,
                     label=fr'$\alpha={alpha}$')

    axes[0].set_xlabel(r'$\mu_1$')
    axes[0].set_xlim(0.0, float(np.pi))
    axes[0].set_ylabel(r'$\max_{\mu_2} \mathrm{SNR}^2 / N$')
    axes[0].set_title(f'SNR — {args.case}, N={target_N}')
    axes[0].set_yscale('log')
    axes[0].grid(True, which='both', ls=':', alpha=0.5)
    axes[0].legend()

    axes[1].set_xlabel(r'$\mu_1$')
    axes[1].set_xlim(0.0, float(np.pi))
    label = (r'$\max_{\mu_2} F_Q^{(\text{avg})} / N^2$' if is_open
                else r'$\max_{\mu_2} F_Q / N^2$')
    axes[1].set_ylabel(label)
    axes[1].set_title(f'QFI — {args.case}, N={target_N}')
    axes[1].set_yscale('log')
    axes[1].grid(True, which='both', ls=':', alpha=0.5)
    axes[1].legend()

    fig.suptitle(
        f'SNR vs QFI — {args.case}, N={target_N}',
        fontsize=14,
    )
    fig.tight_layout()
    out = args.outfile or os.path.join(
        'plots', f'qfi_frontier_{args.case}_N{target_N}.pdf'
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
