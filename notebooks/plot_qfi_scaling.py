#!/usr/bin/env python3
"""
QFI scaling plot: F_Q/N^2 vs N at fixed forward-twist strength mu.

Normal mode:  --case <identifier>, one panel.
Combine-all:  --combine-all, 2x2 panel (rows=closed/open, cols=1D/2D)
              for the codebase fig_ext_9b.pdf.
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

COMBINE_ORDER = [
    ('1d_closed', 0, 0, '1D closed'),
    ('2d_closed', 0, 1, '2D cylinder closed'),
    ('1d_open',   1, 0, '1D open'),
    ('2d_open',   1, 1, '2D cylinder open'),
]


def _collect_qfi(case, mu1, mu_tol):
    root = os.path.dirname(os.path.dirname(__file__))
    is_open = '_open' in case
    qfi_key = 'qfi_avg' if is_open else 'qfi'
    data = {}
    if case == '1d_closed':
        csv_path = os.path.join(
            root, 'data', 'validation_reports', 'qfi_scaling_1d.csv')
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline='') as f:
                    for row in csv.DictReader(f):
                        a = round(float(row['alpha']), 6)
                        n = int(row['N'])
                        q = float(row['qfi'])
                        key = (a, n)
                        if key not in data or q > data[key]:
                            data[key] = q
            except (csv.Error, UnicodeDecodeError):
                pass
                
    dirpath = os.path.join(root, CASE_DIRS[case])
    if os.path.isdir(dirpath):
        for path in sorted(glob.glob(os.path.join(dirpath, '*.npy'))):
            try:
                d = np.load(path, allow_pickle=True).item()
            except Exception:
                continue
                
            if 'N' in d and d['N'] is not None:
                n = int(d['N'])
            elif 'Lx' in d and 'Ly' in d:
                n = int(d['Lx']) * int(d['Ly'])
            else:
                continue

            mu1_val = d.get('mu1')
            if mu1_val is not None:
                val_mu1 = float(mu1_val)
            else:
                val_mu1 = 2.0 * float(d.get('mu', -1)) / n if n > 0 else -1.0

            if abs(val_mu1 - mu1) > mu_tol:
                continue
                
            if qfi_key not in d:
                continue
                
            a = round(float(d['alpha']), 6)
            key = (a, n)
            if key not in data or d[qfi_key] > data[key]:
                data[key] = float(d[qfi_key])
    return data, is_open


def _plot_on_ax(ax, data, is_open, mu):
    if not data:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return
    for a in sorted({k[0] for k in data}):
        Ns = sorted(n for aa, n in data if aa == a)
        FQN2 = [data[(a, n)] / n**2 for n in Ns]
        ax.plot(Ns, FQN2, 'o-', color=ALPHA_COLORS.get(a, '#333'),
                label=fr'$\alpha={a}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$F_Q^{(\mathrm{avg})}/N^2$' if is_open
                  else r'$F_Q/N^2$')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--case', choices=list(CASE_DIRS))
    g.add_argument('--combine-all', action='store_true')
    ap.add_argument('--mu1', type=float, default=np.pi / 2)
    ap.add_argument('--mu-tol', type=float, default=1e-3)
    ap.add_argument('--outfile', default=None)
    args = ap.parse_args()

    plt.rcParams.update({
        'text.usetex': False, 'mathtext.fontset': 'cm',
        'font.family': 'serif', 'figure.dpi': 300,
    })
    root = os.path.dirname(os.path.dirname(__file__))

    if args.combine_all:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for case, i, j, label in COMBINE_ORDER:
            data, is_open = _collect_qfi(case, args.mu1, args.mu_tol)
            ax = axes[i, j]
            _plot_on_ax(ax, data, is_open, args.mu1)
            ax.set_title(label + r', $\mu_1=\pi/2$')
        fig.suptitle('QFI scaling across all symmetric cases',
                     fontsize=14)
        fig.tight_layout()
        for out in [os.path.join(root, 'plots',
                                 'qfi_scaling_all_cases.pdf'),
                    os.path.join(root, 'plots', 'fig_ext_9b.pdf')]:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, bbox_inches='tight')
            print(f"wrote {out}")
        plt.close(fig)
        return

    data, is_open = _collect_qfi(args.case, args.mu1, args.mu_tol)
    if not data:
        print(f"[error] no data for case={args.case}")
        sys.exit(1)
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_on_ax(ax, data, is_open, args.mu1)
    ax.set_title(f'QFI scaling — {args.case}, ' + r'$\mu_1=\pi/2$')
    fig.tight_layout()
    out = args.outfile or os.path.join(
        root, 'plots', f'qfi_scaling_{args.case}.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
