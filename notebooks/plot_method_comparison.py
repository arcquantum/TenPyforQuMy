#!/usr/bin/env python3
"""
notebooks/plot_method_comparison.py


Compares Simple MPO, Bañuls, and TJM on the same axes. The script
looks for forward-mode time series (with keys 'time', 'Sx', 'xi_R2')
in data/tjm_forward/, data/banuls_pts/, data/simple_mpo/. If only
echo-point SNR data exists, it plots a marker comparison at the
requested (N, alpha, gamma1, gamma2).
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import ROOT_DIR, setup_style, PLOTS_DIR


def _scan_dir(dirname, patterns):
    """Return a list of (path, dict) for .npy dict files matching the patterns."""
    import glob
    out = []
    base = os.path.join(ROOT_DIR, 'data', dirname)
    if not os.path.isdir(base):
        return out
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(base, pat))):
            try:
                d = np.load(p, allow_pickle=True).item()
            except Exception:
                continue
            if isinstance(d, dict):
                out.append((p, d))
    return out


def _try_time_series(d):
    """Return (time, Sx, xi_R2) if d looks like a forward-sweep bundle."""
    if all(k in d for k in ('time', 'Sx', 'xi_R2')):
        return (np.asarray(d['time']), np.asarray(d['Sx']), np.asarray(d['xi_R2']))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--gamma1', type=float, default=0.0)
    parser.add_argument('--gamma2', type=float, default=0.1)
    parser.add_argument('--mu', type=float, default=np.pi / 2)
    parser.add_argument('--nu', type=float, default=-np.pi / 2)
    parser.add_argument('--outfile', type=str, default=None)
    args = parser.parse_args()
    setup_style()

    sources = {
        'Simple MPO': _scan_dir('simple_mpo', ['simple_mpo_forward_*.npy',
                                               'simple_mpo_*.npy']),
        'Bañuls':     _scan_dir('banuls_pts', ['banuls_forward_*.npy',
                                               'banuls_*.npy']),
        'TJM':        _scan_dir('tjm_forward', ['tjm_forward_*.npy',
                                                'tjm_*.npy']),
    }

    # Attempt time series first
    got_ts = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for tag, entries in sources.items():
        for _, d in entries:
            ts = _try_time_series(d)
            if ts is None:
                continue
            t, sx, xi = ts
            axes[0].plot(t, sx, label=tag)
            axes[1].plot(t, xi, label=tag)
            got_ts = True

    if got_ts:
        axes[0].set_xlabel(r'$t$')
        axes[0].set_ylabel(r'$\langle S_x \rangle$')
        axes[1].set_xlabel(r'$t$')
        axes[1].set_ylabel(r'$\xi_R^2$')
        axes[1].set_yscale('log')
        for ax in axes:
            ax.legend()
            ax.grid(True, ls=':', alpha=0.5)
        fig.suptitle('Method comparison — forward-mode time series')
        out = args.outfile or os.path.join(PLOTS_DIR, 'method_comparison.pdf')
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f'[plot] wrote {out} (time-series mode)')
        return

    # Fallback: single-point SNR markers
    fig, ax = plt.subplots(figsize=(7, 5))
    any_plot = False
    for tag, entries in sources.items():
        snrs = []
        for _, d in entries:
            if 'snr' in d and abs(d.get('N', args.N) - args.N) < 1e-9:
                snrs.append(float(d['snr']))
        if snrs:
            ax.axhline(np.mean(snrs), label=f'{tag} (mean SNR = {np.mean(snrs):.3f})')
            any_plot = True
    if not any_plot:
        print('[plot] no method-comparison data available in data/')
        sys.exit(0)
    ax.set_title(f'Method comparison at N={args.N}, alpha={args.alpha}')
    ax.set_ylabel('SNR')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    out = args.outfile or os.path.join(PLOTS_DIR, 'method_comparison.pdf')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out} (point-comparison mode)')


if __name__ == '__main__':
    main()