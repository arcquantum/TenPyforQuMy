#!/usr/bin/env python3
"""Pristine single-panel echo landscape for one (case, N, alpha) bundle.

Produces one PDF per (case, N, alpha). Use this for the clean per-tuple
record; run plot_master_landscape_grid.py only after all tuples complete.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
BUNDLES = os.path.join(ROOT, 'data', 'bundles')
PLOTS = os.path.join(ROOT, 'plots')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--case', default='1d_closed')
    p.add_argument('--N', type=int, default=None)
    p.add_argument('--Lx', type=int, default=None)
    p.add_argument('--Ly', type=int, default=None)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--outfile', type=str, default=None)
    args = p.parse_args()

    if args.N is not None:
        fname = f'{args.case}_alpha{args.alpha}_N{args.N}.npy'
        geom_label = f'N={args.N}'
    elif args.Lx is not None and args.Ly is not None:
        fname = f'{args.case}_alpha{args.alpha}_Lx{args.Lx}_Ly{args.Ly}.npy'
        geom_label = f'Lx={args.Lx}, Ly={args.Ly}'
    else:
        sys.exit('Provide --N or --Lx/--Ly')

    path = os.path.join(BUNDLES, fname)
    if not os.path.exists(path):
        sys.exit(f'No bundle: {path}')

    d = np.load(path, allow_pickle=True).item()
    if 'mu1_vals' in d and 'mu2_vals' in d:
        x_vals = np.asarray(d['mu1_vals'], dtype=float)
        y_vals = np.asarray(d['mu2_vals'], dtype=float)
        convention = 'mu1mu2'
    elif 'mu_vals' in d and 'nu_vals' in d:
        x_vals = np.asarray(d['mu_vals'], dtype=float)
        y_vals = np.asarray(d['nu_vals'], dtype=float)
        convention = 'mu_nu'
    else:
        sys.exit(f'Bundle {path} has no (mu, nu) or (mu1, mu2) axes')
    snr = np.asarray(d['snr'], dtype=float)
    chi = d.get('chi_max') or d.get('chi_used')
    n_traj = d.get('n_traj')

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        snr.T, origin='lower', aspect='auto', cmap='viridis',
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
    )
    if convention == 'mu1mu2':
        ax.plot(x_vals, np.zeros_like(x_vals), 'w--', lw=1.6,
                label=r'$\mu_2=0$ (Squeeze)')
        ax.plot(x_vals, x_vals,               'w-.', lw=1.6,
                label=r'$\mu_2=\mu_1$ (ExactEcho)')
        ax.plot(x_vals, 2.0 * x_vals,         'w:',  lw=1.6,
                label=r'$\mu_2=2\mu_1$ (OUT)')
        ax.set_xlabel(r'$\mu_1$')
        ax.set_ylabel(r'$\mu_2$')
    else:
        ax.plot(x_vals, x_vals,                'w--', lw=1.6,
                label=r'$\nu=\mu$ (Squeeze)')
        ax.plot(x_vals, np.zeros_like(x_vals), 'w-.', lw=1.6,
                label=r'$\nu=0$ (ExactEcho)')
        ax.plot(x_vals, -x_vals,               'w:',  lw=1.6,
                label=r'$\nu=-\mu$ (OUT)')
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$\nu$')

    title = f'{args.case}, {geom_label}, $\\alpha={args.alpha}$'
    if chi is not None:
        title += f', $\\chi={chi}$'
    if n_traj:
        title += f', $N_{{\\rm traj}}={n_traj}$'
    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('SNR')
    ax.legend(loc='upper right', framealpha=0.85, fontsize=9)

    out = args.outfile or os.path.join(
        PLOTS,
        f'echo_landscape_{args.case}_{geom_label.replace(" ", "").replace(",", "_")}_alpha{args.alpha}.pdf',
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()
