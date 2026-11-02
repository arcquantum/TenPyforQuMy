#!/usr/bin/env python3
"""
notebooks/plot_chiautomation.py
Effective chi_prod heatmap for Phase-1 tuples.

chi_prod_effective = max(pilot_chi_prod_from_csv, CHI_FLOOR_POLICY[case,geometry,alpha])

Full 30-tuple grid per case. Tuples without a pilot CSV row are
shown as gray (no-pilot). Colour bar reflects the effective production
chi, which is what the workers actually run.
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
RESOURCE = os.path.join(ROOT, 'data', 'resource_tables')
CHI_CONV_CSV = os.path.join(RESOURCE, 'chi_conv.csv')
PLOTS = os.path.join(ROOT, 'plots')

sys.path.insert(0, os.path.join(ROOT, 'scripts', 'maintenance'))
from submit_production import PHASE_TUPLES, chi_prod_for, read_chi_conv  # noqa: E402
sys.path.insert(0, HERE)
from _plot_helpers import get_cmap  # noqa: E402


def read_csv(path: str) -> list[dict]:
    import csv
    if not os.path.exists(path):
        return []
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--outfile', default=None)
    args = p.parse_args()

    # Read pilot CSV into {(geometry, alpha): chi_prod}
    csv_lookup: dict[tuple, int] = {}
    for r in read_csv(CHI_CONV_CSV):
        if r.get('case') != args.case:
            continue
        try:
            alpha = round(float(r['alpha']), 6)
            if r.get('N', '').strip():
                geom = int(float(r['N']))
            else:
                geom = (int(float(r['Lx'])), int(float(r['Ly'])))
            chi = int(float(r.get('chi_prod') or r.get('chi_conv') or 0))
            csv_lookup[(geom, alpha)] = chi
        except (KeyError, ValueError, TypeError):
            continue

    phase = [td for td in PHASE_TUPLES.get(1, []) if td['case'] == args.case]
    if not phase:
        sys.exit(f'No PHASE_TUPLES[1] entries for case={args.case}')

    entries = []
    for td in phase:
        geom = int(td['N']) if td.get('N') is not None else (int(td['Lx']), int(td['Ly']))
        alpha = round(float(td['alpha']), 6)
        chi_conv_table = read_chi_conv()
        chi_eff = chi_prod_for(td, chi_conv_table)
        entries.append((geom, alpha, chi_eff, None, None))

    geoms = sorted({e[0] for e in entries}, key=lambda g: (isinstance(g, tuple), g))
    alphas = sorted({e[1] for e in entries})

    grid = np.full((len(alphas), len(geoms)), np.nan)
    for geom, alpha, chi_eff, _, _ in entries:
        i = alphas.index(alpha)
        j = geoms.index(geom)
        grid[i, j] = chi_eff

    cmap = get_cmap('viridis', copy=True)
    cmap.set_bad(color='#d0d0d0')
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(geoms)),
                                    max(3, 0.75 * len(alphas))))
    im = ax.imshow(np.ma.masked_invalid(grid), origin='lower',
                   aspect='auto', cmap=cmap)
    vmax = np.nanmax(grid) if np.any(np.isfinite(grid)) else 1.0
    for i in range(len(alphas)):
        for j in range(len(geoms)):
            v = grid[i, j]
            txt = f'{int(v)}' if np.isfinite(v) else '–'
            color = 'white' if (np.isfinite(v) and v > 0.6 * vmax) else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=10, color=color)
    ax.set_xticks(range(len(geoms)))
    ax.set_xticklabels([f'N={g}' if not isinstance(g, tuple)
                        else f'({g[0]},{g[1]})' for g in geoms], fontsize=9)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f'{a}' for a in alphas], fontsize=9)
    ax.set_xlabel('Geometry')
    ax.set_ylabel(r'$\alpha$')
    ax.set_title(
        r'Effective $\chi_{\mathrm{prod}} = \max(\chi_{\mathrm{pilot}},\,'
        r'\chi_{\mathrm{floor}})$' + f' — {args.case}'
    )
    if np.any(np.isfinite(grid)):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=r'$\chi_{\mathrm{prod}}$ (effective)')

    out = args.outfile or os.path.join(
        PLOTS, f'chi_automation_{args.case}.pdf'
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()
