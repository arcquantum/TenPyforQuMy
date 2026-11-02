#!/usr/bin/env python
"""
scripts/maintenance/check_qfi_scaling_1d.py
QFI scaling check for 1D closed systems.

For each alpha in {1, 2, 3, 4, 6} and each N in {16, 32, 64, 128, 160, 200},
evolve the CSS under the power-law Ising Hamiltonian to a fixed
forward-twist strength mu/2, extract F_Q, and write a CSV plus a
publication PDF showing F_Q/N^2 vs N.

Answers the physics question: does F_Q/N^2 grow with N at finite
alpha? If yes, non-GHZ Heisenberg scaling exists even though the
SNR may be limited to sqrt(N). If flat, sqrt(N) is the ceiling.

Runs in ~30 min single-core at chi=32.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from src.models import PowerLawOATModel
from src.evolution import initialize_css, evolve_mps_time
from src.observables import compute_qfi

J = 1.0
ALPHAS = [1.0, 2.0, 3.0, 4.0, 6.0]
NS = [16, 32, 64, 128, 160, 200]
CHI = 32
DT = 0.05


def run_sweep(mu1, output_csv, ns_to_run):
    """Iterate over (alpha, N), write incrementally, skip already-done."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fields = ['alpha', 'N', 'mu1', 'qfi', 'qfi_per_N', 'qfi_per_N2']

    done = set()
    existing_rows = []
    if os.path.exists(output_csv):
        with open(output_csv, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    key = (round(float(row['alpha']), 6), int(row['N']))
                    done.add(key)
                    existing_rows.append(dict(row))
                except Exception:
                    continue
        print(f"[resume] {len(done)} existing rows in {output_csv}",
              flush=True)

    f_out = open(output_csv, 'a', newline='')
    writer = csv.DictWriter(f_out, fieldnames=fields)
    if not existing_rows:
        writer.writeheader()

    print(f"{'alpha':>6} {'N':>4} {'qfi':>14} {'qfi/N':>12} {'qfi/N^2':>12}",
          flush=True)

    for alpha in ALPHAS:
        for N in ns_to_run:
            key = (round(alpha, 6), N)
            if key in done:
                print(f"{alpha:>6.1f} {N:>4d}  (already done, skip)",
                      flush=True)
                continue
            t0 = time.time()
            try:
                model = PowerLawOATModel({
                    'N': N, 'J': J, 'alpha': alpha, 'conserve': None,
                    'compression_method': 'exponential',
                })
                psi = initialize_css(model.lat)
                evolve_mps_time(psi, model, N * mu1 / 4.0, CHI, DT)
                F_Q = float(compute_qfi(psi))
                row = {
                    'alpha': alpha, 'N': N, 'mu1': mu1,
                    'qfi': F_Q,
                    'qfi_per_N': F_Q / N,
                    'qfi_per_N2': F_Q / N**2,
                }
                writer.writerow(row)
                f_out.flush()
                os.fsync(f_out.fileno())
                existing_rows.append(row)
                print(f"{alpha:>6.1f} {N:>4d} {F_Q:>14.6e} "
                      f"{row['qfi_per_N']:>12.6e} "
                      f"{row['qfi_per_N2']:>12.6e}  "
                      f"[{time.time()-t0:.1f}s]", flush=True)
            except Exception as exc:
                print(f"{alpha:>6.1f} {N:>4d} ERROR: {exc}", flush=True)

    f_out.close()
    print(f"\nWrote {output_csv}")
    return existing_rows


def plot(results, output_pdf):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {1.0: '#d95f02', 2.0: '#7570b3',
              3.0: '#1b9e77', 4.0: '#66a61e', 6.0: '#e7298a'}
    for alpha in sorted({r['alpha'] for r in results}):
        sub = [r for r in results if r['alpha'] == alpha]
        sub.sort(key=lambda r: r['N'])
        Ns = [r['N'] for r in sub]
        FQN2 = [r['qfi_per_N2'] for r in sub]
        ax.plot(Ns, FQN2, 'o-', color=colors.get(alpha, '#333'),
                label=fr'$\alpha={alpha}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$F_Q / N^2$')
    ax.set_title(r'QFI scaling at fixed $\mu=\pi/2$ (1D closed)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {output_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu1', type=float, default=np.pi / 2,
                        help='Forward twist strength in mu1 units '
                             '(mu1 = 4 J t / N); default pi/2')
    parser.add_argument('--max-N', type=int, default=160,
                        help='Skip N values above this (default 160 to '
                             'avoid the slow N=200 step). Set to 200 to '
                             'include N=200.')
    parser.add_argument('--output-csv', type=str,
                        default='data/validation_reports/qfi_scaling_1d.csv')
    parser.add_argument('--output-pdf', type=str,
                        default='plots/qfi_scaling_1d.pdf')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    ns_to_run = [n for n in NS if n <= args.max_N]
    print(f"Running N in {ns_to_run}", flush=True)
    results = run_sweep(args.mu1, args.output_csv, ns_to_run)
    if results:
        plot(results, args.output_pdf)
    return 0


if __name__ == '__main__':
    sys.exit(main())