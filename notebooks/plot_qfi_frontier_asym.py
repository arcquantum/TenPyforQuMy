#!/usr/bin/env python3
"""SNR vs QFI frontier for asymmetric echo cases."""
import argparse, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

CASE_DIRS = {
    '1d_closed_asym': 'data/echo_pts_1d_closed_asym',
    '2d_closed_asym': 'data/echo_pts_2d_closed_asym',
    '1d_open_asym':   'data/echo_pts_1d_open_asym',
    '2d_open_asym':   'data/echo_pts_2d_open_asym',
}
COLORS = {1.0: '#d95f02', 2.0: '#7570b3', 3.0: '#1b9e77',
          4.0: '#66a61e', 6.0: '#e7298a'}


def _front(d, tp_grid):
    out = np.full(len(tp_grid), np.nan)
    for (tp, tm), v in d.items():
        if not np.isfinite(v):
            continue
        j = int(np.argmin(np.abs(tp_grid - tp)))
        if not np.isfinite(out[j]) or v > out[j]:
            out[j] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=list(CASE_DIRS))
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--outfile', default=None)
    args = ap.parse_args()
    plt.rcParams.update({'text.usetex': False,
                         'mathtext.fontset': 'cm',
                         'font.family': 'serif', 'figure.dpi': 300})
    d_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        CASE_DIRS[args.case])
    if not os.path.isdir(d_path):
        print(f"[error] {d_path} missing"); sys.exit(1)
    is_open = '_open' in args.case
    qk = 'qfi_avg' if is_open else 'qfi'
    groups = {}
    for p in sorted(glob.glob(os.path.join(d_path, '*.npy'))):
        try:
            b = np.load(p, allow_pickle=True).item()
        except Exception:
            continue
        if qk not in b or 'snr' not in b or 't_plus' not in b:
            continue
        a = round(float(b['alpha']), 6)
        N = b.get('N') or (int(b.get('Lx', 0)) * int(b.get('Ly', 0)))
        if N is None:
            continue
        tp, tm = float(b['t_plus']), float(b['t_minus'])
        groups.setdefault((a, int(N)), {})[(tp, tm)] = {
            'snr': float(b['snr']), 'qfi': float(b[qk])}
    if not groups:
        print("[error] no data"); sys.exit(1)
    Ns = sorted({n for (_, n) in groups})
    tN = args.N if args.N is not None else Ns[0]
    if tN not in Ns:
        print(f"[error] N={tN} not in data. Have: {Ns}"); sys.exit(1)
    tp_all = sorted({tp for (_, n), d in groups.items()
                     if n == tN for (tp, _) in d})
    tp_grid = np.array(tp_all)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for a in sorted({a for (a, n) in groups if n == tN}):
        d = groups[(a, tN)]
        sf = _front({k: v['snr'] for k, v in d.items()}, tp_grid)
        qf = _front({k: v['qfi'] for k, v in d.items()}, tp_grid)
        c = COLORS.get(a, '#333')
        axes[0].plot(tp_grid, sf**2 / tN, '-', color=c,
                     label=fr'$\alpha={a}$')
        axes[1].plot(tp_grid, qf / tN**2, '-', color=c,
                     label=fr'$\alpha={a}$')
    axes[0].set_xlabel(r'$t_+$')
    axes[0].set_ylabel(r'$\max_{t_-}\mathrm{SNR}^2/N$')
    axes[0].set_title(f'SNR -- {args.case}, N={tN}')
    axes[0].set_yscale('log'); axes[0].grid(True, ls=':', alpha=0.5)
    axes[0].legend()
    axes[1].set_xlabel(r'$t_+$')
    axes[1].set_ylabel(
        r'$\max_{t_-}F_Q^{(\text{avg})}/N^2$' if is_open
        else r'$\max_{t_-}F_Q/N^2$')
    axes[1].set_title(f'QFI -- {args.case}, N={tN}')
    axes[1].set_yscale('log'); axes[1].grid(True, ls=':', alpha=0.5)
    axes[1].legend()
    fig.suptitle(f'SNR vs QFI -- {args.case}, N={tN}', fontsize=14)
    fig.tight_layout()
    out = args.outfile or os.path.join(
        'plots', f'qfi_frontier_{args.case}_N{tN}.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
