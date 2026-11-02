#!/usr/bin/env python3
"""
notebooks/plot_phase_diagram.py


2D (mu, alpha) phase diagram, colored by dominant protocol
(OUT / ExactEcho / Squeezing / Other).


Produces one PDF per geometry found in the case's bundles. Pilot
bundles at different (alpha, N) tuples often have different
(mu, nu) grid sizes; each is bilinearly resampled onto a common
40 x 40 reference grid before the phase diagram is assembled. This
makes the script robust to the heterogeneous pilot data and
produces a clean (mu, alpha) image for each geometry.

Use --N (1D) or --Lx/--Ly (2D) to restrict to a single geometry.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _plot_helpers import (
    ALL_CASES, setup_style, no_data, discover_bundles,
    parse_bundle_filename, load_bundle, bundle_metadata,
    bundle_axes, bundle_snr_2d, classify_protocol,
    case_label, PLOTS_DIR,
)


REF_GRID_SIZE = 40


# ----------------------------------------------------------------------
# Geometry key / label
# ----------------------------------------------------------------------
def _geom_key(meta: dict):
    if meta.get('Lx') is not None and meta.get('Ly') is not None:
        return (int(meta['Lx']), int(meta['Ly']))
    if meta.get('N') is not None:
        return (int(meta['N']),)
    return None


def _geom_label(geom) -> str:
    if geom is None:
        return 'unknown'
    if len(geom) == 2:
        return f'Lx{geom[0]}_Ly{geom[1]}'
    return f'N{geom[0]}'


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def _resample_snr_to_reference(snr_2d, mu_vals, nu_vals,
                               ref_mu, ref_nu):
    """
    Bilinear-resample a 2D SNR array onto (ref_mu, ref_nu).

    Degenerate source grids (single mu or single nu point) are filled
    with the mean of the source array; that is the correct limit for
    a pilot bundle containing only one grid point.
    """
    from scipy.interpolate import RegularGridInterpolator

    mu_arr = np.asarray(mu_vals, dtype=float)
    nu_arr = np.asarray(nu_vals, dtype=float)
    snr_arr = np.asarray(snr_2d, dtype=float)

    if mu_arr.size < 2 or nu_arr.size < 2:
        v = float(np.nanmean(snr_arr))
        if not np.isfinite(v):
            v = 0.0
        return np.full((len(ref_mu), len(ref_nu)), v)

    i_mu = np.argsort(mu_arr)
    j_nu = np.argsort(nu_arr)
    mu_sorted = mu_arr[i_mu]
    nu_sorted = nu_arr[j_nu]
    snr_sorted = snr_arr[np.ix_(i_mu, j_nu)]

    interp = RegularGridInterpolator(
        (mu_sorted, nu_sorted), snr_sorted,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    MU, NU = np.meshgrid(ref_mu, ref_nu, indexing='ij')
    pts = np.column_stack([MU.ravel(), NU.ravel()])
    return interp(pts).reshape(len(ref_mu), len(ref_nu))


def _dominant_protocol_on_grid(snr_2d, x_vals, y_vals,
                                convention='mu_nu'):
    """
    For each x row, find the argmax over y and classify the
    (x, y_star) point. Returns an integer array over x:
    0 = OUT, 1 = ExactEcho, 2 = Squeezing, 3 = Other.
    """
    n_x = snr_2d.shape[0]
    out = np.full(n_x, 3, dtype=int)
    for i in range(n_x):
        row = snr_2d[i, :]
        if not np.any(np.isfinite(row)):
            continue
        j = int(np.nanargmax(row))
        proto = classify_protocol(float(x_vals[i]), float(y_vals[j]),
                                  convention=convention)
        if proto == 'OUT':
            out[i] = 0
        elif proto == 'ExactEcho':
            out[i] = 1
        elif proto == 'Squeezing':
            out[i] = 2
        else:
            out[i] = 3
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=list(ALL_CASES))
    parser.add_argument('--outfile', type=str, default=None,
                        help='Output PDF path. Only used when a single '
                             'geometry is present in the data or when '
                             '--N (or --Lx/--Ly) is given.')
    parser.add_argument('--N', type=int, default=None,
                        help='Restrict to a single 1D geometry.')
    parser.add_argument('--Lx', type=int, default=None)
    parser.add_argument('--Ly', type=int, default=None)
    args = parser.parse_args()
    setup_style()

    case = args.case
    paths = discover_bundles(case)
    if not paths:
        no_data(f'no bundles for case={case}')

    # ---- collect entries grouped by geometry ----
    by_geom = {}
    for path in paths:
        b = load_bundle(path)
        if b is None:
            continue
        fb = parse_bundle_filename(path) or {}
        meta = bundle_metadata(b, fallback=fb)
        geom = _geom_key(meta)
        if geom is None:
            continue
        try:
            x, y, _, _ = bundle_axes(b)
            snr = bundle_snr_2d(b)
        except Exception:
            continue
        convention = 'mu1mu2' if 'mu1_vals' in b else 'mu_nu'
        by_geom.setdefault(geom, []).append({
            'alpha': float(meta.get('alpha', 1.0)),
            'x_vals': np.asarray(x, dtype=float),
            'y_vals': np.asarray(y, dtype=float),
            'snr':    np.asarray(snr, dtype=float),
            'convention': convention,
        })

    if not by_geom:
        no_data(f'no usable bundles for case={case}')

    # ---- optional geometry filter ----
    if args.N is not None:
        by_geom = {k: v for k, v in by_geom.items()
                   if len(k) == 1 and k[0] == args.N}
    if args.Lx is not None and args.Ly is not None:
        by_geom = {k: v for k, v in by_geom.items()
                   if len(k) == 2 and k == (args.Lx, args.Ly)}
    if not by_geom:
        no_data('no bundles matching the requested geometry filter')

    # ---- common reference grid (x-axis shared; y-axis per convention) ----
    ref_x = np.linspace(0.0, np.pi, REF_GRID_SIZE)
    ref_y_by_conv = {
        'mu_nu':   np.linspace(-np.pi, np.pi, REF_GRID_SIZE),
        'mu1mu2':  np.linspace(0.0, 2.0 * np.pi, REF_GRID_SIZE),
    }

    # ---- colormap ----
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#cccccc']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5],
                        ncolors=len(colors))

    written = []
    single_geom = (len(by_geom) == 1)

    for geom in sorted(by_geom.keys()):
        entries = by_geom[geom]
        conventions = {e['convention'] for e in entries}
        if len(conventions) > 1:
            print(f"[warn] geometry {geom} has mixed conventions "
                  f"{conventions}; using per-entry grid, but the "
                  f"resampled rows will be over different domains.")
        alpha_rows = {}
        for e in entries:
            conv = e['convention']
            ref_y = ref_y_by_conv[conv]
            snr_ref = _resample_snr_to_reference(
                e['snr'], e['x_vals'], e['y_vals'], ref_x, ref_y,
            )
            dom = _dominant_protocol_on_grid(snr_ref, ref_x, ref_y,
                                             convention=conv)
            alpha_rows.setdefault(e['alpha'], []).append(dom)

        alphas = []
        rows = []
        for a in sorted(alpha_rows):
            doms = alpha_rows[a]
            # If an alpha has multiple conventions, take the first —
            # they should be equivalent as protocol classifications.
            alphas.append(a)
            rows.append(doms[0])

        alphas = np.asarray(alphas)
        rows = np.asarray(rows)   # (n_alpha, REF_GRID_SIZE)
        order = np.argsort(alphas)
        alphas = alphas[order]
        rows = rows[order, :]

        fig, ax = plt.subplots(figsize=(7.5, 5))
        im = ax.imshow(
            rows, origin='lower', aspect='auto',
            cmap=cmap, norm=norm,
            extent=[ref_x[0], ref_x[-1],
                    alphas[0], alphas[-1]],
        )
        geom_conventions = {e['convention'] for e in entries}
        xlabel = r'$\mu_1$' if geom_conventions == {'mu1mu2'} else r'$\mu$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\alpha$')
        ax.set_title(
            f'Phase diagram — {case_label(case)}, '
            f'{_geom_label(geom)}'
        )
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['OUT', 'ExactEcho',
                                 'Squeezing', 'Other'])

        if args.outfile and single_geom:
            out = args.outfile
        else:
            out = os.path.join(
                PLOTS_DIR,
                f'phase_diagram_{case}_{_geom_label(geom)}.pdf',
            )
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
        print(f'[plot] wrote {out}')

    if not written:
        no_data(f'no phase diagram produced for case={case}')


if __name__ == '__main__':
    main()