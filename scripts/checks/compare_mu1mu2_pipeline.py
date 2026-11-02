#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mu1mu2_to_mu_nu(N, mu1, mu2):
    # ROUND TO 6 DECIMALS to prevent filename mismatch errors
    mu = round(N * mu1 / 2.0, 6)
    nu = round(N * (mu1 - mu2) / 2.0, 6)
    return mu, nu

def run_one(task):
    N, alpha, mu1, mu2, chi, mode = task
    mu, nu = mu1mu2_to_mu_nu(N, mu1, mu2)
    if mode == 'exact':
        cmd = [sys.executable, 'scripts/workers/closed_1d/exact_echo_worker_1d.py',
               '--N', str(N), '--alpha', str(alpha),
               '--mu', f'{mu:.6f}', '--nu', f'{nu:.6f}']
        out = f"data/exact_1d_closed/exact_echo_1d_N{N}_alpha{alpha}_J1.0_mu{mu:.4f}_nu{nu:.4f}.npy"
    else:
        cmd = [sys.executable, 'scripts/workers/closed_1d/powerlaw_echo_worker.py',
               '--N', str(N), '--alpha', str(alpha), '--chi_max', str(chi),
               '--mu', f'{mu:.6f}', '--nu', f'{nu:.6f}',
               '--method', 'exact_derivative']
        out = f"data/powerlaw_echo_pts/plaw_echo_N{N}_chi{chi}_alpha{alpha}_mu{mu:.4f}_nu{nu:.4f}.npy"
        
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return (mu1, mu2, np.nan, f"rc={r.returncode}: {r.stderr[-300:]}")
    if not os.path.exists(out):
        return (mu1, mu2, np.nan, f"no_file: {out}")
        
    return (mu1, mu2, float(np.load(out, allow_pickle=True).item()['snr']), None)

def grid_fill(results, grid_res, mu1_grid, mu2_grid):
    arr = np.full((grid_res, grid_res), np.nan)
    errors = []
    for (mu1, mu2, val, err) in results:
        i = int(np.argmin(np.abs(mu2_grid - mu2)))
        j = int(np.argmin(np.abs(mu1_grid - mu1)))
        arr[i, j] = val
        if err:
            errors.append(f"  ERROR at (mu1={mu1:.4f}, mu2={mu2:.4f}): {err}")
    return arr, errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--chi', type=int, required=True)
    ap.add_argument('--grid_res', type=int, default=20)
    ap.add_argument('--n_cores', type=int, default=16)
    args = ap.parse_args()

    os.makedirs('plots', exist_ok=True)
    os.makedirs('data/exact_1d_closed', exist_ok=True)
    os.makedirs('data/powerlaw_echo_pts', exist_ok=True)

    json_file = f"sensitivity_landscape_chain_kz_svd_plus_x_N{args.N}_alpha{int(args.alpha)}.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    mu1_vals = np.array(data['mu1_vals'])
    mu2_vals = np.array(data['mu2_vals'])
    records = data['records']

    idx1 = np.linspace(0, len(mu1_vals) - 1, args.grid_res, dtype=int)
    idx2 = np.linspace(0, len(mu2_vals) - 1, args.grid_res, dtype=int)
    mu1_grid = mu1_vals[idx1]
    mu2_grid = mu2_vals[idx2]

    her = np.zeros((args.grid_res, args.grid_res))
    for i in range(args.grid_res):
        for j in range(args.grid_res):
            her[i, j] = records[idx1[j]][idx2[i]]['sensitivity_opt']

    tasks_exact = [(args.N, args.alpha, mu1, mu2, args.chi, 'exact')
                   for mu2 in mu2_grid for mu1 in mu1_grid]
    tasks_mps   = [(args.N, args.alpha, mu1, mu2, args.chi, 'mps')
                   for mu2 in mu2_grid for mu1 in mu1_grid]

    print(f"Running EXACT on {len(tasks_exact)} points with {args.n_cores} cores...")
    with mp.Pool(args.n_cores) as pool:
        exact_res = pool.map(run_one, tasks_exact)
    our_exact, errs_e = grid_fill(exact_res, args.grid_res, mu1_grid, mu2_grid)

    print(f"Running MPS on {len(tasks_mps)} points with {args.n_cores} cores...")
    with mp.Pool(args.n_cores) as pool:
        mps_res = pool.map(run_one, tasks_mps)
    our_mps, errs_m = grid_fill(mps_res, args.grid_res, mu1_grid, mu2_grid)

    if errs_e or errs_m:
        print(f"\nWARNING: {len(errs_e)} exact failures and {len(errs_m)} MPS failures.")
        for e in (errs_e + errs_m)[:5]:
            print(e)
    else:
        print("\nAll grid points computed successfully (no white blocks).")

    with np.errstate(invalid='ignore', divide='ignore'):
        err_e = np.abs(her - our_exact) / np.maximum(np.abs(her), 1e-9)
        err_m = np.abs(our_exact - our_mps) / np.maximum(np.abs(our_exact), 1e-9)
    print(f"Max rel error (Her exact vs Our exact): {np.nanmax(err_e):.3e}")
    print(f"Max rel error (Our exact vs Our MPS):   {np.nanmax(err_m):.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(right=0.88, wspace=0.3) # Make room for colorbar
    X, Y = np.meshgrid(mu1_grid, mu2_grid)
    
    vmin = np.nanmin([np.nanmin(her), np.nanmin(our_exact), np.nanmin(our_mps)])
    vmax = np.nanmax([np.nanmax(her), np.nanmax(our_exact), np.nanmax(our_mps)])
    
    titles = ['Reference exact', 'Annwoy Exact (QuTiP)', rf'Annwoy MPS ($\chi={args.chi}$)']
    data_arrays = [her, our_exact, our_mps]
    
    for ax, Z, title in zip(axes, data_arrays, titles):
        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis_r',
                           vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r'$\mu_1$')
        ax.set_ylabel(r'$\mu_2$')
        ax.plot(mu1_grid, mu1_grid,   'r--', alpha=0.5, label=r'$\mu_2=\mu_1$')
        ax.plot(mu1_grid, 2*mu1_grid, 'g--', alpha=0.5, label=r'$\mu_2=2\mu_1$')
        ax.set_xlim(mu1_grid[0], mu1_grid[-1])
        ax.set_ylim(mu2_grid[0], mu2_grid[-1])
        
    # Bulletproof Colorbar Placement
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Sensitivity')
    
    fig.suptitle(f'N={args.N}, alpha={args.alpha}, chi={args.chi}', fontsize=14)
    outpdf = f'plots/compare_reference_N{args.N}_alpha{int(args.alpha)}.pdf'
    fig.savefig(outpdf, bbox_inches='tight')
    print(f"Saved {outpdf}")

if __name__ == '__main__':
    main()