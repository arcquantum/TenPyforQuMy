"""
scripts/workers/shared/echo_worker.py

Distributed worker for the Echo Protocol Landscape Sweep.
Maps a single SLURM Array Task ID to a specific (mu, nu) coordinate
to compute the MPS time-reversal algorithm in parallel.

Includes HDF5 checkpoint/resume for cluster walltime interruptions.
"""

import numpy as np
import scipy.linalg
import os
import sys
import warnings
import argparse
import h5py
from tenpy.tools import hdf5_io

warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

from src.models import OATModel
from src.evolution import initialize_css, apply_collective_rotation
from src.observables import _get_3x3_covariance
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution


def evolve_mps_time(psi, model, total_time, chi_max):
    """
    Evolves the MPS forward or backward in time depending on the sign of total_time.
    """
    if abs(total_time) < 1e-12:
        return

    dt_max = 0.05
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt_step = total_time / N_steps  # preserves sign for time reversal

    mpo_params = {
        'dt': dt_step,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12}
    }

    engine = ExpMPOEvolution(psi, model, mpo_params)
    engine.options['N_steps'] = N_steps
    engine.run()


def get_Q_mps(psi):
    """
    Extracts the 3x3 quantum covariance matrix using the locked helper.
    """
    _, Cov = _get_3x3_covariance(psi)
    return Cov


def get_M_mps(psi_mu, model, chi_max, nu_minus_mu, delta=1e-3):
    """
    Extracts the 3x3 signal response matrix via central finite differences.
    """
    M = np.zeros((3, 3))
    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']

    for i, axis in enumerate(axes):
        # Forward phase imprint (+delta)
        psi_plus = psi_mu.copy()
        apply_collective_rotation(psi_plus, axis, delta)
        evolve_mps_time(psi_plus, model, nu_minus_mu / 2.0, chi_max)
        means_plus = [np.real(np.sum(psi_plus.expectation_value(a))) for a in op_names]

        # Backward phase imprint (-delta)
        psi_minus = psi_mu.copy()
        apply_collective_rotation(psi_minus, axis, -delta)
        evolve_mps_time(psi_minus, model, nu_minus_mu / 2.0, chi_max)
        means_minus = [np.real(np.sum(psi_minus.expectation_value(a))) for a in op_names]

        for j in range(3):
            M[i, j] = (means_plus[j] - means_minus[j]) / (2.0 * delta)

    return M


def save_checkpoint(checkpoint_file, data_dict):
    """
    Atomically save checkpoint data using HDF5.
    """
    tmp_file = checkpoint_file + ".tmp"
    with h5py.File(tmp_file, 'w') as f:
        hdf5_io.save_to_hdf5(f, data_dict)
    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(checkpoint_file):
    """
    Load checkpoint data if it exists, otherwise return empty dict.
    """
    if os.path.exists(checkpoint_file):
        with h5py.File(checkpoint_file, 'r') as f:
            return hdf5_io.load_from_hdf5(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--chi_max", type=int, default=256)
    parser.add_argument("--grid_size", type=int, default=40)
    args = parser.parse_args()

    # 1. Map 1D Task ID to 2D Grid Coordinates
    row = args.task_id // args.grid_size
    col = args.task_id % args.grid_size

    mu_vals = np.linspace(0, np.pi, args.grid_size)
    nu_vals = np.linspace(-np.pi, np.pi, args.grid_size)

    mu = mu_vals[col]
    nu = nu_vals[row]

    print(f"Task {args.task_id} initialized: Col {col} (mu={mu:.3f}), Row {row} (nu={nu:.3f})", flush=True)

    # Define output and checkpoint paths
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'echo_pts')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, f"pt_{args.task_id}.npy")
    checkpoint_file = os.path.join(DATA_DIR, f"checkpoint_echo_task{args.task_id}.h5")

    # If the final point already exists, nothing to do
    if os.path.exists(out_file):
        print(f"Task {args.task_id} already completed. Exiting.", flush=True)
        return

    # Load checkpoint if present
    chk = load_checkpoint(checkpoint_file)

    model = OATModel({'N': args.N, 'chi_interaction': 1.0, 'conserve': None})

    # 2. Obtain or compute the base state after initial squeezing
    if 'psi_base' in chk:
        psi_base = chk['psi_base']
        print("Loaded psi_base from checkpoint.", flush=True)
    else:
        psi_base = initialize_css(model.lat)
        evolve_mps_time(psi_base, model, mu / 2.0, args.chi_max)
        # Save psi_base immediately
        save_checkpoint(checkpoint_file, {'psi_base': psi_base})
        print("Saved psi_base to checkpoint.", flush=True)

    # 3. Obtain or compute the signal response matrix M
    nu_minus_mu = nu - mu
    if 'M' in chk:
        M = chk['M']
        print("Loaded M from checkpoint.", flush=True)
    else:
        M = get_M_mps(psi_base, model, args.chi_max, nu_minus_mu)
        # Save M
        chk_partial = {'psi_base': psi_base, 'M': M}
        save_checkpoint(checkpoint_file, chk_partial)
        print("Saved M to checkpoint.", flush=True)

    # 4. Obtain or compute psi_q and the covariance Q
    if 'psi_q' in chk:
        psi_q = chk['psi_q']
        print("Loaded psi_q from checkpoint.", flush=True)
    else:
        psi_q = psi_base.copy()
        evolve_mps_time(psi_q, model, nu_minus_mu / 2.0, args.chi_max)
        # Save psi_q
        chk_partial = {'psi_base': psi_base, 'M': M, 'psi_q': psi_q}
        save_checkpoint(checkpoint_file, chk_partial)
        print("Saved psi_q to checkpoint.", flush=True)

    Q = get_Q_mps(psi_q)

    # 5. Extract metrological SNR (with correct null‑direction handling)
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt_evals = np.zeros_like(evals)
    inv_sqrt_evals[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt_evals) @ evecs.T

    F_matrix = M @ Q_inv_sqrt
    singular_values = scipy.linalg.svd(F_matrix, compute_uv=False)
    snr = np.max(singular_values)

    # 6. Save final point
    np.save(out_file, {
        'task_id': args.task_id,
        'row': row,
        'col': col,
        'mu': mu,
        'nu': nu,
        'snr': snr
    })

    # Optionally remove checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Task {args.task_id} securely completed. SNR = {snr:.3f}", flush=True)


if __name__ == "__main__":
    main()