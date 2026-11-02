"""
scripts/workers/shared/main_fss.py

Execution script for standard One-Axis Twisting finite-size scaling.
Upgraded with TeNPy HDF5 serialization and atomic writes for infinite state-resumption.
"""

import sys
import os
import argparse
import numpy as np
import time
import h5py
import warnings
from tenpy.tools import hdf5_io

warnings.filterwarnings("ignore", message="unit_cell_width is a new argument")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

from src.models import OATModel 
from src.observables import compute_squeezing, compute_qfi, compute_entanglement_entropy
from src.evolution import initialize_css, run_evolution_step
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--chi_max", type=int, required=True)
    parser.add_argument("--t_max", type=float, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    print(f"Initializing OAT Resumable Simulation: N={args.N}, chi={args.chi_max}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Define Dual File Architecture (with Atomic Temp File)
    NPY_FILE = os.path.join(DATA_DIR, f"out_N{args.N}_chi{args.chi_max}_tmax{args.t_max}.npy")
    CHK_FILE = os.path.join(DATA_DIR, f"checkpoint_oat_N{args.N}_chi{args.chi_max}.h5")
    TMP_CHK = CHK_FILE + ".tmp"

    model_params = {'N': args.N, 'chi_interaction': 1.0, 'conserve': None}
    model = OATModel(model_params)
    
    # Restored 'compression_method' so TeNPy knows how to truncate the MPS
    mpo_params = {
        'dt': args.dt,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': args.chi_max, 'svd_min': 1e-12}
    }

    # 2. State-Resumption Logic
    if os.path.exists(CHK_FILE):
        print(f"Found HDF5 state! Resuming from: {CHK_FILE}", flush=True)
        with h5py.File(CHK_FILE, 'r') as f:
            chk = hdf5_io.load_from_hdf5(f)
        
        psi = chk['psi']
        start_step = int(chk['step'])
        
        # Explicitly cast deserialized arrays back to Python lists
        times = list(chk['times'])
        squeezing_vals = list(chk['squeezing'])
        qfi_vals = list(chk['qfi'])
        entropy_vals = list(chk['entropy'])
        truncation_errors = list(chk['trunc_error'])
    else:
        print("No checkpoint found. Starting fresh from t=0.0", flush=True)
        psi = initialize_css(model.lat)
        start_step = 0
        times = [0.0]
        squeezing_vals = [compute_squeezing(psi)]
        qfi_vals = [compute_qfi(psi)]
        entropy_vals = [compute_entanglement_entropy(psi)]
        truncation_errors = [0.0]

    engine = ExpMPOEvolution(psi, model, mpo_params)
    num_steps = int(args.t_max / args.dt)

    # 3. Evolution Loop
    for step in range(start_step + 1, num_steps + 1):
        eps = run_evolution_step(engine, N_steps=1)
        current_time = step * args.dt

        xi_R_sq = compute_squeezing(psi)
        qfi = compute_qfi(psi)
        S_vN = compute_entanglement_entropy(psi)

        times.append(current_time)
        squeezing_vals.append(xi_R_sq)
        qfi_vals.append(qfi)
        entropy_vals.append(S_vN)
        truncation_errors.append(eps)

        if step % 10 == 0:
            print(f"t = {current_time:.3f} | Squeezing = {xi_R_sq:.5f} | QFI = {qfi:.2f} | Trunc Error = {eps:.2e}", flush=True)

        # 4. Atomic HDF5 Serialization
        if step % 50 == 0 or step == num_steps:
            np.save(NPY_FILE, {
                'time': np.array(times),
                'squeezing': np.array(squeezing_vals),
                'qfi': np.array(qfi_vals),
                'entropy': np.array(entropy_vals),
                'trunc_error': np.array(truncation_errors),
                'metadata': {'N': args.N, 'chi_max': args.chi_max, 'dt': args.dt}
            })
            
            # Atomic Write: Save to temp, then replace
            with h5py.File(TMP_CHK, 'w') as f:
                hdf5_io.save_to_hdf5(f, {
                    'psi': psi,
                    'step': step,
                    'times': np.array(times),
                    'squeezing': np.array(squeezing_vals),
                    'qfi': np.array(qfi_vals),
                    'entropy': np.array(entropy_vals),
                    'trunc_error': np.array(truncation_errors)
                })
            os.replace(TMP_CHK, CHK_FILE) 

    print(f"Simulation complete in {((time.time() - start_time)/60.0):.2f} minutes.", flush=True)

if __name__ == "__main__":
    main()