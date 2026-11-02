"""
scripts/workers/shared/main_powerlaw.py

The main execution script for the Power-Law One-Axis Twisting finite-size
scaling sweep. Optimized for headless HPC deployment via SLURM job arrays
across the alpha-spectrum. Includes atomic HDF5 state-resumption to survive
cluster wall-time limits.

New in this revision (the documentation of the codebase documentation):
  * ``compression_method='exponential'`` is passed explicitly to
    ``PowerLawOATModel``, so the algebraic 1/r^alpha tail is approximated
    by a sum of exponentials and the MPO bond dimension becomes O(n_exp)
    instead of O(N).
  * Optional ``--Lx`` / ``--Ly`` arguments dispatch to
    ``PowerLawCylinderModel`` (2D cylinder, the documentation). Until that class
    is implemented, 2D dispatch raises a clear SystemExit.
  * The 1D and 2D branches produce identical output schemas, with the 2D
    ``.npy`` gaining ``Lx`` and ``Ly`` keys in its metadata dict, and the
    output filenames gaining ``Lx{Lx}_Ly{Ly}`` in place of ``N{N}``.
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

from src.models import PowerLawOATModel, PowerLawCylinderModel
from src.observables import (compute_squeezing, compute_qfi,
                             compute_entanglement_entropy)
from src.evolution import initialize_css, run_evolution_step
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution


def _build_model(args):
    """
    Build either the 1D chain or the 2D cylinder model, depending on whether
    --Lx / --Ly are provided. Raises a clear error if only one of them is
    given, or if --N is passed together with --Lx/--Ly.

    Returns
    -------
    model : CouplingMPOModel
        The TeNPy model object (1D or 2D).
    n_sites : int
        Total number of physical sites.
    is_2d : bool
        True if the 2D cylinder branch was taken.
    """
    have_Lx = args.Lx is not None
    have_Ly = args.Ly is not None

    if have_Lx != have_Ly:
        raise ValueError(
            "--Lx and --Ly must be provided together, or neither of them."
        )

    if have_Lx and have_Ly:
        # 2D cylinder (the documentation). If the class is still a placeholder, we
        # surface a clear error rather than an obscure ImportError.
        try:
            model = PowerLawCylinderModel({
                'Lx': args.Lx,
                'Ly': args.Ly,
                'J': args.J,
                'alpha': args.alpha,
                'conserve': None,
            })
        except NotImplementedError as exc:
            raise SystemExit(
                f"2D cylinder mode requires PowerLawCylinderModel, which is "
                f"scheduled for the documentation and is not yet available.\n"
                f"Underlying error: {exc}"
            )
        return model, args.Lx * args.Ly, True

    # 1D chain
    model = PowerLawOATModel({
        'N': args.N,
        'J': args.J,
        'alpha': args.alpha,
        'conserve': None,
        'compression_method': 'exponential',
    })
    return model, args.N, False


def _output_basename(args, is_2d):
    """Build the .npy / .h5 filename prefix for 1D or 2D runs."""
    if is_2d:
        return (f"powerlaw_Lx{args.Lx}_Ly{args.Ly}"
                f"_chi{args.chi_max}_alpha{args.alpha}")
    return (f"powerlaw_N{args.N}"
            f"_chi{args.chi_max}_alpha{args.alpha}")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Run MPO time evolution for the Power-Law OUT protocol."
    )
    # --- 1D-exclusive parameter ---
    parser.add_argument("--N", type=int, default=None,
                        help="Number of particles in the 1D chain "
                             "(mutually exclusive with --Lx/--Ly).")
    # --- shared parameters ---
    parser.add_argument("--chi_max", type=int, required=True,
                        help="Maximum MPS bond dimension.")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Power-law decay exponent (0 = all-to-all).")
    parser.add_argument("--t_max", type=float, required=True,
                        help="Total simulation time.")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step size.")
    parser.add_argument("--J", type=float, default=1.0,
                        help="Interaction strength coefficient.")
    # --- 2D-exclusive parameters ---
    parser.add_argument("--Lx", type=int, default=None,
                        help="Cylinder length (2D mode).")
    parser.add_argument("--Ly", type=int, default=None,
                        help="Cylinder circumference (2D mode).")
    args = parser.parse_args()

    # --N is required in 1D mode, forbidden in 2D mode.
    if args.Lx is None and args.Ly is None:
        if args.N is None:
            parser.error("In 1D mode, --N is required.")
    else:
        if args.N is not None:
            parser.error("In 2D mode (--Lx/--Ly given), --N must not be set.")

    model, n_sites, is_2d = _build_model(args)

    geometry_str = (f"Lx={args.Lx}, Ly={args.Ly}" if is_2d
                    else f"N={args.N}")
    print(f"Initializing Power-Law Simulation: {geometry_str}, "
          f"chi_max={args.chi_max}, alpha={args.alpha}, "
          f"J={args.J}, t_max={args.t_max}, dt={args.dt}, "
          f"n_sites={n_sites}, mode={'2D' if is_2d else '1D'}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    basename = _output_basename(args, is_2d)
    NPY_FILE = os.path.join(DATA_DIR, f"{basename}.npy")
    CHK_FILE = os.path.join(DATA_DIR, f"checkpoint_{basename}.h5")
    TMP_CHK = CHK_FILE + ".tmp"

    mpo_params = {
        'dt': args.dt,
        'approximation': 'II',
        'compression_method': 'SVD',
        'trunc_params': {'chi_max': args.chi_max, 'svd_min': 1e-12},
    }

    # -------- state-resumption --------
    if os.path.exists(CHK_FILE):
        print(f"Found HDF5 state! Resuming from: {CHK_FILE}", flush=True)
        with h5py.File(CHK_FILE, 'r') as f:
            chk = hdf5_io.load_from_hdf5(f)
        psi = chk['psi']
        start_step = int(chk['step'])
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

    # -------- evolution loop --------
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
            print(f"t = {current_time:.3f} | Squeezing = {xi_R_sq:.5f} | "
                  f"QFI = {qfi:.2f} | Trunc Error = {eps:.2e}", flush=True)

        if step % 50 == 0 or step == num_steps:
            meta = {
                'chi_max': args.chi_max,
                'alpha': args.alpha,
                'J': args.J,
                'dt': args.dt,
            }
            if is_2d:
                meta['Lx'] = args.Lx
                meta['Ly'] = args.Ly
            else:
                meta['N'] = args.N

            np.save(NPY_FILE, {
                'time': np.array(times),
                'squeezing': np.array(squeezing_vals),
                'qfi': np.array(qfi_vals),
                'entropy': np.array(entropy_vals),
                'trunc_error': np.array(truncation_errors),
                'metadata': meta,
            })

            with h5py.File(TMP_CHK, 'w') as f:
                hdf5_io.save_to_hdf5(f, {
                    'psi': psi,
                    'step': step,
                    'times': np.array(times),
                    'squeezing': np.array(squeezing_vals),
                    'qfi': np.array(qfi_vals),
                    'entropy': np.array(entropy_vals),
                    'trunc_error': np.array(truncation_errors),
                })
            os.replace(TMP_CHK, CHK_FILE)

    end_time = time.time()
    print(f"Simulation complete in {((end_time - start_time) / 60.0):.2f} minutes. "
          f"Data secured.", flush=True)


if __name__ == "__main__":
    main()