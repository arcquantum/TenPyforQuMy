"""
scripts/workers/closed_2d/asymmetric_echo_2d_closed.py

Closed-system asymmetric echo worker for the 2D cylinder.

Protocol
--------
   |psi(phi)> = e^{+i H t_-} e^{-i phi S_n} e^{-i H t_+} |+>^N

with t_plus, t_minus >= 0. The Hamiltonian is the
PowerLawCylinderModel block-site construction of the documentation.

Convention
----------
   t_plus  = mu / 2
   t_minus = (mu - nu) / 2

The time-evolution helper ``evolve_mps_time`` is imported from
src/evolution.py; this file does not define its own copy.

Output
------
data/echo_pts_2d_closed_asym/ with keys
   {Lx, Ly, alpha, J, t_plus, t_minus, snr, method}
"""

import argparse
import os
import sys
import warnings


import numpy as np
import scipy.linalg
import h5py


warnings.filterwarnings("ignore", category=UserWarning, module="tenpy")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)


from tenpy.tools import hdf5_io
from src.models import PowerLawCylinderModel
from src.evolution import (
    initialize_css,
    apply_collective_rotation,
    evolve_mps_time,
)
from src.observables import _get_3x3_covariance, compute_qfi


def get_Q_mps(psi):
    _, Cov = _get_3x3_covariance(psi)
    return Cov


def get_M_mps(psi_base, model, chi_max, t_minus, dt_max=0.05, delta=1e-3):
    """
    M_ij = d/dphi <S_j> |_{phi=0} via central differences.
    The backward evolution is time-reversed: dt = -t_minus.
    """
    M = np.zeros((3, 3))
    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']

    for i, axis in enumerate(axes):
        for sign in (+1, -1):
            psi_copy = psi_base.copy()
            apply_collective_rotation(psi_copy, axis, sign * delta)
            evolve_mps_time(psi_copy, model, -t_minus, chi_max, dt_max)
            means = [np.real(np.sum(psi_copy.expectation_value(a)))
                     for a in op_names]
            for j in range(3):
                M[i, j] += sign * means[j] / (2.0 * delta)
    return M


def get_M_exact(psi_base, model, chi_max, t_minus, dt_max=0.05):
    """
    Analytic signal-response matrix via src.dM.exact_M.
    Passes nu_minus_mu = -2 * t_minus.
    """
    try:
        from src.dM import exact_M
    except ImportError as exc:
        raise SystemExit(
            "--method exact_derivative requires src/dM.py, which is "
            "scheduled for the documentation and is not yet available. "
            "Use --method finite_diff for now."
        ) from exc

    M, _psi_q = exact_M(
        psi_base, model,
        k_axis=None,
        nu_minus_mu=-2.0 * t_minus,
        chi_max=chi_max,
        dt_max=dt_max,
    )
    return M


def save_checkpoint(path, data):
    tmp = path + ".tmp"
    with h5py.File(tmp, 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            return hdf5_io.load_from_hdf5(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Lx", type=int, required=True)
    parser.add_argument("--Ly", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--t_plus", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--t_minus", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--chi_max", type=int, default=64)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--method",
                        choices=['finite_diff', 'exact_derivative'],
                        default='exact_derivative')
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()


    # --- (mu1, mu2) translation (migration) ---
    if getattr(args, "mu1", None) is not None and args.t_plus is None:
        _N = getattr(args, "N", None) or ((getattr(args, "Lx", 0) or 0) * (getattr(args, "Ly", 0) or 0))
        args.t_plus = _N * args.mu1 / 4.0
        args.t_minus = _N * args.mu2 / 4.0
        args.convention = "mu1mu2"
    elif args.t_plus is not None and getattr(args, "mu1", None) is None:
        args.convention = "internal"
    else:
        raise SystemExit("Provide either --t_plus --t_minus or --mu1 --mu2")
    args.mu_suffix = (f"_mu1{args.mu1:.4f}_mu2{args.mu2:.4f}"
                      if args.convention == "mu1mu2"
                      else f"_tplus{args.t_plus:.4f}_tminus{args.t_minus:.4f}")
    # --- End (mu1, mu2) translation ---
    assert args.t_plus >= 0.0, "t_plus must be a non-negative duration"
    assert args.t_minus >= 0.0, "t_minus must be a non-negative duration"

    if args.task_id is not None:
        fname = f"asym_pt_{args.task_id}.npy"
    else:
        fname = (f"asym_pt_Lx{args.Lx}_Ly{args.Ly}_alpha{args.alpha}"
                 f"{args.mu_suffix}.npy")

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'echo_pts_2d_closed_asym')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, fname)
    chk_file = out_file + ".checkpoint.h5"

    if os.path.exists(out_file):
        print(f"Already completed: {out_file}", flush=True)
        return

    chk = load_checkpoint(chk_file)

    model = PowerLawCylinderModel({
        'Lx': args.Lx, 'Ly': args.Ly,
        'J': args.J, 'alpha': args.alpha, 'conserve': None,
    })

    # ---- Stage 1: forward twisting ----
    if 'psi_base' in chk:
        psi_base = chk['psi_base']
        print("Loaded psi_base from checkpoint.", flush=True)
    else:
        psi_base = initialize_css(model.lat)
        evolve_mps_time(psi_base, model, +args.t_plus,
                        args.chi_max, args.dt_max)
        save_checkpoint(chk_file, {'psi_base': psi_base})
        print("Saved psi_base.", flush=True)

    # ---- Stage 2: signal response matrix ----
    if 'M' in chk:
        M = chk['M']
        print("Loaded M from checkpoint.", flush=True)
    else:
        if args.method == 'exact_derivative':
            print("Computing M via exact derivative.", flush=True)
            M = get_M_exact(psi_base, model, args.chi_max,
                            args.t_minus, args.dt_max)
        else:
            print("Computing M via finite differences.", flush=True)
            M = get_M_mps(psi_base, model, args.chi_max,
                          args.t_minus, args.dt_max)
        save_checkpoint(chk_file, {'psi_base': psi_base, 'M': M})
        print("Saved M.", flush=True)

    # ---- Stage 3: final state for covariance ----
    if 'psi_q' in chk:
        psi_q = chk['psi_q']
        print("Loaded psi_q from checkpoint.", flush=True)
    else:
        psi_q = psi_base.copy()
        evolve_mps_time(psi_q, model, -args.t_minus,
                        args.chi_max, args.dt_max)
        save_checkpoint(chk_file,
                        {'psi_base': psi_base, 'M': M, 'psi_q': psi_q})
        print("Saved psi_q.", flush=True)

    Q = get_Q_mps(psi_q)

    # ---- SNR via mask-based pseudo-inverse ----
    evals, evecs = np.linalg.eigh(Q)
    mask = evals > max(1e-12, 1e-8 * evals.max())
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T

    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    snr = float(np.max(sv))

    qfi = float(compute_qfi(psi_q))

    np.save(out_file, {
        'Lx': args.Lx, 'Ly': args.Ly,
        'alpha': args.alpha, 'J': args.J,
        'mu1': getattr(args, 'mu1', None), 'mu2': getattr(args, 'mu2', None), 't_plus': args.t_plus, 't_minus': args.t_minus,
        'snr': snr,
        'qfi': qfi,
        'qfi': qfi,
        'method': args.method,
    })
    if os.path.exists(chk_file):
        os.remove(chk_file)
    print(f"Asymmetric 2D closed SNR = {snr:.6f} "
          f"(Lx={args.Lx}, Ly={args.Ly}, alpha={args.alpha}, "
          f"t_plus={args.t_plus:.4f}, t_minus={args.t_minus:.4f}, "
          f"method={args.method})", flush=True)


if __name__ == "__main__":
    main()