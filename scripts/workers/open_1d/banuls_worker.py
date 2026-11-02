"""
scripts/workers/open_1d/banuls_worker.py

Bañuls locally-purified MPO worker for the 1D open-system power-law
echo protocol. Same CLI as powerlaw_echo_worker.py, plus --gamma1,
--gamma2, --eta_s, --block_size, --purification_dim, and
--disentangle_every.
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
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution

from src.models import PowerLawOATModel
from src.evolution import initialize_css, apply_collective_rotation
from src.observables import _get_3x3_covariance
from src.banuls import apply_dissipative_step, banuls_step


def evolve_mps_with_banuls(psi, model, total_time, chi_max, dt_max,
                           gamma1, gamma2, eta_s, block_size, d_p,
                           disentangle_every):
    """
    Evolve the MPS for a signed total_time, applying:
      - the exact dissipative contraction on each time step
      - the coherent evolution via ExpMPOEvolution
      - one Bañuls step every `disentangle_every` steps
    """
    if abs(total_time) < 1e-15:
        return
    N_steps = int(np.ceil(abs(total_time) / dt_max))
    dt_step = total_time / N_steps

    for step in range(N_steps):
        # Dissipative step (always forward in real time)
        if gamma1 > 0 or gamma2 > 0:
            apply_dissipative_step(psi, gamma1, gamma2, abs(dt_step))

        # Coherent step
        mpo_params = {
            'dt': dt_step,
            'approximation': 'II',
            'compression_method': 'SVD',
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12},
        }
        engine = ExpMPOEvolution(psi, model, mpo_params)
        engine.options['N_steps'] = 1
        engine.run()

        # Periodic Bañuls step
        if disentangle_every > 0 and (step + 1) % disentangle_every == 0:
            banuls_step(psi, chi_max, eta_s,
                        block_size=block_size, d_p=d_p)


def get_Q_mps(psi):
    _, Cov = _get_3x3_covariance(psi)
    return Cov


def get_M_mps(psi_mu, model, chi_max, nu_minus_mu, dt_max,
              gamma1, gamma2, eta_s, block_size, d_p, disentangle_every,
              delta=1e-3):
    M = np.zeros((3, 3))
    axes = ['x', 'y', 'z']
    op_names = ['Sx', 'Sy', 'Sz']

    for i, axis in enumerate(axes):
        psi_plus = psi_mu.copy()
        apply_collective_rotation(psi_plus, axis, delta)
        evolve_mps_with_banuls(psi_plus, model, nu_minus_mu / 2.0,
                               chi_max, dt_max, gamma1, gamma2,
                               eta_s, block_size, d_p, disentangle_every)
        means_plus = [np.real(np.sum(psi_plus.expectation_value(a)))
                      for a in op_names]

        psi_minus = psi_mu.copy()
        apply_collective_rotation(psi_minus, axis, -delta)
        evolve_mps_with_banuls(psi_minus, model, nu_minus_mu / 2.0,
                               chi_max, dt_max, gamma1, gamma2,
                               eta_s, block_size, d_p, disentangle_every)
        means_minus = [np.real(np.sum(psi_minus.expectation_value(a)))
                       for a in op_names]

        for j in range(3):
            M[i, j] = (means_plus[j] - means_minus[j]) / (2.0 * delta)
    return M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--chi_max", type=int, default=64)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--mu1", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--mu2", type=float, default=None)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--dt_max", type=float, default=0.05)
    parser.add_argument("--gamma1", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--eta_s", type=float, default=0.05)
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--purification_dim", type=int, default=2)
    parser.add_argument("--disentangle_every", type=int, default=20)
    args = parser.parse_args()

    # --- (mu1, mu2) translation (migration) ---
    if getattr(args, "mu1", None) is not None and args.mu is None:
        _N = args.N
        args.mu = _N * args.mu1 / 2.0
        args.nu = _N * (args.mu1 - args.mu2) / 2.0
        args.convention = "mu1mu2"
    elif args.mu is not None and getattr(args, "mu1", None) is None:
        args.convention = "internal"
    else:
        raise SystemExit("Provide either --mu --nu or --mu1 --mu2")
    args.mu_suffix = (f"_mu1{args.mu1:.4f}_mu2{args.mu2:.4f}"
                      if args.convention == "mu1mu2"
                      else f"_mu{args.mu:.4f}_nu{args.nu:.4f}")
    # --- End (mu1, mu2) translation ---
    print(f"Initializing Bañuls worker: N={args.N}, chi={args.chi_max}, "
          f"alpha={args.alpha}, mu={args.mu:.6f}, nu={args.nu:.6f}, "
          f"gamma1={args.gamma1:.4e}, gamma2={args.gamma2:.4e}, "
          f"eta_s={args.eta_s}, block_size={args.block_size}, "
          f"d_p={args.purification_dim}, "
          f"disentangle_every={args.disentangle_every}", flush=True)

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'banuls_pts')
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(
        DATA_DIR,
        f"banuls_N{args.N}_chi{args.chi_max}_alpha{args.alpha}"
        f"{args.mu_suffix}"
        f"_g1{args.gamma1:.4e}_g2{args.gamma2:.4e}"
        f"_eta{args.eta_s}.npy"
    )

    if os.path.exists(out_file):
        print(f"Already completed: {out_file}", flush=True)
        return

    model = PowerLawOATModel({
        'N': args.N,
        'J': args.J,
        'alpha': args.alpha,
        'conserve': None,
        'compression_method': 'exponential',
    })

    psi = initialize_css(model.lat)
    evolve_mps_with_banuls(psi, model, args.mu / 2.0,
                           args.chi_max, args.dt_max,
                           args.gamma1, args.gamma2,
                           args.eta_s, args.block_size,
                           args.purification_dim,
                           args.disentangle_every)

    nu_minus_mu = args.nu - args.mu
    M = get_M_mps(psi, model, args.chi_max, nu_minus_mu, args.dt_max,
                  args.gamma1, args.gamma2, args.eta_s,
                  args.block_size, args.purification_dim,
                  args.disentangle_every)

    psi_q = psi.copy()
    evolve_mps_with_banuls(psi_q, model, nu_minus_mu / 2.0,
                           args.chi_max, args.dt_max,
                           args.gamma1, args.gamma2,
                           args.eta_s, args.block_size,
                           args.purification_dim,
                           args.disentangle_every)
    Q = get_Q_mps(psi_q)

    evals, evecs = np.linalg.eigh(Q)
    mask = evals > 1e-12
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    Q_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T

    sv = scipy.linalg.svd(M @ Q_inv_sqrt, compute_uv=False)
    snr = float(np.max(sv))

    np.save(out_file, {
        'N': args.N,
        'chi_max': args.chi_max,
        'alpha': args.alpha,
        'J': args.J,
        'mu1': getattr(args, 'mu1', None),
        'mu2': getattr(args, 'mu2', None),
        'mu': args.mu,
        'nu': args.nu,
        'gamma1': args.gamma1,
        'gamma2': args.gamma2,
        'eta_s': args.eta_s,
        'snr': snr,
    })
    print(f"Bañuls SNR = {snr:.6f}", flush=True)
    print(f"Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()