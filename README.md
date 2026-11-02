# TenPyforQuMy

Tensor Networks for Quantum Metrology.

This repository contains Python code for simulating quantum metrology
protocols based on power-law Ising Hamiltonians. It covers four
combinations of dimension and physical regime:

- 1D chains, closed systems
- 1D chains, open systems with Lindblad dissipation
- 2D cylinders, closed systems
- 2D cylinders, open systems with Lindblad dissipation

For each combination, three echo protocols are simulated: OUT
(over-un-twisting), ExactEcho (exact un-twisting), and Squeezing
(no un-twisting). The figure of merit is the optimal signal-to-noise
ratio of a collective-spin measurement, computed from the signal
response matrix and the collective-spin covariance matrix.

Three simulation engines are included:

- **MPS-TEBD** (closed systems): Matrix Product State time evolution
  with the power-law Ising Hamiltonian.
- **TJM** (open systems): Tensor Jump Method, a Monte Carlo wave-
  function unravelling of the Lindblad equation.
- **Bañuls** (open systems, 1D only): locally purified MPO with exact
  dissipative contraction.

A naive vectorized-MPO worker is also included as a failure baseline.
It demonstrates the positivity violation that the MPS-TEBD and TJM
methods avoid by construction.

Exact QuTiP reference workers (closed and Lindblad) are included for
validation at small system sizes.

## Status

Snapshot as of 2 November 2026. The repository will be updated from
time to time.

## Install

conda create -n tpqm python=3.10
conda activate tpqm
pip install -r requirements.txt
text


## Quickstart

One point of the 1D closed-system echo landscape at N=8, alpha=3,
chi=32, at the OUT protocol point in paper convention
(mu1, mu2) = (pi/8, pi/4):

python3 scripts/workers/closed_1d/powerlaw_echo_worker.py
--N 8 --alpha 3.0 --chi_max 32
--mu1 0.3927 --mu2 0.7854
text


The worker writes a single `.npy` file to `data/powerlaw_echo_pts/`
and prints the SNR to stdout.

Check that the MPO matches its analytic Hamiltonian:

python3 scripts/checks/check_models.py
text


## Conventions

Protocol parameters can be expressed in two coordinate systems:

- **Legacy (mu, nu)**: mu in [0, pi], nu in [-pi, pi].
  CLI: `--mu VAL --nu VAL`.
- **Paper (mu1, mu2)**: mu1 in [0, pi], mu2 in [0, 2 pi].
  CLI: `--mu1 VAL --mu2 VAL`.

Both describe the same physical state, related by

mu1 = 2 * mu / N
mu2 = 2 * (mu - nu) / N
text


The three protocol families map to:

| Family     | Legacy line | Paper line  |
|------------|-------------|-------------|
| OUT        | nu = -mu    | mu2 = 2 mu1 |
| ExactEcho  | nu = 0      | mu2 = mu1   |
| Squeezing  | nu = +mu    | mu2 = 0     |

See `docs/CONVENTIONS.md` for worked examples.

## File map

| Case      | Engine   | Protocol          | Worker |
|-----------|----------|-------------------|--------|
| 1D closed | MPS      | echo (symmetric)  | `scripts/workers/closed_1d/powerlaw_echo_worker.py` |
| 1D closed | MPS      | echo (asymmetric) | `scripts/workers/closed_1d/asymmetric_echo_1d_closed.py` |
| 1D closed | MPS      | echo (3-axis)     | `scripts/workers/closed_1d/scharnagl_echo_1d_closed.py` |
| 1D closed | ED       | echo (reference)  | `scripts/workers/closed_1d/exact_echo_worker_1d.py` |
| 1D open   | TJM      | echo (symmetric)  | `scripts/workers/shared/tjm_worker.py --mode echo` |
| 1D open   | TJM      | echo (asymmetric) | `scripts/workers/open_1d/asymmetric_echo_1d_open.py` |
| 1D open   | TJM      | echo (3-axis)     | `scripts/workers/open_1d/scharnagl_echo_1d_open.py` |
| 1D open   | Lindblad | echo (reference)  | `scripts/workers/open_1d/exact_lindblad_worker_1d.py` |
| 1D open   | Bañuls   | echo              | `scripts/workers/open_1d/banuls_worker.py` |
| 1D open   | Simple   | echo (baseline)   | `scripts/workers/open_1d/simple_mpo_worker.py` |
| 2D closed | MPS      | echo (symmetric)  | `scripts/workers/closed_2d/powerlaw_cylinder_worker.py` |
| 2D closed | MPS      | echo (asymmetric) | `scripts/workers/closed_2d/asymmetric_echo_2d_closed.py` |
| 2D closed | MPS      | echo (3-axis)     | `scripts/workers/closed_2d/scharnagl_echo_2d_closed.py` |
| 2D closed | ED       | echo (reference)  | `scripts/workers/closed_2d/exact_echo_worker_2d.py` |
| 2D open   | TJM      | echo (symmetric)  | `scripts/workers/shared/tjm_worker.py --mode echo --Lx --Ly` |
| 2D open   | TJM      | echo (asymmetric) | `scripts/workers/open_2d/asymmetric_echo_2d_open.py` |
| 2D open   | TJM      | echo (3-axis)     | `scripts/workers/open_2d/scharnagl_echo_2d_open.py` |
| 2D open   | Lindblad | echo (reference)  | `scripts/workers/open_2d/exact_lindblad_worker_2d.py` |

Forward sweeps (time series from the coherent spin state) live in
`scripts/workers/shared/main_powerlaw.py` (power-law) and
`scripts/workers/shared/main_fss.py` (all-to-all).

Full matrix and cross-references: `docs/FILE_MAP.md`.

## License

MIT. See `LICENSE`.

## Contact

Annwoy Roy Choudhury

- ms22095@iisermohali.ac.in
- annoyrc@gmail.com
