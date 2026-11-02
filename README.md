# TenPyforQuMy

Tensor Networks for Quantum Metrology.

This repository contains Python code for simulating quantum metrology
protocols based on power-law Ising Hamiltonians. It covers four
combinations of dimension and physical regime:

- 1D chains, closed systems
- 1D chains, open systems with Lindblad dissipation
- 2D cylinders, closed systems
- 2D cylinders, open systems with Lindblad dissipation

For each combination, the code implements the full echo-protocol
family: symmetric echo (parameterised by forward and backward twist
strengths $(\mu, \nu)$), asymmetric echo (parameterised by
independent forward and backward durations $(t_+, t_-)$), and the
three-axis extension (the second twist axis $\vec{k}$ is free and
parameterised by its polar and azimuthal angles $(\theta_k, \phi_k)$).
Three protocol subclasses are highlighted within the symmetric family:
OUT (over-un-twisting), ExactEcho (exact un-twisting), and Squeezing
(no un-twisting).

The metrological figure of merit is the optimal signal-to-noise ratio
of a collective-spin measurement, computed from the signal-response
matrix $M$ and the collective-spin covariance matrix $Q$ as
$\sigma_{\max}(M Q^{-1/2})$. The quantum Fisher information of the
final state is also computed alongside the SNR, for both closed and
open systems.

Three simulation approaches are included:

- **Taylor-MPO** (all systems): MPS time evolution driven by
  `ExpMPOEvolution`, a second-order Taylor expansion of the
  time-evolution operator applied to the power-law Ising MPO. The
  same engine handles both closed-system (unitary) and open-system
  (effective non-Hermitian) evolution.
- **Tensor Jump Method** (open systems): Monte Carlo wave-function
  unravelling of the Lindblad equation. Uses the Taylor-MPO engine
  for the coherent step, exact single-site dissipative contraction,
  and stochastic quantum jumps.
- **Bañuls** (open systems, 1D only): deterministic locally purified
  MPO with exact dissipative contraction.

A naive vectorized-MPO worker is also included as a failure baseline.
It applies the Taylor-MPO engine to a vectorized density matrix; the
local SVD truncation of the resulting MPO does not preserve positivity
of the reconstructed density matrix, which is the failure mode the
MPS trajectory and Bañuls methods avoid by construction.

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

One point of the 1D closed-system echo landscape at $N = 8$,
$\alpha = 3$, $\chi = 32$, at the OUT protocol point
$(\mu_1, \mu_2) = (\pi/8, \pi/4)$:

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

- **Legacy $(\mu, \nu)$**: $\mu \in [0, \pi]$,
  $\nu \in [-\pi, \pi]$. CLI: `--mu VAL --nu VAL`.
- **Paper $(\mu_1, \mu_2)$**: $\mu_1 \in [0, \pi]$,
  $\mu_2 \in [0, 2\pi]$. CLI: `--mu1 VAL --mu2 VAL`.

Both describe the same physical state, related by

$$
\mu_1 = \frac{2\mu}{N}, \qquad \mu_2 = \frac{2(\mu - \nu)}{N}.
$$

The three protocol subclasses map to:

| Subclass   | Legacy line | Paper line       |
|------------|-------------|------------------|
| OUT        | $\nu = -\mu$ | $\mu_2 = 2\mu_1$ |
| ExactEcho  | $\nu = 0$    | $\mu_2 = \mu_1$  |
| Squeezing  | $\nu = +\mu$ | $\mu_2 = 0$      |

For asymmetric echo, the two parameters are unsigned durations
$(t_+, t_-)$, and the mapping to the symmetric-echo plane is
$\mu = 2 t_+$, $\nu = 2(t_+ - t_-)$. The CLI accepts
`--t_plus VAL --t_minus VAL` or `--mu1 VAL --mu2 VAL`. For the
three-axis extension, `--theta_k VAL --phi_k VAL` specify the second
twist axis on top of any two-axis parameterisation.

See `docs/CONVENTIONS.md` for worked examples.

## File map

| Case      | Approach       | Protocol                  | Worker |
|-----------|----------------|---------------------------|--------|
| 1D closed | Taylor-MPO     | symmetric echo            | `scripts/workers/closed_1d/powerlaw_echo_worker.py` |
| 1D closed | Taylor-MPO     | asymmetric echo           | `scripts/workers/closed_1d/asymmetric_echo_1d_closed.py` |
| 1D closed | Taylor-MPO     | three-axis echo           | `scripts/workers/closed_1d/scharnagl_echo_1d_closed.py` |
| 1D closed | QuTiP exact    | symmetric echo (ref)      | `scripts/workers/closed_1d/exact_echo_worker_1d.py` |
| 1D open   | TJM            | symmetric echo            | `scripts/workers/shared/tjm_worker.py --mode echo` |
| 1D open   | TJM            | asymmetric echo           | `scripts/workers/open_1d/asymmetric_echo_1d_open.py` |
| 1D open   | TJM            | three-axis echo           | `scripts/workers/open_1d/scharnagl_echo_1d_open.py` |
| 1D open   | QuTiP Lindblad | symmetric echo (ref)      | `scripts/workers/open_1d/exact_lindblad_worker_1d.py` |
| 1D open   | Bañuls         | symmetric echo            | `scripts/workers/open_1d/banuls_worker.py` |
| 1D open   | Simple MPO     | symmetric echo (baseline) | `scripts/workers/open_1d/simple_mpo_worker.py` |
| 2D closed | Taylor-MPO     | symmetric echo            | `scripts/workers/closed_2d/powerlaw_cylinder_worker.py` |
| 2D closed | Taylor-MPO     | asymmetric echo           | `scripts/workers/closed_2d/asymmetric_echo_2d_closed.py` |
| 2D closed | Taylor-MPO     | three-axis echo           | `scripts/workers/closed_2d/scharnagl_echo_2d_closed.py` |
| 2D closed | QuTiP exact    | symmetric echo (ref)      | `scripts/workers/closed_2d/exact_echo_worker_2d.py` |
| 2D open   | TJM            | symmetric echo            | `scripts/workers/shared/tjm_worker.py --mode echo --Lx --Ly` |
| 2D open   | TJM            | asymmetric echo           | `scripts/workers/open_2d/asymmetric_echo_2d_open.py` |
| 2D open   | TJM            | three-axis echo           | `scripts/workers/open_2d/scharnagl_echo_2d_open.py` |
| 2D open   | QuTiP Lindblad | symmetric echo (ref)      | `scripts/workers/open_2d/exact_lindblad_worker_2d.py` |

Forward sweeps (time series starting from the coherent spin state)
live in `scripts/workers/shared/main_powerlaw.py` (power-law) and
`scripts/workers/shared/main_fss.py` (all-to-all).

Full matrix and cross-references: `docs/FILE_MAP.md`.

## License

MIT. See `LICENSE`.

## Contact

Annwoy Roy Choudhury

- ms22095@iisermohali.ac.in
- annoyrc@gmail.com
