# File Map

## Workers

| Case      | Protocol          | Engine   | Worker |
|-----------|-------------------|----------|--------|
| 1D closed | echo (symmetric)  | MPS      | `scripts/workers/closed_1d/powerlaw_echo_worker.py` |
| 1D closed | echo (asymmetric) | MPS      | `scripts/workers/closed_1d/asymmetric_echo_1d_closed.py` |
| 1D closed | echo (3-axis)     | MPS      | `scripts/workers/closed_1d/scharnagl_echo_1d_closed.py` |
| 1D closed | echo (reference)  | ED       | `scripts/workers/closed_1d/exact_echo_worker_1d.py` |
| 1D open   | echo (symmetric)  | TJM      | `scripts/workers/shared/tjm_worker.py --mode echo` |
| 1D open   | echo (asymmetric) | TJM      | `scripts/workers/open_1d/asymmetric_echo_1d_open.py` |
| 1D open   | echo (3-axis)     | TJM      | `scripts/workers/open_1d/scharnagl_echo_1d_open.py` |
| 1D open   | echo (reference)  | Lindblad | `scripts/workers/open_1d/exact_lindblad_worker_1d.py` |
| 1D open   | echo              | Bañuls   | `scripts/workers/open_1d/banuls_worker.py` |
| 1D open   | echo (baseline)   | Simple   | `scripts/workers/open_1d/simple_mpo_worker.py` |
| 2D closed | echo (symmetric)  | MPS      | `scripts/workers/closed_2d/powerlaw_cylinder_worker.py` |
| 2D closed | echo (asymmetric) | MPS      | `scripts/workers/closed_2d/asymmetric_echo_2d_closed.py` |
| 2D closed | echo (3-axis)     | MPS      | `scripts/workers/closed_2d/scharnagl_echo_2d_closed.py` |
| 2D closed | echo (reference)  | ED       | `scripts/workers/closed_2d/exact_echo_worker_2d.py` |
| 2D open   | echo (symmetric)  | TJM      | `scripts/workers/shared/tjm_worker.py --mode echo --Lx --Ly` |
| 2D open   | echo (asymmetric) | TJM      | `scripts/workers/open_2d/asymmetric_echo_2d_open.py` |
| 2D open   | echo (3-axis)     | TJM      | `scripts/workers/open_2d/scharnagl_echo_2d_open.py` |
| 2D open   | echo (reference)  | Lindblad | `scripts/workers/open_2d/exact_lindblad_worker_2d.py` |

Forward sweeps (time series starting from the coherent spin state)
live in:

- `scripts/workers/shared/main_powerlaw.py` (1D and 2D, power-law)
- `scripts/workers/shared/main_fss.py` (1D all-to-all)
- `scripts/workers/shared/tjm_worker.py --mode forward` (open systems)

## Engines at a glance

- **MPS**: closed-system time evolution with `ExpMPOEvolution` on the
  power-law Ising MPO. Long-range couplings are compressed by an
  exponential-sum fit (`src/tails.py`) so the MPO bond dimension is
  O(1) in `N`.
- **TJM**: Monte Carlo wave-function unravelling of the Lindblad
  equation. Strang splitting of coherent and dissipative parts, exact
  single-site dissipative contraction, stochastic quantum jumps
  applied per step (`src/tjm.py`).
- **Bañuls**: deterministic locally purified MPO for 1D open systems
  (`src/banuls.py`).
- **Simple MPO**: naive vectorised density matrix with MPO truncation
  (`src/simple_mpo.py`). Included as a failure baseline; it does not
  preserve positivity at finite bond dimension.
- **ED**: exact diagonalization (closed) and exact Lindblad solver
  (open), both via QuTiP, for validation at small `N`.

## Checks

`scripts/checks/` contains the validation suite. Run the full set:

python3 scripts/checks/run_all_checks.py
text


Individual highlights:

- `check_models.py` — MPO vs analytic Hamiltonian on the full basis.
- `check_evolution.py` — norm preservation, time reversal, small-t.
- `check_observables.py` — MPS observables vs exact diagonalization.
- `check_qutip_closed.py` / `check_qutip_open.py` — exact-M vs QuTiP.
- `check_asymmetric_echo.py` — asymmetric vs symmetric at exact echo.
- `check_scharnagl.py` — three-axis extension vs two-axis and QuTiP.
- `check_tjm.py` — TJM vs closed and vs Lindblad, plus positivity.

## Maintenance

`scripts/maintenance/` contains SLURM drivers, resource-table
generators, and results-bundling utilities. Scripts whose name ends
in `.sh` are shell wrappers around the Python drivers. Drivers that
write into `scripts/maintenance/_*_jobs/` are SLURM-specific; their
output is git-ignored. These scripts assume a SLURM cluster; the
underlying Python workers do not.

## Notebooks

`notebooks/` contains one script per figure. All read from
`data/bundles/` (produced by `scripts/maintenance/bundle_results.py`)
and write a single PDF to `plots/`.
