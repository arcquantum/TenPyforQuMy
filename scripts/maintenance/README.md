# Maintenance

SLURM submission, resource-table generation, and results bundling.
The Python workers under `scripts/workers/` do not require a SLURM
cluster; the drivers here do.

## Submission drivers

- `submit_production.py` — main production driver. Enumerates the
  tuples for each phase, looks up per-tuple resources, chunks SLURM
  arrays, submits.
- `submit_asymmetric.py` — asymmetric-echo production driver.
- `submit_scharnagl_stage1.py` — Scharnagl three-axis production
  driver.
- `submit_forward_gaps.py` — resubmission helper.
- `submit_powerlaw_echo.py` — ad-hoc echo submission.
- `run_asym_batch.py` — single-node batch driver for asymmetric echo.

## Resource-table generators

- `auto_chi_pilot.py` — bond-dimension selector.
- `auto_grid_pilot.py` — grid-resolution selector.
- `auto_ntraj_pilot.py` — trajectory-count selector.
- `assemble_1d_closed_chi_conv.py` — assemble per-tuple chi tables.
- `check_qfi_scaling_1d.py` — QFI scaling sweep (offline, no SLURM).

## Results

- `bundle_results.py` — aggregate per-point `.npy` files into
  per-tuple bundles under `data/bundles/`.

## Resubmission

- `auto_resubmit.py` — polls `sacct`, escalates memory on OOM.
- `resubmit_failed_jobs.py` — manual resubmission helper.

## Shell wrappers

`*.sh` — thin wrappers around the Python drivers, ready for `sbatch`.

## Site-specific notes

These scripts assume the SLURM environment at the authors' home
institution. For other clusters, edit the `#SBATCH` headers and the
conda activation line at the top of each `.sh` wrapper. The Python
workers themselves are portable.
