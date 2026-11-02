# Notebooks

Plotting scripts. Each reads from `data/bundles/` (produced by
`scripts/maintenance/bundle_results.py`) and writes a single PDF
to `plots/`.

- `_plot_helpers.py` — shared utilities (path, style, bundle IO)
- `plot_master_landscape_grid.py` — full landscape grid, one panel per tuple
- `plot_frontier_alpha.py` — max-over-nu SNR vs mu, one line per alpha
- `plot_frontier_N.py` — protocol x alpha grid, one line per N
- `plot_phase_diagram.py` — dominant protocol vs (mu, alpha)
- `plot_decoherence_smallmultiples.py` — one panel per gamma
- `plot_chiautomation.py` — effective chi_prod heatmap
- `plot_1d_vs_2d.py` — 1D vs 2D comparison
- `plot_validation_errors.py` — validation error heatmap
- `plot_method_comparison.py` — Simple MPO / Bañuls / TJM on same axes
- `plot_lookup_table.py` — experimental lookup table PDF
- `plot_asymmetric_echo.py` — small multiples over (t_plus, t_minus)
- `plot_forward_spectroscopy.py` — 1D alpha-spectroscopy
- `plot_forward_spectroscopy_2d.py` — 2D alpha-spectroscopy
- `plot_summary_frontier.py` — single-panel summary frontier
- `plot_dephasing_scaling.py` — dephasing-scaling summary
- `plot_qfi_frontier.py` — SNR vs QFI frontier
- `plot_qfi_frontier_asym.py` — asymmetric SNR vs QFI frontier
- `plot_qfi_scaling.py` — QFI/N^2 vs N
- `plot_echo_landscape_single.py` — one panel per tuple
- `plot_baseline.py`, `plot_floors.py`, `plot_scaling_mps.py` — OAT baseline helpers
