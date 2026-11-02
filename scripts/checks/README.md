# Checks

Validation suite. Run all of them:

```
python3 scripts/checks/run_all_checks.py
```


Individual scripts:

- `check_models.py` — MPO vs analytic Hamiltonian on the full basis
- `check_evolution.py` — evolution engine correctness
- `check_observables.py` — MPS observables vs exact diagonalization
- `check_tails.py` — exponential-sum compression of the power-law tail
- `check_exact_M.py` — exact M vs finite differences
- `check_qutip_closed.py` / `check_qutip_closed_2d.py` — closed-system exact-M vs QuTiP
- `check_qutip_open.py` / `check_qutip_open_2d.py` — Lindblad exact-M vs QuTiP
- `check_asymmetric_echo.py` — asymmetric vs symmetric echo workers
- `check_scharnagl.py` — three-axis extension vs two-axis and QuTiP
- `check_tjm.py` — Tensor Jump Method vs closed and Lindblad
- `check_banuls.py` — Bañuls worker closed-system limit
- `check_banuls_open.py` — Bañuls open-system diagnostic
- `check_simple_mpo.py` — failure-baseline positivity violation
- `check_qfi_all_workers.py` — QFI value at the coherent-state limit
- `check_qfi_mixed_bounds.py` — trajectory-averaged QFI bounds
- `check_selector_chi.py`, `check_selector_grid.py`, `check_selector_ntraj.py` — resource-table validators
- `validate_1d_closed.py` — TN vs exact, full chi-scan
- `preflight.py` — ten-point manual inspection before production
- `test_production_ranges.py` — structural test of the pipeline
- `test_stage2b_dispatch.py` — dispatch regression
- `diag_exact_M.py` — single-point diagnostic
