# Workers

Each worker simulates one protocol at one point of the parameter grid
and writes a single `.npy` output. Workers are grouped by physical
case and engine.

- `closed_1d/` — 1D chain, closed system, MPS + exact-diagonalization
- `closed_2d/` — 2D cylinder, closed system, MPS + exact-diagonalization
- `open_1d/` — 1D chain, open system, TJM + exact Lindblad + Bañuls + Simple MPO
- `open_2d/` — 2D cylinder, open system, TJM + exact Lindblad
- `shared/` — workers that handle more than one case (TJM, forward
  sweeps, legacy echo)

See `docs/FILE_MAP.md` for the full case x protocol -> worker table.
