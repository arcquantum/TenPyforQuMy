# Workers

Each worker simulates one protocol at one point of the parameter grid
and writes a single `.npy` output. Workers are grouped by physical
case. Every worker uses the same Taylor-MPO (`ExpMPOEvolution`) engine
for the coherent part of the evolution.

- `closed_1d/` — 1D chain, closed system: symmetric echo, asymmetric
  echo, three-axis echo, and exact-diagonalization reference.
- `closed_2d/` — 2D cylinder, closed system: symmetric echo,
  asymmetric echo, three-axis echo, and exact-diagonalization
  reference.
- `open_1d/` — 1D chain, open system: TJM asymmetric echo, TJM
  three-axis echo, exact-Lindblad reference, Bañuls, and Simple MPO
  failure baseline.
- `open_2d/` — 2D cylinder, open system: TJM asymmetric echo, TJM
  three-axis echo, and exact-Lindblad reference.
- `shared/` — workers that handle more than one case: TJM symmetric
  echo (1D and 2D), forward sweeps (power-law and all-to-all), and the
  legacy OAT echo worker.

See `docs/FILE_MAP.md` for the full case × protocol → worker table.
