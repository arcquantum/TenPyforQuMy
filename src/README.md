# src

Core modules shared by every worker and every check in the
repository. None of these modules writes output on its own; they are
imported by the scripts under `scripts/workers/` and
`scripts/checks/`.

- `models.py` — Tensor-network models. Three classes: `OATModel`
  (collective all-to-all Ising), `PowerLawOATModel` (1D power-law
  Ising with exact or exponential-sum compression), and
  `PowerLawCylinderModel` (2D cylinder, block-site construction).
- `evolution.py` — State initialization and time evolution. Coherent
  spin state construction, collective and single-site rotations, the
  shared `ExpMPOEvolution` wrapper, and the rotation-sandwich helper
  that applies a twist about an arbitrary axis.
- `observables.py` — Collective-spin covariance, Wineland squeezing,
  quantum Fisher information (pure-state value and
  trajectory-ensemble bounds), and half-chain von Neumann entropy.
- `dM.py` — Analytical signal-response matrix and covariance matrix,
  evaluated in the MPS representation. Exact for any power-law
  exponent and for both closed and open systems.
- `dM_dicke.py` — Analytical signal-response and covariance matrices
  in the Dicke basis. Exact for the OAT model and for the all-to-all
  ($\alpha = 0$) power-law Ising model.
- `tails.py` — Exponential-sum fit of the $1/r^{\alpha}$ coupling
  tail. Produces the $(c_k, \lambda_k)$ terms consumed by
  `add_exponentially_decaying_coupling`.
- `tjm.py` — Tensor Jump Method primitives. Single-site dissipative
  contraction, coherent evolution step, stochastic jump process, and
  full-trajectory wrapper.
- `banuls.py` — Bañuls locally purified MPO primitives.
  Dissipative contraction and block-decomposition helpers.
- `simple_mpo.py` — Naive vectorized MPO construction and evolution.
  Included as a failure baseline; the local SVD truncation does not
  preserve positivity at finite bond dimension.
- `selectors.py` — Resource-selection logic. Worker command and
  output-path dispatch, pilot-grid construction, bond-dimension /
  grid-resolution / trajectory-count selectors, and coarse-grid
  verification.
- `checks_helpers_scharnagl.py` — Explicit reference state for the
  three-axis extension, computed directly from matrix exponentials
  on the full Hilbert space. Used by the `check_scharnagl.py`
  validation.
