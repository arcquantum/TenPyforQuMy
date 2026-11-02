# Conventions

Two coordinate systems are used in the code to describe the symmetric
echo protocol. Both describe the same physical state; the choice of
convention is a matter of presentation.

## Legacy convention: (mu, nu)

- `mu`  in `[0, pi]`   — forward twist strength
- `nu`  in `[-pi, pi]` — backward twist strength
- CLI: `--mu VAL --nu VAL`

## Paper convention: (mu1, mu2)

- `mu1` in `[0, pi]`   — forward twist strength
- `mu2` in `[0, 2 pi]` — backward twist strength
- CLI: `--mu1 VAL --mu2 VAL`

## Mapping

The two are related by

mu1 = 2 * mu / N
mu2 = 2 * (mu - nu) / N
text


and inversely by

mu = N * mu1 / 2
nu = N * (mu1 - mu2) / 2
text


where `N` is the total number of spins in the chain or on the cylinder
(`N = Lx * Ly` for the 2D case).

## Protocol families

| Family     | Legacy line | Paper line  |
|------------|-------------|-------------|
| OUT        | nu = -mu    | mu2 = 2 mu1 |
| ExactEcho  | nu = 0      | mu2 = mu1   |
| Squeezing  | nu = +mu    | mu2 = 0     |

## Worked example

Take `N = 8`, the OUT protocol point `(mu, nu) = (pi/2, -pi/2)`. Then

mu1 = 2 * (pi/2) / 8 = pi/8 ~= 0.3927
mu2 = 2 * (pi/2 - (-pi/2))/8 = 2*pi/8 ~= 0.7854
text


Both of the following commands run the same physical protocol and
produce the same SNR:

python3 scripts/workers/closed_1d/powerlaw_echo_worker.py
--N 8 --alpha 3.0 --chi_max 32 --mu 1.5708 --nu -1.5708

python3 scripts/workers/closed_1d/powerlaw_echo_worker.py
--N 8 --alpha 3.0 --chi_max 32 --mu1 0.3927 --mu2 0.7854
text


## Output filenames

Workers name their output files based on the convention used on the
command line:

- `--mu --nu`   -> suffix `_mu{mu}_nu{nu}`
- `--mu1 --mu2` -> suffix `_mu1{mu1}_mu2{mu2}`

Both metadata keys (`mu`, `nu`, `mu1`, `mu2`) are stored inside the
output `.npy` regardless of the CLI form used.
