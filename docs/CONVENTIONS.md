# Conventions

Two coordinate systems are used in the code to describe the symmetric
echo protocol. Both describe the same physical state; the choice of
convention is a matter of presentation.

## Legacy convention: $(\mu, \nu)$

- $\mu \in [0, \pi]$ — forward twist strength
- $\nu \in [-\pi, \pi]$ — backward twist strength
- CLI: `--mu VAL --nu VAL`

## Paper convention: $(\mu_1, \mu_2)$

- $\mu_1 \in [0, \pi]$ — forward twist strength
- $\mu_2 \in [0, 2\pi]$ — backward twist strength
- CLI: `--mu1 VAL --mu2 VAL`

## Mapping

The two are related by

$$
\mu_1 = \frac{2\mu}{N}, \qquad \mu_2 = \frac{2(\mu - \nu)}{N},
$$

and inversely by

$$
\mu = \frac{N \mu_1}{2}, \qquad \nu = \frac{N(\mu_1 - \mu_2)}{2},
$$

where $N$ is the total number of spins in the chain or on the cylinder
($N = L_x L_y$ for the 2D case).

## Protocol subclasses

Within the symmetric echo family, three distinguished lines in the
$(\mu, \nu)$ plane are highlighted:

| Subclass   | Legacy line | Paper line       |
|------------|-------------|------------------|
| OUT        | $\nu = -\mu$ | $\mu_2 = 2\mu_1$ |
| ExactEcho  | $\nu = 0$    | $\mu_2 = \mu_1$  |
| Squeezing  | $\nu = +\mu$ | $\mu_2 = 0$      |

## Other protocol families

- **Asymmetric echo.** The forward and backward durations
  $(t_+, t_-)$ are independent unsigned parameters. The symmetric-echo
  plane corresponds to $t_+ = t_-$, and the mapping is
  $\mu = 2 t_+$, $\nu = 2(t_+ - t_-)$. The CLI accepts
  `--t_plus VAL --t_minus VAL` or, equivalently,
  `--mu1 VAL --mu2 VAL`.
- **Three-axis extension.** The second twist axis $\vec{k}$ is free
  and is specified by its polar and azimuthal angles
  $(\theta_k, \phi_k)$. The CLI accepts `--theta_k VAL --phi_k VAL`
  in addition to any of the two-axis parameterisations above. When
  both angles vanish, the three-axis workers reduce to the
  corresponding symmetric-echo workers.

## Worked example

Take $N = 8$, the OUT protocol point
$(\mu, \nu) = (\pi/2, -\pi/2)$. Then

$$
\mu_1 = \frac{2 \cdot (\pi/2)}{8} = \frac{\pi}{8} \approx 0.3927,
\qquad
\mu_2 = \frac{2 \cdot (\pi/2 - (-\pi/2))}{8} = \frac{\pi}{4} \approx 0.7854.
$$

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

- `--mu --nu` → suffix `_mu{mu}_nu{nu}`
- `--mu1 --mu2` → suffix `_mu1{mu1}_mu2{mu2}`

Both metadata keys (`mu`, `nu`, `mu1`, `mu2`) are stored inside the
output `.npy` regardless of the CLI form used.
