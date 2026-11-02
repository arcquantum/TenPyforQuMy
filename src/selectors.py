"""
src/selectors.py

Selectors for the automatic pilot scans of the documentation (bond dimension, v2),
the documentation (grid resolution), and the documentation (trajectory count).

V2 change: the pilot grid is no longer the fixed 5-point list. It is the
union of the original PILOT_GRID and the (up to) three highest-entanglement
grid points identified from a coarse scan of the (mu, nu) landscape at a
low bond dimension. The union grid is stored per tuple in
data/resource_tables/pilot_grids/.

Provides:
  - PILOT_GRID                     : fallback 5-point grid
  - CHI_SCHEDULE                   : {8, 16, 32, 64, 128, 256}
  - TARGET_CLOSED, TARGET_OPEN     : convergence thresholds
  - classify_case                  : case identifier from geometry/noise
  - worker_command                 : CLI argv for the appropriate worker
  - native_output_path             : the path the worker writes to
  - snr_from_output                : read SNR from an output file
  - s_intermediate_from_output     : read S_intermediate from an output file
  - pilot_grid_filename            : canonical name of the per-tuple grid file
  - read_pilot_grid, write_pilot_grid
  - identify_worst_entanglement_point
  - build_pilot_grid
  - snr_by_chi                     : read SNR over the union grid
  - find_chi_convergence           : unchanged from v1
  - chi_production_from_conv       : unchanged from v1
  - verify_chi_on_coarse_grid      : re-run at chi_conv and 2*chi_conv on the
                                     full coarse grid, return max SNR diff
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np


# ----------------------------------------------------------------------
# Pilot configuration
# ----------------------------------------------------------------------
PILOT_GRID = [
    (np.pi / 2,  np.pi / 2),
    (np.pi / 2,  0.0),
    (np.pi / 2, -np.pi / 2),
    (np.pi / 4,  np.pi / 4),
    (np.pi / 2, -np.pi),
]

# Pilot grid in the (mu1, mu2) reference convention:
#   mu1 = 2 mu / N          (forward twist, in [0, pi])
#   mu2 = 2 (mu - nu) / N   (backward twist, in [0, 2 pi])
# Mapping to internal (mu, nu):
#   mu = N mu1 / 2
#   nu = N (mu1 - mu2) / 2
#
# The 5 points below span the three protocol families (exact echo,
# squeezing, OUT) at the reference peak mu1 = pi/2, plus one mid-range
# point and the upper edge of the mu2 range.
PILOT_GRID_MU1MU2 = [
    (np.pi / 2, np.pi / 2),      # exact echo (nu = 0) at peak
    (np.pi / 2, 0.0),            # squeezing  (nu = +mu) at peak
    (np.pi / 2, np.pi),          # OUT        (nu = -mu) at peak
    (np.pi / 4, np.pi / 4),      # exact echo at mu1 = pi/4
    (np.pi / 2, 2 * np.pi),      # upper edge of mu2 range
]

CHI_SCHEDULE = [8, 16, 32, 64, 128, 256]

TARGET_CLOSED = 1e-3
TARGET_OPEN   = 5e-2

N_TRAJ_PILOT = 100
SEED_PILOT   = 20261024

# Low-bond-dimension pilot used for the entanglement scan. Chosen small
# so that the coarse 20x20 landscape completes in a few hours per tuple.
# The scan does not need to resolve the state accurately; it only needs
# to sort the landscape by entanglement.
ENTANGLEMENT_SCAN_CHI   = 8
ENTANGLEMENT_SCAN_GRID  = 8
ENTANGLEMENT_SCAN_TOP_K = 3
UNION_GRID_CAP          = 12

# Trajectory count for the coarse scan. Smaller than N_TRAJ_PILOT because
# the scan only needs a ranking of the grid points by entropy, not
# high-precision SNR. 20 trajectories gives a ~22% standard error on
# S_intermediate, which is more than enough to identify the top-k.
N_TRAJ_SCAN = 5

# Tolerance for verify_chi_on_coarse_grid.
COARSE_VERIFY_TOL = 1e-3


# ----------------------------------------------------------------------
# Case classification
# ----------------------------------------------------------------------
def classify_case(N=None, Lx=None, Ly=None, gamma1=0.0, gamma2=0.0):
    is_2d   = Lx is not None or Ly is not None
    is_open = (gamma1 > 0.0) or (gamma2 > 0.0)
    if is_2d:
        return '2d_open' if is_open else '2d_closed'
    return '1d_open' if is_open else '1d_closed'


# ----------------------------------------------------------------------
# Worker command construction
# ----------------------------------------------------------------------
def worker_command(case, chi_max, N=None, Lx=None, Ly=None, alpha=1.0,
                   mu=None, nu=None, gamma1=0.0, gamma2=0.0,
                   J=1.0, dt_max=0.05, n_traj=N_TRAJ_PILOT,
                   seed=SEED_PILOT, python_exe=None, root=None,
                   convention="mu_nu", mu1=None, mu2=None):
    """Build the worker CLI argv.

    convention="mu_nu" (default): pass mu=, nu=; emits --mu --nu
    convention="mu1mu2":          pass mu1=, mu2=; emits --mu1 --mu2

    The two argument sets are mutually exclusive: passing the wrong
    one for the chosen convention raises ValueError.
    """
    if python_exe is None:
        python_exe = sys.executable
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if convention == "mu1mu2":
        if mu1 is None or mu2 is None:
            raise ValueError(
                f"convention='mu1mu2' requires mu1 and mu2; "
                f"got mu1={mu1!r}, mu2={mu2!r}"
            )
        twist_args = ["--mu1", f"{mu1:.6f}", "--mu2", f"{mu2:.6f}"]
    elif convention == "mu_nu":
        if mu is None or nu is None:
            raise ValueError(
                f"convention='mu_nu' requires mu and nu; "
                f"got mu={mu!r}, nu={nu!r}"
            )
        twist_args = ["--mu", f"{mu:.6f}", "--nu", f"{nu:.6f}"]
    else:
        raise ValueError(f"Unknown convention: {convention!r}")

    if case == '1d_closed':
        worker = os.path.join(root, 'scripts', 'workers', 'closed_1d', 'powerlaw_echo_worker.py')
        return [python_exe, worker,
                '--N', str(N), '--chi_max', str(chi_max),
                '--alpha', str(alpha), '--J', str(J),
                *twist_args,
                '--dt_max', str(dt_max)]
    if case == '2d_closed':
        worker = os.path.join(root, 'scripts', 'workers', 'closed_2d', 'powerlaw_cylinder_worker.py')
        return [python_exe, worker,
                '--Lx', str(Lx), '--Ly', str(Ly),
                '--chi_max', str(chi_max),
                '--alpha', str(alpha), '--J', str(J),
                *twist_args,
                '--dt_max', str(dt_max)]
    if case == '1d_open':
        worker = os.path.join(root, 'scripts', 'workers', 'shared', 'tjm_worker.py')
        return [python_exe, worker,
                '--mode', 'echo',
                '--N', str(N), '--chi_max', str(chi_max),
                '--alpha', str(alpha), '--J', str(J),
                *twist_args,
                '--gamma1', str(gamma1), '--gamma2', str(gamma2),
                '--n_traj_per_rank', str(n_traj),
                '--seed', str(seed),
                '--dt_max', str(dt_max)]
    if case == '2d_open':
        worker = os.path.join(root, 'scripts', 'workers', 'shared', 'tjm_worker.py')
        return [python_exe, worker,
                '--mode', 'echo',
                '--Lx', str(Lx), '--Ly', str(Ly),
                '--chi_max', str(chi_max),
                '--alpha', str(alpha), '--J', str(J),
                *twist_args,
                '--gamma1', str(gamma1), '--gamma2', str(gamma2),
                '--n_traj_per_rank', str(n_traj),
                '--seed', str(seed),
                '--dt_max', str(dt_max)]
    raise ValueError(f"Unknown case: {case!r}")


# ----------------------------------------------------------------------
# Native output path of each worker
# ----------------------------------------------------------------------
def native_output_path(case, chi_max, N=None, Lx=None, Ly=None,
                       alpha=1.0, mu=None, nu=None,
                       gamma1=0.0, gamma2=0.0,
                       n_traj=N_TRAJ_PILOT, root=None,
                       convention="mu_nu", mu1=None, mu2=None):
    """Return the .npy path the worker writes to.

    convention="mu_nu" (default): filename suffix _mu{mu}_nu{nu}
    convention="mu1mu2":          filename suffix _mu1{mu1}_mu2{mu2}
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if convention == "mu1mu2":
        if mu1 is None or mu2 is None:
            raise ValueError(
                f"convention='mu1mu2' requires mu1 and mu2; "
                f"got mu1={mu1!r}, mu2={mu2!r}"
            )
        twist_suffix = f"_mu1{mu1:.4f}_mu2{mu2:.4f}"
    elif convention == "mu_nu":
        if mu is None or nu is None:
            raise ValueError(
                f"convention='mu_nu' requires mu and nu; "
                f"got mu={mu!r}, nu={nu!r}"
            )
        twist_suffix = f"_mu{mu:.4f}_nu{nu:.4f}"
    else:
        raise ValueError(f"Unknown convention: {convention!r}")

    if case == '1d_closed':
        return os.path.join(
            root, 'data', 'powerlaw_echo_pts',
            f"plaw_echo_N{N}_chi{chi_max}_alpha{alpha}"
            f"{twist_suffix}.npy"
        )
    if case == '2d_closed':
        return os.path.join(
            root, 'data', 'powerlaw_cylinder_pts',
            f"plaw_cyl_Lx{Lx}_Ly{Ly}"
            f"_chi{chi_max}_alpha{alpha}"
            f"{twist_suffix}.npy"
        )
    if case == '1d_open':
        return os.path.join(
            root, 'data', 'tjm_pts',
            f"tjm_N{N}_chi{chi_max}_alpha{alpha}"
            f"{twist_suffix}"
            f"_g1{gamma1:.4e}_g2{gamma2:.4e}"
            f"_ntraj{n_traj}.npy"
        )
    if case == '2d_open':
        return os.path.join(
            root, 'data', 'tjm_pts',
            f"tjm_Lx{Lx}_Ly{Ly}_chi{chi_max}_alpha{alpha}"
            f"{twist_suffix}"
            f"_g1{gamma1:.4e}_g2{gamma2:.4e}"
            f"_ntraj{n_traj}.npy"
        )
    raise ValueError(f"Unknown case: {case!r}")


# ----------------------------------------------------------------------
# Read helpers
# ----------------------------------------------------------------------
def snr_from_output(path):
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True).item()
    except Exception:
        return None
    val = d.get('snr', None)
    if val is None:
        return None
    val = float(val)
    if not np.isfinite(val):
        return None
    return val


def s_intermediate_from_output(path):
    """
    Read S_intermediate from a worker output file. Returns None if the
    file does not exist, is unreadable, or does not contain the key.
    """
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True).item()
    except Exception:
        return None
    val = d.get('S_intermediate', None)
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


# ----------------------------------------------------------------------
# Per-tuple union pilot grid file
# ----------------------------------------------------------------------
def pilot_grid_filename(case, N=None, Lx=None, Ly=None,
                        alpha=1.0, gamma1=0.0, gamma2=0.0,
                        root=None, convention="mu_nu"):
    """
    Canonical filename for the per-tuple union pilot grid.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    grid_dir = os.path.join(root, 'data', 'resource_tables', 'pilot_grids')
    os.makedirs(grid_dir, exist_ok=True)

    if case == '1d_closed':
        name = f"{case}_N{N}_alpha{alpha}.json"
    elif case == '1d_open':
        name = (f"{case}_N{N}_alpha{alpha}"
                f"_g1{gamma1:.4f}_g2{gamma2:.4f}.json")
    elif case == '2d_closed':
        name = f"{case}_Lx{Lx}_Ly{Ly}_alpha{alpha}.json"
    elif case == '2d_open':
        name = (f"{case}_Lx{Lx}_Ly{Ly}_alpha{alpha}"
                f"_g1{gamma1:.4f}_g2{gamma2:.4f}.json")
    else:
        raise ValueError(f"Unknown case: {case!r}")
    if convention == "mu1mu2":
        name = name.replace(".json", "_mu1mu2.json")
    elif convention != "mu_nu":
        raise ValueError(f"Unknown convention: {convention!r}")
    return os.path.join(grid_dir, name)


def read_pilot_grid(case, N=None, Lx=None, Ly=None,
                    alpha=1.0, gamma1=0.0, gamma2=0.0, root=None,
                    convention="mu_nu"):
    """
    Read the union pilot grid from disk, or return PILOT_GRID
    (or PILOT_GRID_MU1MU2 for convention="mu1mu2") if the file
    does not exist. Handles both the old bare-list schema and the
    new dict schema {"convention": ..., "points": ...}.
    """
    path = pilot_grid_filename(case, N=N, Lx=Lx, Ly=Ly,
                               alpha=alpha, gamma1=gamma1, gamma2=gamma2,
                               root=root, convention=convention)
    fallback = (list(PILOT_GRID_MU1MU2) if convention == "mu1mu2"
                else list(PILOT_GRID))
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and "points" in data:
            points = data["points"]
        else:
            points = data
        return [(float(a), float(b)) for a, b in points]
    except Exception:
        return fallback


def write_pilot_grid(case, grid, N=None, Lx=None, Ly=None,
                     alpha=1.0, gamma1=0.0, gamma2=0.0, root=None,
                     convention="mu_nu"):
    """
    Write the union pilot grid to disk.
    """
    path = pilot_grid_filename(case, N=N, Lx=Lx, Ly=Ly,
                               alpha=alpha, gamma1=gamma1, gamma2=gamma2,
                               root=root, convention=convention)
    payload = {
        "convention": convention,
        "points": [[float(a), float(b)] for a, b in grid],
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path

def scan_intermediate_entropy(case, params,
                              low_chi=ENTANGLEMENT_SCAN_CHI,
                              coarse_grid=ENTANGLEMENT_SCAN_GRID,
                              n_traj_scan=N_TRAJ_SCAN,
                              dt_max=0.05,
                              seed=SEED_PILOT,
                              root=None,
                              convention="mu_nu"):
    """Forward-twist-only entropy scan on a coarse grid.

    In mu_nu convention, the grid is (mu, nu) over [0,pi] x [-pi,pi]
    and forward twist time = mu/2.
    In mu1mu2 convention, the grid is (mu1, mu2) over [0,pi] x [0,2pi]
    and forward twist time = N * mu1 / 4  (since mu = N*mu1/2 and
    forward = mu/2).

    Returns a dict keyed by (round(a,4), round(b,4)) where (a, b)
    are the two coordinates of the chosen convention.
    """
    import importlib
    import sys as _sys

    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root not in _sys.path:
        _sys.path.insert(0, root)
    for _subdir in ('shared', 'closed_1d', 'closed_2d', 'open_1d', 'open_2d'):
        _p = os.path.join(root, 'scripts', 'workers', _subdir)
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    from src.models import PowerLawOATModel, PowerLawCylinderModel
    from src.evolution import initialize_css

    gamma1 = params.get('gamma1', 0.0)
    gamma2 = params.get('gamma2', 0.0)
    J      = params.get('J', 1.0)
    alpha  = params['alpha']
    is_open = (gamma1 > 0.0) or (gamma2 > 0.0)
    is_2d   = params.get('Lx') is not None

    # Total physical spin count
    if is_2d:
        N_eff = int(params['Lx']) * int(params['Ly'])
    else:
        N_eff = int(params['N'])

    if is_2d:
        model = PowerLawCylinderModel({
            'Lx': params['Lx'], 'Ly': params['Ly'],
            'J': J, 'alpha': alpha, 'conserve': None,
        })
    else:
        model = PowerLawOATModel({
            'N': params['N'], 'J': J, 'alpha': alpha, 'conserve': None,
            'compression_method': 'exponential',
        })

    if is_open:
        tjm = importlib.import_module('tjm_worker')
        def forward_twist(psi, fwd_time, rng):
            tjm.evolve_trajectory(psi, model, fwd_time,
                                  gamma1, gamma2, dt_max,
                                  low_chi, rng)
    elif is_2d:
        cyl = importlib.import_module('powerlaw_cylinder_worker')
        def forward_twist(psi, fwd_time, rng):
            cyl.evolve_mps_time(psi, model, fwd_time, low_chi, dt_max)
    else:
        echo = importlib.import_module('powerlaw_echo_worker')
        def forward_twist(psi, fwd_time, rng):
            echo.evolve_mps_time(psi, model, fwd_time, low_chi, dt_max)

    # Grid axes and forward-time formula depend on convention
    if convention == "mu1mu2":
        a_vals = np.linspace(0.0, np.pi, coarse_grid)          # mu1
        b_vals = np.linspace(0.0, 2 * np.pi, coarse_grid)      # mu2
        def fwd_time(a):
            return N_eff * a / 4.0     # mu = N*a/2, fwd = mu/2
    elif convention == "mu_nu":
        a_vals = np.linspace(0.0, np.pi, coarse_grid)          # mu
        b_vals = np.linspace(-np.pi, np.pi, coarse_grid)       # nu
        def fwd_time(a):
            return a / 2.0
    else:
        raise ValueError(f"Unknown convention: {convention!r}")

    rng_master = np.random.default_rng(seed)

    table = {}
    n_a = len(a_vals)
    for i_a, a in enumerate(a_vals):
        print(f"[scan] row {i_a + 1}/{n_a} "
              f"({'mu1' if convention == 'mu1mu2' else 'mu'} = {a:.4f})",
              flush=True)
        fwd = fwd_time(a)
        if is_open:
            S_acc = 0.0
            for _ in range(max(n_traj_scan, 1)):
                rng = np.random.default_rng(
                    rng_master.integers(0, 2**32 - 1)
                )
                psi = initialize_css(model.lat)
                forward_twist(psi, fwd, rng)
                S_acc += float(max(psi.entanglement_entropy()))
            S_a = S_acc / max(n_traj_scan, 1)
        else:
            psi = initialize_css(model.lat)
            forward_twist(psi, fwd, rng_master)
            S_a = float(max(psi.entanglement_entropy()))

        for b in b_vals:
            table[(round(float(a), 4), round(float(b), 4))] = S_a

    return table


# ----------------------------------------------------------------------
# Entanglement-guided union grid construction
# ----------------------------------------------------------------------
def identify_worst_entanglement_point(case, params,
                                      low_chi=ENTANGLEMENT_SCAN_CHI,
                                      coarse_grid=ENTANGLEMENT_SCAN_GRID,
                                      top_k=ENTANGLEMENT_SCAN_TOP_K,
                                      n_traj_scan=N_TRAJ_SCAN,
                                      root=None,
                                      runner=subprocess.run,
                                      timeout_per_point=600,
                                      convention="mu_nu"):
    """
    Run the appropriate worker on a coarse_grid x coarse_grid landscape
    at bond dimension low_chi, and identify the top_k grid points with
    the largest half-chain entropy of the intermediate post-twist state.

    Parameters
    ----------
    case : str
        One of '1d_closed', '2d_closed', '1d_open', '2d_open'.
    params : dict
        Case parameters: N (1D), Lx, Ly (2D), alpha, gamma1, gamma2.
    low_chi : int
        Bond dimension for the coarse scan.
    coarse_grid : int
        Grid resolution.
    top_k : int
        Number of top-entanglement points to return.
    n_traj_scan : int
        Trajectory count for the scan (open systems only).
    root : str or None
        Repository root. Defaults to the parent of src/.
    runner : callable
        subprocess.run-compatible callable, for testing.
    timeout_per_point : float
        Wall-time limit for a single worker invocation.

    Returns
    -------
    top_points : list of (mu, nu)
        The top_k points with largest entropy.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # In-process, forward-twist-only scan. S_int depends only on mu,
    # so the scan returns one value per mu row.
    table = scan_intermediate_entropy(
        case, params,
        low_chi=low_chi,
        coarse_grid=coarse_grid,
        n_traj_scan=n_traj_scan,
        root=root,
        convention=convention,
    )

    if not table:
        print("[scan] WARNING: entropy scan returned an empty table. "
              "Falling back to PILOT_GRID.", flush=True)
        return []

    # Group by mu and rank mu rows by S_int. Because S_int is the
    # same for all nu at a given mu, this collapses cleanly to one
    # score per mu.
    by_mu = {}
    for (mu, nu), s in table.items():
        by_mu.setdefault(round(float(mu), 4), []).append(float(s))

    mu_scores = sorted(by_mu.items(),
                       key=lambda x: -float(np.mean(x[1])))
    top_mus = [mu for mu, _ in mu_scores[:max(int(top_k), 1)]]

    # For each top-mu row, probe a spread of nu values that spans
    # the three protocol families and the edges of the Brillouin
    # zone. This ensures the union grid covers the ν dimension even
    # though S_int itself is ν-independent.
    expanded = set()
    if convention == "mu1mu2":
        # top_mus here are actually top-mu1 rows (from the mu1mu2 scan)
        for mu1 in top_mus:
            candidates = [
                2.0 * mu1,      # OUT protocol (nu = -mu => mu2 = 2 mu1)
                mu1,            # exact echo (nu = 0 => mu2 = mu1)
                0.0,            # squeezing  (nu = +mu => mu2 = 0)
                np.pi,          # mid-range mu2
                2.0 * np.pi,    # upper edge of mu2
            ]
            for mu2 in candidates:
                if 0.0 - 1e-9 <= mu2 <= 2.0 * np.pi + 1e-9:
                    expanded.add((round(float(mu1), 4),
                                  round(float(mu2), 4)))
    else:
        for mu in top_mus:
            candidates = [
                -mu,        # OUT protocol
                0.0,        # exact echo
                mu,         # squeezing
                -np.pi,     # left edge
                np.pi,      # right edge
            ]
            for nu in candidates:
                if -np.pi - 1e-9 <= nu <= np.pi + 1e-9:
                    expanded.add((round(float(mu), 4),
                                  round(float(nu), 4)))

    return sorted(expanded)


def build_pilot_grid(case, params,
                     fallback_grid=None,
                     cap=UNION_GRID_CAP,
                     root=None,
                     runner=subprocess.run,
                     convention="mu_nu"):
    """
    Union of PILOT_GRID (or fallback_grid) and the top-entanglement
    points identified by identify_worst_entanglement_point.

    Returns a list of (mu, nu) tuples, capped at `cap` entries.
    """
    if fallback_grid is not None:
        base = list(fallback_grid)
    elif convention == "mu1mu2":
        base = list(PILOT_GRID_MU1MU2)
    else:
        base = list(PILOT_GRID)

    top = identify_worst_entanglement_point(case, params,
                                            root=root, runner=runner,
                                            convention=convention)

    seen = set()
    union = []
    # Add the entanglement-guided points first so they survive the cap.
    for pt in top:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        if key not in seen:
            seen.add(key)
            union.append((float(pt[0]), float(pt[1])))
        if len(union) >= cap:
            return union
    # Then fill with the fixed PILOT_GRID.
    for pt in base:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        if key not in seen:
            seen.add(key)
            union.append((float(pt[0]), float(pt[1])))
        if len(union) >= cap:
            return union
    return union


# ----------------------------------------------------------------------
# Pilot SNR extraction
# ----------------------------------------------------------------------
def snr_by_chi(case, alpha=1.0, N=None, Lx=None, Ly=None,
               gamma1=0.0, gamma2=0.0,
               chi_schedule=CHI_SCHEDULE,
               n_traj=N_TRAJ_PILOT,
               pilot_grid=None,
               root=None,
               convention="mu_nu"):
    """
    Read the pilot SNR for every (chi, grid_point) pair. If pilot_grid
    is None, the union grid is read from disk.
    """
    if pilot_grid is None:
        pilot_grid = read_pilot_grid(case, N=N, Lx=Lx, Ly=Ly,
                                     alpha=alpha,
                                     gamma1=gamma1, gamma2=gamma2,
                                     root=root, convention=convention)

    snr = {}
    for chi in chi_schedule:
        pts = {}
        for (a, b) in pilot_grid:
            if convention == "mu1mu2":
                path = native_output_path(
                    case, chi, N=N, Lx=Lx, Ly=Ly,
                    alpha=alpha, gamma1=gamma1, gamma2=gamma2,
                    n_traj=n_traj, root=root,
                    convention="mu1mu2", mu1=a, mu2=b,
                )
            else:
                path = native_output_path(
                    case, chi, N=N, Lx=Lx, Ly=Ly,
                    alpha=alpha, mu=a, nu=b,
                    gamma1=gamma1, gamma2=gamma2,
                    n_traj=n_traj, root=root,
                )
            val = snr_from_output(path)
            if val is not None:
                pts[(round(a, 4), round(b, 4))] = val
        if pts:
            snr[chi] = pts
    return snr


# ----------------------------------------------------------------------
# Convergence criterion (unchanged from v1)
# ----------------------------------------------------------------------
def find_chi_convergence(snr_by_chi, target_residual):
    residuals = {}
    chi_sorted = sorted(snr_by_chi.keys())
    for i, chi_low in enumerate(chi_sorted[:-1]):
        chi_high = chi_sorted[i + 1]
        pts_low = snr_by_chi[chi_low]
        pts_high = snr_by_chi[chi_high]
        common = set(pts_low.keys()) & set(pts_high.keys())
        if not common:
            continue
        max_res = 0.0
        for pt in common:
            s_lo = pts_low[pt]
            s_hi = pts_high[pt]
            denom = max(abs(s_lo), 1e-12)
            res = abs(s_lo - s_hi) / denom
            max_res = max(max_res, res)
        residuals[(chi_low, chi_high)] = max_res
        if max_res < target_residual:
            return chi_low, residuals
    return None, residuals


def chi_production_from_conv(chi_conv,
                            chi_schedule=CHI_SCHEDULE,
                            chi_cap=256,
                            chi_floor=64):
    if chi_conv is None:
        return None
    target = min(max(1.5 * chi_conv, chi_floor), chi_cap)
    for chi in chi_schedule:
        if chi >= target:
            return chi
    return chi_schedule[-1]


# ----------------------------------------------------------------------
# Coarse-grid verification of the converged chi
# ----------------------------------------------------------------------
def verify_chi_on_coarse_grid(case, params, chi_conv, chi_double,
                              coarse_grid=ENTANGLEMENT_SCAN_GRID,
                              tol=COARSE_VERIFY_TOL,
                              n_traj=N_TRAJ_PILOT,
                              root=None,
                              convention="mu_nu"):
    """
    Read the worker output at chi_conv and 2*chi_conv on the full coarse
    grid and return the maximum relative SNR difference over the grid,
    plus the grid point at which it occurs.

    The function reads existing files only. It does NOT run the worker.
    The caller (auto_chi_pilot --aggregate) is responsible for having
    produced the files at chi_conv and chi_double on the coarse grid,
    either from the Stage A scan or from a dedicated Stage B array.

    Returns
    -------
    max_diff : float
    worst_point : (mu, nu) or None
    n_points_compared : int
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if convention == "mu1mu2":
        a_vals = np.linspace(0.0, np.pi, coarse_grid)
        b_vals = np.linspace(0.0, 2.0 * np.pi, coarse_grid)
    else:
        a_vals = np.linspace(0.0, np.pi, coarse_grid)
        b_vals = np.linspace(-np.pi, np.pi, coarse_grid)

    gamma1 = params.get('gamma1', 0.0)
    gamma2 = params.get('gamma2', 0.0)

    max_diff = 0.0
    worst_point = None
    n_compared = 0

    for a in a_vals:
        for b in b_vals:
            if convention == "mu1mu2":
                p_low = native_output_path(
                    case, chi_conv,
                    N=params.get('N'), Lx=params.get('Lx'),
                    Ly=params.get('Ly'),
                    alpha=params['alpha'],
                    gamma1=gamma1, gamma2=gamma2,
                    n_traj=n_traj, root=root,
                    convention="mu1mu2", mu1=a, mu2=b,
                )
                p_hi = native_output_path(
                    case, chi_double,
                    N=params.get('N'), Lx=params.get('Lx'),
                    Ly=params.get('Ly'),
                    alpha=params['alpha'],
                    gamma1=gamma1, gamma2=gamma2,
                    n_traj=n_traj, root=root,
                    convention="mu1mu2", mu1=a, mu2=b,
                )
            else:
                p_low = native_output_path(
                    case, chi_conv,
                    N=params.get('N'), Lx=params.get('Lx'),
                    Ly=params.get('Ly'),
                    alpha=params['alpha'], mu=a, nu=b,
                    gamma1=gamma1, gamma2=gamma2,
                    n_traj=n_traj, root=root,
                )
                p_hi = native_output_path(
                    case, chi_double,
                    N=params.get('N'), Lx=params.get('Lx'),
                    Ly=params.get('Ly'),
                    alpha=params['alpha'], mu=a, nu=b,
                    gamma1=gamma1, gamma2=gamma2,
                    n_traj=n_traj, root=root,
                )
            s_lo = snr_from_output(p_low)
            s_hi = snr_from_output(p_hi)
            if s_lo is None or s_hi is None:
                continue
            n_compared += 1
            denom = max(abs(s_lo), 1e-12)
            diff = abs(s_lo - s_hi) / denom
            if diff > max_diff:
                max_diff = diff
                worst_point = (float(a), float(b))

    return max_diff, worst_point, n_compared

# =====================================================================
# Grid-resolution selector (the documentation, v2)
# =====================================================================
#
# The v2 grid selector identifies the sharpest features of the
# (mu, nu) landscape (GHZ peak near mu=0, OUT plateau edges, sharp
# transition along nu = ±pi) and then chooses the smallest grid
# resolution whose restricted L2 error on those features is below
# a fixed target.
#
# Configuration
# -------------
GRID_SCHEDULE            = [20, 40, 60, 80]
REFERENCE_GRID           = 100              # common interpolation target
COARSE_GRADIENT_CHI      = 32               # chi for the coarse scan
COARSE_GRADIENT_GRID     = 20               # resolution of the coarse scan
SHARP_POINTS_TOP_K       = 3                # number of top gradient peaks
GRID_PILOT_CAP           = 12               # cap on the union grid size
PILOT_GRID_ERROR_TARGET  = 1e-2             # relative to max SNR

# Trajectory count for the grid-resolution pilot. The pilot only needs
# to distinguish grid resolutions, not to produce a metrologically
# precise SNR, so a small ensemble is sufficient. This is deliberately
# independent of N_TRAJ_PILOT, which governs the chi-convergence pilot
# and the production runs.
N_TRAJ_GRID_PILOT        = 5


# ---------------------------------------------------------------------
# Union-grid file management
# ---------------------------------------------------------------------
def _grid_axes(grid_size, convention="mu_nu"):
    """
    Return (x_vals, y_vals) axis vectors for a pilot grid.
      convention='mu_nu':   x = mu in [0, pi], y = nu in [-pi, pi]
      convention='mu1mu2':  x = mu1 in [0, pi], y = mu2 in [0, 2*pi]
    """
    x = np.linspace(0.0, np.pi, grid_size)
    if convention == "mu1mu2":
        y = np.linspace(0.0, 2.0 * np.pi, grid_size)
    else:
        y = np.linspace(-np.pi, np.pi, grid_size)
    return x, y


def grid_pilot_filename(case, params, root=None, convention="mu_nu"):
    """
    Canonical filename for the per-tuple union pilot grid used by the
    grid-resolution selector.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    grid_dir = os.path.join(root, 'data', 'resource_tables', 'grid_pilots')
    os.makedirs(grid_dir, exist_ok=True)

    N   = params.get('N')
    Lx  = params.get('Lx')
    Ly  = params.get('Ly')
    a   = params.get('alpha', 1.0)
    g1  = params.get('gamma1', 0.0)
    g2  = params.get('gamma2', 0.0)

    if case == '1d_closed':
        name = f"{case}_N{N}_alpha{a}.json"
    elif case == '1d_open':
        name = f"{case}_N{N}_alpha{a}_g1{g1:.4f}_g2{g2:.4f}.json"
    elif case == '2d_closed':
        name = f"{case}_Lx{Lx}_Ly{Ly}_alpha{a}.json"
    elif case == '2d_open':
        name = f"{case}_Lx{Lx}_Ly{Ly}_alpha{a}_g1{g1:.4f}_g2{g2:.4f}.json"
    else:
        raise ValueError(f"Unknown case: {case!r}")
    if convention == "mu1mu2":
        name = name.replace(".json", "_mu1mu2.json")
    return os.path.join(grid_dir, name)


def read_grid_pilot(case, params, root=None, convention="mu_nu"):
    """Read the union grid; return None if not present."""
    path = grid_pilot_filename(case, params, root, convention=convention)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return [(float(mu), float(nu)) for mu, nu in data]
    except Exception:
        return None


def write_grid_pilot(case, params, grid, root=None, convention="mu_nu"):
    """Write the union grid to disk."""
    path = grid_pilot_filename(case, params, root, convention=convention)
    with open(path, 'w') as f:
        json.dump([[float(mu), float(nu)] for mu, nu in grid], f, indent=2)
    return path


# ---------------------------------------------------------------------
# Landscape-array file management
# ---------------------------------------------------------------------
def landscape_grid_filename(case, params, grid_size, chi_max, root=None,
                            convention="mu_nu"):
    """
    Canonical filename for the full SNR landscape at a given grid
    resolution and bond dimension.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    grid_dir = os.path.join(root, 'data', 'resource_tables', 'landscape_grids')
    os.makedirs(grid_dir, exist_ok=True)

    N   = params.get('N')
    Lx  = params.get('Lx')
    Ly  = params.get('Ly')
    a   = params.get('alpha', 1.0)
    g1  = params.get('gamma1', 0.0)
    g2  = params.get('gamma2', 0.0)

    if case == '1d_closed':
        name = f"{case}_N{N}_alpha{a}_g{grid_size}_chi{chi_max}.npy"
    elif case == '1d_open':
        name = (f"{case}_N{N}_alpha{a}_g1{g1:.4f}_g2{g2:.4f}"
                f"_g{grid_size}_chi{chi_max}.npy")
    elif case == '2d_closed':
        name = (f"{case}_Lx{Lx}_Ly{Ly}_alpha{a}"
                f"_g{grid_size}_chi{chi_max}.npy")
    elif case == '2d_open':
        name = (f"{case}_Lx{Lx}_Ly{Ly}_alpha{a}_g1{g1:.4f}_g2{g2:.4f}"
                f"_g{grid_size}_chi{chi_max}.npy")
    else:
        raise ValueError(f"Unknown case: {case!r}")
    if convention == "mu1mu2":
        name = name.replace(".npy", "_mu1mu2.npy")
    return os.path.join(grid_dir, name)


def read_landscape_grid(case, params, grid_size, chi_max, root=None,
                        convention="mu_nu"):
    """Read the landscape .npy file; return None if missing."""
    path = landscape_grid_filename(case, params, grid_size, chi_max, root,
                                   convention=convention)
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True).item()
        return d
    except Exception:
        return None


def write_landscape_grid(case, params, grid_size, chi_max, snr_grid,
                       root=None, convention="mu_nu"):
    """Write the landscape .npy file with axis vectors and metadata."""
    path = landscape_grid_filename(case, params, grid_size, chi_max, root,
                                   convention=convention)
    x_vals, y_vals = _grid_axes(grid_size, convention=convention)
    payload = {
        'case': case,
        'grid_size': int(grid_size),
        'chi_max': int(chi_max),
        'snr_grid': snr_grid,
        'params': dict(params),
        'convention': convention,
    }
    if convention == "mu1mu2":
        payload['mu1_vals'] = x_vals
        payload['mu2_vals'] = y_vals
    else:
        payload['mu_vals'] = x_vals
        payload['nu_vals'] = y_vals
    np.save(path, payload)
    return path


# ---------------------------------------------------------------------
# Interpolation and error metrics
# ---------------------------------------------------------------------
def interpolate_to_reference(snr_grid, ref_grid_size=REFERENCE_GRID,
                            convention="mu_nu"):
    """
    Bilinear interpolation of an N x N SNR landscape to
    ref_grid_size x ref_grid_size.

    The source grid samples mu in [0, pi] and nu in [-pi, pi].
    The destination grid samples the same domain at higher resolution.

    Uses scipy's RegularGridInterpolator, which is a guaranteed-correct
    linear interpolant on the tensor-product grid. Extrapolation is
    handled by clamping at the boundary (fill_value=None uses the
    extrapolation produced by the interpolator, which for the pilot
    is acceptable because the boundary is preserved exactly by the
    linspace construction).

    Parameters
    ----------
    snr_grid : ndarray, shape (N, N)
    ref_grid_size : int

    Returns
    -------
    S_ref : ndarray, shape (ref_grid_size, ref_grid_size)
    """
    from scipy.interpolate import RegularGridInterpolator

    src_size = int(snr_grid.shape[0])
    if snr_grid.shape[0] != snr_grid.shape[1]:
        raise ValueError(f"snr_grid must be square, got shape "
                         f"{snr_grid.shape}")

    x_src, y_src = _grid_axes(src_size, convention=convention)

    interp = RegularGridInterpolator(
        (x_src, y_src), snr_grid,
        method='linear', bounds_error=False, fill_value=None,
    )

    x_dst, y_dst = _grid_axes(ref_grid_size, convention=convention)
    X, Y = np.meshgrid(x_dst, y_dst, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    S_ref = interp(pts).reshape(ref_grid_size, ref_grid_size)
    return S_ref


def error_on_pilot_subset(S_fine, S_coarse, pilot_points,
                        ref_grid_size=REFERENCE_GRID,
                        convention="mu_nu"):
    """
    L2 error between two already-interpolated SNR arrays, restricted
    to the pilot points.

    Both S_fine and S_coarse are ref_grid_size x ref_grid_size arrays
    sharing the same mu/nu grid (as produced by
    interpolate_to_reference). The error is evaluated by nearest-neighbour
    lookup on the reference grid at each pilot point.

    Parameters
    ----------
    S_fine, S_coarse : ndarray, shape (ref, ref)
    pilot_points : iterable of (mu, nu)
    ref_grid_size : int

    Returns
    -------
    err : float
        sqrt( mean over pilot points of (S_fine - S_coarse)^2 ).
    """
    x_dst, y_dst = _grid_axes(ref_grid_size, convention=convention)


    pilot = list(pilot_points)
    if not pilot:
        return 0.0


    sq_errs = []
    for (a, b) in pilot:
        # Nearest grid point in the reference grid.
        i = int(np.argmin(np.abs(x_dst - float(a))))
        j = int(np.argmin(np.abs(y_dst - float(b))))
        i = max(0, min(ref_grid_size - 1, i))
        j = max(0, min(ref_grid_size - 1, j))
        diff = float(S_fine[i, j] - S_coarse[i, j])
        sq_errs.append(diff * diff)

    return float(np.sqrt(np.mean(sq_errs)))


# ---------------------------------------------------------------------
# Sharp-point identification
# ---------------------------------------------------------------------
def identify_sharpest_landscape_points(case, params,
                                     chi_coarse=COARSE_GRADIENT_CHI,
                                     coarse_grid=COARSE_GRADIENT_GRID,
                                     top_k=SHARP_POINTS_TOP_K,
                                     root=None,
                                     convention="mu_nu"):
    """
    Read the coarse grid SNR array and identify the top_k points with
    the largest discrete gradient magnitude, expanded to a 3x3
    neighbourhood.

    The coarse grid array must have been produced by an earlier
    submission of the auto-grid pilot (Stage A). If it is missing,
    this function raises a RuntimeError rather than silently
    re-running the worker, because the worker invocation is the
    responsibility of the driver script and must go through SLURM.

    Parameters
    ----------
    case : str
        One of '1d_closed', '1d_open', '2d_closed', '2d_open'.
    params : dict
        Case parameters: N (1D), Lx, Ly (2D), alpha, gamma1, gamma2.
    chi_coarse, coarse_grid, top_k : int
    root : str or None

    Returns
    -------
    sharp_points : list of (mu, nu)
        Sorted list of the top_k peaks, each expanded to a 3x3
        neighbourhood, deduplicated.
    """
    d = read_landscape_grid(case, params, coarse_grid, chi_coarse, root,
                            convention=convention)
    if d is None:
        path = landscape_grid_filename(case, params, coarse_grid,
                                      chi_coarse, root,
                                      convention=convention)
        raise RuntimeError(
            f"No coarse landscape found for case={case}, "
            f"grid={coarse_grid}, chi={chi_coarse}. "
            f"Expected file: {path}. "
            f"Run the pilot pre-scan (auto_grid_pilot.py --submit) first."
        )

    snr_grid = np.asarray(d['snr_grid'], dtype=float)
    if convention == "mu1mu2":
        mu_vals = np.asarray(d['mu1_vals'], dtype=float)
        nu_vals = np.asarray(d['mu2_vals'], dtype=float)
    else:
        mu_vals = np.asarray(d['mu_vals'], dtype=float)
        nu_vals = np.asarray(d['nu_vals'], dtype=float)

    # Spacing of the source grid
    d_mu = float(mu_vals[1] - mu_vals[0]) if len(mu_vals) > 1 else 1.0
    d_nu = float(nu_vals[1] - nu_vals[0]) if len(nu_vals) > 1 else 1.0

    # Central differences in the interior, one-sided at the boundaries.
    g_mu = np.zeros_like(snr_grid)
    g_nu = np.zeros_like(snr_grid)

    if snr_grid.shape[0] >= 2:
        g_mu[:-1, :] = (snr_grid[1:, :] - snr_grid[:-1, :]) / d_mu
        g_mu[-1, :]  = (snr_grid[-1, :] - snr_grid[-2, :]) / d_mu
    if snr_grid.shape[1] >= 2:
        g_nu[:, :-1] = (snr_grid[:, 1:] - snr_grid[:, :-1]) / d_nu
        g_nu[:, -1]  = (snr_grid[:, -1] - snr_grid[:, -2]) / d_nu

    grad_mag = np.sqrt(g_mu ** 2 + g_nu ** 2)
    grad_mag = np.nan_to_num(grad_mag, nan=0.0, posinf=0.0, neginf=0.0)

    flat_idx = np.argsort(grad_mag.ravel())[::-1]
    top_k = max(int(top_k), 1)
    selected = []
    for flat in flat_idx[:top_k]:
        i, j = np.unravel_index(int(flat), grad_mag.shape)
        selected.append((int(i), int(j)))

    # Expand each selected peak to its 3x3 neighbourhood.
    expanded = set()
    n_mu = len(mu_vals)
    n_nu = len(nu_vals)
    for (i, j) in selected:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii = i + di
                jj = j + dj
                if 0 <= ii < n_mu and 0 <= jj < n_nu:
                    expanded.add((round(float(mu_vals[ii]), 4),
                                  round(float(nu_vals[jj]), 4)))

    return sorted(expanded)


def build_grid_pilot(case, params, fallback_grid=None,
                   cap=GRID_PILOT_CAP, root=None, convention="mu_nu"):
    """
    Union of PILOT_GRID (or fallback_grid) and the sharp-point set
    identified by identify_sharpest_landscape_points.


    The entanglement-guided points are added first so that they
    survive the cap. The fixed PILOT_GRID points fill the remainder.


    Returns a list of (x, y) tuples in the chosen convention,
    capped at `cap` entries.
    """
    if fallback_grid is not None:
        base = list(fallback_grid)
    elif convention == "mu1mu2":
        base = list(PILOT_GRID_MU1MU2)
    else:
        base = list(PILOT_GRID)


    try:
        sharp = identify_sharpest_landscape_points(
            case, params, root=root, convention=convention,
        )
    except RuntimeError as exc:
        print(f"[build_grid_pilot] WARNING: {exc}", flush=True)
        sharp = []

    seen = set()
    union = []

    # Add sharp points first so they survive the cap.
    for pt in sharp:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        if key not in seen:
            seen.add(key)
            union.append((float(pt[0]), float(pt[1])))
        if len(union) >= cap:
            return union

    # Fill with PILOT_GRID.
    for pt in base:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        if key not in seen:
            seen.add(key)
            union.append((float(pt[0]), float(pt[1])))
        if len(union) >= cap:
            return union

    return union


# ---------------------------------------------------------------------
# Resolution selection
# ---------------------------------------------------------------------
def choose_grid_size(case, params, grid_schedule=GRID_SCHEDULE,
                   ref_grid_size=REFERENCE_GRID,
                   target=PILOT_GRID_ERROR_TARGET,
                   chi_max=COARSE_GRADIENT_CHI,
                   root=None, convention="mu_nu"):
    """
    Pick the smallest grid resolution whose restricted L2 error on
    the union pilot grid is below `target * max(SNR)`.

    The comparison is made against the largest available resolution
    (which is treated as the reference for that tuple). If any
    resolution is missing from disk, it is skipped with a warning.

    Returns
    -------
    grid_conv : int or None
    max_error : float
    coarse_snr_max : float
    """
    union_grid = read_grid_pilot(case, params, root, convention=convention)
    if union_grid is None:
        # Fall back to the fixed pilot grid.
        union_grid = (list(PILOT_GRID_MU1MU2) if convention == "mu1mu2"
                      else list(PILOT_GRID))

    # Read every landscape array that is present on disk.
    landscapes = {}
    for g in grid_schedule:
        d = read_landscape_grid(case, params, g, chi_max, root,
                                convention=convention)
        if d is not None:
            landscapes[int(g)] = d
        else:
            print(f"[choose_grid_size] missing landscape at grid={g}, "
                  f"chi={chi_max} for case={case}")

    if len(landscapes) < 2:
        print(f"[choose_grid_size] insufficient landscape data "
              f"({len(landscapes)} grids present) for case={case}")
        return None, float('nan'), float('nan')

    # Interpolate every landscape to the common reference grid.
    interpolated = {}
    for g, d in landscapes.items():
        interpolated[g] = interpolate_to_reference(
            d['snr_grid'], ref_grid_size, convention=convention,
        )

    # Reference: the largest available grid.
    ref_g = max(landscapes.keys())
    S_ref = interpolated[ref_g]
    coarse_snr_max = float(np.nanmax(np.abs(S_ref)))

    abs_target = target * max(coarse_snr_max, 1e-30)

    errors = {}
    for g in sorted(landscapes.keys()):
        if g == ref_g:
            errors[g] = 0.0
        else:
           errors[g] = error_on_pilot_subset(S_ref, interpolated[g],
                                             union_grid, ref_grid_size,
                                             convention=convention)

    # Find the smallest resolution whose error is below the target.
    grid_conv = None
    for g in sorted(landscapes.keys()):
        if errors[g] < abs_target:
            grid_conv = int(g)
            break

    if grid_conv is None:
        grid_conv = int(ref_g)

    max_error = max(errors.values()) if errors else float('nan')
    return grid_conv, float(max_error), float(coarse_snr_max)

# =====================================================================
# Trajectory-count selector (the documentation, v2)
# =====================================================================
#
# The v2 trajectory-count selector identifies the highest-variance
# points of the open-system landscape, then separates the statistical
# error from the systematic floor by fitting
#
#     sigma(N_traj, chi) = a(chi) / sqrt(N_traj) + b(chi)
#
# at each (pilot point, chi).
#
# Configuration
# -------------
N_BATCHES_VARIANCE          = 5
N_TRAJ_PER_BATCH_VARIANCE   = 20
VARIANCE_PILOT_CAP          = 8
VARIANCE_POINTS_TOP_K       = 3
SEED_BATCH_BASE             = 20261024

CHI_LIST_NTRAJ              = [16, 32, 64]
NTRAJ_LIST                  = [50, 100, 200, 500, 1000]
N_SEEDS_NTRAJ               = 5
TARGET_STD_DEFAULT          = 1e-2


# ---------------------------------------------------------------------
# Batch output file management
# ---------------------------------------------------------------------
def batch_output_path(case, params, chi_max, mu=None, nu=None,
                    n_traj=None, seed=None,
                    root=None, convention="mu_nu",
                    mu1=None, mu2=None):
    """
    Canonical filename for one batch trajectory run. Stored under
    data/ntraj_batches/ so that batch files with different seeds
    do not collide with the regular per-point production files.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    batch_dir = os.path.join(root, 'data', 'ntraj_batches')
    os.makedirs(batch_dir, exist_ok=True)

    a  = params.get('alpha', 1.0)
    g1 = params.get('gamma1', 0.0)
    g2 = params.get('gamma2', 0.0)


    if convention == "mu1mu2":
        if mu1 is None or mu2 is None:
            raise ValueError(
                "batch_output_path: convention='mu1mu2' requires "
                "mu1 and mu2"
            )
        twist = f"mu1{mu1:.4f}_mu2{mu2:.4f}"
    else:
        if mu is None or nu is None:
            raise ValueError(
                "batch_output_path: convention='mu_nu' requires "
                "mu and nu"
            )
        twist = f"mu{mu:.4f}_nu{nu:.4f}"


    if case == '1d_open':
        name = (f"1d_open_N{params['N']}_alpha{a}"
                f"_g1{g1:.4e}_g2{g2:.4e}"
                f"_chi{chi_max}_{twist}"
                f"_ntraj{n_traj}_seed{seed}.npy")
    elif case == '2d_open':
        name = (f"2d_open_Lx{params['Lx']}_Ly{params['Ly']}_alpha{a}"
                f"_g1{g1:.4e}_g2{g2:.4e}"
                f"_chi{chi_max}_{twist}"
                f"_ntraj{n_traj}_seed{seed}.npy")
    else:
        raise ValueError(
            f"batch_output_path: case {case!r} is not open-system"
        )
    return os.path.join(batch_dir, name)


def read_batch_snr(case, params, chi_max, mu=None, nu=None,
                 n_traj=None, seed=None,
                 root=None, convention="mu_nu",
                 mu1=None, mu2=None):
    """Read SNR from a batch output file. Returns None if missing."""
    path = batch_output_path(case, params, chi_max,
                             mu=mu, nu=nu, n_traj=n_traj, seed=seed,
                             root=root, convention=convention,
                             mu1=mu1, mu2=mu2)
    return snr_from_output(path)


# ---------------------------------------------------------------------
# Highest-variance point identification
# ---------------------------------------------------------------------
def identify_highest_variance_points(case, params,
                                   chi_coarse=32,
                                   coarse_grid=20,
                                   n_batches=N_BATCHES_VARIANCE,
                                   n_traj_per_batch=N_TRAJ_PER_BATCH_VARIANCE,
                                   top_k=VARIANCE_POINTS_TOP_K,
                                   root=None,
                                   convention="mu_nu"):
    """
    Read the coarse-grid batch outputs and identify the top_k grid
    points with the largest sample standard deviation sigma_SNR
    across the n_batches independent seeds.

    The batch outputs must be produced by auto_ntraj_pilot.py
    --submit (Stage A) before this function is called. If the data
    is absent, returns an empty list and prints a warning.

    Parameters
    ----------
    case : str
        '1d_open' or '2d_open'.
    params : dict
        Must contain N (1D) or Lx, Ly (2D), alpha, gamma1, gamma2.
    chi_coarse : int
    coarse_grid : int
    n_batches : int
    n_traj_per_batch : int
    top_k : int

    Returns
    -------
    top_points : list of (mu, nu)
        Sorted list of top_k peaks, each expanded to a 3x3
        neighbourhood, deduplicated.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if case not in ('1d_open', '2d_open'):
        raise ValueError(
            f"identify_highest_variance_points: case must be an "
            f"open-system identifier, got {case!r}"
        )

    x_vals, y_vals = _grid_axes(coarse_grid, convention=convention)


    # Compute sigma_SNR at each grid point.
    sigma_grid = np.full((coarse_grid, coarse_grid), np.nan)
    n_complete = 0
    n_missing = 0


    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            snrs = []
            for b in range(n_batches):
                seed = SEED_BATCH_BASE + b
                if convention == "mu1mu2":
                    s = read_batch_snr(
                        case, params, chi_coarse,
                        n_traj=n_traj_per_batch, seed=seed, root=root,
                        convention="mu1mu2",
                        mu1=float(x), mu2=float(y),
                    )
                else:
                    s = read_batch_snr(
                        case, params, chi_coarse,
                        mu=float(x), nu=float(y),
                        n_traj=n_traj_per_batch, seed=seed, root=root,
                    )
                if s is not None:
                    snrs.append(s)
            if len(snrs) == n_batches:
                sigma_grid[i, j] = float(np.std(snrs, ddof=1))
                n_complete += 1
            else:
                n_missing += 1

    if n_complete == 0:
        print(f"[identify_highest_variance_points] No batch data found "
              f"for {case} at coarse_grid={coarse_grid}, "
              f"chi={chi_coarse}. Run auto_ntraj_pilot.py --submit first.",
              flush=True)
        return []

    print(f"[variance] {case}: {n_complete} points with full batch data, "
          f"{n_missing} incomplete", flush=True)

    # Rank by sigma, take top_k, expand to 3x3.
    finite_idx = np.argwhere(np.isfinite(sigma_grid))
    if len(finite_idx) == 0:
        return []

    ranked = sorted(
        [(int(i), int(j), float(sigma_grid[i, j]))
         for i, j in finite_idx],
        key=lambda t: -t[2],
    )
    top_k = max(int(top_k), 1)
    expanded = set()
    for (i, j, _sig) in ranked[:top_k]:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < coarse_grid and 0 <= jj < coarse_grid:
                    expanded.add((round(float(x_vals[ii]), 4),
                                  round(float(y_vals[jj]), 4)))

    return sorted(expanded)


def build_traj_pilot(case, params, fallback_grid=None,
                   cap=VARIANCE_PILOT_CAP, root=None, convention="mu_nu"):
    """
    Union of PILOT_GRID (or fallback_grid) and the highest-variance
    set identified by identify_highest_variance_points.


    The highest-variance points are added first so they survive the
    cap. The fixed PILOT_GRID points fill the remainder.
    """
    if fallback_grid is not None:
        base = list(fallback_grid)
    elif convention == "mu1mu2":
        base = list(PILOT_GRID_MU1MU2)
    else:
        base = list(PILOT_GRID)


    high = identify_highest_variance_points(
        case, params, root=root, convention=convention,
    )

    seen = set()
    union = []
    for pt in high:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        if key not in seen:
            seen.add(key)
            union.append((float(pt[0]), float(pt[1])))
        if len(union) >= cap:
            return union
    for pt in base:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
        if key not in seen:
            seen.add(key)
            union.append((float(pt[0]), float(pt[1])))
        if len(union) >= cap:
            return union
    return union


# ---------------------------------------------------------------------
# Convergence fit
# ---------------------------------------------------------------------
def _fit_sigma_vs_ntraj(ntraj_vals, sigma_vals):
    """
    Fit sigma = a / sqrt(N_traj) + b by ordinary least squares on the
    linearized model sigma = a * x + b, where x = 1/sqrt(N_traj).

    Returns
    -------
    (a, b) : tuple of float
        Returns (nan, nan) if the fit fails or if there are fewer
        than two usable points.
    """
    ntraj_arr = np.asarray(ntraj_vals, dtype=float)
    sigma_arr = np.asarray(sigma_vals, dtype=float)

    mask = np.isfinite(sigma_arr) & (ntraj_arr > 0)
    if np.sum(mask) < 2:
        return float('nan'), float('nan')

    x = 1.0 / np.sqrt(ntraj_arr[mask])
    y = sigma_arr[mask]

    # Design matrix for y = a * x + b
    A = np.column_stack([x, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return float('nan'), float('nan')

    a, b = float(sol[0]), float(sol[1])
    # A negative a indicates the model does not fit; treat as failure.
    if a < 0:
        return float('nan'), float('nan')
    return a, b


def converge_ntraj_with_chi(case, params,
                          chi_list=None,
                          ntraj_list=None,
                          n_seeds=N_SEEDS_NTRAJ,
                          target_std=TARGET_STD_DEFAULT,
                          union_grid=None,
                          root=None, convention="mu_nu"):
    """
    Read the batch outputs for the (chi, N_traj, seed) grid, fit the
    statistical-plus-systematic model, and return the smallest
    N_traj such that the statistical part is below target_std and
    the smallest chi for which the systematic floor b is below
    target_std.

    Parameters
    ----------
    case : str
        '1d_open' or '2d_open'.
    params : dict
    chi_list : list of int or None
        Defaults to CHI_LIST_NTRAJ.
    ntraj_list : list of int or None
        Defaults to NTRAJ_LIST.
    n_seeds : int
    target_std : float
    root : str or None
    convention : str
        'mu_nu' (legacy) or 'mu1mu2' (paper convention).

    Returns
    -------
    dict with keys
        'chi_floor'  : smallest chi with b < target_std, or None
        'ntraj_conv' : smallest N_traj with a/sqrt(N) < target_std
                       at the chosen chi, or None
        'per_chi'    : dict {chi: {'a': ..., 'b': ...}}
        'per_point'  : dict {chi: list of (a, b, a_fit, b_fit)}
    """
    if chi_list is None:
        chi_list = list(CHI_LIST_NTRAJ)
    if ntraj_list is None:
        ntraj_list = list(NTRAJ_LIST)

    if union_grid is None:
        union_grid = read_grid_pilot(case, params, root,
                                     convention=convention)
    if union_grid is None:
        union_grid = (list(PILOT_GRID_MU1MU2) if convention == "mu1mu2"
                      else list(PILOT_GRID))

    per_chi_a = {}
    per_chi_b = {}
    per_point = {}

    for chi in chi_list:
        a_at_points = []
        b_at_points = []
        point_fits = []
        for (a, b) in union_grid:
            sigmas = []
            for n_traj in ntraj_list:
                snrs = []
                for s_idx in range(n_seeds):
                    seed = SEED_BATCH_BASE + s_idx
                    if convention == "mu1mu2":
                        s = read_batch_snr(
                            case, params, chi,
                            n_traj=n_traj, seed=seed, root=root,
                            convention="mu1mu2",
                            mu1=float(a), mu2=float(b),
                        )
                    else:
                        s = read_batch_snr(
                            case, params, chi,
                            mu=float(a), nu=float(b),
                            n_traj=n_traj, seed=seed, root=root,
                        )
                    if s is not None:
                        snrs.append(s)
                if len(snrs) >= 2:
                    sigmas.append(float(np.std(snrs, ddof=1)))
                else:
                    sigmas.append(float('nan'))
            a_fit, b_fit = _fit_sigma_vs_ntraj(ntraj_list, sigmas)
            point_fits.append((float(a), float(b), a_fit, b_fit))
            if np.isfinite(a_fit):
                a_at_points.append(a_fit)
            if np.isfinite(b_fit):
                b_at_points.append(b_fit)

        per_chi_a[chi] = (float(np.nanmax(a_at_points))
                          if a_at_points else float('nan'))
        per_chi_b[chi] = (float(np.nanmax(b_at_points))
                          if b_at_points else float('nan'))
        per_point[chi] = point_fits

    # Smallest chi with b < target_std.
    chi_floor = None
    for chi in chi_list:
        b_val = per_chi_b.get(chi, float('nan'))
        if np.isfinite(b_val) and b_val < target_std:
            chi_floor = chi
            break

    # Smallest N_traj at chi_floor with a/sqrt(N) < target_std.
    ntraj_conv = None
    if chi_floor is not None:
        a_val = per_chi_a[chi_floor]
        if np.isfinite(a_val):
            for n_traj in ntraj_list:
                if a_val / np.sqrt(n_traj) < target_std:
                    ntraj_conv = n_traj
                    break

    return {
        'chi_floor': chi_floor,
        'ntraj_conv': ntraj_conv,
        'per_chi': {
            chi: {'a': per_chi_a.get(chi, float('nan')),
                  'b': per_chi_b.get(chi, float('nan'))}
            for chi in chi_list
        },
        'per_point': per_point,
    }