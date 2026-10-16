"""Shared configuration of Test L1: where does the halo end?  Truncation radii from the bound gas (FLAMINGO L1_m9, z = 0).

Test L1 is stack-level work on the per-halo cache of the stacking-choices study (local, 2 threads, < 4 GB RAM).  It
re-uses earlier modules READ-ONLY and configures them in this process only:
  stacking_choices_20261014/core/stackcore.py  cache access, mass bins, paired bootstrap multiplicities
  J_halo_dm_lowmass/j_common.py, j_prepare.py   fine X200 grid, hydro radius x_h, the stack-level 400-particle rule
                                                (greedy_groups / merge), J's crossing interpolation; J's output
                                                directory is redirected to L's results/work/jcfg (env J_RESULTS) so
                                                nothing is written into J_halo_dm_lowmass/
  G_physical_outer/g_physics.py                 LinearHalo (CLASS xi_lin, Tinker10 bias), quad_nodes, the 2-halo template
  analysis/ (study library)                     the partition block (A2-ii complement) and the owner block (counts)

Notation (every symbol; all profiles dimensionless):
  x = r / R200c; s = fine X200 shell [x_s, x_s+1] (67 shells to 4 R200c); V_s = 4 pi/3 (x_s+1^3 - x_s^3)
  P = P_e / P200, P200 = G M200c 200 rho_c f_b / (2 R200c)                     (electron pressure)
  E = n_e mu_XGP / (rho_c f_b), mu_XGP = (m_e + 2 m_p/1.76)/0.9                (free electrons)
  G = rho_gas / (rho_c f_b)                                                   (gas mass density)
  selections (recently AGN-heated gas removed everywhere): 'all' every gas particle in the shell; 'a2ii' the A2-ii
  painting partition; 'hb' host-bound (HBT-bound to the central or its own satellites); 'cb' central-bound only;
  'a2iia2i' A2-ii and not bound to another painted halo; 'p10' / 'p11' the A2-ii complement (gas whose r/R200c is
  smaller to a painted neighbour) not bound / bound to another painted halo (owner 14).  all = a2ii + p10 + p11.
  Stack of bin b: F_b,s = sum_h w_h F_h,s / sum_h w_h (w_h inclusion weight); bootstrap = 200 paired halo resamples
  with stackcore's per-bin seed 20261012 + 1000 * bin index, stratified by the design strata.

Environment: L_MODE = 'full' (default) or 'smoke' (2 bins, 20 bootstrap draws; results_smoke/, figures_smoke/).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"

import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

MODE = os.environ.get("L_MODE", "full")
SMOKE = MODE == "smoke"
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
FLAM = STUDY.parent                                           # flamingo_pressure_calibration/
RESULTS = HERE / ("results_smoke" if SMOKE else "results")
FIGURES = HERE / ("figures_smoke" if SMOKE else "figures")
TABLES = RESULTS / "tables"
INPUTS = HERE / "inputs"
WORK = RESULTS / "work"
for _d in (RESULTS, FIGURES, TABLES, INPUTS, WORK):
    _d.mkdir(parents=True, exist_ok=True)

# ---- configure Test J (and through it F/G) read-only, outputs redirected into L
os.environ["J_MODE"] = "full"                                 # J's full grid/bins; L's own smoke mode selects bins
os.environ["J_RESULTS"] = str(WORK / "jcfg")
os.environ["J_WORKERS"] = "1"
os.environ["STUDY_ROOT"] = str(STUDY)
for _p in (STUDY / "J_halo_dm_lowmass", STUDY / "stacking_choices_20261014" / "core", STUDY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(HERE) in sys.path:                                    # L's own modules first (J also has make_figures etc.)
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import j_common as JC  # noqa: E402
import j_prepare as JP  # noqa: E402
import stackcore as SC  # noqa: E402
import g_physics as GPH  # noqa: E402

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"                                      # (j_common sets 1; 2 is the local limit)

GC = GPH.GC
X = np.asarray(JC.X200, float)                                # 68 edges, R200c units
N_X = JC.N_X                                                  # 67
VOL = np.asarray(JC.VOL, float)                               # 4pi/3 dx^3 per fine shell
CUM_EDGE = np.r_[0.0, np.cumsum(VOL)]
J_BINS = list(JC.J_BINS)                                      # engine bins 1..14 = 12.7-12.9 ... 15.3-15.5
LOW_BINS = list(JC.LOW_BINS)                                  # 1..4 = 12.7-13.5 (stack-level 400-particle rule)
BINS = [3, 10] if SMOKE else J_BINS                           # smoke: 13.1-13.3 and 14.5-14.7
N_BOOT = 20 if SMOKE else 200
SEED0 = SC.STUDY_SEED                                         # 20261012 (+ 1000 x bin index, stackcore convention)
K_CUT = dict(JC.K_CUT)                                        # {1: 50, 2: 60, 4: 67}
E_COSMIC = float(JC.E_COSMIC)                                 # 0.3348 = kappa_E Omega_m (E units)
OMEGA_M = float(GC.OMEGA_M)                                   # 0.306 (FLAMINGO D3A, incl. neutrinos)
OMEGA_B = float(GC.OMEGA_B) if hasattr(GC, "OMEGA_B") else 0.0486
F_B = float(GC.F_B)                                           # 0.158823
H_LITTLE = 0.681
BOX_MPC = 1000.0                                              # L1000N1800 (metadata_plan box_size_cmpc, z = 0)
MPC_M = 3.0856775814913673e22
THR = {"B32": 0.32, "B50": 0.50}
X_CORE = 0.3                                                  # crossings are searched at x >= X_CORE (outside the core)
SLOPE_SIGMA = 0.15                                            # Gaussian kernel width in ln x of the local slope fit
SLOPE_SIGMAS = (0.10, 0.15, 0.25)
SLOPE_RANGE = (0.7, 3.6)                                      # steepest-slope search range [R200c]
SHOCK_RANGE = (1.0, 3.6)
LABELS = {b: JC.FC.bin_label(b) for b in range(16)}
CACHE = SC.CACHE

# cache keys (x200 grid, recently AGN-heated gas removed, geometric estimator)
SEL_TOKEN = {"all": "all", "a2ii": "A2ii", "hb": "host_bound", "cb": "central_bound", "a2iia2i": "A2ii_A2i"}
NORM = {"P": "P200", "E": "rhoc_fb", "G": "rhoc_fb"}


def key(field, sel):
    return f"{field}|geometric|removed|{SEL_TOKEN[sel]}|all|x200|{NORM[field]}"


WEIGHTED_KEYS = {"T_M": "T_M|weighted|removed|all|all|x200|T200", "T_V": "T_V|weighted|removed|all|all|x200|T200",
                 "K": "K|weighted|removed|all|all|x200|K200"}
INPUT_FILE = INPUTS / "l_inputs.npz"
INPUT_META = INPUTS / "l_inputs_meta.json"
J_DIR = STUDY / "J_halo_dm_lowmass"
CODEX_HD = FLAM / "halfdome_sz_contributions_20261015"
CODEX_FL = FLAM / "flamingo_lightcone_sz_comparison_20261015"


# ============================================================================ small utilities
def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return [conv(v) for v in o.tolist()] if o.ndim else conv(o.item())
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    Path(path).write_text(json.dumps(conv(obj), indent=1))


def jload(path):
    return json.loads(Path(path).read_text())


def shell_radius(a, b):
    """Volume-weighted mean radius of the shell [a, b] (G's shell_radius)."""
    return 0.75 * (b**4 - a**4) / (b**3 - a**3)


def mass_bins():
    """stackcore's 15 M200c 0.2-dex bins, keyed by bin index (0 = 12.5-12.7 ... 14 = 15.3-15.5)."""
    return {int(b["index"]): b for b in SC.mass_bins("M200c_0.2")}


def boot_matrix(mbin, n_boot=None):
    """[1 + n_boot, n] statistic weights: row 0 = the inclusion weights, rows 1.. = multiplicity x weight of the paired
    stratified halo resamples (stackcore._bootstrap_counts with the per-bin seed; identical draws to stack_bin)."""
    n_boot = N_BOOT if n_boot is None else n_boot
    n = len(mbin["rows"])
    C = SC._bootstrap_counts(n, n_boot, SEED0 + 1000 * int(mbin["index"]), mbin["strata"])
    w = np.asarray(mbin["weights"], float)
    return np.vstack([w[None, :], C * w[None, :]])


def wstack(Wd, F):
    """Inclusion-weighted mean of the finite per-halo values for every row of Wd: [R, S]."""
    F = np.asarray(F, float)
    ok = np.isfinite(F)
    num = Wd @ np.where(ok, F, 0.0)
    den = Wd @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def wstack_log(Wd, F):
    """exp of the weighted mean of ln F over F > 0 ('geomean')."""
    F = np.asarray(F, float)
    ok = np.isfinite(F) & (F > 0)
    num = Wd @ np.where(ok, np.log(np.where(ok, F, 1.0)), 0.0)
    den = Wd @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))


def group_points(S, groups):
    """Merged stacked values [R, G] (volume-weighted over each group of fine shells) and the group radii [G]."""
    S = np.atleast_2d(np.asarray(S, float))
    vals = np.stack([(S[:, a:e] @ VOL[a:e]) / VOL[a:e].sum() for a, e in groups], axis=1)
    xr = np.array([shell_radius(X[a], X[e]) for a, e in groups])
    return vals, xr


def merged_shells(S, groups):
    """Stacked values with each group's volume-weighted mean assigned to every fine shell of the group: [R, 67]."""
    S = np.atleast_2d(np.asarray(S, float))
    out = np.empty_like(S)
    for a, e in groups:
        out[:, a:e] = ((S[:, a:e] @ VOL[a:e]) / VOL[a:e].sum())[:, None]
    return out


# ============================================================================ radii
def crossing_down(xr, f, thr, x_min=X_CORE):
    """Truncation radius where f falls below thr for good.

    Points (xr, f) with xr >= x_min.  R_t = the last downward crossing (f[i] >= thr > f[i+1]) after which f stays below
    thr to the last point, interpolated linearly in f versus ln x.  Returns dict(R, R_first, n_cross, status) with
    status 'ok', 'above' (f >= thr at the last point: R_t > last point, R = NaN), 'below' (f < thr at every point
    from x_min: R = x_min).  R_first = the first downward crossing (differs from R when f is non-monotonic)."""
    xr = np.asarray(xr, float)
    f = np.asarray(f, float)
    m = (xr >= x_min) & np.isfinite(f)
    x, y = xr[m], f[m]
    above = y >= thr
    if not len(y):
        return dict(R=np.nan, R_first=np.nan, n_cross=0, status="empty")
    if above[-1]:
        return dict(R=np.nan, R_first=np.nan, n_cross=int(np.sum(above[:-1] & ~above[1:])), status="above")
    if not above.any():
        return dict(R=float(x[0]), R_first=float(x[0]), n_cross=0, status="below")
    down = np.flatnonzero(above[:-1] & ~above[1:])

    def at(i):
        t = (y[i] - thr) / (y[i] - y[i + 1])
        return float(np.exp(np.log(x[i]) + t * (np.log(x[i + 1]) - np.log(x[i]))))
    return dict(R=at(int(down[-1])), R_first=at(int(down[0])), n_cross=int(len(down)), status="ok")


def cumulative_radius(F, level, upto=4.0):
    """Radius R where X(<R) = level x X(<upto), X(<R) = sum of V_s F_s with the shell average F_s constant inside each
    fine shell (exact inversion inside the crossing shell: X(<R) = X(<x_a) + F_a 4pi/3 (R^3 - x_a^3))."""
    F = np.asarray(F, float)
    cum = np.r_[0.0, np.cumsum(VOL * F)]
    k_up = int(np.argmin(np.abs(X - upto)))
    target = level * cum[k_up]
    if not np.isfinite(target) or target <= 0:
        return np.nan
    k = int(np.flatnonzero(cum >= target)[0])
    if k == 0:
        return float(X[0])
    a = k - 1
    if F[a] <= 0:
        return float(X[a])
    r3 = X[a]**3 + (target - cum[a]) / (4 * np.pi / 3 * F[a])
    return float(min(r3 ** (1 / 3), X[a + 1]))


def cumulative_at(F, R):
    """X(<R) for the shell-constant profile F (exact inside the shell containing R)."""
    F = np.asarray(F, float)
    if not np.isfinite(R):
        return np.nan
    R = min(max(R, 0.0), X[-1])
    k = int(np.searchsorted(X, R, side="right") - 1)
    k = min(k, N_X - 1)
    cum = np.r_[0.0, np.cumsum(VOL * F)]
    return float(cum[k] + F[k] * 4 * np.pi / 3 * (R**3 - X[k]**3))


def budget_radius(F_paint, total):
    """Radius R where the painted gas inside R equals a given total: sum V F_paint (< R) = total (shell-constant)."""
    F = np.asarray(F_paint, float)
    cum = np.r_[0.0, np.cumsum(VOL * F)]
    if not np.isfinite(total) or total <= 0:
        return np.nan
    if cum[-1] < total:
        return np.nan                                         # painted gas inside 4R holds less than the total
    k = int(np.flatnonzero(cum >= total)[0])
    if k == 0:
        return float(X[0])
    a = k - 1
    r3 = X[a]**3 + (total - cum[a]) / (4 * np.pi / 3 * F[a])
    return float(min(r3 ** (1 / 3), X[a + 1]))


def local_slope(xr, lnF, sigma=SLOPE_SIGMA, grid=None, w_pts=None):
    """Logarithmic slope d ln F / d ln x by local quadratic regression in u = ln x with Gaussian kernel weights
    exp(-(u_i - u0)^2 / 2 sigma^2) (optionally times w_pts).  Returns (grid, slope)."""
    u = np.log(np.asarray(xr, float))
    y = np.asarray(lnF, float)
    ok = np.isfinite(y) & np.isfinite(u)
    u, y = u[ok], y[ok]
    wp = np.ones_like(u) if w_pts is None else np.asarray(w_pts, float)[ok]
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.5), np.log(3.9), 140))
    g = np.log(grid)
    out = np.full(len(g), np.nan)
    for i, u0 in enumerate(g):
        k = wp * np.exp(-0.5 * ((u - u0) / sigma) ** 2)
        if np.sum(k > 1e-3 * k.max()) < 4:
            continue
        A = np.vstack([np.ones_like(u), u - u0, (u - u0) ** 2]).T
        Aw = A * k[:, None]
        try:
            coef = np.linalg.solve(A.T @ Aw, Aw.T @ y)
        except np.linalg.LinAlgError:
            continue
        out[i] = coef[1]
    return grid, out


def steepest(grid, slope, lo, hi):
    """Location and value of the minimum (steepest) slope in [lo, hi]; flags a minimum at the range edge."""
    m = (grid >= lo) & (grid <= hi) & np.isfinite(slope)
    if not m.any():
        return dict(R=np.nan, slope=np.nan, edge=True, depth=np.nan)
    i = np.flatnonzero(m)[np.argmin(slope[m])]
    edge = bool(i == np.flatnonzero(m)[0] or i == np.flatnonzero(m)[-1])
    ends = slope[np.flatnonzero(m)[[0, -1]]]
    depth = float(np.min(ends) - slope[i])           # how much steeper the minimum is than the range ends
    # parabolic refinement in ln x (interior minima only; the vertex then lies within half a grid step)
    R = float(grid[i])
    if not edge and 0 < i < len(grid) - 1 and np.all(np.isfinite(slope[i - 1:i + 2])):
        y0, y1, y2 = slope[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den > 0:
            d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            lg = np.log(grid)
            R = float(np.exp(lg[i] + d * (lg[i + 1] - lg[i])))
    return dict(R=R, slope=float(slope[i]), edge=edge, depth=depth)


def zero_cross_down(grid, slope, lo, hi):
    """First radius in [lo, hi] where the slope falls from > 0 to <= 0 (e.g. the entropy maximum, d ln K/d ln x = 0),
    linear interpolation in ln x; NaN with status 'none' if the slope stays positive (or is never positive)."""
    m = (grid >= lo) & (grid <= hi) & np.isfinite(slope)
    g, y = grid[m], slope[m]
    k = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0))
    if not len(k):
        return dict(R=np.nan, status="none")
    i = int(k[0])
    t = y[i] / (y[i] - y[i + 1])
    return dict(R=float(np.exp(np.log(g[i]) + t * (np.log(g[i + 1]) - np.log(g[i])))), status="ok")


def boot_summary(vals, central):
    """Central value, bootstrap 16/50/84 percentiles (NaN draws excluded), and the NaN fraction."""
    v = np.asarray(vals, float)
    fin = v[np.isfinite(v)]
    q = np.percentile(fin, [16, 50, 84]) if len(fin) else [np.nan] * 3
    return dict(value=central, q16=float(q[0]), q50=float(q[1]), q84=float(q[2]),
                nan_fraction=float(1 - len(fin) / max(len(v), 1)), n=int(len(v)))
