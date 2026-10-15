"""Shared configuration, data access and estimator core of Test B (covariance, weightings, GLS, cut-offs).

Plan: /root/.claude/plans/there-has-been-multiple-indexed-lantern.md, "Test B".  Everything is written inside
B_covariance/; the study products and analysis/ are read-only (python -B).

Conventions (see REPORT.md):
* fit grid: radii234 edges up to 0.8369 R200c plus 1, 1.15156, 1.34, 1.5, 1.75, 2, 2.18026, 2.5, 3, 3.30192,
  3.63424, 4 (35 shells; every plan cut-off is an exact edge; rebinned from the 67-shell X200 storage grid);
* stacks: inclusion-weighted per-shell means over haloes with finite values (excision NaN = missing);
* bootstrap: stratified by (0.2-dex bin, census class) -- the strata of the enlarged design -- so that the CLEAN
  and CONT columns of one replicate are themselves stratified replicates (paired comparisons);
* fit units: bins 13.5-13.7 ... 14.7-14.9 and the pooled >= 14.9 sample (bins 14.9-15.5, N = 154) whose stack is
  a per-shell inclusion-weighted linear regression on log10 M, evaluated at the pooled mean mass ('P') and at the
  mean mass of each 0.2-dex bin ('P12', 'P13', 'P14'; these give the per-bin parameters used by the engine).
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"      # before numpy: single-threaded BLAS in every worker

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
# Study root: env STUDY_ROOT (the cluster copy) or the directory above B_covariance/.  No absolute user paths.
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
sys.path.insert(0, str(ROOT))

import analysis  # noqa: E402,F401
from analysis import covariance as C  # noqa: E402
from analysis import fitting as F  # noqa: E402
from analysis import masks as msk  # noqa: E402
from analysis import stacking as S  # noqa: E402

# ----------------------------------------------------------------------------- run configuration (driver sets env)
MODE = os.environ.get("B_MODE", "full")            # 'full' (cluster) or 'smoke' (tiny local subset)
SMOKE = MODE == "smoke"
RESULTS = Path(os.environ.get("B_RESULTS", str(HERE / ("results_smoke" if SMOKE else "results")))).resolve()
PRODUCTS = RESULTS                                  # compute-stage outputs: npz / csv / json, no figures
FITS_DIR = RESULTS / "fits"
FIGURES = HERE / ("figures_smoke" if SMOKE else "figures")      # make_figures.py only (local)
TABLES = HERE / ("tables_smoke" if SMOKE else "tables")         # make_figures.py only (local)
LOGS = HERE / "logs"


def _env_int(key, full, smoke):
    return int(os.environ.get(key, str(smoke if SMOKE else full)))


WORKERS = max(1, _env_int("B_WORKERS", 4, 1))
REPLICATES = _env_int("B_REPLICATES", 500, 10)     # paired halo-bootstrap replicates (parameter intervals)
INNER_DRAWS = _env_int("B_INNER_DRAWS", 1000, 20)  # bootstrap draws for Cov[ln mean]
LAMBDA_DRAWS = _env_int("B_LAMBDA_DRAWS", 50, 20)  # resamples for Var(r_ij) (shrinkage intensity)
PERMUTATIONS = _env_int("B_PERMUTATIONS", 1000, 20)  # permutation null (B2)
MODEL_BOOT = _env_int("B_MODEL_BOOT", 300, 20)     # bootstrap draws of correlation models (B2)
CROSSFIT_SPLITS = _env_int("B_CROSSFIT_SPLITS", 10, 2)
B3_BOOT = _env_int("B_B3_BOOT", 1000, 20)          # stratified bootstrap of the weighting ratios (B3)
CONTROL_TRIALS = _env_int("B_CONTROL_TRIALS", 40, 2)    # permutation-null false-positive control (B2)
CONTROL_PERMS = _env_int("B_CONTROL_PERMS", 199, 19)
JACKKNIFE_BLOCKS_PER_AXIS = 4                      # 4^3 sub-volumes of the 1 Gpc box
SEED = 20261012                                    # study seed (analysis.stacking.STUDY_SEED)
SEEDS = {"replicates": SEED + 100, "inner": SEED + 200, "lambda": SEED + 300, "crossfit": SEED + 400,
         "permutation": SEED + 500, "model_boot": SEED + 600, "mock": SEED + 700, "smoke": SEED + 800}

CUTOFFS = (1.0, 1.34, 1.5, 1.75, 2.0, 3.0, 4.0)
REF_CUT = 2.0
OBJECTIVES = ("a", "b", "c")
FAMILY = {"pressure": "b12", "electrons": "b16", "gas": "b16"}
PARAMS = {"b12": ("P0", "xc", "beta"), "b16": ("rho0", "alpha", "beta")}
FIELD_LABEL = {"pressure": "electron pressure P_e/P200", "electrons": "electron density mu n_e/(f_b rho_c)",
               "gas": "gas density rho_gas/(f_b rho_c)"}
H_FLAMINGO = 0.681
BOX_MPC = 1000.0

# grid: radii234 edges up to 0.8369 plus the exact cut-off edges (indices into grids.X200 are found at run time)
FIT_EXTRA_EDGES = (1.0, 1.15156, 1.34, 1.5, 1.75, 2.0, 2.18026, 2.5, 3.0, 3.30193, 3.63424, 4.0)
FIT_INNER_MAX = 0.8370
REGION_BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))

MASS_EDGES = np.round(np.arange(12.5, 15.51, 0.2), 1)
if SMOKE:
    # <= 2 mass bins, <= 50 haloes: 14.7-14.9 as a single unit and 14.9-15.1 as a one-bin 'pooled' unit (its
    # log-M-trend target P12), so every code path of the full run is exercised
    SMOKE_BINS = (11, 12)
    SMOKE_PER_BIN = 25
    FIT_BINS = (11,)
    POOL_BINS = (12,)
    UNITS = ("11", "P")
    TARGETS = {"11": ("11",), "P": ("P", "P12")}
    ENGINE_TARGET_OF_BIN = {11: "11", 12: "P12"}
    NEIGHBOUR_ORDER = ("11", "P")
    ENGINE_ELL = (100., 1000., 3000.)
else:
    SMOKE_BINS = None
    SMOKE_PER_BIN = None
    FIT_BINS = (5, 6, 7, 8, 9, 10, 11)        # 13.5-13.7 ... 14.7-14.9 fitted individually
    POOL_BINS = (12, 13, 14)                  # >= 14.9 pooled with a log M trend
    UNITS = ("5", "6", "7", "8", "9", "10", "11", "P")
    TARGETS = {"5": ("5",), "6": ("6",), "7": ("7",), "8": ("8",), "9": ("9",), "10": ("10",), "11": ("11",),
               "P": ("P", "P12", "P13", "P14")}
    ENGINE_TARGET_OF_BIN = {5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "11", 12: "P12", 13: "P13",
                            14: "P14"}
    NEIGHBOUR_ORDER = ("4", "5", "6", "7", "8", "9", "10", "11", "P")
    ENGINE_ELL = (20., 30., 50., 70., 100., 150., 200., 300., 500., 700., 1000., 1500., 2000., 3000., 5000., 8000.)
ALL_TARGETS = tuple(t for u in UNITS for t in TARGETS[u])


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def pool_map(fn, tasks, workers=None, ordered=True):
    """multiprocessing map with fork (workers > 1) or a plain in-process loop (workers == 1)."""
    import multiprocessing as mp
    workers = WORKERS if workers is None else workers
    if workers <= 1:
        for t in tasks:
            yield fn(t)
        return
    with mp.get_context("fork").Pool(workers) as pool:
        it = pool.imap(fn, tasks, chunksize=1) if ordered else pool.imap_unordered(fn, tasks, chunksize=1)
        for r in it:
            yield r


def bin_label(b: int) -> str:
    return f"{MASS_EDGES[b]:.1f}-{MASS_EDGES[b + 1]:.1f}"


def target_label(t: str) -> str:
    if t == "P":
        return ">=14.9 pooled"
    if t.startswith("P"):
        return f"{bin_label(int(t[1:]))} (pooled trend)"
    return bin_label(int(t))


# ----------------------------------------------------------------------------- JSON / CSV
def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path: Path, value) -> None:
    Path(path).write_text(json.dumps(plain(value), indent=1, allow_nan=False) + "\n")


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        Path(path).write_text("")
        return
    fields = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k)) for k in fields})


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- prepared products
def halos() -> dict:
    return dict(np.load(PRODUCTS / "halo_table.npz"))


def profiles(grid: str = "fit") -> dict:
    return dict(np.load(PRODUCTS / f"profiles_{grid}.npz"))


def masks(grid: str = "fit") -> dict:
    return dict(np.load(PRODUCTS / f"masks_{grid}.npz"))


def fit_edges() -> np.ndarray:
    return np.asarray(np.load(PRODUCTS / "profiles_fit.npz")["edges"], float)


def unit_bins(unit: str) -> tuple:
    return POOL_BINS if unit == "P" else (int(unit),)


def unit_rows(h: dict, unit: str) -> np.ndarray:
    return np.flatnonzero(np.isin(h["mass_bin"], unit_bins(unit)))


def strata_of(h: dict, rows: np.ndarray) -> np.ndarray:
    """Stratum id (bin x census class) of each row -- the strata of the enlarged sampling design."""
    return np.asarray(h["mass_bin"][rows], int) * 3 + np.asarray(h["class_main"][rows], int)


def unit_code(unit: str) -> int:
    return 99 if unit == "P" else int(unit)


# ----------------------------------------------------------------------------- stack targets
class StackTarget:
    """Inclusion-weighted per-shell mean of per-halo values (NaN = missing halo-shell), or, with ``x_target``,
    the per-shell weighted linear regression on log10 M evaluated at ``x_target`` (the pooled >= 14.9 trend).

    ``means(counts)`` takes multiplicity matrices [draw, n] (ones = the full sample) and is exactly linear in the
    halo weights, so bootstrap replicates and inner resamples use one code path.
    """

    def __init__(self, values, weights, logm=None, x_target=None, pivot=None):
        self.values = np.asarray(values, float)
        self.finite = np.isfinite(self.values)
        self.Fm = self.finite.astype(float)
        self.v0 = np.where(self.finite, self.values, 0.0)
        self.w = np.asarray(weights, float)
        self.n = len(self.values)
        self.trend = x_target is not None
        if self.trend:
            pivot = float(np.average(logm, weights=self.w)) if pivot is None else float(pivot)
            self.pivot = pivot
            self.x = np.asarray(logm, float) - pivot
            self.xt = float(x_target) - pivot
            self.Fx = self.Fm * self.x[:, None]
            self.Fxx = self.Fm * self.x[:, None] ** 2
            self.vx = self.v0 * self.x[:, None]

    def _moments(self, counts):
        cw = np.atleast_2d(np.asarray(counts, float)) * self.w[None, :]
        out = {"S0": cw @ self.Fm, "Sy": cw @ self.v0}
        if self.trend:
            out.update(S1=cw @ self.Fx, S2=cw @ self.Fxx, Sxy=cw @ self.vx)
        return out

    def means(self, counts) -> np.ndarray:
        m = self._moments(counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            if not self.trend:
                return np.where(m["S0"] > 0, m["Sy"] / m["S0"], np.nan)
            det = m["S0"] * m["S2"] - m["S1"] ** 2
            slope = np.where(det > 0, (m["S0"] * m["Sxy"] - m["S1"] * m["Sy"]) / det, 0.0)
            return m["Sy"] / m["S0"] + slope * (self.xt - m["S1"] / m["S0"])

    def mean_full(self) -> np.ndarray:
        return self.means(np.ones((1, self.n)))[0]

    def sd(self, counts_row=None) -> np.ndarray:
        """Reliability-weighted SD per shell about the mean (or about the regression line for the trend)."""
        c = np.ones(self.n) if counts_row is None else np.asarray(counts_row, float)
        cw = c * self.w
        out = np.full(self.values.shape[1], np.nan)
        for s in range(self.values.shape[1]):
            ok = self.finite[:, s] & (cw > 0)
            if ok.sum() < 3:
                continue
            v, ww = self.values[ok, s], cw[ok]
            if self.trend:
                x = self.x[ok]
                W = ww.sum()
                xm, vm = (ww * x).sum() / W, (ww * v).sum() / W
                sxx = (ww * (x - xm) ** 2).sum()
                slope = (ww * (x - xm) * (v - vm)).sum() / sxx if sxx > 0 else 0.0
                resid = v - (vm + slope * (x - xm))
            else:
                resid = v - (ww * v).sum() / ww.sum()
            # copies of a halo carry its weight each: sum of squared weights over copies = sum c w^2
            w2 = (c[ok] * self.w[ok] ** 2).sum()
            corr = 1 - w2 / ww.sum() ** 2
            out[s] = np.sqrt((ww * resid ** 2).sum() / ww.sum() / corr) if corr > 0 else np.nan
        return out

    def n_eff(self, shell_mask=None) -> float:
        w = self.w
        if shell_mask is not None:
            ok = self.finite[:, np.asarray(shell_mask, bool)].all(axis=1)
            w = w[ok]
        return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def counts_from_positions(positions: np.ndarray, n: int) -> np.ndarray:
    return S.counts_matrix(np.atleast_2d(positions), n)


def log_mean_covariance(target: StackTarget, positions: np.ndarray, inner: np.ndarray, base: np.ndarray,
                        chunk: int = 250) -> dict:
    """Bootstrap Cov[ln mean] on ``base`` shells: inner resamples ``positions[inner]`` of one sample draw."""
    base = np.asarray(base, bool)
    logs = []
    for start in range(0, len(inner), chunk):
        counts = counts_from_positions(positions[inner[start:start + chunk]], target.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs.append(np.log(target.means(counts)[:, base]))
    L = np.vstack(logs)
    good = np.all(np.isfinite(L), axis=1)
    Cv = np.atleast_2d(np.cov(L[good], rowvar=False))
    return {"cov": Cv, "corr": C.cov_to_corr(Cv), "sd": np.sqrt(np.diag(Cv)), "draws": int(good.sum()),
            "failed": int((~good).sum())}


def weighted_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return C.cov_to_corr(C.weighted_cov(X, w))


def correlation_variance(target: StackTarget, base: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Element-wise variance of the inclusion-weighted correlation of the linear per-halo values on ``base``
    (complete cases) over halo resamples ``rows`` [draw, n] (Schaefer-Strimmer numerator by resampling)."""
    base = np.asarray(base, bool)
    X = target.values[:, base]
    ok = np.all(np.isfinite(X), axis=1)
    out = []
    for r in rows:
        rr = r[ok[r]]
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(weighted_corr(X[rr], target.w[rr]))
    out = np.array(out)
    return np.nanvar(out, axis=0, ddof=1)


def sub_index(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.flatnonzero(base), np.flatnonzero(mask))


def fit_mask(base, edges, cut, mean) -> np.ndarray:
    return msk.fit_mask(base, edges, cut, mean)


def band_shares(H: np.ndarray, edges: np.ndarray, mask: np.ndarray, bands=REGION_BANDS) -> np.ndarray:
    """Share of sum|H| per radial band ([param, band]) for influence rows H [param, fit shell]."""
    edges = np.asarray(edges, float)
    centre = np.sqrt(np.maximum(edges[:-1], 1e-6) * edges[1:])[np.asarray(mask, bool)]
    A = np.abs(H)
    tot = A.sum(axis=1, keepdims=True)
    out = np.zeros((H.shape[0], len(bands)))
    for k, (lo, hi) in enumerate(bands):
        sel = (centre >= lo) & (centre < hi)
        out[:, k] = A[:, sel].sum(axis=1)
    return out / np.where(tot > 0, tot, np.nan)


def model_integral(family: str, params, upto: float, n: int = 4000) -> float:
    """4 pi int_0^upto x^2 F(x) dx of the (untruncated) model, dimensionless (Y or N_e in R200c^3 x scale)."""
    x = np.geomspace(1e-6, upto, n)
    f = F.shape(family, x, params)
    return float(np.trapezoid(4 * np.pi * x ** 2 * f, x))


def stack_integral(mean: np.ndarray, edges: np.ndarray, upto: float) -> float:
    """4 pi/3 sum_s F_s (x_hi^3 - x_lo^3) over shells with outer edge <= upto (measured stack; NaN -> 0)."""
    edges = np.asarray(edges, float)
    inside = edges[1:] <= upto + 1e-12
    w = 4 * np.pi / 3 * np.diff(edges ** 3)
    return float(np.nansum(np.where(inside, mean * w, 0.0)))


def plotting():
    from analysis import plotting as PT
    PT.apply_style()
    return PT
