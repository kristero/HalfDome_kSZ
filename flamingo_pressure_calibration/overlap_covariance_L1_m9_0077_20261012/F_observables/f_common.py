"""Shared configuration of Test F (observable-driven calibration), plan section "Test F".

Everything outside F_observables/ is read-only: the analysis library and kernels/engine.py are
imported by exact path (python -B, no byte code), earlier studies are only read through them.

Conventions
-----------
* F grid (35 shells, x = r/R200c): a sliver-free sub-grid of the 67-shell X200 storage grid.  Inside
  0.84 R200c it is the radii234 log grid (0.069 dex); beyond, the exact edges 1, 1.1516, 1.34, 1.5,
  1.75, 2, 2.1803, 2.5, 3, 3.3019, 3.6342, 4 (every requested R_fit / R_cut is an edge).  Every edge
  is bitwise an X200 edge, so profiles are exact np.add.reduceat rebinnings of the stored sums.
* Basis shells = the 35 F shells plus 7 extension shells 4-12 R200c (only models may be non-zero
  there: the truth is defined to 4 R200c).
* Calibrated support = the engine's common support: bins 13.5-13.7 ... 15.3-15.5 (10 bins) and
  log M <= 15.388; the published B12/B16 fallback elsewhere, identical in truth and every model.
* Fields: pressure = Pe/P200 (geometric estimator), electrons = n_e mu_XGP/(rho_c f_b); recently
  AGN-heated gas removed (default), all-gas control.  Gas selection 'all' (literature full signal,
  satellites kept); A2 painting-consistent selections as controls.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------- run configuration (environment)
#   F_MODE      'full' (default) or 'smoke' (tiny subset, see below)
#   STUDY_ROOT  the study directory (default: the parent of this file's directory); no absolute paths are
#               hard-coded here -- earlier studies are reached through the analysis library and the engine,
#               which resolve them relative to their own location (STUDY_ROOT/..)
#   F_WORKERS   worker processes (default 4); F_RESULTS: output directory (default F_observables/results,
#               or results_smoke in smoke mode); big intermediates go to <F_RESULTS>/work
MODE = os.environ.get("F_MODE", "full")
if MODE not in ("full", "smoke"):
    raise ValueError(f"F_MODE must be 'full' or 'smoke', not {MODE!r}")
SMOKE = MODE == "smoke"
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
RESULTS = Path(os.environ.get("F_RESULTS", str(ROOT / "F_observables" / ("results_smoke" if SMOKE else "results")))).resolve()
PRODUCTS = RESULTS                      # compact npz / json
TABLES = RESULTS / "tables"             # csv
WORK = RESULTS / "work"                 # big intermediates (basis kernels, fallback PDF slot)
FIGURES = ROOT / "F_observables" / "figures"     # make_figures.py only
WORKERS = int(os.environ.get("F_WORKERS", "4"))
for _d in (PRODUCTS, TABLES, WORK):
    _d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import analysis  # noqa: E402,F401
from analysis import covariance as COV  # noqa: E402,F401
from analysis import families as FAM  # noqa: E402,F401
from analysis import fitting as F  # noqa: E402,F401
from analysis import masks as MSK  # noqa: E402,F401
from analysis import newdata  # noqa: E402
from analysis import stacking as S  # noqa: E402,F401
from analysis import variants as V  # noqa: E402,F401

SEED = 20261012


def load_engine():
    """kernels/engine.py by exact path (read-only), registered as 'kernel_engine'."""
    if "kernel_engine" in sys.modules:
        return sys.modules["kernel_engine"]
    spec = importlib.util.spec_from_file_location("kernel_engine", ROOT / "kernels" / "engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kernel_engine"] = module
    spec.loader.exec_module(module)
    return module


E = load_engine()

# ----------------------------------------------------------------------------- grids
X200 = np.unique(np.r_[0, np.geomspace(.025, 3, 31), np.geomspace(.04, 1.34, 28), 1, 1.5, 1.75, 2, 2.5,
                       np.geomspace(3, 4, 4)])          # kernels/jacobians.py X200_EDGES (= grids.X200 to 3e-17)
_LOG = np.geomspace(.025, 3, 31)                          # radii234 log edges (0.069 dex)
_F_TARGETS = np.unique(np.r_[0, _LOG[_LOG < .9], 1.0, _LOG[(_LOG > 1.1) & (_LOG < 1.2)], 1.34, 1.5, 1.75, 2.0,
                             _LOG[(_LOG > 2.1) & (_LOG < 2.2)], 2.5, np.geomspace(3, 4, 4)])


def f_edges(x200: np.ndarray) -> np.ndarray:
    """The F grid as exact entries of a given X200 edge array (each target matched to 1e-9 relative)."""
    x200 = np.asarray(x200, float)
    idx = np.array([int(np.argmin(np.abs(x200 - t))) for t in _F_TARGETS])
    assert np.all(np.abs(x200[idx] - _F_TARGETS) <= 1e-9 * np.maximum(_F_TARGETS, 1e-3)), "F target not on X200"
    assert np.all(np.diff(idx) > 0) and len(idx) == 36
    return x200[idx]


F_EDGES = f_edges(X200)
N_F = len(F_EDGES) - 1                                     # 35 shells to 4 R200c
EXT_EDGES = np.array([4.5, 5.0, 6.0, 7.0, 8.5, 10.0, 12.0])
BASIS_EDGES = np.r_[F_EDGES, EXT_EDGES]
N_B = len(BASIS_EDGES) - 1                                 # 42 basis shells to 12 R200c
X_END = float(BASIS_EDGES[-1])
SHELL_VOLUME = 4 * np.pi / 3 * np.diff(BASIS_EDGES**3)     # dimensionless 4pi/3 (x_hi^3 - x_lo^3)
REQUIRED_RADII = (1.0, 1.34, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
R_FIT = (2.0, 4.0) if SMOKE else (1.0, 1.34, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
R_CUT = (2.0, 4.0) if SMOKE else (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
# sampling / pass sizes (smoke: <= 2 bins, <= 50 haloes, <= 20 bootstrap draws, every 25th redshift shell)
BOOT_DRAWS = 20 if SMOKE else 1000
LAMBDA_DRAWS = 10 if SMOKE else 50
LAMBDAS = (1.0, 10.0) if SMOKE else (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FOLDS = 2 if SMOKE else 8
SMOKE_HALOES_PER_BIN = 25
Z_STRIDE = 25 if SMOKE else 1
MECKE_SUBSAMPLE = 8 if SMOKE else 64
# nominal coarse bands of kernels/importance (0.1/0.25/0.5 replaced by the nearest F edges)
BAND_EDGES = np.array([0, .1051, .2335, .5185, 1.0, 1.5, 2.0, 3.0, 4.0])
BAND_LABELS = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")
REGIONS = (("inner_lt_1R", 0.0, 1.0), ("1_to_2R", 1.0, 2.0), ("outer_2_to_4R", 2.0, 4.0))


def edge_index(edges, value) -> int:
    k = int(np.argmin(np.abs(np.asarray(edges) - value)))
    assert abs(edges[k] - value) < 6e-5 * max(1.0, value), (value, edges[k])
    return k


def shells_within(edges, radius) -> np.ndarray:
    """Boolean over shells of ``edges`` with outer edge <= radius (radius must be an edge)."""
    return np.asarray(edges)[1:] <= edges[edge_index(edges, radius)] + 1e-12


def band_of_shell(edges=BASIS_EDGES) -> np.ndarray:
    """Coarse band index (0..7) of every shell inside 4R; -1 beyond."""
    mid = np.sqrt(np.maximum(edges[:-1], 1e-6) * edges[1:])
    out = np.searchsorted(BAND_EDGES, mid) - 1
    out[edges[1:] > 4 + 1e-9] = -1
    return out


# ----------------------------------------------------------------------------- mass bins
SUPPORT = dict(E.support_definition())
MASS_EDGES = np.asarray(SUPPORT["mass_edges"])            # 12.5 ... 15.5 (15 bins)
if SMOKE:                                                  # smoke: two calibrated bins, the rest fallback
    _smoke_support = np.zeros_like(np.asarray(SUPPORT["support"], bool))
    _smoke_support[[10, 11]] = True
    SUPPORT["support"] = _smoke_support
CAL_BINS = [int(b) for b in np.flatnonzero(SUPPORT["support"])]   # full: 5..14 (13.5-15.388)
N_CAL = len(CAL_BINS)
MAX_LOGMASS = float(SUPPORT["max_logmass"])


def bin_label(b: int) -> str:
    if b >= len(MASS_EDGES) - 1:
        return ">=15.5"
    return f"{MASS_EDGES[b]:.1f}-{MASS_EDGES[b + 1]:.1f}"


CAL_LABELS = [bin_label(b) for b in CAL_BINS]
ENGINE_BIN_LABELS = [bin_label(b) for b in range(16)]       # 15 bins + overflow (engine mass_bin)


def make_sky():
    """The frozen lightcone catalogue (engine.Lightcone); smoke mode keeps every Z_STRIDE-th redshift shell."""
    sky = E.Lightcone()
    if Z_STRIDE > 1:
        keep = np.zeros(len(sky.z), bool)
        keep[::Z_STRIDE] = True
        sky.counts = sky.counts * keep[:, None]
    return sky


def pool_map(func, items, workers=None, ordered=True):
    """multiprocessing (fork) map, or in-process for one worker."""
    import multiprocessing
    workers = WORKERS if workers is None else workers
    if workers <= 1:
        for item in items:
            yield func(item)
        return
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        it = pool.imap(func, items) if ordered else pool.imap_unordered(func, items)
        for out in it:
            yield out

# ----------------------------------------------------------------------------- multipoles / observables
ELL = (np.array([50., 100., 200., 300., 500., 1000., 2000., 3000., 5000., 8000., 12000., 20000.]) if SMOKE
       else np.asarray(E.ELL_EXTENDED, float))            # full: 106 values, 20 .. 2e4
SEL_ELL = (100, 300, 1000, 3000, 8000)
SEL_INDEX = [int(np.flatnonzero(ELL == e)[0]) for e in SEL_ELL]
PRIMARY_ELL = (100, 300, 1000, 3000)                       # 8000: core / resolution sensitive (secondary)
QUANTILES = np.asarray(E.QUANTILES, float)
Q_NAMES = ("q16", "q50", "q84", "q975", "q999")

# observable keys -> (family, description). Spectra keys carry the multipole: e.g. 'yytot_1000'.
FAMILIES = {
    "tSZ": [f"yytot_{e}" for e in SEL_ELL],
    "kSZ": [f"ksz1h_{e}" for e in SEL_ELL],
    "DM": ["dm_mean", "dm_var"] + [f"dm_{q}" for q in Q_NAMES],
    "tSZxDM": [f"ydmtot_{e}" for e in SEL_ELL],
}
FAMILY_LABEL = {"tSZ": "tSZ auto (yy total)", "kSZ": "kSZ auto (1h)", "DM": "DM PDF (halo-only, z_s = 1)",
                "tSZxDM": "tSZ x DM (total)"}


def primary(key: str) -> bool:
    """True unless the observable is a secondary (ell = 8000) entry."""
    return not key.endswith("_8000")


def family_members(family: str, primary_only: bool = True) -> list[str]:
    keys = FAMILIES[family]
    return [k for k in keys if primary(k)] if primary_only else list(keys)


# ----------------------------------------------------------------------------- data
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = newdata.load()
    return _DATA


# ----------------------------------------------------------------------------- io helpers
def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(plain(value), indent=1) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        Path(path).write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k)) for k in keys})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.8g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = time.perf_counter() - self.t0
        return self.marks[name]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)
