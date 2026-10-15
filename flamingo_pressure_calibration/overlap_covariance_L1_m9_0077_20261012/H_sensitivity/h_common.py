"""Shared configuration of Test H: sensitivity of tSZ, kSZ and FRB DM at the A2-ii painting target, at the best
stored radial resolution (the 67-shell X200 grid to 4 R200c).

Test H re-uses Test F's and Test G's machinery unchanged (read-only imports of F_observables/f_*.py,
G_physical_outer/g_*.py, the analysis library and kernels/engine.py).  Their modules are configured BEFORE import,
in this process only:
  F_MODE / G_MODE        = H's mode ('full' or 'smoke')
  G_RESULTS              = <H results>/work/g          (G's modules write nothing there unless called to)
  F_RESULTS              = <H results>/work/g/f_base   (= G's FBASE, so g_common's own consistency check passes)
  F_WORKERS / G_WORKERS  = H's worker count
so nothing is ever written into F_observables/ or G_physical_outer/.

Fine grid (the only change of Test F's configuration, made on the imported f_common module object, never in its
file): Test F's 35-shell F grid and 7 extension shells are replaced by
  shells to 4 R200c : the 67 shells of the X200 storage grid (f_common.X200 = the stacking cache's 'x200' edges),
  extension shells  : 4-4.5, 4.5-5, 5-6, 6-7, 7-8, 8-8.5, 8.5-10, 10-12 R200c (Test F's edges plus a break at 8 R200c,
                      so that "painting to 8 R200c" is an exact edge); 75 basis shells in total.
Every F edge is an X200 edge and every F extension edge is a fine extension edge (the grids nest), so a Test-F
shell average is the volume-weighted mean of its fine shells.  f_kernels.basis_grid() then uses the same radial
nodes inside 4 R200c as Test F (its breaks already contain every X200 edge).

Environment (set by run_test_h.py):
  H_MODE     'full' (default) or 'smoke' (Test F's smoke configuration: 2 calibrated bins, F's 25-halo smoke rows,
             every 25th lightcone redshift shell, 12 multipoles)
  H_RESULTS  output directory (default H_sensitivity/results, or results_smoke)
  H_WORKERS  worker processes (default 4; smoke 1)
  H_SCRATCH  optional directory for the fine bases (results/work/g/f_base/work becomes a symlink to it)
  STUDY_ROOT the study directory (default: the parent of this file's directory)
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import contextlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

MODE = os.environ.get("H_MODE", "full")
if MODE not in ("full", "smoke"):
    raise ValueError(f"H_MODE must be 'full' or 'smoke', not {MODE!r}")
SMOKE = MODE == "smoke"
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
RESULTS = Path(os.environ.get("H_RESULTS", str(HERE / ("results_smoke" if SMOKE else "results")))).resolve()
WORK = RESULTS / "work"
TABLES = RESULTS / "tables"
GCFG = WORK / "g"
FBASE = GCFG / "f_base"
INPUTS = HERE / "inputs"
WORKERS = int(os.environ.get("H_WORKERS", "1" if SMOKE else "4"))
for _d in (RESULTS, WORK, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# ---- optional node-local scratch for the big intermediates (the fine bases, ~0.8 GB each): <F_RESULTS>/work becomes a
#      symlink to H_SCRATCH/f_base_work (maas: the home quota is 3.9 GB, /tmp is a 20 GB local disk)
SCRATCH = os.environ.get("H_SCRATCH")
if SCRATCH:
    _sw = Path(SCRATCH).resolve() / "f_base_work"
    _sw.mkdir(parents=True, exist_ok=True)
    FBASE.mkdir(parents=True, exist_ok=True)
    _link = FBASE / "work"
    if _link.is_symlink() and _link.resolve() != _sw:
        _link.unlink()
    if _link.exists() and not _link.is_symlink():
        if any(_link.iterdir()):
            raise SystemExit(f"{_link} is a non-empty directory; remove it or run without H_SCRATCH")
        _link.rmdir()
    if not _link.is_symlink():
        _link.symlink_to(_sw, target_is_directory=True)

# ---- configure Test F / Test G modules (read-only) before import
os.environ["F_MODE"] = MODE
os.environ["F_RESULTS"] = str(FBASE)
os.environ["F_WORKERS"] = str(WORKERS)
os.environ["G_MODE"] = MODE
os.environ["G_RESULTS"] = str(GCFG)
os.environ["G_WORKERS"] = str(WORKERS)
os.environ["STUDY_ROOT"] = str(ROOT)
for _sub in ("F_observables", "G_physical_outer"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))
import f_common as FC  # noqa: E402

assert Path(FC.RESULTS) == FBASE, (FC.RESULTS, FBASE)

# ---- Test F's own grid (kept for parity checks and the coarse comparisons)
F35 = np.array(FC.F_EDGES, float, copy=True)            # 36 edges, 35 shells to 4 R200c
EXT_F = np.array(FC.EXT_EDGES, float, copy=True)
BASIS42 = np.array(FC.BASIS_EDGES, float, copy=True)
N_F35, N_B42 = int(FC.N_F), int(FC.N_B)

# ---- the fine grid
X200 = np.array(FC.X200, float, copy=True)              # 68 edges, 67 shells to 4 R200c
assert len(X200) == 68 and X200[0] == 0.0 and abs(X200[-1] - 4.0) < 1e-12
EXT_H = np.array([4.5, 5.0, 6.0, 7.0, 8.0, 8.5, 10.0, 12.0])
FINE = np.r_[X200, EXT_H]                                # 76 edges, 75 basis shells
N_X = len(X200) - 1                                      # 67
N_FINE = len(FINE) - 1                                   # 75
VOL_FINE = 4 * np.pi / 3 * np.diff(FINE**3)
IDX35 = np.array([int(np.argmin(np.abs(X200 - e))) for e in F35])
assert np.array_equal(X200[IDX35], F35), "F edges are not bitwise X200 edges"
FINE_OF_F = np.searchsorted(IDX35, np.arange(N_X), side="right") - 1      # F shell of every fine shell (to 4R)
IDX42 = np.r_[IDX35, [N_X + int(np.flatnonzero(np.isclose(EXT_H, e))[0]) + 1 for e in EXT_F]]
assert np.allclose(FINE[IDX42], BASIS42, rtol=0, atol=1e-14)
FINE_OF_B42 = np.searchsorted(IDX42, np.arange(N_FINE), side="right") - 1  # Test-F basis shell of every fine shell

_FINE_ATTRS = dict(F_EDGES=X200, N_F=N_X, EXT_EDGES=EXT_H, BASIS_EDGES=FINE, N_B=N_FINE,
                   SHELL_VOLUME=VOL_FINE, X_END=float(FINE[-1]))
_F_ATTRS = dict(F_EDGES=F35, N_F=N_F35, EXT_EDGES=EXT_F, BASIS_EDGES=BASIS42, N_B=N_B42,
                SHELL_VOLUME=4 * np.pi / 3 * np.diff(BASIS42**3), X_END=float(BASIS42[-1]))


def _set(attrs):
    for k, v in attrs.items():
        setattr(FC, k, v)


_set(_FINE_ATTRS)


@contextlib.contextmanager
def f_grid():
    """Temporarily restore Test F's own 35/42-shell grid on the imported f_common (in this process)."""
    _set(_F_ATTRS)
    try:
        yield
    finally:
        _set(_FINE_ATTRS)


# ---- remaining F / G modules (after the patch; their module-level constants then see the fine grid)
import f_basis as FB  # noqa: E402,F401
import f_fit as FF  # noqa: E402,F401
import f_kernels as FK  # noqa: E402,F401
import f_pdf as FP  # noqa: E402,F401

E = FC.E
CAL_BINS = FC.CAL_BINS
N_CAL = FC.N_CAL
CAL_LABELS = FC.CAL_LABELS
ENGINE_LABELS = FC.ENGINE_BIN_LABELS
SEL_ELL = FC.SEL_ELL
SEED = 20261014
QUAD = {"yy1h", "yy2h", "yytot", "ksz1h", "ksz2h", "ksztot", "dmcl1h", "dmcl2h", "dmcltot", "dm_var"}
SELECTIONS = ("a2ii", "main")         # A2-ii painting partition (the target) and all gas (Test F's "main"; comparison)
SEL_LABEL = {"a2ii": "A2-ii painting partition", "main": "all gas (satellites kept; Test F main)"}
Q_NAMES = FC.Q_NAMES
QUANTILES = FC.QUANTILES

# ---- inputs
F_RESULTS_DIR = ROOT / "F_observables" / ("results_smoke" if SMOKE else "results")
F_PROFILES = F_RESULTS_DIR / "profiles.npz"
F_TRUTH = F_RESULTS_DIR / "truth.npz"
F_TRUTH_SUMMARY = F_RESULTS_DIR / "truth_summary.json"
F_IMPORTANCE = F_RESULTS_DIR / "importance_F.npz"
H0_PROFILES = INPUTS / "h0_profiles_x200.npz"
H0_META = INPUTS / "h0_meta.json"
PLAN = ROOT / "metadata_plan.json"


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = time.perf_counter() - self.t0
        return self.marks[name]


write_json = FC.write_json
read_json = FC.read_json
write_csv = FC.write_csv
plain = FC.plain
sha256 = FC.sha256
pool_map = FC.pool_map


def load_npz(path, keys=None):
    """Load an npz into a dict (only ``keys`` / the keys passing the predicate ``keys`` when given)."""
    with np.load(path, allow_pickle=False) as z:
        if keys is None:
            names = z.files
        elif callable(keys):
            names = [k for k in z.files if keys(k)]
        else:
            names = [k for k in keys if k in z.files]
        return {k: z[k] for k in names}


# ---- memory hygiene (maas attempt 2 was stopped at a 32 GiB tree-RSS guard in 'truth': the parent held the three
#      fine bases (3 x 0.74 GB) plus heap left over from the basis build when f_pdf forked 8 workers, and every forked
#      worker's RSS counts the parent's resident copy-on-write pages again, so tree RSS ~ 9 x parent).  Rule now: the
#      parent drops every basis and large temporary before any fork (f_pdf passes, bootstrap pools), returns freed heap
#      to the OS (malloc_trim) and logs its RSS at each fork point.
def _rss_kib(pid="self"):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _pss_kib(pid="self"):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def release():
    """gc + glibc malloc_trim(0): hand freed heap pages back to the OS before forking workers."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:                                   # noqa: BLE001 - non-glibc platforms: gc only
        pass


def mem(tag="", do_release=True):
    """Release memory, then log this process's current RSS / PSS and peak RSS (GiB); returns the current RSS."""
    if do_release:
        release()
    import resource as _res
    rss = _rss_kib() / 1024**2
    pss = _pss_kib() / 1024**2
    peak = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024**2
    log(f"mem {tag}: parent RSS {rss:.2f} GiB, PSS {pss:.2f} GiB, peak {peak:.2f} GiB")
    MEM_LOG.append({"tag": tag, "rss_GiB": rss, "pss_GiB": pss, "peak_GiB": peak, "t": time.time()})
    return rss


MEM_LOG: list = []


# ---- grid helpers
def mids(edges=X200):
    """Volume-weighted shell radius 0.75 (b^4 - a^4) / (b^3 - a^3) (x units)."""
    e = np.asarray(edges, float)
    return 0.75 * (e[1:]**4 - e[:-1]**4) / (e[1:]**3 - e[:-1]**3)


def dlnx(edges=X200):
    """ln(x_hi / x_lo) per shell; the first shell (0 - 0.025) gets ln(0.025 / 0.0179) (its effective centre 0.716 x
    the outer edge, Test F's convention)."""
    e = np.asarray(edges, float)
    lo = np.where(e[:-1] > 0, e[:-1], 0.716 * e[1])
    return np.log(e[1:] / lo)


def to_f35(values, axis=-1, edges_fine=X200):
    """Volume-weighted rebinning of fine-shell averages (67 shells) to Test F's 35 shells."""
    v = np.moveaxis(np.asarray(values, float), axis, -1)
    vol = 4 * np.pi / 3 * np.diff(np.asarray(edges_fine)**3)
    num = np.add.reduceat(v * vol, IDX35[:-1], axis=-1)
    den = np.add.reduceat(vol, IDX35[:-1])
    return np.moveaxis(num / den, -1, axis)


def sum_to_f35(values, axis=-1):
    """Sum of an additive per-shell quantity (importance, Jacobian) over the fine shells of each F shell."""
    v = np.moveaxis(np.asarray(values, float), axis, -1)
    return np.moveaxis(np.add.reduceat(v[..., :N_X], IDX35[:-1], axis=-1), -1, axis)


def edge_index(edges, value, tol=1e-9):
    k = int(np.argmin(np.abs(np.asarray(edges) - value)))
    assert abs(edges[k] - value) <= tol * max(1.0, value), (value, edges[k])
    return k


def inside(edges, radius):
    """Shells with outer edge <= radius."""
    return np.asarray(edges)[1:] <= radius + 1e-9


def interp_crossing(xr, cum, level):
    """First radius where the cumulative curve cum(xr) reaches ``level`` (linear in ln x between edges); NaN if never."""
    cum = np.asarray(cum, float)
    k = np.flatnonzero(cum >= level)
    if not len(k):
        return np.nan
    k = int(k[0])
    if k == 0:
        return float(xr[0])
    a, b = cum[k - 1], cum[k]
    t = (level - a) / (b - a) if b != a else 1.0
    return float(np.exp(np.log(xr[k - 1]) + t * (np.log(xr[k]) - np.log(xr[k - 1]))))
