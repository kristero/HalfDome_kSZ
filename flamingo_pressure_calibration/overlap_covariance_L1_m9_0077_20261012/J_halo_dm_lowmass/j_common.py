"""Shared configuration of Test J: the halo component of FRB DM (three definitions) and the low-mass calibration.

Test J re-uses Test F's / G's machinery unchanged (read-only imports of F_observables/f_*.py, G_physical_outer/g_pdf.py,
the analysis library and kernels/engine.py).  Their modules are configured BEFORE import, in this process only:
  F_MODE / G_MODE        = J's mode ('full' or 'smoke')
  G_RESULTS              = <J results>/work/g
  F_RESULTS              = <J results>/work/g/f_base   (big intermediates of F's code go to f_base/work)
  F_WORKERS / G_WORKERS  = J's worker count
so nothing is written into F_observables/, G_physical_outer/ or H_sensitivity/.

Two configuration changes are made on the imported f_common module object (never in its file), in this process only:
  1. the fine grid of Test H: the 67 X200 shells to 4 R200c + 8 extension shells to 12 R200c (75 basis shells);
  2. the calibrated support.  Test F / H calibrate the engine bins 13.5-13.7 ... 15.3-15.5 with log M <= 15.388
     (kernels/engine.py support_definition; f_common.CAL_BINS).  Test J extends it to the bins 12.7-12.9 ... 15.3-15.5
     (configuration 'C14': 14 calibrated bins; engine bin 12.5-12.7 holds no painted lightcone halo, checked in
     'prepare').  The other two configurations used for comparisons:
        'C10'  Test H's support (13.5-15.388); every other halo painted with the published B12/B16 fallback;
        'C0'   no calibrated bin: every halo painted with the published B12/B16 fallback.
     support(name) switches FC.SUPPORT['support'], FC.CAL_BINS, FC.N_CAL and FC.CAL_LABELS (context manager).
     Haloes above 10^15.388 are fallback-painted in every configuration (no FLAMINGO halo is that massive).

Definitions of the halo electron component (DM; kSZ uses D0 and D3):
  D0  A2-ii gas to 4 R200c (Test H's painting target)
  D1  A2-ii gas inside the 3D sphere r <= R200c       (the profile is cut at x = 1 in 3D, then projected)
  D2  A2-ii gas inside r <= 2 R200c
  D3  gas bound (HBT) to the halo or its own satellites ('host_bound'), all radii to 4 R200c
The published fallback haloes are cut at the same radius as the definition (D1: 1 R200c, D2: 2 R200c, D0/D3: 4 R200c).
Pressure (tSZ side of y x DM) is always the A2-ii target to 4 R200c.

Environment (set by run_test_j.py):
  J_MODE     'full' (default) or 'smoke' (2 calibrated bins 13.1-13.3 and 14.5-14.7 in C14, 14.5-14.7 in C10; 25 haloes
             per engine bin; Test F's smoke sky (every 25th lightcone redshift shell) and 12 multipoles)
  J_RESULTS  output directory (default J_halo_dm_lowmass/results, or results_smoke)
  J_WORKERS  worker processes (default 4; smoke 1)
  J_SCRATCH  optional directory for the bases and other big intermediates (results/work/g/f_base/work becomes a
             symlink to J_SCRATCH/f_base_work)
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

MODE = os.environ.get("J_MODE", "full")
if MODE not in ("full", "smoke"):
    raise ValueError(f"J_MODE must be 'full' or 'smoke', not {MODE!r}")
SMOKE = MODE == "smoke"
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
RESULTS = Path(os.environ.get("J_RESULTS", str(HERE / ("results_smoke" if SMOKE else "results")))).resolve()
WORK = RESULTS / "work"
TABLES = RESULTS / "tables"
GCFG = WORK / "g"
FBASE = GCFG / "f_base"
INPUTS = HERE / "inputs"
WORKERS = int(os.environ.get("J_WORKERS", "1" if SMOKE else "4"))
for _d in (RESULTS, WORK, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# ---- optional node-local scratch for the big intermediates (bases ~1 GB each, fallback pieces)
SCRATCH = os.environ.get("J_SCRATCH")
if SCRATCH:
    _sw = Path(SCRATCH).resolve() / "f_base_work"
    _sw.mkdir(parents=True, exist_ok=True)
    FBASE.mkdir(parents=True, exist_ok=True)
    _link = FBASE / "work"
    if _link.is_symlink() and _link.resolve() != _sw:
        _link.unlink()
    if _link.exists() and not _link.is_symlink():
        if any(_link.iterdir()):
            raise SystemExit(f"{_link} is a non-empty directory; remove it or run without J_SCRATCH")
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
E = FC.E
A = E.A
BIG = FC.WORK                                            # = FBASE/work (the scratch symlink when J_SCRATCH is set)

# ---- the fine grid (Test H's, h_common.py; identical construction)
F35 = np.array(FC.F_EDGES, float, copy=True)
X200 = np.array(FC.X200, float, copy=True)               # 68 edges, 67 shells to 4 R200c
assert len(X200) == 68 and X200[0] == 0.0 and abs(X200[-1] - 4.0) < 1e-12
EXT_H = np.array([4.5, 5.0, 6.0, 7.0, 8.0, 8.5, 10.0, 12.0])
FINE = np.r_[X200, EXT_H]                                # 76 edges, 75 basis shells
N_X = len(X200) - 1                                      # 67
N_FINE = len(FINE) - 1                                   # 75
VOL_FINE = 4 * np.pi / 3 * np.diff(FINE**3)
VOL = VOL_FINE[:N_X]
for _k, _v in dict(F_EDGES=X200, N_F=N_X, EXT_EDGES=EXT_H, BASIS_EDGES=FINE, N_B=N_FINE, SHELL_VOLUME=VOL_FINE,
                   X_END=float(FINE[-1])).items():
    setattr(FC, _k, _v)

# ---- calibrated supports (configurations)
_SUPPORT0 = np.asarray(E.support_definition()["support"], bool).copy()     # Test F / H: bins 5..14
N_ENGINE = len(_SUPPORT0)                                # 15 bins 12.5-15.5 (+ engine overflow bin 15)
ENGINE_LABELS = [FC.bin_label(b) for b in range(16)]
if SMOKE:
    _C14_BINS, _C10_BINS = [3, 10], [10]
else:
    _C14_BINS, _C10_BINS = list(range(1, 15)), [int(b) for b in np.flatnonzero(_SUPPORT0)]
assert SMOKE or _C10_BINS == list(range(5, 15))
SUPPORT_BINS = {"C14": _C14_BINS, "C10": _C10_BINS, "C0": []}
J_BINS = list(_C14_BINS)                                 # the calibrated bins of Test J (index c <-> J_BINS[c])
N_J = len(J_BINS)
LOW_BINS = [b for b in J_BINS if b < 5]                  # newly calibrated (12.7-13.5)
C10_IN_J = [J_BINS.index(b) for b in _C10_BINS]          # positions of the C10 bins in the J index
CONFIG = {"name": None}


def set_support(name):
    bins = SUPPORT_BINS[name]
    mask = np.zeros(N_ENGINE, bool)
    mask[bins] = True
    FC.SUPPORT["support"] = mask
    FC.CAL_BINS = list(bins)
    FC.N_CAL = len(bins)
    FC.CAL_LABELS = [FC.bin_label(b) for b in bins]
    CONFIG["name"] = name


@contextlib.contextmanager
def support(name):
    """Temporarily switch the calibrated support (in this process; forked workers inherit it)."""
    old = CONFIG["name"]
    set_support(name)
    try:
        yield
    finally:
        set_support(old)


set_support("C14")
J_LABELS = [FC.bin_label(b) for b in J_BINS]

# ---- remaining F / G modules (after the patches; their module-level constants see the fine grid)
import f_basis as FB  # noqa: E402,F401
import f_kernels as FK  # noqa: E402,F401
import f_pdf as FP  # noqa: E402,F401
import f1_importance as FI  # noqa: E402,F401
import g_common as GC  # noqa: E402  (constants only)
import g_pdf as GP  # noqa: E402  (fast deposit, bitwise identical)

# cosmic mean electron density in E units: kappa_E Omega_m with the electrons-per-gas-mass factor measured in the 2-4 R200c
# shells, kappa_E = 1.0939938769723199 (G_physical_outer/results/prepare.json: normalisation.KAPPA_E_measured_2_4R;
# G REPORT: 0.3348); the fully ionised primordial value kappa_E = 1.1066 (X_H = 0.752) gives 0.3386
KAPPA_E_MEASURED = 1.0939938769723199
E_COSMIC = float(KAPPA_E_MEASURED * GC.OMEGA_M)
E_COSMIC_THEORY = float(GC.KAPPA_E_THEORY * GC.OMEGA_M)
F_B = float(GC.F_B)
SEL_ELL = FC.SEL_ELL
QUANTILES = FC.QUANTILES
Q_NAMES = FC.Q_NAMES
QUAD = {"yy1h", "yy2h", "yytot", "ksz1h", "ksz2h", "ksztot", "dmcl1h", "dmcl2h", "dmcltot", "dm_var"}
SEED = 20261015

DEFS = {"D0": dict(sel="a2ii", cut=4.0, basis="JA", label="A2-ii gas to 4 R200c (Test H target)"),
        "D1": dict(sel="a2ii", cut=1.0, basis="JA", label="A2-ii gas inside the 3D sphere r <= R200c"),
        "D2": dict(sel="a2ii", cut=2.0, basis="JA", label="A2-ii gas inside the 3D sphere r <= 2 R200c"),
        "D3": dict(sel="hb", cut=4.0, basis="JB", label="host-bound gas (central + own satellites), to 4 R200c")}
CUTS = (1.0, 2.0, 4.0)
TREATMENTS = {"T1": "measured, fine shells merged into stacked shells of >= 400 particles (baseline)",
              "T2": "B16 (electrons) / B12 (pressure) shape inside the hydrodynamic-resolution radius, renormalised per "
                    "halo to the measured N_e (Y) inside it; T1 outside",
              "T3": "raw fine shells (no merging; as Test H uses in 13.5-15.5)"}
MIN_STACK_COUNT = 400
PAINT_MIN_LOGM = float(np.log10(6.6e12))                 # HalfDome painting threshold (12.8195)

# ---- inputs
J0 = INPUTS / "j0_profiles_x200.npz"
J0_META = INPUTS / "j0_meta.json"
H_RESULTS = ROOT / "H_sensitivity" / "results"
F_PROFILES = ROOT / "F_observables" / ("results_smoke" if SMOKE else "results") / "profiles.npz"


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
pool_map = FC.pool_map


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        if keys is None:
            names = z.files
        elif callable(keys):
            names = [k for k in z.files if keys(k)]
        else:
            names = [k for k in keys if k in z.files]
        return {k: z[k] for k in names}


# ---- memory hygiene (as Test H after its attempt 2: release heap before every fork, log RSS)
MEM_LOG: list = []


def _status(key, pid="self", fname="status"):
    try:
        with open(f"/proc/{pid}/{fname}") as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def release():
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:                                   # noqa: BLE001
        pass


def mem(tag="", do_release=True):
    if do_release:
        release()
    import resource as _res
    rss = _status("VmRSS:") / 1024**2
    pss = _status("Pss:", fname="smaps_rollup") / 1024**2
    peak = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024**2
    log(f"mem {tag}: parent RSS {rss:.2f} GiB, PSS {pss:.2f} GiB, peak {peak:.2f} GiB")
    MEM_LOG.append({"tag": tag, "rss_GiB": rss, "pss_GiB": pss, "peak_GiB": peak, "t": time.time()})
    return rss


# ---- grid helpers (h_common.py)
def mids(edges=X200):
    e = np.asarray(edges, float)
    return 0.75 * (e[1:]**4 - e[:-1]**4) / (e[1:]**3 - e[:-1]**3)


def dlnx(edges=X200):
    e = np.asarray(edges, float)
    lo = np.where(e[:-1] > 0, e[:-1], 0.716 * e[1])
    return np.log(e[1:] / lo)


def edge_index(edges, value, tol=1e-9):
    k = int(np.argmin(np.abs(np.asarray(edges) - value)))
    assert abs(edges[k] - value) <= tol * max(1.0, value), (value, edges[k])
    return k


K_CUT = {c: edge_index(X200, c) for c in CUTS}           # shells kept inside the cut: 1R -> 50, 2R -> 60, 4R -> 67


def inside(edges, radius):
    return np.asarray(edges)[1:] <= radius + 1e-9


def interp_crossing(xr, cum, level):
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


def cut_profile(F, cut):
    """Zero a profile [.., 67 or 75] beyond the 3D radius ``cut`` (a fine edge)."""
    F = np.array(F, float, copy=True)
    F[..., K_CUT[cut]:] = 0.0
    return F


def cut_moment(S, cut):
    S = np.array(S, float, copy=True)
    k = K_CUT[cut]
    S[..., k:, :] = 0.0
    S[..., :, k:] = 0.0
    return S


def cut_moment_side(S, cut, side):
    """Cross moment <P_s E_t> with only the E side ('E', axis t) or the P side ('P', axis s) cut."""
    S = np.array(S, float, copy=True)
    k = K_CUT[cut]
    if side == "E":
        S[..., :, k:] = 0.0
    else:
        S[..., k:, :] = 0.0
    return S


def pad75(F):
    F = np.asarray(F, float)
    if F.shape[-1] == N_FINE:
        return F
    return np.pad(F, [(0, 0)] * (F.ndim - 1) + [(0, N_FINE - F.shape[-1])])


def n_seg_upto(radius):
    """Number of basis-grid segments inside ``radius`` (a break of f_kernels.basis_grid)."""
    _, grid, _, _ = FK.basis_grid()
    ends = grid.x[[b for _, b in grid.segments]]
    n = int(np.sum(ends <= radius * (1 + 1e-12)))
    assert abs(ends[n - 1] - radius) < 1e-9 * radius, (radius, ends[n - 1])
    return n


@contextlib.contextmanager
def fallback_cut(radius):
    """f_pdf deposits the published-fallback haloes over the first n_seg_4 basis segments (f_kernels.basis_grid's 4th
    value); this context returns the segment count of ``radius`` instead (the fallback haloes are then cut at that 3D
    radius), in this process and the workers it forks.  Nothing else in f_pdf uses n_seg_4."""
    original = FK.basis_grid
    n_cut = n_seg_upto(radius)

    def patched():
        base, grid, seg_shell, n_seg_4 = original()
        return base, grid, seg_shell, n_cut

    FK.basis_grid = patched
    try:
        yield n_cut
    finally:
        FK.basis_grid = original
