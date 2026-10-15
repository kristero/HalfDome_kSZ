"""Shared configuration of Test G: physically motivated outer components, calibrated per field.

Test G re-uses Test F's machinery unchanged (read-only imports of F_observables/f_*.py, the analysis library
and kernels/engine.py).  F's modules are configured through their environment variables BEFORE import:
  F_MODE    = G's mode ('full' or 'smoke')
  F_RESULTS = <G results>/f_base   (F's stages, when G runs them, write only there; F's work/ = f_base/work)
  F_WORKERS = G's worker count
so nothing is ever written into F_observables/.

Environment (set by run_test_g.py):
  G_MODE     'full' (default) or 'smoke'
  G_RESULTS  output directory (default G_physical_outer/results, or results_smoke)
  G_WORKERS  worker processes (default 4; smoke 1)
  STUDY_ROOT the study directory (default: the parent of this file's directory)

Units and normalisations (verified against the data and code in the prepare stage, prepare.json):
  pressure  P = P_e / P200,  P200 = G M200 (200 rho_c) f_b / (2 R200)            (radii234 pilot_physics.p200)
  electrons E = n_e mu_XGP / (rho_c f_b), mu_XGP = (m_e + m_p 2/(1+0.76)) / 0.9 = 1.2632 m_p (XGPaint)
  gas       G = rho_gas / (rho_c f_b)  ->  cosmic mean gas density = Omega_b / f_b = Omega_m (0.306)
  T200      k T200 = 0.59 m_p G M200 / (2 R200)                                   (Test B halo_table)
  => P/P200 = KAPPA_P (E/200)(T/T200) with KAPPA_P = 0.59 m_p / mu_XGP = 0.4671 (electron-weighted T), and the
     cosmic mean electron density in E units is KAPPA_E Omega_m with KAPPA_E = n_e mu_XGP / rho_gas of
     ionised gas (1.107 for X = 0.752; measured 1.09-1.10 in the 2-4 R200c shells), not Omega_m.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

MODE = os.environ.get("G_MODE", "full")
if MODE not in ("full", "smoke"):
    raise ValueError(f"G_MODE must be 'full' or 'smoke', not {MODE!r}")
SMOKE = MODE == "smoke"
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
RESULTS = Path(os.environ.get("G_RESULTS", str(HERE / ("results_smoke" if SMOKE else "results")))).resolve()
FBASE = RESULTS / "f_base"
TABLES = RESULTS / "tables"
WORK = RESULTS / "work"
WORKERS = int(os.environ.get("G_WORKERS", "1" if SMOKE else "4"))
for _d in (RESULTS, FBASE, TABLES, WORK):
    _d.mkdir(parents=True, exist_ok=True)

# ---- configure and import Test F (read-only modules; outputs to f_base)
os.environ["F_MODE"] = MODE
os.environ["F_RESULTS"] = str(FBASE)
os.environ["F_WORKERS"] = str(WORKERS)
os.environ["STUDY_ROOT"] = str(ROOT)
_F_DIR = ROOT / "F_observables"
if str(_F_DIR) not in sys.path:
    sys.path.insert(0, str(_F_DIR))
import f_common as FC  # noqa: E402
import f_fit as FF  # noqa: E402

assert Path(FC.RESULTS) == FBASE, (FC.RESULTS, FBASE)

# ---- inputs (read-only)
F_RESULTS_DIR = ROOT / "F_observables" / "results"
F_INPUTS = ("profiles.npz", "truth.npz", "truth_summary.json", "importance_F.npz", "models_main.npz")
B_PROFILES = ROOT / "B_covariance" / "results" / "profiles_fit.npz"
B_HALOS = ROOT / "B_covariance" / "results" / "halo_table.npz"
XI_BIAS = HERE / "inputs" / "xi_bias_z0.npz"
PLAN = ROOT / "metadata_plan.json"

# ---- grids (F's)
F_EDGES = FC.F_EDGES
BASIS_EDGES = FC.BASIS_EDGES
N_F, N_B = FC.N_F, FC.N_B
N_CAL = FC.N_CAL
CAL_BINS = FC.CAL_BINS
CAL_LABELS = FC.CAL_LABELS

# ---- Test G configuration
R_FIT = (2.0, 4.0) if SMOKE else (1.0, 1.5, 2.0, 3.0, 4.0)
R_CUT = (2.0, 4.0) if SMOKE else (1.0, 1.5, 2.0, 3.0, 4.0)
OBJECTIVES = ("a", "d")
FIT_TARGETS = ("main", "a2ii")              # all gas (the physics) and the A2-ii stacks (direct fit)
# evaluation targets: (name, fit target, painting factor, truth selection)
EVALS = (("all", "main", False, "main"),      # all-gas model vs the all-gas truth
         ("paint", "main", True, "a2ii"),     # all-gas model x p_keep vs the A2-ii (painting) truth
         ("a2ii", "a2ii", False, "a2ii"))     # model fitted to the A2-ii stacks vs the A2-ii truth
SCATTER = ("det", "S1", "S2")
S2_CUTS = tuple(r for r in R_CUT if r >= 2.0)  # S2 needs an outer aperture 1.5 R .. R_cut
GH1 = 9                                       # Gauss-Hermite nodes, S1 (as F)
GH2 = 4 if SMOKE else 5                       # per dimension, S2 (correlated 2-D)
SEED = 20261013
BOOT_DRAWS = FC.BOOT_DRAWS
M_PIVOT = 10**14.5                            # Msun: pivot of b(M) and of the mass-dependent outer temperature
PDF_GROUP = 64                                # single-entry histograms per PDF pass job group (memory bound)

# ---- physical constants of the normalisation (code constants, verified against the data in 'prepare')
_P = FF.FAM.P                                  # radii234 pilot_physics (read-only)
MU_XGP_OVER_MP = float(_P.MU_XGP_KG / _P.proton_mass)
KAPPA_P = float(0.59 * _P.proton_mass / _P.MU_XGP_KG)
X_H_PRIMORDIAL = 0.752
KAPPA_E_THEORY = float((1 + X_H_PRIMORDIAL) / 2 * MU_XGP_OVER_MP)
_plan = json.loads(PLAN.read_text())
OMEGA_B = float(np.atleast_1d(_plan["cosmology"]["Omega_b"])[0])
F_B = float(_plan["baryon_fraction"])
OMEGA_M = OMEGA_B / F_B                        # = 0.306 (total matter incl. neutrinos; f_b = Omega_b / Omega_m)
H_HUBBLE = float(np.atleast_1d(_plan["cosmology"]["h"])[0])


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


def x_hi_le(edges, radius):
    """Shells of ``edges`` lying entirely inside ``radius`` (an edge)."""
    return np.asarray(edges)[1:] <= radius + 1e-9


def region_masks(edges=F_EDGES):
    out = {}
    for name, a, b in FC.REGIONS:
        out[name] = (np.asarray(edges)[:-1] >= a - 1e-12) & (np.asarray(edges)[1:] <= b + 1e-12)
    return out


def load_prepare():
    z = np.load(RESULTS / "prepare.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def fit_key(field, model, target, objective, r_fit):
    return f"{field}|{model}|{target}|{objective}|{r_fit:g}"


def parse_key(key):
    f, m, t, o, r = key.split("|")
    return f, m, t, o, float(r)
