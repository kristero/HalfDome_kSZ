"""Shared, configuration-free definitions of Test J2 (fits of the halo-component definitions D0-D3 and of the newly
calibrated low-mass bins).  numpy only: imported by every J2 process (j2_inputs.py locally; j2_fits.py in Test G's
35-shell configuration and j2_eval.py in Test J's fine-grid configuration on maas; j2_post.py locally), which cannot
share one process because Test J patches Test F's grid on the imported module object.

Definitions of the halo electron component (Test J1, j_common.DEFS):
  D0  A2-ii gas to 4 R200c (the painting target of Tests H / I)
  D1  A2-ii gas inside the 3D sphere r <= R200c   (profile cut at x = 1 in 3D, then projected)
  D2  A2-ii gas inside r <= 2 R200c
  D3  gas bound (HBT) to the halo or its own satellites ('host_bound'), all radii to 4 R200c
Calibrated bins: the 14 engine bins 12.7-12.9 ... 15.3-15.5 (log10 M200c [Msun], 0.2 dex; configuration C14 of J1);
'low' bins = 12.7-13.5 (engine bins 1-4, calibrated for the first time in Test J).

Fit grid (as Test I): Test F's 35-shell grid to 4 R200c (every F edge is an X200 edge, so a 35-shell value is the
volume-weighted mean of the fine X200 shells, exactly).  Fit masks:
  bins >= 13.5   Test F / G / I strict resolution rule (G_physical_outer/results/f_base/profiles.npz base_mask, equal to
                 strict_mask in these bins): every halo of the bin resolves the shell
  bins 12.7-13.5 'S400' stack-level rule (replaces the per-halo strict rule, which trusts only 2.18 / 1.5 / 1.0 / 0.52
                 R200c outward there): an F shell is used if x_lo >= max(0.04, median 4 eps/R200c) (gravitational
                 softening) and the stack holds >= 400 gas particles of the selection in it (A2-ii counts for D0-D2
                 and pressure, host-bound counts for D3; 1/sqrt(400) = 5% Poisson error of the stacked particle sum)
  every bin, D3  additionally >= 400 stacked host-bound particles (never binds above 10^13.5 inside the strict shells)
  RC variant (bins 12.7-13.5 only): the 'resolved shape + core' grid: one shell [0, x_h] (x_h = hydrodynamic-resolution
                 radius of J1: inside it only the integral is measured) plus J1's merged >= 400-particle T1 shells
                 outside x_h; shells of the RC grid are fitted with their exact model volume averages.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
MODE = os.environ.get("J_MODE", "full")
SMOKE = MODE == "smoke"
J2_RESULTS = Path(os.environ.get("J2_RESULTS", str(HERE / ("results_j2_smoke" if SMOKE else "results_j2")))).resolve()
J1_RESULTS = HERE / ("results_smoke" if SMOKE else "results")
INPUTS = HERE / "inputs"
J2_INPUT = INPUTS / ("j2_fitdata_smoke.npz" if SMOKE else "j2_fitdata.npz")
J2_INPUT_META = INPUTS / ("j2_fitdata_smoke_meta.json" if SMOKE else "j2_fitdata_meta.json")
J1_FULL_RESULTS = HERE / "results"
# J1 truth used as the reference of every J2 painting (smoke: the J1 smoke truth re-run in <J2 results>/j1base)
J1_TRUTH = (J2_RESULTS / "j1base" / "truth_summary.json") if SMOKE else (J1_FULL_RESULTS / "truth_summary.json")
J0 = INPUTS / "j0_profiles_x200.npz"
G_PROFILES35 = ROOT / "G_physical_outer" / "results" / "f_base" / "profiles.npz"
G_PREPARE = ROOT / "G_physical_outer" / "results" / "prepare.npz"
I_RESULTS = ROOT / "I_fitting_a2ii" / "results"
SCRATCH_DEFAULT = "/tmp/nagainis_halfdome_testJ"

F35 = np.array([0., 0.025, 0.02932554, 0.0343995, 0.04035136, 0.04733302, 0.05552265, 0.06512928, 0.07639805,
                0.08961657, 0.10512219, 0.1233106, 0.14464601, 0.16967291, 0.19903, 0.23346651, 0.27386128,
                0.32124522, 0.37682761, 0.44202696, 0.51850721, 0.6082202, 0.71345548, 0.83689876, 1., 1.15155585,
                1.34, 1.5, 1.75, 2., 2.18026444, 2.5, 3., 3.30192725, 3.63424119, 4.])   # rounded; exact from G
N_ENGINE = 15
J_BINS = [3, 10] if SMOKE else list(range(1, 15))
LOW_BINS = [b for b in J_BINS if b < 5]
HIGH_BINS = [b for b in J_BINS if b >= 5]
N_J = len(J_BINS)
MIN_STACK_COUNT = 400
SOFTENING_FLOOR = 0.04
SEED = 20261015
LOGM_EDGES = np.round(np.arange(12.5, 15.51, 0.2), 1)


def bin_label(b):
    return f"{LOGM_EDGES[b]:.1f}-{LOGM_EDGES[b + 1]:.1f}"


J_LABELS = [bin_label(b) for b in J_BINS]

# ---- definitions: selection, boundary (3D cut = R_cut), fine basis set of J1, fit radii, ladder radii, outer options
DEFS = {
    "D0": dict(sel="a2ii", R=4.0, basis="JA", label="A2-ii gas to 4 R200c",
               rfit=(2.0, 3.0, 4.0), lad_rfit=(4.0,), outers=("O0", "O1", "O2")),
    "D1": dict(sel="a2ii", R=1.0, basis="JA", label="A2-ii gas, 3D sphere r <= R200c",
               rfit=(1.0, 1.34), lad_rfit=(1.0,), outers=("O0",)),
    "D2": dict(sel="a2ii", R=2.0, basis="JA", label="A2-ii gas, r <= 2 R200c",
               rfit=(1.34, 2.0, 3.0), lad_rfit=(2.0,), outers=("O0",)),
    "D3": dict(sel="hb", R=4.0, basis="JB", label="host-bound gas to 4 R200c",
               rfit=(1.34, 2.0, 3.0, 4.0), lad_rfit=(2.0, 4.0), outers=("O0", "O1", "O2")),
}
if SMOKE:
    DEFS["D0"]["rfit"] = (4.0,)
    DEFS["D2"]["rfit"] = (2.0,)
    DEFS["D3"]["rfit"] = (2.0, 4.0)
    DEFS["D3"]["lad_rfit"] = (4.0,)
OBJECTIVES = ("a", "d", "L")
SEL_ELL = (100, 300, 1000, 3000, 8000)
FAM_ELL = (100, 300, 1000, 3000)                   # family members (Test I: 100 <= ell <= 3000)
Q_NAMES = ("q16", "q50", "q84", "q975", "q999")
DM_FAMILY = ("dm_mean", "dm_var", "q16", "q50", "q84", "q975", "q999")


def dm_family(d):
    """DM family members of a definition.  D1's 16% quantile lies inside the no-crossing atom (34.7% of sightlines
    to z_s = 1 cross no R200c sphere): it equals the first grid cell for the truth and for every painted profile, so
    it carries no profile information and is left out of D1's family."""
    return tuple(k for k in DM_FAMILY if not (d == "D1" and k == "q16"))


IMP_DM = ("dm_mean", "dm_var", "dm_q16", "dm_q50", "dm_q84", "dm_q975", "dm_q999")


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: plain(v) for k, v in r.items()})


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        names = z.files if keys is None else [k for k in keys if k in z.files]
        return {k: z[k] for k in names}


def smoke_rows(mass_bin):
    """J1's smoke selection (j_prepare.select_rows): 25 haloes per engine bin, default_rng(SEED)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(N_ENGINE):
        r = np.flatnonzero(mass_bin == b)
        rows.append(np.sort(rng.choice(r, size=min(25, len(r)), replace=False)))
    return np.concatenate(rows)


# ---- grids
def vol(edges):
    e = np.asarray(edges, float)
    return 4 * np.pi / 3 * np.diff(e**3)


def edge_index(edges, values):
    e = np.asarray(edges, float)
    idx = np.array([int(np.argmin(np.abs(e - v))) for v in np.atleast_1d(values)])
    assert np.max(np.abs(e[idx] - np.atleast_1d(values))) < 1e-7, (values, e[idx])
    return idx


def rebin(values, fine_edges, idx):
    """Volume-weighted mean of fine shells over the coarse shells [idx[k], idx[k+1]) (fine-edge indices)."""
    v = vol(fine_edges)
    num = np.add.reduceat(np.asarray(values, float) * v, idx[:-1], axis=-1)
    den = np.add.reduceat(v, idx[:-1])
    return num / den


def inside(edges, radius):
    return np.asarray(edges)[1:] <= radius + 1e-9


def mids(edges):
    e = np.asarray(edges, float)
    return 0.75 * (e[1:]**4 - e[:-1]**4) / (e[1:]**3 - e[:-1]**3)


# ---- scores (Test I's formula, i_common.sens_score, on the stack's own shells)
def sens_score(d, w):
    """S = sqrt(sum w d^2 / sum w): importance-weighted RMS of d = ln(model/stack) (d = -1 for removed shells)."""
    W = np.asarray(w, float)
    return float(np.sqrt(np.sum(W * np.asarray(d)**2) / np.sum(W)))


def score_d(M, m, r_cut, edges):
    """d = ln(M/m) inside r_cut (shells with m > 0), -1 for shells beyond r_cut with m > 0, 0 where m <= 0."""
    ins = inside(edges, r_cut)
    ok = m > 0
    d = np.zeros_like(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(np.where(M > 0, M, np.nan) / np.where(ok, m, 1.0))
    good = ok & ins[None, :] & np.isfinite(lr)
    d[good] = lr[good]
    d[ok & ~ins[None, :]] = -1.0
    d[ok & ins[None, :] & ~np.isfinite(lr)] = -1.0
    return d


def family_rms(errs, keys):
    vals = [errs[k] for k in keys if k in errs and errs[k] is not None and np.isfinite(errs[k])]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else np.nan
