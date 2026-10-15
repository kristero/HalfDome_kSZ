"""Shared, configuration-free helpers of Test I (fitting on the A2-ii painting stacks).  numpy only: imported by both
stage processes (i_g35: Test G's configuration on Test F's 35-shell grid; i_fine: Test H's configuration on the
75-shell fine grid), which cannot share one process because H patches Test F's grid on the imported module object.

Paths (environment, set by run_test_i.py):
  I_MODE      'full' or 'smoke'
  I_RESULTS   Test I results directory (default I_fitting_a2ii/results or results_smoke)
  I_H_RESULTS Test H results directory read (default H_sensitivity/results or results_smoke)
  STUDY_ROOT  the study directory

Definitions used throughout (x = r/R200c; c = calibrated bin 13.5-15.5 in log10 M200c [Msun], 0.2 dex; s = shell):
  importance        I_O[b, s] = (dO/d ln F_{b,s}) / (k_O O)  (Test H, A2-ii target; k_O = 2 for quadratic
                    observables, 1 for linear ones; sum over all bins and shells = 1)
  family weight     w_fam[c, s] = mean over the family's observables O (and multipoles 100 <= ell <= 3000) of
                    |I_O[c, s]|, calibrated bins only:
                      tSZ    yytot (pressure);  kSZ  ksz1h (electrons);  DM  dm_mean, dm_var, dm_q16 ... dm_q999;
                      tSZxDM ydmtot_P (pressure side) / ydmtot_E (electron side)
  sensitive-part score  S_fam = sqrt( sum_{c,s} w_fam[c,s] d[c,s]^2 / sum_{c,s} w_fam[c,s] ),
                    d[c,s] = ln(M[c,s] / m[c,s]) for shells inside R_cut (M = the painted model's shell average,
                    m = the A2-ii stacked mean, both on the fine 67-shell grid), d = -1 for shells beyond R_cut
                    (a removed shell is a -100% profile error; ln would diverge); shells with m <= 0 are skipped.
                    Dimensionless; 0.05 = a 5% RMS profile error where the family's signal comes from.
  linear response   L_O = k_O sum_{c,s} I_O[c,s] (M[c,s]/m[c,s] - 1): the first-order fractional change of O
                    (calibrated bins; a diagnostic next to the exact in-basis evaluation).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STUDY_ROOT", str(HERE.parent))).resolve()
MODE = os.environ.get("I_MODE", "full")
SMOKE = MODE == "smoke"
RESULTS = Path(os.environ.get("I_RESULTS", str(HERE / ("results_smoke" if SMOKE else "results")))).resolve()
H_RESULTS = Path(os.environ.get("I_H_RESULTS", str(ROOT / "H_sensitivity" / ("results_smoke" if SMOKE else "results"))))
H_RESULTS = H_RESULTS.resolve()
TABLES = RESULTS / "tables"
WORK = RESULTS / "work"
INPUTS = HERE / "inputs"
VARIANTS_NPZ = INPUTS / "i0_variants.npz"
for _d in (RESULTS, TABLES, WORK):
    _d.mkdir(parents=True, exist_ok=True)

R_FIT_I1 = (1.0, 1.34, 1.5, 2.0, 3.0, 4.0)
R_CUT_I1 = (1.0, 1.5, 2.0, 3.0, 4.0)
R_FIT_I5 = (2.0, 4.0)
OBJECTIVES = ("a", "d", "L")
VARIANTS = ("a2ii", "a2ii_agn", "allgas", "a2i", "unweighted", "median", "m05", "perhalo", "xray")
FAM_OBS = {"tSZ": ("yytot",), "kSZ": ("ksz1h",), "DM": ("dm_mean", "dm_var", "dm_q16", "dm_q50", "dm_q84",
                                                          "dm_q975", "dm_q999"),
           "tSZxDM_P": ("ydmtot_P",), "tSZxDM_E": ("ydmtot_E",)}
FIELD_FAMS = {"P": ("tSZ", "tSZxDM_P"), "E": ("kSZ", "DM", "tSZxDM_E")}
ELL_RANGE = (100.0, 3000.0)


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(path, obj):
    Path(path).write_text(json.dumps(plain(obj), indent=1) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    import csv
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


def family_weights(sens, cal_bins, n_x=67):
    """w_fam[c, s] (calibrated bins, fine shells to 4 R200c) from Test H's A2-ii importance maps (sens.npz dict)."""
    ell = np.asarray(sens["ell"], float)
    ok_ell = (ell >= ELL_RANGE[0] - 1e-9) & (ell <= ELL_RANGE[1] + 1e-9)
    out = {}
    for fam, names in FAM_OBS.items():
        acc = []
        for nm in names:
            M = np.abs(np.asarray(sens[f"imp__a2ii__{nm}"], float))[list(cal_bins)][:, :n_x]
            if M.ndim == 3:
                M = M[..., ok_ell].mean(axis=-1)
            acc.append(M)
        out[fam] = np.mean(acc, axis=0)
    return out


def importance_maps(sens, cal_bins, n_x=67):
    """Signed importance I_O[c, s(, ell)] of the calibrated bins for the observables used in the linear response."""
    keys = ["yytot", "ksz1h", "dm_mean", "dm_var", "ydmtot_P", "ydmtot_E"] + [f"dm_{q}" for q in
                                                                                ("q16", "q50", "q84", "q975", "q999")]
    return {k: np.asarray(sens[f"imp__a2ii__{k}"], float)[list(cal_bins)][:, :n_x] for k in keys}


def sens_score(M, m, w, x_edges, r_cut):
    """Importance-weighted RMS of ln(M/m) (see module docstring); M, m, w [c, 67]."""
    inside = np.asarray(x_edges)[1:] <= r_cut + 1e-9
    ok = m > 0
    d = np.zeros_like(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(M / np.where(ok, m, 1.0))
    good_in = ok & inside[None, :] & (M > 0)
    d[good_in] = lr[good_in]
    d[ok & ~inside[None, :]] = -1.0
    bad = ok & inside[None, :] & ~(M > 0)
    d[bad] = -1.0
    W = np.where(ok, w, 0.0)
    return float(np.sqrt(np.sum(W * d**2) / np.sum(W))), float(np.sum(W[:, ~inside]) / np.sum(W))


def score_regions(M, m, w, x_edges, r_cut, regions=((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))):
    """Weighted mean of d = ln(M/m) (signed) and the weight share per radial region (where each model fails)."""
    e = np.asarray(x_edges)
    inside = e[1:] <= r_cut + 1e-9
    ok = m > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(ok & (M > 0), np.log(M / np.where(ok, m, 1.0)), -1.0)
    d = np.where(inside[None, :], d, -1.0)
    W = np.where(ok, w, 0.0)
    out = {}
    for a, b in regions:
        sel = (e[:-1] >= a - 1e-9) & (e[1:] <= b + 1e-9)
        ws = W[:, sel].sum()
        out[f"{a:g}-{b:g}"] = {"weight_share": float(ws / W.sum()),
                               "mean_d": float(np.sum(W[:, sel] * d[:, sel]) / ws) if ws > 0 else None,
                               "rms_d": float(np.sqrt(np.sum(W[:, sel] * d[:, sel]**2) / ws)) if ws > 0 else None}
    return out


def linear_response(M, m, I, k):
    ok = m > 0
    delta = np.where(ok, M / np.where(ok, m, 1.0) - 1, 0.0)
    if I.ndim == 3:
        return k * np.einsum("cs,csl->l", delta, I)
    return float(k * np.sum(I * delta))
