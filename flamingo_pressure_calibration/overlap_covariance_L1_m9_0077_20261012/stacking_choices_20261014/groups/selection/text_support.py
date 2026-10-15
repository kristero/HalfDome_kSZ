#!/usr/bin/env python
"""Group 'selection': supporting numbers for the report text (added after the 2026-10-14 verifier review).

Adds a 'text_support' block to results.json so that every corrected statement of the report can be cited as
results.json:text_support.<...>.  Run after compute_selection.py and summarize.py.  Uses the validated core with the
REFERENCE settings (inclusion-weighted mean, strict rule, n_boot = 200, the core's default per-bin seeds, design
strata), so all bootstrap replicates stay paired with those of compute_selection.py.

Blocks
  agn_kept_extremes        largest |kept/removed - 1| per field over strict-resolved SHELLS and over band medians
  agn_heated_location      where the recently AGN-heated pressure (Y) and electrons sit inside 4 R200c
  agn_share_innermost      heated share of the innermost shell vs the maximum over shells, per bin
  cold_sf_innermost        cold/SF share of the innermost shell, per bin and field
  impact_significance      |dO/O| / paired-bootstrap SD for every variant and observable
  clean_per_bin            per-bin contributions of the CLEAN-only impacts (first order, they add up)
  cb_xray_minus_central_bound   paired difference of the two central-bound variants (percentage points)
  soap_only_conversion     electrons from gas mass with a single global X_H (what SOAP-only data allow)
  A2ii_inside_1p5R         largest |band median| of the A2-ii partition inside 1.5 R200c, per field
  sample_classes           what the 'overlapper inside 1R' class contains; CLEAN counts per bin
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

R = json.loads((HERE / "results.json").read_text())
GRID, SCHEME, NB = "fit", "M200c_0.2", 200
RES = dict(min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)
EDGES = sc.edges(GRID)
LO, HI = EDGES[:-1], EDGES[1:]
X = 0.75 * (HI**4 - LO**4) / (HI**3 - LO**3)
VOL = sc.shell_volumes(GRID)
BINS = sc.mass_bins(SCHEME)
CAL = [b for b in BINS if b["lo"] >= 13.5 - 1e-9]
CAL_MAIN = [b["label"] for b in CAL if b["label"] != "15.3-15.5"]
TOP = "15.3-15.5"
BANDS = list(R["bands"])
MU_XGP = (1.0 / 1836.15267343 + 2.0 / 1.76) / 0.9          # XGPaint mu in m_p (build_cache.py convention)
TAB = sc.halo_table()
V = R["variants"]


def sig(x, n=4):
    return sc._sig(float(x), n)


def arr(a):
    return np.array([np.nan if v is None else v for v in a], float)


def mask(lab):
    return np.array(V["ref"]["shell_ratio"]["P"][lab]["resolved"], bool)


ts = {"note": "Supporting numbers for the report text; fractions are dimensionless unless the key ends in _pct. "
              "'cal' = the nine bins 13.5-15.3, 'top' = 15.3-15.5 (N = 6)."}
t0 = time.time()

# ------------------------------------------------------------------ (a) AGN kept: shells vs band medians
ext = {}
for f in ("P", "E", "G", "T_e", "K"):
    out = {}
    for name, labs in (("cal", CAL_MAIN), ("top", [TOP])):
        best_shell, best_band = (0.0, None, None), (0.0, None, None)
        for lab in labs:
            d = V["selection:agn_kept"]["shell_ratio"][f][lab]
            dev = arr(d["ratio"]) - 1
            m = np.array(d["resolved"], bool) & np.isfinite(dev)
            if m.any():
                i = np.flatnonzero(m)[np.argmax(np.abs(dev[m]))]
                if abs(dev[i]) > abs(best_shell[0]):
                    best_shell = (float(dev[i]), lab, float(X[i]))
            for bd in BANDS:
                br = V["selection:agn_kept"]["band_ratio"][f][lab][bd]
                if br and abs(br["dev"]) > abs(best_band[0]):
                    best_band = (float(br["dev"]), lab, bd)
        out[name] = {"max_abs_shell_dev": sig(best_shell[0]), "shell_bin": best_shell[1],
                     "shell_x": sig(best_shell[2]) if best_shell[2] is not None else None,
                     "max_abs_band_median_dev": sig(best_band[0]), "band_bin": best_band[1], "band": best_band[2]}
    ext[f] = out
ts["agn_kept_extremes"] = ext

# ------------------------------------------------------------------ (b) where the heated gas sits
REG = {"<0.1": (0.0, 0.1), "0.1-1": (0.1, 1.0), "1-2": (1.0, 2.0), "2-4": (2.0, 4.0)}
REG_M = {k: (LO >= lo - 1e-9) & (HI <= hi + 1e-9) for k, (lo, hi) in REG.items()}
loc = {}
for field in ("P", "E"):
    loc[field] = {}
    key = sc.make_key(field, agn="only", grid=GRID)
    for b in BINS:
        res = sc.stack_bin(key, b, statistic="mean", rule="strict", n_boot=NB, resolution_options=RES,
                           return_boot=True)
        v = np.nan_to_num(res["value"]) * VOL
        bt = np.nan_to_num(res["boot"]) * VOL[None, :]
        tot, totb = v.sum(), bt.sum(axis=1)
        ent = {}
        for k, m in REG_M.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                fb = bt[:, m].sum(axis=1) / totb
            ent[k] = {"fraction": sig(v[m].sum() / tot) if tot > 0 else None,
                      "err": sig(np.nanstd(fb, ddof=1)) if np.isfinite(fb).sum() > 1 else None}
        loc[field][b["label"]] = ent
ts["agn_heated_location"] = {
    "description": "fraction of the recently AGN-heated stacked Y(<4R200c) (P) or N_e(<4R200c) (E) that lies in "
                   "each radial region; err = SD over the 200 paired replicates",
    **loc}

# ------------------------------------------------------------------ (c) heated share: innermost shell vs max
inner = {}
for lab in R["agn_share_of_total"]["P"]:
    d = R["agn_share_of_total"]["P"][lab]
    sh = arr(d["shell"])
    m = np.array(d["resolved"], bool)
    inner[lab] = {"innermost_pct": sig(100 * sh[0]), "innermost_x": sig(X[0]),
                  "max_pct": sig(100 * np.nanmax(sh)), "max_x": sig(X[np.nanargmax(sh)]),
                  "max_strict_shell_pct": sig(100 * np.nanmax(sh[m])) if m.any() else None,
                  "max_strict_shell_x": sig(X[np.flatnonzero(m)[np.nanargmax(sh[m])]]) if m.any() else None}
ts["agn_share_innermost_P"] = inner

# ------------------------------------------------------------------ (d) cold/SF share of the innermost shell
ts["cold_sf_innermost_pct"] = {f: {lab: sig(100 * arr(R["cold_sf_share_of_total"][f][lab]["shell"])[0])
                                   for lab in R["cold_sf_share_of_total"][f]} for f in ("P", "E", "G")}

# ------------------------------------------------------------------ (e) significance of every impact
sigt = {}
for vid, v in V.items():
    if vid == "ref" or not v["impact"]:
        continue
    sigt[vid] = {}
    for f, d in v["impact"].items():
        for o, x in d.items():
            sigt[vid][o] = {"value_pct": x["dO_over_O_pct"], "err_pct": x["err_pct"],
                            "sigma": sig(abs(x["dO_over_O_pct"]) / x["err_pct"], 3) if x["err_pct"] else None}
ts["impact_significance"] = sigt

# ------------------------------------------------------------------ (f) CLEAN per-bin contributions
cl = {}
for f, obs in (("P", ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000")),
               ("E", ("kSZ_1h_ell1000", "DM_mean"))):
    for o in obs:
        pb = arr(V["selection:clean"]["impact"][f][o]["per_bin_pct"])
        cl[o] = {"per_bin_pct": dict(zip([b["label"] for b in CAL], [sig(x) for x in pb])),
                 "total_pct": sig(pb.sum()), "without_top_bin_pct": sig(pb[:-1].sum()),
                 "top_bin_pct": sig(pb[-1]),
                 "N_clean_top_bin": R["subset_composition"]["clean"][TOP]["N"]}
pbk = arr(V["selection:exc_all_k1"]["impact"]["P"]["tSZ_yy_total_ell1000"]["per_bin_pct"])
ts["clean_per_bin"] = cl
ts["exc_all_k1_tSZ1000_per_bin"] = {"total_pct": sig(pbk.sum()), "top_bin_pct": sig(pbk[-1]),
                                    "without_top_bin_pct": sig(pbk[:-1].sum())}

# ------------------------------------------------------------------ (g) paired difference cb_xray - central_bound
OBS = {"P": ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000"],
       "E": ["kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var", "DM_q16"]}


def stacks(key):
    vals, boots = [], []
    for b in CAL:
        r = sc.stack_bin(key, b, statistic="mean", rule="strict", n_boot=NB, resolution_options=RES,
                         return_boot=True)
        vals.append(r["value"])
        boots.append(r["boot"])
    return np.stack(vals), np.stack(boots, axis=1)          # [10, S], [nb, 10, S]


diff = {}
for f in ("P", "E"):
    Fr, Frb = stacks(sc.make_key(f, grid=GRID))
    Fa, Fab = stacks(sc.make_key(f, selection="central_bound", grid=GRID))
    Fc, Fcb = stacks(sc.make_key(f, selection="central_bound", phase="xray", grid=GRID))
    ia, ic = sc.impact_from_stacks(Fa, Fr, f), sc.impact_from_stacks(Fc, Fr, f)
    ba = [sc.impact_from_stacks(Fab[k], Frb[k], f) for k in range(NB)]
    bc = [sc.impact_from_stacks(Fcb[k], Frb[k], f) for k in range(NB)]
    for o in OBS[f]:
        d = 100 * (ic[o]["dO_over_O"] - ia[o]["dO_over_O"])
        db = 100 * np.array([c[o]["dO_over_O"] - a[o]["dO_over_O"] for a, c in zip(ba, bc)])
        db = db[np.isfinite(db)]
        diff[o] = {"central_bound_pct": sig(100 * ia[o]["dO_over_O"]), "cb_xray_pct": sig(100 * ic[o]["dO_over_O"]),
                   "difference_pp": sig(d), "paired_err_pp": sig(np.std(db, ddof=1)),
                   "sigma": sig(abs(d) / np.std(db, ddof=1), 3) if np.std(db, ddof=1) > 0 else None}
ts["cb_xray_minus_central_bound"] = diff

# ------------------------------------------------------------------ (h) SOAP-only electron conversion
# E units n_e mu_XGP/(rho_c f_b), G units rho/(rho_c f_b).  Full ionisation with ONE global hydrogen fraction X:
# n_e = rho (1+X)/(2 m_p)  ->  E_X = G * MU_XGP * (1+X)/2 exactly, shell by shell (the stack is linear).
# The stacked ratios G/E are read from the e_gas_muxgp variant (ratio of the G and E stacks).
vg = V["selection:e_gas_muxgp"]
ann = list(R["annuli"])


def conv(Xh):
    c = MU_XGP * (1 + Xh) / 2
    out = {"factor_vs_gas_over_muXGP": sig(c, 6), "integral_dev_pct": {}, "band_dev_pct": {}}
    for a in ("0-1", "0-2", "0-4"):
        vals = {lab: 100 * (c * (1 + vg["integral_ratio"]["E"][lab][a]["dev"]) - 1) for lab in CAL_MAIN + [TOP]}
        out["integral_dev_pct"][a] = {"cal": [sig(min(vals[l] for l in CAL_MAIN)), sig(max(vals[l] for l in CAL_MAIN))],
                                      "top": sig(vals[TOP])}
    for bd in BANDS:
        vals = [100 * (c * (1 + vg["band_ratio"]["E"][lab][bd]["dev"]) - 1)
                for lab in CAL_MAIN if vg["band_ratio"]["E"][lab][bd]]
        out["band_dev_pct"][bd] = [sig(min(vals)), sig(max(vals))] if vals else None
    return out


ratio4 = np.array([1 + vg["integral_ratio"]["E"][lab]["0-4"]["dev"] for lab in CAL_MAIN])
x_null = 2.0 / (MU_XGP * ratio4) - 1.0                       # X that nulls N_e(<4R) per bin
x_glob = float(np.mean(x_null))
fullion = V["selection:e_fullion"]["integral_ratio"]["E"]
ts["soap_only_conversion"] = {
    "description": "electrons from gas mass assuming full H+He ionisation with one global hydrogen mass fraction "
                   "X (SOAP stores no hydrogen mass or electron number): E_X = G x mu_XGP x (1+X)/2, exact at the "
                   "stack level; deviations from SWIFT ElectronNumberDensities in %",
    "mu_XGP_mp": sig(MU_XGP, 7),
    "X_0.76": conv(0.76),
    "X_0.745": conv(0.745),
    "X_null_Ne4R_per_bin": dict(zip(CAL_MAIN, [sig(x) for x in x_null])),
    "X_null_Ne4R_mean_cal": sig(x_glob),
    "X_null_mean_applied": conv(x_glob),
    "per_particle_fullion_Ne4R_dev_pct": [sig(100 * min(fullion[l]["0-4"]["dev"] for l in CAL_MAIN)),
                                          sig(100 * max(fullion[l]["0-4"]["dev"] for l in CAL_MAIN))]}
# effective mass per electron and the implied H fraction (full ionisation, no metals) in the hot 1-1.5 R200c band
r15 = np.array([1 + vg["band_ratio"]["E"][lab]["1-1.5"]["dev"] for lab in CAL_MAIN])
ts["soap_only_conversion"]["band_1-1.5"] = {
    "gas_over_E_ratio": [sig(r15.min()), sig(r15.max())],
    "mass_per_electron_mp": [sig(MU_XGP * r15.min()), sig(MU_XGP * r15.max())],
    "implied_X_if_fully_ionised": [sig(2 / (MU_XGP * r15.max()) - 1), sig(2 / (MU_XGP * r15.min()) - 1)]}
# halo-shells with zero SWIFT electrons but non-zero full-ionisation electrons, and how many are strict-resolved
E0 = np.asarray(sc.per_halo(sc.make_key("E", grid=GRID)))
EF = np.asarray(sc.per_halo(sc.make_key("E_fullion", grid=GRID)))
zero = (E0 == 0) & (EF > 0)
n_res_cal = 0
for b in CAL:
    n_res_cal += int(zero[b["rows"]][:, mask(b["label"])].sum())
ts["electron_zero_shells"] = {"E_zero_but_fullion_positive": int(zero.sum()),
                              "in_strict_resolved_shells_of_calibrated_bins": n_res_cal}

# ------------------------------------------------------------------ (i) A2-ii inside 1.5 R200c
a2 = {}
for f in ("P", "E", "G"):
    best = (0.0, None, None)
    for lab in CAL_MAIN + [TOP]:
        for bd in ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5"):
            br = V["selection:A2ii"]["band_ratio"][f][lab][bd]
            if br and abs(br["dev"]) > abs(best[0]):
                best = (br["dev"], lab, bd)
    a2[f] = {"max_abs_band_dev_pct": sig(100 * best[0]), "bin": best[1], "band": best[2]}
ts["A2ii_inside_1p5R"] = a2

# ------------------------------------------------------------------ (j) sample classes
cm, cn = TAB["class_main"], TAB["class_neighbours_mu0p1"]
in1 = cm == 1
u, c = np.unique(cn[in1], return_counts=True)
ts["sample_classes"] = {
    "class_main_counts": {int(k): int(n) for k, n in zip(*np.unique(cm, return_counts=True))},
    "class_main1_by_class_neighbours_mu0p1": {int(k): int(n) for k, n in zip(u, c)},
    "note": "class_neighbours_mu0p1 = 0: no neighbouring (non-satellite) overlapper >= 0.1 M200c inside 2 R200c",
    "clean_N_weight_share": {lab: [d["N"], d["weight_share"]] for lab, d in R["subset_composition"]["clean"].items()}}

ts["runtime_s"] = round(time.time() - t0, 1)
R["text_support"] = ts
(HERE / "results.json").write_text(json.dumps(R, separators=(",", ":"), allow_nan=False))

if __name__ == "__main__":
    print(json.dumps({k: v for k, v in ts.items() if k not in ("impact_significance",)}, indent=1))
    print("impact significance (sigma) of selected entries:")
    for vid in ("selection:clean", "selection:agn_kept", "selection:A2ii", "selection:exc_all_k1"):
        print(vid, {o: (d["value_pct"], d["err_pct"], d["sigma"]) for o, d in sigt[vid].items()
                    if o in ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "kSZ_1h_ell1000",
                             "DM_mean", "DM_var")})
