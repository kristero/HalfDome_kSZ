"""Third set of checks behind SECTION.md of group 'resolution' (final text corrections, 2026-10-14).

Recomputes, from results.json, work/arrays.npz and the core cache (plain numpy; stackcore only for the bins), the
numbers that the verifier's final pass flagged and the related numbers the corrected text quotes:
  * h-alone (factor 1) vs count-alone (>= 20) first edges and their gap in shells;
  * the two h-passing shells of 12.5-12.7 and which haloes fail the count there;
  * P / extrapolation above / below 1 per rejected shell, recent-AGN effect and deficit closed, all calibrated bins;
  * hot / cold shares of the n_e excess (central sphere and volume-integrated), incl. 15.3-15.5;
  * resolved weight fraction and median x_in / h in the rejected shells of 14.9-15.1;
  * per-halo band medians and the partially resolved shells inside bands >= 0.25 R200c;
  * largest |dO/O| per setting over the ten observables (2R and 4R fits vs the reference fit);
  * the L2p8 binding counts under strict and frac95;
  * contiguity of the >= 500 masks.
Writes section_checks_final.json next to this script.  Run:
  LD_LIBRARY_PATH=<env>/lib <env>/bin/python -B section_checks_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent
OUT = G.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

R = json.loads((G / "results.json").read_text())
PS = R["per_setting"]
A = np.load(G / "work" / "arrays.npz")
E = np.load(OUT / "cache" / "edges_fit.npy")
XIN = E[:-1]
C = np.load(OUT / "cache" / "counts_fit.npy")
H = np.load(OUT / "cache" / "hsml_fit.npy")
BINS = {b["label"]: b for b in sc.mass_bins("M200c_0.2")}
LABS = list(BINS)
CAL = LABS[5:]
OBS = ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "tSZ_yy_1h_ell3000", "y_mean",
       "kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var"]
XV = 0.75 * (E[1:] ** 4 - E[:-1] ** 4) / (E[1:] ** 3 - E[:-1] ** 3)
VOL = 4 * np.pi / 3 * np.diff(E ** 3)
out = {}


def first(mask):
    i = np.flatnonzero(mask)
    return (round(float(E[i[0]]), 4), int(i[0])) if len(i) else (None, None)


# ------------------------------------------------------------------ grid
out["grid_edges"] = [round(float(v), 4) for v in E]
out["grid_dex"] = [round(float(v), 4) for v in np.log10(E[2:] / E[1:-1])]

# ------------------------------------------------------------------ h-alone (factor 1) vs count-alone (>= 20)
gap = {}
for lab in LABS:
    rows = BINS[lab]["rows"]
    with np.errstate(invalid="ignore"):
        h1 = (XIN[None, :] >= 1.0 * H[rows]).all(0)
    cnt = (C[rows] >= 20).all(0)
    (eh, ih), (ec, ic) = first(h1), first(cnt)
    gap[lab] = {"h1_alone": eh, "count20_alone": ec,
                "count_edge_shells_outside_h1": None if ih is None or ic is None else ic - ih,
                "hf1_eq_hf0_mask": PS["resolution:hf1"]["masks"][lab]["kept"] == PS["resolution:hf0"]["masks"][lab]["kept"]}
out["h1_vs_count20"] = gap
d = [gap[l]["count_edge_shells_outside_h1"] for l in CAL[:-2]]
out["h1_vs_count20_gap_range_13.5_15.1"] = [min(d), max(d)]

# ------------------------------------------------------------------ 12.5-12.7
rows = BINS["12.5-12.7"]["rows"]
with np.errstate(invalid="ignore"):
    h2 = (XIN[None, :] >= 2.0 * H[rows]).all(0)
info = {}
for i in range(31, 35):
    low = np.flatnonzero(C[rows, i] < 20)
    info[f"{E[i]:.3f}-{E[i + 1]:.3f}"] = {"h_pass": bool(h2[i]), "n_haloes_lt20": int(len(low)),
                                          "min_count": int(C[rows, i].min()),
                                          "halo_rows_lt20": [int(rows[j]) for j in low]}
out["bin_12.5-12.7"] = info

# ------------------------------------------------------------------ P / extrapolation and recent-AGN gas
ID = R["inner_decomposition"]
pinfo = {}
for lab in CAL:
    d = ID[lab]
    kept = np.array(d["kept_reference"], bool)
    rej = np.flatnonzero(~kept)
    p = np.array(d["P_over_ref2R_extrapolation"], float)
    k = np.array(d["P_agn_kept_over_removed"], float)
    below = [i for i in rej if p[i] < 1]
    above = [i for i in rej if p[i] >= 1]
    pinfo[lab] = {
        "central_P_over_extrap": float(p[0]), "central_agn_kept_over_removed": float(k[0]),
        "central_deficit_closed": None if p[0] >= 1 else float(p[0] * (k[0] - 1) / (1 - p[0])),
        "max_agn_effect_outside_centre": float(max(k[i] for i in rej if i > 0)),
        "below_shells": [[round(float(E[i]), 4), round(float(E[i + 1]), 4), round(float(p[i]), 4)] for i in below],
        "above_shells": [[round(float(E[i]), 4), round(float(E[i + 1]), 4), round(float(p[i]), 4)] for i in above],
        "x0.027": {"P": float(p[1]), "agn": float(k[1]),
                   "deficit": float(1 - p[1]), "closed": float(p[1] * (k[1] - 1) / (1 - p[1])) if p[1] < 1 else None},
        "x0.052": {"P": float(p[5]), "agn": float(k[5])} if 5 in rej else None,
        "P_xray_x0.027_0.052": [float(ID[lab]["P_xray_over_ref2R_extrapolation"][1]),
                                float(ID[lab]["P_xray_over_ref2R_extrapolation"][5])]}
out["P_inner"] = pinfo

# ------------------------------------------------------------------ n_e: hot / cold shares
ne = {}
for lab in CAL:
    Ea = A[f"stack|ref|E|{lab}|value"]
    Ex = A[f"stack|ref_E_xray|{lab}|value"]
    m = A[f"fit|ref|E|2|{lab}|model"]
    kept = np.asarray(PS["ref"]["masks"][lab]["kept"], bool)
    sel = ~kept & (E[1:] <= 1 + 1e-12) & (Ea > m)
    den = ((Ea - m) * VOL)[sel].sum()
    ne[lab] = {"hot_share_int": float(((Ex - m) * VOL)[sel].sum() / den) if den > 0 else None,
               "n_shells_int": int(sel.sum()),
               "central_hot_share_of_excess": float((Ex[0] - m[0]) / (Ea[0] - m[0])) if Ea[0] > m[0] else None,
               "central_cold_share_of_ne": float(ID[lab]["E_cold_sf_share"][0]),
               "x0.027_cold_share_of_ne": float(ID[lab]["E_cold_sf_share"][1]),
               "hot_ne_over_extrap_x0.027_0.052": [float(ID[lab]["E_xray_over_ref2R_extrapolation"][1]),
                                                   float(ID[lab]["E_xray_over_ref2R_extrapolation"][5])]}
out["ne_inner"] = ne

# ------------------------------------------------------------------ 14.9-15.1 rejected shells: x_in / h, resolved fraction
rf = R["resolved_fraction_per_shell"]["ref"]["14.9-15.1"]
out["rejected_14.9-15.1"] = {"resolved_weight_fraction": rf[:6], "xh_median": ID["14.9-15.1"]["xh_median"][:6],
                             "count_median": ID["14.9-15.1"]["count_median"][:6]}

# ------------------------------------------------------------------ per-halo bands
sbr = PS["resolution:per_halo"]["stack_band_ratios"]
out["per_halo_bands_cal"] = {f: {lab: {b: sbr[f][lab][b] for b in ("0-0.1", "0.1-0.25")} for lab in CAL}
                             for f in ("P", "E")}
expl = {f: [(lab, b, sbr[f][lab][b]["ratio"]) for lab in LABS[:5] for b in ("0.25-0.5", "0.5-1")
            if sbr[f][lab][b]["ratio"] is not None] for f in ("P", "E")}
out["per_halo_exploratory_0.25_1"] = {f: {"min": min(v, key=lambda t: t[2]), "max": max(v, key=lambda t: t[2])}
                                      for f, v in expl.items()}
part = {}
for lab in CAL:
    rfl = R["resolved_fraction_per_shell"]["ref"][lab]
    for f in ("P", "E"):
        rat = A[f"stack|per_halo|{f}|{lab}|value"] / A[f"stack|ref|{f}|{lab}|value"]
        sel = [i for i in range(35) if 0.1 <= XV[i] < 0.25 and 0 < rfl[i] < 1]
        if sel:
            part[f"{lab}|{f}"] = [[round(float(E[i]), 4), round(float(E[i + 1]), 4), round(float(rat[i]), 4),
                                   rfl[i]] for i in sel]
out["per_halo_partial_shells_0.1_0.25"] = part
out["per_halo_cal_ge025_all_one"] = {
    f: all(sbr[f][lab][b]["ratio"] in (None, 1.0) for lab in CAL
           for b in ("0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")) for f in ("P", "E", "G", "T_e", "K")}

# ------------------------------------------------------------------ largest |dO/O| per setting
mx = {}
for s in ("resolution:hf1", "resolution:mc200", "resolution:frac95", "resolution:per_halo",
          "resolution:per_halo_f50", "resolution:none", "resolution:diagnostic"):
    row = {}
    for tag in ("impact_fit4R_vs_ref_fit", "impact_fit2R_vs_ref_fit"):
        dd = PS[s]["summary"][tag]
        o = max(OBS, key=lambda q: abs(dd[q]["dO_over_O"]))
        row[tag] = {"observable": o, "percent": round(100 * dd[o]["dO_over_O"], 3),
                    "err": round(100 * dd[o]["err"], 3),
                    "range": [round(100 * min(dd[q]["dO_over_O"] for q in OBS), 3),
                              round(100 * max(dd[q]["dO_over_O"] for q in OBS), 3)]}
    mx[s] = row
out["max_abs_impact"] = mx
ph2 = PS["resolution:per_halo"]["summary"]["impact_fit2R_vs_ref_fit"]
out["per_halo_2R"] = {o: [round(100 * ph2[o]["dO_over_O"], 2), round(100 * ph2[o]["err"], 2)]
                      for o in ("tSZ_yy_total_ell3000", "kSZ_1h_ell1000", "DM_mean", "DM_var")}

# ------------------------------------------------------------------ L2p8 binding
emp = R["redshift"]["L2p8_empirical"]
l2 = {}
for rule, key in (("strict", "strict_first_edge"), ("frac95", "frac95_first_edge")):
    n = nh = 0
    exc = []
    for z, dz in emp.items():
        for b, dd in dz["bins"].items():
            n += 1
            fe = dd[key]
            bc = dd["by_criterion"]
            binds = [k for k in bc if bc[k][rule] is not None and fe is not None and abs(bc[k][rule] - fe) < 1e-6]
            if "smoothing_length" in binds:
                nh += 1
            else:
                exc.append({"z": z, "bin": b, "N": dd["N"], "first_edge": fe, "alone": {k: bc[k][rule] for k in bc},
                            "binding": binds})
    l2[rule] = {"n_cases": n, "h_binds": nh, "exceptions": exc}
Ns = [dd["N"] for dz in emp.values() for dd in dz["bins"].values()]
l2["N_range"] = [min(Ns), max(Ns)]
out["L2p8"] = l2

# ------------------------------------------------------------------ >= 500 masks: contiguity
nc = {}
for lab in CAL:
    m = PS["resolution:mc500"]["masks"][lab]
    if not m["contiguous"]:
        k = np.flatnonzero(m["kept"])
        nc[lab] = {"kept_shells": [[round(float(E[i]), 4), round(float(E[i + 1]), 4)] for i in k],
                   "n_kept": int(len(k)), "n_kept_to_2R": int(m["n_kept_to_2R"]),
                   "n_shells_from_first_edge_to_4R": int(35 - k[0])}
out["mc500_noncontiguous"] = nc

json.dump(out, open(G / "section_checks_final.json", "w"), indent=1, default=str)
for k, v in out.items():
    print(k, json.dumps(v, default=str)[:1500])
print("wrote section_checks_final.json")
