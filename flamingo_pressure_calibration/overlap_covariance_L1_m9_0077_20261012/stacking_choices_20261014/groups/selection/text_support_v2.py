#!/usr/bin/env python
"""Group 'selection': supporting numbers for the second text correction (2026-10-14, after the second verifier review).

Adds a 'text_support_v2' block to results.json.  Run after compute_selection.py, summarize.py and text_support.py
(text_support.py replaces only results.json:text_support, so this block survives a rerun of it).  Read-only on the
cache; no other key of results.json is changed.

Blocks
  top_bin_agn_per_halo       15.3-15.5 (N = 6): each halo's share of the bin's recently AGN-heated Y inside 0.1 R200c
                             and the fraction of its own heated Y(<4R) that lies inside 0.1 R200c (same regions as
                             text_support.agn_heated_location: shells with outer edge <= 0.1 R200c)
  importance_share           sum of each observable's importance over the calibrated engine bins 13.5-15.5 (the only
                             bins sc.impact_from_stacks perturbs) and over the painted fallback bins 12.7-13.5
                             (HalfDome paints M200c >= 6.6e12 Msun; 12.5-12.7 carries zero importance)
  all_painted_bins_estimate  first-order ESTIMATE of a global choice applied to every painted bin (engine bins
                             12.5-15.5, all shells, inclusion-weighted mean stacks), next to the calibrated-bins-only
                             value from the same code (which must reproduce results.json impact)
  low_mass_cold_sf           cold/SF share of N_e(<R200c) and N_e(<4R200c) in the bins 12.5-13.5
  A2ii_single_shells         largest |F_A2ii/F_ref - 1| over strict-resolved shells with outer edge <= 1 and <= 1.5
                             R200c, 13.5-15.5
  ownership_groups           tSZ ell = 100 range of the variants that remove own-satellite gas; largest |impact| of
                             the neighbour-only variants
  largest_options            largest |impact| among the measured options of each choice
  clean_per_bin_groups       CLEAN tSZ per-bin contributions summed over bin groups
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

t0 = time.time()
R = json.loads((HERE / "results.json").read_text())
V = R["variants"]
S = R["summary"]["variants"]
GRID, SCHEME = "fit", "M200c_0.2"
EDGES = sc.edges(GRID)
LO, HI = EDGES[:-1], EDGES[1:]
X = 0.75 * (HI**4 - LO**4) / (HI**3 - LO**3)
VOL = sc.shell_volumes(GRID)
BINS = sc.mass_bins(SCHEME)
LABELS = [b["label"] for b in BINS]
CAL = R["calibrated_bins"]
TOP = "15.3-15.5"
IMP = sc._importance()
CALB = list(sc.CAL_BINS)
PAINTED_FALLBACK = [LABELS.index(lab) for lab in ("12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5")]
assert list(IMP["engine_bin_labels"][:len(LABELS)]) == LABELS


def sig(x, n=4):
    return None if x is None or not np.isfinite(x) else sc._sig(float(x), n)


def imp_matrix(name):
    ikey, at, k, fld, _ = sc.OBSERVABLES[name]
    M = IMP[ikey]
    if at is not None:
        M = M[..., int(np.flatnonzero(IMP["ell"] == at)[0])]
    return M, k, fld


OBS_MAIN = ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "y_mean",
            "kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var", "DM_q16"]
ts = {"note": "Second set of supporting numbers (2026-10-14, after the second verifier review). Percentages unless "
              "the key says otherwise. 'cal' = 13.5-15.3, 'top' = 15.3-15.5 (N = 6)."}

# ------------------------------------------------------------------ (a) top bin: which haloes carry the heated core
b = BINS[LABELS.index(TOP)]
only = np.nan_to_num(np.asarray(sc.per_halo(sc.make_key("P", agn="only", grid=GRID))[b["rows"]])) * VOL[None, :]
w = np.asarray(b["weights"], float)
inner_m = HI <= 0.1 + 1e-9
inner, tot = only[:, inner_m].sum(axis=1), only.sum(axis=1)
ts["top_bin_agn_per_halo"] = {
    "description": "15.3-15.5: per halo, share of the bin's (inclusion-weighted) recently AGN-heated Y inside 0.1 "
                   "R200c, and the fraction of the halo's own heated Y(<4R200c) inside 0.1 R200c",
    "weights": [sig(x) for x in w],
    "logm200c": [sig(x, 5) for x in np.asarray(sc.halo_table()["logm200c"])[b["rows"]]],
    "share_of_bin_inner_heated_Y": [sig(x) for x in w * inner / (w @ inner)],
    "own_inner_fraction": [sig(i / t) if t > 0 else None for i, t in zip(inner, tot)],
    "bin_inner_fraction": sig((w @ inner) / (w @ tot)),
    "n_haloes_with_heated_gas_inside_0.1R": int((inner > 0).sum())}

# ------------------------------------------------------------------ (b) importance shares
ishare = {}
for o in OBS_MAIN:
    M, k, fld = imp_matrix(o)
    pb = M.sum(axis=1)
    ishare[o] = {"calibrated_13.5-15.5": sig(pb[CALB].sum()), "painted_fallback_12.7-13.5": sig(pb[PAINTED_FALLBACK].sum()),
                 "bin_12.5-12.7": sig(pb[0]), "bin_ge_15.5": sig(pb[15])}
ts["importance_share"] = {"description": "fraction of each observable's importance (sum of imp over shells) per bin "
                                         "group; sc.impact_from_stacks perturbs only the calibrated bins", **ishare}

# ------------------------------------------------------------------ (c) all-painted-bins estimate for global choices
STACK = {}


def stacked(key):
    if key not in STACK:
        STACK[key] = np.stack([sc.stack_bin(key, bb, statistic="mean", rule="strict", n_boot=0)["value"]
                               for bb in BINS])
    return STACK[key]


est = {"description": "first-order ESTIMATE: dO/O = k sum_{b,s} imp[b,s] (F_var/F_ref - 1)[b,s] over all shells of "
                      "the engine bins 12.5-15.5 (12.5-12.7 has zero importance), i.e. as if the choice were applied "
                      "to every painted halo, including the 12.7-13.5 haloes that HalfDome paints with the published "
                      "fallback profile (their inner shells are unresolved). 'calibrated_only' is the same sum over "
                      "13.5-15.5 and reproduces results.json impact."}
for vid in ("selection:xray", "selection:e_gas_muxgp", "selection:e_fullion", "selection:agn_kept",
            "selection:A2ii"):
    est[vid] = {}
    for fld, key in V[vid]["cache_keys"].items():
        if fld not in ("P", "E"):
            continue
        Fv, Fr = stacked(key), stacked(V["ref"]["cache_keys"][fld])
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where((Fv > 0) & (Fr > 0) & np.isfinite(Fv) & np.isfinite(Fr), Fv / Fr - 1, 0.0)
        for o in OBS_MAIN:
            M, k, f_o = imp_matrix(o)
            if f_o != fld:
                continue
            cal = 100 * k * (M[CALB] * d[CALB]).sum()
            allb = 100 * k * (M[:15] * d).sum()
            ref = V[vid]["impact"][fld][o]["dO_over_O_pct"]
            assert abs(cal - ref) <= 2e-3 * max(abs(ref), 1e-3), (vid, o, cal, ref)
            est[vid][o] = {"calibrated_only": sig(cal), "all_painted_bins_ESTIMATE": sig(allb),
                           "from_12.7-13.5": sig(allb - cal)}
ts["all_painted_bins_estimate"] = est

# ------------------------------------------------------------------ (d) cold/SF electrons at low mass
cs = R["cold_sf_share_of_total"]["E"]
ts["low_mass_cold_sf"] = {lab: {"Ne_R200c_pct": sig(100 * cs[lab]["integral"]["0-1"]["value"]),
                                "Ne_4R_pct": sig(100 * cs[lab]["integral"]["0-4"]["value"])}
                          for lab in ("12.5-12.7", "12.7-12.9", "12.9-13.1", "13.1-13.3", "13.3-13.5")}

# ------------------------------------------------------------------ (e) A2-ii single shells inside 1 and 1.5 R200c
a2 = {}
for f in ("P", "E", "G"):
    a2[f] = {}
    for lim in (1.0, 1.5):
        best = (0.0, None, None)
        for lab in CAL:
            sr = V["selection:A2ii"]["shell_ratio"][f][lab]
            dev = np.array([np.nan if r is None else r - 1 for r in sr["ratio"]])
            m = np.array(sr["resolved"], bool) & np.isfinite(dev) & (HI <= lim + 1e-9)
            if m.any():
                i = np.flatnonzero(m)[np.argmax(np.abs(dev[m]))]
                if abs(dev[i]) > abs(best[0]):
                    best = (dev[i], lab, X[i])
        a2[f][f"inside_{lim:g}R"] = {"max_abs_shell_dev_pct": sig(100 * best[0]), "bin": best[1],
                                     "x": sig(best[2]) if best[2] is not None else None}
ts["A2ii_single_shells"] = a2

# ------------------------------------------------------------------ (f) ownership groups
own_sat = ["selection:rm_ov_mu0.03", "selection:rm_ov_mu0.1", "selection:rm_ov_mu0.3", "selection:rm_ov_mu0.1_reach1R",
           "selection:exc_all_k0.5", "selection:exc_all_k1", "selection:exc_all_k2"]
nbr = ["selection:rm_ov_mu0.1_keepsat", "selection:exc_nbr_k1"]
t100 = {v: S[v]["impact_pct"]["tSZ_yy_total_ell100"]["value"] for v in own_sat}
ts["ownership_groups"] = {
    "own_satellite_removing_tSZ_ell100": {"values": t100, "range": [min(t100.values()), max(t100.values())]},
    "neighbour_only_tSZ_ell100": {v: [S[v]["impact_pct"]["tSZ_yy_total_ell100"]["value"],
                                      S[v]["impact_pct"]["tSZ_yy_total_ell100"]["err"]] for v in nbr},
    "neighbour_only_max_abs_impact": {v: max(((o, x["value"]) for o, x in S[v]["impact_pct"].items()),
                                             key=lambda t: abs(t[1])) for v in nbr},
    "bound_only_integral_dev_0-4": {v: {f: S[v]["integral_dev_pct"][f]["0-4"]["cal"] for f in ("P", "E")}
                                    for v in ("selection:central_bound", "selection:host_bound")}}

# ------------------------------------------------------------------ (g) largest option per choice
GROUPS = {"agn": ["selection:agn_kept"],
          "phase": ["selection:xray", "selection:cb_xray"],
          "electrons": ["selection:e_fullion", "selection:e_gas_muxgp"],
          "ownership": ["selection:central_bound", "selection:host_bound", "selection:no_other",
                        "selection:no_satellites_or_other", *own_sat, *nbr, "selection:A2i", "selection:A2ii",
                        "selection:A2ii_A2i"],
          "sample": ["selection:clean", "selection:overlapped", "selection:overlap_in1R", "selection:overlap_annulus",
                     "selection:clean_mu0.03", "selection:clean_mu0.3"]}
lo = {}
for g, vids in GROUPS.items():
    lo[g] = {}
    for o in OBS_MAIN:
        cand = [(v, S[v]["impact_pct"][o]["value"], S[v]["impact_pct"][o]["err"]) for v in vids
                if o in S[v]["impact_pct"]]
        if cand:
            v, val, err = max(cand, key=lambda t: abs(t[1]))
            lo[g][o] = {"variant": v, "value": val, "err": err}
ts["largest_options"] = lo

# ------------------------------------------------------------------ (h) CLEAN per-bin groups
cpb = R["text_support"]["clean_per_bin"]


def gsum(o, labs):
    return sig(sum(cpb[o]["per_bin_pct"][lab] for lab in labs))


ts["clean_per_bin_groups"] = {
    o: {"13.5-14.9": gsum(o, CAL[:7]), "14.9-15.5": gsum(o, CAL[7:]), "14.9-15.1": gsum(o, ["14.9-15.1"]),
        "15.1-15.3": gsum(o, ["15.1-15.3"]), "15.3-15.5": gsum(o, [TOP]), "14.3-15.1": gsum(o, CAL[4:8]),
        "13.5-13.9": gsum(o, CAL[:2]), "total": cpb[o]["total_pct"]}
    for o in ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000")}

ts["runtime_s"] = round(time.time() - t0, 1)
R["text_support_v2"] = ts
(HERE / "results.json").write_text(json.dumps(R, separators=(",", ":"), allow_nan=False))

if __name__ == "__main__":
    print(json.dumps(ts, indent=1))
