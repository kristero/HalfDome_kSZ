#!/usr/bin/env python
"""Second (adversarial) verification of group 'selection': independent numpy recomputation.

Reads the cache arrays, halo_table.npz, edges and importance_F.npz directly.  Does NOT import the group's scripts
and does not call stackcore's stacking / resolution / impact functions: the inclusion-weighted mean, the strict mask,
the bootstrap (same algorithm as documented, so replicates can be paired), the band medians, the integrals and the
first-order observable propagation are re-implemented here.  Compares with results.json and writes
v2_independent.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
OUT = GROUP.parents[1]
STUDY = OUT.parent
CACHE = OUT / "cache"
IDX = json.loads((CACHE / "index.json").read_text())["variants"]
R = json.loads((GROUP / "results.json").read_text())
V = R["variants"]
SUMM = R["summary"]["variants"]
TS = R["text_support"]

with np.load(CACHE / "halo_table.npz") as z:
    TAB = {k: z[k] for k in z.files}
with np.load(STUDY / "F_observables" / "results" / "importance_F.npz") as z:
    IMP = {k: z[k] for k in z.files}

E = np.load(CACHE / "edges_fit.npy")
LO, HI = E[:-1], E[1:]
S = len(LO)
X = 0.75 * (HI**4 - LO**4) / (HI**3 - LO**3)
VOL = 4 * np.pi / 3 * (HI**3 - LO**3)
CNT = np.load(CACHE / "counts_fit.npy")
HS = np.load(CACHE / "hsml_fit.npy")
EPS = TAB["eps_over_r200c"]
NB = 200
SEED0 = 20261012
assert np.allclose(IMP["shell_edges"], E)

# ------------------------------------------------------------------ own mass bins
me = TAB["mass_edges"]
BINS = []
for k in range(len(me) - 1):
    rows = np.flatnonzero((TAB["logm200c"] >= me[k]) & (TAB["logm200c"] < me[k + 1]))
    if len(rows):
        BINS.append(dict(k=k, label=f"{me[k]:.1f}-{me[k+1]:.1f}", rows=rows, w=TAB["weight"][rows],
                         strata=(TAB["mass_bin"] * 3 + TAB["class_main"])[rows]))
LAB = [b["label"] for b in BINS]
CAL = [b for b in BINS if b["k"] >= 5]
CAL_MAIN = [b["label"] for b in CAL if b["label"] != "15.3-15.5"]


def key(field, est="geometric", agn="removed", sel="all", phase="all", norm=None):
    if norm is None:
        norm = {"P": "P200", "E": "rhoc_fb", "G": "rhoc_fb", "E_fullion": "rhoc_fb", "T_e": "T200", "K": "K200"}[field]
    return "|".join((field, est, agn, sel, phase, "fit", norm))


_A = {}


def arr(k):
    if k not in _A:
        _A[k] = np.load(CACHE / IDX[k]["file"])
    return _A[k]


def strict(b, mc=20, sf=4.0, hf=2.0, me_=0.04):
    r = b["rows"]
    geo = LO[None, :] >= np.maximum(me_, sf * EPS[r][:, None])
    with np.errstate(invalid="ignore"):
        res = geo & (CNT[r] >= mc) & (LO[None, :] >= hf * HS[r])
    return res.all(axis=0)


_C = {}


def boot_counts(b, seed=None):
    """Multiplicity matrix, stratified resampling within strata (documented core algorithm)."""
    seed = SEED0 + 1000 * b["k"] if seed is None else seed
    kk = (b["k"], seed)
    if kk not in _C:
        n = len(b["rows"])
        rng = np.random.default_rng(seed)
        pos = np.empty((NB, n), dtype=np.int64)
        st = b["strata"]
        for s in np.unique(st):
            wh = np.flatnonzero(st == s)
            pos[:, wh] = wh[rng.integers(0, len(wh), size=(NB, len(wh)))]
        C = np.zeros((NB, n))
        for i in range(NB):
            C[i] = np.bincount(pos[i], minlength=n)
        _C[kk] = C
    return _C[kk]


def wmean(vals, w, C=None, sub=None):
    """Inclusion-weighted mean over finite values; returns (value [S], boot [NB, S])."""
    ww = w if sub is None else w * sub
    ok = np.isfinite(vals)
    v0 = np.where(ok, vals, 0.0)
    val = (ww @ v0) / (ww @ ok)
    if C is None:
        return val, None
    W = C * ww[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = (W @ v0) / (W @ ok)
    return val, bt


def stack(k, b, sub=None, seed=None):
    vals = arr(k)[b["rows"]]
    return wmean(vals, b["w"], boot_counts(b, seed), None if sub is None else sub[b["rows"]].astype(float))


BANDS = {"<0.1": (0, .1), "0.1-0.25": (.1, .25), "0.25-0.5": (.25, .5), "0.5-1": (.5, 1), "1-1.5": (1, 1.5),
         "1.5-2": (1.5, 2), "2-3": (2, 3), "3-4": (3, 4)}


def band_median(ratio_v, mask, band):
    lo, hi = BANDS[band]
    sel = (X >= lo) & (X < hi) & mask & np.isfinite(ratio_v)
    return float(np.median(ratio_v[sel])) if sel.any() else None


def integ(v, lo, hi):
    m = (LO >= lo - 1e-12) & (HI <= hi + 1e-12)
    return np.nansum((np.nan_to_num(v) * VOL)[..., m], axis=-1)


OBS = {"tSZ_yy_total_ell100": ("imp__yytot", 100, 2), "tSZ_yy_total_ell1000": ("imp__yytot", 1000, 2),
       "tSZ_yy_total_ell3000": ("imp__yytot", 3000, 2), "kSZ_1h_ell100": ("imp__ksz1h", 100, 2),
       "kSZ_1h_ell1000": ("imp__ksz1h", 1000, 2), "kSZ_1h_ell3000": ("imp__ksz1h", 3000, 2),
       "DM_mean": ("imp__dm_mean", None, 1), "DM_var": ("imp__dm_var", None, 2), "DM_q16": ("imp__dm_q16", None, 1)}
PFIELD = {o: ("P" if o.startswith("tSZ") else "E") for o in OBS}


def impact(Fv, Fr, obs):
    """dO/O in % from [10, 35] stacks (first order, NaN cells = no change)."""
    ik, ell, kfac = OBS[obs]
    M = IMP[ik]
    if ell is not None:
        M = M[..., int(np.flatnonzero(IMP["ell"] == ell)[0])]
    M = M[5:15]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = Fv / Fr - 1
    good = np.isfinite(d) & (Fv > 0) & (Fr > 0)
    d = np.where(good, d, 0.0)
    per_bin = kfac * (M * d).sum(axis=-1)
    return 100 * per_bin.sum(axis=-1), 100 * per_bin


def stacks_cal(k, sub=None, seed=None):
    vs, bs = [], []
    for b in CAL:
        v, bt = stack(k, b, sub, seed)
        vs.append(v)
        bs.append(bt)
    return np.stack(vs), np.stack(bs, axis=1)       # [10, S], [NB, 10, S]


def impact_with_err(kv, kr, obs, sub=None, seed=None):
    Fv, Fvb = stacks_cal(kv, sub, seed)
    Fr, Frb = stacks_cal(kr, None, seed)
    val, pb = impact(Fv, Fr, obs)
    bt = np.array([impact(Fvb[i], Frb[i], obs)[0] for i in range(NB)])
    return float(val), float(np.std(bt, ddof=1)), pb


out = {}
cmp_rows = []


def compare(name, mine, group, tol):
    ok = None
    try:
        ok = bool(np.all(np.abs(np.asarray(mine, float) - np.asarray(group, float)) <= tol))
    except Exception:  # noqa: BLE001
        ok = False
    cmp_rows.append({"name": name, "mine": mine, "group": group, "tol": tol, "match": ok})
    print(f"{'OK ' if ok else 'XX '} {name}: mine {mine} | group {group}")


# ------------------------------------------------------------------ 0. bins, strict first edge
for b in BINS:
    m = strict(b)
    fe = float(LO[m].min()) if m.any() else None
    contiguous = bool(m.any() and np.all(m[np.flatnonzero(m)[0]:]))
    rb = [x for x in R["bins"] if x["label"] == b["label"]][0]
    compare(f"strict first edge {b['label']} (contiguous {contiguous})", fe, rb["strict_first_edge"]
            if rb["strict_first_edge"] is not None else np.nan, 1e-3) if fe is not None else \
        compare(f"strict none {b['label']}", 0, rb["n_strict"], 0)

# ------------------------------------------------------------------ 1. AGN kept: tSZ ell 3000 (and all observables)
ref_P, ref_E = key("P"), key("E")
for obs in ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000"):
    v, e, _ = impact_with_err(key("P", agn="kept"), ref_P, obs)
    g = V["selection:agn_kept"]["impact"]["P"][obs]
    compare(f"agn_kept {obs} (value, err) %", [round(v, 5), round(e, 5)], [g["dO_over_O_pct"], g["err_pct"]], 2e-4)
for obs in ("kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var"):
    v, e, _ = impact_with_err(key("E", agn="kept"), ref_E, obs)
    g = V["selection:agn_kept"]["impact"]["E"][obs]
    compare(f"agn_kept {obs} (value, err) %", [round(v, 5), round(e, 5)], [g["dO_over_O_pct"], g["err_pct"]], 2e-4)

# resolved-only AGN kept: max over observables
mx = 0.0
for fld, ref in (("P", ref_P), ("E", ref_E)):
    Fv, _ = stacks_cal(key(fld, agn="kept"))
    Fr, _ = stacks_cal(ref)
    Mres = np.stack([strict(b) for b in CAL])
    for obs in OBS:
        if PFIELD[obs] != fld:
            continue
        mx = max(mx, abs(impact(np.where(Mres, Fv, Fr), Fr, obs)[0]))
out["agn_kept_resolved_only_max_abs_pct"] = mx
print("agn_kept resolved-only max |impact| %:", mx)

# ------------------------------------------------------------------ 2. heated share of Y(<R200c), Y(<4R)
agn_tab = {}
for b in BINS:
    vo, _ = stack(key("P", agn="only"), b)
    vr, _ = stack(ref_P, b)
    agn_tab[b["label"]] = {"Y200_share_pct": 100 * integ(vo, 0, 1) / (integ(vo, 0, 1) + integ(vr, 0, 1)),
                           "Y4R_share_pct": 100 * integ(vo, 0, 4) / (integ(vo, 0, 4) + integ(vr, 0, 4)),
                           "innermost_share_pct": 100 * vo[0] / (vo[0] + vr[0]),
                           "Y4R_inside0.1_pct": 100 * integ(vo, 0, 0.1) / integ(vo, 0, 4),
                           "Y4R_2to4_pct": 100 * integ(vo, 2, 4) / integ(vo, 0, 4)}
out["agn_table"] = agn_tab
for lab, want in (("13.5-13.7", (0.044, 0.025, 42, 60, 28)), ("14.1-14.3", (0.0085, 0.0067, 16, 65, 17)),
                  ("14.7-14.9", (0.0017, 0.0019, 1.6, 51, 30)), ("15.1-15.3", (0.0004, 0.0009, 0.43, 31, 54)),
                  ("12.5-12.7", (1.59, 0.22, 15, 65, 30))):
    a = agn_tab[lab]
    mine = [a["Y200_share_pct"], a["Y4R_share_pct"], a["innermost_share_pct"], a["Y4R_inside0.1_pct"],
            a["Y4R_2to4_pct"]]
    print("AGN table", lab, ["%.4g" % x for x in mine], "text:", want)
    rel_ok = all(abs(m - t) <= max(0.06 * abs(t), 0.5 * 10**(np.floor(np.log10(abs(t))) - 1)) for m, t in zip(mine, want))
    cmp_rows.append({"name": f"AGN table row {lab}", "mine": [float("%.4g" % x) for x in mine], "group": list(want),
                     "match": bool(rel_ok)})

# top bin: is the heated Y inside 0.1R dominated by one halo?
b = BINS[-1]
vo = arr(key("P", agn="only"))[b["rows"]]
inner = (np.nan_to_num(vo) * VOL)[:, HI <= 0.1 + 1e-12].sum(axis=1) * b["w"]
out["top_bin_heated_inner_per_halo_share"] = (inner / inner.sum()).round(4).tolist()
print("top bin heated Y(<0.1R) per-halo shares:", out["top_bin_heated_inner_per_halo_share"])

# ------------------------------------------------------------------ 3. gas phase
for obs in ("kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var", "DM_q16"):
    v, e, _ = impact_with_err(key("E", phase="xray"), ref_E, obs)
    g = V["selection:xray"]["impact"]["E"][obs]
    compare(f"xray {obs} (value, err) %", [round(v, 4), round(e, 4)], [g["dO_over_O_pct"], g["err_pct"]], 2e-3)
tsz_x = [impact_with_err(key("P", phase="xray"), ref_P, o)[0] for o in
         ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000")]
out["xray_tSZ_pct"] = tsz_x
print("xray tSZ %:", tsz_x)

cold = {}
for fld in ("P", "E", "G"):
    cold[fld] = {}
    for b in BINS:
        vc, _ = stack(key(fld, phase="cold_sf"), b)
        vr, _ = stack(key(fld), b)
        m = strict(b)
        rr = vc / vr
        cold[fld][b["label"]] = {"0-1": 100 * integ(vc, 0, 1) / integ(vr, 0, 1),
                                 "2-4": 100 * integ(vc, 2, 4) / integ(vr, 2, 4),
                                 "0-4": 100 * integ(vc, 0, 4) / integ(vr, 0, 4),
                                 "band3-4": 100 * band_median(rr, m, "3-4") if band_median(rr, m, "3-4") else None,
                                 "inner": 100 * rr[0]}


def rng_cal(d, k):
    v = [d[l][k] for l in CAL_MAIN if d[l][k] is not None]
    return [round(min(v), 4), round(max(v), 4)]


cold_rng = {f: {k: rng_cal(cold[f], k) for k in ("0-1", "2-4", "0-4", "band3-4")} for f in cold}
cold_rng["inner_13.5_14.7"] = {f: [round(cold[f]["13.5-13.7"]["inner"], 4), round(cold[f]["14.7-14.9"]["inner"], 4)]
                               for f in cold}
cold_rng["expl_12.5_0-1"] = {f: round(cold[f]["12.5-12.7"]["0-1"], 4) for f in cold}
out["cold_sf"] = cold_rng
print("cold/SF ranges:", json.dumps(cold_rng))

# T_e x-ray vs all: band medians
te = {}
for b in CAL:
    if b["label"] == "15.3-15.5":
        continue
    vx, _ = stack(key("T_e", est="weighted", phase="xray"), b)
    vr, _ = stack(key("T_e", est="weighted"), b)
    te[b["label"]] = {bd: band_median(vx / vr, strict(b), bd) for bd in BANDS}
te_rng = {}
for bd in BANDS:
    v = [te[l][bd] - 1 for l in te if te[l][bd] is not None]
    te_rng[bd] = [round(100 * min(v), 3), round(100 * max(v), 3)] if v else None
out["T_e_xray_band_dev_pct"] = te_rng
print("T_e xray band dev %:", te_rng)

# ------------------------------------------------------------------ 4. electron counting
MU = (1.0 / 1836.15267343 + 2.0 / 1.76) / 0.9
ne4 = {}
for b in CAL:
    vg, _ = stack(key("G"), b)
    ve, _ = stack(ref_E, b)
    vf, _ = stack(key("E_fullion"), b)
    m = strict(b)
    rg = integ(vg, 0, 4) / integ(ve, 0, 4)
    ne4[b["label"]] = {"G/E-1": 100 * (rg - 1), "full/E-1": 100 * (integ(vf, 0, 4) / integ(ve, 0, 4) - 1),
                       "X_null": 2 / (MU * rg) - 1,
                       "X076_04": 100 * (MU * 1.76 / 2 * rg - 1),
                       "X076_band01": 100 * (MU * 1.76 / 2 * band_median(vg / ve, m, "<0.1") - 1)
                       if band_median(vg / ve, m, "<0.1") else None,
                       "rg_band01": band_median(vg / ve, m, "<0.1")}
cm_ = [l for l in CAL_MAIN]
xn = np.array([ne4[l]["X_null"] for l in cm_])
xg = float(xn.mean())
for l in cm_:
    rb = ne4[l]["rg_band01"]
    ne4[l]["Xeff_band01"] = 100 * (MU * (1 + xg) / 2 * rb - 1) if rb else None
    ne4[l]["Xeff_04"] = 100 * (MU * (1 + xg) / 2 * (1 + ne4[l]["G/E-1"] / 100) - 1)
elec = {k: [round(min(ne4[l][k] for l in cm_ if ne4[l][k] is not None), 4),
            round(max(ne4[l][k] for l in cm_ if ne4[l][k] is not None), 4)]
        for k in ("G/E-1", "full/E-1", "X_null", "X076_04", "X076_band01", "Xeff_band01", "Xeff_04")}
elec["X_null_mean"] = round(xg, 4)
out["electrons"] = elec
print("electron counting:", json.dumps(elec))
compare("G/E N_e(<4R) dev % range", elec["G/E-1"], [-8.752, -8.494], 2e-3)
compare("X=0.76 N_e(<4R) dev % range", elec["X076_04"], [1.435, 1.722], 2e-3)
compare("X=0.76 <0.1 band dev %", elec["X076_band01"], [4.339, 5.757], 2e-3)
compare("X_eff <0.1 band dev %", elec["Xeff_band01"], [2.752, 4.149], 2e-3)
compare("X_eff N_e(<4R) dev %", elec["Xeff_04"], [-0.1078, 0.1746], 2e-3)
for obs in ("kSZ_1h_ell1000", "DM_mean"):
    v, e, _ = impact_with_err(key("G"), ref_E, obs)
    g = V["selection:e_gas_muxgp"]["impact"]["E"][obs]
    compare(f"e_gas_muxgp {obs} %", round(v, 3), g["dO_over_O_pct"], 5e-3)  # group value has 4 sig. digits

# ------------------------------------------------------------------ 5. ownership
a2 = {}
for fld in ("P", "E"):
    for b in CAL:
        va, _ = stack(key(fld, sel="A2ii"), b)
        vr, _ = stack(key(fld), b)
        m = strict(b)
        rr = va / vr
        a2.setdefault(fld, {})[b["label"]] = {
            "band3-4": 100 * (band_median(rr, m, "3-4") - 1),
            "I04": 100 * (integ(va, 0, 4) / integ(vr, 0, 4) - 1),
            "max_shell_inside1.5": 100 * float(np.nanmax(np.abs(rr[m & (HI <= 1.5 + 1e-9)] - 1)))
            if (m & (HI <= 1.5 + 1e-9)).any() else None}
a2r = {f: {k: [round(min(a2[f][l][k] for l in CAL_MAIN if a2[f][l][k] is not None), 4),
               round(max(a2[f][l][k] for l in CAL_MAIN if a2[f][l][k] is not None), 4)]
           for k in ("band3-4", "I04", "max_shell_inside1.5")} for f in a2}
a2r["top_max_shell_inside1.5"] = {f: a2[f]["15.3-15.5"]["max_shell_inside1.5"] for f in a2}
out["A2ii"] = a2r
print("A2ii:", json.dumps(a2r))
for obs, want in (("tSZ_yy_total_ell100", (-1.602, 0.068)), ("tSZ_yy_total_ell1000", (-0.354, 0.027)),
                  ("tSZ_yy_total_ell3000", (0.038, 0.010)), ("DM_mean", (-3.231, 0.114)), ("DM_q16", (-5.52, 0.20))):
    v, e, _ = impact_with_err(key(PFIELD[obs], sel="A2ii"), key(PFIELD[obs]), obs)
    compare(f"A2ii {obs} (value, err) %", [round(v, 3), round(e, 3)], list(want), [2e-3 + 0.005 * abs(want[0]), 5e-3])  # text values rounded

for sel in ("central_bound", "host_bound", "no_satellites_or_other", "no_other"):
    v, e, _ = impact_with_err(key("E", sel=sel), ref_E, "DM_mean")
    v2, e2, _ = impact_with_err(key("P", sel=sel), ref_P, "tSZ_yy_total_ell1000")
    out[f"{sel}_DMmean_tSZ1000"] = [v, e, v2, e2]
    print(f"{sel}: DM mean {v:.3f} +- {e:.3f} %, tSZ1000 {v2:.3f} +- {e2:.3f} %")

t100 = {}
for sel in ("rm_ov_mu0.03", "rm_ov_mu0.1", "rm_ov_mu0.3", "rm_ov_mu0.1_keepsat", "rm_ov_mu0.1_reach1R",
            "exc_all_k0.5", "exc_all_k1", "exc_all_k2", "exc_nbr_k1", "A2i", "A2ii_A2i"):
    v, e, pb = impact_with_err(key("P", sel=sel), ref_P, "tSZ_yy_total_ell100")
    v1, e1, pb1 = impact_with_err(key("P", sel=sel), ref_P, "tSZ_yy_total_ell1000")
    t100[sel] = {"ell100": [round(v, 3), round(e, 3)], "ell1000": [round(v1, 3), round(e1, 3)],
                 "ell1000_top_bin": round(float(pb1[-1]), 3)}
out["ownership_tSZ"] = t100
print("ownership tSZ:", json.dumps(t100))

# unbound share of pressure: no_other - host_bound
unb = {}
for b in CAL:
    vn, _ = stack(key("P", sel="no_other"), b)
    vh, _ = stack(key("P", sel="host_bound"), b)
    vr, _ = stack(ref_P, b)
    unb[b["label"]] = [100 * integ(vn - vh, 1, 2) / integ(vr, 1, 2), 100 * integ(vn - vh, 2, 4) / integ(vr, 2, 4)]
out["unbound_share_P"] = {"1-2": [round(min(unb[l][0] for l in CAL_MAIN), 2), round(max(unb[l][0] for l in CAL_MAIN), 2)],
                          "2-4": [round(min(unb[l][1] for l in CAL_MAIN), 2), round(max(unb[l][1] for l in CAL_MAIN), 2)]}
compare("unbound share of P 1-2R, 2-4R %", [out["unbound_share_P"]["1-2"], out["unbound_share_P"]["2-4"]],
        [[58.8, 64.4], [74.4, 89.9]], 0.06)

# ------------------------------------------------------------------ 6. halo sample
clean = TAB["class_main"] == 0
for obs, want in (("tSZ_yy_total_ell100", (0.80, 1.57)), ("tSZ_yy_total_ell1000", (1.72, 1.82)),
                  ("tSZ_yy_total_ell3000", (4.16, 1.00)), ("DM_mean", (-1.93, 0.20))):
    v, e, pb = impact_with_err(key(PFIELD[obs]), key(PFIELD[obs]), obs, sub=clean)
    compare(f"CLEAN {obs} (value, err) %", [round(v, 3), round(e, 3)], list(want), 0.006)
    if obs == "tSZ_yy_total_ell100":
        compare("CLEAN tSZ100 top-bin contribution", round(float(pb[-1]), 3), 0.8615, 2e-3)
    # robustness of the error to the bootstrap seed (independent, unpaired seed)
    v2, e2, _ = impact_with_err(key(PFIELD[obs]), key(PFIELD[obs]), obs, sub=clean, seed=777)
    out[f"clean_{obs}_err_seed777"] = e2
    print(f"  CLEAN {obs}: err with another seed {e2:.3f}")
ws = []
for b in CAL:
    if b["label"] == "15.3-15.5":
        continue
    sub = clean[b["rows"]]
    ws.append(b["w"][sub].sum() / b["w"].sum())
compare("CLEAN weight share 13.5-15.3", [round(min(ws), 3), round(max(ws), 3)], [0.66, 0.81], 0.006)
out["clean_N_top"] = int(clean[BINS[-1]["rows"]].sum())
in1 = TAB["class_main"] == 1
out["class_main1_vs_neighbours"] = {int(k): int(n) for k, n in zip(*np.unique(TAB["class_neighbours_mu0p1"][in1],
                                                                            return_counts=True))}
print("class_main==1 by class_neighbours_mu0p1:", out["class_main1_vs_neighbours"], "N in1R", int(in1.sum()))
# excision k1 tSZ 1000
v, e, pb = impact_with_err(key("P", sel="exc_all_k1"), ref_P, "tSZ_yy_total_ell1000")
compare("exc_all_k1 tSZ1000 (value, err, top)", [round(v, 2), round(e, 2), round(float(pb[-1]), 2)], [-5.56, 0.90, -2.34],
        0.011)

out["comparisons"] = cmp_rows
out["n_mismatch"] = int(sum(1 for c in cmp_rows if not c["match"]))
(HERE / "v2_independent.json").write_text(json.dumps(out, indent=1, default=float))
print("mismatches:", out["n_mismatch"], "of", len(cmp_rows))
