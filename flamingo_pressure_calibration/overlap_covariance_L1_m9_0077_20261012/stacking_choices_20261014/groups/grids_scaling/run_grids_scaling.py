"""Group 'grids_scaling': radial grid, radius / normalisation scaling, radial range / truncation, clumping.

Run:  LD_LIBRARY_PATH=<halfdome env>/lib <halfdome env>/bin/python -B run_grids_scaling.py
Writes (this directory only): web_variants.json, work/main_results.json, work/figdata_main.pkl
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np

import gs_common as C

sc = C.sc
T0 = time.time()
HERE = C.HERE
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)

TAB = sc.halo_table()
BINS = C.bins()
CAL = C.calibrated(BINS)
CAL_LABELS = [b["label"] for b in CAL]
FIT_E = sc.edges("fit")
XF = C.xcentre(FIT_E)
H = float(sc.load_index()["hubble"])
H70 = H / 0.7
FB = float(sc.load_index()["baryon_fraction"])
FIELDS_WEB = ("P", "E", "G", "T_e", "K")

# per-halo scales (physical units)
RHO_C200 = TAB["m200c"] / (4 * np.pi / 3 * TAB["r200c"]**3) / 200.0      # Msun / Mpc^3 (halo's own M, R)
RHO_C500 = TAB["m500c"] / (4 * np.pi / 3 * TAB["r500c"]**3) / 500.0
Q500 = TAB["r500c"] / TAB["r200c"]                                       # R500c / R200c
A10M = (TAB["m500c"] * H70 / 3e14)**0.12                                  # A10's extra M^0.12 factor

# conversion factors variant units -> reference units (multiply), per halo
FACT = {
    ("P", "P200"): np.ones_like(Q500),
    ("P", "B24"): TAB["p500_B24_pa"] / TAB["p200_pa"],
    ("P", "A10"): TAB["p500_A10_pa"] / TAB["p200_pa"],
    ("P", "self_similar_500"): TAB["p500_ss_pa"] / TAB["p200_pa"],
    ("P", "A10M"): TAB["p500_A10_pa"] * A10M / TAB["p200_pa"],
    ("E", "rhoc_fb"): np.ones_like(Q500),
    # E_ref = n_e mu_XGP / (rho_c200 f_b); E_B24 = n_e / n_e,500  ->  E_ref = E_B24 n_e,500 mu_XGP / (rho_c200 f_b)
    ("E", "B24"): TAB["ne500_B24_m3"] * C.MU_XGP_MSUN / (RHO_C200 * FB / C.MPC_M**3),
    ("G", "rhoc_fb"): np.ones_like(Q500),
    ("G", "B24"): 500 * 0.159 * RHO_C500 / (RHO_C200 * FB),
    ("T_e", "T200"): np.ones_like(Q500),
    ("T_e", "T500"): TAB["T500_K"] / TAB["T200_K"],
    ("K", "K200"): np.ones_like(Q500),
    ("K", "K500"): (TAB["T500_K"] / TAB["T200_K"]) * (TAB["ne500_B24_m3"] / TAB["ne200_SS_m3"])**(-2 / 3),
}


def key(field, grid="fit", norm=None, estimator=None):
    est = estimator or ("weighted" if field in ("T_e", "K", "T_M", "T_V", "T_X", "C_rho", "C_P") else "geometric")
    return sc.make_key(field, estimator=est, grid=grid, norm=norm)


def settings(template, grid, extra=None):
    s = {"key_template": template, "statistic": "mean", "rule": "strict", "min_count": 20, "softening_factor": 4,
         "hsml_factor": 2, "min_edge": 0.04, "scheme": C.SCHEME, "sample": "all haloes of the bin (inclusion weights)",
         "grid": grid, "n_boot": C.N_BOOT}
    if extra:
        s.update(extra)
    return s


TEMPLATE_STD = "{field}|{geometric|weighted}|removed|all|all|%s|{default norm}"

# ============================================================================ variant definitions
# spec: ('key', key, grid, norm)  |  ('derived', fn(rows)->values, grid, norm)  |  ('paint', src_vid, field)
#       |  ('haloremap', src_key)
VARIANTS = []


def add(vid, dimension, label, description, fields, grid, x_unit, template, extra=None):
    VARIANTS.append({"id": vid, "dimension": dimension, "label": label, "description": description,
                     "fields": fields, "grid": grid, "x_unit": x_unit,
                     "settings": settings(template, grid, extra)})


DEF_NORM_200 = {"P": "P200", "E": "rhoc_fb", "G": "rhoc_fb", "T_e": "T200", "K": "K200"}
DEF_NORM_500 = {"P": "B24", "E": "B24", "G": "B24", "T_e": "T500", "K": "K500"}

add("ref", "Reference", "Reference: fit grid (35 shells), r/R200c, P200 / rho_c f_b",
    "The reference stack: inclusion-weighted mean on the 35-shell fit grid to 4 R200c, radii in R200c, pressure in "
    "P200 and electrons in n_e mu_XGP/(rho_c f_b); geometric estimator (ratio-of-sums 'weighted' estimator for "
    "T_e and K, the only one cached for those fields), recent-AGN gas removed, all gas.",
    {f: ("key", key(f, "fit"), "fit", DEF_NORM_200[f]) for f in FIELDS_WEB}, "fit", "R200c", TEMPLATE_STD % "fit")
add(f"{C.GROUP}:x200", "Radial grid", "x200 storage grid (67 shells)",
    "Same particle sums on the 67-shell X200 storage grid (union of the radii234, Lee22 and fit edges, with "
    "thin sliver shells) instead of the 35-shell fit grid.",
    {f: ("key", key(f, "x200"), "x200", DEF_NORM_200[f]) for f in FIELDS_WEB}, "x200", "R200c", TEMPLATE_STD % "x200")
add(f"{C.GROUP}:lee22", "Radial grid", "Lee22 grid (27 shells, 0.04-1.34 R200c)",
    "Lee et al. (2022)'s 27 logarithmic shells between 0.04 and 1.34 R200c; nothing is measured beyond 1.34 R200c.",
    {f: ("key", key(f, "lee22"), "lee22", DEF_NORM_200[f]) for f in FIELDS_WEB}, "lee22", "R200c",
    TEMPLATE_STD % "lee22")
add(f"{C.GROUP}:old", "Radial grid", "radii234 grid (36 shells)",
    "The earlier radii234 36-shell grid to 4 R200c (log shells to 3 R200c plus edges at 1, 2 and 3.3-4); it has a "
    "sliver at 0.98-1 R200c.",
    {f: ("key", key(f, "old"), "old", DEF_NORM_200[f]) for f in FIELDS_WEB}, "old", "R200c", TEMPLATE_STD % "old")
add(f"{C.GROUP}:x500_B24", "Radius scaling and normalisation", "r/R500c, B24 scales (x500 grid)",
    "Profiles against r/R500c on the 40-shell X500 grid to 6 R500c, normalised with Braspenning+24 500c scales "
    "(P_e,500 = n_e,500 kT500, n_e,500 = 500 f_B rho_c/(mu_e m_p), T500, K500).",
    {f: ("key", key(f, "x500"), "x500", DEF_NORM_500[f]) for f in FIELDS_WEB}, "x500", "R500c", TEMPLATE_STD % "x500")
add(f"{C.GROUP}:x500_A10", "Radius scaling and normalisation", "r/R500c, P/P500 (A10)",
    "Pressure against r/R500c normalised by Arnaud+10's P500 = 1.65e-3 E(z)^8/3 (M500/3e14 h70^-1)^2/3 h70^2 keV/cm^3 "
    "(no M^0.12 term).",
    {"P": ("key", key("P", "x500", "A10"), "x500", "A10")}, "x500", "R500c", "P|geometric|removed|all|all|x500|A10")
add(f"{C.GROUP}:x500_A10M", "Radius scaling and normalisation", "r/R500c, P/[P500 (M500/3e14)^0.12] (A10 full)",
    "As the A10 variant, but also divided by A10's mass term (M500/3e14 h70^-1 Msun)^0.12 (derived here per halo).",
    {"P": ("derived", lambda rows: np.asarray(sc.per_halo(key("P", "x500", "A10"))[rows]) / A10M[rows][:, None],
           "x500", "A10M")}, "x500", "R500c", "P|geometric|removed|all|all|x500|A10 / (M500 h70/3e14)^0.12")
add(f"{C.GROUP}:x500_ss500", "Radius scaling and normalisation", "r/R500c, G M500 rho500 f_b/(2 R500c)",
    "Pressure against r/R500c normalised by the self-similar thermal-pressure scale at 500c (the P200 form at "
    "Delta = 500; no mu/mu_e factor), = 1.930 x the B24 P_e,500.",
    {"P": ("key", key("P", "x500", "self_similar_500"), "x500", "self_similar_500")}, "x500", "R500c",
    "P|geometric|removed|all|all|x500|self_similar_500")
add(f"{C.GROUP}:x500_P200", "Radius scaling and normalisation", "r/R500c, P200 / rho_c f_b (x500 grid)",
    "Radii in R500c (X500 grid) but the reference normalisations P200 and n_e mu_XGP/(rho_c f_b): isolates the "
    "radius unit.",
    {"P": ("key", key("P", "x500", "P200"), "x500", "P200"), "E": ("key", key("E", "x500", "rhoc_fb"), "x500",
                                                                     "rhoc_fb")},
    "x500", "R500c", "{P|E}|geometric|removed|all|all|x500|{P200|rhoc_fb}")
add(f"{C.GROUP}:b24_B24", "Radius scaling and normalisation", "B24 grid (30 shells, 0.01-3 R500c), B24 scales",
    "Braspenning+24's 30 log shells over 0.01-3 R500c with their 500c normalisations (statistic still the "
    "inclusion-weighted mean, estimator still geometric).",
    {"P": ("key", key("P", "b24"), "b24", "B24"), "E": ("key", key("E", "b24"), "b24", "B24")}, "b24", "R500c",
    "{P|E}|geometric|removed|all|all|b24|B24")
add(f"{C.GROUP}:fit_P500", "Radius scaling and normalisation", "r/R200c, P/P_e,500 (B24) (fit grid)",
    "Radii in R200c on the fit grid but pressure divided by the B24 P_e,500 instead of P200 (derived per halo): "
    "isolates the pressure normalisation.",
    {"P": ("derived", lambda rows: np.asarray(sc.per_halo(key("P", "fit"))[rows]) / FACT[("P", "B24")][rows][:, None],
           "fit", "B24")}, "fit", "R200c", "P|geometric|removed|all|all|fit|P200 x P200/P_e,500(B24)")
add(f"{C.GROUP}:x500_halo", "Radius scaling and normalisation", "measured in R500c shells, converted per halo to R200c",
    "The X500-grid sums of each halo remapped to the fit grid with its own R500c/R200c (conservative remap: monotone "
    "PCHIP of the cumulative in ln x) and stacked in R200c/P200 units: a consistency check of the two storage "
    "grids. Shells beyond 6 R500c of any halo of the bin are left empty.",
    {"P": ("haloremap", key("P", "x500", "P200")), "E": ("haloremap", key("E", "x500", "rhoc_fb"))}, "fit", "R200c",
    "{P|E}|geometric|removed|all|all|x500|{P200|rhoc_fb} remapped per halo to fit")
add(f"{C.GROUP}:x500_paint", "Radius scaling and normalisation", "R500c-scaled stack painted at M200c",
    "The r/R500c, B24-normalised stack converted to r/R200c and P200 (or rho_c f_b) with the bin's mean R500c/R200c "
    "and mean P_e,500/P200, as HalfDome (M200c only) would paint it; remapped to the fit grid (monotone PCHIP of the "
    "cumulative in ln x). Shells beyond 6 R500c (at the bin-mean ratio) are empty.",
    {"P": ("paint", f"{C.GROUP}:x500_B24", "P"), "E": ("paint", f"{C.GROUP}:x500_B24", "E")}, "fit", "R200c",
    "x500 B24 stack x <P_e,500/P200>_bin at r = <R500c/R200c>_bin x r500")

VBYID = {v["id"]: v for v in VARIANTS}

# ============================================================================ stacks
STK = {}          # (vid, field, label) -> result dict (own grid / units)
REF_FIT = {}      # (field, label) -> [1+B, 35] reference rows
CONV = {}         # (vid, field, label) -> dict(rows [1+B, 35] in ref units on the fit grid, ok [35])
SCALE = {}        # (vid, field, label) -> (rho_b, c_b)


def bin_mean(arr, b):
    return float(np.average(arr[b["rows"]], weights=b["weights"]))


def mask_uncovered(res, cover):
    """Per-halo X500 -> fit remaps: a fit shell beyond 6 R500c of some halo of the bin would be a mean over the
    covered (more concentrated) haloes only; set such shells to NaN (missing) in the stack and its replicates."""
    bad = ~np.asarray(cover, bool)
    for k in ("value", "err", "sd", "sd_ln", "q16", "q84", "boot_q16", "boot_q84"):
        if k in res and res[k] is not None:
            res[k] = np.where(bad, np.nan, np.asarray(res[k], float))
    if "boot" in res:
        res["boot"] = np.where(bad[None, :], np.nan, res["boot"])
    res["uncovered_shells_set_to_nan"] = int(bad.sum())


def first_resolved_edge(mask, edges):
    idx = np.flatnonzero(mask)
    return float(edges[idx[0]]) if idx.size else np.nan


for v in VARIANTS:
    for f, spec in v["fields"].items():
        kind = spec[0]
        if kind in ("paint",):
            continue
        for b in BINS:
            if kind == "key":
                res = C.stack_key(spec[1], b)
            elif kind == "derived":
                res = C.stack_values(spec[1](b["rows"]), b, spec[2])
            elif kind == "haloremap":
                src = np.asarray(sc.per_halo(spec[1])[b["rows"]])
                vals = np.empty((len(b["rows"]), len(FIT_E) - 1))
                for i, r in enumerate(b["rows"]):
                    vals[i] = C.remap_rows(src[i:i + 1], sc.edges("x500"), FIT_E, scale=Q500[r])[0]
                res = C.stack_values(vals, b, "fit")
                # resolved: reference strict mask, every halo covered, and inside every halo's x500 strict range
                m500 = sc.resolution_mask(b["rows"], "x500", "strict")
                e0 = first_resolved_edge(m500, sc.edges("x500"))
                lo_needed = e0 * np.max(Q500[b["rows"]])
                cover = FIT_E[1:] <= 6.0 * np.min(Q500[b["rows"]]) + 1e-9
                res["resolved"] = sc.resolution_mask(b["rows"], "fit", "strict") & cover & (FIT_E[:-1] >= lo_needed)
                mask_uncovered(res, cover)
            STK[(v["id"], f, b["label"])] = res
    print(f"stacked {v['id']}  {time.time() - T0:.0f} s", flush=True)

# reference rows on the fit grid
for f in FIELDS_WEB:
    for b in BINS:
        REF_FIT[(f, b["label"])] = C.all_rows(STK[("ref", f, b["label"])])

# ============================================================================ conversion to reference units, fit grid
for v in VARIANTS:
    for f, spec in v["fields"].items():
        for b in BINS:
            lab = b["label"]
            if spec[0] == "paint":
                src = STK[(spec[1], spec[2], lab)]
                g = src["grid"]
                norm = VBYID[spec[1]]["fields"][spec[2]][3]
                rho_b = bin_mean(Q500, b)
                c_b = bin_mean(FACT[(spec[2], norm)], b)
                rows = C.remap_rows(C.all_rows(src), sc.edges(g), FIT_E, scale=rho_b, factor=c_b)
                ok = C.remap_mask(src["resolved"], sc.edges(g), FIT_E, scale=rho_b)
                SCALE[(v["id"], f, lab)] = (rho_b, c_b)
                n_used = np.where(np.isfinite(rows[0]), b["N"], 0)
                err = np.nanstd(rows[1:], axis=0, ddof=1)
                q = np.nanpercentile(rows[1:], [16, 84], axis=0)
                STK[(v["id"], f, lab)] = {"value": rows[0], "err": err, "q16": None, "q84": None, "n_used": n_used,
                                          "resolved": ok, "boot": rows[1:], "boot_q16": q[0], "boot_q84": q[1],
                                          "grid": "fit", "sd": np.full(35, np.nan), "sd_ln": np.full(35, np.nan),
                                          "conversion": {"rho_b": rho_b, "c_b": c_b}}
                CONV[(v["id"], f, lab)] = {"rows": rows, "ok": ok & STK[("ref", f, lab)]["resolved"]}
                continue
            res = STK[(v["id"], f, lab)]
            g = res["grid"] if "grid" in res else spec[2]
            norm = spec[3] if spec[0] in ("key", "derived") else DEF_NORM_200[f]
            if spec[0] == "haloremap":
                rho_b, c_b = 1.0, 1.0
            else:
                rho_b = bin_mean(Q500, b) if sc.grid_radius(g) == "R500c" else 1.0
                c_b = bin_mean(FACT[(f, norm)], b)
            SCALE[(v["id"], f, lab)] = (rho_b, c_b)
            rows_own = C.all_rows(res)
            if g == "fit" and rho_b == 1.0:
                rows = rows_own * c_b
                ok = res["resolved"].copy()
            else:
                rows = C.remap_rows(rows_own, sc.edges(g), FIT_E, scale=rho_b, factor=c_b)
                ok = C.remap_mask(res["resolved"], sc.edges(g), FIT_E, scale=rho_b)
            CONV[(v["id"], f, lab)] = {"rows": rows, "ok": ok & STK[("ref", f, lab)]["resolved"]}
print(f"converted  {time.time() - T0:.0f} s", flush=True)

# ============================================================================ band ratios and impacts
VARIANT_RESULTS = {}
for v in VARIANTS:
    vr = {"dimension": v["dimension"], "label": v["label"], "band_ratios": {}, "impact": {}, "conversion": {}}
    for f in v["fields"]:
        vr["band_ratios"][f] = {}
        vr["conversion"][f] = {}
        for b in BINS:
            lab = b["label"]
            cv = CONV[(v["id"], f, lab)]
            vr["band_ratios"][f][lab] = C.band_ratios(cv["rows"], REF_FIT[(f, lab)], cv["ok"], XF)
            rho_b, c_b = SCALE[(v["id"], f, lab)]
            vr["conversion"][f][lab] = {"R_var_over_R200c_bin_mean": rho_b, "norm_to_ref_factor_bin_mean": c_b,
                                        "calibrated": lab in CAL_LABELS}
        if f in ("P", "E"):
            var_cal = np.stack([CONV[(v["id"], f, lab)]["rows"] for lab in CAL_LABELS])     # [10, 1+B, 35]
            ref_cal = np.stack([REF_FIT[(f, lab)] for lab in CAL_LABELS])
            vr["impact"][f] = C.impact_rows(var_cal, ref_cal, f, C.OBS_P if f == "P" else C.OBS_E)
    VARIANT_RESULTS[v["id"]] = vr
print(f"band ratios / impacts  {time.time() - T0:.0f} s", flush=True)

# ============================================================================ remap-method sensitivity (R500c -> R200c)
# x500_paint: the bin-mean conversion of the x500 B24 stack with each remap method; x500_halo: the per-halo conversion
# of the X500 sums with each method (where the R200c truth is known, so this measures the method error itself).
REMAP_SENS = {"note": "pchip is the method used for the variants; constant_density, powerlaw_cumulative "
                      "(conservative) and loglog_points (non-conservative) are alternatives for the sensitivity test.",
              "x500_paint": {}, "x500_halo": {}, "x500_paint_method_corrected": {}}
ROWS_SENS = {}


def _band_range(band_by_bin):
    out = {}
    for bl in C.BAND_LABELS:
        v = [band_by_bin[l][bl]["ratio"] for l in band_by_bin if band_by_bin[l][bl]["ratio"] is not None]
        out[bl] = [min(v), max(v)] if v else None
    return out


for meth, fn in C.REMAP_METHODS.items():
    for kind in ("x500_paint", "x500_halo"):
        d = {}
        for f in ("P", "E"):
            obs = C.OBS_P if f == "P" else C.OBS_E
            rows_cal, ref_cal, band = [], [], {}
            for b in CAL:
                lab = b["label"]
                vid = f"{C.GROUP}:{kind}"
                ok = CONV[(vid, f, lab)]["ok"]
                if kind == "x500_paint":
                    if meth == "pchip":
                        rows = CONV[(vid, f, lab)]["rows"]
                    else:
                        rho_b, c_b = SCALE[(vid, f, lab)]
                        rows = fn(C.all_rows(STK[(f"{C.GROUP}:x500_B24", f, lab)]), sc.edges("x500"), FIT_E,
                                  scale=rho_b, factor=c_b)
                else:
                    if meth == "pchip":
                        rows = CONV[(vid, f, lab)]["rows"]
                    else:
                        src = np.asarray(sc.per_halo(VBYID[vid]["fields"][f][1])[b["rows"]])
                        vals = np.vstack([fn(src[i:i + 1], sc.edges("x500"), FIT_E, scale=Q500[r_])
                                          for i, r_ in enumerate(b["rows"])])
                        st_ = C.stack_values(vals, b, "fit")
                        mask_uncovered(st_, FIT_E[1:] <= 6.0 * np.min(Q500[b["rows"]]) + 1e-9)
                        rows = C.all_rows(st_)
                rows_cal.append(rows)
                ref_cal.append(REF_FIT[(f, lab)])
                ROWS_SENS[(kind, meth, f, lab)] = rows
                band[lab] = C.band_ratios(rows, REF_FIT[(f, lab)], ok, XF)
            imp = C.impact_rows(np.stack(rows_cal), np.stack(ref_cal), f, obs)
            d[f] = {"impact": imp, "band_ratio_range_calibrated": _band_range(band), "band_ratios": band}
        REMAP_SENS[kind][meth] = d
    # paint bias with this method's own error removed: F_paint x F_ref / F_halo (the per-halo conversion of the same
    # X500 sums measures the method error where the truth is known); paired replicates, first order
    d = {}
    for f in ("P", "E"):
        obs = C.OBS_P if f == "P" else C.OBS_E
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.stack([ROWS_SENS[("x500_paint", meth, f, l)] * REF_FIT[(f, l)] / ROWS_SENS[("x500_halo", meth, f, l)]
                             for l in CAL_LABELS])
        d[f] = {"impact": C.impact_rows(corr, np.stack([REF_FIT[(f, l)] for l in CAL_LABELS]), f, obs)}
    REMAP_SENS["x500_paint_method_corrected"][meth] = d
    print(f"remap sensitivity {meth}  {time.time() - T0:.0f} s", flush=True)

# ============================================================================ web_variants.json
web = {"group": C.GROUP, "variants": [], "records": []}
for v in VARIANTS:
    web["variants"].append({"id": v["id"], "dimension": v["dimension"], "label": v["label"],
                            "description": v["description"], "fields": list(v["fields"]), "scheme": C.SCHEME,
                            "grid": v["grid"], "settings": v["settings"]})
    for f, spec in v["fields"].items():
        for b in BINS:
            res = STK[(v["id"], f, b["label"])]
            g = res.get("grid", v["grid"])
            web["records"].append(C.web_record(v["id"], f, b, sc.edges(g), sc.grid_radius(g), res, res["resolved"]))
(HERE / "web_variants.json").write_text(json.dumps(web, separators=(",", ":")))
print(f"web_variants.json {(HERE / 'web_variants.json').stat().st_size / 1e6:.2f} MB, "
      f"{len(web['records'])} records", flush=True)

# ============================================================================ (1) rebinning consistency
def rebin_values(vals, src_edges, tgt_edges, weights=None):
    """Exact rebin of shell averages onto a sub-grid (tgt edges must be src edges): volume (or given) weights."""
    se, te = np.asarray(src_edges), np.asarray(tgt_edges)
    idx = [int(np.argmin(np.abs(se - t))) for t in te]
    if not np.allclose(se[idx], te, rtol=0, atol=1e-12):
        raise ValueError("target edges are not a subset of the source edges")
    w = np.diff(se**3) if weights is None else weights
    w = np.broadcast_to(w, vals.shape)
    out = np.empty(vals.shape[:-1] + (len(te) - 1,))
    for k in range(len(te) - 1):
        sl = slice(idx[k], idx[k + 1])
        ws = w[..., sl]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., k] = np.nansum(vals[..., sl] * ws, axis=-1) / np.sum(np.where(np.isfinite(vals[..., sl]), ws, 0),
                                                                            axis=-1)
    return out


def maxrel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    return float(np.max(np.abs(a[ok] / b[ok] - 1))) if ok.any() else None


rebin = {"per_halo": {}, "stack": {}, "ratio_fields_after_stacking": {}, "median_after_stacking": {}}
for tgt in ("fit", "lee22", "old"):
    for f in ("P", "E", "G"):
        src = np.asarray(sc.per_halo(key(f, "x200")))
        rebin["per_halo"][f"{f}: x200->{tgt}"] = maxrel(rebin_values(src, sc.edges("x200"), sc.edges(tgt)),
                                                        sc.per_halo(key(f, tgt)))
    # T_e: electron-weighted (E x V) rebin of the per-halo ratio
    E200 = np.asarray(sc.per_halo(key("E", "x200")))
    T200 = np.asarray(sc.per_halo(key("T_e", "x200")))
    w = E200 * np.diff(sc.edges("x200")**3)[None, :]
    rebin["per_halo"][f"T_e: x200->{tgt} (electron-weighted)"] = maxrel(
        rebin_values(T200, sc.edges("x200"), sc.edges(tgt), weights=np.where(w > 0, w, 0)), sc.per_halo(key("T_e", tgt)))
    # K: K E^(2/3) / T_e is a per-halo constant on every grid
    Kc = np.asarray(sc.per_halo(key("K", tgt))) * np.asarray(sc.per_halo(key("E", tgt)))**(2 / 3) / np.asarray(
        sc.per_halo(key("T_e", tgt)))
    K2 = np.asarray(sc.per_halo(key("K", "x200"))) * E200**(2 / 3) / T200
    const = np.nanmedian(K2, axis=1)
    rebin["per_halo"][f"K: K E^(2/3)/T_e constant per halo, x200 and {tgt}"] = max(
        maxrel(Kc, np.broadcast_to(const[:, None], Kc.shape)), maxrel(K2, np.broadcast_to(const[:, None], K2.shape)))
for f in ("P", "E", "G"):
    rebin["per_halo"][f"{f}: x500->b24"] = maxrel(
        rebin_values(np.asarray(sc.per_halo(key(f, "x500"))), sc.edges("x500"), sc.edges("b24")),
        sc.per_halo(key(f, "b24")))
# stack level: the fit-grid stack (sample and every bootstrap replicate) vs the rebinned x200 stack
for f in ("P", "E", "G"):
    worst = 0.0
    for b in BINS:
        a = rebin_values(C.all_rows(STK[(f"{C.GROUP}:x200", f, b["label"])]), sc.edges("x200"), FIT_E)
        worst = max(worst, maxrel(a, REF_FIT[(f, b["label"])]))
    rebin["stack"][f"{f}: mean stack + 200 replicates, x200 rebinned vs fit"] = worst
# ratio fields cannot be rebinned after stacking: volume- or stacked-electron-weighted rebin of the x200 T_e stack
dev_v, dev_e = [], []
for b in CAL:
    lab = b["label"]
    t = STK[(f"{C.GROUP}:x200", "T_e", lab)]["value"]
    e = STK[(f"{C.GROUP}:x200", "E", lab)]["value"]
    ok = STK[("ref", "T_e", lab)]["resolved"]
    ref = STK[("ref", "T_e", lab)]["value"]
    tv = rebin_values(t[None], sc.edges("x200"), FIT_E)[0]
    te = rebin_values(t[None], sc.edges("x200"), FIT_E, weights=(e * np.diff(sc.edges("x200")**3))[None])[0]
    dev_v.append(np.nanmax(np.abs(tv[ok] / ref[ok] - 1)))
    dev_e.append(np.nanmax(np.abs(te[ok] / ref[ok] - 1)))
rebin["ratio_fields_after_stacking"] = {
    "T_e stack: volume-weighted rebin of x200 stack vs fit stack, max |dev| over calibrated bins, resolved shells":
        float(np.max(dev_v)),
    "T_e stack: stacked-electron-weighted rebin vs fit stack, max |dev|": float(np.max(dev_e))}
# medians do not rebin
devm = {}
for f in ("P", "E"):
    d = []
    for b in CAL:
        mf = sc.stack(np.asarray(sc.per_halo(key(f, "fit"))[b["rows"]]), b["weights"], "median", n_boot=0)["value"]
        mx = sc.stack(np.asarray(sc.per_halo(key(f, "x200"))[b["rows"]]), b["weights"], "median", n_boot=0)["value"]
        mxr = rebin_values(mx[None], sc.edges("x200"), FIT_E)[0]
        ok = STK[("ref", f, b["label"])]["resolved"]
        d.append(float(np.nanmax(np.abs(mxr[ok] / mf[ok] - 1))))
    devm[f] = {"max_abs_dev_over_calibrated_bins": max(d), "per_bin": dict(zip(CAL_LABELS, d))}
rebin["median_after_stacking"] = devm

# conservative-remap accuracy on grids where the truth is known (old -> fit, lee22 -> fit): PCHIP (used) vs
# constant density inside each source shell; max |dev| and median |dev| over resolved shells of the calibrated bins
remap_acc = {}
for g in ("old", "lee22"):
    for f in ("P", "E"):
        for meth, fn in C.REMAP_METHODS.items():
            d, where, allabs = [], [], []
            for b in CAL:
                v = STK[(f"{C.GROUP}:{g}", f, b["label"])]
                ref = REF_FIT[(f, b["label"])][0]
                rm = fn(v["value"], sc.edges(g), FIT_E)[0]
                ok = C.remap_mask(v["resolved"], sc.edges(g), FIT_E) & STK[("ref", f, b["label"])]["resolved"] \
                    & np.isfinite(rm)
                dev = np.abs(rm[ok] / ref[ok] - 1)
                allabs.extend(dev.tolist())
                k = int(np.argmax(dev))
                d.append(float(dev[k]))
                where.append([float(FIT_E[np.flatnonzero(ok)[k]]), float(FIT_E[np.flatnonzero(ok)[k] + 1])])
            j = int(np.argmax(d))
            remap_acc[f"{f}: {g} -> fit, {meth}"] = {"max_abs_dev": max(d), "median_abs_dev": float(np.median(allabs)),
                                                     "worst_bin": CAL_LABELS[j], "worst_shell_R200c": where[j]}
rebin["remap_accuracy"] = remap_acc
print(f"rebinning checks  {time.time() - T0:.0f} s", flush=True)

# resolution masks per grid
resmask = {}
for g in ("fit", "x200", "lee22", "old", "x500", "b24"):
    e = sc.edges(g)
    resmask[g] = {}
    for b in BINS:
        m = sc.resolution_mask(b["rows"], g, "strict")
        idx = np.flatnonzero(m)
        holes = [[float(e[k]), float(e[k + 1])] for k in range(idx[0], idx[-1]) if not m[k]] if idx.size else []
        resmask[g][b["label"]] = {"first_resolved_edge": float(e[idx[0]]) if idx.size else None,
                                  "first_resolved_edge_R200c": (float(e[idx[0]]) * bin_mean(Q500, b)
                                                                if (idx.size and sc.grid_radius(g) == "R500c")
                                                                else (float(e[idx[0]]) if idx.size else None)),
                                  "n_resolved": int(m.sum()), "holes": holes}

# ============================================================================ normalisation checks
norm_checks = {}
rows_all = np.arange(len(Q500))
P_fit = np.asarray(sc.per_halo(key("P", "fit")))
E_fit = np.asarray(sc.per_halo(key("E", "fit")))
P_x500 = np.asarray(sc.per_halo(key("P", "x500")))
Y200 = sc.integral(P_fit, "fit", 1.0) * TAB["p200_pa"] * TAB["r200c"]**3 / C.PE_FACTOR
Y4R = sc.integral(P_fit, "fit", 4.0) * TAB["p200_pa"] * TAB["r200c"]**3 / C.PE_FACTOR
Y500 = sc.integral(P_x500, "x500", 1.0) * TAB["p500_B24_pa"] * TAB["r500c"]**3 / C.PE_FACTOR
NE200 = sc.integral(E_fit, "fit", 1.0) * TAB["r200c"]**3 * RHO_C200 * FB / C.MU_XGP_MSUN
NE4R = sc.integral(E_fit, "fit", 4.0) * TAB["r200c"]**3 * RHO_C200 * FB / C.MU_XGP_MSUN


def relstats(a, b):
    d = a / b - 1
    return {"median": float(np.median(d)), "max_abs": float(np.max(np.abs(d)))}


norm_checks["Y(<R200c) from P/P200 x P200 R200c^3 vs halo_table Y200_own"] = relstats(Y200, TAB["Y200_own_mpc2"])
norm_checks["Y(<4R200c) vs Y4R_own"] = relstats(Y4R, TAB["Y4R_own_mpc2"])
norm_checks["Y(<R500c) from x500 P/P_e,500(B24) x P_e,500 R500c^3 vs Y500_own"] = relstats(Y500, TAB["Y500_own_mpc2"])
norm_checks["N_e(<R200c) from E x rho_c f_b / mu_XGP vs Ne200_own"] = relstats(NE200, TAB["Ne200_own"])
norm_checks["N_e(<4R200c) vs Ne4R_own"] = relstats(NE4R, TAB["Ne4R_own"])
for (f, n500, n200) in (("P", "B24", "P200"), ("P", "A10", "P200"), ("P", "self_similar_500", "P200"),
                        ("E", "B24", "rhoc_fb")):
    a = np.asarray(sc.per_halo(key(f, "x500", n200)))
    bb = np.asarray(sc.per_halo(key(f, "x500", n500)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nanmedian(np.where(bb > 0, a / bb, np.nan), axis=1)
    ok = np.isfinite(r)
    norm_checks[f"{f}: cached x500 {n200}/{n500} array ratio vs analytic per-halo factor"] = relstats(
        r[ok], FACT[(f, n500)][ok])
norm_checks["P_e,500(A10)/P_e,500(B24): median, min, max"] = [float(np.median(FACT[("P", "A10")] / FACT[("P", "B24")])),
                                                              float(np.min(FACT[("P", "A10")] / FACT[("P", "B24")])),
                                                              float(np.max(FACT[("P", "A10")] / FACT[("P", "B24")]))]
norm_checks["A10 implied f_b(A10)/f_B(B24) = 0.175/0.159"] = 0.175 / 0.159
norm_checks["self_similar_500/B24 (= (f_b/f_B) mu_e/mu)"] = [float(np.median(FACT[("P", "self_similar_500")] /
                                                                             FACT[("P", "B24")])),
                                                            (FB / 0.159) * 1.14 / 0.59]
norm_checks["mu/mu_e = 0.59/1.14 (electron/thermal pressure; B12 0.5176 prefactor)"] = 0.59 / 1.14
norm_checks["P_e,500(B24)/P200: median, 5th, 95th pct"] = [float(np.median(FACT[("P", "B24")])),
                                                           *map(float, np.percentile(FACT[("P", "B24")], [5, 95]))]
norm_checks["E_ref/E_B24 constant (mu_e m_p ... ): median, max rel spread"] = [
    float(np.median(FACT[("E", "B24")])), float(np.max(np.abs(FACT[("E", "B24")] / np.median(FACT[("E", "B24")]) - 1)))]
norm_checks["rho_c from (M200c,R200c) per halo: max rel spread"] = float(np.max(np.abs(RHO_C200 / np.median(RHO_C200)
                                                                                       - 1)))
norm_checks["rho_c from (M500c,R500c) per halo: max rel spread"] = float(np.max(np.abs(RHO_C500 / np.median(RHO_C500)
                                                                                       - 1)))
norm_checks["rho_c(cosmology) = 2.775e11 h^2 Msun/Mpc^3 vs median halo rho_c200"] = float(
    np.median(RHO_C200) / (2.77536627e11 * H**2) - 1)
norm_checks["R500c/R200c: median, 5th, 95th, min, max"] = [float(np.median(Q500)),
                                                           *map(float, np.percentile(Q500, [5, 95, 0, 100]))]
norm_checks["fraction of haloes with 6 R500c > 4 R200c (unweighted sample of 4588)"] = float(np.mean(6 * Q500 > 4))
norm_checks["fraction with 6 R500c > 4 R200c, inclusion-weighted population 12.5-15.5"] = float(
    np.average(6 * Q500 > 4, weights=TAB["weight"]))
_calrows = np.concatenate([b["rows"] for b in CAL])
_calw = np.concatenate([b["weights"] for b in CAL])
norm_checks["fraction with 6 R500c > 4 R200c, inclusion-weighted population 13.5-15.5"] = float(
    np.average(6 * Q500[_calrows] > 4, weights=_calw))
norm_checks["6 R500c / R200c: median"] = float(np.median(6 * Q500))
norm_checks["T500/T200 median; K500/K200 median"] = [float(np.median(FACT[("T_e", "T500")])),
                                                    float(np.median(FACT[("K", "K500")]))]
print(f"normalisation checks  {time.time() - T0:.0f} s", flush=True)

# ============================================================================ (2) self-similarity and scatter
QMED = float(np.median(Q500))
X_EQ = np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])     # R200c-equivalent radii
X_DENSE = np.geomspace(0.05, 3.8, 120)
WELL = [b for b in CAL if b["hi"] <= 15.1 + 1e-9]                                  # 13.5-15.1 (N >= 127)
SCALINGS = {
    "P": {"R200c, P200 (ref)": ("ref", "P"), "R200c, P_e,500 (B24)": (f"{C.GROUP}:fit_P500", "P"),
          "R500c, P200": (f"{C.GROUP}:x500_P200", "P"), "R500c, P_e,500 (B24)": (f"{C.GROUP}:x500_B24", "P"),
          "R500c, P500 (M500/3e14)^0.12 (A10 full)": (f"{C.GROUP}:x500_A10M", "P")},
    "E": {"R200c, rho_c f_b (ref)": ("ref", "E"), "R500c, n_e,500 (B24)": (f"{C.GROUP}:x500_B24", "E")},
    "T_e": {"R200c, T200 (ref)": ("ref", "T_e"), "R500c, T500": (f"{C.GROUP}:x500_B24", "T_e")},
    "K": {"R200c, K200 (ref)": ("ref", "K"), "R500c, K500": (f"{C.GROUP}:x500_B24", "K")},
}


def interp_ln(res, xq, edges, use_resolved=True, what="value"):
    """ln(res[what]) at xq by linear interpolation in (ln x, ln F) between shell centres (no extrapolation)."""
    xc = C.xcentre(edges)
    vals = np.asarray(res[what], float)
    ok = np.isfinite(vals) & (vals > 0)
    if use_resolved:
        ok &= np.asarray(res["resolved"], bool)
    out = np.full(len(xq), np.nan)
    if ok.sum() < 2:
        return out
    lx, lv = np.log(xc[ok]), np.log(vals[ok])
    inside = (np.log(xq) >= lx[0]) & (np.log(xq) <= lx[-1])
    # do not interpolate across holes in the resolved range wider than one unresolved shell
    out[inside] = np.interp(np.log(xq[inside]), lx, lv)
    return out


selfsim = {}
figss = {}
for f, scal in SCALINGS.items():
    selfsim[f] = {}
    figss[f] = {}
    for name, (vid, fld) in scal.items():
        g = VBYID[vid]["fields"][fld][2] if VBYID[vid]["fields"][fld][0] in ("key", "derived") else "fit"
        r500 = sc.grid_radius(g) == "R500c"
        xq = X_EQ / QMED if r500 else X_EQ
        xd = X_DENSE / QMED if r500 else X_DENSE
        lnF, lnFd, sdl, sdld, ferr, fes = [], [], [], [], [], []
        for b in CAL:
            res = STK[(vid, fld, b["label"])]
            lnF.append(interp_ln(res, xq, sc.edges(g)))
            lnFd.append(interp_ln(res, xd, sc.edges(g)))
            sdl.append(np.exp(interp_ln({"value": res["sd_ln"], "resolved": res["resolved"]}, xq, sc.edges(g))))
            sdld.append(np.exp(interp_ln({"value": res["sd_ln"], "resolved": res["resolved"]}, xd, sc.edges(g))))
            fe = np.asarray(res["err"]) / np.asarray(res["value"])
            ferr.append(np.exp(interp_ln({"value": fe, "resolved": res["resolved"]}, xq, sc.edges(g))))
            fes.append(np.exp(interp_ln({"value": fe, "resolved": res["resolved"]}, xd, sc.edges(g))))
        lnF, lnFd, sdl, sdld, ferr, fes = map(np.asarray, (lnF, lnFd, sdl, sdld, ferr, fes))
        logm = np.array([b["mean_logm200c"] for b in CAL])
        well = np.array([b["hi"] <= 15.1 + 1e-9 for b in CAL])
        spread, spread_all, slope, nb, noise = [], [], [], [], []
        for j in range(len(X_EQ)):
            okw = well & np.isfinite(lnF[:, j])
            oka = np.isfinite(lnF[:, j])
            nb.append(int(okw.sum()))
            spread.append(float(np.std(lnF[okw, j], ddof=1)) if okw.sum() >= 5 else None)
            spread_all.append(float(np.std(lnF[oka, j], ddof=1)) if oka.sum() >= 5 else None)
            slope.append(float(np.polyfit(logm[okw], lnF[okw, j], 1)[0]) if okw.sum() >= 5 else None)
            noise.append(float(np.nanmedian(ferr[okw, j])) if okw.sum() >= 5 else None)
        dense_spread = [float(np.std(lnFd[well & np.isfinite(lnFd[:, j]), j], ddof=1))
                        if (well & np.isfinite(lnFd[:, j])).sum() >= 5 else np.nan for j in range(len(X_DENSE))]
        dense_noise = [float(np.nanmedian(fes[well & np.isfinite(lnFd[:, j]), j]))
                       if (well & np.isfinite(lnFd[:, j])).sum() >= 5 else np.nan for j in range(len(X_DENSE))]
        selfsim[f][name] = {
            "x_R200c_equivalent": X_EQ, "x_own_units": xq, "grid": g,
            "spread_sd_lnF_across_bins_13.5-15.1": spread, "spread_sd_lnF_across_all_10_calibrated_bins": spread_all,
            "dlnF_dlog10M200c_13.5-15.1": slope, "n_bins_used": nb,
            "median_bootstrap_fractional_error_of_stack": noise,
            "within_bin_sd_ln_per_bin": {b["label"]: sdl[i] for i, b in enumerate(CAL)},
            "within_bin_sd_ln_median_over_13.5-15.1": [float(np.nanmedian(sdl[well, j])) if np.isfinite(
                sdl[well, j]).sum() >= 3 else None for j in range(len(X_EQ))]}
        figss[f][name] = {"x_dense": X_DENSE, "spread_dense": np.asarray(dense_spread), "noise_dense":
                          np.asarray(dense_noise), "sdln_dense": sdld, "lnF_dense": lnFd, "logm": logm, "well": well,
                          "grid": g, "r500": r500}
# common-bin comparison: per field, only the 13.5-15.1 bins resolved at that radius in EVERY scaling of the field
for f in figss:
    names = list(figss[f])
    lnq = {}
    for name, (vid, fld) in SCALINGS[f].items():
        g = VBYID[vid]["fields"][fld][2] if VBYID[vid]["fields"][fld][0] in ("key", "derived") else "fit"
        xq = X_EQ / QMED if sc.grid_radius(g) == "R500c" else X_EQ
        lnq[name] = np.asarray([interp_ln(STK[(vid, fld, b["label"])], xq, sc.edges(g)) for b in CAL])
        sd_q = np.asarray([np.exp(interp_ln({"value": STK[(vid, fld, b["label"])]["sd_ln"],
                                             "resolved": STK[(vid, fld, b["label"])]["resolved"]}, xq, sc.edges(g)))
                           for b in CAL])
        lnq[name + "__sd"] = sd_q
    well = np.array([b["hi"] <= 15.1 + 1e-9 for b in CAL])
    common = well[:, None] & np.all([np.isfinite(lnq[n]) for n in names], axis=0)
    for name in names:
        sp, sdm, nb = [], [], []
        for j in range(len(X_EQ)):
            m = common[:, j]
            nb.append(int(m.sum()))
            sp.append(float(np.std(lnq[name][m, j], ddof=1)) if m.sum() >= 4 else None)
            sdm.append(float(np.median(lnq[name + "__sd"][m, j])) if m.sum() >= 4 else None)
        selfsim[f][name]["common_bins_spread_sd_lnF"] = sp
        selfsim[f][name]["common_bins_within_bin_sd_ln_median"] = sdm
        selfsim[f][name]["common_bins_n"] = nb
print(f"self-similarity  {time.time() - T0:.0f} s", flush=True)

# shot-noise context: median gas-particle count per shell at the comparison radii (fit vs x500)
counts_ctx = {}
for g in ("fit", "x500"):
    cnt = np.asarray(sc.counts(g))
    xq = X_EQ / QMED if g == "x500" else X_EQ
    idx = np.clip(np.searchsorted(sc.edges(g), xq) - 1, 0, len(sc.edges(g)) - 2)
    counts_ctx[g] = {b["label"]: [float(np.median(cnt[b["rows"], k])) for k in idx] for b in CAL}

# ============================================================================ (3) cumulative budgets
def fit_edge(u):
    """The stored fit-grid edge closest to u (asserting it is within 1e-3): sc.integral(..., upto) counts shells with
    outer edge <= upto + 1e-12, so a rounded non-round edge (e.g. 3.6342 for 3.634241) would drop a shell."""
    k = int(np.argmin(np.abs(FIT_E - u)))
    if abs(FIT_E[k] - u) > 1e-3:
        raise ValueError(f"{u} is not a fit-grid edge")
    return float(FIT_E[k])


CUM_X = (0.442, 1.0, 1.34, 1.5, 2.0, 2.5, 3.0)       # 0.442027 is the fit-grid edge nearest 0.5 R200c
cumul = {}
figcum = {}
for f in ("P", "E", "G"):
    cumul[f] = {}
    figcum[f] = {}
    for b in BINS:
        res = STK[("ref", f, b["label"])]
        rows = C.all_rows(res)
        tot = sc.integral(rows, "fit", 4.0)
        cum_edges = np.stack([sc.integral(rows, "fit", u) for u in FIT_E[1:]], axis=1) / tot[:, None]
        out = {}
        for u in CUM_X:
            fr = 1 - sc.integral(rows, "fit", fit_edge(u)) / tot
            out[f"beyond_{u:g}"] = {"value": float(fr[0]), "err": float(np.std(fr[1:], ddof=1)),
                                    "edge_R200c": fit_edge(u)}
        # share per unit ln r over the outermost shells (how fast the integral still grows at 4R); exact stored
        # edges (the earlier version used 3.6342 < 3.634241 and so divided two shells by the width of one)
        for (a, bb, lab_) in ((fit_edge(3.0), fit_edge(4.0), "3-4"), (fit_edge(3.634), fit_edge(4.0), "3.634-4")):
            sh = (sc.integral(rows, "fit", bb) - sc.integral(rows, "fit", a)) / tot / np.log(bb / a)
            out[f"dlnI_dlnx_{lab_}"] = {"value": float(sh[0]), "err": float(np.std(sh[1:], ddof=1)),
                                        "edges_R200c": [a, bb]}
        last = (sc.integral(rows, "fit", fit_edge(4.0)) - sc.integral(rows, "fit", fit_edge(3.634))) / tot
        out["share_in_last_shell_3.634-4"] = {"value": float(last[0]), "err": float(np.std(last[1:], ddof=1))}
        # local log slope of the stacked profile between the 3-3.30 and 3.63-4 shells (exact shell indices)
        v = res["value"]
        i1 = int(np.flatnonzero(np.isclose(FIT_E, fit_edge(3.0)))[0])
        i2 = int(np.flatnonzero(np.isclose(FIT_E, fit_edge(3.634)))[0])
        out["profile_log_slope_3.15_to_3.8"] = float(np.log(v[i2] / v[i1]) / np.log(XF[i2] / XF[i1]))
        out["profile_log_slope_shell_radii"] = [float(XF[i1]), float(XF[i2])]
        # per-halo fractions (population spread)
        ph = np.asarray(sc.per_halo(key(f, "fit"))[b["rows"]])
        pt = sc.integral(ph, "fit", 4.0)
        for u in (1.0, 2.0):
            fr_h = 1 - sc.integral(ph, "fit", fit_edge(u)) / pt
            o = np.argsort(fr_h)
            cw = np.cumsum(b["weights"][o]) / b["weights"].sum()
            out[f"per_halo_beyond_{u:g}_q16_q50_q84"] = [float(fr_h[o][np.searchsorted(cw, q)]) for q in
                                                         (0.16, 0.5, 0.84)]
        cumul[f][b["label"]] = out
        figcum[f][b["label"]] = {"cum": cum_edges[0], "cum_lo": np.percentile(cum_edges[1:], 16, axis=0),
                                 "cum_hi": np.percentile(cum_edges[1:], 84, axis=0), "logm": b["mean_logm200c"]}

# R500c-unit budgets on the x500 grid, M500c_0.2 bins: Y(<R500c)/Y(<5 R500c) (A10 universal shape: computed below)
cum500 = {}
for f in ("P", "E"):
    cum500[f] = {}
    for b in sc.mass_bins("M500c_0.2"):
        if b["lo"] < 13.5 - 1e-9:
            continue
        res = sc.stack_bin(key(f, "x500"), b, "mean", "strict", n_boot=C.N_BOOT, return_boot=True)
        rows = C.all_rows(res)
        r = sc.integral(rows, "x500", 1.0) / sc.integral(rows, "x500", 5.0)
        r6 = sc.integral(rows, "x500", 1.0) / sc.integral(rows, "x500", 6.0)
        cum500[f][b["label"]] = {"I(<R500c)/I(<5R500c)": {"value": float(r[0]), "err": float(np.std(r[1:], ddof=1))},
                                 "I(<R500c)/I(<6R500c)": {"value": float(r6[0]), "err": float(np.std(r6[1:], ddof=1))},
                                 "N": b["N"]}


def a10_I(x, P0=8.403, c500=1.177, gam=0.3081, alp=1.0510, bet=5.4905):
    u = np.geomspace(1e-6, x, 20001)
    cx = c500 * u
    p = P0 / (cx**gam * (1 + cx**alp)**((bet - gam) / alp))
    return np.trapz(3 * p * u**2, u)


cum500["A10_universal_shape_I(1)/I(5)"] = a10_I(1.0) / a10_I(5.0)
cum500["A10_universal_shape_I(1)"] = a10_I(1.0)
cum500["A10_universal_shape_I(5)"] = a10_I(5.0)

# extraction-sphere reach per bin
reach = {b["label"]: {"mean_6R500c_over_R200c": float(np.average(6 * Q500[b["rows"]], weights=b["weights"])),
                      "fraction_6R500c_gt_4R200c": float(np.average(6 * Q500[b["rows"]] > 4, weights=b["weights"])),
                      "mean_R500c_over_R200c": bin_mean(Q500, b)} for b in BINS}

# first-order truncation impacts (cut applied to the reference stack: shells with outer edge > cut removed)
CUTS = (1.0, 1.34, 1.5, 1.75, 2.0, 2.5, 3.0)
trunc = {}
IMP = sc._importance()          # read-only: Test F's importance Jacobians [16 engine bins, 35 fit shells(, ell)]
CALB = list(sc.CAL_BINS)


def imp_matrix(o):
    ikey, at, k, _fld, _desc = sc.OBSERVABLES[o]
    M = IMP[ikey]
    if at is not None:
        M = M[..., int(np.flatnonzero(IMP["ell"] == at)[0])]
    return np.asarray(M, float), k


for f, obs in (("P", C.OBS_P), ("E", C.OBS_E)):
    trunc[f] = {}
    for cut in CUTS:
        removed = FIT_E[1:] > fit_edge(cut) + 1e-9
        dln = np.where(removed[None, :], -50.0, 0.0) * np.ones((len(CAL), 1))
        r = C.impact_dln(dln, f, obs)
        for o in obs:
            M, k = imp_matrix(o)
            rem_cal = float(M[CALB][:, removed].sum())          # removed importance, calibrated bins only
            rem_all = float(M[:, removed].sum())                # all 16 engine bins (fallback profiles outside)
            r[o]["removed_importance_calibrated_bins"] = rem_cal
            r[o]["removed_importance_all_engine_bins"] = rem_all
            r[o]["dO_over_O_first_order_cut_all_engine_bins"] = -k * rem_all
            if k == 2:
                # estimates of the exact (second-order) change of a quadratic observable
                # (a) single coherent amplitude: O ~ A^2, A -> A (1 - rem)   -> (1 - rem)^2 - 1
                # (b) independent bins (1-halo form): sum_b share_b [(1 - rem_b/share_b)^2 - 1]
                sb, rb = M[CALB].sum(axis=1), M[CALB][:, removed].sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    per_bin = np.where(sb > 0, sb * ((1 - rb / sb)**2 - 1), 0.0)
                sba, rba = M.sum(axis=1), M[:, removed].sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    per_bin_a = np.where(sba > 0, sba * ((1 - rba / sba)**2 - 1), 0.0)
                r[o]["quadratic_estimate_single_amplitude_calibrated_bins"] = (1 - rem_cal)**2 - 1
                r[o]["quadratic_estimate_independent_bins_calibrated_bins"] = float(per_bin.sum())
                r[o]["quadratic_estimate_single_amplitude_all_engine_bins"] = (1 - rem_all)**2 - 1
                r[o]["quadratic_estimate_independent_bins_all_engine_bins"] = float(per_bin_a.sum())
        trunc[f][f"{cut:g}"] = r
trunc["note"] = ("dO_over_O: first order (sc.observable_impact), cut applied to the 10 calibrated bins only (uncalibrated "
                 "haloes held fixed; exact for y_mean and DM_mean for that partial cut).  *_all_engine_bins: the same "
                 "cut applied to all 16 Test F engine bins (fallback profiles outside 13.5-15.5), first order from "
                 "importance_F.npz.  quadratic_estimate_*: second-order estimates for k = 2 observables, (a) one "
                 "coherent amplitude (1 - removed)^2 - 1, (b) independent bins sum_b share_b [(1 - removed_b/share_b)^2 "
                 "- 1]; the per-halo exact 1-halo change lies closer to zero than (b) when the removed fraction varies "
                 "between haloes.")
print(f"cumulative / truncation  {time.time() - T0:.0f} s", flush=True)

# ============================================================================ (4) clumping and SPH volume filling
clump = {}
figclump = {}
for b in BINS:
    lab = b["label"]
    rows = b["rows"]
    Ggeo = np.asarray(sc.per_halo(key("G", "fit"))[rows])
    Gqv = np.asarray(sc.per_halo(key("G", "fit", estimator="Q_V"))[rows])
    Gqm = np.asarray(sc.per_halo(key("G", "fit", estimator="Q_M"))[rows])
    Pqv = np.asarray(sc.per_halo(key("P", "fit", estimator="Q_V"))[rows])
    Pqvy = np.asarray(sc.per_halo(key("P", "fit", estimator="Q_V_y"))[rows])
    Crho = np.asarray(sc.per_halo(key("C_rho", "fit", "none"))[rows])
    Cp = np.asarray(sc.per_halo(key("C_P", "fit", "none"))[rows])
    with np.errstate(invalid="ignore", divide="ignore"):
        fV = np.where(Gqv > 0, Ggeo / Gqv, np.nan)
        ident = (Gqm / Ggeo) / (Crho / fV) - 1
    st = {}
    for name, arr in (("C_rho", Crho), ("C_P", Cp), ("f_V", fV)):
        st[name] = C.stack_values(arr, b, "fit")
    ok = STK[("ref", "P", lab)]["resolved"]
    pg = STK[("ref", "P", lab)]["value"]
    gg = STK[("ref", "G", lab)]["value"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = {"P_QV/P_geo (ratio of stacks)": np.average(np.where(np.isfinite(Pqv), Pqv, 0), axis=0,
                                                             weights=b["weights"]) / pg,
                  "P_QVy/P_geo (ratio of stacks)": np.average(np.where(np.isfinite(Pqvy), Pqvy, 0), axis=0,
                                                              weights=b["weights"]) / pg,
                  "G_QM/G_geo (ratio of stacks)": np.average(np.where(np.isfinite(Gqm), Gqm, 0), axis=0,
                                                             weights=b["weights"]) / gg}
    bandv = {}
    for nm, arr in [("C_rho", st["C_rho"]["value"]), ("C_P", st["C_P"]["value"]), ("f_V", st["f_V"]["value"])] + \
            list(ratios.items()):
        bandv[nm] = {}
        for (a, bb), bl in zip(C.BANDS, C.BAND_LABELS):
            sel = ok & (XF >= a) & (XF < bb) & np.isfinite(arr)
            bandv[nm][bl] = float(np.median(arr[sel])) if sel.any() else None
    clump[lab] = {"band_medians_resolved": bandv,
                  "identity_G_QM/G_geo_eq_C_rho/f_V_per_halo_max_abs_dev": float(np.nanmax(np.abs(ident)))}
    figclump[lab] = {"C_rho": st["C_rho"]["value"], "C_rho_err": st["C_rho"]["err"], "C_P": st["C_P"]["value"],
                     "C_P_err": st["C_P"]["err"], "f_V": st["f_V"]["value"], "f_V_err": st["f_V"]["err"],
                     "resolved": ok, "logm": b["mean_logm200c"], **{k: v for k, v in ratios.items()}}
print(f"clumping  {time.time() - T0:.0f} s", flush=True)

# ============================================================================ save
main = {"group": C.GROUP, "reference": {"key_template": TEMPLATE_STD % "fit", "statistic": "mean", **C.RES,
                                        "scheme": C.SCHEME, "n_boot": C.N_BOOT, "seed": "core default per bin"},
        "bands_R200c": C.BAND_LABELS, "calibrated_bins": CAL_LABELS, "variants": VARIANT_RESULTS,
        "rebinning": rebin, "resolution_masks": resmask, "normalisation_checks": norm_checks,
        "self_similarity": selfsim, "counts_context_median_per_shell": counts_ctx, "R500c_over_R200c_median": QMED,
        "cumulative_fractions_fit_grid": cumul, "cumulative_R500c_units_M500c_bins": cum500,
        "extraction_reach": reach, "truncation_first_order": trunc, "clumping": clump,
        "remap_method_sensitivity": REMAP_SENS,
        "runtime_s": time.time() - T0}
(WORK / "main_results.json").write_text(json.dumps(C.jsonify(main), indent=1))

fig = {"stk": {k: {kk: vv for kk, vv in v.items() if kk not in ("boot",)} for k, v in STK.items()},
       "conv": {k: {"value": v["rows"][0], "err": np.nanstd(v["rows"][1:], axis=0, ddof=1), "ok": v["ok"]}
                for k, v in CONV.items()},
       "selfsim": figss, "cum": figcum, "clump": figclump, "bins": [{k: b[k] for k in ("label", "lo", "hi",
                                                                                        "mean_logm200c", "N")}
                                                                   for b in BINS],
       "cal": CAL_LABELS, "q500": Q500, "variants": [{k: v[k] for k in ("id", "label", "grid", "x_unit")}
                                                     for v in VARIANTS]}
with open(WORK / "figdata_main.pkl", "wb") as fh:
    pickle.dump(fig, fh)
print(f"done  {time.time() - T0:.0f} s", flush=True)
