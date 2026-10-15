"""Checks behind the corrected SECTION.md of group 'resolution' (text-only corrections, 2026-10-14).

Reads results.json (and, for the few numbers not stored there, the cache arrays through res_common / plain numpy)
and prints / writes every number the corrected section quotes that the verifier flagged or that needed a re-check.
Writes section_checks.json next to this script.  Run:
  LD_LIBRARY_PATH=<env>/lib <env>/bin/python -B section_checks.py
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
E = np.load(OUT / "cache" / "edges_fit.npy")
XIN = E[:-1]
C = np.load(OUT / "cache" / "counts_fit.npy")
H = np.load(OUT / "cache" / "hsml_fit.npy")
T = np.load(OUT / "cache" / "halo_table.npz")
EPS = T["eps_over_r200c"]
BINS = {b["label"]: b for b in sc.mass_bins("M200c_0.2")}
LABS = list(BINS)
CAL = LABS[5:]
OBS = ["tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000", "tSZ_yy_1h_ell3000", "y_mean",
       "kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var"]
out = {}


def first(mask):
    i = np.flatnonzero(mask)
    return (float(E[i[0]]), int(i[0])) if len(i) else (None, None)


def pr(*a):
    print(*a)


# ---------------------------------------------------------------- 1. fit grid
w = np.log10(E[2:] / E[1:-1])
out["grid"] = {"n_shells": len(E) - 1, "central_sphere": [0.0, float(E[1])],
               "dex_0.025_to_0.837": [float(w[:22].min()), float(w[:22].max())], "n_log_shells": 22,
               "dex_0.837_to_1": float(w[22]), "edges_beyond_1": [float(v) for v in E[24:]],
               "dex_beyond_1": [float(w[23:].min()), float(w[23:].max())], "n_beyond_1": int(len(E) - 1 - 24)}
pr("grid", out["grid"])

# ---------------------------------------------------------------- 2. h-alone edges at factor 1 / 2, count-alone
alone = {}
for lab in LABS:
    rows = BINS[lab]["rows"]
    c, h = C[rows], H[rows]
    with np.errstate(invalid="ignore"):
        h1 = (XIN[None, :] >= 1.0 * h).all(0)
        h2 = (XIN[None, :] >= 2.0 * h).all(0)
    cnt = (c >= 20).all(0)
    alone[lab] = {"h1": first(h1)[0], "h2": first(h2)[0], "count20": first(cnt)[0],
                  "h1_idx": first(h1)[1], "count20_idx": first(cnt)[1],
                  "hf1_first_edge": PS["resolution:hf1"]["summary"]["first_edge"][lab],
                  "hf0_first_edge": PS["resolution:hf0"]["summary"]["first_edge"][lab],
                  "mconly20_first_edge": PS["resolution:mconly20"]["summary"]["first_edge"][lab]}
out["alone_edges"] = alone
pr("\nh-alone (factor 1) vs count-alone (>= 20) edges")
for lab in LABS:
    a = alone[lab]
    d = None if a["h1_idx"] is None or a["count20_idx"] is None else a["count20_idx"] - a["h1_idx"]
    pr(f"  {lab}: h1 {a['h1']}  count20 {a['count20']}  (count edge {d} shells outside h1)  hf1 {a['hf1_first_edge']}"
       f"  hf0 {a['hf0_first_edge']}  mconly20 {a['mconly20_first_edge']}")

# ---------------------------------------------------------------- 3. 12.5-12.7: why nothing is kept
rows = BINS["12.5-12.7"]["rows"]
c, h = C[rows], H[rows]
with np.errstate(invalid="ignore"):
    hpass = (XIN[None, :] >= 2.0 * h).all(0)
i_h = np.flatnonzero(hpass)
info = {"h_pass_shells": [[float(E[i]), float(E[i + 1])] for i in i_h]}
for i in range(len(XIN) - 3, len(XIN)):
    low = np.flatnonzero(c[:, i] < 20)
    info[f"{E[i]:.3f}-{E[i + 1]:.3f}"] = {"n_haloes_lt20": int(len(low)), "min_count": int(c[:, i].min()),
                                          "halo_rows_lt20": [int(rows[j]) for j in low[:3]],
                                          "h_pass": bool(hpass[i])}
out["bin_12.5-12.7"] = info
pr("\n12.5-12.7", json.dumps(info))

# ---------------------------------------------------------------- 4. per-halo band ratios (all calibrated bins)
sbr = PS["resolution:per_halo"]["stack_band_ratios"]
bands = {}
for f in ("P", "E"):
    for lab in LABS:
        for b in ("0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1"):
            v = sbr[f][lab][b]
            bands[f"{f}|{lab}|{b}"] = v
out["per_halo_band"] = bands
pr("\nper-halo / reference band ratios (median over kept shells) [ratio, err, n_shells]")
for f in ("P", "E"):
    for b in ("0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1"):
        pr(f"  {f} {b}: " + "; ".join(f"{lab} {bands[f'{f}|{lab}|{b}']['ratio']}+-{bands[f'{f}|{lab}|{b}']['err']}"
                                       f"(n{bands[f'{f}|{lab}|{b}']['n_shells']})" for lab in LABS))

# per-shell ratio and resolved fraction in 13.5-13.7 (bands >= 0.25): needs per-shell per-halo / all-halo ratio
A = np.load(G / "work" / "arrays.npz")
keys = [k for k in A.files if "per_halo" in k and "13.5-13.7" in k]
pr("\narrays with per_halo & 13.5-13.7:", keys[:20])
rf = R["resolved_fraction_per_shell"]["ref"]["13.5-13.7"] if "13.5-13.7" in R["resolved_fraction_per_shell"]["ref"] \
    else None
out["resolved_fraction_ref_13.5-13.7"] = rf
xv = 0.75 * (E[1:] ** 4 - E[:-1] ** 4) / (E[1:] ** 3 - E[:-1] ** 3)
shells_025_05 = [i for i in range(35) if 0.25 <= xv[i] < 0.5]
pr("shells in band 0.25-0.5:", [(i, round(float(E[i]), 4), round(float(E[i + 1]), 4)) for i in shells_025_05])
for f in ("P", "E"):
    kv, kr = f"stack|per_halo|{f}|13.5-13.7|value", f"stack|ref|{f}|13.5-13.7|value"
    if kv in A.files and kr in A.files:
        ratio = A[kv] / A[kr]
        out[f"per_halo_shell_ratio_13.5-13.7_{f}"] = [float(v) for v in ratio]
        pr(f"  {f} per-shell per_halo/ref 13.5-13.7 at shells {shells_025_05}:",
           [round(float(ratio[i]), 4) for i in shells_025_05])
# exploratory bins, bands 0.25-1
expl = {f: [sbr[f][lab][b]["ratio"] for lab in LABS[:5] for b in ("0.25-0.5", "0.5-1")
            if sbr[f][lab][b]["ratio"] is not None] for f in ("P", "E")}
out["per_halo_exploratory_0.25_1_range"] = {f: [min(v), max(v)] for f, v in expl.items()}
pr("exploratory 0.25-1 band range:", out["per_halo_exploratory_0.25_1_range"])
cal_ge025 = {f: [sbr[f][lab][b]["ratio"] for lab in CAL for b in ("0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")
                 if sbr[f][lab][b]["ratio"] is not None] for f in ("P", "E")}
out["per_halo_cal_ge025_range"] = {f: [min(v), max(v)] for f, v in cal_ge025.items()}
pr("calibrated >=0.25 band range:", out["per_halo_cal_ge025_range"])

# ---------------------------------------------------------------- 5. AGN and P / extrapolation in 13.5-13.7 etc.
ID = R["inner_decomposition"]
agn = {}
for lab in ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.3-14.5", "14.9-15.1"):
    d = ID[lab]
    p = np.array(d["P_over_ref2R_extrapolation"], float)
    k = np.array(d["P_agn_kept_over_removed"], float)
    kept = np.array(d["kept_reference"], bool)
    rej = np.flatnonzero(~kept)
    agn[lab] = {"x_vol": [round(float(xv[i]), 4) for i in rej],
                "P_over_extrap": [float(p[i]) for i in rej],
                "P_agn_kept_over_removed": [float(k[i]) for i in rej],
                "P_kept_over_extrap": [float(p[i] * k[i]) for i in rej],
                "deficit_closed_fraction": [None if p[i] >= 1 else float((p[i] * k[i] - p[i]) / (1 - p[i]))
                                            for i in rej],
                "rejected_shells_below_extrap": [[float(E[i]), float(E[i + 1])] for i in rej if p[i] < 1],
                "rejected_shells_above_extrap": [[float(E[i]), float(E[i + 1])] for i in rej if p[i] >= 1]}
out["agn"] = agn
pr("\nAGN / P extrapolation")
for lab, d in agn.items():
    pr(" ", lab)
    for kk in ("P_over_extrap", "P_agn_kept_over_removed", "P_kept_over_extrap", "deficit_closed_fraction"):
        pr("    ", kk, [None if v is None else round(v, 4) for v in d[kk]])
    pr("     below:", d["rejected_shells_below_extrap"])
    pr("     above:", d["rejected_shells_above_extrap"])

# ---------------------------------------------------------------- 6. n_e hot / cold per bin
hot = {}
for lab in CAL[:-1]:
    d = ID[lab]
    kept = np.array(d["kept_reference"], bool)
    rej = np.flatnonzero(~kept)
    hot[lab] = {k: [None if d[k][i] is None else round(float(d[k][i]), 4) for i in rej]
                for k in ("E_over_ref2R_extrapolation", "E_xray_over_ref2R_extrapolation", "E_cold_sf_share",
                          "E_excess_hot_share", "count_median", "xh_median", "count_zero_fraction",
                          "no_nonagn_gas_fraction", "P_over_ref2R_extrapolation", "P_xray_over_ref2R_extrapolation",
                          "P_agn_kept_over_removed")}
out["inner_rejected"] = hot
pr("\nrejected-shell decomposition (shell 0 = 0-0.025 sphere)")
for lab, d in hot.items():
    pr(" ", lab)
    for k, v in d.items():
        pr("    ", k, v)

# volume-integrated hot share of the n_e excess (rejected shells inside R200c with n_e > model)
VOL = 4 * np.pi / 3 * np.diff(E ** 3)
hint = {}
for lab in CAL[:-1]:
    Ea = A[f"stack|ref|E|{lab}|value"]
    Ex = A[f"stack|ref_E_xray|{lab}|value"]
    m = A[f"fit|ref|E|2|{lab}|model"]
    kept = np.asarray(PS["ref"]["masks"][lab]["kept"], bool)
    sel = ~kept & (E[1:] <= 1 + 1e-12) & (Ea > m)
    den = ((Ea - m) * VOL)[sel].sum()
    hint[lab] = {"hot_share_int": float(((Ex - m) * VOL)[sel].sum() / den),
                 "innermost_hot_share": float((Ex[0] - m[0]) / (Ea[0] - m[0])) if Ea[0] > m[0] else None,
                 "E_cold_sf_share_shell0": float(ID[lab]["E_cold_sf_share"][0])}
out["hot_share_integrated"] = hint
pr("\nvolume-integrated hot share of the n_e excess:", {k: {kk: round(vv, 3) if vv is not None else None
                                                          for kk, vv in v.items()} for k, v in hint.items()})

# ---------------------------------------------------------------- 7. impacts: frac95 max, tables, errors
def imp(sid, tag):
    return PS[sid]["summary"][tag]


fr = {}
for tag in ("impact_fit2R_vs_ref_fit", "impact_fit4R_vs_ref_fit"):
    vals = {o: imp("resolution:frac95", tag)[o]["dO_over_O"] for o in OBS}
    o_max = max(vals, key=lambda o: abs(vals[o]))
    fr[tag] = {"max_abs_percent": 100 * abs(vals[o_max]), "observable": o_max}
out["frac95_max"] = fr
pr("\nfrac95 max |dO/O|:", fr)

SETS = ["mc50", "mc100", "mc200", "mc500", "hf1", "hf0", "hf3", "me0", "me0.1", "frac95", "frac84", "diagnostic",
        "per_halo_f50", "per_halo", "none", "mc0", "mc5", "mc10", "mc32", "sf0", "sf2", "sf3", "sf6", "mconly500"]
tab = {}
for s in SETS:
    sid = f"resolution:{s}"
    row = {}
    for tag in ("impact_fit4R_vs_ref_fit", "impact_fit2R_vs_ref_fit", "impact_fit4R_vs_truth", "impact_stack"):
        row[tag] = {o: [round(100 * imp(sid, tag)[o]["dO_over_O"], 3), round(100 * imp(sid, tag)[o]["err"], 3),
                        round(100 * imp(sid, tag)[o]["missing_share"], 2)] for o in OBS}
    tab[s] = row
tab["ref"] = {"impact_fit4R_vs_truth": {o: [round(100 * imp("ref", "impact_fit4R_vs_truth")[o]["dO_over_O"], 3),
                                            round(100 * imp("ref", "impact_fit4R_vs_truth")[o]["err"], 3)]
                                        for o in OBS}}
out["impacts"] = tab
pr("\n4R vs ref fit [%, err, missing%]")
for s in SETS:
    pr(f"  {s}: " + " | ".join(f"{o.replace('_yy_total', '').replace('_ell', '')} {v[0]:+.2f}+-{v[1]:.2f}"
                                + (f" m{v[2]}" if v[2] else "") for o, v in tab[s]["impact_fit4R_vs_ref_fit"].items()))
pr("\n4R vs truth")
for s in ("ref", "frac95", "mc100", "hf3", "mc500", "per_halo", "none"):
    pr(f"  {s}: " + " | ".join(f"{o} {v[0]:+.2f}+-{v[1]:.2f}" for o, v in tab[s]["impact_fit4R_vs_truth"].items()))
pr("\n2R vs ref fit, selected")
for s in ("none", "per_halo", "mc500", "frac95", "diagnostic"):
    pr(f"  {s}: " + " | ".join(f"{o} {v[0]:+.2f}+-{v[1]:.2f} m{v[2]}" for o, v in tab[s]["impact_fit2R_vs_ref_fit"].items()))
pr("\nstack impacts, per_halo:", tab["per_halo"]["impact_stack"])

# largest non-per-halo errors per tag
errs = {}
for tag in ("impact_fit4R_vs_ref_fit", "impact_fit2R_vs_ref_fit", "impact_fit4R_vs_truth"):
    lst = []
    for sid, d in PS.items():
        if "per_halo" in sid or sid == "ref":
            continue
        for o in OBS:
            lst.append((100 * d["summary"][tag][o]["err"], sid, o))
    lst.sort(reverse=True)
    errs[tag] = lst[:6]
out["largest_errors"] = errs
pr("\nlargest non-per-halo errors:", json.dumps(errs, indent=0))
ph_err = [100 * PS["resolution:per_halo"]["summary"]["impact_fit4R_vs_ref_fit"][o]["err"] for o in OBS]
out["per_halo_err_range_4R"] = [min(ph_err), max(ph_err)]
pr("per_halo 4R err range over all 10 observables:", out["per_halo_err_range_4R"])

# missing fit bins: which settings, which fits
mfb = {}
for sid, d in PS.items():
    for fit in ("fit2R", "fit4R"):
        for f in ("P", "E"):
            v = d["impacts"]["missing_fit_bins"][fit][f]["variant"]
            if v:
                mfb[f"{sid}|{fit}|{f}"] = v
out["missing_fit_bins"] = mfb
pr("\nmissing fit bins:", mfb)

# ---------------------------------------------------------------- 8. fits: integrals, rms, params
fits = {}
for s in ("ref", "resolution:frac95", "resolution:mc100", "resolution:hf3", "resolution:mc500", "resolution:none",
          "resolution:per_halo"):
    F = PS[s]["fits"]
    row = {}
    for name, f, outer, key in (("Y1_2R", "P", "2R", "int_lt1_vs_truth"), ("Y1_4R", "P", "4R", "int_lt1_vs_truth"),
                                ("Y4_4R", "P", "4R", "int_lt4_vs_truth"), ("N4_4R", "E", "4R", "int_lt4_vs_truth")):
        vals = {lab: F[f][outer][lab][key]["ratio"] for lab in CAL
                if F[f][outer][lab].get("ok") and F[f][outer][lab].get(key)}
        vv = {lab: round(100 * (v - 1), 2) for lab, v in vals.items() if v is not None}
        lo = min(vv, key=vv.get)
        hi = max(vv, key=vv.get)
        row[name] = {"min": [vv[lo], lo], "max": [vv[hi], hi], "n_bins": len(vv)}
    fits[s] = row
out["fit_integrals"] = fits
pr("\nfit integrals (fit / particle sum - 1, %):")
for s, row in fits.items():
    pr(" ", s, row)
rms = {}
for outer in ("2R", "4R"):
    rms[outer] = {lab: PS["ref"]["fits"]["P"][outer][lab]["rms_percent"] for lab in CAL}
out["rms_ref_P"] = rms
pr("\nref B12 rms %:", rms)
pr("Y1 4R ref 13.5-13.7:", PS["ref"]["fits"]["P"]["4R"]["13.5-13.7"]["int_lt1_vs_truth"],
   PS["ref"]["fits"]["P"]["4R"]["13.5-13.7"]["int_lt4_vs_truth"])
for s, f, outer, lab in (("resolution:per_halo", "P", "4R", "13.5-13.7"), ("resolution:per_halo", "P", "2R", "13.5-13.7"),
                         ("resolution:none", "E", "2R", "13.5-13.7"), ("resolution:mc500", "P", "4R", "13.7-13.9")):
    pr(f"dln params {s} {f} {outer} {lab}:", PS[s]["fits"][f][outer][lab].get("dln_params_vs_ref"),
       PS[s]["fits"][f][outer][lab].get("dln_params_vs_ref_err"))
for s in ("resolution:mc500", "resolution:mc100"):
    pr(f"band ratio vs ref {s} P 4R 13.7-13.9:",
       {b: v for b, v in PS[s]["fits"]["P"]["4R"]["13.7-13.9"]["band_ratio_vs_ref"].items() if b in ("0-0.1", "0.1-0.25")})

# dropped shells for mc100 / mc200 (calibrated): radial range of the shells removed
drop = {}
for s in ("mc50", "mc100", "mc200", "hf3"):
    rng = []
    for lab in CAL:
        a = PS["ref"]["summary"]["first_shell_index"][lab]
        b = PS[f"resolution:{s}"]["summary"]["first_shell_index"][lab]
        if b is not None and a is not None and b > a:
            rng.append((float(E[a]), float(E[b])))
    drop[s] = [min(r[0] for r in rng), max(r[1] for r in rng)] if rng else None
out["dropped_shell_range"] = drop
pr("\ndropped-shell radial range (calibrated):", drop)

# ---------------------------------------------------------------- 9. softening / min_edge / diagnostic masks
for s in ("sf0", "sf2", "sf3", "sf6"):
    diff = [lab for lab in LABS if PS[f"resolution:{s}"]["summary"]["first_edge"][lab] != PS["ref"]["summary"]["first_edge"][lab]
            or PS[f"resolution:{s}"]["summary"]["n_kept_to_4R"][lab] != PS["ref"]["summary"]["n_kept_to_4R"][lab]]
    pr(s, "bins with a different mask:", diff)
pr("diagnostic first edges:", PS["resolution:diagnostic"]["summary"]["first_edge"])
pr("me0 first edges:", {l: PS["resolution:me0"]["summary"]["first_edge"][l] for l in CAL[-2:]})
pr("me0.1 first edges:", {l: PS["resolution:me0.1"]["summary"]["first_edge"][l] for l in CAL})
pr("softening-alone edges:", {l: PS["ref"]["masks"][l]["first_edge_by_criterion"]["softening"] for l in LABS})
pr("h-alone edges (factor 2):", {l: PS["ref"]["masks"][l]["first_edge_by_criterion"]["smoothing_length"] for l in LABS})
pr("n_kept_mconly:", {k: v for k, v in R["report_numbers"].items() if k.startswith("n_kept_mconly")})

# ---------------------------------------------------------------- 10. N dependence
nd = R["n_dependence_of_strict"]
rows_nd = []
for lab in CAL[:-2]:
    d = nd[lab]
    for n in (30, 100, 300):
        if f"subset_{n}" in d:
            rows_nd.append((lab, n, d["full"], d[f"subset_{n}"]["median"], d["full_frac95"],
                            d[f"frac95_subset_{n}"]["median"]))
ns = sum(abs(r[2] - r[3]) > 1e-6 for r in rows_nd)
nf = sum(abs(r[4] - r[5]) > 1e-6 for r in rows_nd)
out["n_dependence"] = {"n_tests": len(rows_nd), "strict_diff": ns, "frac95_diff": nf,
                       "frac95_diff_cases": [r[:2] for r in rows_nd if abs(r[4] - r[5]) > 1e-6]}
pr("\nN dependence:", out["n_dependence"])
for lab in ("14.5-14.7", "13.7-13.9", "13.1-13.3"):
    d = nd[lab]
    pr(" ", lab, d["N"], d["full"], {n: d.get(f"subset_{n}", {}).get("median") for n in (300, 100, 30)},
       d["full_frac95"], {n: d.get(f"frac95_subset_{n}", {}).get("median") for n in (300, 100, 30)})

# ---------------------------------------------------------------- 11. L2p8 redshift: which criterion binds
emp = R["redshift"]["L2p8_empirical"]
cases = []
for z, dz in emp.items():
    for b, d in dz["bins"].items():
        fe = d["frac95_first_edge"]
        bc = d["by_criterion"]
        crit = {k: bc[k]["frac95"] for k in bc}
        binds = [k for k, v in crit.items() if v is not None and fe is not None and abs(v - fe) < 1e-6]
        cases.append({"z": z, "bin": b, "N": d["N"], "first_edge": fe, "alone": crit, "binding": binds})
h_binds = [c for c in cases if "smoothing_length" in c["binding"]]
not_h = [c for c in cases if "smoothing_length" not in c["binding"]]
out["L2p8_binding"] = {"n_cases": len(cases), "n_h_binds": len(h_binds), "exceptions": not_h,
                       "soft_alone_cases": [c for c in cases if c["binding"] == ["softening"]],
                       "soft_ties": [(c["z"], c["bin"], c["binding"]) for c in cases if "softening" in c["binding"]],
                       "N_range": [min(c["N"] for c in cases), max(c["N"] for c in cases)]}
pr("\nL2p8:", json.dumps(out["L2p8_binding"], indent=0)[:3000])
for z in ("0", "0.5", "1", "1.5", "2", "3", "4"):
    for b in ("13.4-13.6", "14.0-14.2"):
        d = emp[z]["bins"].get(b)
        if d:
            pr(f"  z={z} {b}: frac95 edge {d['frac95_first_edge']}, N {d['N']}, median count@0.1 "
               f"{d['median_count_at_x0.1']}, x/h@0.1 {d['median_x_over_h_at_x0.1']}, x_ref {d['x_ref']}")
ana = R["redshift"]["analytic_L1m9_bins"]
pr("analytic z=3 13.5-13.7:", ana["3"]["bins"]["13.5-13.7"], {z: ana[z]["eps_over_R_factor_vs_z0"] for z in ana})

# ---------------------------------------------------------------- 12. grids
gd = R["grid_dependence"]
pr("\ngrid first edges:")
for g in gd:
    pr(f"  {g} ({gd[g].get('radius')}, median dex {gd[g].get('dex_per_shell_median')}):",
       {l: gd[g]["bins"][l]["first_edge"] for l in LABS})
gg = R["grid_gaps_strict"]["x200"]
allg = [x for l in gg for x in gg[l]]
co = [x for x in allg if x["n_fail_smoothing_length"] == 0 and x["n_fail_geometric"] == 0]
out["x200_gaps"] = {"n": len(allg), "n_bins": len(gg), "count_only": len(co),
                    "width_all": [min(x["width_dex"] for x in allg), max(x["width_dex"] for x in allg)],
                    "width_count_only": [min(x["width_dex"] for x in co), max(x["width_dex"] for x in co)]}
pr("x200 gaps:", out["x200_gaps"])

# ---------------------------------------------------------------- 13. per-halo bias by fraction (P range in < 0.05)
phb = R["per_halo_bias_by_fraction"]
pr("\nper_halo_bias_by_fraction P keys:", list(phb["P"].keys())[:12])
pr(json.dumps(phb["P"])[:1500])

# ---------------------------------------------------------------- 14. web variants count
W = json.loads((G / "web_variants.json").read_text())
vids = sorted({v["variant"] if isinstance(v, dict) and "variant" in v else None for v in
               (W["records"] if isinstance(W, dict) and "records" in W else [])} - {None})
pr("\nweb keys:", list(W.keys()) if isinstance(W, dict) else type(W))
out["web"] = {"n_variant_ids": len(vids), "n_fit4": len([v for v in vids if "fit4" in v]),
              "n_setting": len([v for v in vids if "fit4" not in v and v != "ref"])}
pr("web:", out["web"])

json.dump(out, open(G / "section_checks.json", "w"), indent=1, default=str)
print("\nwrote section_checks.json")
