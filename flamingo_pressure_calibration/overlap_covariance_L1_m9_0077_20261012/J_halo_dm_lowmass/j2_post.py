"""Test J2 post-processing (local, numpy only, < 1 GB): errors, family RMS, sensitive-part scores, minimal adequate
models, the low-mass fallback replacement and the ladder comparison with Test I, from the fetched maas results.

Inputs (read-only): results_j2/ (fits.json, fits_parity.json, ladder_summary.json, ladder_parity.json, ladder_<D>.npz,
tables/J2_eval_long.csv, tables/J2b_ladder_*.csv, evaluate.npz, evaluate_items.json, eval_parity.json,
j1base/stacks.npz), inputs/j2_fitdata.npz (importance-derived weights), Test J1's results/sens.npz (per-statistic
importance maps) and post/j_post.json (J1's exact bound-only kSZ), Test I's tables (comparison).

Definitions (x = r/R200c; c = calibrated bin; g = a shell of the definition's own stack: the fine X200 shells in
13.5-15.5, J1's merged >= 400-particle T1 shells in 12.7-13.5; O = an observable):
  error           e_O = O_model / O_ref - 1; S1 painting against the scatter-inclusive truth ('S1'), det painting
                  against the truth mean ('profile-only', isolates the profile shape)
  DM family RMS   eps_DM = sqrt(mean over the DM members of e_O^2); members: mean, variance, Q16, Q50, Q84, Q97.5, Q99.9
                  of the halo DM PDF to z_s = 1 (D1 without Q16: inside the no-crossing atom for every painting)
  DM-field / y x DM / kSZ / tSZ family RMS: the same over ell = 100, 300, 1000, 3000 of dmcltot / ydmtot / ksz1h / yytot
  importance      I_O[c, s] = (dO/d ln F_c,s) / (k_O O) (J1, definition D, C14); family weight w[c, s] = mean of |I_O|
                  over the family's members (and 100 <= ell <= 3000); summed over the fine shells of each group g
  score           S = sqrt( sum_{c,g} w[c,g] d[c,g]^2 / sum_{c,g} w[c,g] ), d = ln(M_g / m_g), M_g = volume average of
                  the painted model over g, m_g = the stack's value (T1); d = -1 beyond R_cut (removed signal; absent
                  here because every painting is cut at the definition's boundary, where the weight ends); S per DM
                  statistic uses w = |I_O| of that statistic alone; dimensionless (0.05 = 5% RMS profile error where
                  the signal comes from); S is not an observable error (signed residuals cancel in observables)
  minimal adequate model: the fewest parameters k with both the S1 and the profile-only DM family RMS <= threshold
Outputs: post_j2/J2_scores.csv, J2_summary.json, J2_lowmass.csv, J2_ladder_compare.csv, J2_fit_parameters.csv.
"""
from __future__ import annotations

import csv
import json

import j2_common as C
import numpy as np

R = C.J2_RESULTS
POST = C.HERE / ("post_j2_smoke" if C.SMOKE else "post_j2")
REGIONS = ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))
STAT_MAPS = {"mean": "dm_mean", "var": "dm_var", "q16": "dm_q16", "q50": "dm_q50", "q84": "dm_q84",
             "q975": "dm_q975", "q999": "dm_q999"}
THRESHOLDS = (0.01, 0.02, 0.05, 0.10)


def fnum(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_long():
    out = {}
    with open(R / "tables" / "J2_eval_long.csv") as fh:
        for r in csv.DictReader(fh):
            out[(r["label"], r["prediction"])] = r
    return out


def groups_for(stacks, D, X):
    """Per J bin: list of (a, e) fine-shell index ranges of the definition's own stack shells."""
    key = "E_a2ii" if C.DEFS[D]["sel"] == "a2ii" else "E_hb"
    out = []
    for b in C.J_BINS:
        g = stacks.get(f"groups_{key}_{b}") if b in C.LOW_BINS else None      # T1 merging only in 12.7-13.5
        out.append([tuple(map(int, x)) for x in g] if g is not None else [(s, s + 1) for s in range(len(X) - 1)])
    return out


def aggregate(F, groups, X):
    """Volume averages over each group: list per bin of arrays."""
    V = C.vol(X)
    return [np.array([F[c, a:e] @ V[a:e] / V[a:e].sum() for a, e in groups[c]]) for c in range(len(groups))]


def gsum(W, groups):
    return [np.array([W[c, a:e].sum() for a, e in groups[c]]) for c in range(len(groups))]


def score(Mg, mg, wg, lo, hi, r_cut, sel=None, xmin=None):
    """xmin [bin]: only shells with x_lo >= xmin[c] (the fitted region; None: every shell)."""
    num = den = 0.0
    for c in range(len(Mg)):
        if sel is not None and not sel[c]:
            continue
        m, M, w = mg[c], Mg[c], wg[c]
        ok = m > 0
        if xmin is not None:
            ok = ok & (lo[c] >= xmin[c] - 1e-9)
        ins = hi[c] <= r_cut + 1e-9
        d = np.zeros_like(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(np.where(M > 0, M, np.nan) / np.where(ok, m, 1.0))
        d[ok & ins & np.isfinite(lr)] = lr[ok & ins & np.isfinite(lr)]
        d[ok & ~(ins & np.isfinite(lr))] = -1.0
        num += float(np.sum(np.where(ok, w, 0) * d**2))
        den += float(np.sum(np.where(ok, w, 0)))
    return float(np.sqrt(num / den)) if den > 0 else None


def region_mean(Mg, mg, wg, lo, hi, r_cut):
    out = {}
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    for a, b in REGIONS:
        num = den = 0.0
        for c in range(len(Mg)):
            sel = (mid[c] >= a) & (mid[c] < b) & (mg[c] > 0) & (hi[c] <= r_cut + 1e-9)
            if not sel.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.log(np.where(Mg[c][sel] > 0, Mg[c][sel], np.nan) / mg[c][sel])
            d = np.where(np.isfinite(d), d, -1.0)
            num += float(np.sum(wg[c][sel] * d))
            den += float(np.sum(wg[c][sel]))
        out[f"{a:g}-{b:g}"] = num / den if den > 0 else None
    return out


def run2_extras(summ, rows, by, fits, models):
    """Run-2 additions (verifier round 2): ladder completeness, the best painted mass-cell law per k, outer options vs
    O0 per inner law, Test I's own fits vs J2's dI refit, and the first fitted shell of each (L) fit in 12.7-13.5."""
    def worst(r):
        return max(r["rms_DM"], r["prof_DM"] if r.get("prof_DM") is not None else np.inf)

    comp, dmk, outer = {}, {}, {}
    for D, spec in C.DEFS.items():
        res = json.loads(str(np.load(R / f"ladder_{D}.npz")["results_json"]))
        mrows = {(r["model"], float(r["R_fit"])): r for r in models.get(D, []) if r["dset"] == "mass"}
        for rf in spec["lad_rfit"]:
            fitted = sorted(r["model"] for r in res if r["dset"] == "mass" and abs(float(r["R_fit"]) - rf) < 1e-9
                            and "floor" in r["fits"])
            painted = sorted(r["model"] for r in rows if r["D"] == D and r["block"] == "J2b" and r.get("dset") == "mass"
                             and abs(float(r["R_fit"]) - rf) < 1e-9 and r.get("rms_DM") is not None)
            comp[f"{D}|Rf{rf:g}"] = {"n_fitted": len(fitted), "n_painted": len(painted),
                                     "complete": fitted == painted, "missing": sorted(set(fitted) - set(painted))}
            lad = [r for r in rows if r["D"] == D and r["block"] == "J2b" and r.get("dset") == "mass"
                   and abs(float(r["R_fit"]) - rf) < 1e-9 and r.get("rms_DM") is not None]
            perk = []
            for k in sorted({int(r["k"]) for r in lad}):
                rk = [r for r in lad if int(r["k"]) == k]
                b = min(rk, key=worst)
                mr = mrows.get((b["model"], rf), {})
                perk.append({"k": k, "label": b["label"], "model": b["model"], "rms_DM": b["rms_DM"],
                             "prof_DM": b["prof_DM"], "max_S1_profile": worst(b), "n_models_with_k": len(rk),
                             "chi2nu_floor": fnum(mr.get("chi2nu_floor")), "dBIC_floor": fnum(mr.get("dBIC_floor"))})
            # running minimum: the lowest max(S1, profile) DM error reachable with at most k parameters
            run = np.inf
            for e in perk:
                run = min(run, e["max_S1_profile"])
                e["best_up_to_k"] = run
            dmk[f"{D}|Rf{rf:g}"] = perk
            if len(spec["outers"]) > 1:
                inners = sorted({m.split("+")[0] for (m, r_) in mrows if abs(r_ - rf) < 1e-9})
                tab = {}
                for inn in inners:
                    ent = {}
                    for o in spec["outers"]:
                        mr = mrows.get((f"{inn}+{o}", rf))
                        if mr is None:
                            continue
                        lab = f"l|{D}|{inn}+{o}|mass|Rf{rf:g}"
                        ent[o] = {"dBIC_floor": fnum(mr["dBIC_floor"]), "chi2nu_floor": fnum(mr["chi2nu_floor"]),
                                  "k": int(float(mr["k"])), "rms_DM": by.get(lab, {}).get("rms_DM"),
                                  "prof_DM": by.get(lab, {}).get("prof_DM")}
                    if "O0" in ent:
                        best_o = min(ent, key=lambda o: ent[o]["dBIC_floor"])
                        ent["best_by_BIC"] = best_o
                        ent["outer_beats_O0_by_BIC"] = best_o != "O0"
                    tab[inn] = ent
                outer[f"{D}|Rf{rf:g}"] = {"per_inner": tab,
                                          "inners_where_an_outer_option_beats_O0": sorted(
                                              i for i, v in tab.items() if v.get("outer_beats_O0_by_BIC")),
                                          "best_overall_by_BIC": min(((i, o) for i, v in tab.items() for o in v
                                                                      if o in spec["outers"]),
                                                                     key=lambda t: tab[t[0]][t[1]]["dBIC_floor"])}
    summ["ladder"]["completeness"] = comp
    summ["ladder"]["dm_by_k"] = dmk
    summ["ladder"]["outer_vs_O0"] = outer
    # ---- Test I's published E|d|4 fits vs J2's dI refit (13.5-15.5) and the exact effect on the paintings
    ifp = C.I_RESULTS / "fits35.json"
    if ifp.exists() and not C.SMOKE:
        TI = C.read_json(ifp)["fits"]
        per = {}
        for b in C.HIGH_BINS:
            a = np.asarray(fits["fits"][f"D0|dI|4|{b}"]["parameters"], float)
            t = np.asarray(TI[f"E|d|4|{b - 5}"]["parameters"], float)
            per[C.bin_label(b)] = {"J2_dI": a, "TestI_E_d_4": t, "max_rel_param": float(np.max(np.abs(a / t - 1)))}
        fam = ("rms_DM", "prof_DM", "rms_kSZ", "prof_kSZ", "rms_kSZ2h", "rms_tSZ", "rms_DMcl", "rms_ydm")
        summ["dI_vs_TestI"] = {
            "per_bin": per, "max_rel_param": max(v["max_rel_param"] for v in per.values()),
            "families": {lab: {k: by.get(lab, {}).get(k) for k in fam}
                         for lab in ("lm|C14|B12a+B16dIT", "lm|C14|B12a+B16dI", "lm|C10|B12a+B16dIT",
                                     "lm|C10|B12a+B16dI", "a|D0|B16|dIT|Rf4", "a|D0|B16|dI|Rf4")},
            "note": "dIT = Test I's own fits (I_fitting_a2ii/results/fits35.json E|d|4, P|a|4; weights from Test H's "
                    "C10 maps) in 13.5-15.5 + J2's D0|dI|4 / P|a|4 fits in 12.7-13.5; dI = Test I's weight formula "
                    "applied to J1's C14 maps, refitted in all 14 bins"}
        summ["dI_vs_TestI"]["change_dIT_minus_dI_pp"] = {
            k: 100 * (by["lm|C14|B12a+B16dIT"][k] - by["lm|C14|B12a+B16dI"][k])
            for k in ("rms_DM", "rms_kSZ", "rms_kSZ2h", "rms_tSZ") if "lm|C14|B12a+B16dIT" in by}
    # ---- first fitted 35-shell edge of each fit in 12.7-13.5 vs the first shell of the bin's 35-shell fit mask
    inp = C.load_npz(C.J2_INPUT, keys=["f35_edges", "mask35_a2ii", "mask35_hb"])
    F35e = np.asarray(inp["f35_edges"], float)
    fe = {}
    for key, r in fits["fits"].items():
        if r is None or r.get("mask") is None or key.startswith("P|"):
            continue
        D, obj, rf, b = key.split("|")
        b = int(b)
        if b not in C.LOW_BINS:
            continue
        mk = np.asarray(r["mask"], bool)
        m35 = np.asarray(inp["mask35_" + C.DEFS[D]["sel"]], bool)[b]
        if mk.any() and m35.any():
            fe[key] = {"first_fitted_edge": float(F35e[np.flatnonzero(mk)[0]]),
                       "mask35_first_edge": float(F35e[np.flatnonzero(m35)[0]])}
    summ["first_fitted_edge_low_bins"] = fe


def main():
    POST.mkdir(exist_ok=True)
    ev = C.load_npz(R / "evaluate.npz")
    meta = C.read_json(R / "evaluate_items.json")
    items = meta["items"]
    long = load_long()
    inp = C.load_npz(C.J2_INPUT)
    stacks = C.load_npz(R / "j1base" / "stacks.npz")
    sens = C.load_npz(C.J1_FULL_RESULTS / "sens.npz", keys=None)
    X = np.asarray(ev["x_edges"], float)
    jb = C.J_BINS
    low_sel = np.array([b < 5 for b in jb])
    # ---- per-definition aggregation helpers
    G, LO, HI, MSTACK, W = {}, {}, {}, {}, {}
    for D, spec in C.DEFS.items():
        G[D] = groups_for(stacks, D, X)
        LO[D] = [np.array([X[a] for a, _ in g]) for g in G[D]]
        HI[D] = [np.array([X[e] for _, e in g]) for g in G[D]]
        key = "E_a2ii_T1" if spec["sel"] == "a2ii" else "E_hb_T1"
        MSTACK[D] = aggregate(np.asarray(stacks[key], float)[jb], G[D], X)
        W[D] = {"DM": gsum(np.asarray(inp[f"wfine_DM_{D}"]), G[D])}
        if f"wfine_kSZ_{D}" in inp:
            W[D]["kSZ"] = gsum(np.asarray(inp[f"wfine_kSZ_{D}"]), G[D])
        for st, nm in STAT_MAPS.items():
            k = f"imp__{D}__{nm}"
            if k in sens and not (D == "D1" and st == "q16"):
                W[D][st] = gsum(np.abs(np.asarray(sens[k], float))[jb][:, :len(X) - 1], G[D])
    # innermost fitted edge per bin (35-shell masks of j2_inputs) and the DM weight share of the unfitted cores
    F35e = np.asarray(inp["f35_edges"], float)
    XMIN, CORE = {}, {}
    for D, spec in C.DEFS.items():
        mk = np.asarray(inp["mask35_" + spec["sel"]], bool)
        XMIN[D] = [float(F35e[np.flatnonzero(mk[b])[0]]) for b in jb]
        w = W[D]["DM"]
        tot = sum(float(x.sum()) for x in w)
        core = [float(w[c][LO[D][c] < XMIN[D][c] - 1e-9].sum()) for c in range(len(jb))]
        CORE[D] = {"xmin_per_bin": dict(zip([C.bin_label(b) for b in jb], XMIN[D])),
                   "DM_weight_share_inside_xmin": sum(core) / tot,
                   "DM_weight_share_inside_xmin_low_bins": sum(core[c] for c in range(len(jb)) if low_sel[c]) / tot,
                   "DM_weight_share_low_bins": sum(float(w[c].sum()) for c in range(len(jb)) if low_sel[c]) / tot}
    # ---- scores and errors per painting
    rows = []
    for it in items:
        lab, D = it["label"], it["D"]
        M4 = np.asarray(ev[f"M4__{lab}"], float)
        Mg = aggregate(M4, G[D], X)
        rc = it["R_cut"]
        row = {k: it.get(k) for k in ("label", "block", "D", "model", "objective", "R_fit", "R_cut", "k", "rule",
                                      "dset", "config", "selected")}
        row["roles"] = ";".join(it.get("roles", []) or [])
        for fam, w in W[D].items():
            row[f"S_{fam}"] = score(Mg, MSTACK[D], w, LO[D], HI[D], rc)
        row["S_DM_low"] = score(Mg, MSTACK[D], W[D]["DM"], LO[D], HI[D], rc, sel=low_sel)
        row["S_DM_high"] = score(Mg, MSTACK[D], W[D]["DM"], LO[D], HI[D], rc, sel=~low_sel)
        # the same restricted to the fitted region (x >= the innermost fitted 35-shell edge of each bin): the cores
        # inside it (unresolved in every bin; in 12.7-13.5 they hold the central spike inside the softening limit)
        # are painted by extrapolation and do not constrain the fits
        row["S_DM_fit"] = score(Mg, MSTACK[D], W[D]["DM"], LO[D], HI[D], rc, xmin=XMIN[D])
        row["S_DM_low_fit"] = score(Mg, MSTACK[D], W[D]["DM"], LO[D], HI[D], rc, sel=low_sel, xmin=XMIN[D])
        row["S_DM_high_fit"] = score(Mg, MSTACK[D], W[D]["DM"], LO[D], HI[D], rc, sel=~low_sel, xmin=XMIN[D])
        if "kSZ" in W[D]:
            row["S_kSZ_fit"] = score(Mg, MSTACK[D], W[D]["kSZ"], LO[D], HI[D], rc, xmin=XMIN[D])
        for st in ("mean", "var", "q16", "q50", "q84", "q975", "q999"):
            if st in W[D]:
                row[f"S_{st}_fit"] = score(Mg, MSTACK[D], W[D][st], LO[D], HI[D], rc, xmin=XMIN[D])
        for reg, v in region_mean(Mg, MSTACK[D], W[D]["DM"], LO[D], HI[D], rc).items():
            row[f"dmean_DM_{reg}"] = v
        for pred, tag in (("S1", ""), ("det", "_det")):
            r = long.get((lab, pred))
            if r is None:
                continue
            fam = C.dm_family(D)
            e_tr = {k: fnum(r.get(f"err_{k}")) for k in fam}
            e_m = {k: fnum(r.get(f"errm_{k}")) for k in fam}
            if pred == "S1":
                row["rms_DM"] = C.family_rms(e_tr, fam)
                for k in fam:
                    row[f"e_{k}"] = e_tr[k]
                for name, key in (("DMcl", "dmcltot"), ("ydm", "ydmtot"), ("kSZ", "ksz1h"), ("kSZ2h", "ksz2h"),
                                  ("tSZ", "yytot")):
                    ks = [f"{key}_{e}" for e in C.FAM_ELL]
                    ee = {k: fnum(r.get(f"err_{k}")) for k in ks}
                    row[f"rms_{name}"] = C.family_rms(ee, ks)
                    for k in ks:
                        row[f"e_{k}"] = ee[k]
            else:
                row["prof_DM"] = C.family_rms(e_m, fam)
                row["rms_DM_det_vs_truth"] = C.family_rms(e_tr, fam)
                for k in fam:
                    row[f"p_{k}"] = e_m[k]
                for name, key in (("DMcl", "dmcltot"), ("ydm", "ydmtot"), ("kSZ", "ksz1h"), ("tSZ", "yytot")):
                    ks = [f"{key}_{e}" for e in C.FAM_ELL]
                    row[f"prof_{name}"] = C.family_rms({k: fnum(r.get(f"errm_{k}")) for k in ks}, ks)
        rows.append(row)
    C.write_csv(POST / "J2_scores.csv", rows)
    by = {r["label"]: r for r in rows}
    C.write_json(POST / "J2_scores.json", by)                 # the same rows keyed by label (report registry)
    # ---- summary per definition
    summ = {"definitions": __doc__, "per_definition": {}, "parity": {}, "lowmass": {}, "ladder": {}, "rc": {}}
    for name in ("fits_parity.json", "ladder_parity.json", "eval_parity.json"):
        p = R / name
        summ["parity"][name] = C.read_json(p) if p.exists() else None
    fits = C.read_json(R / "fits.json")
    lsum = C.read_json(R / "ladder_summary.json")
    for D, spec in C.DEFS.items():
        own = [r for r in rows if r["D"] == D and r["block"] in ("J2a", "J2b") and r.get("rms_DM") is not None]
        # candidates of the recommendation rule, the minimal adequate model and the per-statistic choices: B16 per bin
        # (every objective, R_fit, RC, dIT) and every mass-cell ladder law (complete since run 2).  The concentration-cell
        # laws are painted at the bin-mean concentration, which is not how they would be used, and HalfDome has no
        # concentrations: they are reported separately and are not candidates.
        conc = [r for r in own if r["block"] == "J2b" and r.get("dset") == "massc"]
        own = [r for r in own if not (r["block"] == "J2b" and r.get("dset") == "massc")]
        ref = by.get(f"ref|{D}|stack", {})
        rec = {"boundary_R200c": spec["R"], "label": spec["label"],
               "S1_floor_DM_rms": ref.get("rms_DM"), "S1_floor_per_stat": {k: ref.get(f"e_{k}") for k in
                                                                         C.dm_family(D)},
               "S1_floor_DMcl_rms": ref.get("rms_DMcl"), "S1_floor_ydm_rms": ref.get("rms_ydm")}
        b16 = [r for r in own if r["model"] == "B16"]
        lad = [r for r in own if r["block"] == "J2b"]

        def brief(r):
            if r is None:
                return None
            keys = ("label", "model", "objective", "R_fit", "R_cut", "k", "rule", "dset", "rms_DM", "prof_DM",
                    "rms_DMcl", "rms_ydm", "rms_kSZ", "S_DM", "S_DM_low", "S_DM_high", "S_mean", "S_var", "S_q50",
                    "S_q84", "S_q999", "roles", "S_DM_fit", "S_DM_low_fit", "S_DM_high_fit", "S_mean_fit", "S_var_fit",
                    "S_q16_fit", "S_q50_fit", "S_q84_fit", "S_q975_fit", "S_q999_fit", "rms_kSZ2h", "k", "S_kSZ_fit",
                    "rms_DM_det_vs_truth") + tuple(f"e_{k}" for k in C.dm_family(D)) + \
                tuple(f"p_{k}" for k in C.dm_family(D))
            return {k: r.get(k) for k in keys if k in r}

        rec["best_B16_any"] = brief(min(b16, key=lambda r: r["rms_DM"])) if b16 else None
        rec["best_B16_not_d"] = brief(min([r for r in b16 if r["objective"] not in ("d", "dI", "dIT")],
                                          key=lambda r: r["rms_DM"])) if b16 else None
        rec["best_B16_profile_only"] = brief(min(b16, key=lambda r: r["prof_DM"])) if b16 else None
        rec["B16_at_boundary"] = {f"{r['rule']}|{r['objective']}": brief(r) for r in b16
                                  if abs(float(r["R_fit"]) - spec["R"]) < 1e-9}
        rec["B16_all"] = {r["label"]: brief(r) for r in b16}
        rec["ladder_all"] = {r["label"]: brief(r) for r in lad}
        rec["concentration_laws"] = {r["label"]: brief(r) for r in conc}
        rec["n_candidates"] = {"B16_per_bin": len(b16), "ladder_mass_cells": len(lad), "total": len(own)}
        rec["ladder_selected"] = {r["label"]: brief(r) for r in lad if r.get("selected") in (True, "True")}
        rec["best_ladder"] = brief(min(lad, key=lambda r: r["rms_DM"])) if lad else None
        rec["best_ladder_k_le_14"] = brief(min([r for r in lad if int(r["k"]) <= 14], key=lambda r: r["rms_DM"])) \
            if [r for r in lad if int(r["k"]) <= 14] else None
        rec["lowest_S_DM"] = brief(min(own, key=lambda r: r["S_DM"])) if own else None
        rec["per_statistic"] = {}
        for st in ("mean", "var", "q16", "q50", "q84", "q975", "q999"):
            if st == "q16" and D == "D1":
                continue
            k = "dm_mean" if st == "mean" else ("dm_var" if st == "var" else st)
            cand = [r for r in own if r.get(f"e_{k}") is not None]
            if not cand:
                continue
            bst = min(cand, key=lambda r: abs(r[f"e_{k}"]))
            low = [r for r in own if r.get(f"S_{st}_fit") is not None]
            bS = min(low, key=lambda r: r[f"S_{st}_fit"]) if low else None
            rec["per_statistic"][st] = {"smallest_error": {"label": bst["label"], "error": bst[f"e_{k}"],
                                                           "k": bst["k"]},
                                        "lowest_score_fitted_region": None if bS is None else {
                                            "label": bS["label"], "S": bS[f"S_{st}_fit"], "k": bS["k"],
                                            "error": bS.get(f"e_{k}")}}
        rec["minimal_adequate"] = {}
        rec["minimal_adequate_ties"] = {}
        for thr in THRESHOLDS:
            ok = [r for r in own if r["rms_DM"] <= thr and r.get("prof_DM") is not None and r["prof_DM"] <= thr]
            rec["minimal_adequate"][f"{thr:g}"] = brief(min(ok, key=lambda r: (int(r["k"]), r["rms_DM"]))) if ok \
                else None
            if ok:
                kmin = min(int(r["k"]) for r in ok)
                rec["minimal_adequate_ties"][f"{thr:g}"] = sorted(r["label"] for r in ok if int(r["k"]) == kmin)
        # RC vs S400 (same objective, R_fit = boundary)
        rc = {}
        for obj in ("a", "d"):
            a = by.get(f"a|{D}|B16|{obj}|Rf{spec['R']:g}")
            b = by.get(f"rc|{D}|B16|{obj}|Rf{spec['R']:g}")
            if a and b:
                rc[obj] = {"S400": brief(a), "RC": brief(b)}
        rec["rc_vs_s400"] = rc
        # recommendation rule (data-driven): minimise max(S1 DM family RMS, profile-only DM family RMS), so that a
        # small S1 error obtained by cancellation against the scatter model is not preferred
        def worst(r):
            return max(r["rms_DM"], r.get("prof_DM") if r.get("prof_DM") is not None else np.inf)

        notd = [r for r in b16 if r["objective"] not in ("d", "dI", "dIT")]
        comp = [r for r in lad if int(r["k"]) <= 14]
        rec_r = {"rule": "minimise max(S1 DM family RMS vs truth, profile-only DM family RMS vs truth mean) over every "
                         "fitted painting (B16 per bin and ladder laws); compact: ladder laws with k <= 14",
                 "primary": brief(min(b16 + lad, key=worst)) if b16 + lad else None,
                 "best_B16": brief(min(b16, key=worst)) if b16 else None,
                 "primary_not_d": brief(min(notd + lad, key=worst)) if notd + lad else None,
                 "compact": brief(min(comp, key=worst)) if comp else None}
        for kk in ("primary", "best_B16", "primary_not_d", "compact"):
            if rec_r[kk] is not None:
                rec_r[kk]["max_S1_profile"] = worst(by[rec_r[kk]["label"]])
        summ.setdefault("recommendations", {})[D] = rec_r
        rec["core"] = CORE[D]
        rec["ladder_chosen"] = lsum["chosen"].get(D)
        rec["ladder_sigma_floor"] = lsum["sigma_floor"].get(D)
        summ["per_definition"][D] = rec
    # ---- RC vs S400 (same objective, R_fit = boundary): change of the DM family errors
    ch = {}
    for D, spec in C.DEFS.items():
        for obj in ("a", "d"):
            a = by.get(f"a|{D}|B16|{obj}|Rf{spec['R']:g}")
            b = by.get(f"rc|{D}|B16|{obj}|Rf{spec['R']:g}")
            if a and b:
                ch[f"{D}|{obj}"] = {"S400_rms_DM": a["rms_DM"], "RC_rms_DM": b["rms_DM"],
                                    "change_rms_DM": b["rms_DM"] - a["rms_DM"],
                                    "change_prof_DM": b["prof_DM"] - a["prof_DM"]}
    summ["rc_change"] = ch
    summ["rc_max_abs_change_rms_DM"] = max(abs(v["change_rms_DM"]) for v in ch.values())
    summ["rc_max_abs_change_prof_DM"] = max(abs(v["change_prof_DM"]) for v in ch.values())
    sub = [v for k, v in ch.items() if not k.startswith("D3")]
    summ["rc_max_abs_change_rms_DM_D0_D2"] = max(abs(v["change_rms_DM"]) for v in sub)
    sub2 = [v for k, v in ch.items() if not k.startswith("D3") and k != "D0|a"]
    summ["rc_max_abs_change_rms_DM_D0_D2_except_D0a"] = max(abs(v["change_rms_DM"]) for v in sub2)
    # ---- RC fits: points per bin and unfittable bins
    for key, r in fits["rc"].items():
        summ["rc"][key] = {k: r.get(k) for k in ("n_points", "x_h", "parameters", "chi2_per_dof", "note", "bin")}
    # ---- low-mass calibration with Test I's recipe: C14 (fits everywhere) vs C10 (published fallback below 13.5)
    lm_rows = []
    T0 = meta["truths"].get("D0", {}).get("truth", {})
    for obj in ("dIT", "dI", "a"):
        for pred in ("S1", "det"):
            r14 = long.get((f"lm|C14|B12a+B16{obj}", pred))
            r10 = long.get((f"lm|C10|B12a+B16{obj}", pred))
            if r14 is None:
                continue
            for k in sorted(T0):
                if k.endswith("_8000"):
                    continue
                o14 = fnum(r14.get(f"obs_{k}"))
                o10 = fnum(r10.get(f"obs_{k}")) if r10 else None
                lm_rows.append({"objective": obj, "prediction": pred, "observable": k, "truth_C14_D0": T0[k],
                                "C14_fit": o14, "C10_fit": o10,
                                "change_C14_over_C10": (o14 / o10 - 1) if (o14 is not None and o10) else None,
                                "err_C14_fit": fnum(r14.get(f"err_{k}")),
                                "err_C10_fit": fnum(r10.get(f"err_{k}")) if r10 else None})
    C.write_csv(POST / "J2_lowmass.csv", lm_rows)
    C.write_json(POST / "J2_lowmass.json", {f"{r['objective']}|{r['prediction']}|{r['observable']}": r for r in lm_rows})
    lm = {}
    for r in lm_rows:
        lm.setdefault(f"{r['objective']}|{r['prediction']}", {})[r["observable"]] = {
            k: r[k] for k in ("change_C14_over_C10", "err_C14_fit", "err_C10_fit")}
    summ["lowmass"] = lm
    # families of the lm items (tSZ, kSZ, DM)
    for lab in [r["label"] for r in rows if r["block"] == "LM"]:
        summ["lowmass"][f"families|{lab}"] = {k: by[lab].get(k) for k in ("rms_DM", "prof_DM", "rms_tSZ", "rms_kSZ",
                                                                          "rms_kSZ2h", "rms_DMcl", "rms_ydm",
                                                                          "prof_tSZ", "prof_kSZ")}
    # low-mass fits: parameters (D0, Test I recipe) and their stacks' integrals
    prow = []
    for key, r in fits["fits"].items():
        if r is None:
            continue
        D, obj, rf, b = key.split("|")
        prow.append({"key": key, "D": D, "objective": obj, "R_fit": rf, "engine_bin": b, "bin": r.get("bin"),
                     "rule": r.get("rule"), "p1": r["parameters"][0], "p2": r["parameters"][1],
                     "p3": r["parameters"][2], "chi2_per_dof": r.get("chi2_per_dof"), "n_shells": r.get("n_shells"),
                     "rms_percent": r.get("rms_percent"), "accepted": r.get("accepted"), "hartlap": r.get("hartlap")})
    C.write_csv(POST / "J2_fit_parameters.csv", prow)
    # ---- ladder comparison with Test I (D0) and per-definition parsimony ladders
    comp = []
    for D in list(C.DEFS) + (["D0parityI"] if not C.SMOKE else []):
        p = R / "tables" / f"J2b_ladder_ladder_{D}.csv"
        if not p.exists():
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                comp.append({"source": f"J2 {D}", **{k: r.get(k) for k in ("dset", "R_fit", "k", "model", "chi2_floor",
                                                                         "chi2nu_floor", "dBIC_floor", "pass_bic",
                                                                         "pass_cv", "selected")}})
    ip = C.I_RESULTS / "tables" / "I2_ladder_ladder.csv"
    if ip.exists():
        with open(ip) as fh:
            for r in csv.DictReader(fh):
                if r["field"] == "E":
                    comp.append({"source": "Test I A2-ii (10 bins)", **{k: r.get(k) for k in (
                        "dset", "R_fit", "k", "model", "chi2_floor", "chi2nu_floor", "dBIC_floor", "pass_bic", "pass_cv",
                        "selected")}})
    C.write_csv(POST / "J2_ladder_compare.csv", comp)
    models = {}
    for D in C.DEFS:
        p = R / "tables" / f"J2b_ladder_models_{D}.csv"
        if p.exists():
            with open(p) as fh:
                models[D] = list(csv.DictReader(fh))
    summ["ladder"]["n_models"] = {D: len(v) for D, v in models.items()}
    keymods = {}
    for D, ms in models.items():
        for r in ms:
            if r["dset"] == "mass" and r["model"] in ("PL3g+O0", "PL3+O0", "BPLall+O0", "PB+O0", "PL1+O0", "PL2+O0",
                                                       "PL3gM+O0", "BPLall+O2", "PB+O2", "PL1+O2", "BPLall_L22+O0",
                                                       "BPLs+O0", "BPLa+O0", "BPLa_L22+O0", "BPLs_L22+O0", "PL1s+O0"):
                keymods.setdefault(D, {})[f"{r['model']}|Rf{float(r['R_fit']):g}"] = {
                    k: fnum(r.get(k)) for k in ("k", "chi2nu_stat", "chi2nu_floor", "chi2nu_L22", "dBIC_floor",
                                                "cv_minus_best", "cv_scatter_best")} | {
                    "pass_bic": r.get("pass_bic"), "pass_cv": r.get("pass_cv"), "selected": r.get("selected"),
                    "params": r.get("params")}
    summ["ladder"]["key_models"] = keymods
    mc = {}
    for D, ms in models.items():
        for r in ms:
            if r.get("params") and "log10Mcut" in r["params"]:
                mc.setdefault(D, {})[f"{r['model']}|{r['dset']}|Rf{float(r['R_fit']):g}"] = \
                    json.loads(r["params"])["log10Mcut"]
    summ["ladder"]["log10Mcut_free"] = mc
    d3 = [r for r in rows if r["D"] == "D3" and r["block"] in ("J2a", "J2b") and r.get("rms_kSZ") is not None
          and r.get("dset") != "massc"]
    if d3:
        b = min(d3, key=lambda r: r["rms_kSZ"])
        summ["kSZ_D3_best"] = {"label": b["label"], "rms_kSZ": b["rms_kSZ"], "rms_DM": b["rms_DM"]}
    # the per-halo strict rule's first trusted F shell in the low-mass bins (Test F/G masks; replaced by S400 here)
    g = C.load_npz(C.G_PROFILES35, keys=["f_edges", "strict_mask"])
    summ["strict_rule_first_edge_low_bins"] = {C.bin_label(b): float(g["f_edges"][np.flatnonzero(g["strict_mask"][b])[0]])
                                               for b in C.LOW_BINS if np.any(g["strict_mask"][b])}
    run2_extras(summ, rows, by, fits, models)
    C.write_json(POST / "J2_summary.json", summ)
    print(json.dumps({D: {"floor": v["S1_floor_DM_rms"], "best_B16": (v["best_B16_any"] or {}).get("label"),
                          "best_B16_rms": (v["best_B16_any"] or {}).get("rms_DM"),
                          "best_ladder": (v["best_ladder"] or {}).get("label"),
                          "best_ladder_rms": (v["best_ladder"] or {}).get("rms_DM"),
                          "chosen": v["ladder_chosen"]} for D, v in summ["per_definition"].items()}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
