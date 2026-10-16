"""L stage 'global' (local): global-radius candidates, unit dependence, and how much of each 4R-painted observable comes
from inside each radius (exact for the additive terms, from Test J's fetched truncation curves; no new painting).
User clarification (2026-10-16): the radius is the FIT range; HalfDome still paints the fitted law to 4 R200c.  T below
is therefore the share of the observable fixed by data inside the radius (= what a hard cut would keep, which is not
the plan); 1 - T is the part that depends on the extrapolated region (for all A2-ii gas there; see l_extrap.py).

1. Signal weights.  w_b(O) = Test J's importance share of observable O carried by engine bin b (C14, D0 = A2-ii to 4R;
   J results/sens_summary.json: shares['D0__O']), renormalised over the 14 calibrated bins.  Families:
     P: tSZ C_ell^yy (total) at ell = 100 / 1000 / 3000, mean y, the P side of y x DM at ell = 1000;
     E: kSZ 1-halo at ell = 1000, kSZ total at ell = 100 / 1000 / 3000, mean / variance / median DM, DM-field C_ell at
        ell = 1000, the E side of y x DM at ell = 1000.
   Family weight w_b(F) = mean over the family's observables of w_b(O).
2. Signal-weighted mean radius of a rule: Rbar = sum_b w_b R_t,b (per field; 'common' = mean of the P and E values),
   in R200c, R200m and R500c units; 16-84% from the 200 per-bin bootstrap replicates (independent between bins).
   Mass dependence: CV = sqrt(sum_b w_b (R_t,b - Rbar)^2) / Rbar per unit, and max/min over 12.9-15.3.
3. Exact additive terms.  Test J's per-bin 'own' truncation curves (J results/sens.npz: trunc_own__D0__<O> [bin, edge,
   ell]) give each bin's 1-halo term (mean y, mean DM, C_ell^yy 1h, kSZ 1h, DM-field 1h, DM variance) with that bin's
   haloes cut at each fine edge; these terms add over bins:  O_1h(cuts) = O_fb + sum_b own_b(R_b).
   own_b(R) between two fine edges is interpolated linearly in R^3 (exact at the edges; ESTIMATE inside a shell).
   O_fb (the fallback haloes, i.e. the 9 lightcone haloes above 10^15.388 Msun that keep the published profiles) is
   taken from J's trunc_all__D0__<O>[0] for mean y, mean DM, DM variance, kSZ 1h and DM-field 1h.  C_ell^yy 1h has NO
   fallback term (J stores no trunc_all__D0__yy1h), so for yy1h O_fb = 0 and T is a share of the calibrated haloes'
   yy1h only; the size of the omitted term is bracketed in l_extrap.py (extrap.json:
   observable_level.yy1h_fallback_ESTIMATE; note added 2026-10-16, round 2 of review; no number here changed).
   Reported: T = O(cuts) / O(4R) for the per-bin cuts R_t,b of a rule and for its global radius Rbar applied to every
   bin, and own_D0(cuts) / own_D3(4R) (D3 = host-bound electrons to 4R, J's truth): the all-gas content inside the
   radius relative to the whole bound component.
4. Truncated-target weights: w'_b(O) = own_b(R_t,b) / sum_b' own_b'(R_t,b') for the 1-halo / linear observables.
5. D3 convergence radii from J's truth curves (own_radii and radii of D3: radius holding 90 / 95% of the bound
   component's contribution) next to the stack-level C90 / C95.
6. Observable-optimal single cut (exact additive terms only): the x minimising the RMS over a family's 1-halo / linear
   terms of T(x) / T(per-bin cuts) - 1, for B32, Req and B50 (observable_optimal; added 2026-10-16 after review).
7. Sensitivity of the signal-weighted radius: leave one bin out, and the range over single-observable weights
   (global_sensitivity; added 2026-10-16 after review).
Output: results/global.json, tables/L1_global.csv, tables/L1_weights.csv.
"""
from __future__ import annotations

import csv

import l_common as LC
import l_radii as LR
import numpy as np

SEL_ELL = (100, 300, 1000, 3000, 8000)
FAMILY = {"P": ["yytot@100", "yytot@1000", "yytot@3000", "ymean@scalar", "ydmtot_P@1000"],
          "E": ["ksz1h@1000", "ksztot@100", "ksztot@1000", "ksztot@3000", "dm_mean@scalar", "dm_var@scalar",
                "dm_q50@scalar", "dmcltot@1000", "ydmtot_E@1000"]}
OWN_OBS = {"P": [("ymean", None), ("yy1h", 100), ("yy1h", 1000), ("yy1h", 3000)],
           "E": [("dm_mean", None), ("dm_var", None), ("ksz1h", 100), ("ksz1h", 1000), ("ksz1h", 3000),
                 ("dmcl1h", 1000)]}
MAIN_RULES = ["B32", "B32u", "B32lo", "B32hi", "B50", "B32cb", "B50cb", "C68", "C90", "C95", "Req", "Req_all",
              "SB", "SB_G", "2H2", "2H1", "MDK15", "CLg"]


def load_j():
    sens = np.load(LC.J_DIR / "results" / "sens.npz")
    summ = LC.jload(LC.J_DIR / "results" / "sens_summary.json")
    return sens, summ


def weights(summ):
    labs = [LC.LABELS[b] for b in LC.J_BINS]
    W = {}
    for f, obs in FAMILY.items():
        fam = np.zeros(len(labs))
        for o in obs:
            sh = np.array([summ["shares"][f"D0__{o}"][lab] for lab in labs], float)
            sh = np.clip(sh, 0, None)
            W[f"{f}:{o}"] = sh / sh.sum()
            fam += W[f"{f}:{o}"]
        W[f"{f}:family"] = fam / len(obs)
    return labs, W


def own_curve(sens, defn, name, ell):
    a = np.asarray(sens[f"trunc_own__{defn}__{name}"], float)
    if a.ndim == 3:
        a = a[:, :, SEL_ELL.index(ell)]
    return a                                                   # [14 bins, 68 edges]


def own_at_vec(curve_b, R):
    """own_at for an array of radii (same formula, vectorised)."""
    R = np.clip(np.asarray(R, float), 0.0, 4.0)
    k = np.minimum(np.searchsorted(LC.X, R, side="right") - 1, LC.N_X - 1)
    t = (R**3 - LC.X[k]**3) / (LC.X[k + 1]**3 - LC.X[k]**3)
    return curve_b[k] + t * (curve_b[k + 1] - curve_b[k])


def own_at(curve_b, R):
    """own_b(R): exact at fine edges, linear in R^3 inside a shell."""
    R = float(min(max(R, 0.0), 4.0))
    k = int(np.searchsorted(LC.X, R, side="right") - 1)
    k = min(k, LC.N_X - 1)
    t = (R**3 - LC.X[k]**3) / (LC.X[k + 1]**3 - LC.X[k]**3)
    return float(curve_b[k] + t * (curve_b[k + 1] - curve_b[k]))


OPT_RULES = ("B32", "Req", "B50")
OPT_GRID = np.round(np.arange(1.0, 2.5 + 1e-9, 1e-4), 4)       # candidate single cuts x [R200c], step 1e-4


def observable_optimal(out, sens, rad):
    """The single cut x (R200c, same for every bin) that best reproduces a rule's per-bin-truncated target on the exact
    additive terms:  minimise RMS_F(x) = sqrt( mean_{O in F} [ T_O(x) / T_O(per-bin) - 1 ]^2 )  over x in OPT_GRID,
    T_O(x) = (O_fb + sum_b own_b(x)) / (O_fb + sum_b own_b(4)); F = OWN_OBS[P] (4 terms), OWN_OBS[E] (6) or both
    ('common', 10).  Only the 1-halo / linear terms of Test J's own curves enter (two-halo, total spectra and
    quantiles need Test L2's exact painting on maas)."""
    res = {"definition": observable_optimal.__doc__, "grid_step": 1e-4, "grid": [float(OPT_GRID[0]), float(OPT_GRID[-1])]}
    for rule in OPT_RULES:
        curves, targets, fams, parts = {}, {}, {}, {}
        for f, obs in OWN_OBS.items():
            for name, ell in obs:
                tag = f"{name}@{ell}" if ell else name
                at = out["additive_terms"].get(tag, {})
                if rule not in at:
                    continue
                c = own_curve(sens, "D0", name, ell)
                fb = at["fallback"] or 0.0
                full = at["own_full_sum"]
                curves[tag] = (fb + sum(own_at_vec(c[k], OPT_GRID) for k in range(len(c)))) / (fb + full)
                parts[tag] = (c, fb, full)
                targets[tag] = at[rule]["T_perbin"]
                fams.setdefault(f, []).append(tag)
        if not curves:
            continue                                   # smoke mode: per-bin targets need every bin
        fams["common"] = fams.get("P", []) + fams.get("E", [])
        res[rule] = {}
        for fam, tags in fams.items():
            dev = np.array([curves[t] / targets[t] - 1 for t in tags])        # [n_obs, n_grid]
            rms = np.sqrt(np.mean(dev**2, axis=0))
            i = int(np.argmin(rms))
            g = None
            if fam in ("P", "E"):
                g = out["global"][rule][fam]["R"]["family"].get("value")
            else:
                g = out["global"][rule]["common"]["R"]
            dev_g = None if g is None else [float((parts[t][1] + sum(own_at(parts[t][0][k], g) for k in range(len(parts[t][0]))))
                                                  / (parts[t][1] + parts[t][2]) / targets[t] - 1) for t in tags]
            res[rule][fam] = {
                "x_opt": float(OPT_GRID[i]), "rms_at_opt": float(rms[i]), "max_abs_dev_at_opt": float(np.abs(dev[:, i]).max()),
                "dev_at_opt": dict(zip(tags, dev[:, i].tolist())),
                "x_signal_weighted": g,
                "rms_at_signal_weighted": None if dev_g is None else float(np.sqrt(np.mean(np.square(dev_g)))),
                "dev_at_signal_weighted": None if dev_g is None else dict(zip(tags, dev_g)),
                "rms_within_0.001_of_min": [float(OPT_GRID[rms <= rms[i] + 1e-3].min()), float(OPT_GRID[rms <= rms[i] + 1e-3].max())],
                "observables": tags}
    return res


def global_sensitivity(out, rad, rnpz, W, bins_ok, idx):
    """How the family-weighted B32 radius depends on single bins: leave one bin out (weights renormalised over the
    rest; 16-84% from the same per-bin bootstrap rows), and the range of single-observable weightings."""
    res = {"definition": global_sensitivity.__doc__}
    labs_ok = [LC.LABELS[b] for b in bins_ok]
    for rule in ("B32", "Req", "B50"):
        res[rule] = {}
        for f in ("P", "E"):
            vals = np.array([rnpz[f"R__{rule}__{f}__{b}"] for b in bins_ok], float)
            vals = np.where(np.isfinite(vals), vals, 4.0)
            w_all = W[f"{f}:family"][idx]
            loo = {}
            for j, lab in enumerate(labs_ok):
                keep = np.array([i != j for i in range(len(labs_ok))])
                if keep.sum() == 0:
                    continue
                w = w_all[keep] / w_all[keep].sum()
                r = LC.boot_summary(w @ vals[keep, 1:], float(w @ vals[keep, 0]))
                r["weight_of_left_out_bin"] = float(w_all[j] / w_all.sum())
                loo[lab] = r
            singles = {o: out["global"][rule][f]["R"][o].get("value") for o in FAMILY[f]}
            sv = {o: v for o, v in singles.items() if v is not None}
            res[rule][f] = {"leave_one_bin_out": loo,
                            "single_observable": singles,
                            "single_observable_min": (min(sv.values()) if sv else None),
                            "single_observable_max": (max(sv.values()) if sv else None),
                            "single_observable_argmin": (min(sv, key=sv.get) if sv else None),
                            "single_observable_argmax": (max(sv, key=sv.get) if sv else None)}
        both = {f"{f}:{o}": v for f in ("P", "E") for o, v in res[rule][f]["single_observable"].items() if v is not None}
        if both:
            res[rule]["both_fields_single_observable"] = {"min": min(both.values()), "max": max(both.values()),
                                                         "argmin": min(both, key=both.get), "argmax": max(both, key=both.get)}
    return res


def main():
    t0 = LC.time.perf_counter()
    rad = LC.jload(LC.RESULTS / "radii.json")
    rnpz = np.load(LC.RESULTS / "radii.npz")
    sens, jsumm = load_j()
    labs, W = weights(jsumm)
    bins_ok = [b for b in LC.J_BINS if LC.LABELS[b] in rad["bins"]]
    idx = [LC.J_BINS.index(b) for b in bins_ok]
    out = {"weights": {k: dict(zip(labs, v.tolist())) for k, v in W.items()}, "global": {}, "mass_dependence": {},
           "additive_terms": {}, "truncated_target_weights": {}, "d3_convergence": {}, "definitions": __doc__}
    # ---------------------------------------------------------------- 2. signal-weighted radii
    for rule in MAIN_RULES:
        out["global"][rule] = {}
        for f in ("P", "E"):
            out["global"][rule][f] = {}
            for unit in ("R", "R200m", "R500c"):
                vals = np.array([rnpz[f"{unit}__{rule}__{f}__{b}"] for b in bins_ok], float)    # [nb, rows]
                st = np.array([rad["bins"][LC.LABELS[b]]["rules"][rule][f]["status"] for b in bins_ok])
                cens = np.array([rad["bins"][LC.LABELS[b]]["rules"][rule][f].get("censored_fraction_boot", 0) for b in bins_ok])
                if unit == "R":
                    vals = np.where(np.isfinite(vals), vals, 4.0)                 # censored -> 4R (lower limit)
                for wname in ("family",) + tuple(o for o in FAMILY[f]):
                    w = W[f"{f}:{wname}"][idx]
                    w = w / w.sum()
                    v0 = vals[:, 0]
                    if not np.all(np.isfinite(v0)):
                        res = {"value": None, "note": "undefined in some bin (censored)"}
                    else:
                        Rbar = float(w @ v0)
                        boot = w @ np.where(np.isfinite(vals[:, 1:]), vals[:, 1:], np.nan)
                        res = LC.boot_summary(boot, Rbar)
                        res["cv"] = float(np.sqrt(w @ (v0 - Rbar) ** 2) / Rbar)
                        sel = [i for i, b in enumerate(bins_ok) if 2 <= b <= 13]
                        res["max_over_min_12.9-15.3"] = float(v0[sel].max() / v0[sel].min()) if sel else None
                    res["bins_censored_central"] = int(np.sum(st == "above"))
                    res["max_censored_fraction_boot"] = float(cens.max())
                    out["global"][rule][f].setdefault(unit, {})[wname] = res
            # common radius (both fields): mean of the two family-weighted values
        for unit in ("R", "R200m", "R500c"):
            a = out["global"][rule]["P"][unit]["family"].get("value")
            b_ = out["global"][rule]["E"][unit]["family"].get("value")
            out["global"][rule].setdefault("common", {})[unit] = (None if a is None or b_ is None else 0.5 * (a + b_))
    # ---------------------------------------------------------------- 3./4. additive terms (exact own curves)
    for f, obs in OWN_OBS.items():
        for name, ell in obs:
            tag = f"{name}@{ell}" if ell else name
            cD0 = own_curve(sens, "D0", name, ell)
            has_d3 = f"trunc_own__D3__{name}" in sens.files and f == "E"
            cD3 = own_curve(sens, "D3", name, ell) if has_d3 else None
            full = cD0[:, -1].sum()
            fb = None
            if name in ("ymean", "dm_mean", "dm_var", "ksz1h", "dmcl1h"):
                al = np.asarray(sens[f"trunc_all__D0__{name}"], float)
                if al.ndim == 2:
                    li = int(np.argmin(np.abs(sens["ell"] - ell)))
                    fb = float(al[0, li])
                else:
                    fb = float(al[0])
            res = {"own_full_sum": float(full), "fallback": fb}
            for rule in MAIN_RULES:
                Rb = []
                for b in LC.J_BINS:
                    lab = LC.LABELS[b]
                    if lab not in rad["bins"]:
                        Rb.append(np.nan)
                        continue
                    v = rad["bins"][lab]["rules"][rule][f]["value"]
                    Rb.append(4.0 if v is None else min(v, 4.0))
                Rb = np.array(Rb)
                if not np.all(np.isfinite(Rb)):
                    continue
                per = np.array([own_at(cD0[c], Rb[c]) for c in range(len(Rb))])
                g = out["global"][rule][f]["R"]["family"].get("value")
                glob = np.array([own_at(cD0[c], min(g, 4.0)) for c in range(len(Rb))]) if g else None
                fbv = 0.0 if fb is None else fb
                r = {"T_perbin": float((fbv + per.sum()) / (fbv + full)),
                     "T_perbin_calibrated_only": float(per.sum() / full),
                     "R_global": g,
                     "T_global": None if glob is None else float((fbv + glob.sum()) / (fbv + full)),
                     "shares_truncated_target": dict(zip(labs, (per / per.sum()).tolist())),
                     "shares_D0": dict(zip(labs, (cD0[:, -1] / full).tolist()))}
                if cD3 is not None:
                    r["perbin_over_D3"] = float(per.sum() / cD3[:, -1].sum())
                    r["global_over_D3"] = None if glob is None else float(glob.sum() / cD3[:, -1].sum())
                res[rule] = r
            out["additive_terms"][tag] = res
    # self-consistent family weights for the truncated target (1-halo / linear observables of each field)
    for rule in MAIN_RULES:
        out["truncated_target_weights"][rule] = {}
        for f, obs in OWN_OBS.items():
            tags = [f"{n}@{e}" if e else n for n, e in obs]
            ws = [np.array([out["additive_terms"][t][rule]["shares_truncated_target"][lab] for lab in labs])
                  for t in tags if rule in out["additive_terms"][t]]
            if not ws:
                continue
            wf = np.mean(ws, axis=0)
            Rv = np.array([rad["bins"][lab]["rules"][rule][f]["value"] if lab in rad["bins"] else np.nan for lab in labs],
                          dtype=float)
            ok = np.isfinite(Rv)
            out["truncated_target_weights"][rule][f] = {
                "weights": dict(zip(labs, wf.tolist())),
                "R_weighted": float(np.sum(wf[ok] * Rv[ok]) / np.sum(wf[ok])) if ok.any() else None,
                "observables": tags}
    # ---------------------------------------------------------------- 6. observable-optimal single cut (additive terms)
    out["observable_optimal"] = observable_optimal(out, sens, rad)
    # ---------------------------------------------------------------- 7. sensitivity of the signal-weighted radius
    out["global_sensitivity"] = global_sensitivity(out, rad, rnpz, W, bins_ok, idx)
    # ---------------------------------------------------------------- band bound shares (stack level)
    import l_stacks as LS
    bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    bb = {}
    for b in LC.BINS:
        st = LS.bin_stacks(b)["S"]
        for f in ("P", "E"):
            for den in ("a2ii", "all"):
                vals = []
                for lo, hi in bands:
                    m = (LC.X[:-1] >= lo - 1e-9) & (LC.X[1:] <= hi + 1e-9)
                    vals.append(float((st[f"{f}_hb"][0, m] @ LC.VOL[m]) / (st[f"{f}_{den}"][0, m] @ LC.VOL[m])))
                bb.setdefault(LC.LABELS[b], {})[f"{f}_hb_over_{den}"] = vals
    dev = {}
    for f in ("P", "E"):
        m = 0.0
        for b in LC.BINS:
            v, xr, _ = LS.fraction(b, f, "a2ii/all")
            m = max(m, float(1 - np.nanmin(v[0, xr < 1.5])))
        dev[f] = m
    out["a2ii_vs_all_inside_1.5R"] = {"max_1_minus_a2ii_over_all": dev,
                                      "definition": "largest 1 - (A2-ii / all gas) of a stacked shell with radius < 1.5 R200c"}
    out["band_bound_shares"] = {"bands": bands, "bins": bb,
                                "definition": "X_hb(band) / X_ref(band), X = sum_s V_s F_s over the band's fine shells"}
    # per-shell (per-group) bound fractions inside each band: min / max of the stacked f (added 2026-10-16 after review)
    psr = {}
    for b in LC.BINS:
        for f in ("P", "E"):
            for name in ("hb/all", "cb/all"):
                v, xr, _ = LS.fraction(b, f, name)
                for lo, hi in bands[1:]:
                    m = (xr > lo) & (xr < hi)
                    psr.setdefault(LC.LABELS[b], {})[f"{f}_{name.replace('/', '_over_')}_{lo:g}-{hi:g}"] = (
                        [float(np.nanmin(v[0, m])), float(np.nanmax(v[0, m]))] if m.any() else None)
    out["per_shell_fraction_ranges"] = {"bins": psr, "definition": "min and max of the stacked per-shell (per-group in "
                                        "12.7-13.5) bound fraction f over the shells whose volume-weighted radius x_g "
                                        "lies inside the band (lo < x_g < hi)"}
    # B32 bootstrap half-widths (q84 - q16) / 2 per bin and field (added 2026-10-16 after review)
    hw = {}
    for lab, d in rad["bins"].items():
        for f in ("P", "E"):
            s_ = d["rules"]["B32"][f]
            hw.setdefault(lab, {})[f] = float(0.5 * (s_["q84"] - s_["q16"]))
    below = {f"{lab}:{f}": v for lab, d in hw.items() for f, v in d.items() if float(lab.split("-")[0]) < 15.3 - 1e-9}
    out["B32_bootstrap_halfwidths"] = {
        "bins": hw, "below_15.3_min": min(below.values()) if below else None,
        "below_15.3_max": max(below.values()) if below else None,
        "below_15.3_argmin": min(below, key=below.get) if below else None,
        "below_15.3_argmax": max(below, key=below.get) if below else None,
        "below_15.3_median": float(np.median(list(below.values()))) if below else None,
        "definition": "(q84 - q16) / 2 of the 200 paired halo-bootstrap B32 radii [R200c]; 'below_15.3' = bins 12.7-15.3"}
    # low-mass (12.7-13.5) and high-mass (>= 14.5) shares of each weight set
    low = [lab for lab in labs if float(lab.split("-")[0]) < 13.5 - 1e-9]
    high = [lab for lab in labs if float(lab.split("-")[0]) >= 14.5 - 1e-9]
    ws = {}
    for k, v in out["weights"].items():
        ws[f"D0:{k}"] = {"low_12.7-13.5": float(sum(v[lab] for lab in low)), "high_ge_14.5": float(sum(v[lab] for lab in high))}
    for rule, d in out["truncated_target_weights"].items():
        for f, v in d.items():
            ws[f"truncated:{rule}:{f}"] = {"low_12.7-13.5": float(sum(v["weights"][lab] for lab in low)),
                                           "high_ge_14.5": float(sum(v["weights"][lab] for lab in high))}
    for f, obs in OWN_OBS.items():                    # D0 own-term shares of the same 1-halo / linear observables
        tags = [f"{n}@{e}" if e else n for n, e in obs]
        if not all("B32" in out["additive_terms"][t] for t in tags):
            continue                                  # smoke mode: not every bin evaluated
        wf = np.mean([[out["additive_terms"][t]["B32"]["shares_D0"][lab] for lab in labs] for t in tags], axis=0)
        ws[f"D0own:{f}"] = {"low_12.7-13.5": float(sum(w_ for w_, lab in zip(wf, labs) if lab in low)),
                            "high_ge_14.5": float(sum(w_ for w_, lab in zip(wf, labs) if lab in high))}
    out["weight_mass_shares"] = ws
    # ---------------------------------------------------------------- 5. D3 convergence (J's truth curves)
    d3 = {}
    for k, v in jsumm["own_radii"].items():
        if not k.startswith("D3__"):
            continue
        _, obs, lab = k.split("__")
        d3.setdefault(obs, {})[lab] = {kk: v[kk] for kk in ("R50", "R80", "R90", "R95", "R99")}
    out["d3_convergence"]["own_bin"] = d3
    out["d3_convergence"]["global"] = {o: {kk: v[kk] for kk in ("R50", "R90", "R95", "R99") if kk in v}
                                       for o, v in jsumm["radii"]["D3"].items()}
    out["d3_convergence"]["source"] = "J_halo_dm_lowmass/results/sens_summary.json: own_radii['D3__*'], radii['D3']"
    out["seconds"] = LC.time.perf_counter() - t0
    LC.jdump(out, LC.RESULTS / "global.json")
    # tables
    with open(LC.TABLES / "L1_global.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "field", "unit", "weights", "Rbar", "q16", "q84", "cv", "max_over_min_12.9-15.3"])
        for rule in MAIN_RULES:
            for f in ("P", "E"):
                for unit in ("R", "R200m", "R500c"):
                    for wn, r in out["global"][rule][f][unit].items():
                        w.writerow([rule, f, unit if unit != "R" else "R200c", wn, r.get("value"), r.get("q16"),
                                    r.get("q84"), r.get("cv"), r.get("max_over_min_12.9-15.3")])
    with open(LC.TABLES / "L1_weights.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["weights"] + labs)
        for k, v in W.items():
            w.writerow([k] + [f"{x:.5g}" for x in v])
    LC.log(f"global done {out['seconds']:.1f} s")


if __name__ == "__main__":
    main()
