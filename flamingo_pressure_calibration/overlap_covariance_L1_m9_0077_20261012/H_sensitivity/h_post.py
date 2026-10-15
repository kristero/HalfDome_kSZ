"""Post-run corrections of Test H (local, written 2026-10-14 after the round-2 verification; no new physics).

Reads ONLY the fetched maas results (results/*.npz, *.json), the H0 inputs (inputs/h0_profiles_x200.npz) and Test F's
truth (F_observables/results/truth.npz).  No fine basis is needed: every quantity below is an exact function of the
stored arrays, except where a value is labelled 'estimate'.  Writes post/h_post.json and post/H_post_*.csv.

1. fine_vs_F_truth_absmax: |O_fine / O_F - 1| (fine 67-shell truth vs Test F's 35-shell truth of the same selection)
   over all 106 multipoles, over ell <= 3000 and over the five selected multipoles, per observable.
2. Calibrated-halo-only shares.  The stored importance maps I_O[b, s] of engine bin 15.3-15.5 contain, besides the
   FLAMINGO-calibrated haloes (10^15.3 <= M200c <= 10^15.388), the lightcone haloes above 10^15.388 painted with the
   published B12/B16 fallback.  The calibrated-halo share of the importance, c_cal = (1 / k O) dO/d ln A_cal (A_cal = a
   common amplitude of all calibrated haloes), follows exactly from three stored numbers: O (truth to 4 R200c),
   O_cal (truthcal: calibrated haloes only) and O_fb (truncation curve at R_t = 0: fallback haloes only):
     linear and one-halo-type observables (mean y, mean DM, 1-halo C_ell, DM variance):  c_cal = 1 - O_fb / O
     observables with a two-halo term (total and 2-halo C_ell; O = O_cal + O_fb + 2X, X the cal x fb cross term):
                                                                            c_cal = (O + O_cal - O_fb) / (2 O)
     each side of the y x DM cross spectrum (degree 1 in that side's profiles):
                                                    c_cal,P = 1 - O(P_cal = 0) / O,  c_cal,E = 1 - O(E_cal = 0) / O
     with O(P_cal = 0) the stored P-side truncation curve at R_t = 0 (trunc_all__*__ydmtot_Pcut[0]), same for E.
     DM quantiles (Mecke Jacobians): exact per shell, the fallback part is the stored Jqfb in the calibrated bins.
   fallback inside the calibrated engine bins = sum_{b in cal} I_O[b, .] - c_cal.
3. Per-shell calibrated-halo importance for the tolerance: exact for mean y and mean DM (each bin's own linear
   contribution, differences of trunc_own) and for the DM quantiles (Jqfb); for the quadratic and cross observables
   an ESTIMATE that removes the fallback part of bin 15.3-15.5 in proportion to that bin's per-shell importance.
4. Unresolved halo-shells (base mask false) as % of the total observable, of the calibrated-engine-bin importance and
   of the calibrated-halo importance.
5. Report tables at full precision (rounded once in the report): peaks, truncation radii and losses, per-halo
   DM-quantile losses, coherent band tolerances, double counting per shell, and five radial bands of whole fine shells.
   Main band edges (bands5): 0, 0.2335, 0.5185, 1, 2, 4 R200c = Test F's shell edges (also fine-grid edges), labels
   '<0.233R', '0.233-0.519R', '0.519-1R', '1-2R', '2-4R'.  Alternative (bands5_alt_edges): the fine edges nearest
   below 0.25 and 0.5 R200c (0.2471, 0.4734), i.e. each shell assigned by its outer edge to <=0.25 / <=0.5 R200c;
   it shows how strongly the split near the tSZ peak depends on the edge choice.  [Round-3 fix, 2026-10-14: the
   first version labelled the main bands '<0.25R / 0.25-0.5R / 0.5-1R' although the edges are 0.2335 / 0.5185.]
   The maas run's 8-band table (results/tables/H3_bands.csv) has the same rounding in its labels (edges 0.1051 /
   0.2335 / 0.5185 labelled 0.1 / 0.25 / 0.5): post/H_post_bands8.csv repeats it with the edges used, each value
   recomputed from results/sens.npz (bands8_relabel).
6. Mass-binning family RMS at R_cut = 4 (binning_family_rms_pct_Rcut4), every entry named by what it is:
     det_binning_only      RMS of e_O(scheme, det) - e_O(M200c_0.2 piecewise, det), e_O vs the truth mean; the
                           reference is exact (e = 0), so this equals binning_summary's family_rms_vs_truth_mean
     S1_binning_only_dO    RMS of d_O = e_O(scheme, S1) - e_O(M200c_0.2 piecewise, S1), e_O vs the scatter-inclusive
                           truth: the binning-only error with S1 scatter (S1-model error and truncation cancel)
     S1_total_vs_scatter_truth  RMS of e_O(scheme, S1) itself = binning + S1-model error (family_rms_vs_truth)
   families: tSZ = C^yy at ell 100/300/1000/3000; kSZ = 1-halo at the same ell; tSZxDM = total at the same ell;
   DM = mean, variance and the five quantiles Q16/Q50/Q84/Q97.5/Q99.9 (all evaluated at R_cut = 4);
   DM_moments = mean and variance only.  [Round-3 fix: the first version stored the S1 total error under the same
   key as the det binning-only error.]
7. Diagonal (own-shell) fraction of the calibrated haloes' 1-halo terms, with both normalisations:
     f_diag,cal = sum_{c,s} K_{c,ss} <F_s^2>_c / O_cal^1h     (results/sens_summary.json:self_fraction)
     f_diag,tot = sum_{c,s} K_{c,ss} <F_s^2>_c / O^1h  = f_diag,cal x O_cal^1h / O^1h
   O_cal^1h = truthcal 1-halo term (calibrated haloes only), O^1h = truth 1-halo term (all haloes).
Usage: python -B h_post.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import report_numbers as RN          # local post-processing helpers (crossing / converged radii); read-only use

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "post"
F_TRUTH = HERE.parent / "F_observables" / "results" / "truth.npz"
SEL_ELL = (100, 300, 1000, 3000, 8000)
QN = ("q16", "q50", "q84", "q975", "q999")
LIN = {"ymean", "dm_mean"}
ONEH = {"yy1h", "ksz1h", "dmcl1h", "dm_var", "ydm1h"}
TWOH = {"yytot", "yy2h", "ksztot", "ksz2h", "dmcltot", "dmcl2h", "ydmtot", "ydm2h"}
K_OF = {k: 2 for k in ("yytot", "yy1h", "yy2h", "ksz1h", "ksz2h", "ksztot", "dm_var", "dmcltot", "dmcl1h", "dmcl2h")}
B5 = (0.0, 0.2335, 0.5185, 1.0, 2.0, 4.0)                 # Test F's shell edges (fine-grid edges 0.23347, 0.51851)
B5_LAB = ("<0.233R", "0.233-0.519R", "0.519-1R", "1-2R", "2-4R")
B5_ALT = (0.0, 0.2471, 0.4734, 1.0, 2.0, 4.0)              # fine edges nearest below 0.25 / 0.5 (0.24707, 0.47341)
B5_ALT_LAB = ("<0.247R", "0.247-0.473R", "0.473-1R", "1-2R", "2-4R")
BIN_FAM = {"tSZ": [f"yytot_{e}" for e in (100, 300, 1000, 3000)], "kSZ": [f"ksz1h_{e}" for e in (100, 300, 1000, 3000)],
           "DM": ["dm_mean", "dm_var"] + [f"dm_{q}" for q in QN], "DM_moments": ["dm_mean", "dm_var"],
           "tSZxDM": [f"ydmtot_{e}" for e in (100, 300, 1000, 3000)]}
REPORT_ROWS = [("tSZ yy ell=100", "yytot", 100), ("tSZ yy ell=1000", "yytot", 1000), ("tSZ yy ell=3000", "yytot", 3000),
               ("mean y", "ymean", None), ("kSZ 1h ell=100", "ksz1h", 100), ("kSZ 1h ell=1000", "ksz1h", 1000),
               ("kSZ 1h ell=3000", "ksz1h", 3000), ("mean DM", "dm_mean", None), ("DM variance", "dm_var", None),
               ("DM Q16", "dm_q16", None), ("DM Q50", "dm_q50", None), ("DM Q84", "dm_q84", None),
               ("DM Q99.9", "dm_q999", None), ("DM-field C ell=1000", "dmcltot", 1000),
               ("tSZxDM ell=1000, P side", "ydmtot_P", 1000), ("tSZxDM ell=1000, e side", "ydmtot_E", 1000),
               ("tSZ yy 1h ell=100", "yy1h", 100), ("tSZ yy 1h ell=1000", "yy1h", 1000)]


def edge(x, v):
    k = int(np.argmin(np.abs(x - v)))
    assert abs(x[k] - v) < 1e-3, (v, x[k])
    return k


def main():
    OUT.mkdir(exist_ok=True)
    S = dict(np.load(RES / "sens.npz"))
    T = dict(np.load(RES / "truth.npz"))
    FT = dict(np.load(F_TRUTH))
    ts = json.loads((RES / "truth_summary.json").read_text())
    ss = json.loads((RES / "sens_summary.json").read_text())
    h0 = np.load(HERE / "inputs" / "h0_profiles_x200.npz")
    x, dln, ell = S["x_edges"], S["dlnx"], S["ell"]
    li = {e: int(np.flatnonzero(ell == e)[0]) for e in SEL_ELL}
    cal = [int(b) for b in S["cal_bins"]]
    calm = np.zeros(16, bool)
    calm[cal] = True
    labels = [str(v) for v in S["engine_bin_labels"]]
    top = cal[-1]                                                  # engine bin 15.3-15.5
    mid = np.sqrt(np.maximum(x[:-1], 1e-6) * x[1:])                # as h_sens (tolerance / peak positions)
    out = {"definitions": __doc__, "inputs": {"results": str(RES), "F_truth": str(F_TRUTH)}}

    # ------------------------------------------------------------------ 1. fine vs F truth, unsigned maxima
    fv, rows_f = {}, []
    for sel in ("a2ii", "main"):
        fv[sel] = {}
        for k in sorted(T):
            if not k.startswith(f"truth_{sel}__") or k not in FT:
                continue
            name = k.split("__")[1]
            d = np.abs(np.asarray(T[k], float) / np.asarray(FT[k], float) - 1)
            if d.ndim == 0:
                rec = {"scalar": float(d)}
            else:
                rec = {"all_ell": float(d.max()), "ell_at_max": float(ell[int(np.argmax(d))]),
                       "ell_le_3000": float(d[ell <= 3000].max()),
                       "ell_at_max_le_3000": float(ell[ell <= 3000][int(np.argmax(d[ell <= 3000]))]),
                       "sel_ell": float(d[[li[e] for e in SEL_ELL]].max()),
                       "sel_ell_at_max": float(SEL_ELL[int(np.argmax(d[[li[e] for e in SEL_ELL]]))])}
            fv[sel][name] = rec
            rows_f.append(dict(selection=sel, observable=name, **rec))
        arr = {n: r for n, r in fv[sel].items() if "all_ell" in r}
        sc = {n: r["scalar"] for n, r in fv[sel].items() if "scalar" in r}
        worst = lambda f: max(((r[f], n) for n, r in arr.items()), key=lambda t: t[0])   # noqa: E731
        fv[sel]["_max"] = {"all_ell": worst("all_ell") + (arr[worst("all_ell")[1]]["ell_at_max"],),
                           "ell_le_3000": worst("ell_le_3000") + (arr[worst("ell_le_3000")[1]]["ell_at_max_le_3000"],),
                           "sel_ell": worst("sel_ell") + (arr[worst("sel_ell")[1]]["sel_ell_at_max"],),
                           "scalars": max(sc.values()) if sc else None}
    out["fine_vs_F_truth_absmax"] = fv

    # ------------------------------------------------------------------ 2. calibrated-halo shares (exact)
    Q = {"imp": np.asarray(ts["pdf"]["truth_a2ii"]["quantiles"], float),
         "impdet": np.asarray(ts["pdf"]["truthmean_a2ii"]["quantiles"], float)}
    JQ = {"imp": (T["Jq_truth_a2ii_subsample"], T["Jqfb_truth_a2ii_subsample"]),
          "impdet": (T["Jq_truthmean_a2ii"], T["Jqfb_truthmean_a2ii"])}
    # reconstruct the stored quantile maps from truth.npz (check) and keep the fallback part in the calibrated bins
    qcheck, qfb = 0.0, {}
    for tag in ("imp", "impdet"):
        Jq, Jqfb = JQ[tag]
        for p, qn in enumerate(QN):
            M = np.array(Jqfb[p], float, copy=True)
            for c, b in enumerate(cal):
                M[b] += Jq[p, c, :len(dln)]
            M /= Q[tag][p]
            qcheck = max(qcheck, float(np.max(np.abs(M - S[f"{tag}__a2ii__dm_{qn}"]))))
            qfb[(tag, qn)] = np.asarray(Jqfb[p], float)[calm] / Q[tag][p]          # [10, 67]
    out["quantile_map_reconstruction_max_abs"] = qcheck

    def tv(key, e, tag="truth"):
        v = np.asarray(T[f"{tag}_a2ii__{key}"], float)
        return float(v if v.ndim == 0 else v[li[e]])

    def curve0(key, e, sel="a2ii"):
        c = np.asarray(S[f"trunc_all__{sel}__{key}"], float)
        return float(c[0] if c.ndim == 1 else c[0, li[e]])

    def imp(name, e, tag="imp", sel="a2ii"):
        M = S[f"{tag}__{sel}__{name}"]
        return M if M.ndim == 2 else M[..., li[e]]

    shares, rows_s = {}, []
    names = ["yytot", "yy1h", "yy2h", "ymean", "ksz1h", "ksz2h", "ksztot", "dm_mean", "dm_var", "dmcltot", "dmcl1h",
             "dmcl2h", "ydmtot_P", "ydmtot_E", "ydm1h_P", "ydm1h_E"] + [f"dm_{q}" for q in QN]
    for name in names:
        ats = [None] if (name in LIN or name == "dm_var" or name.startswith("dm_q") or name == "dm_mean") else SEL_ELL
        for e in ats:
            m = imp(name, e)
            per_bin = m.sum(axis=1)
            cal_engine = float(per_bin[calm].sum())
            fb_out = float(per_bin[~calm].sum())
            if name.startswith("dm_q"):
                fbin = float(qfb[("imp", name[3:])].sum())
                ccal = cal_engine - fbin
                method = "Mecke Jacobians: fallback part of the calibrated bins = stored Jqfb (exact)"
                check = None
            elif name.startswith("ydm"):
                side = name[-1]
                base = name[:-2]                                   # ydmtot / ydm1h
                O = tv(base, e)
                O0 = curve0(f"{base}_{side}cut", e)
                ccal = 1 - O0 / O
                fbin = cal_engine - ccal
                method = f"1 - O({side}_cal = 0) / O (stored {side}-side truncation curve at R_t = 0)"
                check = None
                if base == "ydm1h":
                    check = abs(ccal - tv("ydm1h", e, "truthcal") / O)
            else:
                O, Ofb = tv(name, e), curve0(name, e)
                Ocal = tv(name, e, "truthcal")
                if name in TWOH:
                    ccal = (O + Ocal - Ofb) / (2 * O)
                    method = "(O + O_cal - O_fb) / (2 O)"
                    check = None
                else:
                    ccal = 1 - Ofb / O
                    method = "1 - O_fb / O"
                    check = abs(ccal - Ocal / O)                   # one-halo-type / linear: equals O_cal / O
                fbin = cal_engine - ccal
            key = f"{name}@{e if e is not None else 'scalar'}"
            rec = {"calibrated_haloes": ccal, "calibrated_engine_bins_incl_fallback": cal_engine,
                   "fallback_inside_calibrated_bins": fbin, "fallback_outside_calibrated_bins": fb_out,
                   "bin_15.3-15.5_total": float(per_bin[top]),
                   "bin_15.3-15.5_calibrated_haloes": float(per_bin[top]) - fbin,
                   "fallback_share_of_bin_15.3-15.5": fbin / float(per_bin[top]) if per_bin[top] else None,
                   "sum_all_bins": float(per_bin.sum()), "method": method}
            if check is not None:
                rec["check_vs_truthcal"] = check
            shares[key] = rec
            rows_s.append(dict(observable=name, ell=e if e is not None else "", **{k: v for k, v in rec.items()
                                                                                    if k != "method"}))
    out["calibrated_shares"] = shares
    # checks of the decomposition with independent stored arrays
    #  (a) additivity: c_cal(tot) O_tot = c_cal(1h) O_1h + c_cal(2h) O_2h
    #  (b) one-halo terms: calibrated part of bin 15.3-15.5 = that bin's own 1-halo signal (trunc_own, full radius)
    #      / O, so fallback_inside = I[15.3-15.5] - own / O (independent of the truncation zero point)
    #  (c) bounds: 0 <= fallback inside the calibrated bins <= importance of bin 15.3-15.5 (every observable)
    chk = {"additivity_max_abs": 0.0, "bin_top_1h_vs_own_max_abs": 0.0}
    for tot, a, b2 in (("yytot", "yy1h", "yy2h"), ("ksztot", "ksz1h", "ksz2h"), ("dmcltot", "dmcl1h", "dmcl2h")):
        for e in SEL_ELL:
            lhs = shares[f"{tot}@{e}"]["calibrated_haloes"] * tv(tot, e)
            rhs = shares[f"{a}@{e}"]["calibrated_haloes"] * tv(a, e) + shares[f"{b2}@{e}"]["calibrated_haloes"] * tv(b2, e)
            chk["additivity_max_abs"] = max(chk["additivity_max_abs"], abs(lhs / tv(tot, e) - rhs / tv(tot, e)))
            own1 = float(S[f"trunc_own__a2ii__{a}"][-1, -1, li[e]])
            alt = float(imp(a, e)[top].sum()) - own1 / tv(a, e)
            chk["bin_top_1h_vs_own_max_abs"] = max(chk["bin_top_1h_vs_own_max_abs"],
                                                   abs(alt - shares[f"{a}@{e}"]["fallback_inside_calibrated_bins"]))
    chk["bounds_violations"] = [k for k, v in shares.items() if not (-1e-12 <= v["fallback_inside_calibrated_bins"]
                                                                     <= v["bin_15.3-15.5_total"] + 1e-12)]
    out["calibrated_shares_checks"] = chk

    # ------------------------------------------------------------------ 3. tolerance: calibrated haloes per shell
    tol, rows_t = {}, []
    own_lin = {"ymean": S["trunc_own__a2ii__ymean"], "dm_mean": S["trunc_own__a2ii__dm_mean"]}
    lin_check = 0.0
    for name, e in (("yytot", 100), ("yytot", 1000), ("yytot", 3000), ("ksz1h", 1000), ("ksz1h", 100),
                    ("dm_mean", None), ("dm_var", None), ("dm_q50", None), ("dm_q16", None), ("dm_q999", None),
                    ("ymean", None), ("ydmtot_E", 1000), ("ydmtot_P", 1000), ("dmcltot", 1000)):
        m = imp(name, e)
        k = K_OF.get(name, 1)
        per_eng = m[calm].sum(axis=0)
        key = f"{name}@{e if e is not None else 'scalar'}"
        if name in own_lin:
            per_cal = np.diff(own_lin[name], axis=1).sum(axis=0) / tv(name, None)
            kind = "exact (each bin's own linear contribution)"
            # bins 13.5-15.3 hold calibrated haloes only: their stored rows must equal the own contributions
            dif = np.diff(own_lin[name], axis=1) / tv(name, None)
            lin_check = max(lin_check, float(np.max(np.abs(dif[:-1] - m[cal[:-1]]))))
        elif name.startswith("dm_q"):
            per_cal = per_eng - qfb[("imp", name[3:])].sum(axis=0)
            kind = "exact (Mecke Jacobians minus stored Jqfb)"
        else:
            fbin = shares[key]["fallback_inside_calibrated_bins"]
            row = m[top]
            per_cal = per_eng - row * (fbin / row.sum())
            kind = "estimate (fallback part of bin 15.3-15.5 removed in proportion to its per-shell importance)"
        with np.errstate(divide="ignore"):
            t_eng = 0.01 / np.abs(k * (per_eng / dln) * 0.1 * np.log(10))
            t_cal = 0.01 / np.abs(k * (per_cal / dln) * 0.1 * np.log(10))
        je, jc = int(np.argmin(t_eng)), int(np.argmin(t_cal))
        at = lambda t, r: float(np.interp(np.log(r), np.log(mid), t))   # noqa: E731
        radii = (0.1, 0.2, 0.3, 0.43, 0.5, 0.6, 1.0, 1.5, 2.0, 3.0, 3.8)
        tol[key] = {"calibrated_engine_bins": {"min": float(t_eng[je]), "at_x": float(mid[je]),
                                               "at": {f"{r:g}": at(t_eng, r) for r in radii}},
                    "calibrated_haloes": {"min": float(t_cal[jc]), "at_x": float(mid[jc]),
                                          "at": {f"{r:g}": at(t_cal, r) for r in radii}, "kind": kind},
                    "max_rel_change_over_shells": float(np.nanmax(np.abs(t_cal / t_eng - 1)[np.isfinite(t_eng)]))}
        # range of the tolerance over 1-4R and 1.5-4R (calibrated engine bins, unsmoothed shells)
        for lo, hi in ((1.0, 4.0), (1.5, 4.0), (0.3, 0.6)):
            msk = (mid >= lo) & (mid <= hi)
            tol[key]["calibrated_engine_bins"][f"range_{lo:g}_{hi:g}R"] = [float(t_eng[msk].min()),
                                                                          float(t_eng[msk].max())]
        for s in range(len(dln)):
            rows_t.append(dict(observable=name, ell=e if e is not None else "", x_lo=x[s], x_hi=x[s + 1],
                               importance_calibrated_engine_bins=per_eng[s], importance_calibrated_haloes=per_cal[s],
                               tol_per_0p1dex_engine_bins=t_eng[s], tol_per_0p1dex_calibrated_haloes=t_cal[s],
                               kind=kind))
    out["tolerance"] = tol
    out["linear_own_contribution_vs_stored_rows_13.5-15.3_max_abs"] = lin_check

    # ------------------------------------------------------------------ 4. unresolved halo-shells
    base = h0["base_mask"]
    unres = {}
    for name, e in (("yytot", 100), ("yytot", 1000), ("yytot", 3000), ("ksz1h", 1000), ("dm_mean", None),
                    ("dm_var", None), ("dm_q999", None), ("dm_q50", None)):
        m = imp(name, e)
        u = float(sum(m[b][~base[b]].sum() for b in cal))
        key = f"{name}@{e if e is not None else 'scalar'}"
        unres[key] = {"pct_of_total_observable": 100 * u,
                      "pct_of_calibrated_engine_bin_importance": 100 * u / float(m[calm].sum()),
                      "pct_of_calibrated_halo_importance": 100 * u / shares[key]["calibrated_haloes"]}
    out["unresolved"] = unres
    out["first_resolved_inner_edge"] = {labels[b]: float(x[int(np.argmax(base[b]))]) for b in cal}

    # ------------------------------------------------------------------ 5a. bands (whole fine shells), peaks
    bk = [edge(x, v) for v in B5]
    bka = [edge(x, v) for v in B5_ALT]
    bands, bands_alt, rows_b = {}, {}, []
    for lab, name, e in REPORT_ROWS:
        m = imp(name, e)
        per = m.sum(axis=0)
        pc = m[calm].sum(axis=0)
        b_all = [float(per[bk[q]:bk[q + 1]].sum()) for q in range(5)]
        b_cal = [float(pc[bk[q]:bk[q + 1]].sum()) for q in range(5)]
        b_alt = [float(per[bka[q]:bka[q + 1]].sum()) for q in range(5)]
        dens = per / dln
        key = f"{name}@{e if e is not None else 'scalar'}"
        bands[lab] = {"key": key, "all_bins": dict(zip(B5_LAB, b_all)), "calibrated_engine_bins": dict(zip(B5_LAB, b_cal)),
                      "sum_all": float(per.sum()), "peak_x": float(mid[int(np.argmax(dens))]),
                      "calibrated_haloes_share": shares[key]["calibrated_haloes"],
                      "calibrated_engine_bins_share": shares[key]["calibrated_engine_bins_incl_fallback"]}
        bands_alt[lab] = {"key": key, "all_bins": dict(zip(B5_ALT_LAB, b_alt))}
        rows_b.append(dict(row=lab, key=key, **{f"all_{k}": v for k, v in zip(B5_LAB, b_all)},
                           **{f"calengine_{k}": v for k, v in zip(B5_LAB, b_cal)}, sum_all=float(per.sum()),
                           peak_x=bands[lab]["peak_x"], calibrated_haloes=shares[key]["calibrated_haloes"],
                           **{f"alt_all_{k}": v for k, v in zip(B5_ALT_LAB, b_alt)}))
    out["band_edges"] = {"bands5": [float(x[k]) for k in bk], "bands5_alt_edges": [float(x[k]) for k in bka],
                         "note": "bands are sums over whole fine shells between these grid edges (R200c); bands5 uses "
                                 "Test F's shell edges 0.2335 / 0.5185, bands5_alt_edges the fine edges nearest below "
                                 "0.25 / 0.5 (assignment of each shell by its outer edge)"}
    out["bands5"] = bands
    out["bands5_alt_edges"] = bands_alt
    # the maas run's 8-band table (results/tables/H3_bands.csv, sens_summary.json:bands) uses whole fine shells between
    # the edges 0.1051 / 0.2335 / 0.5185 R200c but labels them 0.1 / 0.25 / 0.5: relabel with the edges used and check
    # every value against the stored importance maps (results/ itself stays unchanged)
    e8 = (0.0, 0.1051, 0.2335, 0.5185, 1.0, 1.5, 2.0, 3.0, 4.0)
    k8 = [edge(x, v) for v in e8]
    lab_old = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")
    lab_new = ("<0.105", "0.105-0.233", "0.233-0.519", "0.519-1", "1-1.5", "1.5-2", "2-3", "3-4")
    rows_b8, b8_check = [], 0.0
    with open(RES / "tables" / "H3_bands.csv") as f:
        for r in csv.DictReader(f):
            q = lab_old.index(r["band_R200c"])
            M = S[f"imp__{r['selection']}__{r['observable']}"]
            m = M if M.ndim == 2 else M[..., li[int(r["ell"])]]
            val = float(m[:, k8[q]:k8[q + 1]].sum())
            b8_check = max(b8_check, abs(val - float(r["importance"])))
            rows_b8.append(dict(selection=r["selection"], observable=r["observable"], ell=r["ell"],
                                band_R200c=lab_new[q], band_lo=float(x[k8[q]]), band_hi=float(x[k8[q + 1]]),
                                label_in_results=r["band_R200c"], importance=val))
    out["bands8_relabel"] = {"edges_used": [float(x[k]) for k in k8], "label_in_results": lab_old,
                             "label_with_edges_used": lab_new, "recompute_vs_stored_max_abs": b8_check,
                             "table": "post/H_post_bands8.csv"}
    # coherent tolerance over a whole band (calibrated stacks): 0.01 / (k * importance of the band)
    coh = {}
    for name, e in (("dm_mean", None), ("ksz1h", 100), ("ksz1h", 1000), ("dm_q16", None), ("dm_q50", None),
                    ("yytot", 1000)):
        key = f"{name}@{e if e is not None else 'scalar'}"
        m = imp(name, e)
        k = K_OF.get(name, 1)
        i_all = float(m[:, bk[4]:bk[5]].sum())
        i_cal = float(m[calm][:, bk[4]:bk[5]].sum())
        coh[key] = {"importance_2-4R_all_bins": i_all, "importance_2-4R_calibrated_engine_bins": i_cal,
                    "coherent_2-4R_error_for_1pct_calibrated": 0.01 / (k * i_cal),
                    "coherent_2-4R_error_for_1pct_if_fallback_also_changed": 0.01 / (k * i_all)}
    out["coherent_band_tolerance"] = coh

    # ------------------------------------------------------------------ 5b. truncation radii and losses
    rad, rows_r = {}, []
    for sel in ("a2ii", "main"):
        for name, e in [("yytot", v) for v in SEL_ELL] + [("ymean", None)] + [("ksz1h", v) for v in SEL_ELL] + \
                       [("ksz2h", 100), ("ksztot", 100), ("dm_mean", None), ("dm_var", None), ("dmcltot", 1000),
                        ("ydmtot", 1000), ("ydmtot_Pcut", 1000), ("ydmtot_Ecut", 1000), ("yy1h", 1000)]:
            c = S[f"trunc_all__{sel}__{name}"]
            c = c if c.ndim == 1 else c[:, li[e]]
            st = RN.curve_stats(x, np.asarray(c, float))
            st["loss_total_4R"] = 0.0
            key = f"{sel}|{name}@{e if e is not None else 'scalar'}"
            rad[key] = st
            rows_r.append(dict(key=key, **st))
        q = S[f"trunc_all__{sel}__dmq_mean"]
        for p, qn in enumerate(QN):
            st = RN.curve_stats(x, np.asarray(q[:, p], float))
            key = f"{sel}|dm_{qn}@truthmean"
            rad[key] = st
            rows_r.append(dict(key=key, **st))
        qh, kk = S[f"trunc_halo__{sel}__dmq"], S[f"trunc_halo__{sel}__k"]
        for p, qn in enumerate(QN):
            rad[f"{sel}|dm_{qn}@perhalo"] = {f"loss_total_{x[k]:.4g}R": float(1 - qh[j, p] / qh[-1, p])
                                             for j, k in enumerate(kk[:-1])}
    out["radii"] = rad
    # C overshoot at high ell (1-halo interference)
    out["C_max"] = {f"{n}@{e}": rad[f"a2ii|{n}@{e}"]["C_max"] for n in ("yytot", "ksz1h") for e in (3000, 8000)}

    # ------------------------------------------------------------------ 5c. double counting per shell
    B = np.load(RES / "beyond.npz")
    bs = json.loads((RES / "beyond_summary.json").read_text())
    dc = {}
    for f in ("P", "E"):
        r = B[f"share_a2ii_over_all__{f}"]                      # [10 calibrated bins, 67 shells]
        for lo, hi in ((2.0, 3.0), (3.0, 4.0), (3.6342, 4.0)):
            msk = (x[:-1] >= lo - 1e-9) & (x[1:] <= hi + 1e-9)
            vw = list(bs["double_counting"][f"{f}_{lo:g}-{hi:g}R"].values())
            dc[f"{f}_{lo:g}-{hi:g}R"] = {"volume_weighted_range_over_bins": [min(vw), max(vw)],
                                         "per_shell_range_over_bins_and_shells": [float(r[:, msk].min()),
                                                                                  float(r[:, msk].max())],
                                         "neighbour_fraction_volume_weighted": [1 - max(vw), 1 - min(vw)]}
    out["double_counting"] = dc

    # ------------------------------------------------------------------ 5d. other report values at full precision
    bn = json.loads((RES / "binning_summary.json").read_text())["errors"]

    def fam_rms(e, keys, ref=None):
        d = [e[k] - (0.0 if ref is None else ref[k]) for k in keys if k in e and (ref is None or k in ref)]
        return (100 * float(np.sqrt(np.mean(np.square(d)))) if d else None), len(d)

    binrec, bin_check = {}, 0.0
    rdet = bn["M200c_0.2|piecewise|4|det"]["vs_truth_mean"]
    rs1 = bn["M200c_0.2|piecewise|4|S1"]["vs_truth"]
    for sc in ("M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5"):
        for mode in ("piecewise", "interp"):
            det, s1 = bn[f"{sc}|{mode}|4|det"], bn[f"{sc}|{mode}|4|S1"]
            rec = {"det_binning_only": {}, "S1_binning_only_dO": {}, "S1_total_vs_scatter_truth": {},
                   "n_terms": {}}
            for fam, keys in BIN_FAM.items():
                rec["det_binning_only"][fam], n = fam_rms(det["vs_truth_mean"], keys, rdet)
                rec["S1_binning_only_dO"][fam], n1 = fam_rms(s1["vs_truth"], keys, rs1)
                rec["S1_total_vs_scatter_truth"][fam], _ = fam_rms(s1["vs_truth"], keys)
                rec["n_terms"][fam] = {"det": n, "S1": n1}
            # the stored family RMS must equal these (det: exact reference; S1 total: same definition)
            for fam in ("tSZ", "kSZ", "DM", "tSZxDM"):
                bin_check = max(bin_check,
                                abs(rec["det_binning_only"][fam] - 100 * det["family_rms_vs_truth_mean"][fam]),
                                abs(rec["S1_total_vs_scatter_truth"][fam] - 100 * s1["family_rms_vs_truth"][fam]))
            rec["dm_quantiles_included"] = {"det": bool(det.get("dm_quantiles_included")),
                                            "S1": bool(s1.get("dm_quantiles_included"))}
            binrec[f"{sc}|{mode}"] = rec
    out["binning_family_rms_pct_Rcut4"] = {
        "definitions": "det_binning_only: RMS over the family of e_O(scheme, det) - e_O(M200c_0.2 piecewise, det), "
                       "e_O = O_painted / O_truthmean - 1 (reference exact); S1_binning_only_dO: RMS of "
                       "e_O(scheme, S1) - e_O(M200c_0.2 piecewise, S1), e_O = O_painted / O_truth - 1 (scatter-inclusive "
                       "truth); S1_total_vs_scatter_truth: RMS of e_O(scheme, S1) itself (binning + S1-model error). "
                       "Families: tSZ C^yy, kSZ 1h, tSZxDM total at ell 100/300/1000/3000; DM = mean, variance, "
                       "Q16/Q50/Q84/Q97.5/Q99.9; DM_moments = mean, variance. Values in %.",
        "values": binrec,
        "check_vs_stored_family_rms_max_abs_pct": bin_check}
    # diagonal (own-shell) fraction of the calibrated haloes' 1-halo terms, both normalisations
    sf = {}
    for name in ("yy1h", "ksz1h", "dmcl1h"):
        st = ss["self_fraction"][f"a2ii__{name}"]
        sf[name] = {e: {"over_O_cal_1h": float(st[str(e)]),
                        "over_O_1h": float(st[str(e)]) * tv(name, e, "truthcal") / tv(name, e),
                        "O_cal_1h_over_O_1h": tv(name, e, "truthcal") / tv(name, e)} for e in SEL_ELL}
    out["self_fraction_a2ii"] = sf
    dg = json.loads((RES / "tables" / "H_digest.json").read_text())
    out["coherent_beta_sensitivity_yy"] = {e: dg["H6_coherent_sensitivity"][f"Rf4|P|yytot@{e}"]["p3"]
                                           for e in (100, 300, 1000, 3000)}
    out["top_bin_shares_yytot@100"] = {labels[b]: float(imp("yytot", 100)[b].sum()) for b in cal[-3:]}
    sh_top = shares["yytot@100"]
    out["top_bin_15.3-15.5_split_yytot@100"] = {
        "engine_bin_share": sh_top["bin_15.3-15.5_total"],
        "lightcone_haloes_10^15.3-10^15.388_painted_with_the_FLAMINGO_stack": sh_top["bin_15.3-15.5_calibrated_haloes"],
        "lightcone_haloes_above_10^15.388_painted_with_the_fallback": sh_top["fallback_inside_calibrated_bins"],
        "note": "the shares belong to the HalfDome lightcone haloes of the engine bin; the stack that paints the "
                "calibrated part is built from the FLAMINGO haloes counted in halo_counts_calibrated_bins"}
    out["halo_counts_calibrated_bins"] = {labels[b]: int((h0["mass_bin"] == b).sum()) for b in cal}
    out["halo_counts_note"] = "FLAMINGO L1_m9 haloes in each calibrated stack (not lightcone haloes)"
    out["tolerance_stored_sens_summary_yytot@1000"] = ss["tolerance"]["a2ii__yytot@1000"]
    out["sum_rules"] = {"moments_max_abs": max(v for k, v in ss["sum_rules"].items()
                                               if not isinstance(v, dict) and "dm_q" not in k),
                        "quantile_subsample_sums": {qn: float(S[f"imp__a2ii__dm_{qn}"].sum()) for qn in QN},
                        "quantile_truthmean_max_abs": max(v for k, v in ss["sum_rules"].items()
                                                          if not isinstance(v, dict) and k.startswith("impdet__")
                                                          and "dm_q" in k)}
    (OUT / "h_post.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for fname, rows in (("H_post_fine_vs_F_truth.csv", rows_f), ("H_post_calibrated_shares.csv", rows_s),
                        ("H_post_tolerance_shells.csv", rows_t), ("H_post_bands5.csv", rows_b),
                        ("H_post_bands8.csv", rows_b8),
                        ("H_post_radii.csv", rows_r)):
        cols = []
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for c, v in r.items()})
    print("quantile map reconstruction", qcheck, "| linear own-vs-stored", lin_check)
    print("fine vs F max:", json.dumps({s: fv[s]["_max"] for s in fv}, default=float))


if __name__ == "__main__":
    main()
