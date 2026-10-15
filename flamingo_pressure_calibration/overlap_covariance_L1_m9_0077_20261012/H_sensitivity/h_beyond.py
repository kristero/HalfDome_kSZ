"""H4: "until what distance" - what painting beyond 4 R200c would add, under explicit (model-dependent) assumptions.

The FLAMINGO truth is measured to 4 R200c only.  Beyond, the A2-ii (and, for comparison, the all-gas) profile of every
calibrated bin is extended by
  (i)   zero                         the current painting (hard cut at 4 R200c) = the truth;
  (ii)  B12 / B16 extrapolation      the objective-(a) fit to the bin's stack out to R_fit = 4 R200c (h_fits), evaluated
                                     beyond 4 R200c without a cut;
  (iii) log-linear extrapolation     a power law A x^n fitted to the stacked shells at 3-4 R200c (iii-a) or 2.5-4 R200c
                                     (iii-b), continued as exact shell averages;
painted out to R_ext = 4.5, 5, 6, 7, 8, 8.5, 10, 12 R200c (basis extension edges).  Each halo of the bin receives the
same extension (the bin mean), so <F_s F_t> beyond 4R = m_s e_t (+ e_s e_t), with e the extension.
Moment observables: exact in-basis; DM quantiles: DM-PDF passes of the truth mean + extension (deterministic, all
R_ext) and of the per-halo truth + extension at 12 R200c (models ii and iii-a; checks the deterministic shortcut).
Double counting: A2-ii's share of all gas, m_A2ii / m_all, in the shells 2-4 R200c per bin (the rest belongs to
neighbouring painted haloes and would be painted twice by an all-gas extension).
Outputs: results/beyond.npz, results/beyond_summary.json, tables/H4_beyond.csv.
"""
from __future__ import annotations

import h_common as HC
import h_fits as HF
import h_pdf as HP
import h_truth as HT
import numpy as np

FC, FB = HC.FC, HC.FB
R_EXT = tuple(float(e) for e in HC.EXT_H)
REPORT_EXT = (5.0, 6.0, 8.0, 12.0)
KEYS = ("yytot", "yy1h", "yy2h", "ymean", "ksz1h", "ksz2h", "dm_mean", "dm_var", "dmcltot", "ydmtot")


def extend_moments(mom, extP, extE, k_ext):
    """Moments with the first ``k_ext`` extension shells (beyond 4R) filled by the bin-mean extension."""
    ns, nx = HC.N_FINE, HC.N_X
    keep = np.zeros(ns, bool)
    keep[nx:nx + k_ext] = True
    eP, eE = extP * keep, extE * keep
    P = HP.pad(mom["P"]) + eP
    E = HP.pad(mom["E"]) + eE
    pad2 = lambda S: np.pad(S, ((0, 0), (0, ns - S.shape[1]), (0, ns - S.shape[2])))      # noqa: E731
    mP, mE = HP.pad(mom["P"]), HP.pad(mom["E"])
    o = lambda a, b: np.einsum("cs,ct->cst", a, b)                                         # noqa: E731
    SPP = pad2(mom["SPP"]) + o(mP, eP) + o(eP, mP) + o(eP, eP)
    SEE = pad2(mom["SEE"]) + o(mE, eE) + o(eE, mE) + o(eE, eE)
    SPE = pad2(mom["SPE"]) + o(mP, eE) + o(eP, mE) + o(eP, eE)
    return dict(P=P, E=E, SPP=SPP, SEE=SEE, SPE=SPE)


def main():
    timer = HC.Timer()
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    tr = HC.load_npz(HC.RESULTS / "truth.npz")
    ts = HC.read_json(HC.RESULTS / "truth_summary.json")
    fallback = np.load(HC.WORK / "pdf_fallback_h.npz")
    fallback = (fallback["hist"], float(fallback["overflow"]))
    out = {"r_ext": np.array(R_EXT), "fine_edges": HC.FINE}
    summary = {"definitions": __doc__, "models": {}, "changes": {}, "double_counting": {}, "fits": {}}
    rows = []
    li = [int(np.flatnonzero(FC.ELL == e)[0]) for e in HC.SEL_ELL]
    cal = FC.CAL_BINS
    for sel in HC.SELECTIONS:
        B = FB.Basis(HT.BASIS_OF[sel])
        mom = HT.cal_moments(pr, sel)
        base = B.observables(**mom)
        # ---- extension models
        exts = {}
        fitP, fitE = [], []
        for c in range(FC.N_CAL):
            fP = HF.fit(HF.bin_data(pr, "P", sel, c), 4.0)
            fE = HF.fit(HF.bin_data(pr, "E", sel, c), 4.0)
            fitP.append(fP)
            fitE.append(fE)
        extP = np.array([HF.model_shells("P", f["parameters"], cut=None) for f in fitP])
        extE = np.array([HF.model_shells("E", f["parameters"], cut=None) for f in fitE])
        extP[:, :HC.N_X] = 0.0
        extE[:, :HC.N_X] = 0.0
        exts["ii_B12B16_fit4R"] = (extP, extE)
        summary["fits"][sel] = {"P": [f["parameters"] for f in fitP], "E": [f["parameters"] for f in fitE],
                                "P_chi2_per_dof": [f["chi2_per_dof"] for f in fitP],
                                "E_chi2_per_dof": [f["chi2_per_dof"] for f in fitE]}
        for lab, x_from in (("iiia_loglin_3to4R", 3.0), ("iiib_loglin_2.5to4R", 2.5)):
            eP, sP, aP = HF.loglinear_extension(mom["P"], x_from)
            eE, sE, aE = HF.loglinear_extension(mom["E"], x_from)
            exts[lab] = (eP, eE)
            summary["models"].setdefault(sel, {})[lab] = {"slope_P": sP, "slope_E": sE}
        for lab, (eP, eE) in exts.items():
            out[f"ext__{sel}__{lab}__P"] = eP
            out[f"ext__{sel}__{lab}__E"] = eE
            # ratio of the extension in the first extension shell to the stack's last shell (continuity)
            summary["models"].setdefault(sel, {}).setdefault(lab, {})["first_ext_over_last_shell_P"] = \
                (eP[:, HC.N_X] / mom["P"][:, HC.N_X - 1]).tolist()
            summary["models"][sel][lab]["first_ext_over_last_shell_E"] = (eE[:, HC.N_X] / mom["E"][:, HC.N_X - 1]).tolist()
            # Y and N_e added beyond 4R (z = 0, per bin) relative to Y(<4R), N_e(<4R)
            vol = HC.VOL_FINE
            Y4 = mom["P"] @ vol[:HC.N_X]
            N4 = mom["E"] @ vol[:HC.N_X]
            for j, r in enumerate(R_EXT):
                k = j + 1
                eP_k = eP[:, HC.N_X:HC.N_X + k] @ vol[HC.N_X:HC.N_X + k]
                eE_k = eE[:, HC.N_X:HC.N_X + k] @ vol[HC.N_X:HC.N_X + k]
                out[f"budget__{sel}__{lab}__Y_add_R{r:g}"] = eP_k / Y4
                out[f"budget__{sel}__{lab}__Ne_add_R{r:g}"] = eE_k / N4
                o = B.observables(**extend_moments(mom, eP, eE, k))
                ch = {}
                for key in KEYS:
                    v = np.asarray(o[key]) / np.asarray(base[key]) - 1
                    out[f"chg__{sel}__{lab}__{key}__R{r:g}"] = v
                    if np.ndim(v):
                        for e, i in zip(HC.SEL_ELL, li):
                            ch[f"{key}_{e}"] = float(v[i])
                    else:
                        ch[key] = float(v)
                summary["changes"].setdefault(sel, {}).setdefault(lab, {})[f"R{r:g}"] = ch
                rows.append(dict(selection=sel, model=lab, R_ext=r, quantity="moments", **ch,
                                 Y_added_median=float(np.median(eP_k / Y4)), Ne_added_median=float(np.median(eE_k / N4))))
        # ---- double counting: A2-ii share of all gas at 2-4R (from the stacks)
        if sel == "a2ii":
            mA = {f: pr[f"mean{f}_a2ii"][cal] for f in ("P", "E")}
            mM = {f: pr[f"mean{f}_main"][cal] for f in ("P", "E")}
            for f in ("P", "E"):
                ratio = mA[f] / np.where(mM[f] > 0, mM[f], np.nan)
                out[f"share_a2ii_over_all__{f}"] = ratio
                vol = HC.VOL_FINE[:HC.N_X]
                for lo, hi in ((2.0, 3.0), (3.0, 4.0), (3.6342, 4.0)):
                    m = (HC.X200[:-1] >= lo - 1e-9) & (HC.X200[1:] <= hi + 1e-9)
                    share = (mA[f][:, m] @ vol[m]) / (mM[f][:, m] @ vol[m])
                    summary["double_counting"][f"{f}_{lo:g}-{hi:g}R"] = dict(zip(HC.CAL_LABELS, share.tolist()))
        HC.log(f"beyond moments {sel}", timer.mark(f"mom_{sel}"))
        del B, base                            # no basis in memory at the DM-PDF fork below
        HC.mem(f"beyond after moments {sel}")
    # ---- DM quantiles (A2-ii): truth mean + extension (all R_ext), per-halo + extension (12R; ii and iii-a)
    sel = "a2ii"
    mom = HT.cal_moments(pr, sel)
    rows_bin = [HT.bin_rows(pr, c) for c in range(FC.N_CAL)]
    w_bin = [pr["weight"][r] / pr["weight"][r].sum() for r in rows_bin]
    specs = []
    for lab in ("ii_B12B16_fit4R", "iiia_loglin_3to4R", "iiib_loglin_2.5to4R"):
        eE = out[f"ext__{sel}__{lab}__E"]
        for j, r in enumerate(R_EXT):
            keep = np.zeros(HC.N_FINE, bool)
            keep[HC.N_X:HC.N_X + j + 1] = True
            specs.append((f"mean|{lab}|{r:g}", HP.det_entries(HP.pad(mom["E"]) + eE * keep)))
    for lab in ("ii_B12B16_fit4R", "iiia_loglin_3to4R"):
        eE = out[f"ext__{sel}__{lab}__E"]
        per = []
        for c in range(FC.N_CAL):
            v = HP.pad(np.nan_to_num(pr[f"E_{sel}"][rows_bin[c]], nan=0.0)) + eE[c][None, :]
            per.append((v, w_bin[c]))
        specs.append((f"halo|{lab}|12", per))
    res, _, tim = HP.run_specs(specs, HT.BASIS_OF[sel], fallback=fallback, label="pdf beyond")
    qmean0 = np.asarray(ts["pdf"][f"truthmean_{sel}"]["quantiles"], float)
    qhalo0 = np.asarray(ts["pdf"][f"truth_{sel}"]["quantiles"], float)
    for lab_full, s in res.items():
        kind, lab, r = lab_full.split("|")
        ref = qmean0 if kind == "mean" else qhalo0
        q = np.asarray(s["quantiles"], float)
        out[f"dmq__{kind}__{lab}__R{r}"] = q
        summary["changes"].setdefault("dm_quantiles", {}).setdefault(kind, {}).setdefault(lab, {})[f"R{r}"] = \
            dict(zip([f"dm_{qn}" for qn in HC.Q_NAMES], (q / ref - 1).tolist()))
    summary["pdf_timing"] = tim
    np.savez_compressed(HC.RESULTS / "beyond.npz", **out)
    summary["seconds"] = timer.mark("total")
    HC.write_json(HC.RESULTS / "beyond_summary.json", summary)
    HC.write_csv(HC.TABLES / "H4_beyond.csv", rows)
    HC.log("beyond done", summary["seconds"])
    return summary


if __name__ == "__main__":
    main()
