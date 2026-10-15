"""H stages 'basis' (H1) and 'truth' (H2, with the parity gates against Test F).

basis: f_kernels machinery on the fine grid for three reference sets (h_basis): FP (Test F's shapes and all-gas
       coefficients: parity), HA (A2-ii fine shapes and means: the painting target), HM (all-gas fine shapes and
       means: the comparison).
truth: per selection s in {a2ii, main}, in its own basis (HA, HM):
  TRUTH (scatter-inclusive): every calibrated bin enters with its haloes' inclusion-weighted second moments
      <F_s F_t> (1-halo terms, DM variance) and means (linear and 2-halo terms); the DM PDF deposits every halo's own
      column.  TRUTH MEAN: the bin-mean profile alone (one profile per bin).  Both to 4 R200c (nothing beyond).
  Mecke DM-quantile Jacobians dQ/dln F[c, s] of the per-halo truth (Test F's recipe: <= 64 haloes per bin drawn with
      default_rng(f_common.SEED + 5), weights rescaled) and of the truth mean, with the fallback haloes' pieces.
  Cross-basis checks: each selection also evaluated in the other basis and in FP (within-shell shape sensitivity),
      and the fine truth against Test F's 35-shell truth (sub-shell resolution effect).
Parity (FP): Test F's per-halo shell averages mapped onto the fine shells through F's own within-shell shapes must
  reproduce F's truth.npz / truth_summary.json (moment observables, DM quantiles) and importance_F.npz (imp__* maps,
  fine Jacobians summed into F's shells), to rounding.
Outputs: results/truth.npz, results/truth_summary.json, results/parity.json, results/work/pdf_fallback_h.npz.
"""
from __future__ import annotations

import h_basis as HB
import h_common as HC
import h_pdf as HP
import numpy as np

FC, FB = HC.FC, HC.FB
import f1_importance as FI  # noqa: E402  (Test F, read-only: full_map, fallback_J, OBS_ROWS)

BASIS_OF = {"a2ii": "HA", "main": "HM"}


# ============================================================================ helpers
def bin_rows(pr, c):
    return np.flatnonzero(pr["mass_bin"] == FC.CAL_BINS[c])


def cal_moments(pr, sel):
    cal = FC.CAL_BINS
    return dict(P=pr[f"meanP_{sel}"][cal], E=pr[f"meanE_{sel}"][cal], SPP=pr[f"SPP_{sel}"][cal],
                SEE=pr[f"SEE_{sel}"][cal], SPE=pr[f"SPE_{sel}"][cal])


def subsample(pr):
    """Test F's Mecke subsample (f1_truth.main (b)): rows and rescaled weights per calibrated bin."""
    rng = np.random.default_rng(FC.SEED + 5)
    rows_out, w_out = [], []
    for c in range(FC.N_CAL):
        rows = bin_rows(pr, c)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        pick = rng.choice(len(rows), size=min(FC.MECKE_SUBSAMPLE, len(rows)), replace=False)
        rows_out.append(rows[pick])
        w_out.append(w[pick] * (w.sum() / w[pick].sum()))
    return rows_out, w_out


def shapes(mean, x, edges):
    return HB.truth_shapes(mean, x, edges)


def specs_for_basis(pr, fpr):
    """Reference shapes and coefficients of the three basis sets."""
    _, gridobj, seg_shell, _ = HB.grid()
    cal = FC.CAL_BINS
    x = gridobj.x
    mP35, mE35 = fpr["meanP_main"][cal], fpr["meanE_main"][cal]
    gP, gE = shapes(mP35, x, HC.F35), shapes(mE35, x, HC.F35)
    rP, _, _ = HB.fp_ratios(gP, gridobj, seg_shell)
    rE, _, _ = HB.fp_ratios(gE, gridobj, seg_shell)
    pad42 = lambda a: np.pad(a, ((0, 0), (0, HC.N_B42 - a.shape[1])))                    # noqa: E731
    specs = {"FP": dict(gP=gP, gE=gE, coefP=pad42(mP35)[:, HC.FINE_OF_B42] * rP,
                        coefE=pad42(mE35)[:, HC.FINE_OF_B42] * rE)}
    for name, sel in (("HA", "a2ii"), ("HM", "main")):
        mP, mE = pr[f"meanP_{sel}"][cal], pr[f"meanE_{sel}"][cal]
        specs[name] = dict(gP=shapes(mP, x, HC.X200), gE=shapes(mE, x, HC.X200), coefP=HP.pad(mP), coefE=HP.pad(mE))
    return specs, (rP, rE)


def basis_stage():
    pr = HC.load_npz(HC.FC.PRODUCTS / "profiles.npz")
    fpr = HC.load_npz(HC.F_PROFILES)
    specs, _ = specs_for_basis(pr, fpr)
    timing = HB.build(specs)
    _, gridobj, _, _ = HB.grid()
    timing["truth_shapes_identity_vs_f_kernels"] = max(
        HB.check_truth_shapes(fpr["meanP_main"][FC.CAL_BINS], gridobj.x),
        HB.check_truth_shapes(fpr["meanE_main"][FC.CAL_BINS], gridobj.x))
    HC.write_json(HC.RESULTS / "basis.json", timing)
    return timing


def f_implied(fpr, rP, rE, sel):
    """Test F's per-bin moments of selection ``sel`` on the fine shells through F's within-shell shapes."""
    cal = FC.CAL_BINS
    m = FINE_MAP
    pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, HC.N_B42 - a.shape[-1])])      # noqa: E731
    pad2 = lambda a: np.pad(a, [(0, 0), (0, HC.N_B42 - a.shape[1]), (0, HC.N_B42 - a.shape[2])])  # noqa: E731
    P = pad(fpr[f"meanP_{sel}"][cal])[:, m] * rP
    E = pad(fpr[f"meanE_{sel}"][cal])[:, m] * rE
    SPP = pad2(fpr[f"SPP_{sel}"][cal])[:, m][:, :, m] * rP[:, :, None] * rP[:, None, :]
    SEE = pad2(fpr[f"SEE_{sel}"][cal])[:, m][:, :, m] * rE[:, :, None] * rE[:, None, :]
    SPE = pad2(fpr[f"SPE_{sel}"][cal])[:, m][:, :, m] * rP[:, :, None] * rE[:, None, :]
    return dict(P=P, E=E, SPP=SPP, SEE=SEE, SPE=SPE)


FINE_MAP = HC.FINE_OF_B42


def f_implied_halo(fpr, rE, key, rows, c):
    """Per-halo fine electron rows of F's per-halo profiles (F grid) for bin c."""
    v = np.pad(np.nan_to_num(fpr[key][rows], nan=0.0), ((0, 0), (0, HC.N_B42 - HC.N_F35)))
    return v[:, FINE_MAP] * rE[c][None, :]


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(a / b - 1)
    r = r[np.isfinite(r) & (b != 0)]
    return float(r.max()) if r.size else 0.0


def imp_maps(B, J, truth):
    """Normalised importance maps imp[name] = (dO/dln F) / (k O) [16, 67(, ell)] (f1_importance.OBS_ROWS)."""
    out = {}
    for name, jkey, tkey, fam, text in FI.OBS_ROWS:
        M = FI.full_map(B, J, jkey)
        k = 2 if jkey in FI.QUAD else 1
        O = truth[tkey]
        out[name] = M / (k * (O if np.ndim(O) == 0 else O[None, None, :]))
    return out


# ============================================================================ truth stage
def main():
    timer = HC.Timer()
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    fpr = HC.load_npz(HC.F_PROFILES)
    ftr = HC.load_npz(HC.F_TRUTH)
    fts = HC.read_json(HC.F_TRUTH_SUMMARY)
    fimp = HC.load_npz(HC.F_IMPORTANCE)
    specs, (rP, rE) = specs_for_basis(pr, fpr)
    del specs
    out, summary, parity = {}, {"observables": {}, "pdf": {}}, {}
    parity["deposit_identity_max_abs"] = HP.identity_check()
    # memory: one fine basis (0.74 GB in full mode) in memory at a time, none at the DM-PDF forks below
    HC.mem("truth start")
    # ---------------------------------------------------------------- parity: moment observables
    BF = FB.Basis("FP")
    worst = {}
    for sel in ("main", "a2ii"):
        mom = f_implied(fpr, rP, rE, sel)
        for tag, o in (("truth", BF.observables(**mom)), ("truthmean", BF.observables(mom["P"], mom["E"]))):
            for k, v in o.items():
                ref = ftr.get(f"{tag}_{sel}__{k}")
                if ref is not None:
                    worst[f"{tag}_{sel}__{k}"] = rel(v, ref)
    parity["moments_max_rel"] = max(worst.values())
    parity["moments_by_key"] = worst
    HC.log("parity moments", parity["moments_max_rel"])
    # ---------------------------------------------------------------- parity: importance maps (moment observables)
    momF = f_implied(fpr, rP, rE, "main")
    JF = BF.jacobians(**momF)
    truthF = {k.split("__")[1]: ftr[k] for k in ftr if k.startswith("truth_main__")}
    impF = imp_maps(BF, JF, truthF)
    del JF
    worst_imp = {}
    for name, M in impF.items():
        coarse = HC.sum_to_f35(M, axis=1)
        ref = fimp[f"imp__{name}"]
        scale = np.max(np.abs(ref))
        worst_imp[name] = float(np.max(np.abs(coarse - ref)) / scale)
    del impF
    parity["importance_max_abs_over_max"] = max(worst_imp.values())
    parity["importance_by_observable"] = worst_imp
    HC.log("parity importance", parity["importance_max_abs_over_max"])
    # within-shell shape sensitivity (each selection's moments in every basis): observables per (selection, basis)
    moms = {sel: cal_moments(pr, sel) for sel in HC.SELECTIONS}
    cross = {(sel, "FP"): BF.observables(**moms[sel]) for sel in HC.SELECTIONS}
    del BF
    HC.mem("after FP basis")
    # ---------------------------------------------------------------- production moment truths (one basis at a time)
    full_by_sel = {}
    for name in ("HA", "HM"):
        B = FB.Basis(name)
        for sel in HC.SELECTIONS:
            if BASIS_OF[sel] != name:
                cross[(sel, name)] = B.observables(**moms[sel])
                continue
            mom = moms[sel]
            full = B.observables(**mom)
            mean = B.observables(mom["P"], mom["E"])
            calonly = B.observables(**mom, cal_only=True)
            full_by_sel[sel] = full
            for tag, o in (("truth", full), ("truthmean", mean), ("truthcal", calonly)):
                for k, v in o.items():
                    out[f"{tag}_{sel}__{k}"] = np.asarray(v)
                summary["observables"][f"{tag}_{sel}"] = B.scalars(o)
            bud = B.budgets(mom["P"], mom["E"])
            out[f"Y4_{sel}"], out[f"Ne4_{sel}"] = bud["Y"], bud["Ne"]
        del B
        HC.mem(f"after {name} basis")
    for sel in HC.SELECTIONS:
        full = full_by_sel[sel]
        for other in ("FP", "HA", "HM"):
            if other == BASIS_OF[sel]:
                continue
            o2 = cross[(sel, other)]
            summary.setdefault("basis_shape_sensitivity", {})[f"{sel}_in_{other}"] = {
                k: rel(o2[k], full[k]) for k in full}
        # fine truth vs Test F's 35-shell truth (sub-shell resolution of the profile)
        summary.setdefault("fine_vs_F_truth", {})[sel] = {
            k: float(np.max(np.asarray(full[k]) / np.asarray(ftr[f"truth_{sel}__{k}"]) - 1))
            if np.ndim(full[k]) == 0 else
            {str(e): float(full[k][i] / ftr[f"truth_{sel}__{k}"][i] - 1) for e, i in zip(FC.SEL_ELL, FC.SEL_INDEX)}
            for k in full if f"truth_{sel}__{k}" in ftr}
    del cross, full_by_sel
    HC.log("moments", timer.mark("moments"))
    # ---------------------------------------------------------------- DM PDF pass 1 (FP parity, then HA / HM)
    # FP: per-halo F-implied main truth + truth mean (fallback slot deposited here; same for every basis)
    per_halo_F = []
    rows_bin = [bin_rows(pr, c) for c in range(FC.N_CAL)]
    w_bin = [pr["weight"][r] / pr["weight"][r].sum() for r in rows_bin]
    for c in range(FC.N_CAL):
        per_halo_F.append((f_implied_halo(fpr, rE, "E_main", rows_bin[c], c), w_bin[c]))
    specsF = [("F_truth_main", per_halo_F), ("F_truthmean_main", HP.det_entries(momF["E"]))]
    resF, fallback, tF = HP.run_specs(specsF, "FP", keep_density=True, label="pdf FP")
    np.savez_compressed(HC.WORK / "pdf_fallback_h.npz", hist=fallback[0], overflow=fallback[1])
    parity["pdf"] = {}
    for lab, fl in (("F_truth_main", "truth_main"), ("F_truthmean_main", "truthmean_main")):
        q = np.asarray(resF[lab]["quantiles"], float)
        ref = np.asarray(fts["pdf"][fl]["quantiles"], float)
        parity["pdf"][fl] = {"H_FP": q, "F": ref, "max_rel": rel(q, ref)}
    parity["pdf_max_rel"] = max(v["max_rel"] for v in parity["pdf"].values())
    HC.log("parity pdf", parity["pdf_max_rel"])
    pdf_ref = {}
    for sel in HC.SELECTIONS:
        which = BASIS_OF[sel]
        mom = cal_moments(pr, sel)
        specs_s = [(f"truth_{sel}", HP.halo_entries(pr, f"E_{sel}", rows_bin, w_bin)),
                   (f"truthmean_{sel}", HP.det_entries(mom["E"]))]
        res, _, tim = HP.run_specs(specs_s, which, fallback=fallback, keep_density=True, label=f"pdf {sel}")
        for lab, s in res.items():
            pdf_ref[lab] = s
            summary["pdf"][lab] = {k: s[k] for k in ("quantiles", "mean", "variance", "mass_on_grid", "negative_mass")}
            out[f"pdf_{lab}__quantiles"] = s["quantiles"]
            out[f"pdf_{lab}__density"] = s["density"]
            out[f"pdf_{lab}__mean"] = s["mean"]
            out[f"pdf_{lab}__variance"] = s["variance"]
        summary.setdefault("pdf_timing", {})[sel] = tim
        for tag in ("truth", "truthmean"):
            s = summary["pdf"][f"{tag}_{sel}"]
            mo = summary["observables"][f"{tag}_{sel}"]
            summary.setdefault("pdf_vs_moments", {})[f"{tag}_{sel}"] = {
                "mean": s["mean"] / mo["dm_mean"] - 1, "variance": s["variance"] / mo["dm_var"] - 1}
    HC.log("pdf pass 1", timer.mark("pdf1"))
    # ---------------------------------------------------------------- Mecke quantile Jacobians
    sub_rows, sub_w = subsample(pr)
    itemsF = [("F_truth_main_subsample", resF["F_truth_main"],
               [(f_implied_halo(fpr, rE, "E_main", sub_rows[c], c), sub_w[c]) for c in range(FC.N_CAL)], True),
              ("F_truthmean_main", resF["F_truthmean_main"], HP.det_entries(momF["E"]), True)]
    mF, _ = HP.run_mecke(itemsF, "FP", fallback, label="mecke FP")
    worst_q = {}
    for lab, fl in (("F_truth_main_subsample", "truth_main_subsample"), ("F_truthmean_main", "truthmean_main")):
        J, Jfb = mF[lab]
        Jc = np.add.reduceat(J, HC.IDX42[:-1], axis=-1)                       # fine -> F basis shells
        Jfbc = HC.sum_to_f35(Jfb, axis=-1)
        worst_q[fl] = {"J_max_abs_over_max": float(np.max(np.abs(Jc - ftr[f"Jq_{fl}"])) / np.max(np.abs(ftr[f"Jq_{fl}"]))),
                       "Jfb_max_abs_over_max": float(np.max(np.abs(Jfbc - ftr[f"Jqfb_{fl}"])) / np.max(np.abs(ftr[f"Jqfb_{fl}"])))}
    # the normalised quantile importance in Test F's shells
    Jq, Jqfb = mF["F_truth_main_subsample"]
    Qt = np.asarray(fts["pdf"]["truth_main"]["quantiles"], float)
    wq = {}
    for p, qn in enumerate(FC.Q_NAMES):
        M = np.array(Jqfb[p], float, copy=True)
        for c, b in enumerate(FC.CAL_BINS):
            M[b] += Jq[p, c, :HC.N_X]
        coarse = HC.sum_to_f35(M / Qt[p], axis=1)
        ref = fimp[f"imp__dm_{qn}"]
        wq[qn] = float(np.max(np.abs(coarse - ref)) / np.max(np.abs(ref)))
    parity["mecke"] = worst_q
    parity["importance_quantiles_max_abs_over_max"] = max(wq.values())
    parity["importance_quantiles_by_q"] = wq
    HC.log("parity mecke", worst_q, wq)
    for sel in HC.SELECTIONS:
        which = BASIS_OF[sel]
        mom = cal_moments(pr, sel)
        items = [(f"truth_{sel}_subsample", pdf_ref[f"truth_{sel}"],
                  [(np.nan_to_num(pr[f"E_{sel}"][sub_rows[c]], nan=0.0), sub_w[c]) for c in range(FC.N_CAL)], True),
                 (f"truthmean_{sel}", pdf_ref[f"truthmean_{sel}"], HP.det_entries(mom["E"]), True)]
        m, tim = HP.run_mecke(items, which, fallback, label=f"mecke {sel}")
        for lab, (J, Jfb) in m.items():
            out[f"Jq_{lab}"] = J
            out[f"Jqfb_{lab}"] = Jfb
            Q = np.asarray(pdf_ref[lab.replace("_subsample", "")]["quantiles"], float)
            summary.setdefault("mecke_sum_rule", {})[lab] = ((J.sum(axis=(1, 2)) + Jfb.sum(axis=(1, 2))) / Q - 1).tolist()
        summary.setdefault("mecke_timing", {})[sel] = tim
    out["subsample_rows"] = np.concatenate(sub_rows)
    HC.log("mecke", timer.mark("mecke"))
    parity["passed"] = bool(parity["deposit_identity_max_abs"] == 0.0 and parity["moments_max_rel"] < 1e-8
                            and parity["importance_max_abs_over_max"] < 1e-8 and parity["pdf_max_rel"] < 1e-6
                            and parity["importance_quantiles_max_abs_over_max"] < 1e-6)
    parity["seconds"] = timer.mark("total")
    np.savez_compressed(HC.RESULTS / "truth.npz", **out)
    summary["seconds"] = timer.mark("total")
    HC.write_json(HC.RESULTS / "truth_summary.json", summary)
    HC.write_json(HC.RESULTS / "parity.json", parity)
    HC.log("truth stage done; parity passed:", parity["passed"])
    return parity
