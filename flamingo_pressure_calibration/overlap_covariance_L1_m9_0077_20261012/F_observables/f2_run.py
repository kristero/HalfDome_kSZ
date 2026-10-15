"""F2-F4 on the full sample: fit every method, evaluate every model against the truth.

For each model (basis-shell averages per calibrated bin, pressure and electrons):
  * deterministic predictions (one profile per bin, as painted today) of every observable, in-basis;
  * scatter-aware predictions: lognormal amplitude scatter per bin (sigma from per-halo Y / N_e(<1R) and
    (<4R) moments: 1h, DM variance and y x DM 1h scaled by <A^2>, <A_P A_E>; DM PDF by a 9-node
    Gauss-Hermite amplitude mixture) and the measured second-moment correction (1h only: model x the
    bin's measured <F_s F_t>/(m_s m_t));
  * DM PDF quantiles from one f_pdf pass over all model variants (fallback slot from the truth pass);
  * profile residuals vs the measured mean on resolved shells: region RMS and importance-weighted RMS.
Errors: Delta O / O against the scatter-inclusive TRUTH, and against the deterministic truth mean
(isolates the profile / fit / cut effect).  Controls (agn_kept, a2i, a2ii): the objective-(a) grid and the
stacks refitted on, and scored against, their own truths.
Outputs: products/models_main.npz, products/fit_records.csv (tables/), tables/F_errors_long.csv,
tables/F_model_scores.csv, products/run_summary.json.
"""
from __future__ import annotations

import f_basis as FB
import f_common as FC
import f_fit as FF
import f_methods as FM
import f_pdf as FP
import numpy as np

SCALAR_KEYS = None
SCATTER_PDF_MODELS = ("grid_a_Rf2_Rc4", "grid_a_Rf4_Rc4", "tab_stack", "flex_env_a", "grid_c_Rf2_Rc4")


def truth_dict(pr, tr, ts, sel):
    cal = FC.CAL_BINS
    obs = dict(ts["observables"][f"truth_{sel}"])
    for p, q in enumerate(FC.Q_NAMES):
        obs[f"dm_{q}"] = float(ts["pdf"][f"truth_{sel}"]["quantiles"][p])
    mean_obs = dict(ts["observables"][f"truthmean_{sel}"])
    for p, q in enumerate(FC.Q_NAMES):
        mean_obs[f"dm_{q}"] = float(ts["pdf"][f"truthmean_{sel}"]["quantiles"][p])
    return {"P": pr[f"meanP_{sel}"][cal], "E": pr[f"meanE_{sel}"][cal], "SPP": pr[f"SPP_{sel}"][cal],
            "SEE": pr[f"SEE_{sel}"][cal], "SPE": pr[f"SPE_{sel}"][cal], "obs": obs, "mean_obs": mean_obs,
            "Qdet": np.asarray(ts["pdf"][f"truthmean_{sel}"]["quantiles"], float), "Jq_det": tr["Jq_truthmean_main"]}


def scatter_moments(P, E, kind, mom=None, truth=None, B=None):
    """Second moments of a scatter-aware model: lognormal amplitudes or the measured correction."""
    if kind == "lognormal":
        a = np.exp(mom[:, 0])[:, None, None]
        b = np.exp(mom[:, 1])[:, None, None]
        x = mom[:, 2][:, None, None]
        return (a * np.einsum("cs,ct->cst", P, P), b * np.einsum("cs,ct->cst", E, E), x * np.einsum("cs,ct->cst", P, E))
    out = []
    for S, m1, m2, M1, M2 in ((truth["SPP"], truth["P"], truth["P"], P, P), (truth["SEE"], truth["E"], truth["E"], E, E),
                              (truth["SPE"], truth["P"], truth["E"], P, E)):
        den = np.einsum("cs,ct->cst", m1, m2)
        ratio = np.ones((FC.N_CAL, B.ns, B.ns))                 # extension shells: no correction
        ratio[:, :FC.N_F, :FC.N_F] = np.where(den > 0, S / np.where(den > 0, den, 1), 1.0)
        out.append(ratio * np.einsum("cs,ct->cst", M1, M2))
    return tuple(out)


def family_scores(err: dict) -> dict:
    out = {}
    for fam, keys in FC.FAMILIES.items():
        vals = [err[k] for k in keys if FC.primary(k) and k in err and np.isfinite(err[k])]
        out[f"rms_{fam}"] = float(np.sqrt(np.mean(np.square(vals)))) if vals else np.nan
        out[f"maxabs_{fam}"] = float(np.max(np.abs(vals))) if vals else np.nan
    return out


SPECTRA = ("yy1h", "yy2h", "yytot", "ksz1h", "ksz2h", "dmcl1h", "dmcl2h", "dmcltot", "ydm1h", "ydm2h", "ydmtot")


def evaluate(B, models, truth, wfam, pdf_quantiles=None, scatter=None, pdf_scatter=None, spectra=None):
    """Long rows (one per model x observable) and one score row per model; ``spectra`` (dict) collects the
    full multipole spectra of the deterministic and lognormal_1R predictions for the figures."""
    rows, scores = [], []
    for label, m in models.items():
        P, E = m["P"], m["E"]
        variants = {"deterministic": B.observables(P, E)}
        for sname, spec in (scatter or {}).items():
            SPP, SEE, SPE = scatter_moments(P, E, *spec, truth=truth, B=B)
            variants[sname] = B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE)
        P35, E35 = P[:, :FC.N_F], E[:, :FC.N_F]
        resP, _ = FF.residual_metrics(P35, truth["P"], wfam["base"], {k[2:]: v for k, v in wfam.items() if k.startswith("P_")})
        resE, _ = FF.residual_metrics(E35, truth["E"], wfam["base"], {k[2:]: v for k, v in wfam.items() if k.startswith("E_")})
        if spectra is not None:
            for vname in ("deterministic", "lognormal_1R"):
                if vname in variants:
                    spectra[(label, vname)] = np.array([variants[vname][k] for k in SPECTRA])
        for vname, o in variants.items():
            sc = B.scalars(o)
            q = None
            if vname == "deterministic" and pdf_quantiles is not None:
                q = pdf_quantiles.get(label)
            elif pdf_scatter is not None:
                q = pdf_scatter.get((label, vname))
            if q is not None:
                for p, qn in enumerate(FC.Q_NAMES):
                    sc[f"dm_{qn}"] = float(q[p])
            err = {k: sc[k] / truth["obs"][k] - 1 for k in sc if k in truth["obs"] and truth["obs"][k]}
            errm = {k: sc[k] / truth["mean_obs"][k] - 1 for k in sc if k in truth["mean_obs"] and truth["mean_obs"][k]}
            for k in sorted(err):
                rows.append({"model": label, "prediction": vname, "observable": k, "value": sc[k],
                             "truth": truth["obs"][k], "rel_error": err[k], "rel_error_vs_truth_mean": errm.get(k, np.nan)})
            srow = {"model": label, "prediction": vname, **{f"P_{k}": v for k, v in resP.items()},
                    **{f"E_{k}": v for k, v in resE.items()}, **family_scores(err)}
            srow.update({f"{k}_vs_mean": v for k, v in family_scores(errm).items()})
            Yb = B.budgets(P, E)
            Yt = B.budgets(truth["P"], truth["E"])
            srow["Y4_rms_bins"] = float(np.sqrt(np.mean((Yb["Y"] / Yt["Y"] - 1)**2)))
            srow["Ne4_rms_bins"] = float(np.sqrt(np.mean((Yb["Ne"] / Yt["Ne"] - 1)**2)))
            Yall = B.budgets(P, E, radius=FC.X_END)
            srow["Y_all_rms_bins"] = float(np.sqrt(np.mean((Yall["Y"] / Yt["Y"] - 1)**2)))
            srow["Ne_all_rms_bins"] = float(np.sqrt(np.mean((Yall["Ne"] / Yt["Ne"] - 1)**2)))
            meta = m.get("meta", {})
            for k in ("objective", "R_fit", "R_cut", "edge", "family", "method"):
                if k in meta:
                    srow[k] = meta[k] if not isinstance(meta[k], dict) else FC.json.dumps(FC.plain(meta[k]))
            scores.append(srow)
    return rows, scores


def pdf_for_models(models, B, scatter_specs=None, which="T"):
    """One PDF pass over every model (deterministic) and the lognormal variants of SCATTER_PDF_MODELS."""
    fb = np.load(FC.WORK / "pdf_fallback.npz")
    ent = FP.Entries(B.ns)
    index = {}
    for label, m in models.items():
        h = ent.new_hist(label)
        index[("det", label)] = h
        for c in range(FC.N_CAL):
            ent.add(c, m["E"][c], h, 1.0)
    for (label, sname), mom in (scatter_specs or {}).items():
        h = ent.new_hist(f"{label}|{sname}")
        index[(sname, label)] = h
        for c in range(FC.N_CAL):
            amp, wq = gauss_hermite(mom[c, 1])
            ent.add(c, amp[:, None] * models[label]["E"][c][None, :], h, wq)
    ent.freeze()
    res = FP.run(ent, which, include_fallback=False, fallback=(fb["hist"], float(fb["overflow"])))
    det = {lab: res["summaries"][h]["quantiles"] for (k, lab), h in index.items() if k == "det"}
    sca = {(lab, k): res["summaries"][h]["quantiles"] for (k, lab), h in index.items() if k != "det"}
    return det, sca, res["timing"]


def gauss_hermite(sigma2, n=9):
    t, w = np.polynomial.hermite.hermgauss(n)
    s = np.sqrt(max(sigma2, 0.0))
    return np.exp(s * np.sqrt(2) * t - sigma2 / 2), w / np.sqrt(np.pi)


def main():
    timer = FC.Timer()
    pr = dict(np.load(FC.PRODUCTS / "profiles.npz"))
    tr = dict(np.load(FC.PRODUCTS / "truth.npz"))
    ts = FC.read_json(FC.PRODUCTS / "truth_summary.json")
    im = dict(np.load(FC.PRODUCTS / "importance_F.npz"))
    B = FB.Basis("T")
    imp = {k: im[k] for k in ("w_P", "w_E", "K_P", "K_E")}
    base = pr["base_mask"][FC.CAL_BINS]
    wfam = {"base": base}
    for k in im:
        if k.startswith("Wfam_"):
            _, f, fam = k.split("_", 2)
            wfam[f"{f}_{fam}"] = im[k]
    truth = truth_dict(pr, tr, ts, "main")
    models, records, info = FM.fit_all(pr, "main", imp, B, truth, log=FC.log)
    FC.log(f"{len(models)} models fitted", timer.mark("fits"))
    # scatter specifications: lognormal amplitude (1R, 4R apertures) and the measured correction
    scatter = {"lognormal_1R": ("lognormal", tr["lognormal_main_1R"]), "lognormal_4R": ("lognormal", tr["lognormal_main_4R"]),
               "measured_2nd_moment": ("measured", None)}
    pdf_scatter_specs = {(lab, s): tr[f"lognormal_main_{s.split('_')[1]}"] for lab in SCATTER_PDF_MODELS if lab in models
                         for s in ("lognormal_1R", "lognormal_4R")}
    det_q, sca_q, pdf_timing = pdf_for_models(models, B, pdf_scatter_specs)
    FC.log("pdf pass done", timer.mark("pdf"))
    spectra = {}
    rows, scores = evaluate(B, models, truth, wfam, det_q, scatter, sca_q, spectra=spectra)
    keys = sorted(spectra)
    np.savez_compressed(FC.PRODUCTS / "model_spectra.npz", ell=B.ell, spectra_names=np.array(SPECTRA),
                        models=np.array([k[0] for k in keys]), predictions=np.array([k[1] for k in keys]),
                        values=np.array([spectra[k] for k in keys]),
                        truth=np.array([tr[f"truth_main__{k}"] for k in SPECTRA]),
                        truth_mean=np.array([tr[f"truthmean_main__{k}"] for k in SPECTRA]))
    # ---------------- controls: objective-(a) grid and stacks on their own truths
    control_rows, control_scores = [], []
    for sel in ("agn_kept", "a2i", "a2ii"):
        tsel = truth_dict(pr, tr, ts, sel)
        cm, crec, _ = FM.fit_all(pr, sel, imp, B, tsel, parts=("grid", "tab"), objectives=("a",))
        cm = {k: v for k, v in cm.items() if k.startswith("grid_a_") or k.startswith("tab_")}
        dq, _, _ = pdf_for_models(cm, B)
        r, s = evaluate(B, cm, tsel, wfam, dq)
        for x in r:
            x["selection"] = sel
        for x in s:
            x["selection"] = sel
        control_rows += r
        control_scores += s
        FC.log(f"control {sel} done", timer.mark(sel))
    # ---------------- save
    labels = list(models)
    np.savez_compressed(FC.PRODUCTS / "models_main.npz", labels=np.array(labels),
                        P=np.array([models[k]["P"] for k in labels]), E=np.array([models[k]["E"] for k in labels]),
                        meta_json=np.array(FC.json.dumps(FC.plain({k: models[k]["meta"] for k in labels}))))
    FC.write_csv(FC.TABLES / "F_fit_records.csv", records)
    FC.write_csv(FC.TABLES / "F_errors_long.csv", rows)
    FC.write_csv(FC.TABLES / "F_model_scores.csv", scores)
    FC.write_csv(FC.TABLES / "F_controls_errors_long.csv", control_rows)
    FC.write_csv(FC.TABLES / "F_controls_scores.csv", control_scores)
    FC.write_json(FC.PRODUCTS / "run_summary.json", {"models": len(models), "lambda_choice": info["lambda_choice"],
                                                     "pdf_timing": pdf_timing, "seconds": timer.marks,
                                                     "scatter_moments_1R": tr["lognormal_main_1R"],
                                                     "scatter_moments_4R": tr["lognormal_main_4R"]})
    FC.log("f2_run done", timer.mark("total"))


if __name__ == "__main__":
    main()
