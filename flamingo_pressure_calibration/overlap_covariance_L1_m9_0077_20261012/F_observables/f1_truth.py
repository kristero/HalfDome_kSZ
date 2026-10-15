"""F1/F2 truth: observables of the measured per-halo profiles (to 4 R200c) through the lightcone model.

TRUTH (scatter-inclusive): every calibrated bin contributes with its measured haloes' own profiles:
1h terms and the DM variance use the inclusion-weighted second moments <F_s F_t> (= mean over the bin's
haloes of |y_ell,i|^2, exact in the basis), linear terms the mean, and the DM PDF deposits every halo's
own column (per-halo compound Poisson).  'truth mean' (deterministic) = the mean profile alone, as a
one-profile-per-bin painting would use it.  Selections: main (all gas, recent-AGN gas removed) and the
controls agn_kept / a2i / a2ii.  Also: lognormal amplitude-scatter models of the mean (sigma from per-halo
Y(<R) / N_e(<R) moments, R = 1, 4 R200c), the per-fold training means (spatial CV references), and the
Mecke quantile Jacobians of the deterministic truth mean, of the per-halo truth (64-halo subsample per
bin, reweighted) and of the 8 training means.
Outputs: products/truth.npz, products/truth_summary.json, products/pdf_fallback.npz.
"""
from __future__ import annotations

import f_basis as FB
import f_common as FC
import f_pdf as FP
import numpy as np

SELECTIONS = ("main", "agn_kept", "a2i", "a2ii")
GH_NODES = 9
APERTURES = (1.0, 4.0)
MECKE_SUBSAMPLE = FC.MECKE_SUBSAMPLE


def bin_rows(pr, c):
    return np.flatnonzero(pr["mass_bin"] == FC.CAL_BINS[c])


def lognormal_moments(pr, sel, aperture):
    """Per calibrated bin: sigma_P^2, sigma_E^2 (ln <X^2>/<X>^2) and <Y N>/(<Y><N>) from per-halo
    Y(<aperture) and N_e(<aperture) (inclusion-weighted)."""
    out = np.zeros((FC.N_CAL, 3))
    for c in range(FC.N_CAL):
        rows = bin_rows(pr, c)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        Y = pr[f"P_{sel}_int{aperture:g}R"][rows]
        Ne = pr[f"E_{sel}_int{aperture:g}R"][rows]
        out[c] = (np.log(w @ Y**2 / (w @ Y)**2), np.log(w @ Ne**2 / (w @ Ne)**2), (w @ (Y * Ne)) / ((w @ Y) * (w @ Ne)))
    return out


def gauss_hermite(sigma2, n=GH_NODES):
    """Amplitudes A_q = exp(sigma sqrt2 t_q - sigma^2/2) and weights (unit mean, <A^2> = exp(sigma^2))."""
    t, w = np.polynomial.hermite.hermgauss(n)
    s = np.sqrt(max(sigma2, 0.0))
    return np.exp(s * np.sqrt(2) * t - sigma2 / 2), w / np.sqrt(np.pi)


def fold_means(pr, key, fold):
    out = np.zeros((FC.N_CAL, FC.N_F))
    for c in range(FC.N_CAL):
        rows = bin_rows(pr, c)
        rows = rows[pr["fold"][rows] != fold]
        w = pr["weight"][rows]
        out[c] = (w / w.sum()) @ pr[key][rows]
    return out


def main():
    timer = FC.Timer()
    pr = dict(np.load(FC.PRODUCTS / "profiles.npz"))
    B = FB.Basis("T")
    cal = FC.CAL_BINS
    out, summary = {}, {"observables": {}}
    # ---------------- moment observables
    for sel in SELECTIONS:
        P, Ee = pr[f"meanP_{sel}"][cal], pr[f"meanE_{sel}"][cal]
        S = dict(SPP=pr[f"SPP_{sel}"][cal], SEE=pr[f"SEE_{sel}"][cal], SPE=pr[f"SPE_{sel}"][cal])
        full = B.observables(P, Ee, **S)
        mean = B.observables(P, Ee)
        calonly = B.observables(P, Ee, **S, cal_only=True)
        for tag, o in (("truth", full), ("truthmean", mean), ("truthcal", calonly)):
            for k, v in o.items():
                out[f"{tag}_{sel}__{k}"] = np.asarray(v)
            summary["observables"][f"{tag}_{sel}"] = B.scalars(o)
        bud = B.budgets(P, Ee)
        out[f"Y4_{sel}"], out[f"Ne4_{sel}"] = bud["Y"], bud["Ne"]
    # exact scatter boost per bin and multipole (main): tr(K S) / m^T K m
    P, Ee = pr["meanP_main"][cal], pr["meanE_main"][cal]
    Pp, Ep = B.pad(P), B.pad(Ee)
    for name, Kk, S, a, b in (("yy1h", B.K_yy1h, pr["SPP_main"][cal], Pp, Pp), ("ksz1h", B.K_ksz1h, pr["SEE_main"][cal], Ep, Ep),
                              ("dmcl1h", B.K_dm1h, pr["SEE_main"][cal], Ep, Ep), ("ydm1h", B.K_ydm1h, pr["SPE_main"][cal], Pp, Ep)):
        num = np.einsum("cstl,cst->cl", Kk, B.pad2(S), optimize=True)
        den = np.einsum("cstl,cs,ct->cl", Kk, a, b, optimize=True)
        out[f"boost_{name}"] = num / den
    # lognormal amplitude moments
    for ap in APERTURES:
        for sel in ("main",):
            out[f"lognormal_{sel}_{ap:g}R"] = lognormal_moments(pr, sel, ap)
    FC.log("moments", timer.mark("moments"))
    # ---------------- PDF pass 1: truths, means, lognormal models, fold means
    ent = FP.Entries(B.ns)
    hists = {}
    for sel in SELECTIONS:
        h = ent.new_hist(f"truth_{sel}")
        hists[f"truth_{sel}"] = h
        for c in range(FC.N_CAL):
            rows = bin_rows(pr, c)
            w = pr["weight"][rows] / pr["weight"][rows].sum()
            ent.add(c, pr[f"E_{sel}"][rows], h, w)
        h = ent.new_hist(f"truthmean_{sel}")
        hists[f"truthmean_{sel}"] = h
        for c in range(FC.N_CAL):
            ent.add(c, pr[f"meanE_{sel}"][cal[c]], h, 1.0)
    for ap in APERTURES:
        h = ent.new_hist(f"lognormal_main_{ap:g}R")
        hists[f"lognormal_main_{ap:g}R"] = h
        mom = out[f"lognormal_main_{ap:g}R"]
        for c in range(FC.N_CAL):
            amp, wq = gauss_hermite(mom[c, 1])
            ent.add(c, amp[:, None] * pr["meanE_main"][cal[c]][None, :], h, wq)
    folds = FC.FOLDS
    for k in range(folds):
        h = ent.new_hist(f"trainmean_fold{k}")
        hists[f"trainmean_fold{k}"] = h
        fm = fold_means(pr, "E_main", k)
        out[f"trainmeanE_fold{k}"] = fm
        out[f"trainmeanP_fold{k}"] = fold_means(pr, "P_main", k)
        for c in range(FC.N_CAL):
            ent.add(c, fm[c], h, 1.0)
    ent.freeze()
    first = FP.run(ent, "T", keep_density=True)
    np.savez_compressed(FC.WORK / "pdf_fallback.npz", hist=first["fallback"][0], overflow=first["fallback"][1])
    summary["pdf"] = {}
    for label, s in zip(first["labels"], first["summaries"]):
        summary["pdf"][label] = {"quantiles": s["quantiles"], "mean": s["mean"], "variance": s["variance"],
                                 "mass_on_grid": s["mass_on_grid"], "negative_mass": s["negative_mass"]}
        out[f"pdf_{label}__quantiles"] = s["quantiles"]
        out[f"pdf_{label}__density"] = s["density"]
        out[f"pdf_{label}__density_at_quantiles"] = s["density_at_quantiles"]
        out[f"pdf_{label}__mean"] = s["mean"]
        out[f"pdf_{label}__variance"] = s["variance"]
    summary["pdf_pass1_timing"] = first["timing"]
    FC.log("pdf pass 1", timer.mark("pdf1"))
    # ---------------- PDF pass 2: Mecke Jacobians
    mecke = []
    labels = []
    # (a) deterministic truth mean (main), with fallback pieces
    ent2 = FP.Entries(B.ns)
    h0 = ent2.new_hist("truthmean_main")
    plan = {"fallback": True}
    for c in range(FC.N_CAL):
        plan[c] = ent2.add(c, pr["meanE_main"][cal[c]], h0, 1.0)
    mecke.append((first["summaries"][hists["truthmean_main"]], plan))
    labels.append("truthmean_main")
    # (b) per-halo truth (main), subsample of <= 64 haloes per bin, weights rescaled
    rng = np.random.default_rng(FC.SEED + 5)
    h1 = ent2.new_hist("truth_main_subsample")
    plan = {"fallback": True}
    for c in range(FC.N_CAL):
        rows = bin_rows(pr, c)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        pick = rng.choice(len(rows), size=min(MECKE_SUBSAMPLE, len(rows)), replace=False)
        plan[c] = ent2.add(c, pr["E_main"][rows[pick]], h1, w[pick] * (w.sum() / w[pick].sum()))
    mecke.append((first["summaries"][hists["truth_main"]], plan))
    labels.append("truth_main_subsample")
    # (c) per-fold training means
    for k in range(folds):
        hk = ent2.new_hist(f"trainmean_fold{k}")
        plan = {"fallback": False}
        for c in range(FC.N_CAL):
            plan[c] = ent2.add(c, out[f"trainmeanE_fold{k}"][c], hk, 1.0)
        mecke.append((first["summaries"][hists[f"trainmean_fold{k}"]], plan))
        labels.append(f"trainmean_fold{k}")
    ent2.freeze()
    second = FP.run(ent2, "T", mecke=mecke, include_fallback=True)
    for lab, J, Jfb in zip(labels, second["J"], second["Jfb"]):
        out[f"Jq_{lab}"] = J                      # [quantile, cal bin, basis shell]
        out[f"Jqfb_{lab}"] = Jfb                  # [quantile, engine bin, F shell] (fallback haloes)
    summary["mecke"] = {lab: {"sum_rule_over_Q": ((second["J"][i].sum(axis=(1, 2)) + second["Jfb"][i].sum(axis=(1, 2)))
                                                  / mecke[i][0]["quantiles"] - 1).tolist()}
                        for i, lab in enumerate(labels)}
    summary["pdf_pass2_timing"] = second["timing"]
    # consistency: PDF mean/variance vs moment-form mean DM and DM variance
    summary["pdf_vs_moments"] = {}
    for sel in SELECTIONS:
        for tag in ("truth", "truthmean"):
            s = summary["pdf"][f"{tag}_{sel}"]
            mo = summary["observables"][f"{tag}_{sel}"]
            summary["pdf_vs_moments"][f"{tag}_{sel}"] = {"mean": s["mean"] / mo["dm_mean"] - 1,
                                                         "variance": s["variance"] / mo["dm_var"] - 1}
    np.savez_compressed(FC.PRODUCTS / "truth.npz", **out)
    summary["seconds"] = timer.mark("total")
    FC.write_json(FC.PRODUCTS / "truth_summary.json", summary)
    FC.log("done", summary["seconds"])


if __name__ == "__main__":
    main()
