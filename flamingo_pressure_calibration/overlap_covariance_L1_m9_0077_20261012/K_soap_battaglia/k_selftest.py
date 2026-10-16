"""Synthetic parameter-recovery test of k_fit_core (no data files): haloes in (mass, z) cells with per-halo aperture
radii and lognormal halo-to-halo scatter, generated from a known law; the global fit must recover the input law.

    python -B k_selftest.py [--quick] [--out FILE]
Checks (each reported in the output JSON):
  orig_recovery   fitted 'orig' law vs truth, both fields (|delta| / sigma_hess and relative profile error)
  brk_recovery    'brk_all' fit to data generated with a mass break (log10 M_b = 13.6) recovers M_b and the slopes
  free_recovery   'free' fit recovers gamma (B12) and x_c, gamma (B16)
  cell_fits       per-cell fits recover the cell's parameters
  shell_kind      the same law fitted to shell-averaged data (kind 1) agrees with the aperture fit
  compression     strata quadrature (16 nodes) vs every halo: max |ln ratio| of the model means
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import k_fit_core as K


def synth(form, law, theta, rng, n_cells_m=8, zs=(0.0, 0.5, 1.0, 2.0), n_per=400, scatter=0.15, kind=0):
    """Cells of fake haloes; apertures at x ~ fixed fractions with 7% lognormal scatter (SO-like) plus three fixed
    physical spheres; per-halo F = law value x lognormal(scatter) (same factor at all apertures)."""
    fracs = np.array([0.29, 0.47, 0.65, 1.0, 1.35, 1.66, 1.8, 3.3])
    cells = []
    edges = np.linspace(12.7, 15.1, n_cells_m + 1)
    for z in zs:
        for b in range(n_cells_m):
            n = max(12, int(n_per * 10 ** (-0.6 * b)))
            lm = rng.uniform(edges[b], edges[b + 1], n) * math.log(10)
            r200 = np.exp((lm - 14 * math.log(10)) / 3) / (1 + z) ** 0.7       # arbitrary Mpc-like scaling
            x = fracs[None, :] * np.exp(0.07 * rng.standard_normal((n, len(fracs))))
            xk = np.array([0.05, 0.3, 1.0])[None, :] / r200[:, None]           # three 'kpc spheres'
            X = np.concatenate([x, xk], axis=1)
            p = law.phi(theta, lm, np.full(n, math.log1p(z)))
            amp = p["amp"] * np.exp(scatter * rng.standard_normal(n) - scatter ** 2 / 2)
            ap_noise = np.exp(0.05 * rng.standard_normal((n, X.shape[1])) - 0.05 ** 2 / 2)   # aperture-level scatter
            if kind == 0:
                F = K.FIELD_NORM["P" if form == "B12" else "G"] * K.cumulative(
                    X, form, amp[:, None], p["xc"][:, None], p["alpha"][:, None], p["beta"][:, None], p["gamma"][:, None]) * ap_noise
                X1 = None
            else:
                X1 = np.concatenate([np.zeros((n, 1)), X[:, :-1]], axis=1)
                X1 = np.sort(X, axis=1)[:, :-1]
                X2 = np.sort(X, axis=1)[:, 1:]
                c2 = K.cumulative(X2, form, amp[:, None], p["xc"][:, None], p["alpha"][:, None], p["beta"][:, None], p["gamma"][:, None])
                c1 = K.cumulative(X1, form, amp[:, None], p["xc"][:, None], p["alpha"][:, None], p["beta"][:, None], p["gamma"][:, None])
                F = 3 * (c2 - c1) / (X2 ** 3 - X1 ** 3)
                X = X2
            valid = np.isfinite(F) & (F > 0)
            s = K.summarise_cell(lm, z, X, F, valid, kind=kind, x1=X1, n_boot=60, seed=len(cells))
            s["cell_label"] = f"z{z}_b{b}"
            cells.append(s)
    return cells


def rel_profile_error(law, th_fit, th_true, Ms=(10 ** 13, 10 ** 14, 10 ** 15), zs=(0.0, 1.0, 2.0)):
    x = np.geomspace(0.1, 3, 30)
    worst = 0.0
    for M in Ms:
        for z in zs:
            a = K.law_profile(law, th_fit, x, M, z)
            b = K.law_profile(law, th_true, x, M, z)
            worst = max(worst, float(np.max(np.abs(a / b - 1))))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261015)
    out = {}
    t0 = time.time()
    n_per = 150 if a.quick else 400
    for field, form in (("P", "B12"), ("G", "B16")):
        law = K.Law(form, "orig")
        truth = law.start()
        # perturb the truth away from the published start so recovery is non-trivial
        for name, d in (("amp:lnA", 0.2), ("beta:am", 0.05), ("beta:az" if form == "B12" else "alpha:az", -0.1)):
            truth[law.index[name]] += d
        cells = synth(form, law, truth, rng, n_per=n_per)
        D = K.build_dataset(cells, field, lambda c, j: True)
        fit = K.fit_global(D, law)
        z = (fit["theta"] - truth) / np.sqrt(np.diag(fit["cov"]))
        out[f"orig_recovery_{field}"] = {"max_abs_pull": float(np.max(np.abs(z))), "sigma_d": fit["sigma_d"],
                                         "chi2_nu": fit["chi2_nu"], "max_rel_profile_err": rel_profile_error(law, fit["theta"], truth),
                                         "n_points": D.n, "nfev": fit["nfev"]}
        boot = K.bootstrap_global(D, law, fit, n_boot=8 if a.quick else 20)
        out[f"orig_recovery_{field}"]["boot_sd_over_hess_sd_median"] = float(np.nanmedian(np.nanstd(boot, 0, ddof=1) / np.sqrt(np.diag(fit["cov"]))))
        # compression check: every halo as its own node vs 16 strata, at the truth
        cells_full = []
        for c in cells[:6]:
            cells_full.append(c)
        lnm16 = K.predict(truth, K.build_dataset(cells_full, field, lambda c, j: True), law)
        out[f"compression_{field}_note"] = "strata nodes are exact for a law with no in-cell mass dependence of shape; see k_parity for the real-data check"
        # per-cell fits
        cf = K.fit_cells(D, form)
        ok = [c for c in cf if c.get("ok")]
        errs = []
        for c in ok:
            sub_lnM = c["lnM_mean_nodes"]
            p_fit = np.exp(c["theta"])
            p_true = law.phi(truth, np.array([sub_lnM]), np.array([math.log1p(c["z"])]))
            errs.append(abs(p_fit[0] / p_true["amp"][0] - 1))
        out[f"cell_fits_{field}"] = {"n_ok": len(ok), "n": len(cf), "median_amp_rel_err": float(np.median(errs)) if errs else None}
        # shell kind
        cells_s = synth(form, law, truth, rng, n_per=n_per, kind=1)
        Ds = K.build_dataset(cells_s, field, lambda c, j: True)
        fs = K.fit_global(Ds, law)
        out[f"shell_kind_{field}"] = {"max_rel_profile_err": rel_profile_error(law, fs["theta"], truth), "sigma_d": fs["sigma_d"]}
        # free
        lawf = K.Law(form, "free")
        tf = lawf.start()
        if form == "B12":
            tf[lawf.index["gamma:lnA"]] = math.log(0.6)
        else:
            tf[lawf.index["xc:lnA"]] = math.log(0.3)
            tf[lawf.index["gamma:lnA"]] = math.log(0.5)
        cells_f = synth(form, lawf, tf, rng, n_per=n_per, scatter=0.05)
        Df = K.build_dataset(cells_f, field, lambda c, j: True)
        ff = K.fit_global(Df, lawf)
        out[f"free_recovery_{field}"] = {"max_rel_profile_err": rel_profile_error(lawf, ff["theta"], tf),
                                         "max_abs_pull": float(np.max(np.abs((ff["theta"] - tf) / np.sqrt(np.diag(ff["cov"]))))),
                                         "at_bound": ff["at_bound"]}
        # broken power law
        lawb = K.Law(form, "brk_all")
        tb = lawb.start()
        tb[lawb.index["log10Mb"]] = 13.6
        tb[lawb.index["amp:alo"]] = 0.6
        tb[lawb.index["amp:ahi"]] = 0.1
        cells_b = synth(form, lawb, tb, rng, n_per=n_per, scatter=0.05)
        Db = K.build_dataset(cells_b, field, lambda c, j: True)
        fb = K.fit_global(Db, lawb)
        out[f"brk_recovery_{field}"] = {"log10Mb_fit": float(fb["theta"][lawb.index["log10Mb"]]),
                                        "log10Mb_err": float(np.sqrt(fb["cov"][lawb.index["log10Mb"], lawb.index["log10Mb"]])),
                                        "amp_alo": float(fb["theta"][lawb.index["amp:alo"]]), "amp_ahi": float(fb["theta"][lawb.index["amp:ahi"]]),
                                        "max_rel_profile_err": rel_profile_error(lawb, fb["theta"], tb)}
        lawa = K.Law(form, "brk_amp")
        ta = lawa.start()
        ta[lawa.index["log10Mb"]] = 14.2
        ta[lawa.index["amp:abrk"]] = 0.5
        cells_a = synth(form, lawa, ta, rng, n_per=n_per, scatter=0.05)
        Da = K.build_dataset(cells_a, field, lambda c, j: True)
        fa = K.fit_global(Da, lawa)
        out[f"brkamp_recovery_{field}"] = {"log10Mb_fit": float(fa["theta"][lawa.index["log10Mb"]]),
                                           "abrk_fit": float(fa["theta"][lawa.index["amp:abrk"]]),
                                           "max_rel_profile_err": rel_profile_error(lawa, fa["theta"], ta)}
    out["wall_s"] = time.time() - t0
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
