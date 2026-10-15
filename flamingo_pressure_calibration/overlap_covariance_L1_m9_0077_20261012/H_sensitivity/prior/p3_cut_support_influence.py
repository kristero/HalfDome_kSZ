"""P3: Test B results on fit adequacy vs cut radius, parameter influence radii and parameter precision (read only).

Definitions (Test B):
* sigma_d: white, constant-fractional model-discrepancy term solved so that the GLS chi^2/nu = 1
  (r^T (C + sigma_d^2 I)^-1 r = nu, r = ln model - ln stack on the fitted shells).  sigma_d ~ 0 => B12/B16 describe
  the stack within its statistical errors.
* influence H = (J^T W J)^-1 J^T W  (J = d ln model / d theta, W = fit weights): row p gives how much each shell's
  ln F moves parameter p; band fractions are sum |H_ps| over the band / sum over all fitted shells.
* fit units: the seven 0.2-dex bins 13.5-14.9 (targets 5..11) and the pooled >= 14.9 trend fit ('P').

Sources: B_covariance/tables/b5_cut_support.csv, b4_influence_bands.csv, b4_fits.csv; results/b6_intervals.csv.
Outputs: tables/b_cut_support_summary.csv, b_influence_summary.csv, b_param_precision.csv, b_cut_observables.csv,
p3_summary.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import prior_common as C

UNITS = ["5", "6", "7", "8", "9", "10", "11", "P"]
CUTS = [1.0, 1.34, 1.5, 1.75, 2.0, 3.0, 4.0]


def med_rng(s):
    s = s.dropna()
    if len(s) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(s)), float(s.min()), float(s.max()))


def main():
    S = {}
    cs = pd.read_csv(C.B_TABLES / "b5_cut_support.csv", dtype={"target": str})
    cs = cs[cs.target.isin(UNITS)]
    rows = []
    for spec in ("ALL_P", "ALL_E", "A2ii_P", "A2ii_E", "CLEAN_P", "CLEAN_E"):
        for obj in ("a", "c"):
            sub = cs[(cs.spec == spec) & (cs.objective == obj)]
            if len(sub) == 0:
                continue
            for cut in CUTS:
                q = sub[np.isclose(sub.cut, cut)]
                if len(q) == 0:
                    continue
                row = {"spec": spec, "objective": obj, "cut": cut, "n_units": len(q)}
                for col in ("sigma_d", "chi2_per_dof_sigma_d_0", "rms_lt1", "rms_1_2", "rms_2_4", "rms_in_fit_range",
                            "rms_beyond_cut_to_4R", "mean_resid_beyond_cut"):
                    m, lo, hi = med_rng(q[col])
                    scale = 100.0 if col == "sigma_d" else 1.0
                    row[f"{col}_median"] = m * scale
                    row[f"{col}_min"] = lo * scale
                    row[f"{col}_max"] = hi * scale
                rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_csv(C.TABLES / "b_cut_support_summary.csv", index=False, float_format="%.4g")
    S["cut_support"] = cdf.round(3).to_dict(orient="records")

    # influence: which radii drive each parameter (diagonal a vs GLS c), cuts 2 and 4
    inf = pd.read_csv(C.B_TABLES / "b4_influence_bands.csv", dtype={"target": str})
    inf = inf[inf.target.isin(UNITS)]
    bands = ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4"]
    irows = []
    for spec in ("ALL_P", "ALL_E", "A2ii_P", "A2ii_E"):
        for obj in ("a", "c"):
            for cut in (1.34, 2.0, 4.0):
                sub = inf[(inf.spec == spec) & (inf.objective == obj) & np.isclose(inf.cut, cut)]
                for par in sorted(sub.parameter.unique()):
                    q = sub[sub.parameter == par]
                    irows.append({"spec": spec, "objective": obj, "cut": cut, "parameter": par, "n_units": len(q),
                                  **{b: float(np.median(q[b])) for b in bands},
                                  "n_eff_shells_median": float(np.median(q.n_eff_shells))})
    idf = pd.DataFrame(irows)
    idf.to_csv(C.TABLES / "b_influence_summary.csv", index=False, float_format="%.4g")
    S["influence"] = idf.round(3).to_dict(orient="records")

    # parameter precision (bootstrap SD of ln theta) vs cut
    fits = pd.read_csv(C.B_TABLES / "b4_fits.csv", dtype={"target": str})
    fits = fits[fits.target.isin(UNITS)]
    prow = []
    for spec, pars in (("ALL_P", ("P0", "xc", "beta")), ("ALL_E", ("rho0", "alpha", "beta")),
                       ("A2ii_P", ("P0", "xc", "beta")), ("A2ii_E", ("rho0", "alpha", "beta"))):
        for obj in ("a", "c"):
            for cut in CUTS:
                q = fits[(fits.spec == spec) & (fits.objective == obj) & np.isclose(fits.cut, cut)]
                if len(q) == 0:
                    continue
                row = {"spec": spec, "objective": obj, "cut": cut}
                for p in pars:
                    row[f"sd_ln_{p}_boot_median_pct"] = 100 * float(np.nanmedian(q[f"sd_ln_{p}_boot"]))
                    row[f"neff_shells_{p}_median"] = float(np.nanmedian(q[f"neff_shells_{p}"]))
                prow.append(row)
    pdf = pd.DataFrame(prow)
    pdf.to_csv(C.TABLES / "b_param_precision.csv", index=False, float_format="%.4g")
    S["param_precision"] = pdf.round(3).to_dict(orient="records")

    # observables vs cut (GLS, relative to 2R) and A2-ii vs ALL (same cut)
    iv = pd.read_csv(C.B_RESULTS / "b6_intervals.csv")
    keep = iv[iv.group.str.startswith("B5 cut-off") | (iv.group.str.startswith("B5 sample")
                                                       & iv.variant.str.startswith(("A2ii", "A2i|", "CLEAN")))]
    keep = keep[["group", "field", "variant", "reference", "observable", "exact_percent", "p16_percent",
                 "p84_percent"]]
    keep.to_csv(C.TABLES / "b_cut_observables.csv", index=False, float_format="%.4g")
    C.write_json(C.PRIOR / "p3_summary.json", S)
    print(cdf[(cdf.objective == "c")][["spec", "cut", "sigma_d_median", "sigma_d_min", "sigma_d_max",
                                       "rms_2_4_median", "rms_beyond_cut_to_4R_median"]].round(2).to_string())


if __name__ == "__main__":
    main()
