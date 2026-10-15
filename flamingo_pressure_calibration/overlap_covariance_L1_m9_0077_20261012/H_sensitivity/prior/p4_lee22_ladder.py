"""P4: Test G's Lee22-style parametrisation ladder, reorganised (read only).

Inner profile (both fields): F_in(x) = A (x/x_c)^gamma (1 + x/x_c)^(-beta), gamma = -0.3 unless freed.
Mass laws per parameter p (in ln p): const, pl: ln p0 + a_m ln(M/10^14.5), bpl: broken at one common M_cut (free or
Lee22's 10^13.77-13.78 Msun), +c: + a_c ln(c200c/10), gamma free (g) or a mass law (gM).
Outer options: O0 none; O1 the physical 2-halo term (electrons 0 parameters; pressure tau0, q); O2 = O1 with the inner
part times [1 + (x/x_t)^4]^(-3/2), x_t = x_t0 (M/10^14.5)^a_xt (+2).
Likelihood 'floor': chi^2 = sum_c r_c^T (C_c + sigma_d^2 I)^-1 r_c, sigma_d set so the per-bin reference has chi^2/nu = 1;
BIC = chi^2 + k ln n.  Selection: fewest k with dBIC <= 6 and CV chi^2 within the best CV model's fold scatter.
All ladder fits use ALL-GAS stacks (dset 'mass' = 10 mass bins; 'massc' = mass x concentration cells).

Sources: G_physical_outer/results/tables/G_selection_{models,ladder,observables}.csv.
Outputs: tables/lee22_ladder_selected.csv, lee22_ladder_steps.csv, lee22_named_models.csv, lee22_observables.csv,
p4_summary.json.
"""
from __future__ import annotations

import json

import pandas as pd

import prior_common as C


def main():
    m = pd.read_csv(C.G_TABLES / "G_selection_models.csv")
    lad = pd.read_csv(C.G_TABLES / "G_selection_ladder.csv")
    obs = pd.read_csv(C.G_TABLES / "G_selection_observables.csv")
    S = {}
    sel = m[m.selected == True][["field", "dset", "R_fit", "model", "k", "chi2nu_floor", "dBIC_floor", "chi2nu_stat",
                                 "chi2nu_L22", "sigma_floor", "cv_minus_best", "cv_scatter_best", "params"]].copy()
    sel["log10Mcut"] = [json.loads(p).get("log10Mcut") for p in sel.params]
    sel["sigma_floor_pct"] = 100 * sel.sigma_floor
    sel.drop(columns=["params", "sigma_floor"]).to_csv(C.TABLES / "lee22_ladder_selected.csv", index=False,
                                                       float_format="%.4g")
    S["selected"] = sel.drop(columns=["params"]).round(3).to_dict(orient="records")

    steps = lad[["field", "dset", "R_fit", "k", "model", "chi2nu_floor", "dBIC_floor", "pass_bic", "pass_cv",
                 "selected"]]
    steps.to_csv(C.TABLES / "lee22_ladder_steps.csv", index=False, float_format="%.4g")

    # named models of interest at every R_fit (mass cells): Lee22's own choices, free gamma, PB references
    names = ["PL1+O0", "PL3+O0", "PL3+O1", "PL3+O2", "BPLa_L22+O0", "BPLs_L22+O0", "BPLa_L22+O1", "BPLs_L22+O1",
             "BPLa_L22+O2", "BPLs_L22+O2", "BPLall_L22+O2", "BPLa+O2", "BPLs+O2", "BPLall+O0", "BPLall+O1",
             "BPLall+O2", "PL3g+O2", "PL3gM+O2", "PB+O0", "PB+O1", "PB+O2"]
    nm = m[(m.dset == "mass") & m.model.isin(names)][["field", "R_fit", "model", "k", "chi2nu_floor", "dBIC_floor",
                                                      "chi2nu_L22", "cv_minus_best", "params"]].copy()
    nm["log10Mcut"] = [json.loads(p).get("log10Mcut") for p in nm.params]
    nm = nm.drop(columns=["params"])
    nm.to_csv(C.TABLES / "lee22_named_models.csv", index=False, float_format="%.4g")
    S["named_mass"] = nm.round(3).to_dict(orient="records")
    # concentration cells
    cnames = ["PL3+O2", "PL3cA+O2", "PL3c3+O2", "L22best+O2", "L22best_L22+O2", "PL1cA+O2", "PBcell+O2", "PL3+O0",
              "PL3c3+O0", "L22best+O0", "L22best_L22+O0", "PBcell+O0"]
    nc = m[(m.dset == "massc") & m.model.isin(cnames)][["field", "R_fit", "model", "k", "chi2nu_floor", "dBIC_floor",
                                                        "chi2nu_L22"]]
    nc.to_csv(C.TABLES / "lee22_concentration_models.csv", index=False, float_format="%.4g")
    S["named_massc"] = nc.round(3).to_dict(orient="records")
    # observables of the ladder's selected minimal models and references (% RMS)
    o = obs.copy()
    for c in ("rms_tSZ", "rms_tSZ_vs_mean", "rms_kSZ", "rms_kSZ_vs_mean", "rms_DM", "rms_DM_vs_mean"):
        o[c] = 100 * o[c]
    o.to_csv(C.TABLES / "lee22_observables.csv", index=False, float_format="%.4g")
    S["observables_paint"] = o[o["eval"] == "paint"].round(3).to_dict(orient="records")
    C.write_json(C.PRIOR / "p4_summary.json", S)
    print(sel.drop(columns=["params"]).round(3).to_string())


if __name__ == "__main__":
    main()
