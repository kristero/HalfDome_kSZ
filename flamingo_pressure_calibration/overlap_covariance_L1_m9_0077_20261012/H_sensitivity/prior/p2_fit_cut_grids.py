"""P2: R_fit x R_cut error grids of the earlier tests, reorganised (read only).

Family score = RMS over the family's observables of eps_O = O_model / O_truth - 1 (in %):
  tSZ: C_ell^yy (1h+2h) at ell = 100, 300, 1000, 3000; kSZ: C_ell^kSZ,1h at the same ell;
  DM: mean, variance and the 16/50/84/97.5/99.9 % quantiles of the halo DM PDF (z_s = 1);
  tSZxDM: C_ell^{y DM} at the same ell.
"truth" = measured per-halo profiles (scatter-inclusive); "vs mean" = deterministic prediction against the truth built
from the stacked mean profile only (isolates fit + cut).

Sources: F_observables/results/tables/F_model_scores.csv, F_controls_scores.csv (Test F, B12/B16 only);
G_physical_outer/results/tables/G_scores.csv (Test G, B12/B16 = P0/E0 and the physical / empirical outer models).
Outputs: tables/grid_*.csv (long and pivoted) and p2_summary.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import prior_common as C

FAM = ("tSZ", "kSZ", "DM", "tSZxDM")


def pct(x):
    return 100.0 * x


def pivots(df, value, index="R_fit", columns="R_cut"):
    return df.pivot_table(index=index, columns=columns, values=value, aggfunc="first")


def main():
    S = {}
    # ------------------------------------------------------------------ Test F: B12/B16 grids (main truth = all gas)
    f = pd.read_csv(C.F_TABLES / "F_model_scores.csv")
    g = f[f.model.str.startswith("grid_") & (f.edge == "hard")].copy()
    rows = []
    for _, r in g.iterrows():
        row = {"source": "F_model_scores.csv", "truth": "all gas", "objective": r.objective, "prediction": r.prediction,
               "R_fit": r.R_fit, "R_cut": r.R_cut}
        for fam in FAM:
            row[f"rms_{fam}"] = pct(r[f"rms_{fam}"])
            row[f"rms_{fam}_vs_mean"] = pct(r[f"rms_{fam}_vs_mean"])
        for reg in ("inner_lt_1R", "1_to_2R", "outer_2_to_4R"):
            row[f"P_rms_{reg}"] = r[f"P_rms_{reg}"]
            row[f"E_rms_{reg}"] = r[f"E_rms_{reg}"]
        rows.append(row)
    Fg = pd.DataFrame(rows)
    Fg.to_csv(C.TABLES / "grid_F_B12B16_long.csv", index=False, float_format="%.4g")
    det_a = Fg[(Fg.objective == "a") & (Fg.prediction == "deterministic")]
    S["F_objective_a_deterministic_vs_truth"] = {fam: pivots(det_a, f"rms_{fam}").round(2).to_dict() for fam in FAM}
    S["F_objective_a_deterministic_vs_mean"] = {fam: pivots(det_a, f"rms_{fam}_vs_mean").round(2).to_dict() for fam in FAM}
    ln4 = Fg[(Fg.objective == "a") & (Fg.prediction == "lognormal_4R")]
    S["F_objective_a_lognormal4R_vs_truth"] = {fam: pivots(ln4, f"rms_{fam}").round(2).to_dict() for fam in FAM}
    # best (R_fit, R_cut) per family and objective (deterministic, vs truth), and the Pareto set
    best = []
    for obj in sorted(Fg.objective.dropna().unique()):
        sub = Fg[(Fg.objective == obj) & (Fg.prediction == "deterministic")]
        for fam in FAM:
            i = sub[f"rms_{fam}"].idxmin()
            best.append({"objective": obj, "family": fam, "R_fit": sub.loc[i, "R_fit"], "R_cut": sub.loc[i, "R_cut"],
                         "rms_pct": sub.loc[i, f"rms_{fam}"],
                         "rms_at_4_4": float(sub[(sub.R_fit == 4) & (sub.R_cut == 4)][f"rms_{fam}"].iloc[0]),
                         "rms_at_2_4": float(sub[(sub.R_fit == 2) & (sub.R_cut == 4)][f"rms_{fam}"].iloc[0])})
    pd.DataFrame(best).to_csv(C.TABLES / "grid_F_best_per_family.csv", index=False, float_format="%.4g")
    # profile accuracy along R_fit at R_cut = 4 (objective a): the inner-accuracy cost of fitting further out
    prof = det_a[det_a.R_cut == 4][["R_fit", "P_rms_inner_lt_1R", "P_rms_1_to_2R", "P_rms_outer_2_to_4R",
                                    "E_rms_inner_lt_1R", "E_rms_1_to_2R", "E_rms_outer_2_to_4R",
                                    "rms_tSZ", "rms_kSZ", "rms_DM", "rms_tSZxDM"]]
    prof.to_csv(C.TABLES / "grid_F_profile_rms_vs_Rfit_cut4.csv", index=False, float_format="%.4g")
    S["F_profile_rms_vs_Rfit_cut4"] = prof.round(2).to_dict(orient="records")
    # objectives at R_cut = 4
    objc = Fg[(Fg.R_cut == 4) & (Fg.prediction == "deterministic")][
        ["objective", "R_fit", "rms_tSZ", "rms_kSZ", "rms_DM", "rms_tSZxDM"]].sort_values(["objective", "R_fit"])
    objc.to_csv(C.TABLES / "grid_F_objectives_cut4.csv", index=False, float_format="%.4g")

    # ------------------------------------------------------------------ Test F controls: B12/B16 fitted to A2-ii stacks
    c = pd.read_csv(C.F_TABLES / "F_controls_scores.csv")
    c = c[c.model.str.startswith("grid_a_")]
    crow = []
    for _, r in c.iterrows():
        row = {"selection": r.selection, "R_fit": r.R_fit, "R_cut": r.R_cut}
        for fam in FAM:
            row[f"rms_{fam}"] = pct(r[f"rms_{fam}"])
            row[f"rms_{fam}_vs_mean"] = pct(r[f"rms_{fam}_vs_mean"])
        for reg in ("inner_lt_1R", "1_to_2R", "outer_2_to_4R"):
            row[f"P_rms_{reg}"] = r[f"P_rms_{reg}"]
            row[f"E_rms_{reg}"] = r[f"E_rms_{reg}"]
        crow.append(row)
    Cg = pd.DataFrame(crow)
    Cg.to_csv(C.TABLES / "grid_F_controls_B12B16_long.csv", index=False, float_format="%.4g")
    a2 = Cg[Cg.selection == "a2ii"]
    S["F_controls_a2ii_objective_a_deterministic_vs_own_truth"] = {fam: pivots(a2, f"rms_{fam}").round(2).to_dict()
                                                                  for fam in FAM}
    S["F_controls_a2ii_vs_mean"] = {fam: pivots(a2, f"rms_{fam}_vs_mean").round(2).to_dict() for fam in FAM}

    # ------------------------------------------------------------------ Test G grids (5 x 5)
    G = pd.read_csv(C.G_TABLES / "G_scores.csv")
    G["fam_rms_pct"] = np.where(G.field == "P", G.rms_tSZ, np.nan)
    keep = []
    for _, r in G.iterrows():
        d = {"field": r.field, "model": r.model, "fit_target": r.fit_target, "objective": r.objective,
             "R_fit": r.R_fit, "R_cut": r.R_cut, "eval": r["eval"], "prediction": r.prediction,
             "dm_quantiles_included": r.dm_quantiles_included}
        if r.field == "P":
            d["rms_tSZ"] = pct(r.rms_tSZ)
            d["rms_tSZ_vs_mean"] = pct(r.rms_tSZ_vs_mean)
        else:
            d["rms_kSZ"] = pct(r.rms_kSZ)
            d["rms_kSZ_vs_mean"] = pct(r.rms_kSZ_vs_mean)
            d["rms_DM"] = pct(r.rms_DM)
            d["rms_DM_vs_mean"] = pct(r.rms_DM_vs_mean)
            d["DM_mean_var_only_rms"] = pct(r.DM_mean_var_only_rms)
        d["profile_rms_lt1R"] = r.profile_rms_inner_lt_1R
        d["profile_rms_1_2R"] = r.profile_rms_1_to_2R
        d["profile_rms_2_4R"] = r.profile_rms_outer_2_to_4R
        keep.append(d)
    Gl = pd.DataFrame(keep)
    Gl.to_csv(C.TABLES / "grid_G_long.csv", index=False, float_format="%.4g")
    out = {}
    for model, val in (("P0", "rms_tSZ"), ("P2m", "rms_tSZ"), ("P1", "rms_tSZ"), ("E0", "rms_kSZ"), ("E0", "rms_DM"),
                       ("E2gf", "rms_kSZ"), ("E2gf", "rms_DM"), ("E1", "rms_kSZ"), ("E1", "rms_DM")):
        for tgt, ev in (("main", "paint"), ("main", "all"), ("a2ii", "a2ii")):
            for pred in ("det", "S1"):
                sub = Gl[(Gl.model == model) & (Gl.fit_target == tgt) & (Gl["eval"] == ev) & (Gl.prediction == pred)
                         & (Gl.objective == "a")]
                if len(sub):
                    out[f"{model}|{val}|fit={tgt}|eval={ev}|{pred}|obj=a"] = pivots(sub, val).round(2).to_dict()
    S["G_grids"] = out

    # ------------------------------------------------------------------ parity check G P0/E0 (all, det, a) vs F grid a
    par = []
    for _, r in Gl[(Gl.model.isin(["P0", "E0"])) & (Gl.fit_target == "main") & (Gl["eval"] == "all")
                   & (Gl.prediction == "det") & (Gl.objective == "a")].iterrows():
        fr = det_a[(det_a.R_fit == r.R_fit) & (det_a.R_cut == r.R_cut)]
        if len(fr) == 0:
            continue
        if r.model == "P0":
            par.append(abs(fr.rms_tSZ.iloc[0] - r.rms_tSZ))
        else:
            par.append(abs(fr.rms_kSZ.iloc[0] - r.rms_kSZ))
    S["parity_G_P0E0_vs_F_grid_a_max_abs_pct_points"] = float(max(par)) if par else None
    S["parity_n"] = len(par)
    # importance-weighted profile RMS (weights = F1 importance of the family on resolved shells): how well each
    # fitted form reproduces the *sensitive* part of the stacked profile
    irm = []
    for _, r in f[(f.prediction == "deterministic") & ((f.R_cut == 4) | f.R_cut.isna())].iterrows():
        irm.append({"source": "F", "model": r.model, "objective": r.objective, "R_fit": r.R_fit, "R_cut": r.R_cut,
                    "P_irms_tSZ": r.P_irms_tSZ, "E_irms_kSZ": r.E_irms_kSZ, "E_irms_DM": r.E_irms_DM,
                    "P_rms_lt1R": r.P_rms_inner_lt_1R, "P_rms_2_4R": r.P_rms_outer_2_to_4R,
                    "E_rms_lt1R": r.E_rms_inner_lt_1R, "E_rms_2_4R": r.E_rms_outer_2_to_4R})
    for _, r in G[(G.prediction == "det") & (G.R_cut == 4) & (G["eval"].isin(["paint", "a2ii"]))].iterrows():
        irm.append({"source": "G", "model": r.model, "fit_target": r.fit_target, "objective": r.objective,
                    "R_fit": r.R_fit, "R_cut": r.R_cut, "P_irms_tSZ": r.profile_irms_tSZ,
                    "E_irms_kSZ": r.profile_irms_kSZ, "E_irms_DM": r.profile_irms_DM,
                    "rms_lt1R": r.profile_rms_inner_lt_1R, "rms_2_4R": r.profile_rms_outer_2_to_4R})
    pd.DataFrame(irm).to_csv(C.TABLES / "importance_weighted_profile_rms.csv", index=False, float_format="%.4g")
    C.write_json(C.PRIOR / "p2_summary.json", S)
    print("parity G vs F:", S["parity_G_P0E0_vs_F_grid_a_max_abs_pct_points"], "n", len(par))


if __name__ == "__main__":
    main()
