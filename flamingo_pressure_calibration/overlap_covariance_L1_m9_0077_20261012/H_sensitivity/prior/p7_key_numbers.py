"""P7: collect the headline numbers quoted in the 'prior results' section into one table with their sources.

Reads only tables written by p1-p5 (and, through them, the earlier tests' stored results).
Output: tables/prior_key_numbers.csv, prior_summary.json.
"""
from __future__ import annotations

import json

import pandas as pd

import prior_common as C

T = C.TABLES
rows = []


def add(name, value, source, note=""):
    rows.append({"name": name, "value": value, "source": source, "note": note})


def main():
    p1 = json.load(open(C.PRIOR / "p1_summary.json"))
    for o in ("yytot_100", "yytot_1000", "yytot_3000", "ksz1h_100", "ksz1h_1000", "ksz1h_3000", "dm_mean", "dm_var",
              "dm_q16", "dm_q50", "dm_q999", "ydmtot_P_1000", "ydmtot_E_1000", "ymean"):
        v = p1[o]
        add(f"{o}: share inside 1R / 1-2R / 2-4R [%]",
            f"{100 * v['inside_1R']:.1f} / {100 * v['1_2R']:.1f} / {100 * v['2_4R']:.1f}",
            "tables/imp_bands_F.csv (from F_observables/results/importance_F.npz)")
        r = v["radii_all"]
        add(f"{o}: R50 / R90 / R95 [R200c], all masses",
            f"{r['R50']:.2f} / {r['R90']:.2f} / {r['R95']:.2f}", "tables/imp_radii_F.csv")
        add(f"{o}: dI/dlnx in the last shell (3.63-4R) / peak", f"{r['last_shell_dI_dlnx_over_peak']:.2f}",
            "tables/imp_radii_F.csv", "1.00 = still at its maximum at the 4R edge")
        add(f"{o}: calibrated-mass share [%]", f"{100 * v['calibrated_share']:.0f}", "tables/imp_bands_F.csv")
        add(f"{o}: dominant engine bin (share %)", f"{v['dominant_bin']} ({100 * v['dominant_bin_share']:.0f})",
            "p1_summary.json")
    add("quantile importance maps: sum-rule deviation (max)", f"{p1['sum_rule_max_abs_dev']:.4f}", "p1_summary.json",
        "DM quantile maps (Mecke, 64-halo subsample) sum to 0.990-1.002; bands renormalised")
    rad = pd.read_csv(T / "imp_radii_F.csv")
    for o in ("yytot_1000", "ksz1h_1000", "dm_mean", "dm_q999"):
        q = rad[(rad.observable == o) & ~rad.masses.isin(["all", "calibrated 13.5-15.5"])]
        add(f"{o}: R90 range over calibrated bins (13.5-13.7 -> 15.3-15.5) [R200c]",
            f"{q.R90.iloc[0]:.2f} -> {q.R90.iloc[-1]:.2f}", "tables/imp_radii_F.csv")
    bud = pd.read_csv(T / "budget_bands_F.csv")
    for qn in ("Y(<4R)", "Ne(<4R)"):
        q = bud[bud.quantity == qn]
        add(f"{qn}: share inside 2R over calibrated bins [%]",
            f"{100 * q.inside_2R.min():.0f}-{100 * q.inside_2R.max():.0f}", "tables/budget_bands_F.csv")
        add(f"{qn}: d ln I(<x)/d ln x at 4R over calibrated bins",
            f"{q['dlnI_dlnx_last_shell_3.63_4R'].min():.2f}-{q['dlnI_dlnx_last_shell_3.63_4R'].max():.2f}",
            "tables/budget_bands_F.csv")
    F = pd.read_csv(T / "grid_F_B12B16_long.csv")
    d = F[(F.objective == "a") & (F.prediction == "deterministic")]
    for rf, rc in ((2, 4), (4, 4), (1, 4), (4, 2), (4, 3)):
        q = d[(d.R_fit == rf) & (d.R_cut == rc)].iloc[0]
        add(f"B12/B16 objective a, (R_fit,R_cut)=({rf},{rc}): tSZ/kSZ/DM/tSZxDM RMS vs truth [%]",
            f"{q.rms_tSZ:.1f} / {q.rms_kSZ:.1f} / {q.rms_DM:.1f} / {q.rms_tSZxDM:.1f}", "tables/grid_F_B12B16_long.csv")
        add(f"B12/B16 objective a, ({rf},{rc}): vs truth mean [%]",
            f"{q.rms_tSZ_vs_mean:.1f} / {q.rms_kSZ_vs_mean:.1f} / {q.rms_DM_vs_mean:.1f} / "
            f"{q.rms_tSZxDM_vs_mean:.1f}", "tables/grid_F_B12B16_long.csv")
    for pred in ("lognormal_4R", "measured_2nd_moment"):
        q = F[(F.objective == "a") & (F.prediction == pred) & (F.R_cut == 4)]
        for fam in ("tSZ", "kSZ", "DM"):
            i = q[f"rms_{fam}"].idxmin()
            add(f"B12/B16 objective a, R_cut=4, prediction {pred}: best R_fit for {fam} (RMS %)",
                f"{q.loc[i, 'R_fit']:g} ({q.loc[i, f'rms_{fam}']:.2f}); at R_fit=4: "
                f"{float(q[q.R_fit == 4][f'rms_{fam}'].iloc[0]):.2f}", "tables/grid_F_B12B16_long.csv")
    Cc = pd.read_csv(T / "grid_F_controls_B12B16_long.csv")
    a2 = Cc[Cc.selection == "a2ii"]
    for rf in (2, 4):
        q = a2[(a2.R_fit == rf) & (a2.R_cut == 4)].iloc[0]
        add(f"B12/B16 refit to the A2-ii stacks, ({rf},4): tSZ/kSZ/DM/tSZxDM RMS vs A2-ii truth [%]",
            f"{q.rms_tSZ:.1f} / {q.rms_kSZ:.1f} / {q.rms_DM:.1f} / {q.rms_tSZxDM:.1f}",
            "tables/grid_F_controls_B12B16_long.csv")
        add(f"B12/B16 on A2-ii, ({rf},4): profile RMS P <1R/1-2R/2-4R; E <1R/1-2R/2-4R [%]",
            f"{q.P_rms_inner_lt_1R:.1f}/{q.P_rms_1_to_2R:.1f}/{q.P_rms_outer_2_to_4R:.1f}; "
            f"{q.E_rms_inner_lt_1R:.1f}/{q.E_rms_1_to_2R:.1f}/{q.E_rms_outer_2_to_4R:.1f}",
            "tables/grid_F_controls_B12B16_long.csv")
    cs = pd.read_csv(T / "b_cut_support_summary.csv")
    for spec in ("ALL_P", "ALL_E", "A2ii_P", "A2ii_E"):
        q = cs[(cs.spec == spec) & (cs.objective == "c")]
        add(f"{spec}: GLS model-discrepancy sigma_d median [%] at cuts 1/1.34/2/3/4",
            " / ".join(f"{q[q.cut == c].sigma_d_median.iloc[0]:.2f}" for c in (1, 1.34, 2, 3, 4)),
            "tables/b_cut_support_summary.csv (B_covariance/tables/b5_cut_support.csv)")
    mc = pd.read_csv(T / "model_comparison_4R_S1.csv")
    for _, r in mc.iterrows():
        add(f"Test G 4/4 S1, {r.family}: {r.model} [{r.target}]", f"{r.rms:.2f}", "tables/model_comparison_4R_S1.csv")
    sel = pd.read_csv(T / "lee22_ladder_selected.csv")
    for _, r in sel[sel.dset == "mass"].iterrows():
        add(f"Lee22 ladder selected ({r.field}, R_fit={r.R_fit:g})",
            f"{r.model}, k={r.k}, chi2/nu(floor)={r.chi2nu_floor:.2f}, log10 Mcut={r.log10Mcut:.2f}",
            "tables/lee22_ladder_selected.csv")
    sch = pd.read_csv(T / "binning_scheme_impacts.csv")
    for _, r in sch.iterrows():
        add(f"binning scheme {r.scheme} vs M200c_0.2 (stack level): yy ell=1000 / kSZ ell=1000 / mean DM [%]",
            f"{r.tSZ_yy_total_ell1000_pct:+.2f} / {r.kSZ_1h_ell1000_pct:+.2f} / {r.DM_mean_pct:+.2f}",
            "tables/binning_scheme_impacts.csv (stacking_choices mass_binning/res_schemes.json)")
    pd.DataFrame(rows).to_csv(T / "prior_key_numbers.csv", index=False)
    C.write_json(C.PRIOR / "prior_summary.json", {"key_numbers": rows})
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
