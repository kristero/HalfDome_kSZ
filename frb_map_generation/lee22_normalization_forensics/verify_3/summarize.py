"""verify_3: collect the key C4 numbers from the verify_3 outputs into one JSON."""
import json, os
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3"
c4 = json.load(open(os.path.join(O, "c4_checks.json")))
pop = json.load(open(os.path.join(O, "pop22_anchor.json")))
kz = json.load(open(os.path.join(O, "konietzka_lowz_anchor.json")))
fig = json.load(open(os.path.join(O, "fig_reference_curves.json")))
z0 = c4["z0_table"]
s = {
 "Tnw_over_T200_z0_range_13.0-14.5": [min(r["Tnw_over_T200_0_1"] for r in z0), max(r["Tnw_over_T200_0_1"] for r in z0)],
 "Mgas_over_fbM200_z0_range": [min(r["Mgas_over_fbM200"] for r in z0), max(r["Mgas_over_fbM200"] for r in z0)],
 "local_gas_over_fb_total_at_R200_range": [min(r["local_gas_over_fb_total_at_R200_lee22c"] for r in z0), max(r["local_gas_over_fb_total_at_R200_lee22c"] for r in z0)],
 "FnHSE_vol0.2-1_thermal_range": [min(r["FnHSE_vol_0p2_1_lee22c_fnt0.0"] for r in z0), max(r["FnHSE_vol_0p2_1_lee22c_fnt0.0"] for r in z0)],
 "FnHSE_vol0.2-1_fnt0.15_range": [min(r["FnHSE_vol_0p2_1_lee22c_fnt0.15"] for r in z0), max(r["FnHSE_vol_0p2_1_lee22c_fnt0.15"] for r in z0)],
 "FnHSE_after_a3_a2_vs_z": {f"{r['log10Mh']}|z={r['z']}": r["FnHSE_vol_0p2_1_after_a3"] for r in c4["z_dependence"] if r["pressure"] == "P_x(1+z)^2"},
 "Ysph500_Lee_over_UPP": {f"{r['mass']}|{r['c']}": r["Ysph500_Lee_over_UPP"] for r in c4["pressure_z0"]},
 "Pop22_F_T_ce_range": [min(r["F_T_ce"] for r in pop["pop22_rows"]), max(r["F_T_ce"] for r in pop["pop22_rows"])],
 "Pop22_F_gas500_range": [min(r["F_gas500"] for r in pop["pop22_rows"]), max(r["F_gas500"] for r in pop["pop22_rows"])],
 "Pop22_TX_over_T200_range": [min(r["TX_over_T200_Pop22"] for r in pop["pop22_rows"]), max(r["TX_over_T200_Pop22"] for r in pop["pop22_rows"])],
 "Konietzka_13to14_zs0.1": kz["mass13to14|zs=0.1"],
 "TNG_literature_hmf_weighted_fgas_R200c_13to14": [pop[k] for k in pop if k.startswith("hmf_weighted")],
 "fig5_cyan_over_B16printed_noFb": [fig["fig5_p0|M=bin centre as Msun|B16 exponent printed"], fig["fig5_p1|M=bin centre as Msun|B16 exponent printed"]],
 "fig2_cyan_over_B12Pth": [fig["fig2_p0|M=bin centre as Msun|B12 P_th"], fig["fig2_p1|M=bin centre as Msun|B12 P_th"]],
 "sim_stack_Tnw_over_T200": {k: v for k, v in fig.items() if k.startswith("sim_p") and "Tnw" in k},
}
json.dump(s, open(os.path.join(O, "c4_verify_summary.json"), "w"), indent=1)
print(json.dumps(s, indent=1)[:3000])
