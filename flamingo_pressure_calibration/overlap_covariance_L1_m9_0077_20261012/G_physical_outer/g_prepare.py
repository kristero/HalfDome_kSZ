"""G stages 'inputs' and 'prepare'.

inputs   full : copy Test F's products (profiles.npz, truth.npz, truth_summary.json, importance_F.npz,
                models_main.npz) from F_observables/results into <results>/f_base (SHA256 recorded); F's basis is
                rebuilt by the 'kernels' stage (F's work/ was deleted on maas).
         smoke: regenerate them with Test F's own stages in smoke mode (profiles, kernels T, truth, importance),
                written to <results_smoke>/f_base only.
prepare  physics inputs and per-bin tables (results/prepare.npz, prepare.json):
         * normalisation checks: E/G per band (-> KAPPA_E), P / [(E/200)(T_M/T200)] per band (-> KAPPA_P);
         * xi_lin / b(M) spot values (reproduce inputs/make_xi_bias.py's printout);
         * 2-halo templates <b xi_lin(x R)> per calibrated bin at the basis-shell nodes;
         * measured / predicted 2-halo ratios per band (all gas and A2-ii; Ebar = Omega_m as in the brief, and
           Ebar = KAPPA_E Omega_m);
         * outer temperatures: T_M/T200, electron-weighted T_e/T200, pressure-implied T_P/T200 per band; and
           the brief's 'implied' temperature (P/P200) / [Omega_m (1 + b xi)/200];
         * R200m/R200c (NFW, SOAP c200c) per bin -> splashback and accretion-shock expectations;
         * p_keep fits (P, E); S1 / S2 scatter tables (main, a2ii).
"""
from __future__ import annotations

import shutil

import g_common as GC
import g_physics as PH
import numpy as np


def inputs():
    timer = GC.Timer()
    out = {"mode": GC.MODE, "f_base": str(GC.FBASE), "files": {}}
    if GC.SMOKE:
        import f0_profiles
        import f1_importance
        import f1_truth
        import f_kernels
        f0_profiles.main()
        out["kernels_timing"] = f_kernels.build(sets=("T",), workers=GC.WORKERS)
        f1_truth.main()
        f1_importance.main()
        out["how"] = "regenerated with Test F's stages in smoke mode (profiles, kernels T, truth, importance)"
    else:
        for name in GC.F_INPUTS:
            src = GC.F_RESULTS_DIR / name
            dst = GC.FBASE / name
            shutil.copyfile(src, dst)
            out["files"][name] = {"source": str(src), "sha256": GC.sha256(dst), "bytes": dst.stat().st_size}
        out["how"] = "copied from F_observables/results (Test F's maas run, 2026-10-12)"
    for name in ("profiles.npz", "truth.npz", "truth_summary.json", "importance_F.npz"):
        p = GC.FBASE / name
        out["files"].setdefault(name, {"sha256": GC.sha256(p), "bytes": p.stat().st_size})
    out["seconds"] = timer.mark("total")
    GC.write_json(GC.RESULTS / "inputs.json", out)
    GC.log("inputs done", out["seconds"])
    return out


def main():
    timer = GC.Timer()
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    bprof = np.load(GC.B_PROFILES)
    bhalo = np.load(GC.B_HALOS)
    lh = PH.LinearHalo()
    info, arrays = {"mode": GC.MODE}, {}
    # ---- consistency of B's per-halo arrays with F's (same rows, same profiles)
    rows = pr["halo_rows"]
    info["B_vs_F"] = {
        "orig_row_equal": bool(np.array_equal(bhalo["orig_row"][rows], rows)),
        "edges_equal": float(np.max(np.abs(bprof["edges"] - GC.F_EDGES))),
        "P_geo_vs_P_main_maxabs": float(np.nanmax(np.abs(bprof["P_geo"][rows] - pr["P_main"]))),
        "E_geo_vs_E_main_maxabs": float(np.nanmax(np.abs(bprof["E_geo"][rows] - pr["E_main"]))),
        "P_A2ii_vs_P_a2ii_maxabs": float(np.nanmax(np.abs(bprof["P_A2ii"][rows] - pr["P_a2ii"]))),
        "E_A2ii_vs_E_a2ii_maxabs": float(np.nanmax(np.abs(bprof["E_A2ii"][rows] - pr["E_a2ii"])))}
    # ---- normalisation checks
    norm = PH.normalisation_checks(pr, bprof, bhalo)
    arrays.update(norm_E_over_G=norm["E_over_G"], norm_P_over_ET=norm["P_over_ET"])
    outer = [k for k, (lo, hi) in enumerate(PH.BANDS) if lo >= 2.0]
    inner = [k for k, (lo, hi) in enumerate(PH.BANDS) if hi <= 1.0]
    w_bins = np.array([pr["weight"][pr["mass_bin"] == b].sum() for b in GC.CAL_BINS])
    kappa_e = float(np.mean(norm["E_over_G"][:, outer]))
    info["normalisation"] = {
        "mu_XGP_over_m_p": GC.MU_XGP_OVER_MP, "KAPPA_P_code": GC.KAPPA_P, "KAPPA_E_theory_X0.752": GC.KAPPA_E_THEORY,
        "Omega_b": GC.OMEGA_B, "f_b": GC.F_B, "Omega_m = Omega_b/f_b": GC.OMEGA_M,
        "E_over_G_bands": [f"{lo:g}-{hi:g}" for lo, hi in PH.BANDS],
        "E_over_G_range_inside_1R": [float(norm["E_over_G"][:, inner].min()), float(norm["E_over_G"][:, inner].max())],
        "E_over_G_range_2_4R": [float(norm["E_over_G"][:, outer].min()), float(norm["E_over_G"][:, outer].max())],
        "KAPPA_E_measured_2_4R": kappa_e,
        "P_over_ET_range_inside_1R": [float(norm["P_over_ET"][:, inner].min()), float(norm["P_over_ET"][:, inner].max())],
        "P_over_ET_range_all_bands": [float(norm["P_over_ET"].min()), float(norm["P_over_ET"].max())],
        "Ebar_brief_Omega_m": GC.OMEGA_M, "Ebar_used": kappa_e * GC.OMEGA_M,
        "note": "E = n_e mu_XGP/(rho_c f_b) with mu_XGP = 1.2632 m_p (XGPaint, includes 1/0.9): ionised gas has "
                "E/G = (1+X)/2 mu_XGP/m_p; the cosmic mean gas density is Omega_m in G units, so the mean electron "
                "density is KAPPA_E Omega_m (stars neglected). P200 = G M (200 rho_c) f_b/(2R) and kT200 = "
                "0.59 m_p G M/(2R) give P/P200 = KAPPA_P (E/200)(T_e/T200) with KAPPA_P = 0.59 m_p/mu_XGP."}
    ebar = kappa_e * GC.OMEGA_M
    # ---- 2-halo templates (all haloes of each bin)
    tpl = PH.bin_templates(lh, pr)
    arrays.update(T2h=tpl["T2h"], bin_b=tpl["b"], bin_logm=tpl["logm"], bin_r200=tpl["r200"])
    info["xi_bias"] = {"meta": lh.meta, "spot_values": lh.spot_values(), "omega_m_table": lh.omega_m}
    info["bins"] = {"labels": GC.CAL_LABELS, "mean_b": tpl["b"], "mean_logm": tpl["logm"], "mean_r200c_mpc": tpl["r200"],
                    "N": [int(np.sum(pr["mass_bin"] == b)) for b in GC.CAL_BINS], "b_pivot": float(lh.b(GC.M_PIVOT))}
    # ---- measured / predicted 2-halo ratios
    for sel in ("main", "a2ii"):
        r_brief, _ = PH.two_halo_ratios(pr, tpl, GC.OMEGA_M, sel)
        r_used, pred = PH.two_halo_ratios(pr, tpl, ebar, sel)
        arrays[f"ratio2h_brief_{sel}"] = r_brief
        arrays[f"ratio2h_{sel}"] = r_used
        arrays["E2h_shell_pred"] = pred
    # ---- temperatures
    for sel in ("main", "a2ii"):
        t = PH.temperature_profiles(pr, bprof, bhalo, sel)
        for k, v in t.items():
            arrays[f"temp_{k}_{sel}"] = v
    arrays["temp_shell_Te_main"] = PH.temperature_shells(pr, bprof, bhalo, "main")
    vol = np.diff(GC.F_EDGES**3)
    x, wq = PH.quad_nodes(GC.F_EDGES)
    T35 = np.sum(tpl["T2h"][:, :GC.N_F] * wq[None], axis=2)
    brief_T = np.zeros((GC.N_CAL, len(PH.BANDS)))
    for c, b in enumerate(GC.CAL_BINS):
        for k, (lo, hi) in enumerate(PH.BANDS):
            sh = PH.band_mask(lo, hi)
            brief_T[c, k] = (pr["meanP_main"][b][sh] @ vol[sh]) / (GC.OMEGA_M * ((1 + T35[c, sh]) @ vol[sh]) / 200.0)
    arrays["temp_brief_implied"] = brief_T
    arrays["temp_corrected_implied"] = brief_T * GC.OMEGA_M / (GC.KAPPA_P * ebar)
    # ---- R200m / R200c and the splashback / accretion-shock expectations
    ratio_m = PH.r200m_over_r200c(bhalo["concentration_200c"][rows])
    r_m, nu = np.zeros(GC.N_CAL), np.zeros(GC.N_CAL)
    for c, b in enumerate(GC.CAL_BINS):
        sel_rows = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][sel_rows]
        ok = np.isfinite(ratio_m[sel_rows])
        r_m[c] = np.sum(w[ok] * ratio_m[sel_rows][ok]) / np.sum(w[ok])
        nu[c] = float(np.sum(w * lh.nu_of(pr["m200c_msun"][sel_rows])) / np.sum(w))
    arrays.update(r200m_over_r200c=r_m, bin_nu=nu)
    info["splashback"] = {"R200m_over_R200c": r_m, "nu": nu, "x_sp_expected_1.0_1.5_R200m": [list(1.0 * r_m), list(1.5 * r_m)],
                          "x_shock_expected_1.5_2_x_sp": [list(1.5 * 1.0 * r_m), list(2.0 * 1.5 * r_m)]}
    # ---- p_keep
    cal = GC.CAL_BINS
    for f in ("P", "E"):
        fit = PH.fit_pkeep(pr[f"mean{f}_main"][cal], pr[f"mean{f}_a2ii"][cal], tpl["logm"], pr["neff_main"][cal])
        arrays[f"pkeep_params_{f}"] = fit["params"]
        arrays[f"pkeep_measured_{f}"] = fit["measured_ratio"]
        arrays[f"pkeep_smoothed_{f}"] = fit["smoothed_at_shells"]
        info[f"pkeep_{f}"] = {"params_f0_f1_l0_l1_n": fit["params"], "x_h_at_14.5": float(np.exp(fit["params"][2])),
                              "rms_rel_resid_per_bin": fit["rms_rel_per_bin"], "success": fit["success"]}
    # ---- scatter tables
    for sel in ("main", "a2ii"):
        s1, s2 = PH.scatter_tables(pr, sel)
        arrays[f"S1_{sel}"] = s1
        for rc, M in s2.items():
            arrays[f"S2_{sel}_Rc{rc:g}"] = M
        info[f"S2_{sel}"] = {f"Rc{rc:g}": [PH.s2_summary(M[c]) for c in range(GC.N_CAL)] for rc, M in s2.items()}
        info[f"S1_{sel}"] = {"sigma2_P": s1[:, 0], "sigma2_E": s1[:, 1], "x_PE": s1[:, 2]}
    # consistency with Test F's lognormal_main_4R (main selection)
    tr = np.load(GC.FBASE / "truth.npz")
    info["S1_vs_F_lognormal_main_4R_maxabs"] = float(np.max(np.abs(arrays["S1_main"] - tr["lognormal_main_4R"])))
    arrays.update(kappa_e=np.array(kappa_e), ebar=np.array(ebar), kappa_p=np.array(GC.KAPPA_P),
                  b_pivot=np.array(float(lh.b(GC.M_PIVOT))), omega_m=np.array(GC.OMEGA_M))
    np.savez_compressed(GC.RESULTS / "prepare.npz", **arrays)
    info["seconds"] = timer.mark("total")
    GC.write_json(GC.RESULTS / "prepare.json", info)
    GC.log("prepare done", info["seconds"])
    return info


if __name__ == "__main__":
    main()
