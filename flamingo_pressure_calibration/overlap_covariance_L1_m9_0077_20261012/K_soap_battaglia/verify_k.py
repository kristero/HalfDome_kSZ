"""Test K verification (local, from the fetched results) -> verification.json.

  python -B verify_k.py [--results DIR] [--out FILE]
Checks: K0 inventory complete; K1 every output read, SO radii physical (M200c / (4 pi R200c^3 / 3) = 200 rho_c(z) from
the header H(z)), f_b, fill condition; 32-node quadrature error (published, perturbed and every fitted law; at the fit
points and its linear effect on the fits, from results/k7_fit_audit.json); K2 / K3 convergence (non-converged fits
flagged in k_params.json and on the page), parameters at bounds, bootstrap completeness; per-output counts; K5 parity
files; synthetic recovery self-test; result size and SHA256SUMS.  Run after report_k.py (the page is checked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_common as KC  # noqa: E402
import k_fit_core as K  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def integral_at_nodes(R, fits, n_top=40, n_rand=40, seed=3):
    """k_fit_core.cumulative at the final parameters of every fit vs adaptive quadrature (scipy quad in ln x, dense
    breakpoints across the transition at x_c, epsrel 1e-12), at the n_top nodes with the largest ln s = alpha ln(x/x_c)
    of the fit's dataset and n_rand random nodes."""
    from scipy.integrate import IntegrationWarning, quad
    import k_fits as KF
    warnings.simplefilter("ignore", IntegrationWarning)     # quad's roundoff notes near the 1e-12 target; the result is checked
    warnings.simplefilter("ignore", RuntimeWarning)
    rng = np.random.default_rng(seed)
    out = {}
    for key, c in fits["configs"].items():
        if "theta" not in c:
            continue
        field, r, p = key.split("|")
        D = KF.dataset(field, r, R)
        law = K.Law(KC.FIELDS[field]["form"], p)
        ph = law.phi(np.array(c["theta"]), D.node_lnM, D.node_lz)
        x = D.node_x2
        ls = ph["alpha"] * np.log(x / ph["xc"])
        idx = np.unique(np.r_[np.argsort(ls)[-n_top:], rng.choice(len(x), size=min(n_rand, len(x)), replace=False)])
        worst, wls = 0.0, None
        n_bad_gamma = int(np.sum(ph["gamma"] <= -3.0))            # model undefined there (k_fit_core returns +inf)
        for i in idx:
            amp, xc, al, be, ga = (float(ph[q][i]) for q in K.QNAMES)
            if ga <= -3.0:
                continue
            got = float(K.cumulative(x[i], law.form, amp, xc, al, be, ga))
            f = lambda u: float(K.profile(math.exp(u), law.form, amp, xc, al, be, ga)) * math.exp(3 * u)   # noqa: E731
            uc, ux = math.log(xc), math.log(x[i])
            lo = ux - 60.0 / (3.0 + ga)
            pts = sorted({lo, ux} | {uc + k / al for k in np.arange(-12, 13, 0.5) if lo < uc + k / al < ux}
                         | set(np.linspace(lo, ux, 40).tolist()))
            ref = sum(quad(f, pts[j], pts[j + 1], epsabs=0, epsrel=1e-12, limit=200)[0] for j in range(len(pts) - 1))
            e = abs(got / ref - 1)
            if e > worst:
                worst, wls = e, float(ls[i])
        out[key] = {"n_nodes": int(len(idx)), "max_rel_err": worst, "ln_s_at_worst": wls, "max_ln_s_in_dataset": float(ls.max()),
                    "nodes_with_gamma_le_-3": n_bad_gamma, "min_gamma": float(np.min(ph["gamma"])), "max_alpha": float(np.max(ph["alpha"]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "verification.json"))
    a = ap.parse_args()
    R = Path(a.results)
    v = {"checks": {}, "numbers": {}}
    ck = v["checks"]

    inv = json.loads((R / "k0_inventory.json").read_text()) if (R / "k0_inventory.json").exists() else None
    if inv:
        files = inv["files"]
        ck["K0_71_files"] = len(files) == 71 and not inv["missing_files"]
        ck["K0_no_missing_columns"] = all(not f["missing_columns"] for f in files.values())
        v["numbers"]["redshifts"] = {s: f["Redshift"] for s, f in files.items()}
        v["numbers"]["soap_TB"] = sum(f["bytes"] for f in files.values()) / 1e12
        v["numbers"]["smoke_read_0077"] = inv.get("smoke_read_0077")
        filt = {s: f.get("heating_filter", {}).get("a_limit") for s, f in files.items()}
        v["numbers"]["agn_filter_a_limit_range"] = [min(x for x in filt.values() if x), max(x for x in filt.values() if x)]
    st = json.loads((R / "k1_status.json").read_text()) if (R / "k1_status.json").exists() else []
    if st:
        ck["K1_all_outputs_ok"] = all(s["ok"] for s in st)
        rr = [s["checks"].get("rho200_over_rhocrit_H_median") for s in st if s.get("checks")]
        rr = [x for x in rr if x is not None]
        v["numbers"]["rho200_over_rhocrit_H_median_range"] = [min(rr), max(rr)] if rr else None
        ck["K1_SO_radii_physical (|rho200/200rho_c(z)-1| < 1e-3)"] = bool(rr) and max(abs(x - 1) for x in rr) < 1e-3
        fb = [s["checks"]["f_b_header_incl_nu"] for s in st if s.get("checks")]
        if fb:
            v["numbers"]["f_b_header_range"] = [min(fb), max(fb)]
            ck["K1_f_b_matches_header"] = max(abs(x - KC.F_B) for x in fb) < 1e-4
        low = {f"{s['snap']:04d}": s["checks"].get("Nbound_min") for s in st if s.get("n_selected") and (s["checks"].get("Nbound_min") or 0) < 100}
        v["numbers"]["outputs_with_centrals_below_100_bound_particles (min Nbound)"] = low
        vf = [min(s["checks"]["n_valid_frac_per_aperture"]) for s in st if s.get("n_selected")]
        v["numbers"]["min_valid_fraction_any_aperture_any_output"] = min(vf)
        # unfilled apertures (SORadius = 0; spheres of subhaloes with < 100 bound particles) are masked per halo
        ck["K1_unfilled_apertures_masked (valid fraction >= 0.999 everywhere)"] = min(vf) >= 0.999
        v["numbers"]["n_selected_per_output"] = {f"{s['snap']:04d}": s.get("n_selected") for s in st}
        v["numbers"]["n_selected_total"] = int(sum(s.get("n_selected") or 0 for s in st))
        v["numbers"]["read_seconds_total"] = float(sum(s.get("seconds_read") or 0 for s in st))
        v["numbers"]["z0_study_rows_found"] = [s.get("z0_study_rows_found") for s in st if s["snap"] == 77]
    cells = sorted((R / "cells").glob("cells_*.npz"))
    if cells:
        ce, cp, ncell, nuse = [], [], 0, 0
        for f in cells:
            d = np.load(f)
            ncell += len(d["cell_N"]); nuse += int(np.sum(d["cell_N"] >= KC.MIN_CELL_HALOES))
            for key, lst in (("comp_err_pub", ce), ("comp_err_pert", cp)):
                x = d[key]
                use = (d["cell_N"] >= KC.MIN_CELL_HALOES)[:, None, None] & np.isfinite(x)
                lst.append(np.abs(x[use]))
        ce, cp = np.concatenate(ce), np.concatenate(cp)
        v["numbers"]["cells_total"] = ncell
        v["numbers"]["cells_with_N>=10"] = nuse
        v["numbers"]["compression_abs_ln_err_published"] = {"max": float(ce.max()), "p99": float(np.percentile(ce, 99)), "median": float(np.median(ce))}
        v["numbers"]["compression_abs_ln_err_perturbed"] = {"max": float(cp.max()), "p99": float(np.percentile(cp, 99)), "median": float(np.median(cp))}
        ck["K1_compression_p99_below_0.2pct (published / perturbed law, every aperture of the used cells)"] = float(np.percentile(np.r_[ce, cp], 99)) < 0.002
        au_p = R / "k7_fit_audit.json"
        if au_p.exists():
            au = json.loads(au_p.read_text())
            cf = au["compression_k1_at_fit_points"]
            big = {(b["cell"], b["aperture"], b["N"], round(b["sd_lnx"], 3)) for c in cf.values() for b in c["points_gt_1pct"]}
            v["numbers"]["compression_at_fit_points"] = {k: c["max_of_both"] for k, c in cf.items()}
            v["numbers"]["compression_note"] = (
                "32 nodes vs the exact mean over every valid halo (results/k7_fit_audit.json): over all apertures of the used cells "
                f"median {np.median(ce):.1e}, p99 {np.percentile(ce, 99):.2%} (published law) / {np.percentile(cp, 99):.2%} (perturbed); at the R3 fit points "
                f"max(|pub|, |pert|) has p99 {cf['P|R3']['max_of_both']['p99']:.2%} (P) / {cf['G|R3']['max_of_both']['p99']:.2%} (G) and max "
                f"{max(c['max_of_both']['max'] for c in cf.values()):.2%}; {cf['P|R3']['max_of_both']['n_gt_0.2pct']} of {cf['P|R3']['max_of_both']['n']} P and "
                f"{cf['G|R3']['max_of_both']['n_gt_0.2pct']} of {cf['G|R3']['max_of_both']['n']} G points exceed 0.2%; the points above 1% are "
                + "; ".join(f"{c} {a} (N = {n}, sd ln x = {s_})" for c, a, n, s_ in sorted(big)) + ". Propagated linearly, the compression error of "
                f"each fitted law shifts no law parameter by more than {max(x['max_abs_shift_hess_sigma'] for x in au['linear_response'].values()):.2f} Hessian sigma.")
            v["numbers"]["compression_linear_response_max_shift_hess_sigma"] = {k: x["max_abs_shift_hess_sigma"] for k, x in au["linear_response"].items()}
            v["numbers"]["compression_fitted_laws_at_fit_points"] = {k: x["all"] for k, x in au["compression_fitted"].items()}
            v["numbers"]["compression_published_law_recomputed_vs_stored_max_abs_diff"] = max(
                x["check_published_law_vs_stored_comp_err_pub_max_abs_diff"] for x in au["compression_fitted"].values())
            ck["K7_compression_recomputed_matches_stored (published law, N <= 256 cells, < 1e-6)"] = \
                v["numbers"]["compression_published_law_recomputed_vs_stored_max_abs_diff"] < 1e-6
            ck["K7_compression_shifts_below_0.5_sigma (linear response, 24 P/G fits)"] = \
                max(x["max_abs_shift_hess_sigma"] for x in au["linear_response"].values()) < 0.5
            conv = au["convergence"]
            v["numbers"]["fits_not_converged"] = {k: {"chi2_nu_at_quoted_sigma_d": c["chi2_nu_at_reported_sigma_d"], "sigma_d_quoted": c["sigma_d_reported"],
                                                      "sigma_d_chi2nu1_at_final_theta": c["sigma_d_chi2nu1_at_final_theta"],
                                                      "status_per_iteration": [it["status"] for it in c["iterations"]]}
                                                  for k, c in conv.items() if not c["converged"]}
            ck["K7_converged_fits_have_chi2nu_1 (|chi2/nu - 1| < 1e-5)"] = all(abs(c["chi2_nu_at_reported_sigma_d"] - 1) < 1e-5 for c in conv.values() if c["converged"])
            kp = json.loads((R / "k_params.json").read_text())
            page = (HERE / "report" / "fit_soap_battaglia.txt")
            ptxt = page.read_text() if page.exists() else ""
            ck["K7_nonconverged_fits_flagged (k_params.json fit_status, page)"] = all(
                (not kp["fits"][k]["fit_status"]["converged"]) and (not kp["fits"][k]["fit_status"]["usable"]) and "not converged" in ptxt
                for k in v["numbers"]["fits_not_converged"])
            pv_p = R / "k7_params_provenance.json"
            if pv_p.exists():
                pv = json.loads(pv_p.read_text())
                orig = json.loads((HERE / pv["maas_copy"]).read_text())
                same = all({kk: vv for kk, vv in kp["fits"][k].items() if kk != "fit_status"} == f for k, f in orig["fits"].items()) and \
                    {kk: vv for kk, vv in kp["conventions"].items() if kk != "fit_status"} == orig["conventions"] and set(kp["fits"]) == set(orig["fits"])
                ck["K7_k_params_maas_fields_unchanged (local fit_status / electron_factors added)"] = same and sha(HERE / pv["maas_copy"]) == pv["maas_sha256"] \
                    and sha(R / "k_params.json") == pv["local_sha256"]
                v["numbers"]["k_params_provenance"] = pv
        else:
            v["numbers"]["compression_note"] = "results/k7_fit_audit.json missing: run k_fit_audit.py"
        v["numbers"]["results_MB"] = sum(p.stat().st_size for p in R.rglob("*") if p.is_file() and "work" not in p.parts) / 1e6
        ck["results_below_300MB"] = v["numbers"]["results_MB"] < 300
    fits = json.loads((R / "k3_fits.json").read_text()) if (R / "k3_fits.json").exists() else None
    if fits:
        conf = fits["configs"]
        ck["K3_all_configs_fitted"] = all("theta" in c for c in conf.values())
        v["numbers"]["K3_at_bound"] = {k: c.get("at_bound") for k, c in conf.items() if c.get("at_bound")}
        v["numbers"]["K3_boot_ok"] = {k: c.get("boot_n_ok") for k, c in conf.items()}
        req = fits.get("n_boot_requested", {"orig": 100, "other": 40, "check": 20})
        need = lambda k: (req["orig"] if k.endswith("|orig") else req["other"]) if k.split("|")[0] in ("P", "G") else req["check"]  # noqa: E731
        ck["K3_bootstrap_complete(>=90%)"] = all((c.get("boot_n_ok") or 0) >= 0.9 * need(k) for k, c in conf.items())
        v["numbers"]["K3_sigma_d"] = {k: c.get("sigma_d") for k, c in conf.items()}
        v["numbers"]["published_sigma_d"] = {k: p.get("sigma_d") for k, p in fits["published"].items()}
        # compression for the fitted laws on the stored subsamples (exact 256-halo mean vs 16 strata of the same haloes)
        worst = {}
        if cells:
            d = np.load(cells[-1])          # z = 0 output
            dz = np.load(cells[len(cells) // 2])
            for key, c in conf.items():
                if "theta" not in c:
                    continue
                field, r, p = key.split("|")
                form = KC.FIELDS[field]["form"]; fld = "P" if form == "B12" else "G"
                law = K.Law(form, p); th = np.array(c["theta"])
                w = 0.0
                for dd in (d, dz):
                    for ci in range(len(dd["cell_N"])):
                        n = int(dd["sub_n"][ci])
                        if n < 32:
                            continue
                        lnM = dd["sub_lnM"][ci, :n].astype(float)
                        zz = float(dd["z"][ci])
                        for j in range(KC.N_SO if fld == "P" else KC.N_AP):
                            ok = dd["sub_valid"][ci, :n, j] & np.isfinite(dd["sub_x"][ci, :n, j])
                            if ok.sum() < 32:
                                continue
                            xj = dd["sub_x"][ci, :n, j][ok].astype(float); mj = lnM[ok]
                            ph = law.phi(th, mj, np.full(len(mj), math.log1p(zz)))
                            exact = np.mean(K.FIELD_NORM[fld] * K.cumulative(xj, form, ph["amp"], ph["xc"], ph["alpha"], ph["beta"], ph["gamma"]))
                            gs = K.strata_groups(mj, np.log(xj), KC.N_STRATA_M, KC.N_STRATA_X)
                            nm = np.array([mj[g].mean() for g in gs]); nx = np.array([np.exp(np.log(xj[g]).mean()) for g in gs])
                            ww = np.array([len(g) for g in gs], float) / len(xj)
                            pn = law.phi(th, nm, np.full(len(nm), math.log1p(zz)))
                            approx = np.sum(ww * K.FIELD_NORM[fld] * K.cumulative(nx, form, pn["amp"], pn["xc"], pn["alpha"], pn["beta"], pn["gamma"]))
                            w = max(w, abs(math.log(approx / exact)))
                worst[key] = w
            v["numbers"]["compression_fitted_laws_max_abs_ln"] = worst
            ck["compression_fitted_laws_below_0.3pct (256-halo subsamples)"] = max(worst.values()) < 0.003 if worst else None
    # the aperture integral at the nodes of every fitted law (incl. the largest s = (x/x_c)^alpha of each dataset) vs quadrature
    if fits:
        v["numbers"]["integral_at_fitted_nodes"] = integral_at_nodes(R, fits)
        ck["integral_at_fitted_nodes(<1e-9)"] = max(d["max_rel_err"] for d in v["numbers"]["integral_at_fitted_nodes"].values()) < 1e-9
        ck["no_fitted_node_with_gamma_le_-3"] = all(d["nodes_with_gamma_le_-3"] == 0 for d in v["numbers"]["integral_at_fitted_nodes"].values())
    # sphere points within each range's x cut (k_common.RANGES sphere_xmax)
    pred_path = R / "k3_predictions.npz"
    if pred_path.exists():
        pr = np.load(pred_path)
        worst = {}
        okc = True
        for field in ("G", "Gcore"):
            for r, spec in KC.RANGES.items():
                nm = pr[f"{field}__{r}__name"].astype(str)
                xb = pr[f"{field}__{r}__xbar"]
                isph = np.char.startswith(nm, "IS/")
                mx = float(xb[isph].max()) if isph.any() else None
                worst[f"{field}|{r}"] = {"n_sphere_points": int(isph.sum()), "max_mean_x": mx, "cut": spec["xmax"]}
                okc &= (mx is None) or (mx <= spec["xmax"] + 1e-12)
        v["numbers"]["sphere_points_vs_cut"] = worst
        ck["sphere_points_within_range_cuts (1 / 1.8 / 3.3)"] = bool(okc)
    rn_p = HERE / "report" / "report_numbers.json"
    if rn_p.exists() and fits:
        po = json.loads(rn_p.read_text()).get("per_output_counts", {}).get("value")
        if po:
            t = po["totals"]
            ck["per_output_counts_sum_to_totals (cells, points per field x range)"] = (
                t["cells_used"] == v["numbers"].get("cells_with_N>=10") and len(po["rows"]) == 71
                and all(t[f"points|{k}"] == fits["datasets"][k]["n_points"] for k in ("P|R1", "P|R2", "P|R3", "G|R1", "G|R2", "G|R3")))
    p1 = R / "k1_provenance_check.json"
    if p1.exists():
        d1 = json.loads(p1.read_text())
        ck["K1_provenance_recheck_identical (3 outputs, final code)"] = bool(d1.get("all_core_arrays_identical"))
        v["numbers"]["K1_provenance_recheck"] = {k: {kk: vv for kk, vv in o.items() if kk != "non_identical"} for k, o in d1.get("outputs", {}).items()}
    ps = HERE / "report" / "xgpaint_snippet_check.json"
    if ps.exists():
        d2 = json.loads(ps.read_text())
        ck["xgpaint_snippet_runs_and_matches (<1e-7)"] = bool(d2.get("passed"))
        v["numbers"]["xgpaint_snippet_max_rel_diff"] = d2.get("max_rel_diff")
    for name in ("k5_parity_local.json", "k5_selftest.json"):
        p = R / name
        if p.exists():
            d = json.loads(p.read_text())
            if name.startswith("k5_parity"):
                for k in ("integral", "shells", "xgpaint", "particles"):
                    if k in d:
                        ck[f"K5_{k}"] = d[k].get("pass")
                v["numbers"]["K5_xgpaint_max_rel_diff"] = d.get("xgpaint", {}).get("max_rel_diff")
                v["numbers"]["K5_integral_max_rel_err"] = d.get("integral", {}).get("max_rel_err")
                v["numbers"]["K5_particles_max_abs_median_dev"] = d.get("particles", {}).get("max_abs_median_dev_above_13.5")
            else:
                v["numbers"]["selftest"] = d
                ck["selftest_orig_profile_recovery(<5%)"] = all(d[f"orig_recovery_{f}"]["max_rel_profile_err"] < 0.05 for f in ("P", "G"))
    if (R / "SHA256SUMS").exists():
        bad = []
        for line in (R / "SHA256SUMS").read_text().splitlines():
            if not line.strip():
                continue
            h, rel = line.split("  ", 1)
            p = R / rel
            if not p.exists() or sha(p) != h:
                bad.append(rel)
        ck["SHA256SUMS_match"] = not bad
        v["numbers"]["SHA256SUMS_mismatch"] = bad
    v["all_pass"] = all(x is True for x in ck.values() if x is not None)
    Path(a.out).write_text(json.dumps(v, indent=1, default=float) + "\n")
    print(json.dumps(ck, indent=1))
    print("all_pass", v["all_pass"])


if __name__ == "__main__":
    main()
