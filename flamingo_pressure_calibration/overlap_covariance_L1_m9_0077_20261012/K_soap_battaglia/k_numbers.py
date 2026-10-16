"""Key numbers of Test K for the report, each with its source (file / field) -> report/report_numbers.json.

  python -B k_numbers.py [--results DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_common as KC  # noqa: E402
import k_fit_core as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "report" / "report_numbers.json"))
    a = ap.parse_args()
    R = Path(a.results)
    N = {}
    fits = json.loads((R / "k3_fits.json").read_text())
    st = json.loads((R / "k1_status.json").read_text()) if (R / "k1_status.json").exists() else []
    if st:
        N["haloes_total"] = {"value": int(sum(s.get("n_selected") or 0 for s in st)), "source": "results/k1_status.json n_selected (sum)"}
        N["haloes_z0"] = {"value": [s.get("n_selected") for s in st if s["snap"] == 77], "source": "results/k1_status.json snap 77"}
        zs = sorted((s["z"], s["snap"], s.get("n_selected")) for s in st if s.get("z") is not None)
        if zs:
            N["z_range"] = {"value": [zs[0][0], zs[-1][0]], "source": "results/k1_status.json z"}
        N["outputs_with_selected"] = {"value": int(sum(1 for s in st if (s.get("n_selected") or 0) > 0)), "source": "results/k1_status.json"}
        N["per_output"] = {"value": {f"{s['snap']:04d}": [round(s.get("z", float("nan")), 3), s.get("n_selected")] for s in st}, "source": "results/k1_status.json"}
    for key, d in fits["datasets"].items():
        N[f"dataset|{key}"] = {"value": d, "source": f"results/k3_fits.json datasets[{key}]"}
    for key, p in fits["published"].items():
        N[f"published_sigma_d|{key}"] = {"value": p["sigma_d"], "source": f"results/k3_fits.json published[{key}].sigma_d"}
        N[f"published_rms|{key}"] = {"value": p["rms_resid"], "source": f"results/k3_fits.json published[{key}].rms_resid"}
    for key, c in fits["configs"].items():
        if "theta" not in c:
            continue
        N[f"sigma_d|{key}"] = {"value": c["sigma_d"], "source": f"results/k3_fits.json configs[{key}].sigma_d"}
        N[f"chi2_nu|{key}"] = {"value": c["chi2_nu"], "source": f"results/k3_fits.json configs[{key}].chi2_nu"}
        N[f"BIC|{key}"] = {"value": c["information"]["BIC"], "source": f"results/k3_fits.json configs[{key}].information.BIC (at the orig sigma_d)"}
        N[f"at_bound|{key}"] = {"value": c["at_bound"], "source": f"results/k3_fits.json configs[{key}].at_bound"}
        law_ = K.Law(KC.FIELDS[key.split("|")[0]]["form"], key.split("|")[2]); lo_, hi_ = law_.bounds(); th_ = np.array(c["theta"])
        N[f"near_bound|{key}"] = {"value": [n for i, n in enumerate(law_.names) if (np.isfinite(lo_[i]) and th_[i] - lo_[i] < 1e-3)
                                            or (np.isfinite(hi_[i]) and hi_[i] - th_[i] < 1e-3)],
                                  "source": f"results/k3_fits.json configs[{key}].theta within 1e-3 (fitted coordinate) of k_fit_core.Law.bounds()"}
    # profile ratio refitted / published at EXACTLY x = 0.1, 0.3, 0.5, 1, 2, 3 (k4_profile_ratios.json; not grid points)
    PR = json.loads((R / "k4_profile_ratios.json").read_text())
    for key, tab in PR["ratios"].items():
        N[f"profile_ratio_refit_over_published|{key}"] = {
            "value": {mz: {f"x{xx:g}": round(float(v), 4) for xx, v in zip(PR["x"], vals)} for mz, vals in tab.items()},
            "source": f"results/k4_profile_ratios.json ratios[{key}] (k_fit_core.law_profile at exactly x = {PR['x']})"}
    # the five profile parameters of every law on a mass x z grid and over its data domain (k4_law_values.json)
    LV = json.loads((R / "k4_law_values.json").read_text())
    for key, g in LV["grid"].items():
        N[f"law_values|{key}"] = {"value": {"log10_M200c": LV["log10_M200c"], "z": LV["z"], **g},
                                  "source": f"results/k4_law_values.json grid[{key}] (parameter[i_mass][i_z])"}
    for key, d in LV["data_domain"].items():
        N[f"law_domain|{key}"] = {"value": d, "source": f"results/k4_law_values.json data_domain[{key}] (min/max over the fitted cells)"}
    # published laws: normalised aperture integrals at R200c (1e14 Msun, z = 0)
    for fld, form in (("P", "B12"), ("G", "B16")):
        law = K.Law(form, "orig")
        ph = {q: float(np.ravel(v_)[0]) for q, v_ in K.law_profile_params(law, law.start(), 1e14, 0.0).items()}
        N[f"published_integral_R200c_1e14_z0|{fld}"] = {"value": float(K.FIELD_NORM[fld] * K.cumulative(1.0, form, *(ph[q] for q in K.QNAMES))),
                                                       "source": "k_fit_core: FIELD_NORM * cumulative(x = 1) of the published law at M200c = 1e14 Msun, z = 0 (Ytilde or Mtilde)"}
    # redshift coverage: highest-mass cell with >= MIN_CELL_HALOES haloes at the output nearest each z
    files = sorted((R / "cells").glob("cells_*.npz"))
    cz, cb, cn = [], [], []
    for f in files:
        d = np.load(f)
        cz.append(d["z"]); cb.append(d["cell_bin"]); cn.append(d["cell_N"])
    cz, cb, cn = np.concatenate(cz), np.concatenate(cb), np.concatenate(cn)
    uz = np.unique(cz)
    cov = {}
    for zt in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0):
        zn = float(uz[np.argmin(np.abs(uz - zt))])
        m = (np.abs(cz - zn) < 1e-9) & (cn >= KC.MIN_CELL_HALOES)
        if not m.any():
            cov[f"z{zt:g}"] = {"z_output": zn, "top_cell": None}
            continue
        i = np.nonzero(m)[0][np.argmax(cb[m])]
        cov[f"z{zt:g}"] = {"z_output": zn, "top_cell": f"{KC.MASS_EDGES[cb[i]]:.1f}-{KC.MASS_EDGES[cb[i] + 1]:.1f}", "N": int(cn[i]),
                           "n_cells_N>=10": int(m.sum())}
    N["coverage_top_cell_by_z"] = {"value": cov, "source": "results/cells/cells_*.npz (z, cell_bin, cell_N): highest 0.2-dex bin with cell_N >= 10 at the output nearest z"}
    N["cells_total"] = {"value": int(len(cn)), "source": "results/cells/cells_*.npz cell_N (all cells)"}
    N["cells_used"] = {"value": int(np.sum(cn >= KC.MIN_CELL_HALOES)), "source": "results/cells/cells_*.npz cell_N >= 10"}
    N["outputs_with_used_cells"] = {"value": int(len(np.unique(cz[cn >= KC.MIN_CELL_HALOES]))), "source": "results/cells/cells_*.npz"}
    N["z_max_used"] = {"value": float(cz[cn >= KC.MIN_CELL_HALOES].max()), "source": "results/cells/cells_*.npz"}
    # per output: selected haloes, cells (all / used), haloes in the used cells, mass range of the used cells and the
    # number of fit points of every field x range (user decision: report how many haloes and cells enter per output)
    pred_ = np.load(R / "k3_predictions.npz")
    lab = {f"{fr}|{r}": np.array([s[:4] for s in pred_[f"{fr}__{r}__cell"].astype(str)]) for fr in ("P", "G") for r in KC.RANGES}
    sel_ = {f"{s['snap']:04d}": s.get("n_selected") for s in st}
    zsn = {f"{s['snap']:04d}": s.get("z") for s in st}
    rows_, tot = [], {"cells_all": 0, "cells_used": 0, "haloes_in_used_cells": 0, **{f"points|{k}": 0 for k in lab}}
    for f in files:
        d = np.load(f)
        sn = f.stem.split("_")[1]
        use = d["cell_N"] >= KC.MIN_CELL_HALOES
        b = d["cell_bin"][use]
        row = {"snap": sn, "z": round(float(zsn.get(sn, d["z"][0] if len(d["z"]) else float("nan"))), 3), "haloes_selected": sel_.get(sn),
               "cells_all": int(len(d["cell_N"])), "cells_used": int(use.sum()), "haloes_in_used_cells": int(d["cell_N"][use].sum()),
               "used_log10M_range": [float(KC.MASS_EDGES[b.min()]), float(KC.MASS_EDGES[b.max() + 1])] if use.any() else None,
               **{f"points|{k}": int(np.sum(v == sn)) for k, v in lab.items()}}
        rows_.append(row)
        for k in tot:
            tot[k] += row[k]
    N["per_output_counts"] = {"value": {"rows": rows_, "totals": tot},
                              "source": "results/k1_status.json n_selected; results/cells/cells_<snap>.npz cell_N, cell_bin (used: cell_N >= 10); "
                                        "results/k3_predictions.npz <field>__<range>__cell (points per output)"}
    # aperture integrals law / SOAP summary (R3 evaluation dataset)
    p = R / "k4_aperture_integrals.csv"
    if p.exists():
        rows = list(csv.DictReader(open(p)))
        for q in ("Y500", "Y200", "Mgas200"):
            for model in ["published"] + [f"fit_{r}_{pz}" for r in KC.RANGES for pz in KC.PARAMETRIZATIONS]:
                v = np.array([float(rw[f"ratio_{model}"]) for rw in rows if rw["quantity"] == q and rw["eval_dataset"] == "R3" and rw.get(f"ratio_{model}")])
                if len(v):
                    N[f"aperture_ratio|{q}|{model}"] = {"value": {"median": float(np.median(v)), "p16": float(np.percentile(v, 16)), "p84": float(np.percentile(v, 84)),
                                                                  "min": float(v.min()), "max": float(v.max()), "n_cells": int(len(v))},
                                                        "source": f"results/k4_aperture_integrals.csv ratio_{model} (quantity {q}, eval_dataset R3)"}
            # by redshift band for the published law and the R2 orig fit
            for model in ("published", "fit_R2_orig", "fit_R3_orig", "fit_R3_brk_amp", "fit_R3_free"):
                for zlo, zhi in ((0, 0.25), (0.25, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 9)):
                    v = np.array([float(rw[f"ratio_{model}"]) for rw in rows if rw["quantity"] == q and rw["eval_dataset"] == "R3"
                                  and zlo <= float(rw["z"]) < zhi and rw.get(f"ratio_{model}")])
                    if len(v):
                        N[f"aperture_ratio_z|{q}|{model}|{zlo}-{zhi}"] = {"value": {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))},
                                                                          "source": f"results/k4_aperture_integrals.csv ratio_{model}, {zlo} <= z < {zhi}"}
    # rms of ln(law / SOAP) per aperture and regime (R3 evaluation datasets: every usable aperture)
    pred = np.load(R / "k3_predictions.npz")
    regimes = {"all": lambda z, m: np.ones(len(z), bool), "z<1,M>=10^13.5": lambda z, m: (z < 1) & (m >= 13.5),
               "z<1,M<10^13.5": lambda z, m: (z < 1) & (m < 13.5), "z>=1": lambda z, m: z >= 1}
    for fld in ("P", "G"):
        base = f"{fld}__R3"
        nm = pred[base + "__name"]; zz = pred[base + "__z"]; lm = pred[base + "__cell_lnM"] / math.log(10)
        models = [k for k in pred.files if k.startswith(base + "__fit_") or k == base + "__published"]
        for ap in ("SO/2500_crit", "SO/500_crit", "SO/200_crit", "SO/50_crit", "SO/5xR_500_crit", "IS/300kpc", "IS/1000kpc"):
            s_ = nm == ap
            if not s_.any():
                continue
            for k in models:
                r = pred[k][s_] - pred[base + "__lnd"][s_]
                tab = {}
                for rg, f in regimes.items():
                    m = f(zz[s_], lm[s_])
                    if m.any():
                        tab[rg] = {"rms": float(np.sqrt(np.mean(r[m] ** 2))), "mean": float(np.mean(r[m])), "n": int(m.sum())}
                N[f"aperture_ln_law_over_soap|{fld}|{ap}|{k.split('__')[-1]}"] = {
                    "value": tab, "source": f"results/k3_predictions.npz {k} - {base}__lnd at {base}__name == {ap}; regimes by cell z and mean log10 M200c"}
    # z = 0 particle stacks: A2-ii / all gas at 1-4 R200c (neighbour gas that SOAP's inclusive apertures count)
    pz = R / "k4_particle_z0.npz"
    if pz.exists():
        P_ = np.load(pz)
        e = P_["edges"]
        for fld in ("P", "G"):
            tab = {}
            for b, lab in enumerate(P_["bin_labels"]):
                a2, al = P_[f"{fld}_A2ii_mean"][b], P_[f"{fld}_all_mean"][b]
                for lo, hi in ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)):
                    sel = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
                    vol = np.diff(e ** 3)[sel]
                    tab.setdefault(str(lab), {})[f"{lo:g}-{hi:g}"] = float(np.sum(a2[sel] * vol) / np.sum(al[sel] * vol))
            N[f"particles_z0_A2ii_over_allgas|{fld}"] = {"value": tab, "source": f"results/k4_particle_z0.npz {fld}_A2ii_mean / {fld}_all_mean, volume-weighted over the fit-grid shells in each x interval"}
            # removed fraction 1 - A2ii/all over the bins 10^12.9-10^14.7 and over all bins
            for name, lo_m, hi_m in (("12.9-14.7", 12.9, 14.7), ("all_bins", -1, 99)):
                rng_tab = {}
                for xr in ("1-2", "2-3", "3-4"):
                    v = [1 - tab[str(lab)][xr] for lab in P_["bin_labels"]
                         if float(str(lab).split("-")[0]) >= lo_m - 1e-9 and float(str(lab).split("-")[1]) <= hi_m + 1e-9]
                    rng_tab[xr] = [float(min(v)), float(max(v))] if v else None
                N[f"particles_z0_A2ii_removed_range|{fld}|{name}"] = {
                    "value": rng_tab, "source": f"1 - particles_z0_A2ii_over_allgas|{fld} over the 0.2-dex bins in {name} (min, max)"}
    # statistical errors of the cell means and point/cell counts
    for fld in ("P", "G"):
        for r in KC.RANGES:
            base = f"{fld}__{r}"
            sd = pred[base + "__sd_boot"]
            nm = pred[base + "__name"].astype(str)
            N[f"sd_boot|{fld}|{r}"] = {"value": {"median": float(np.median(sd)), "p90": float(np.percentile(sd, 90)), "max": float(sd.max())},
                                       "source": f"results/k3_predictions.npz {base}__sd_boot (bootstrap sd of ln cell mean)"}
            cells_all = np.unique(pred[base + "__cell"])
            cells_2500 = np.unique(pred[base + "__cell"][nm == "SO/2500_crit"])
            N[f"cells_with_R2500c|{fld}|{r}"] = {"value": {"with": int(len(cells_2500)), "of": int(len(cells_all))},
                                                "source": f"results/k3_predictions.npz {base}__cell where {base}__name == SO/2500_crit"}
            isph = np.char.startswith(nm, "IS/")
            N[f"sphere_points|{fld}|{r}"] = {"value": {"n": int(isph.sum()), "max_xbar": float(pred[base + "__xbar"][isph].max()) if isph.any() else None},
                                            "source": f"results/k3_predictions.npz {base}__name / __xbar (physical-kpc spheres in the dataset)"}
    # bootstrap / Hessian error ratio, starts, correlations and law tables per fit
    for key, c in fits["configs"].items():
        if "theta" not in c:
            continue
        N[f"starts|{key}"] = {"value": {"winner": c.get("start"), "all": c.get("starts")}, "source": f"results/k3_fits.json configs[{key}].starts"}
        N[f"corr|{key}"] = {"value": {"names": c["names"], "corr": c["corr"]}, "source": f"results/k3_fits.json configs[{key}].corr"}
        N[f"table|{key}"] = {"value": c["table"], "source": f"results/k3_fits.json configs[{key}].table"}
        if not c.get("boot_sd"):
            continue
        h = np.sqrt(np.clip(np.diag(np.array(c["cov"])), 0, None))
        b = np.array(c["boot_sd"], float)
        ok = np.isfinite(b) & (h > 0)
        N[f"boot_over_hess|{key}"] = {"value": {"median": float(np.median(b[ok] / h[ok])), "min": float(np.min(b[ok] / h[ok])),
                                                "max": float(np.max(b[ok] / h[ok])), "n_boot_ok": c.get("boot_n_ok")},
                                      "source": f"results/k3_fits.json configs[{key}] boot_sd / sqrt(diag cov)"}
    # K2: per-cell x_c (pressure) and alpha (gas) at z = 0 vs mass
    if (R / "k2_cell_fits.json").exists():
        k2all = json.loads((R / "k2_cell_fits.json").read_text())["fits"]
        for fld, q in (("P", "xc"), ("G", "alpha")):
            for r in KC.RANGES:
                rows = sorted([(d["lnM_mean_nodes"] / math.log(10), d.get(q), bool(d.get("at_bound"))) for d in k2all
                               if d["field"] == fld and d["range"] == r and abs(d["z"]) < 1e-9 and d.get("ok") and q in d])
                N[f"k2_z0_{q}_vs_mass|{fld}|{r}"] = {"value": [[round(a, 3), b, c] for a, b, c in rows],
                                                     "source": f"results/k2_cell_fits.json fits (field {fld}, range {r}, z = 0): [log10 M (node mean), {q}, at a bound]"}
    info = json.loads((R / "k4_information.json").read_text()) if (R / "k4_information.json").exists() else {}
    for k, v in info.items():
        N[f"information|{k}"] = {"value": v, "source": f"results/k4_information.json {k}"}
    # K2 cell-fit summary
    if (R / "k2_cell_fits.json").exists():
        k2 = json.loads((R / "k2_cell_fits.json").read_text())["fits"]
        for f in ("P", "G"):
            for r in KC.RANGES:
                sub = [d for d in k2 if d["field"] == f and d["range"] == r]
                N[f"k2_cells|{f}|{r}"] = {"value": {"n": len(sub), "ok": int(sum(1 for d in sub if d.get("ok"))),
                                                     "at_bound": int(sum(1 for d in sub if d.get("at_bound")))},
                                          "source": f"results/k2_cell_fits.json fits (field {f}, range {r})"}
    # fit audit (k_fit_audit.py): convergence, 32-node compression error at the fit points and its effect on the fits
    au_p = R / "k7_fit_audit.json"
    if au_p.exists():
        au = json.loads(au_p.read_text())
        for key, c in au["convergence"].items():
            N[f"fit_convergence|{key}"] = {"value": {k_: c[k_] for k_ in ("converged", "nfev_final_solve", "chi2_nu_at_reported_sigma_d",
                                                                          "sigma_d_reported", "sigma_d_chi2nu1_at_final_theta", "rel_diff_sigma_d")}
                                                   | {"status_per_iteration": [it["status"] for it in c["iterations"]]},
                                           "source": f"results/k7_fit_audit.json convergence[{key}]"}
        N["fits_not_converged"] = {"value": sorted(k for k, c in au["convergence"].items() if not c["converged"]),
                                   "source": "results/k7_fit_audit.json convergence[*].converged (scipy least_squares success of the final solve)"}
        for key, c in au["compression_k1_at_fit_points"].items():
            N[f"compression_k1_at_fit_points|{key}"] = {"value": {k_: c[k_] for k_ in ("published", "perturbed", "max_of_both", "points_gt_1pct")}
                                                        | {"max_cell_N_of_points_gt_0.2pct": max(c["cell_N_of_points_gt_0.2pct"]) if c["cell_N_of_points_gt_0.2pct"] else None},
                                                        "source": f"results/k7_fit_audit.json compression_k1_at_fit_points[{key}] (|comp_err_pub|, |comp_err_pert| at the fit points)"}
        for key, c in au["compression_fitted"].items():
            N[f"compression_fitted|{key}"] = {"value": c, "source": f"results/k7_fit_audit.json compression_fitted[{key}]"}
            N[f"compression_linear_response|{key}"] = {"value": au["linear_response"][key], "source": f"results/k7_fit_audit.json linear_response[{key}]"}
        lr = au["linear_response"]
        kmax = max(lr, key=lambda k_: lr[k_]["max_abs_shift_hess_sigma"])
        N["compression_linear_response_max"] = {"value": {"key": kmax, "max_abs_shift_hess_sigma": lr[kmax]["max_abs_shift_hess_sigma"], "param": lr[kmax]["param"],
                                                          "max_delta_model_rms_ln": max(v_["delta_model_rms_ln"] for v_ in lr.values())},
                                                "source": "results/k7_fit_audit.json linear_response (max over the 24 P/G fits)"}
        cf = au["compression_fitted"]
        N["compression_fitted_max"] = {"value": {"p99_max": max(v_["all"]["p99"] for v_ in cf.values()), "max": max(v_["all"]["max"] for v_ in cf.values()),
                                                 "pub_check_max_abs_diff": max(v_["check_published_law_vs_stored_comp_err_pub_max_abs_diff"] for v_ in cf.values())},
                                       "source": "results/k7_fit_audit.json compression_fitted (max over the 24 P/G fits)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(N, indent=1, default=float) + "\n")
    print("wrote", a.out, len(N), "entries")


if __name__ == "__main__":
    main()
