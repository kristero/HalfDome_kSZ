"""Test K fit audit (local, from the fetched results; no refit) -> results/k7_fit_audit.json.

  python -B k_fit_audit.py [--results DIR]

1. Convergence of every global fit (results/k3_fits.json): the least-squares status of every sigma_d iteration, and the
   sigma_d that makes chi^2/nu = 1 at the FINAL theta (k_fit_core.sigma_for_chi2 on the residuals of the stored theta).
   A fit counts as converged if scipy's least_squares returned success (status > 0, not max_nfev) in its final solve.
2. The 32-node compression error at the points that enter each fit (field x range):
   a. Test K's own K1 diagnostic comp_err_pub / comp_err_pert (ln[32-node model / exact mean over EVERY valid halo of
      the cell], for the published law and a perturbed law), restricted to the fit points;
   b. the same for each FITTED law: exact where the stored 256-halo subsample is the whole cell (sub_n = cell_N), and
      for larger cells the 32 strata of the subsample against its exact mean (a proxy, as in verify_k.py);
   c. the linear response of each fit to that error: dtheta = -(J_w^T J_w)^-1 J_w^T W delta (J_w = W J, W the fit's
      whitener at its sigma_d), in units of the Hessian sigma of each law parameter.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_common as KC  # noqa: E402
import k_fit_core as K  # noqa: E402
import k_fits as KF  # noqa: E402


def point_rows(C, field, r):
    """(cell row, aperture index) of every point of KF.dataset(field, r), in dataset order (replicates build_dataset)."""
    q = KC.FIELDS[field]["quantity"]
    cells, sel = KF.cell_dicts(C, field, r)
    out = []
    for c, s in zip(cells, sel):
        i = c["row"]
        for j in range(len(c["names"])):
            if c["nodes"][j] is None or j not in s:
                continue
            m = C[f"mean_{q}"][i][j]
            if np.isfinite(m) and m > 0:
                out.append((i, j))
    return out


def stats(v):
    v = np.abs(np.asarray(v, float))
    v = v[np.isfinite(v)]
    return {"n": int(len(v)), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "n_gt_0.2pct": int(np.sum(v > 0.002)), "n_gt_1pct": int(np.sum(v > 0.01))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args()
    R = Path(a.results)
    fits = json.loads((R / "k3_fits.json").read_text())
    C = KF.load_cells(R)
    files = sorted((R / "cells").glob("cells_*.npz"))
    S = {k: np.concatenate([np.load(f)[k] for f in files]) for k in ("sub_n", "sub_lnM", "sub_x", "sub_valid")}
    out = {"description": __doc__.strip(), "convergence": {}, "compression_k1_at_fit_points": {}, "compression_fitted": {},
           "linear_response": {}}
    # ---------------------------------------------------------------- 1. convergence
    for key, c in fits["configs"].items():
        field, r, p = key.split("|")
        D = KF.dataset(field, r, R)
        law = K.Law(KC.FIELDS[field]["form"], p)
        th = np.array(c["theta"], float)
        resid = D.lnd - K.predict(th, D, law)
        nu = D.n - law.n
        s_fin = K.sigma_for_chi2(D, resid, nu)
        chi2_rep = float(D.chi2_cells(resid, c["sigma_d"]).sum())
        out["convergence"][key] = {
            "success_final_solve": bool(c["success"]), "nfev_final_solve": int(c["nfev"]),
            "iterations": [{"sigma_d": h["sigma_d"], "nfev": h["nfev"], "status": h["status"]} for h in c.get("history", [])],
            "max_nfev": 600, "converged": bool(c["success"]),
            "sigma_d_reported": c["sigma_d"], "chi2_nu_at_reported_sigma_d": chi2_rep / nu,
            "chi2_nu_stored": c["chi2_nu"], "sigma_d_chi2nu1_at_final_theta": s_fin,
            "rel_diff_sigma_d": s_fin / c["sigma_d"] - 1,
            "bootstrap_status_counts": c.get("boot_status_counts"),
        }
    # ---------------------------------------------------------------- 2a. K1 diagnostic at the fit points
    for field, kf in (("P", 0), ("G", 1)):
        for r in KC.RANGES:
            rows = point_rows(C, field, r)
            D = KF.dataset(field, r, R)
            assert len(rows) == D.n, (field, r, len(rows), D.n)
            assert all(KC.AP_NAMES[j] == D.meta["point_name"][k] for k, (i, j) in enumerate(rows))
            pub = np.array([C["comp_err_pub"][i, j, kf] for i, j in rows])
            pert = np.array([C["comp_err_pert"][i, j, kf] for i, j in rows])
            both = np.maximum(np.abs(pub), np.abs(pert))
            big = [{"cell": KC.cell_label(int(C["snap"][i]), int(C["cell_bin"][i])), "z": float(C["z"][i]), "aperture": KC.AP_NAMES[j],
                    "N": int(C["cell_N"][i]), "sd_lnx": float(C["sd_lnx"][i, j]), "abs_ln_err": float(both[k])}
                   for k, (i, j) in enumerate(rows) if both[k] > 0.01]
            out["compression_k1_at_fit_points"][f"{field}|{r}"] = {
                "published": stats(pub), "perturbed": stats(pert), "max_of_both": stats(both), "points_gt_1pct": big,
                "cell_N_of_points_gt_0.2pct": [int(C["cell_N"][i]) for k, (i, j) in enumerate(rows) if both[k] > 0.002]}
    # ---------------------------------------------------------------- 2b/2c. fitted laws
    for key, c in fits["configs"].items():
        field, r, p = key.split("|")
        if field not in ("P", "G"):
            continue
        form = KC.FIELDS[field]["form"]
        D = KF.dataset(field, r, R)
        law = K.Law(form, p)
        th = np.array(c["theta"], float)
        rows = point_rows(C, field, r)
        delta = np.full(D.n, np.nan)
        exact_flag = np.zeros(D.n, bool)
        pub_check = []
        lawp = K.Law(form, "orig"); thp = lawp.start()
        for k, (i, j) in enumerate(rows):
            n = int(S["sub_n"][i])
            ok = S["sub_valid"][i, :n, j] & np.isfinite(S["sub_x"][i, :n, j])
            xj = S["sub_x"][i, :n, j][ok].astype(float); mj = S["sub_lnM"][i, :n][ok].astype(float)
            lz = math.log1p(float(C["z"][i]))
            full = n == int(C["cell_N"][i])
            if full:
                assert ok.sum() == int(C["nval"][i, j]), (key, i, j)
                w = C["node_w"][i, j]; m = w > 0
                nl, nx, nw = C["node_lnM"][i, j][m], C["node_x"][i, j][m], w[m] / w[m].sum()
            else:
                gs = K.strata_groups(mj, np.log(xj), KC.N_STRATA_M, KC.N_STRATA_X)
                nl = np.array([mj[g].mean() for g in gs]); nx = np.array([np.exp(np.log(xj[g]).mean()) for g in gs])
                nw = np.array([len(g) for g in gs], float) / len(xj)
            for lw, tt, tgt in ((law, th, "fit"), (lawp, thp, "pub")):
                if tgt == "pub" and not full:
                    continue
                ph = lw.phi(tt, mj, np.full(len(mj), lz))
                exact = np.mean(K.FIELD_NORM[field] * K.cumulative(xj, form, ph["amp"], ph["xc"], ph["alpha"], ph["beta"], ph["gamma"]))
                pn = lw.phi(tt, nl, np.full(len(nl), lz))
                approx = np.sum(nw * K.FIELD_NORM[field] * K.cumulative(nx, form, pn["amp"], pn["xc"], pn["alpha"], pn["beta"], pn["gamma"]))
                if tgt == "fit":
                    delta[k] = math.log(approx / exact); exact_flag[k] = full
                else:
                    pub_check.append(abs(math.log(approx / exact) - C["comp_err_pub"][i, j, 0 if field == "P" else 1]))
        out["compression_fitted"][key] = {"all": stats(delta), "exact_cells_N<=256": stats(delta[exact_flag]),
                                          "subsample_proxy_cells_N>256": stats(delta[~exact_flag]) if (~exact_flag).any() else None,
                                          "check_published_law_vs_stored_comp_err_pub_max_abs_diff": float(max(pub_check)) if pub_check else None}
        # linear response of the fit to the compression error
        lnm, J = K.predict(th, D, law, need_jac=True)
        W = D.whitener(c["sigma_d"])
        Jw = W @ J
        F = Jw.T @ Jw
        cov = np.linalg.pinv(F)
        dth = -cov @ (Jw.T @ (W @ delta))
        sig = np.sqrt(np.clip(np.diag(np.array(c["cov"])), 0, None))
        shift = dth / sig
        imax = int(np.argmax(np.abs(shift)))
        out["linear_response"][key] = {"max_abs_shift_hess_sigma": float(abs(shift[imax])), "param": law.names[imax],
                                       "shift_hess_sigma": {n: float(v) for n, v in zip(law.names, shift)},
                                       "delta_model_rms_ln": float(np.sqrt(np.mean((J @ dth) ** 2)))}
        print(key, out["compression_fitted"][key]["all"]["max"], out["linear_response"][key]["max_abs_shift_hess_sigma"], flush=True)
    KC.write_json(R / "k7_fit_audit.json", out)
    print("wrote", R / "k7_fit_audit.json")


if __name__ == "__main__":
    main()
