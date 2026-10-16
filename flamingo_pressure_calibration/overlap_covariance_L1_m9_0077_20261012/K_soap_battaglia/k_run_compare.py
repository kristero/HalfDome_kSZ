"""Run 2 (first report, archived in attempts/run2_results_20261015) vs run 3 (current results): sigma_d, chi^2/nu,
points and parameter shifts per fit (local, small) -> results/k6_run2_vs_run3.json.

  python -B k_run_compare.py
Run 3 differs from run 2 by (i) the corrected aperture integral (k_fit_core.cumulative: exact 1 - t for large
(x/x_c)^alpha; +inf for gamma <= -3), (ii) gas sphere cuts mean x <= 1.8 / 3.3 instead of 1.9 / 3.6 (R2 / R3), (iii)
extra start vectors (k_starts_run2.json), (iv) 20 instead of 12 bootstrap refits for brk_amp / brk_all / free.
Shifts are (theta_3 - theta_2) / sigma_3 with sigma_3 the run-3 Hessian error (law parameters in their fitted
coordinates: ln A, alpha_m, alpha_z, ...).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    old = json.loads((HERE / "attempts" / "run2_results_20261015" / "results" / "k3_fits.json").read_text())
    new = json.loads((HERE / "results" / "k3_fits.json").read_text())
    out = {"description": __doc__.strip().splitlines()[0], "configs": {}, "datasets": {}}
    for key, d in new["datasets"].items():
        o = old["datasets"].get(key, {})
        out["datasets"][key] = {"n_points_run2": o.get("n_points"), "n_points_run3": d["n_points"],
                                "n_cells_run2": o.get("n_cells"), "n_cells_run3": d["n_cells"]}
    for key, c in new["configs"].items():
        o = old["configs"].get(key)
        if not o or "theta" not in c or "theta" not in o:
            continue
        t2, t3 = np.array(o["theta"]), np.array(c["theta"])
        s3 = np.sqrt(np.clip(np.diag(np.array(c["cov"])), 1e-300, None))
        sh = (t3 - t2) / s3
        i = int(np.argmax(np.abs(sh)))
        out["configs"][key] = {"sigma_d_run2": o["sigma_d"], "sigma_d_run3": c["sigma_d"],
                               "n_points_run2": o["n_points"], "n_points_run3": c["n_points"],
                               "max_abs_shift_hess_sigma": float(np.abs(sh[i])), "param_of_max_shift": c["names"][i],
                               "shift_hess_sigma": dict(zip(c["names"], np.round(sh, 2).tolist())),
                               "start_run3": c.get("start"), "at_bound_run2": o.get("at_bound"), "at_bound_run3": c.get("at_bound")}
    # the run-2 integral (archived code) vs the corrected one: the verifier's example and the nodes of the run-2 free gas fits
    import importlib.util
    import math
    import sys
    sys.path.insert(0, str(HERE))
    import k_fit_core as Knew
    import k_fits as KF
    spec = importlib.util.spec_from_file_location("k_fit_core_run2", HERE / "attempts" / "run2_k2k3_20261015_code" / "k_fit_core.py")
    Kold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Kold)
    ex = []
    for al in (10.0, 11.6):
        for be in (2.8, 2.95, 3.01, 3.2):
            xc, ga = 0.24, -0.55
            with np.errstate(all="ignore"):
                o = float(Kold.cumulative(13.75 * xc, "B16", 1.0, xc, al, be, ga))
            n = float(Knew.cumulative(13.75 * xc, "B16", 1.0, xc, al, be, ga))
            ex.append({"alpha": al, "beta": be, "gamma": ga, "x_over_xc": 13.75, "old_over_new_minus_1": (o / n - 1) if np.isfinite(o) else None})
    out["old_integral_example_B16"] = ex
    nodes = {}
    for key in ("G|R1|free", "G|R2|free", "G|R3|free"):
        f, r, p = key.split("|")
        D = KF.dataset(f, r, HERE / "results")
        law = Knew.Law("B16", p)
        th2 = np.array(old["configs"][key]["theta"])
        ph = law.phi(th2, D.node_lnM, D.node_lz)
        args = (D.node_x2, "B16", ph["amp"], ph["xc"], ph["alpha"], ph["beta"], ph["gamma"])
        with np.errstate(all="ignore"):
            o = Kold.cumulative(*args)
        n = Knew.cumulative(*args)
        ok = np.isfinite(n) & (n > 0)
        rel = np.where(ok & np.isfinite(o), np.abs(o / np.where(ok, n, 1.0) - 1), np.inf)
        nodes[key] = {"nodes": int(len(n)), "nodes_gamma_le_-3": int(np.sum(ph["gamma"] <= -3)), "min_gamma": float(ph["gamma"].min()),
                      "max_alpha": float(ph["alpha"].max()), "nodes_rel_err_gt_1e-6": int(np.sum(rel[ok] > 1e-6)),
                      "max_rel_err_finite_nodes": float(np.max(rel[ok & np.isfinite(o)])) if np.any(ok & np.isfinite(o)) else None,
                      "nodes_nonfinite_old": int(np.sum(ok & ~np.isfinite(o)))}
    out["run2_free_gas_thetas_at_run3_nodes"] = nodes
    out["run2_free_gas_note"] = ("run-2 theta of the free gas fits evaluated on the run-3 datasets (sphere cuts 1.8 / 3.3) with the run-2 "
                                 "(archived) and the corrected integral; nodes with gamma <= -3 have an infinite true integral")
    (HERE / "results" / "k6_run2_vs_run3.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["configs"].items():
        print(f"{k:16s} sigma_d {v['sigma_d_run2']:.4f} -> {v['sigma_d_run3']:.4f}  max shift {v['max_abs_shift_hess_sigma']:7.2f} "
              f"({v['param_of_max_shift']})  start {v['start_run3']}")


if __name__ == "__main__":
    main()
