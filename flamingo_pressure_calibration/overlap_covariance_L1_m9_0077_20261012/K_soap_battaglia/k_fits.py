"""Test K stages K2 (per-cell free fits) and K3 (global law fits + bootstrap), from results/cells/cells_*.npz.

Datasets (k_fit_core.Dataset), one per field x fit range:
  field P    : ln mean Ytilde_noAGN at the usable SO apertures of every cell with >= 10 centrals
  field G    : ln mean Mtilde at the usable SO apertures and physical spheres (>= 100 kpc and mean x >= 0.1)
  field Pagn : as P with Ytilde including recently AGN-heated gas (check; 'orig' only)
  field Gcore: as G with the spheres >= 30 kpc down to 0.025 R200c (central-galaxy gas; check, 'orig' only)
  ranges R1 / R2 / R3 (k_common.RANGES); usability and de-duplication in k_read.usable_apertures / select_points.
K2: k_fit_core.fit_cells (3 free parameters per cell, published fixed shape, C_boot only) for P and G in each range.
K3: k_fit_core.fit_global for each field x range x parametrization (orig, brk_amp, brk_all, free; Pagn: orig) with
    sigma_d (chi^2/nu = 1), Hessian covariance, bootstrap refits at fixed sigma_d (100 for orig, 40 for the other
    parametrisations, 20 for the checks; run_test_k.py --boot overrides); starts: the published law (orig), the orig
    fit (other parametrisations; three break masses for brk_*), plus any extra start vectors of $K_STARTS (best
    sigma_d wins; every start's result is recorded under 'starts'); the published law scored on
    the same data; AIC / BIC at the orig fit's sigma_d; predictions of every fit (and of the published law) at every
    usable point of every range (residual maps, extrapolation).
Outputs: results/k2_cell_fits.json, results/k3_fits.json, results/k3_boot.npz, results/k3_predictions.npz,
results/k_params.json (machine-readable parameter file).
"""
from __future__ import annotations

import json
import math
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import k_common as KC
import k_fit_core as K
import k_read as KR

_CACHE = {}


def load_cells(results=None):
    """Concatenated cell arrays of all outputs (sorted by snapshot)."""
    results = Path(results or KC.RESULTS)
    files = sorted((results / "cells").glob("cells_*.npz"))
    parts = [dict(np.load(f)) for f in files]
    keep = ("cell_bin", "cell_N", "cell_mean_lnM", "cell_median_lnM", "z", "snap", "mean_Yn", "mean_Ya", "mean_Mg",
            "boot_Yn", "boot_Ya", "boot_Mg", "nval", "xbar", "med_ngas", "frac_valid", "node_lnM", "node_x", "node_w",
            "sd_Yn", "sd_Mg", "hmask_mean_Yn", "hmask_mean_Mg", "comp_err_pub", "comp_err_pert", "shell_P", "shell_G",
            "x_p16", "x_p84", "sd_lnx")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keep if all(k in p for p in parts)}
    return out


def cell_dicts(C, field, rng_name):
    """Cell dicts in k_fit_core.build_dataset format, with the selected point indices of each cell."""
    q = KC.FIELDS[field]["quantity"]
    cells, sel = [], []
    R = KC.RANGES[rng_name]
    for i in range(len(C["cell_N"])):
        if C["cell_N"][i] < KC.MIN_CELL_HALOES:
            continue
        stats = {"xbar": C["xbar"][i], "frac_valid": C["frac_valid"][i], "med_ngas": C["med_ngas"][i]}
        use = KR.usable_apertures(stats, field)
        keep = KR.select_points(C["xbar"][i], use, names_allowed=R["so"], xmax=R["xmax"])
        nodes = []
        for j in range(KC.N_AP):
            w = C["node_w"][i, j]
            m = w > 0
            nodes.append({"lnM": C["node_lnM"][i, j][m], "x1": np.zeros(m.sum()), "x2": C["node_x"][i, j][m], "w": w[m]}
                         if m.any() else None)
        cells.append({"z": float(C["z"][i]), "kind": 0, "mean": C[f"mean_{q}"][i], "boot_mean": C[f"boot_{q}"][i],
                      "nodes": nodes, "xbar": C["xbar"][i], "names": list(KC.AP_NAMES), "n_halo": int(C["cell_N"][i]),
                      "mean_lnM": float(C["cell_mean_lnM"][i]), "cell_label": KC.cell_label(int(C["snap"][i]), int(C["cell_bin"][i])),
                      "row": i})
        sel.append(set(keep))
    return cells, sel


def dataset(field, rng_name, results=None):
    key = (field, rng_name)
    if key not in _CACHE:
        C = _CACHE.setdefault("cells", load_cells(results))
        cells, sel = cell_dicts(C, field, rng_name)
        lut = {c["cell_label"]: s for c, s in zip(cells, sel)}
        D = K.build_dataset(cells, KC.DATA_FIELD[field], lambda c, j: j in lut[c["cell_label"]])
        _CACHE[key] = D
    return _CACHE[key]


def configs():
    out = []
    for field in ("P", "G"):
        for r in KC.RANGES:
            for p in KC.PARAMETRIZATIONS:
                out.append((field, r, p))
    for r in KC.RANGES:
        out.append(("Pagn", r, "orig"))
        out.append(("Gcore", r, "orig"))
    return out


def _form(field):
    return KC.FIELDS[field]["form"]


def task_k2(args):
    field, r = args
    D = dataset(field, r)
    res = K.fit_cells(D, _form(field))
    out = []
    labels = D.meta["cell_label"]
    for item, lab in zip(res, labels):
        d = {"cell": lab, "field": field, "range": r, "n_points": item["n_points"], "z": item["z"],
             "lnM_mean_nodes": item["lnM_mean_nodes"], "ok": item.get("ok", False)}
        if item.get("ok") is not None and "theta" in item:
            th = item["theta"]; sd = np.sqrt(np.clip(np.diag(item["cov"]), 0, None))
            law = K.Law(_form(field), "cell", 1)
            for i, n in enumerate(law.names):
                q = n.split(":")[0]
                d[q] = K.QSIGN[q] * math.exp(th[i])
                d[q + "_err_ln"] = float(sd[i])
            d["chi2"] = item["chi2"]
            d["at_bound"] = item["at_bound"]
        else:
            d["reason"] = item.get("reason")
        out.append(d)
    return out


def task_fit(args):
    field, r, p = args
    t0 = time.time()
    done = Path(KC.RESULTS) / "work" / f"fit_{field}_{r}_{p}.json"
    if KC.RESUME and done.exists():                # resumed K3: the nominal fit of an earlier invocation
        prev = json.loads(done.read_text())
        prev["resumed"] = True
        return prev
    D = dataset(field, r)
    law = K.Law(_form(field), p)
    orig = None
    f0 = Path(KC.RESULTS) / "work" / f"fit_{field}_{r}_orig.json"
    if p != "orig" and f0.exists():               # start from the orig fit of the same field and range
        prev = json.loads(f0.read_text())
        orig = dict(zip(prev["names"], prev["theta"]))
    if p in ("brk_amp", "brk_all"):
        starts = [(f"orig_fit, log10 M_b = {lmb}", start_from_orig(law, orig, lmb)) for lmb in (13.3, 13.8, 14.3)]
    else:
        starts = [("orig_fit" if p != "orig" else "published", start_from_orig(law, orig, 14.0))]
    for label, th in extra_starts(f"{field}|{r}|{p}", law):
        starts.append((label, th))
    best, scores = None, []
    for label, st in starts:
        try:
            fit = K.fit_global(D, law, theta0=st)
        except Exception as exc:   # noqa: BLE001
            fit = {"error": repr(exc)}
        if "error" in fit:
            scores.append({"start": label, "error": fit["error"]})
            continue
        scores.append({"start": label, "sigma_d": fit["sigma_d"], "chi2": fit["chi2"], "nfev": fit["nfev"]})
        if best is None or fit["sigma_d"] < best["sigma_d"] - 1e-6 or (abs(fit["sigma_d"] - best["sigma_d"]) <= 1e-6 and fit["chi2"] < best["chi2"]):
            best = fit
            best["start"] = label
    if best is None:
        return {"config": [field, r, p], "error": "all starts failed", "starts": scores}
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best.items() if k not in ("lnm", "resid")}
    out["config"] = [field, r, p]
    out["n_starts"] = len(starts)
    out["starts"] = scores
    out["seconds"] = time.time() - t0
    (Path(KC.RESULTS) / "work").mkdir(parents=True, exist_ok=True)
    KC.write_json(Path(KC.RESULTS) / "work" / f"fit_{field}_{r}_{p}.json", out)
    return out


def extra_starts(key, law):
    """Additional start vectors for a configuration from the JSON file named by $K_STARTS (run_test_k.py --starts):
    {"starts": {key: {"names": [...], <label>: theta, ...}}}; used only when the parameter names match."""
    path = os.environ.get("K_STARTS")
    if not path or not Path(path).exists():
        return []
    entry = json.loads(Path(path).read_text()).get("starts", {}).get(key)
    if not entry or list(entry.get("names", [])) != list(law.names):
        return []
    lo, hi = law.bounds()
    return [(f"{Path(path).name}:{label}", np.clip(np.array(th, float), lo, hi)) for label, th in entry.items() if label != "names"]


def start_from_orig(law, orig, lmb):
    """Start vector of `law` from an orig fit ({name: value} or None = published): pl laws copied, bpl laws with
    Q_b = A (M_b/1e14)^alpha_m and both slopes = alpha_m, break at 10^lmb, alpha_break = 0."""
    th = law.start()
    src = orig if orig is not None else dict(zip(K.Law(law.form, "orig").names, K.Law(law.form, "orig").start()))
    for q in K.QNAMES:
        if f"{q}:lnA" not in src:
            continue
        lnA, am, az = src[f"{q}:lnA"], src[f"{q}:am"], src[f"{q}:az"]
        k = law.qkind[q]
        if k == "pl":
            th[law.index[f"{q}:lnA"]], th[law.index[f"{q}:am"]], th[law.index[f"{q}:az"]] = lnA, am, az
        elif k == "bpl":
            th[law.index[f"{q}:lnQb"]] = lnA + am * (lmb - 14.0) * math.log(10)
            th[law.index[f"{q}:alo"]] = th[law.index[f"{q}:ahi"]] = am
            th[law.index[f"{q}:az"]] = az
    if "log10Mb" in law.index:
        th[law.index["log10Mb"]] = lmb
    return th


def task_boot(args):
    field, r, p, which = args
    chunk = Path(KC.RESULTS) / "work" / f"boot_{field}_{r}_{p}_{which[0]:03d}-{which[-1]:03d}.npz"
    if KC.RESUME and chunk.exists():
        z = np.load(chunk)
        return (field, r, p, list(which), z["B"], z["status"])
    D = dataset(field, r)
    law = K.Law(_form(field), p)
    fit = json.loads((Path(KC.RESULTS) / "work" / f"fit_{field}_{r}_{p}.json").read_text())
    fit["theta"] = np.array(fit["theta"])
    B, status = K.bootstrap_global(D, law, fit, which=which, return_status=True)
    np.savez(chunk, B=B, status=status)
    return (field, r, p, list(which), B, status)


def run_pool(fn, tasks, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    with get_context("fork").Pool(workers, maxtasksperchild=8) as pool:
        return pool.map(fn, tasks, chunksize=1)


def stage_k2(workers):
    t0 = time.time()
    tasks = [(f, r) for f in ("P", "G") for r in KC.RANGES]
    res = run_pool(task_k2, tasks, workers)
    flat = [d for part in res for d in part]
    KC.write_json(Path(KC.RESULTS) / "k2_cell_fits.json", {"fits": flat, "seconds": time.time() - t0,
                                                            "method": "k_fit_core.fit_cells: 3 free parameters per cell "
                                                            "(B12: P0, x_c, beta; B16: rho0, alpha, beta), published fixed "
                                                            "shape, covariance C_boot + sigma_0^2 (no sigma_d), 3 starts"})
    KC.log(f"K2 done: {len(flat)} cell fits in {time.time() - t0:.0f} s")


def stage_k3(workers):
    t0 = time.time()
    cfg = configs()
    # 1) orig first (start values for the others), then the rest
    first = [c for c in cfg if c[2] == "orig"]
    rest = [c for c in cfg if c[2] != "orig"]
    fits = {}
    for part in (first, rest):
        for res in run_pool(task_fit, part, workers):
            fits["|".join(res["config"])] = res
    KC.log(f"K3 nominal fits done in {time.time() - t0:.0f} s")
    # 2) bootstrap refits
    nb = KC.N_BOOT_FIT
    btasks = []
    for c in cfg:
        if "error" in fits["|".join(c)]:
            continue
        n = (nb if c[2] == "orig" else KC.N_BOOT_OTHER) if c[0] in ("P", "G") else KC.N_BOOT_CHECK
        for chunk in np.array_split(np.arange(n), max(1, n // 10)):
            btasks.append(c + (tuple(int(i) for i in chunk),))
    boot, bstat = {}, {}
    for field, r, p, which, B, status in run_pool(task_boot, btasks, workers):
        key = f"{field}|{r}|{p}"
        boot.setdefault(key, {})
        for i, row, s_ in zip(which, B, status):
            boot[key][i] = row
            bstat.setdefault(key, {})[i] = int(s_)
    bootarr = {k: np.array([v[i] for i in sorted(v)]) for k, v in boot.items()}
    np.savez_compressed(Path(KC.RESULTS) / "k3_boot.npz", **{k.replace("|", "__"): v for k, v in bootarr.items()})
    KC.log(f"K3 bootstrap done in {time.time() - t0:.0f} s")
    # 3) tables, published scores, information criteria, predictions everywhere
    summary = {"configs": {}, "published": {}, "datasets": {}}
    pred = {}
    for field in ("P", "G", "Pagn", "Gcore"):
        for r in KC.RANGES:
            D = dataset(field, r)
            summary["datasets"][f"{field}|{r}"] = {"n_points": D.n, "n_cells": D.n_cells,
                                                  "z_range": [float(D.z.min()), float(D.z.max())] if D.n else None,
                                                  "points_per_aperture": {n: int(np.sum(D.meta["point_name"] == n)) for n in np.unique(D.meta["point_name"])},
                                                  "cells_per_output": int(D.n_cells)}
            lawp = K.Law(_form(field), "orig")
            ev = K.evaluate_fixed(D, lawp, lawp.start())
            summary["published"][f"{field}|{r}"] = {k: v for k, v in ev.items() if k not in ("lnm", "resid")}
    for c in cfg:
        key = "|".join(c)
        f = fits[key]
        if "error" in f:
            summary["configs"][key] = f
            continue
        field, r, p = c
        D = dataset(field, r)
        law = K.Law(_form(field), p)
        th = np.array(f["theta"]); cov = np.array(f["cov"])
        B = bootarr.get(key)
        ref = fits.get(f"{field}|{r}|orig", f)
        resid = D.lnd - K.predict(th, D, law)
        ic = K.information_criteria(D, resid, ref["sigma_d"], law.n)
        tab = K.law_table(law, th, cov, B)
        item = dict(f)
        st_ = np.array([bstat.get(key, {}).get(i, -9) for i in sorted(bstat.get(key, {}))])
        item.update(boot_status_counts={str(int(u)): int(np.sum(st_ == u)) for u in np.unique(st_)} if len(st_) else {})
        item.update(table=tab, information=ic, corr=K.correlation(cov).tolist(),
                    boot_n_ok=int(np.sum(np.all(np.isfinite(B), axis=1))) if B is not None else 0,
                    boot_sd=(np.nanstd(B, 0, ddof=1).tolist() if B is not None and len(B) > 1 else None),
                    boot_p16=(np.nanpercentile(B, 16, 0).tolist() if B is not None and len(B) > 1 else None),
                    boot_p84=(np.nanpercentile(B, 84, 0).tolist() if B is not None and len(B) > 1 else None))
        summary["configs"][key] = item
    # predictions at every usable point (R3 datasets hold every usable aperture of R1/R2 as well, except where the
    # de-duplication kept a different aperture; evaluate on each range's dataset and on R3)
    for field in ("P", "G", "Pagn", "Gcore"):
        for r_eval in KC.RANGES:
            D = dataset(field, r_eval)
            base = f"{field}__{r_eval}"
            pred[base + "__lnd"] = D.lnd
            pred[base + "__z"] = D.z
            pred[base + "__cell"] = np.array(D.meta["cell_label"])[D.cell]
            pred[base + "__cell_lnM"] = np.asarray(D.meta["cell_lnM"])[D.cell]
            pred[base + "__cell_n"] = np.asarray(D.meta["cell_n"])[D.cell]
            pred[base + "__name"] = D.meta["point_name"]
            pred[base + "__xbar"] = D.meta["point_xbar"]
            pred[base + "__sd_boot"] = np.nanstd(D.boot, axis=1) if D.boot is not None else np.full(D.n, np.nan)
            lawp = K.Law(_form(field), "orig")
            pred[base + "__published"] = K.predict(lawp.start(), D, lawp)
            for c in cfg:
                if c[0] != field or "error" in fits["|".join(c)]:
                    continue
                law = K.Law(_form(field), c[2])
                pred[base + f"__fit_{c[1]}_{c[2]}"] = K.predict(np.array(fits["|".join(c)]["theta"]), D, law)
    np.savez_compressed(Path(KC.RESULTS) / "k3_predictions.npz", **pred)
    summary["seconds"] = time.time() - t0
    summary["n_boot_requested"] = {"orig": KC.N_BOOT_FIT, "other": KC.N_BOOT_OTHER, "check": KC.N_BOOT_CHECK}
    summary["ranges"] = {r: {"label": v["label"], "so": list(v["so"]), "sphere_xmax": v["xmax"]} for r, v in KC.RANGES.items()}
    summary["starts_file"] = os.environ.get("K_STARTS")
    KC.write_json(Path(KC.RESULTS) / "k3_fits.json", summary)
    KC.write_json(Path(KC.RESULTS) / "k_params.json", params_file(summary))
    KC.log(f"K3 done in {time.time() - t0:.0f} s")


def params_file(summary):
    """Machine-readable parameter file: every fit with its convention, units, data range and uncertainties."""
    out = {"description": "Battaglia B12 (pressure) / B16 (gas density) laws refitted to FLAMINGO L1_m9 HYDRO_FIDUCIAL "
                          "SOAP-HBT aperture integrals (Test K)",
           "conventions": {
               "x": "r / R200c (physical)",
               "mass": "M200c in Msun (no h); pivot 1e14 Msun; A * (M200c/1e14)^alpha_m * (1+z)^alpha_z",
               "B12": "P_e/P200 = 0.5176 * P0 * (x/x_c)^gamma * (1+(x/x_c)^alpha)^(-beta); P200 = G M200c 200 rho_c(z) f_b / (2 R200c); "
                      "XGPaint Battaglia16ThermalSZProfile(P0_amp, P0_alpha_m, P0_alpha_z, x_c_amp, ..., beta_amp, ...) takes these numbers directly",
               "B16": "rho_gas/(f_b rho_c(z)) = rho0 * (x/x_c)^gamma * (1+(x/x_c)^alpha)^(-(beta+gamma)/alpha); XGPaint BattagliaTauProfile[Physical](P0_amp=rho0 law, ...) "
                      "paints n_e = 0.9 rho_gas / mu_e (mu_e = (m_e + 2 m_p/(1+X)) with X = 0.76); see 'electron_factors' for the particle-measured n_e / (rho_gas/mu_e)",
               "brk_amp": "amplitude law times (M200c/M_b)^alpha_break for M200c < M_b (XGPaint BreakModel: alpha_break, M_break = M_b)",
               "brk_all": "ln Q = ln Q_b + (alpha_m_low if M < M_b else alpha_m_high) ln(M/M_b) + alpha_z ln(1+z); Q_b = value at M_b and z = 0",
               "f_b": KC.F_B, "data": "SOAP-HBT inclusive SO and physical-kpc sphere integrals, central haloes M200c >= 10^12.7, 0.2-dex cells at 71 outputs",
           },
           "fits": {}}
    for key, item in summary["configs"].items():
        if "table" not in item:
            continue
        ds = summary["datasets"][f"{item['config'][0]}|{item['config'][1]}"]
        out["fits"][key] = {"field": item["config"][0], "range": item["config"][1], "range_label": KC.RANGES[item["config"][1]]["label"],
                            "parametrization": item["config"][2], "law": item["table"], "theta_names": item["names"],
                            "theta": item["theta"], "cov_hessian": item["cov"], "boot_sd": item["boot_sd"],
                            "sigma_d": item["sigma_d"], "chi2_nu": item["chi2_nu"], "n_points": item["n_points"],
                            "n_cells": item["n_cells"], "z_range": ds["z_range"], "information": item["information"],
                            "at_bound": item["at_bound"]}
    return out
