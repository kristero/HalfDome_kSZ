"""Fit native FLAMINGO pressure stacks to 1R; preserve original 2/3/4R fits.

All four variants share the same halo population, original resolved lower mask,
and diagonal empirical objective. The exported curves use a common calibration
support and published fallback, so changing fit extent cannot silently change
which masses use simulations. No snapshot particles are read by this script.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parents[1] / "radii234_contamination_L1_m9_0077_20261008"
sys.path.insert(0, str(CALIBRATION))
import pilot_physics as P
from models import fit_profile, shape, shell, BOUNDS

RADII = np.array([1., 2., 3., 4.])
BOOTSTRAPS = 128


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def dump(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def accepted(fit):
    return bool(fit and fit["success"] and fit["bound_distance"] > 1e-5
                and fit["jacobian_condition"] < 1e6)


def fit(mean, std, mask, start=None):
    valid = mask & np.isfinite(mean) & (mean > 0)
    if valid.sum() < 6:
        return None
    error = np.maximum(np.divide(std, mean, out=np.ones_like(std), where=mean > 0), .05)
    return fit_profile("b12", mean, valid, error, start=start)


def wide_fit(mean, std, mask, start):
    """Sensitivity only: expand every coefficient bound, leave data/objective fixed."""
    lower = np.array([1e-5, 1e-3, .05])
    upper = np.array([1e6, 100., 100.])
    lo, hi = np.log(lower), np.log(upper)
    error = np.maximum(std/mean, .05)
    def residual(theta):
        return (np.log(shell("b12", np.exp(theta))[mask]) - np.log(mean[mask])) / error[mask]
    solution = least_squares(residual, np.clip(np.log(start), lo+1e-7, hi-1e-7),
                             bounds=(lo, hi), max_nfev=2000,
                             ftol=1e-9, xtol=1e-9, gtol=1e-9)
    p = np.exp(solution.x)
    singular = np.linalg.svd(solution.jac, compute_uv=False)
    return {"parameters": p, "success": solution.success,
            "bound_distance": np.min(np.minimum(solution.x-lo, hi-solution.x)),
            "jacobian_condition": singular[0]/max(singular[-1], 1e-30),
            "rms_percent": 100*np.sqrt(np.mean((shell("b12", p)[mask]/mean[mask]-1)**2)),
            "objective": float(np.sum(solution.fun**2)),
            "bounds": [lower, upper]}


def relative_rms(prediction, measured, mask):
    valid = mask & np.isfinite(measured) & (measured > 0)
    return float(100*np.sqrt(np.mean((prediction[valid]/measured[valid]-1)**2))) if valid.any() else np.nan


def published(mass, x):
    parameters = P.B12[:3] * (mass/1e14)**P.B12[3:6]
    return shape("b12", x, parameters)


def integrate(parameters, radius=4.):
    nodes, weights = np.polynomial.legendre.leggauss(128)
    x = (nodes+1)*radius/2
    return float(4*np.pi*np.sum(shape("b12", x, parameters)*x*x*weights*radius/2))


def main():
    started, cpu = time.perf_counter(), time.process_time()
    summary_path = CALIBRATION/"products/study_summary.json"
    cases = json.loads(summary_path.read_text())["bins"]
    halos = dict(np.load(CALIBRATION/"selected_halos.npz"))
    halos["logmass"] = np.log10(halos["m200c_msun"])
    data = dict(np.load(CALIBRATION/"measurements.npz"))
    nbin, nshell = len(cases), len(P.EDGES)-1
    parameters = np.full((nbin, 4, 3), np.nan)
    bootstrap = np.full((nbin, 4, BOOTSTRAPS, 3), np.nan)
    masks = np.zeros((nbin, 4, nshell), bool)
    means, stds = np.empty((nbin, nshell)), np.empty((nbin, nshell))
    stable = np.zeros((nbin, 4), bool)
    strict = np.array([case["strict_resolution_supported"] for case in cases])
    fit_flags, rows_csv = [], []
    no_other = np.empty((nbin, 3))
    wide = np.full((nbin, 3), np.nan)
    for b, case in enumerate(cases):
        rows = np.flatnonzero(halos["mass_bin"] == b)
        stored_profiles = np.load(CALIBRATION/"products"/(case["label"]+"_profiles.npz"))
        values = data["pressure_components"][rows].sum(axis=1)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        means[b], stds[b] = mean, std
        base = np.array(case["fit_base_mask"], bool)
        one_mask = base & (P.EDGES[1:] <= 1 + 1e-12) & (mean > 0)
        one = fit(mean, std, one_mask)
        native = [one] + [entry["fit"] for entry in case["selections"]["all"]["pressure"]]
        old_boot = np.load(CALIBRATION/"products"/(case["label"]+"_parameters.npz"))["bootstrap"][:,0]
        bootstrap[b, 1:] = old_boot
        no_other[b] = case["selections"]["no_other"]["pressure"][-1]["fit"]["parameters"]
        record = {"label": case["label"], "mass_edges": case["mass_edges"],
                  "halos": len(rows), "strict_resolution_supported": strict[b],
                  "strict_1R_shells": int(np.sum(stored_profiles["strict_mask"] & (P.EDGES[1:]<=1))),
                  "fit_1R_shells": int(one_mask.sum()), "fits": native}
        for ri, (radius, fitted) in enumerate(zip(RADII, native)):
            mask = base & (P.EDGES[1:] <= radius+1e-12) & (mean > 0)
            masks[b, ri] = mask
            if fitted is None:
                continue
            parameters[b, ri] = fitted["parameters"]
            stable[b, ri] = accepted(fitted)
            prediction = shell("b12", parameters[b, ri])
            row = {"bin_low": case["mass_edges"][0], "bin_high": case["mass_edges"][1],
                   "halos": len(rows), "fit_radius": radius,
                   "P0": parameters[b,ri,0], "xc": parameters[b,ri,1], "beta": parameters[b,ri,2],
                   "fit_shells": int(mask.sum()), "resolution_supported": int(strict[b]),
                   "stable": int(stable[b,ri]), "bound_distance": fitted["bound_distance"],
                   "jacobian_condition": fitted["jacobian_condition"],
                   "fit_RMS_percent": relative_rms(prediction, mean, mask),
                   "common_inner_1R_RMS_percent": relative_rms(prediction, mean, one_mask),
                   "resolved_all_4R_RMS_percent": relative_rms(prediction, mean, base),
                   "outer_1to4_RMS_percent": relative_rms(prediction, mean, base&(P.EDGES[:-1]>=1)),
                   "Y4_dimensionless": integrate(parameters[b,ri])}
            rows_csv.append(row)
        if one:
            wide_result = wide_fit(mean, std, one_mask, one["parameters"])
            wide[b] = wide_result["parameters"]
            record["wide_1R"] = wide_result
            alt = fit(mean, std, one_mask, start=parameters[b,1])
            record["old_2R_start_1R"] = alt
            record["multistart_max_relative_parameter_difference"] = np.max(np.abs(np.array(alt["parameters"])/parameters[b,0]-1))
            record["wide_bound_Y4_ratio"] = integrate(wide[b])/integrate(parameters[b,0])
        fit_flags.append(record)
        print(json.dumps({"bin":case["label"], "strict":bool(strict[b]),
                          "1R_shells":int(one_mask.sum()), "stable":stable[b].tolist(),
                          "1R_parameters":parameters[b,0].tolist(),
                          "1R_condition":None if one is None else one["jacobian_condition"]}), flush=True)

    # Identical support for every radius variant is essential for a fit-only test.
    common = strict & np.all(stable, axis=1) & (masks[:,0].sum(axis=1)>=6)
    dump(HERE/"common_support.json", {"common_support":common,
         "supported_mass_bins":[cases[b]["mass_edges"] for b in np.flatnonzero(common)],
         "strict_resolution_supported":strict,"stable_all_radii":stable,
         "minimum_1R_shells":6,"published_fallback_shared_between_all_fit_radii":True,
         "measured_max_logmass":float(halos["logmass"].max()),"fit_flags":fit_flags})
    np.savez_compressed(HERE/"fit_parameters.npz",parameters=parameters,
                        fit_radii=RADII, bootstrap=bootstrap, masks=masks,
                        mean=means,std=stds,edges=P.EDGES,stable=stable,
                        strict_resolution_supported=strict,common_support=common,
                        no_other_R4_parameters=no_other,wide_bound_1R_parameters=wide,
                        mass_edges=halos["mass_edges"])
    print("POINT_FITS_READY", flush=True)

    # Exactly the original halo-resampling sequence: all 128 draws are paired
    # with the stored 2/3/4R fits; rejected fits remain NaN in their original slot.
    for b, case in enumerate(cases):
        if not np.all(np.isfinite(parameters[b,0])):
            continue
        rows = np.flatnonzero(halos["mass_bin"]==b)
        values = data["pressure_components"][rows].sum(axis=1)
        rng = np.random.default_rng(20261008+b)
        chosen = rng.integers(0,len(rows),size=(BOOTSTRAPS,len(rows)))
        for draw, selected in enumerate(chosen):
            sample = values[selected]
            result = fit(sample.mean(axis=0),sample.std(axis=0,ddof=1),masks[b,0],start=parameters[b,0])
            if accepted(result):
                bootstrap[b,0,draw] = result["parameters"]
        print(json.dumps({"bootstrap_bin":case["label"],"accepted_1R":int(np.all(np.isfinite(bootstrap[b,0]),axis=1).sum())}),flush=True)

    grid_logmass = np.linspace(12.,16.,201)
    x = np.geomspace(1e-5,4.,1024)
    profiles = np.empty((4,len(grid_logmass),len(x)))
    extra_names = np.array(["no_other_fit4","published","Y4_matched_fit1","Y4_matched_fit2","Y4_matched_fit3","wide_bounds_fit1"])
    extras = np.empty((len(extra_names),len(grid_logmass),len(x)))
    source_bin = np.clip(np.searchsorted(halos["mass_edges"],grid_logmass,side="right")-1,0,nbin-1)
    calibrated = np.zeros(len(grid_logmass),bool)
    maximum = float(halos["logmass"].max())
    for mi,lm in enumerate(grid_logmass):
        b = source_bin[mi]
        fallback = published(10**lm,x)
        valid = halos["mass_edges"][0] <= lm <= maximum and common[b]
        calibrated[mi] = valid
        profiles[:,mi] = [shape("b12",x,p) for p in parameters[b]] if valid else fallback
        extras[:,mi] = fallback
        if valid:
            extras[0,mi] = shape("b12",x,no_other[b])
            for ri in range(3):
                extras[2+ri,mi] = profiles[ri,mi]*integrate(parameters[b,3])/integrate(parameters[b,ri])
            extras[-1,mi] = shape("b12",x,wide[b])
    assert np.all(np.isfinite(profiles)) and np.all(profiles>0)
    assert np.all(np.isfinite(extras)) and np.all(extras>0)
    np.savez_compressed(HERE/"profile_shapes.npz",fit_radii=RADII,
                        grid_logmass=grid_logmass,x=x,profiles=profiles,
                        extra_case_names=extra_names,extra_profiles=extras,
                        mass_edges=halos["mass_edges"],common_support=common,
                        calibrated_logmass_max=maximum,calibrated_grid=calibrated,
                        source_bin=source_bin)
    # Bootstrap summaries remain conditional on the sampled halo population.
    for row in rows_csv:
        b = int(np.argmin(np.abs(halos["mass_edges"][:-1]-row["bin_low"])))
        ri = int(row["fit_radius"])-1
        draws = bootstrap[b,ri]
        draws = draws[np.all(np.isfinite(draws),axis=1)]
        row["bootstrap_accepted"] = len(draws)
        row["common_support"] = int(common[b])
        for j,name in enumerate(("P0","xc","beta")):
            for percentile in (16,50,84):
                row[name+f"_p{percentile}"] = np.percentile(draws[:,j],percentile) if len(draws) else np.nan
    with (HERE/"coefficients.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows_csv[0]))
        writer.writeheader();writer.writerows(rows_csv)
    np.savez_compressed(HERE/"fit_parameters.npz",parameters=parameters,
                        fit_radii=RADII,bootstrap=bootstrap,masks=masks,
                        mean=means,std=stds,edges=P.EDGES,stable=stable,
                        strict_resolution_supported=strict,common_support=common,
                        no_other_R4_parameters=no_other,wide_bound_1R_parameters=wide,
                        mass_edges=halos["mass_edges"])
    input_hashes={str(p.relative_to(CALIBRATION)):sha(p) for p in
                  [summary_path,CALIBRATION/"selected_halos.npz",CALIBRATION/"measurements.npz"]}
    dump(HERE/"fit_input_summary.json",{
        "common_support_bins":[cases[b]["mass_edges"] for b in np.flatnonzero(common)],
        "common_support_count":int(common.sum()),"fit_flags":fit_flags,
        "native_2_3_4_coefficients_preserved":True,
        "bootstrap_draws":128,"bootstrap_seed":"20261008+bin_index",
        "bootstrap_scope":"conditional mean-fit uncertainty; no spatial/cosmic variance or individual-scatter realization",
        "profile_convention":"F(x)=Pe/P200=.5176 P0 (x/xc)^(-.3) (1+x/xc)^(-beta)",
        "objective":"log-shell residual / max(halo std/mean,0.05); shell-volume average; original mask",
        "bounds_primary":BOUNDS["b12"],
        "fallback":"ALL variants use exact published B12 at the actual mass outside common support/max measured mass",
        "mass_table_policy":"piecewise constant measured shapes within source bins; published varies with mass; do not interpolate across unsupported boundaries",
        "units":"native Pe/P200, physical Msun; do not apply additional h or hydrogen factor",
        "physical_units_code":str(CALIBRATION/"pilot_physics.py"),
        "input_SHA256":input_hashes,"wall_seconds":time.perf_counter()-started,
        "CPU_seconds":time.process_time()-cpu,
        "peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "threads":1,"raw_particle_reads":0,"submitted_jobs":0})
    print(json.dumps({"complete":True,"wall_seconds":time.perf_counter()-started,
                      "common_support":common.tolist()}),flush=True)


if __name__=="__main__":
    main()
