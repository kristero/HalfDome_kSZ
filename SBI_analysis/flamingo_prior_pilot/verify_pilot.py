"""Validate scientific provenance and export reusable tables for the pilot."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import toml

from prior_model import (FIDUCIAL, HIGH, KEYS, LOW, NAMES, bin_cl,
    comparison_metrics, save_json, sha256)


def parameter_receipt(root, campaign, output, status):
    path = output / "candidate_parameters.toml"
    if not path.exists():
        # A resume-compatible receipt for maps launched before this diagnostic
        # was added. Same archived command/configuration, no repainting.
        runtime = campaign / "runtime"
        environment = os.environ.copy()
        environment.update(HALFDOME_SOURCE_DIR=str(campaign / "code/halfdome"),
            FLAMINGO_CAMPAIGN=str(campaign), PILOT_OUTPUT=str(output), PILOT_VALIDATE_ONLY="1",
            JULIA_DEPOT_PATH=str(runtime / "depot")+":/home/kristero10/.julia",
            LD_LIBRARY_PATH=str(runtime / "julia-1.12.2/lib/julia")+":"+environment.get("LD_LIBRARY_PATH", ""),
            JULIA_PKG_PRECOMPILE_AUTO="0", HDF5_USE_FILE_LOCKING="FALSE")
        with (output / "parameter_receipt.log").open("w") as stream:
            subprocess.run(status["command"], env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=120)
    actual = toml.load(path)
    values = [actual[key.replace("battaglia_", "")] for key in KEYS]
    np.testing.assert_array_equal(values, status["theta"])
    assert actual["alpha_amp"] == 1. and actual["gamma_amp"] == -.3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root, campaign = args.root, args.campaign
    result = json.loads((root / "pilot_results.json").read_text())
    cache = toml.load(campaign / "cache/complete.toml")
    baseline = toml.load(campaign / "results/HalfDome/complete.toml")
    checked = []
    runtimes, memory = [], []
    for path in sorted((root / "results").glob("*/full_map_complete.json")):
        status = json.loads(path.read_text())
        output = path.parent
        for filename, digest in status["files"].items():
            assert sha256(output / filename) == digest, str(output / filename)
        parameter_receipt(root, campaign, output, status)
        metadata = toml.load(output / "complete.toml")
        for key in ("cache_file_sha256", "mask_pixel_sha256", "noise1_pixel_sha256", "noise2_pixel_sha256"):
            assert metadata[key] == cache[key], (output, key)
        assert metadata["runtime"] == baseline["runtime"], output
        assert metadata["nside"] == 4096 and metadata["ell_max"] == 7979
        assert metadata["beam_fwhm_arcmin"] == 2.
        for filename in ("masked_clean_cl.npy", "unmasked_clean_cl.npy", "masked_noisy_cross_cl.npy"):
            values = np.load(output / filename)
            assert values.shape == (7980,) and np.isfinite(values).all(), output
        theta = np.array(status["theta"])
        assert np.all(theta >= LOW) and np.all(theta <= HIGH)
        checked.append(output.name)
        runtimes.append(status["elapsed_seconds"])
        memory.append(status["peak_rss_GiB"])
    # The Mstar warm start changes only P0. Two independently repainted maps
    # therefore test linear signal scaling and the fixed-noise cross terms.
    base = root / "results/fit_L1_m9_iter2"
    scaled = root / "results/fit_Mstar-1sigma_iter0"
    first = json.loads((base / "full_map_complete.json").read_text())["theta"]
    second = json.loads((scaled / "full_map_complete.json").read_text())["theta"]
    np.testing.assert_array_equal(first[1:], second[1:])
    amplitude = second[0]/first[0]
    signal = np.load(base / "masked_clean_cl.npy")
    noisy = np.load(base / "masked_noisy_cross_cl.npy")
    noise_cross = np.load(campaign / "cache/masked_noise_cross_cl.npy")
    predicted_clean = amplitude**2*signal
    predicted_noisy = amplitude**2*signal+amplitude*(noisy-signal-noise_cross)+noise_cross
    measured_clean = np.load(scaled / "masked_clean_cl.npy")
    measured_noisy = np.load(scaled / "masked_noisy_cross_cl.npy")
    clean_error = float(np.max(abs(measured_clean[80:]/predicted_clean[80:]-1)))
    noisy_error = float(np.max(abs(measured_noisy[80:]-predicted_noisy[80:]))
                        /np.max(abs(predicted_noisy[80:])))
    assert clean_error < 1e-10 and noisy_error < 1e-10
    save_json(root / "audit/amplitude_scaling_validation.json", dict(
        status="passed", base=base.name, scaled=scaled.name, amplitude_ratio=amplitude,
        maximum_clean_relative_error=clean_error,
        maximum_noisy_absolute_error_over_peak=noisy_error,
        scope="Two independently painted maps; ell>=80; same fixed noise and exact quadratic/linear cross terms"))
    parameter_rows = [dict(name="Battaglia12", kind="reference", **dict(zip(NAMES, FIDUCIAL)))]
    table = {}
    scale_metrics = {}
    for variant, fitted in result["fits"].items():
        best = fitted["best"]
        assert best["name"] in checked
        output = root / "results" / best["name"]
        requested = json.loads((output / "full_map_complete.json").read_text())["theta"]
        np.testing.assert_array_equal(best["theta"], requested)
        ell, prediction = bin_cl(np.load(output / "masked_clean_cl.npy"))
        target = bin_cl(np.load(campaign / "results" / variant / "masked_clean_cl.npy"))[1]
        metrics = comparison_metrics(prediction, target)
        scale_metrics[variant] = {
            "ell_ge_"+str(lower): dict(
                bins=int(np.sum(ell >= lower)),
                **comparison_metrics(prediction[ell >= lower], target[ell >= lower]))
            for lower in (80, 500, 1000, 2000)
        }
        scale_metrics[variant]["worst_bin_ell"] = float(ell[np.argmax(abs(prediction/target-1))])
        for key in metrics:
            assert np.isclose(metrics[key], best[key], rtol=1e-12)
        parameter_rows.append(dict(name=variant, kind="clean_spectrum_fit",
            **dict(zip(NAMES, best["theta"]))))
        table["ell"] = ell
        table[variant+"_target_clean"] = target
        table[variant+"_fit_clean"] = prediction
        table[variant+"_target_noisy"] = bin_cl(np.load(campaign / "results" / variant / "masked_noisy_cross_cl.npy"))[1]
        table[variant+"_fit_noisy"] = bin_cl(np.load(output / "masked_noisy_cross_cl.npy"))[1]
    for name, data in result["extremes"].items():
        parameter_rows.append(dict(name=name, kind=data["status"], **dict(zip(NAMES, data["theta"]))))
    for p0 in (1., 60.):
        theta = FIDUCIAL.copy()
        theta[0] = p0
        parameter_rows.append(dict(name="P0_"+str(int(p0)), kind="exact_amplitude_rescaling",
                                   **dict(zip(NAMES, theta))))
    with (root / "parameters.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "kind"]+list(NAMES))
        writer.writeheader()
        writer.writerows(parameter_rows)
    names = list(table)
    np.savetxt(root / "fit_spectra.csv", np.column_stack([table[name] for name in names]),
               delimiter=",", header=",".join(names), comments="", fmt="%.17g")
    save_json(root / "audit/fit_scale_metrics.json", dict(
        fits=scale_metrics,
        scope="Post-fit diagnostics with the same best parameters; no bins were removed from fitting"))
    unfinished = {
        p.parent.name: json.loads(p.read_text())
        for p in (root / "results").glob("*/run_status.json")
        if not (p.parent / "full_map_complete.json").exists()
    }
    cache_sizes = [p.stat().st_size for p in (root / "paint_cache").glob("*.jld2")]
    resource = dict(completed_full_maps=len(checked), maximum_map_RSS_GiB=max(memory),
        median_map_seconds=float(np.median(runtimes)), maximum_map_seconds=max(runtimes),
        estimated_serial_hours_per_1000_maps=float(np.median(runtimes)*1000/3600),
        estimated_core_hours_per_1000_maps=float(np.median(runtimes)*1000/3600*26),
        estimate_scope="Completed full map, clean/noisy spectra and profile checks; excludes optimizer and failed attempts",
        incomplete_map_attempts={name: dict(exit_code=data["exit_code"],
            elapsed_seconds=data["elapsed_seconds"]) for name, data in unfinished.items()},
        retained_interpolator_cache_bytes=sum(cache_sizes),
        retained_interpolator_caches=len(cache_sizes),
        median_interpolator_cache_bytes=float(np.median(cache_sizes)),
        estimated_cache_GiB_per_1000_models=float(np.median(cache_sizes)*1000/1024**3))
    save_json(root / "resource_usage.json", resource)
    assert len(list((root / "plots").glob("*.png"))) == 9
    save_json(root / "verification.json", dict(status="passed", full_maps_verified=checked,
        operator="Original archived Julia source/runtime, exact mask and noise pixel hashes, 40-bin clean statistic",
        parameter_receipts="All nine parsed Julia parameters equal recorded requested values exactly",
        completed_plots=len(list((root / "plots").glob("*.png")))))
    save_json(root / "artifact_manifest.json", dict(files={
        p.relative_to(root).as_posix(): sha256(p)
        for directory in ("code", "audit", "results", "proposals", "plots")
        for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }))
    print("Pilot outputs, parameters and exact noise/operator provenance verified", flush=True)


if __name__ == "__main__":
    main()
