#!/usr/bin/env python3
"""Run approved map operations sequentially and preserve a receipt per stage."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import toml

from preflight import sha256
from download_maps import save_json


def execute(command, environment, root, stage):
    log_path = root / "logs" / (stage + ".log")
    time_path = root / "logs" / (stage + ".time.txt")
    status_path = root / "logs" / (stage + ".status.json")
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.run(["/usr/bin/time", "-v", "-o", str(time_path)] + command,
                                 env=environment, stdout=log, stderr=subprocess.STDOUT)
    status = dict(command=command, exit_code=process.returncode,
                  elapsed_seconds=time.monotonic()-started,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  job_id=os.environ.get("PBS_JOBID"))
    for line in time_path.read_text().splitlines():
        if "Maximum resident set size (kbytes):" in line:
            status["peak_rss_kbytes"] = int(line.split(":")[-1].strip())
    save_json(status_path, status)
    print(stage + ": " + json.dumps(status), flush=True)
    if process.returncode:
        raise RuntimeError("Stage failed: {}. See {}".format(stage, log_path))
    if status.get("peak_rss_kbytes", 0) > 60 * 1024 ** 2:
        raise RuntimeError("Peak RSS approached the approved 64-GB limit; stopping before another map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--julia")
    parser.add_argument("--julia-project")
    parser.add_argument("--halfdome-catalogue", default="/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5")
    args = parser.parse_args()
    root = args.campaign.resolve()
    args.julia = args.julia or str(root / "runtime/julia-1.12.2/bin/julia")
    args.julia_project = args.julia_project or str(root / "runtime/julia_env")
    metadata = json.loads((root / "preflight/metadata_manifest.json").read_text())
    approval = json.loads((root / "approval.json").read_text())
    if approval["download_approved"] is not True:
        raise ValueError("Campaign has no approval")
    download = json.loads((root / "inputs/download_complete.json").read_text())
    if download["manifest_sha256"] != approval["manifest_sha256"]:
        raise ValueError("Download manifest changed")
    for record in download["files"].values():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Downloaded input changed")
    for relative, expected in json.loads((root / "code/source_manifest.json").read_text()).items():
        if sha256(root / "code/halfdome" / relative) != expected:
            raise ValueError("Changed HalfDome source: " + relative)
    runtime_manifest = json.loads((root / "code/runtime_source_manifest.json").read_text())
    for relative, expected in runtime_manifest["source_hashes"].items():
        if sha256(root / "runtime/XGPaint" / relative) != expected:
            raise ValueError("Changed XGPaint source: " + relative)
    request = metadata["halfdome_reference"]["simulation_request"]
    if Path(args.halfdome_catalogue).stat().st_size != request["catalogue"]["size"]:
        raise ValueError("HalfDome control catalogue size differs")
    noise = root / "code/halfdome/other_sims/SO/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
    if sha256(noise) != request["noise_sha256"]:
        raise ValueError("Noise table differs from HalfDome reference")
    settings = dict(part.split("=", 1) for part in request["command"]
                    if "=" in part and not part.startswith("--"))
    settings.update(halfdome_path=args.halfdome_catalogue,
                    baseline_noise_path=str(noise),
                    goal_noise_path=str(noise.with_name("SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt")),
                    output_dir=str(root / "results/HalfDome/raw"),
                    cache_dir=str(root / "paint_cache"))
    julia_command = [args.julia, "--startup-file=no", "--threads=26",
                     "--project=" + args.julia_project, str(root / "code/process_maps.jl")]
    julia_command += [key + "=" + value for key, value in settings.items()]
    environment = os.environ.copy()
    original_library_path = environment.get("LD_LIBRARY_PATH")
    for key in list(environment):
        if key.startswith(("TSZ_", "BATTAGLIA_")):
            environment.pop(key)
    environment.update(HALFDOME_SOURCE_DIR=str(root / "code/halfdome"),
                       FLAMINGO_CAMPAIGN=str(root), HDF5_USE_FILE_LOCKING="FALSE",
                       JULIA_DEPOT_PATH=str(root / "runtime/depot") + ":/home/kristero10/.julia",
                       JULIA_NUM_PRECOMPILE_TASKS="8", JULIA_PKG_PRECOMPILE_AUTO="0",
                       LD_LIBRARY_PATH=str(root / "runtime/julia-1.12.2/lib/julia") + ":" + (original_library_path or ""),
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="26")
    environment.update(FLAMINGO_MODE="probe", FLAMINGO_OUTPUT=str(root / "preflight/production_probe.toml"))
    execute(julia_command, environment, root, "operator_probe")
    actual = toml.load(root / "preflight/production_probe.toml")
    expected = toml.load(root / "code/local_operator_probe.toml")
    if actual["mask_pixel_sha256"] != expected["mask_pixel_sha256"]:
        raise ValueError("Mask RNG differs from saved-reference environment")
    for key in ("rng_uniform", "rng_normal", "noise_first_pixels", "beam_first_pixels"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-11, atol=1e-20,
                                   err_msg="Operator/RNG differs across runtimes: " + key)
    save_json(root / "preflight/operator_comparison.json", dict(
        status="passed", actual=actual, reference=expected,
        note="Small operator regression only; production maps share the same cached arrays pixel for pixel"))
    environment["FLAMINGO_MODE"] = "cache"
    execute(julia_command, environment, root, "shared_noise_cache")
    # First a fresh HalfDome reference, then one FLAMINGO map at a time.
    for name in ("HalfDome", "L1_m9", "fgas-8sigma", "Mstar-1sigma"):
        directory = root / "results" / name
        if (directory / "complete.toml").exists():
            print("Reusing completed map stage " + name, flush=True)
            continue
        environment.update(FLAMINGO_MODE="map", FLAMINGO_OUTPUT=str(directory),
                           FLAMINGO_INPUT="halfdome" if name == "HalfDome" else download["files"][name]["path"])
        execute(julia_command, environment, root, "map_" + name)
    environment.update(OMP_NUM_THREADS="26", OPENBLAS_NUM_THREADS="26", MKL_NUM_THREADS="26",
                       APP_THREADS="26", MPLBACKEND="Agg")
    if original_library_path is None:
        environment.pop("LD_LIBRARY_PATH", None)
    else:
        environment["LD_LIBRARY_PATH"] = original_library_path
    command = [sys.executable, str(root / "code/compare_inference.py"), "--campaign", str(root),
               "--helpers", str(root / "code/inference_helpers"),
               "--bundle", metadata["verified_bundle"]["path"],
               "--reference", metadata["halfdome_reference"]["path"]]
    execute(command, environment, root, "inference_comparison")
    save_json(root / "campaign_complete.json", dict(status="completed", job_id=os.environ.get("PBS_JOBID"),
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completion_sha256=sha256(root / "comparison/complete.json")))


if __name__ == "__main__":
    main()
