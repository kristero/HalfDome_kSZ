"""Run one full-resolution candidate and record its actual resource usage."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np

from prior_model import HIGH, KEYS, LOW, domain_metrics, save_json, sha256


def full_map(root, campaign, name, theta, timeout=1200):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (9,) or np.any(theta < LOW) or np.any(theta > HIGH):
        raise ValueError("Candidate outside the authorized pilot prior")
    minimum_beta = float(domain_metrics(theta)["min_beta"])
    if minimum_beta <= .75:
        raise ValueError("Unreliable/divergent LOS tail on interpolation grid")
    output = root / "results" / name
    if (output / "full_map_complete.json").exists():
        saved = json.loads((output / "full_map_complete.json").read_text())
        np.testing.assert_array_equal(theta, saved["theta"])
        return output
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((campaign / "preflight/metadata_manifest.json").read_text())
    request = metadata["halfdome_reference"]["simulation_request"]
    settings = dict(part.split("=", 1) for part in request["command"]
                    if "=" in part and not part.startswith("--"))
    noise = campaign / "code/halfdome/other_sims/SO"
    settings.update(halfdome_path="/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5",
        output_dir=str(output / "raw"), cache_dir=str(root / "paint_cache"),
        baseline_noise_path=str(noise / "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"),
        goal_noise_path=str(noise / "SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"),
        enforce_battaglia_guardrails="false")
    settings.update({key: format(value, ".17g") for key, value in zip(KEYS, theta)})
    runtime = campaign / "runtime"
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("TSZ_", "BATTAGLIA_")):
            environment.pop(key)
    environment.update(HALFDOME_SOURCE_DIR=str(campaign / "code/halfdome"),
        FLAMINGO_CAMPAIGN=str(campaign), PILOT_OUTPUT=str(output),
        JULIA_DEPOT_PATH=str(runtime / "depot") + ":/home/kristero10/.julia",
        JULIA_PKG_PRECOMPILE_AUTO="0", HDF5_USE_FILE_LOCKING="FALSE",
        LD_LIBRARY_PATH=str(runtime / "julia-1.12.2/lib/julia")+":"+environment.get("LD_LIBRARY_PATH", ""),
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="26")
    command = [str(runtime / "julia-1.12.2/bin/julia"), "--startup-file=no", "--threads=26",
               "--project="+str(runtime / "julia_env"), str(root / "code/paint_candidate.jl")]
    command += [key+"="+value for key, value in settings.items()]
    log = root / "logs" / (name+".log")
    timing = root / "logs" / (name+".time.txt")
    previous = None
    if (output / "run_status.json").exists():
        previous = json.loads((output / "run_status.json").read_text())
        attempt_number = len(previous.get("attempts", [previous]))
        if log.exists():
            shutil.copy2(log, log.with_suffix(".attempt"+str(attempt_number)+".log"))
        if timing.exists():
            shutil.copy2(timing, timing.with_suffix(".attempt"+str(attempt_number)+".txt"))
    started = time.monotonic()
    print("Full map starting: "+name, flush=True)
    with log.open("w") as stream:
        proc = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(timing)]+command,
            stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            exit_code = 124
    status = dict(theta=theta.tolist(), command=command, exit_code=exit_code,
                  elapsed_seconds=time.monotonic()-started, minimum_grid_beta=minimum_beta,
                  job_id=os.environ.get("PBS_JOBID"), log=str(log))
    if timing.exists():
        for line in timing.read_text().splitlines():
            if "Maximum resident set size (kbytes):" in line:
                status["peak_rss_GiB"] = int(line.split(":")[-1])/1024**2
    current_attempt = {key: status.get(key) for key in
        ("exit_code", "elapsed_seconds", "peak_rss_GiB", "job_id")}
    old_attempts = previous.get("attempts", [{key: previous.get(key) for key in current_attempt}]) if previous else []
    status["attempts"] = old_attempts+[current_attempt]
    status["elapsed_seconds"] = sum(a["elapsed_seconds"] for a in status["attempts"])
    status["peak_rss_GiB"] = max(a.get("peak_rss_GiB") or 0 for a in status["attempts"])
    save_json(output / "run_status.json", status)
    if exit_code:
        raise RuntimeError("Full map failed: "+name+"; see "+str(log))
    if status.get("peak_rss_GiB", 0) > 60:
        raise RuntimeError("Candidate approached the 64-GB allocation")
    status["files"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    save_json(output / "full_map_complete.json", status)
    print("Full map complete: {} in {:.1f}s".format(name, status["elapsed_seconds"]), flush=True)
    return output
