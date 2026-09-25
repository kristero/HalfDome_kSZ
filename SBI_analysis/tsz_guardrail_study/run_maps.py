"""Run isolated, paired clean maps under a PBS allocation; record all failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
KEYS = ("P0_amp", "x_c_amp", "beta_amp", "P0_alpha_m", "x_c_alpha_m",
        "beta_alpha_m", "P0_alpha_z", "x_c_alpha_z", "beta_alpha_z")


def _run_case(root, name, theta, grid, refinement):
    protocol = json.loads((HERE / "protocol.json").read_text())
    campaign = Path(protocol["campaign"])
    pixel_nside = os.environ.get('STUDY_PIXEL_NSIDE', '8192')
    variant = grid+str(refinement) if grid != 'pixel' else 'pixel'+pixel_nside+'_'+str(refinement)
    output = root / "maps" / name / variant
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        old = json.loads((output / "status.json").read_text())
        if old["exit_code"] == 0:
            return
    snapshot = output / "source"
    (snapshot / "baseline").mkdir(parents=True, exist_ok=True)
    for source_name in ("map_experiment.jl", "baseline/stable_los.jl", "protocol.json"):
        shutil.copy2(HERE / source_name, snapshot / source_name)
    source_hashes = {name: hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
        for name in ("map_experiment.jl", "baseline/stable_los.jl", "protocol.json")}
    request = json.loads((campaign / "preflight/metadata_manifest.json").read_text())
    command = request["halfdome_reference"]["simulation_request"]["command"]
    settings = dict(arg.split("=", 1) for arg in command if "=" in arg and not arg.startswith("--"))
    noise = campaign / "code/halfdome/other_sims/SO"
    settings.update(output_dir=str(output / "raw"), cache_dir=str(output / "cache"),
        halfdome_path="/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5",
        enforce_battaglia_guardrails="false", model_exists="false", reuse_existing_cache="false",
        baseline_noise_path=str(noise / "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"),
        goal_noise_path=str(noise / "SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"))
    settings.update({"battaglia_"+key: format(float(value), ".17g") for key, value in zip(KEYS, theta)})
    if grid == 'pixel':
        settings['nside'] = pixel_nside
    runtime = campaign / "runtime"
    threads = int(os.environ.get("STUDY_THREADS", protocol["threads"]))
    env = {key: value for key, value in os.environ.items() if not key.startswith(("TSZ_", "BATTAGLIA_"))}
    env.update(HALFDOME_SOURCE_DIR=str(campaign / "code/halfdome"), FLAMINGO_CAMPAIGN=str(campaign),
        STUDY_OUTPUT=str(output), STUDY_GRID=grid, STUDY_REFINEMENT=str(refinement),
        JULIA_DEPOT_PATH=str(runtime / "depot")+":/home/kristero10/.julia",
        LD_LIBRARY_PATH=str(runtime / "julia-1.12.2/lib/julia")+":"+env.get("LD_LIBRARY_PATH", ""),
        JULIA_PKG_PRECOMPILE_AUTO="0", HDF5_USE_FILE_LOCKING="FALSE",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS=str(threads))
    command = [str(runtime / "julia-1.12.2/bin/julia"), "--startup-file=no", "--threads="+str(threads),
        "--project="+str(runtime / "julia_env"), str(snapshot / "map_experiment.jl")]
    command += [key+"="+value for key, value in settings.items()]
    started = time.monotonic()
    with (output / "run.log").open("w") as stream:
        process = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(output / "time.txt")]+command,
            env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=protocol["case_timeout_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    status = dict(name=name, theta=theta, grid=grid, refinement=refinement, exit_code=code,
        seconds=time.monotonic()-started, command=command, job_id=os.environ.get("PBS_JOBID"),
        source_sha256=source_hashes)
    (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
    print(name, grid, refinement, code, status["seconds"], flush=True)


def run_case(root, name, theta, grid, refinement):
    import fcntl  # Cluster-only advisory locking; --help remains usable elsewhere.
    lock_root = root/'locks'
    lock_root.mkdir(exist_ok=True)
    label = name+'_'+grid+str(refinement)+'_'+os.environ.get('STUDY_PIXEL_NSIDE','8192')
    with (lock_root/(label+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_case(root,name,theta,grid,refinement)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--grids", default="historical1,logz1,logz2")
    args = parser.parse_args()
    cases = json.loads((args.root / "inputs/cases.json").read_text())
    for name in args.cases:
        for grid in args.grids.split(","):
            run_case(args.root, name, cases[name], grid[:-1], int(grid[-1]))
