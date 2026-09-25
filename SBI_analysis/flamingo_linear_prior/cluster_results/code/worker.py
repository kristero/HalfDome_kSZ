"""One PBS chunk: immutable row IDs, transactional results and bounded scratch.

No failed row is replaced or silently dropped. Retrying a chunk uses precisely
the original parameters. Successful rows survive a timeout or node failure.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import h5py
import numpy as np

from prior import JointPrior, KEYS, digest, write_json
from noise_seeds import ALGORITHM, PREFLIGHT_OFFSET, split_seeds

SPECTRA = ("unmasked_clean_cl", "masked_clean_cl", "masked_noisy_cross_cl")
BINNED_NAMES = {"unmasked_clean_cl": "unmasked_clean_dl40",
                "masked_clean_cl": "masked_clean_dl40",
                "masked_noisy_cross_cl": "masked_noisy_cross_dl40"}
EDGES = np.append(np.arange(80, 7881, 200), 7980)


def bin_cl(cl):
    ell = np.arange(7980)
    dl = cl*ell*(ell+1)/(2*np.pi)
    return np.array([np.average(dl[a:b], weights=2*ell[a:b]+1)
                     for a,b in zip(EDGES[:-1], EDGES[1:])])


def verify_frozen(root):
    manifest = json.loads((root/"manifest.json").read_text())
    for folder, key in (("code", "code_sha256"), ("design", "design_sha256")):
        for name, expected in manifest[key].items():
            if digest(root/folder/name) != expected:
                raise RuntimeError("Frozen input changed: " + str(root/folder/name))
    for name, expected in manifest.get("dependency_sha256", {}).items():
        if digest(name) != expected:
            raise RuntimeError("Archived dependency changed: " + name)
    if digest(root/"run_config.json") != manifest["run_config_sha256"]:
        raise RuntimeError("Run configuration changed; use a new run root")
    if digest(root/"preflight/cases.json") != manifest["preflight_cases_sha256"]:
        raise RuntimeError("Preflight cases changed")
    return manifest


def run_row(root, run, theta, label):
    assert run["training_noise_mode"] == "independent_per_row"
    index = int(label.rsplit("_", 1)[1])
    is_preflight = label.startswith("preflight_")
    seed_row_id = (index + PREFLIGHT_OFFSET if is_preflight else
                   int(np.load(root/"design/noise_seed_row_ids.npy",mmap_mode="r")[index]))
    seeds = split_seeds(seed_row_id, run["noise_master_seed"])
    if not is_preflight:
        np.testing.assert_array_equal(seeds, np.load(root/"design/noise_split_seeds.npy", mmap_mode="r")[index])
    campaign = Path(run["campaign"])
    request = json.loads((campaign/"preflight/metadata_manifest.json").read_text())
    arguments = request["halfdome_reference"]["simulation_request"]["command"]
    settings = dict(part.split("=",1) for part in arguments if "=" in part and not part.startswith("--"))
    started = time.monotonic()
    # TemporaryDirectory cleanup is restricted to this newly created row tree.
    with tempfile.TemporaryDirectory(prefix=label+"_", dir=str(root/"scratch")) as temp:
        temp = Path(temp)
        output = temp/"result"
        output.mkdir()
        noise = campaign/"code/halfdome/other_sims/SO"
        settings.update(halfdome_path=run["catalogue"], output_dir=str(output/"raw"),
            cache_dir=str(temp/"cache"), enforce_battaglia_guardrails="false",
            baseline_noise_path=str(noise/"SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"),
            goal_noise_path=str(noise/"SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"))
        settings.update({key: format(value,".17g") for key,value in zip(KEYS,theta)})
        environment = {k:v for k,v in os.environ.items() if not k.startswith(("TSZ_", "BATTAGLIA_"))}
        runtime = campaign/"runtime"
        environment.update(HALFDOME_SOURCE_DIR=str(campaign/"code/halfdome"),
            FLAMINGO_CAMPAIGN=str(campaign), EXTENDED_ROW_OUTPUT=str(output),
            EXTENDED_NOISE_SPLIT_SEEDS=",".join(map(str, seeds)),
            EXTENDED_NOISE_ROW_ID=str(seed_row_id),
            EXTENDED_NOISE_MASTER_SEED=str(run["noise_master_seed"]),
            EXTENDED_NOISE_SEED_ALGORITHM=ALGORITHM,
            EXTENDED_NOISE_VALIDATE="1" if is_preflight and index == 0 else "0",
            JULIA_DEPOT_PATH=str(runtime/"depot")+":/home/kristero10/.julia",
            JULIA_PKG_PRECOMPILE_AUTO="0", HDF5_USE_FILE_LOCKING="FALSE",
            LD_LIBRARY_PATH=str(runtime/"julia-1.12.2/lib/julia")+":"+environment.get("LD_LIBRARY_PATH",""),
            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS=str(run["threads"]))
        command = [str(runtime/"julia-1.12.2/bin/julia"), "--startup-file=no",
            "--threads="+str(run["threads"]), "--project="+str(runtime/"julia_env"),
            str(root/"code/paint_row.jl")]+[k+"="+v for k,v in settings.items()]
        timing, log = temp/"time.txt", temp/"row.log"
        with log.open("w") as stream:
            proc = subprocess.Popen(["/usr/bin/time","-v","-o",str(timing)]+command,
                stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            try:
                exit_code = proc.wait(timeout=run["row_timeout_seconds"])
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                exit_code = 124
        status = dict(label=label, theta=list(map(float,theta)), exit_code=exit_code,
            elapsed_seconds=time.monotonic()-started, job_id=os.environ.get("PBS_JOBID"),
            command=command)
        for line in timing.read_text().splitlines() if timing.exists() else []:
            if "Maximum resident set size (kbytes):" in line:
                status["peak_rss_GiB"] = int(line.split(":")[-1])/1024**2
        if exit_code:
            failure = root/"logs"/(label+"_"+str(time.time_ns()))
            shutil.copy2(log, str(failure)+".log")
            write_json(str(failure)+".json", status)
            raise RuntimeError("Row failed without replacement: " + str(failure))
        data = {name: np.load(output/(name+".npy")) for name in SPECTRA}
        for name, values in data.items():
            if values.shape != (7980,) or not np.isfinite(values).all():
                failure = root/"logs"/(label+"_validation_"+str(time.time_ns()))
                shutil.copy2(log, str(failure)+".log")
                status.update(validation_error="invalid spectrum: "+name,
                              shape=list(values.shape),nonfinite=int(np.sum(~np.isfinite(values))))
                write_json(str(failure)+".json",status)
                np.savez(str(failure)+".npz",**data)
                raise RuntimeError("Invalid spectrum: " + name)
            if "noisy" not in name and np.any(values < 0):
                raise RuntimeError("Negative clean auto spectrum")
        # Check that Julia parsed exactly the submitted parameters.
        import toml
        parsed = toml.load(str(output/"parameters.toml"))
        for key,value in zip(KEYS,theta):
            # Configuration uses P0_amp, x_c_amp etc. without battaglia_.
            actual = parsed.get(key, parsed.get(key.removeprefix("battaglia_")
                                if hasattr(key,"removeprefix") else key[len("battaglia_"):]))
            if actual != value:
                raise RuntimeError("Parsed parameter mismatch: " + key)
        status["operator"] = toml.load(str(output/"complete.toml"))
        status["numerics"] = toml.load(str(output/"numerics.toml"))
        status["noise"] = toml.load(str(output/"noise.toml"))
        assert status["noise"]["split_seeds"] == seeds
        assert status["noise"]["seed_row_id"] == seed_row_id
        for key in ("mask_pixel_sha256", "noise1_pixel_sha256", "noise2_pixel_sha256"):
            assert status["noise"][key] == status["operator"][key]
        return data, status, log.read_bytes()


def store_row(root, target, theta, data, status, log, full_cl):
    """Write a temporary HDF5 file then atomically publish one completed row.

    Chunk packing happens only after all rows finish, so a killed writer cannot
    corrupt previously completed rows. At most chunk_size row files per worker.
    """
    temporary = target.with_suffix(".tmp.h5")
    with h5py.File(temporary,"w") as handle:
        handle["theta"] = theta
        for name, values in data.items():
            handle[BINNED_NAMES[name]] = bin_cl(values)
            if full_cl:
                handle[name] = values
        handle.attrs["status_json"] = json.dumps(status)
        handle.create_dataset("log",data=np.frombuffer(log,dtype=np.uint8),compression="gzip")
        handle.attrs["manifest_sha256"] = digest(root/"manifest.json")
        handle.attrs["complete"] = True
        handle.flush()
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--chunk",type=int,required=True)
    parser.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify_frozen(root)
    run = json.loads((root/"run_config.json").read_text())
    if args.preflight:
        # Scheduling-only override; output records actual Julia thread count.
        run["threads"] = int(os.environ.get("PREFLIGHT_THREADS",run["threads"]))
        assert 1 <= run["threads"] <= 26
        cases = json.loads((root/"preflight/cases.json").read_text())
        rows = [(i, cases[i]["theta"]) for i in range(args.chunk*2,min(len(cases),args.chunk*2+2))]
        directory = root/"preflight"/str(args.chunk)
    else:
        gate = root/"preflight/quality_gate.json"
        gate_data = json.loads(gate.read_text()) if gate.exists() else {}
        if not gate_data.get("passed") or gate_data.get("manifest_sha256") != digest(root/"manifest.json"):
            raise RuntimeError("Full-map preflight must pass before dataset generation")
        theta = np.load(root/"design/theta.npy",mmap_mode="r")
        first = args.chunk*run["chunk_size"]
        rows = [(i,theta[i]) for i in range(first,min(len(theta),first+run["chunk_size"]))]
        directory = root/"chunks"/("chunk_%05d" % args.chunk)
    if args.chunk < 0 or not rows:
        raise RuntimeError("Invalid or empty chunk")
    directory.mkdir(exist_ok=True)
    lock_name = ("preflight_" if args.preflight else "chunk_")+str(args.chunk)+".lock"
    with (root/"locks"/lock_name).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        packed = directory.with_suffix(".h5")
        if packed.exists():
            with h5py.File(packed,"r") as handle:
                source_hashes = {digest(root/"manifest.json")}
                if args.preflight:
                    source_hashes.add(manifest.get("preflight_generation_manifest_sha256"))
                assert handle.attrs["manifest_sha256"] in source_hashes
                assert len(handle) == len(rows)
            print("Chunk already complete",flush=True)
            return
        start = time.monotonic()
        prior = JointPrior(manifest["prior"])
        for index, values in rows:
            if time.monotonic()-start > run["worker_budget_seconds"]-run["row_timeout_seconds"]-120:
                print("Stopping before walltime; resubmit this chunk",flush=True)
                return
            target = directory/("row_%07d.h5"%index)
            if target.exists():
                with h5py.File(target,"r") as handle:
                    np.testing.assert_array_equal(handle["theta"][:],values)
                    assert handle.attrs["manifest_sha256"] == digest(root/"manifest.json")
                    assert handle.attrs["complete"]
                continue
            assert prior.contains(values), "Row outside frozen joint prior"
            label = ("preflight_" if args.preflight else "row_")+str(index)
            print("Starting",label,flush=True)
            data,status,log = run_row(root,run,values,label)
            store_row(root,target,values,data,status,log,args.preflight or run["retain_full_cl"])
            print("Completed",label,"seconds",status["elapsed_seconds"],flush=True)
        temporary = packed.with_suffix(".tmp.h5")
        with h5py.File(temporary,"w") as dest:
            for index, values in rows:
                with h5py.File(directory/("row_%07d.h5"%index),"r") as source:
                    group = dest.create_group(str(index))
                    for name in source:
                        source.copy(name,group)
                    for name,value in source.attrs.items():
                        group.attrs[name] = value
            dest.attrs["manifest_sha256"] = digest(root/"manifest.json")
        temporary.replace(packed)
        # Delete only verified, freshly packed row files; leave unknown files.
        for index,_ in rows:
            (directory/("row_%07d.h5"%index)).unlink()
        try:
            directory.rmdir()
        except OSError:
            pass
        print("Packed",packed,flush=True)


if __name__ == "__main__":
    main()
