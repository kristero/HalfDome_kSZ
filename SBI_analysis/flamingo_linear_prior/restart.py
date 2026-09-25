"""Audited restart: preserve old rows, import compatible rows, submit PBS jobs.

All job selection uses the exact RUN_ROOT variable and the current PBS owner.
No simulation file in the previous run is deleted or overwritten.
"""
import argparse
import fcntl
import getpass
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import h5py
import numpy as np

from prior import digest, write_json
from worker import verify_frozen, BINNED_NAMES


def live_jobs(root=None):
    result = subprocess.run(["qselect", "-u", getpass.getuser()], capture_output=True, text=True)
    ids = result.stdout.split()
    if not ids:
        return {}
    # This PBS client opens a connection per explicit job ID. Query bounded
    # batches to avoid its "Too many open connections" failure at 65 jobs.
    jobs = {}
    for start in range(0, len(ids), 12):
        batch = ids[start:start + 12]
        result = subprocess.run(["qstat", "-f", "-F", "json"] + batch, capture_output=True, text=True)
        if not result.stdout.strip():
            raise RuntimeError("PBS status failed: " + result.stderr.strip())
        jobs.update(json.loads(result.stdout).get("Jobs", {}))
    selected = {}
    for job_id, job in jobs.items():
        if job.get("job_state") in ("F", "X"):
            continue
        variables = job.get("Variable_List", {})
        if isinstance(variables, str):
            variables = dict(item.split("=", 1) for item in variables.split(",") if "=" in item)
        if root is None or variables.get("RUN_ROOT") == str(root):
            if job["Job_Owner"].split("@")[0] != getpass.getuser():
                raise RuntimeError("Refusing to act on another user's job")
            selected[job_id] = job
    return selected


def completed_files(root):
    files = [p for p in (root / "chunks").glob("chunk_*/row_*.h5")
             if re.fullmatch(r"row_\d{7}\.h5", p.name)]
    files += [p for p in (root / "chunks").glob("chunk_*.h5")
              if re.fullmatch(r"chunk_\d{5}\.h5", p.name)]
    return {str(p): digest(p) for p in sorted(files)}


def retire_previous(root, run):
    previous = Path(run["previous_run"]).resolve()
    if previous == root or root in previous.parents or previous in root.parents:
        raise RuntimeError("Previous and replacement run roots must be separate")
    path = root / "audit/previous_run_retirement.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("retired") and not live_jobs(previous):
            print("Previous run already retired", flush=True)
            return
    verify_frozen(previous)
    before = completed_files(previous)
    jobs = live_jobs(previous)
    record = dict(previous_root=str(previous), replacement_root=str(root),
                  completed_files_before=before, jobs_before=jobs, cancelled=[], retired=False,
                  authorization="User requested restart with the broad linear-prior version and retention of compatible profiles")
    write_json(path, record)
    # Descendants/collector have later IDs, so remove them before their parents.
    for job_id in sorted(jobs, key=lambda s: int(s.split(".")[0]), reverse=True):
        current = live_jobs(previous)
        if job_id not in current:
            continue
        result = subprocess.run(["qdel", job_id], capture_output=True, text=True)
        if result.returncode and job_id in live_jobs(previous):
            raise RuntimeError("Could not retire {}: {}".format(job_id, result.stderr))
        record["cancelled"].append(job_id)
        write_json(path, record)
        print("Retired", job_id, current[job_id]["job_state"], flush=True)
    deadline = time.monotonic() + 90
    while live_jobs(previous):
        if time.monotonic() > deadline:
            raise RuntimeError("Previous jobs are still leaving PBS; rerun retire")
        time.sleep(2)
    after = completed_files(previous)
    assert all(after.get(name) == sha for name, sha in before.items())
    verify_frozen(previous)
    record.update(retired=True, completed_files_after=after,
                  all_preexisting_completed_files_unchanged=True,
                  completed_file_count=len(after), completed_at_unix=time.time())
    write_json(path, record)
    print("Preserved", len(after), "completed files; old PBS jobs retired", flush=True)


def import_rows(root, run):
    previous = Path(run["previous_run"]).resolve()
    if live_jobs(previous):
        raise RuntimeError("Retire previous workers before auditing imports")
    verify_frozen(previous)
    for name in ("paint_row.jl", "stable_los.jl", "independent_noise.jl", "noise_seeds.py"):
        assert digest(previous / "code" / name) == digest(root / "code" / name), name
    previous_run = json.loads((previous / "run_config.json").read_text())
    for key in ("campaign", "catalogue", "training_noise_mode", "noise_master_seed", "threads"):
        assert previous_run[key] == run[key], key
    theta = np.load(root / "design/theta.npy")
    source_rows = np.load(root / "design/source_old_row.npy")
    seed_ids = np.load(root / "design/noise_seed_row_ids.npy")
    seeds = np.load(root / "design/noise_split_seeds.npy")
    old_manifest_hash = digest(previous / "manifest.json")
    new_manifest_hash = digest(root / "manifest.json")
    copied, unavailable = [], []
    for new_id in np.flatnonzero(source_rows >= 0):
        old_id = int(source_rows[new_id])
        old_chunk = old_id // previous_run["chunk_size"]
        source = previous / "chunks" / ("chunk_%05d" % old_chunk) / ("row_%07d.h5" % old_id)
        group_name = None
        if not source.exists():
            source = previous / "chunks" / ("chunk_%05d.h5" % old_chunk)
            group_name = str(old_id)
        if not source.exists():
            unavailable.append(old_id)
            continue
        directory = root / "chunks" / ("chunk_%05d" % (new_id // run["chunk_size"]))
        directory.mkdir(exist_ok=True)
        target = directory / ("row_%07d.h5" % new_id)
        provenance = dict(source=str(source), source_group=group_name, source_sha256=digest(source),
                          old_row_id=old_id, new_row_id=int(new_id), source_manifest_sha256=old_manifest_hash,
                          selection="P0*xc/240 thinning of the full frozen old design; independent of completion")
        with h5py.File(source, "r") as handle:
            src = handle[group_name] if group_name is not None else handle
            assert src.attrs["complete"] and src.attrs["manifest_sha256"] == old_manifest_hash
            np.testing.assert_array_equal(src["theta"][:], theta[new_id])
            status = json.loads(src.attrs["status_json"])
            assert status["exit_code"] == 0
            assert status["noise"]["seed_row_id"] == int(seed_ids[new_id]) == old_id
            np.testing.assert_array_equal(status["noise"]["split_seeds"], seeds[new_id])
            assert status["numerics"]["maximum_column_relative_error"] < 1e-7
            if target.exists():
                with h5py.File(target, "r") as existing:
                    assert existing.attrs["manifest_sha256"] == new_manifest_hash
                    assert json.loads(existing.attrs["reuse_provenance_json"]) == provenance
            else:
                temporary = target.with_suffix(".tmp.h5")
                with h5py.File(temporary, "w") as dest:
                    for name in src:
                        src.copy(name, dest)
                    for key, value in src.attrs.items():
                        dest.attrs[key] = value
                    dest.attrs["manifest_sha256"] = new_manifest_hash
                    dest.attrs["generation_manifest_sha256"] = old_manifest_hash
                    dest.attrs["reuse_provenance_json"] = json.dumps(provenance)
                temporary.replace(target)
            with h5py.File(target, "r") as dest:
                for name in ("theta",) + tuple(BINNED_NAMES.values()):
                    np.testing.assert_array_equal(dest[name][:], src[name][:])
                    assert np.isfinite(dest[name][:]).all()
        copied.append(provenance)
    old_files = completed_files(previous)
    retirement = json.loads((root / "audit/previous_run_retirement.json").read_text())
    assert old_files == retirement["completed_files_after"]
    report = dict(passed=True, manifest_sha256=new_manifest_hash, reused_count=len(copied),
                  selected_old_points=int((source_rows >= 0).sum()), unavailable_old_selected_count=len(unavailable),
                  reused=copied, all_previous_completed_files_preserved=True,
                  previous_completed_files=len(old_files))
    write_json(root / "audit/reuse_import.json", report)
    print("Imported", len(copied), "exact profiles and noise realizations", flush=True)


def submit_preflight(root):
    if live_jobs(root):
        print("Replacement already has active jobs", flush=True)
        return
    cases = json.loads((root / "preflight/cases.json").read_text())
    records = []
    for chunk in range((len(cases) + 1) // 2):
        if (root / "preflight" / (str(chunk) + ".h5")).exists():
            continue
        command = ["qsub", "-q", "mini", "-N", "lp" + digest(root / "manifest.json")[:6] + "p" + str(chunk),
                   "-v", "RUN_ROOT={},CHUNK={},PREFLIGHT=1".format(root, chunk),
                   "-o", str(root / "logs"), str(root / "code/run_worker.pbs")]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        records.append(dict(chunk=chunk, job_id=result.stdout.strip(), command=command))
        write_json(root / "audit/preflight_submission.json", dict(jobs=records))
        print("Preflight", chunk, result.stdout.strip(), flush=True)


def submit_production(root, run, normal, background):
    manifest_hash = digest(root / "manifest.json")
    gate = json.loads((root / "preflight/quality_gate.json").read_text())
    reuse = json.loads((root / "audit/reuse_import.json").read_text())
    assert gate["passed"] and gate["manifest_sha256"] == manifest_hash
    assert reuse["passed"] and reuse["manifest_sha256"] == manifest_hash
    if live_jobs(Path(run["previous_run"])):
        raise RuntimeError("Previous production still active")
    journal_path = root / "audit/production_submission.json"
    if journal_path.exists():
        raise RuntimeError("Submission journal already exists; inspect it before resubmission")
    if live_jobs(root):
        raise RuntimeError("Wait for preflight PBS jobs to finish")
    if not 1 <= normal <= 6 or not 0 <= background <= 16:
        raise ValueError("Use 1..6 mini workers and 0..16 mini_B workers")
    lanes = ["mini"] * normal + ["mini_B"] * background
    previous = [None] * len(lanes)
    records = []
    journal = dict(root=str(root), manifest_sha256=manifest_hash, jobs=records,
                   phase="staging_held_jobs", concurrency=len(lanes), queues=lanes)
    write_json(journal_path, journal)
    count = (run["count"] + run["chunk_size"] - 1) // run["chunk_size"]
    for chunk in range(count):
        if (root / "chunks" / ("chunk_%05d.h5" % chunk)).exists():
            continue
        lane = chunk % len(lanes)
        command = ["qsub", "-h", "-q", lanes[lane], "-N", "lp" + manifest_hash[:6] + "d" + str(chunk),
                   "-v", "RUN_ROOT={},CHUNK={},PREFLIGHT=0".format(root, chunk), "-o", str(root / "logs")]
        if previous[lane]:
            command += ["-W", "depend=afterok:" + previous[lane]]
        command += [str(root / "code/run_worker.pbs")]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        records.append(dict(chunk=chunk, lane=lane, queue=lanes[lane], job_id=job_id, command=command,
                            parent=previous[lane]))
        previous[lane] = job_id
        write_json(journal_path, journal)
    terminals = [job for job in previous if job]
    command = ["qsub", "-h", "-q", "mini", "-N", "linear_collect", "-v", "RUN_ROOT=" + str(root),
               "-o", str(root / "logs"), "-W", "depend=afterok:" + ":".join(terminals),
               str(root / "code/run_collect.pbs")]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    collector = result.stdout.strip()
    journal.update(collector_job_id=collector, collector_command=command, terminal_job_ids=terminals)
    write_json(journal_path, journal)
    jobs = live_jobs(root)
    assert len({r["chunk"] for r in records}) == count
    for record in records:
        job = jobs[record["job_id"]]
        assert "u" in job["Hold_Types"]
        assert job["queue"] == record["queue"] and int(job["Resource_List"]["ncpus"]) == 26
        assert job["Resource_List"]["walltime"] == "23:59:00"
        assert job["Resource_List"]["mem"] == "64gb"
        if record["parent"]:
            assert record["parent"] in job.get("depend", "")
    assert "u" in jobs[collector]["Hold_Types"]
    verify_frozen(root)
    journal["phase"] = "releasing"
    journal["released"] = []
    write_json(journal_path, journal)
    for job_id in [collector] + [record["job_id"] for record in reversed(records)]:
        subprocess.run(["qrls", "-h", "u", job_id], capture_output=True, text=True, check=True)
        journal["released"].append(job_id)
        write_json(journal_path, journal)
    journal["phase"] = "released"
    journal["submitted_at_unix"] = time.time()
    write_json(journal_path, journal)
    print("Released", len(records), "production chunks with concurrency cap", len(lanes),
          "and collector", collector, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--action", choices=("retire", "import", "preflight", "production", "status"), required=True)
    parser.add_argument("--normal-workers", type=int, default=6)
    parser.add_argument("--background-workers", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    verify_frozen(root)
    run = json.loads((root / "run_config.json").read_text())
    with (root / "locks/restart.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "retire":
            retire_previous(root, run)
        elif args.action == "import":
            import_rows(root, run)
        elif args.action == "preflight":
            submit_preflight(root)
        elif args.action == "production":
            submit_production(root, run, args.normal_workers, args.background_workers)
        else:
            jobs = live_jobs(root)
            states = {state:sum(j["job_state"] == state for j in jobs.values()) for state in ("R", "Q", "H", "E")}
            print(json.dumps(dict(states=states, jobs={k:{field:v.get(field) for field in
                  ("Job_Name", "job_state", "queue", "comment")} for k,v in jobs.items()},
                  completed_files=len(completed_files(root))), indent=2))


if __name__ == "__main__":
    main()
