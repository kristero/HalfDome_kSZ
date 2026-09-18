"""Expand the initial four production lanes without editing frozen inputs.

This administrative tool is deliberately outside the generation code manifest.
It keeps running chunks, stages replacement pending jobs under user holds, then
removes the old pending schedule and releases the replacements. Every submission
and cancellation is recorded immediately. It never redraws parameters or seeds.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode:
        raise RuntimeError("Command failed: {}\n{}".format(command, result.stderr))
    return result.stdout.strip()


def jobs():
    return json.loads(run(["qstat", "-f", "-F", "json"])).get("Jobs", {})


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_snapshot(root):
    manifest = json.loads((root / "manifest.json").read_text())
    paths = [root / "manifest.json", root / "run_config.json"]
    for folder, key in (("code", "code_sha256"), ("design", "design_sha256")):
        paths.extend(root / folder / name for name in manifest[key])
    return {str(path.relative_to(root)): digest(path) for path in paths}


def plan(root, normal, background):
    sys.path.insert(0, str(root / "code"))
    from worker import verify_frozen
    verify_frozen(root)
    config = json.loads((root / "run_config.json").read_text())
    gate = json.loads((root / "preflight/quality_gate.json").read_text())
    assert gate["passed"] and gate["manifest_sha256"] == digest(root / "manifest.json")
    previous = json.loads((root / "production_submission.json").read_text())
    records = sorted(previous["jobs"], key=lambda row: row["chunk"])
    count = (config["count"] + config["chunk_size"] - 1) // config["chunk_size"]
    assert [row["chunk"] for row in records] == list(range(count))
    live = jobs()
    queue_info = json.loads(run(["qstat", "-Qf", "-F", "json"]))["Queue"]
    assert 4 <= normal <= int(queue_info["mini"]["max_user_run"])
    assert 0 <= background <= int(queue_info["mini_B"]["max_user_run"])
    assert normal + background > 4
    selected = {}
    for row in records:
        job = live[row["job_id"]]
        environment = job["Variable_List"]
        assert environment["RUN_ROOT"] == str(root)
        assert int(environment["CHUNK"]) == row["chunk"]
        assert int(environment.get("PREFLIGHT", 0)) == 0
        assert job["queue"] == "mini"
        assert job["Resource_List"]["ncpus"] == config["threads"] == 26
        assert job["job_state"] == ("R" if row["chunk"] < 4 else "H")
        assert "u" not in job.get("Hold_Types", "")
        selected[row["job_id"]] = job
    collector = previous["collector_job_id"]
    assert live[collector]["job_state"] == "H"
    assert "u" not in live[collector].get("Hold_Types", "")
    assert live[collector]["Variable_List"]["RUN_ROOT"] == str(root)
    selected[collector] = live[collector]
    owner = selected[records[0]["job_id"]]["Job_Owner"].split("@", 1)[0]
    other = [job for name, job in live.items() if name not in selected
             and job.get("Job_Owner", "").split("@", 1)[0] == owner
             and job.get("queue") in ("mini", "mini_B")]
    assert not other, "Account has other work; recalculate the lane allocation"
    return dict(root=str(root), created_utc=datetime.datetime.utcnow().isoformat()+"Z",
        normal_workers=normal, background_workers=background, concurrency=normal+background,
        row_count=config["count"], chunk_size=config["chunk_size"],
        original_jobs=records, original_collector=collector, original_pbs=selected,
        queue_limits={q:queue_info[q] for q in ("mini", "mini_B")},
        frozen_hashes=frozen_snapshot(root),
        completed_row_hashes={str(p.relative_to(root)):digest(p)
                              for p in (root / "chunks").glob("chunk_*/row_*.h5")},
        replacements=[], events=[], phase="planned")


def apply(root, audit, state):
    record_path = audit / "schedule.json"
    originals = state["original_jobs"]
    old_pending = [row["job_id"] for row in originals[4:]]
    old_pending.append(state["original_collector"])

    def checkpoint(phase=None):
        if phase:
            state["phase"] = phase
        write_json(record_path, state)

    def action(command):
        output = run(command)
        state["events"].append(dict(command=command, output=output,
            utc=datetime.datetime.utcnow().isoformat()+"Z"))
        checkpoint()
        return output

    try:
        # Prevent the old pending jobs from starting while replacement jobs are staged.
        action(["qhold", "-h", "u"] + old_pending)
        checkpoint("old_pending_held")
        live = jobs()
        assert all(live[j]["job_state"] == "H" and "u" in live[j]["Hold_Types"]
                   for j in old_pending)
        previous = [None] * state["concurrency"]
        for row in originals[:4]:
            previous[row["chunk"]] = row["job_id"]
        for row in originals[4:]:
            chunk = row["chunk"]
            lane = chunk % state["concurrency"]
            queue = "mini" if lane < state["normal_workers"] else "mini_B"
            environment = "RUN_ROOT={},CHUNK={},PREFLIGHT=0".format(root, chunk)
            command = ["qsub", "-h", "-q", queue, "-N",
                       state["original_pbs"][row["job_id"]]["Job_Name"],
                       "-v", environment, "-o", str(root / "logs")]
            parent = previous[lane]
            if parent:
                command += ["-W", "depend=afterok:" + parent]
            command.append(str(root / "code/run_worker.pbs"))
            job_id = action(command)
            state["replacements"].append(dict(chunk=chunk, lane=lane, queue=queue,
                job_id=job_id, predecessor=parent, command=command))
            previous[lane] = job_id
            checkpoint()
        state["terminal_job_ids"] = previous
        command = ["qsub", "-h", "-q", "mini", "-W", "depend=afterok:"+":".join(previous),
                   "-v", "RUN_ROOT="+str(root), "-o", str(root / "logs"),
                   str(root / "code/run_collect.pbs")]
        state["collector_job_id"] = action(command)
        checkpoint("replacements_staged")
        current = jobs()
        new_ids = [row["job_id"] for row in state["replacements"]]
        new_ids.append(state["collector_job_id"])
        assert all(current[j]["job_state"] == "H" and "u" in current[j]["Hold_Types"]
                   for j in new_ids)
        assert all(current[j]["job_state"] == "H" for j in old_pending)
        parents = {row["job_id"]: None for row in originals[:4]}
        parents.update({row["job_id"]:row["predecessor"] for row in state["replacements"]})
        covered = set()
        for terminal in previous:
            chain = set()
            node = terminal
            while node is not None:
                assert node in parents and node not in chain
                chain.add(node)
                node = parents[node]
            assert not covered.intersection(chain), "Two lanes share a production job"
            covered.update(chain)
        assert covered == set(parents) and len(covered) == len(originals)
        for row in state["replacements"]:
            job = current[row["job_id"]]
            assert job["queue"] == row["queue"]
            assert int(job["Variable_List"]["CHUNK"]) == row["chunk"]
            assert job["Variable_List"]["RUN_ROOT"] == str(root)
            assert job["Resource_List"]["ncpus"] == 26
            assert job["Resource_List"]["mem"] == "64gb"
            assert job["Resource_List"]["walltime"] == "23:59:00"
        assert frozen_snapshot(root) == state["frozen_hashes"]
        print("Replacement schedule staged under holds:", len(new_ids), "jobs", flush=True)

        # Delete descendants first, so PBS does not automatically cancel jobs
        # still present in our cancellation list. Running ancestors are retained.
        checkpoint("retiring_old_pending")
        action(["qdel", state["original_collector"]])
        for row in reversed(originals[4:]):
            action(["qdel", row["job_id"]])
        current = jobs()
        assert not set(old_pending).intersection(current)
        assert all(row["job_id"] in current for row in originals[:4])
        assert len({row["chunk"] for row in state["replacements"]}) == len(originals)-4
        checkpoint("old_pending_retired")
        action(["qrls", "-h", "u"] + new_ids)
        checkpoint("released")
        current = jobs()
        assert all(j in current and "u" not in current[j].get("Hold_Types", "") for j in new_ids)
        assert frozen_snapshot(root) == state["frozen_hashes"]
        assert all(digest(root / name) == value for name, value in state["completed_row_hashes"].items())
        state["checks"] = dict(frozen_inputs_unchanged=True, existing_rows_unchanged=True,
            existing_rows_checked=len(state["completed_row_hashes"]),
            each_chunk_scheduled_once=True, collector_waits_for_all_lanes=True)
        checkpoint("verified")
    except Exception as error:
        state["error"] = str(error)
        checkpoint()
        if state["phase"] in ("planned", "old_pending_held", "replacements_staged"):
            # Before retirement, restore the old schedule on a staging failure.
            rollback_ids = [r["job_id"] for r in state["replacements"]]
            if state.get("collector_job_id"):
                rollback_ids.append(state["collector_job_id"])
            for job_id in reversed(rollback_ids):
                subprocess.run(["qdel", job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["qrls", "-h", "u"]+old_pending,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            checkpoint("staging_failed_original_schedule_restored")
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--normal-workers", type=int, default=6)
    parser.add_argument("--background-workers", type=int, default=16)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root, audit = args.root.resolve(), args.audit_root.resolve()
    assert audit != root and audit != root / "code"
    audit.mkdir(parents=True, exist_ok=True)
    assert not (audit / "schedule.json").exists(), "Inspect the existing transaction before retrying"
    state = plan(root, args.normal_workers, args.background_workers)
    write_json(audit / "plan.json", state)
    print(json.dumps({key:state[key] for key in ("row_count", "chunk_size", "normal_workers",
                     "background_workers", "concurrency")}, indent=2), flush=True)
    if args.apply:
        apply(root, audit, state)
        print(json.dumps(dict(phase=state["phase"], checks=state["checks"],
            new_heads=[r for r in state["replacements"] if r["predecessor"] is None],
            collector=state["collector_job_id"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
