#!/usr/bin/env python3
"""Submit a staged, isolated five-job comparison; dry-run unless --submit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if any(character in str(campaign) for character in ",\n "):
        raise ValueError("PBS campaign path must not contain spaces, commas or newlines")
    code, logs = campaign / "code", campaign / "logs"
    script = code / "run_so_unbinned_moped_stage.pbs"
    if not script.is_file():
        raise FileNotFoundError(script)
    receipt_path = campaign / "submission.json"
    if receipt_path.exists() or (campaign / "analysis").exists():
        raise ValueError("Use a fresh campaign to avoid duplicate submissions")
    plan = [("prepare", "unbinned_moped", []),
            ("train_evaluate", "unbinned_moped", [0]),
            ("evaluate", "bins40", [0]), ("evaluate", "moped40", [0]),
            ("summarize", "unbinned_moped", [1, 2, 3])]
    receipt = dict(campaign=str(campaign), jobs=[], source_sha256={
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(code.iterdir()) if path.is_file()})
    for number, (stage, method, dependencies) in enumerate(plan):
        command = ["qsub", "-N", f"U9_{number}_{method[:5]}", "-v",
                   f"CAMPAIGN={campaign},STAGE={stage},METHOD={method}"]
        if stage == "prepare":
            command += ["-l", "select=1:ncpus=26:mpiprocs=1:mem=64gb"]
        if dependencies:
            job_ids = [receipt["jobs"][i]["job_id"] for i in dependencies]
            command += ["-W", "depend=afterok:" + ":".join(job_ids)]
        command.append(str(script))
        if args.submit:
            logs.mkdir(exist_ok=True)
            job_id = subprocess.check_output(command, text=True).strip()
            if not job_id or "\n" in job_id:
                raise RuntimeError(f"Unexpected qsub output: {job_id!r}")
        else:
            job_id = f"DRY_RUN_{number}"
        receipt["jobs"].append(dict(stage=stage, method=method, job_id=job_id, command=command))
        if args.submit:
            receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps(receipt["jobs"][-1]), flush=True)


if __name__ == "__main__":
    main()
