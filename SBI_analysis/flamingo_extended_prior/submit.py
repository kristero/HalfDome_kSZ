"""Print or submit a bounded wave of PBS jobs; rerun to resume pending chunks.

Default is dry-run. --submit actually calls qsub. The run size comes only from
the frozen run_config.json, so switching to 524k cannot happen accidentally.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from prior import digest, write_json
from worker import verify_frozen


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--max-jobs",type=int,default=4)
    parser.add_argument("--queue",default="mini")
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--all",dest="all_chunks",action="store_true",
                        help="Queue all chunks in serial dependency chains, with bounded concurrency")
    parser.add_argument("--preflight-threads",type=int,default=26)
    args = parser.parse_args()
    root = args.root.resolve()
    verify_frozen(root)
    if args.max_jobs < 1 or args.max_jobs > 6:
        parser.error("max-jobs must be in 1..6 (mini per-user limit)")
    if not 1 <= args.preflight_threads <= 26:
        parser.error("preflight-threads must be in 1..26")
    run = json.loads((root/"run_config.json").read_text())
    if args.preflight:
        count = (len(json.loads((root/"preflight/cases.json").read_text()))+1)//2
    else:
        gate = json.loads((root/"preflight/quality_gate.json").read_text())
        assert gate["passed"] and gate["manifest_sha256"] == digest(root/"manifest.json")
        count = (run["count"]+run["chunk_size"]-1)//run["chunk_size"]
    # Live PBS state protects against duplicate submission after this driver exits.
    selection = subprocess.run(["qselect","-u",getpass.getuser()],capture_output=True,text=True)
    job_ids = selection.stdout.split()
    if selection.returncode and selection.stderr.strip():
        raise RuntimeError(selection.stderr)
    jobs = {}
    if job_ids:
        state = subprocess.run(["qstat","-f","-F","json"]+job_ids,
                               capture_output=True,text=True)
        # A job can finish between qselect and qstat. Still inspect all jobs
        # returned, and fail closed if PBS gave no parseable state.
        jobs = json.loads(state.stdout).get("Jobs",{})
    prefix = "xp"+digest(root/"manifest.json")[:6]+("p" if args.preflight else "d")
    active = {j["Job_Name"] for j in jobs.values() if j.get("job_state") not in ("F","X")}
    available = max(0,args.max_jobs-sum(name.startswith(prefix) for name in active))
    submitted = []
    resources = []
    extra_env = ""
    if args.preflight:
        resources = ["-l","select=1:ncpus=%d:mpiprocs=1:mem=64gb"%args.preflight_threads]
        extra_env = ",PREFLIGHT_THREADS="+str(args.preflight_threads)
    if args.all_chunks:
        if any(name.startswith(prefix) for name in active):
            raise SystemExit("This campaign already has queued/running jobs; wait or use bounded waves")
        # PBS 19.1 on idark rejects both -J ...%4 and max_run_subjobs.
        # Four afterok chains provide a portable, exact concurrency bound.
        previous = [None]*args.max_jobs
        records = []
        record_path = root/"logs"/("all_chunks_submission_"+str(time.time_ns())+".json")
        for chunk in range(count):
            packed = root/("preflight" if args.preflight else "chunks")/(
                str(chunk)+".h5" if args.preflight else "chunk_%05d.h5"%chunk)
            if packed.exists():
                continue
            lane = chunk % args.max_jobs
            env = "RUN_ROOT="+str(root)+",CHUNK="+str(chunk)+",PREFLIGHT="+str(int(args.preflight))+extra_env
            command = ["qsub","-q",args.queue,"-N",prefix+str(chunk),"-v",env,
                       "-o",str(root/"logs")]+resources
            if previous[lane]:
                command += ["-W","depend=afterok:"+previous[lane]]
            command += [str(root/"code/run_worker.pbs")]
            print(" ".join(shlex.quote(v) for v in command),flush=True)
            record = dict(chunk=chunk,lane=lane,command=command)
            if args.submit:
                result = subprocess.run(command,capture_output=True,text=True,check=True)
                record["job_id"] = result.stdout.strip()
                print(record["job_id"],flush=True)
            previous[lane] = record.get("job_id","DRYRUN_CHUNK_%d"%chunk)
            records.append(record)
            if args.submit:
                write_json(record_path,dict(jobs=records,terminal_job_ids=previous,
                                           concurrency=args.max_jobs))
        print("Prepared %d chunks in %d dependency chains"%(len(records),args.max_jobs))
        return
    for chunk in range(count):
        packed = root/("preflight" if args.preflight else "chunks")/(
            str(chunk)+".h5" if args.preflight else "chunk_%05d.h5"%chunk)
        name = prefix+str(chunk)
        if packed.exists() or name in active:
            continue
        if len(submitted) >= available:
            break
        env = "RUN_ROOT="+str(root)+",CHUNK="+str(chunk)+",PREFLIGHT="+str(int(args.preflight))+extra_env
        command = ["qsub","-q",args.queue,"-N",name,"-v",env,
                   "-o",str(root/"logs")]+resources+[str(root/"code/run_worker.pbs")]
        print(" ".join(shlex.quote(v) for v in command),flush=True)
        record = dict(chunk=chunk,command=command,preflight=args.preflight)
        if args.submit:
            result = subprocess.run(command,capture_output=True,text=True,check=True)
            record["job_id"] = result.stdout.strip()
            print(record["job_id"],flush=True)
        submitted.append(record)
    if args.submit:
        write_json(root/"logs"/("submission_"+str(time.time_ns())+".json"),submitted)
    print("Wave size:",len(submitted),"; rerun this command after completion for the next wave.")


if __name__ == "__main__":
    main()
