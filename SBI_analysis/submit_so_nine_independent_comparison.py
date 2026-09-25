#!/usr/bin/env python3
"""Submit one matched-analysis stage with explicit dependencies; dry run by default.

Calibration must be initialized first. Run prepare after calibration is complete;
run train after preparation is complete. Submission never implies completion.
"""
import argparse
import json
import shlex
import subprocess
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("calibration","prepare","train"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--calibration-root",type=Path,required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--generation-manifest",type=Path)
    p.add_argument("--python",default="python3")
    p.add_argument("--julia",default="julia")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--submit",action="store_true",help="Actually call qsub; default prints commands")
    args=p.parse_args()
    if args.workers<1 or args.workers>5:
        p.error("Use 1..5 simultaneous calibration workers")
    # Logs live next to the root: the preparation stage requires a fresh root.
    logs=args.root.with_name(args.root.name+"_pbs")
    logs.mkdir(parents=True,exist_ok=True)
    record=logs/f"{args.stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.jsonl"
    base=dict(PROJECT_DIR=str(args.project.resolve()),RUN_ROOT=str(args.root.resolve()),
        CALIBRATION_ROOT=str(args.calibration_root.resolve()),PYTHON=args.python,JULIA=args.julia)
    def submit(stage,extra=None,dependencies=()):
        variables=dict(base,STAGE=stage,**(extra or {}))
        if any(any(c in str(v) for c in (",","\n","\r")) for v in variables.values()):
            raise ValueError("PBS environment values cannot contain commas/newlines")
        command=["qsub","-q","mini","-N","SO9_"+stage[:8],"-o",str(logs.resolve()),
            "-l","select=1:ncpus=26:mpiprocs=1:mem=128gb","-l","walltime=23:59:00",
            "-v",",".join(f"{k}={v}" for k,v in variables.items())]
        if dependencies:
            command += ["-W","depend=afterok:"+":".join(dependencies)]
        command += [str(args.project/"SBI_analysis/run_so_nine_independent_stage.pbs")]
        print(shlex.join(command),flush=True)
        job=subprocess.check_output(command,text=True).strip() if args.submit else f"dry{len(jobs)}"
        if args.submit:
            with record.open("a") as f:
                f.write(json.dumps(dict(job_id=job,stage=stage,command=command))+"\n")
        jobs.append(job)
        return job
    # Prevent accidentally overlapping submissions from this wrapper. Finished
    # IDs disappear on some PBS installations, so missing IDs are not completion.
    if args.submit:
        for path in logs.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                old=json.loads(line)
                status=subprocess.run(["qstat","-f",old["job_id"]],text=True,capture_output=True)
                if status.returncode==0 and any(f"job_state = {s}" in status.stdout for s in ("Q","R","H","W","T","E")):
                    raise RuntimeError(f"Previous campaign job {old['job_id']} is active; wait before submitting another stage")
    jobs=[]
    if args.stage=="calibration":
        if not (args.calibration_root/"manifest.json").is_file():
            p.error("Initialize the calibration design first")
        dependencies=[submit("calibration",dict(WORKER=i,WORKERS=args.workers)) for i in range(args.workers)]
        submit("combine",dependencies=dependencies)
    elif args.stage=="prepare":
        if not args.dataset or not args.generation_manifest:
            p.error("prepare needs --dataset and --generation-manifest")
        if not (args.calibration_root/"calibration_complete.json").is_file():
            p.error("Calibration has not completed")
        submit("prepare",dict(DATASET=str(args.dataset.resolve()),GENERATION_MANIFEST=str(args.generation_manifest.resolve())))
    else:
        config=json.loads((args.root/"experiment.json").read_text())
        dependencies=[]
        # One lane per method; each method
        # receives identical architecture, row counts, splits and epoch budget.
        for method in config["methods"]:
            previous=[]
            for n in config["sizes"]:
                job=submit("train_evaluate",dict(N_TRAIN=n,METHOD=method),previous)
                previous=[job]
            dependencies+=previous
        submit("summarize",dependencies=dependencies)
    print("Submitted job receipt: "+str(record) if args.submit else "Dry run only. Add --submit to launch.")


if __name__=="__main__":
    main()
