"""Submit remaining calibration lanes and the dependent Fisher/SBI analysis."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pilot-job',required=True)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    receipt=args.root/'submission.json'
    if receipt.exists():
        raise ValueError('Submission already recorded; do not duplicate jobs')
    jobs=[];records=[]
    for worker in range(4):
        command=['qsub','-q','mini','-l','select=1:ncpus=26:mpiprocs=1:mem=64gb',
            '-v',f'RUN_ROOT={args.root},WORKER={worker},CPUS=26',
            '-o',str(args.root/'logs'),str(args.root/'analysis_code/run_calibration.pbs')]
        print(command,flush=True)
        job=subprocess.check_output(command,text=True).strip() if args.submit else f'dry{worker}'
        jobs.append(job);records.append(dict(worker=worker,job=job,command=command))
        if args.submit:
            receipt.write_text(json.dumps(dict(pilot_job=args.pilot_job,jobs=records),indent=2)+'\n')
    command=['qsub','-q','mini','-W','depend=afterok:'+':'.join([args.pilot_job]+jobs),
        '-v',f'RUN_ROOT={args.root},CPUS=26','-o',str(args.root/'logs'),
        str(args.root/'analysis_code/run_analysis.pbs')]
    print(command,flush=True)
    job=subprocess.check_output(command,text=True).strip() if args.submit else 'dry_analysis'
    records.append(dict(stage='analysis',job=job,command=command))
    if args.submit:
        receipt.write_text(json.dumps(dict(pilot_job=args.pilot_job,jobs=records),indent=2)+'\n')


if __name__=='__main__':
    main()
