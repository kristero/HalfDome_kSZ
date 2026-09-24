"""Short PBS dispatch stages; dependencies never consume a worker while waiting."""
import argparse
import json
import subprocess
from launch import ROOT,atomic_json


def submit(name,script,dependencies=(),variables=None,queue=None):
    cmd=['qsub','-N',name,'-o',str(ROOT/(name+'.scheduler.log'))]
    if queue:cmd+=['-q',queue]
    if dependencies:cmd+=['-W','depend=afterok:'+':'.join(dependencies)]
    if variables:cmd+=['-v',','.join(k+'='+str(v) for k,v in variables.items())]
    job=subprocess.check_output(cmd+[str(ROOT/script)],text=True).strip()
    with (ROOT/'submission_journal.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(name=name,job=job,command=cmd+[str(ROOT/script)]))+'\n')
    return job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['spots','production'])
    args=parser.parse_args()
    marker=ROOT/(args.phase+'_submission.json')
    if marker.exists():raise RuntimeError('Submission already recorded; refusing duplicate jobs')
    if args.phase=='spots':
        from manage import prepare_spots
        prepare_spots()
        jobs={}
        for group in range(2):
            jobs['spot'+str(group)]=submit('tsz256_spot'+str(group),'spot.pbs',variables={'SPOT':group},queue='mini2')
        initial=json.loads((ROOT/'test_submission.json').read_text())
        software=json.loads((ROOT/'software_submission.json').read_text())['job']
        dependencies=list(jobs.values())+[initial['anchors'],software]
        # Completed job IDs may already have left PBS history. The numerical
        # gate checks files and exit statuses too; only unfinished jobs need a
        # scheduler dependency.
        active=[]
        for job in dependencies:
            q=subprocess.run(['qstat','-f',job],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if q.returncode==0:active.append(job)
        jobs['final_dispatch']=submit('tsz256_start','start.pbs',active)
    else:
        import final_gate
        from manage import prepare_batches
        final_gate.main()
        prepare_batches(workers=6,batch_size=4)
        jobs={}
        for number in range(6):
            jobs['worker'+str(number)]=submit('tsz256_w'+str(number),'production.pbs',variables={'WORKER':number},
                queue='mini2' if number>=4 else 'mini')
        # The last active worker submits analysis when all 256 rows are present.
        # This avoids waiting for unused workers that never left the queue.
    atomic_json(marker,jobs)
    print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
