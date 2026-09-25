"""Save a scheduler snapshot and conservative, explicitly conditional capacity."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def remote_snapshot():
    result = dict(utc=datetime.now(timezone.utc).isoformat())
    raw = subprocess.check_output(['qstat','-f','-F','json'], text=True)
    data = json.loads(raw)
    jobs = []
    for identifier, record in data.get('Jobs', {}).items():
        jobs.append(dict(id=identifier, state=record.get('job_state'),
            owner=record.get('Job_Owner','').split('@')[0],
            name=record.get('Job_Name'), queue=record.get('queue'),
            requested=record.get('Resource_List',{}),
            used=record.get('resources_used',{}),
            estimated=record.get('estimated',{}), comment=record.get('comment','')))
    result['jobs']=jobs
    result['nodes']=subprocess.check_output(['pbsnodes','-aSj'],text=True)
    result['queues']=subprocess.check_output(['qstat','-Qf','mini','mini2','small','mini_B'],text=True)
    print(json.dumps(result))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--remote',action='store_true');args=ap.parse_args()
    if args.remote:
        remote_snapshot()
    else:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
            'idark','/home/anaconda3/bin/python3','-','--remote'],
            input=Path(__file__).read_text(),text=True,capture_output=True,check=True)
        result=json.loads(proc.stdout)
        (ROOT/'results').mkdir(exist_ok=True)
        (ROOT/'results/cluster_snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
        mine=[r for r in result['jobs'] if r['owner']=='kristero10']
        print(json.dumps(dict(utc=result['utc'],own_jobs=mine,
            active_records=len(result['jobs'])),indent=2))
