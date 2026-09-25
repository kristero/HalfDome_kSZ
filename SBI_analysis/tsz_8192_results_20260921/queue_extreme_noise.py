"""Matched 128-draw noise controls for faint and bright prior extremes."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'tsz_8192_validation_20260920'
NEW=HERE.parent/'tsz_8192_noise_extremes_20260921'
REMOTE='/lustre/work/kristero10/tsz_8192_noise_extremes_20260921'

def main():
    NEW.mkdir(exist_ok=True);(NEW/'inputs').mkdir(exist_ok=True)
    for name in ['control.jl','fullsky_test.jl','spherical_truncation_profiles.jl','worker.py','external_dependencies.json']:
        shutil.copy2(OLD/name,NEW/name)
    shutil.copy2(OLD/'inputs/frozen_unbinned_moped_transform.npz',NEW/'inputs/frozen_unbinned_moped_transform.npz')
    plan=json.loads((OLD/'plan.json').read_text());tasks=[]
    for identifier,original in [(86,2),(87,4)]:
        task=plan['tasks'][original].copy();task.update(id=identifier,noise_draws=128,paired_4096=True)
        task.pop('identity_sha256',None)
        task['identity_sha256']=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest();tasks.append(task)
    (NEW/'plan.json').write_text(json.dumps(dict(tasks=tasks,scope='Row-specific conditional noise at compact and extended-shallow extremes'),indent=2))
    (NEW/'row.py').write_bytes(b'''import json,sys,pathlib,hashlib
import worker
root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['source_sha256'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
task=next(t for t in json.loads((root/'plan.json').read_text())['tasks'] if t['id']==int(sys.argv[1]))
(root/('claim_'+str(task['id']))).mkdir()
worker.run_task(task)
''')
    pbs='''#!/bin/bash
#PBS -N tsz_extreme_noise
#PBS -q mini
#PBS -l select=1:ncpus=26:mem=64gb
#PBS -l walltime=06:00:00
#PBS -j oe
set -euo pipefail
cd /lustre/work/kristero10/tsz_8192_noise_extremes_20260921
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=26 MKL_NUM_THREADS=1
/home/anaconda3/bin/python3 row.py "$TASK_ID"
'''
    (NEW/'row.pbs').write_bytes(pbs.encode())
    files=[p for p in NEW.iterdir() if p.is_file() and p.name not in ['source.tar.gz','manifest.json']]+list((NEW/'inputs').iterdir())
    hashes={str(p.relative_to(NEW)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (NEW/'manifest.json').write_text(json.dumps(dict(source_sha256=hashes),indent=2))
    with tarfile.open(NEW/'source.tar.gz','w:gz') as archive:
        for p in files+[NEW/'manifest.json']:archive.add(p,arcname=str(p.relative_to(NEW)))
    exists=subprocess.run(['ssh','idark','test','-e',REMOTE+'/manifest.json'])
    if exists.returncode!=1:raise RuntimeError('Remote root exists or connectivity failed')
    subprocess.run(['ssh','idark','mkdir','-p',REMOTE],check=True)
    subprocess.run(['scp',str(NEW/'source.tar.gz'),'idark:'+REMOTE+'/source.tar.gz'],check=True)
    subprocess.run(['ssh','idark','tar','-xzf',REMOTE+'/source.tar.gz','-C',REMOTE],check=True)
    jobs=[]
    for identifier in [86,87]:
        job=subprocess.check_output(['ssh','idark','qsub','-v','TASK_ID='+str(identifier),'-o',REMOTE+f'/task_{identifier}.log',REMOTE+'/row.pbs'],text=True).strip()
        jobs.append(dict(task=identifier,job=job))
        (HERE/'results/extreme_noise_submission.json').write_text(json.dumps(jobs,indent=2))
    print(json.dumps(jobs,indent=2))

if __name__=='__main__':main()
