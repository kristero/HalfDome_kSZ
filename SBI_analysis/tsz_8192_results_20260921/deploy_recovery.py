"""Resume verified controls into a new root and submit disjoint shards."""
from pathlib import Path,PurePosixPath
import hashlib,json,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent/'tsz_8192_validation_recovery_20260921'
REMOTE='/lustre/work/kristero10/tsz_8192_validation_recovery_20260921'

remote_code=r'''
import pathlib,json,shutil,hashlib,subprocess,numpy as np
old=pathlib.Path('/lustre/work/kristero10/tsz_8192_validation_20260920')
new=pathlib.Path('/lustre/work/kristero10/tsz_8192_validation_recovery_20260921')
manifest=json.loads((new/'manifest.json').read_text())
for name,sha in manifest['source_sha256'].items():
 assert hashlib.sha256((new/name).read_bytes()).hexdigest()==sha,name
plan=json.loads((new/'plan.json').read_text());retained=[]
for task in plan['tasks'][:80]:
 path=old/'controls'/f"{task['id']:03d}";marker=path/'status.json'
 if not marker.exists() or json.loads(marker.read_text())['returncode']!=0:continue
 request=json.loads((path/'request.json').read_text());assert request['task']==task
 for name,sha in request['source_sha256'].items():assert hashlib.sha256((new/name).read_bytes()).hexdigest()==sha
 files=[path/'masked_clean_cl.npy',path/'unmasked_clean_cl.npy']
 if task.get('paired_4096'):files.append(path/'paired4096_clean_cl.npy')
 for i in range(task['noise_draws']):
  files.append(path/'noise'/f'{i:03d}.npy');assert (path/'noise'/f'{i:03d}.npy.toml').exists()
  if task.get('paired_4096'):files.append(path/'noise'/f'{i:03d}.npy.paired4096.npy')
 for file in files:
  a=np.load(file);assert a.shape==(7980,) and np.isfinite(a).all(),str(file)
 destination=new/'controls'/path.name
 if not destination.exists():shutil.copytree(path,destination)
 retained.append(dict(task_id=task['id'],files={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
(new/'preserved_results.json').write_text(json.dumps(dict(origin=str(old),retained=retained,
 caveat='Prior localflock caused duplicate identical tasks. Structure and source hashes verified; six clean rechecks queued.'),indent=2))
journal=new/'submission_recovery.json'
state=json.loads(journal.read_text()) if journal.exists() else dict(workers=[],analysis=None,retained=len(retained),diagnostic_256_submitted=False)
def save():journal.write_text(json.dumps(state,indent=2))
while len(state['workers'])<4:
 i=len(state['workers'])
 job=subprocess.check_output(['qsub','-N',f'tszfix{i}','-v',f'TSZ_SHARD={i}',
  '-o',str(new/f'shard_{i}.log'),str(new/'partition.pbs')],text=True).strip()
 state['workers'].append(job);save()
if not state['analysis']:
 state['analysis']=subprocess.check_output(['qsub','-N','tszfix_report','-W','depend=afterany:'+':'.join(state['workers']),
  '-o',str(new/'analysis.log'),str(new/'analyze.pbs')],text=True).strip();save()
print(json.dumps(state,indent=2))
'''

def main():
    subprocess.run([sys.executable,str(ROOT/'prepare_recovery.py')],check=True)
    pbs='''#!/bin/bash
#PBS -N tsz_partition
#PBS -q mini
#PBS -l select=1:ncpus=26:mem=64gb
#PBS -l walltime=23:59:00
#PBS -j oe
set -euo pipefail
cd /lustre/work/kristero10/tsz_8192_validation_recovery_20260921
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=26 MKL_NUM_THREADS=1
/home/anaconda3/bin/python3 partition_worker.py --shard "$TSZ_SHARD"
'''
    (NEW/'partition.pbs').write_bytes(pbs.encode())
    (NEW/'analyze.pbs').write_bytes((NEW/'analyze.pbs').read_text().replace('tsz_8192_validation_20260920','tsz_8192_validation_recovery_20260921').encode())
    files=[p for p in NEW.iterdir() if p.suffix in ['.py','.jl','.pbs','.json'] and p.name!='manifest.json']+list((NEW/'inputs').glob('*'))
    hashes={str(p.relative_to(NEW)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (NEW/'manifest.json').write_text(json.dumps(dict(source_sha256=hashes,physical_producer_changed=False),indent=2))
    with tarfile.open(NEW/'recovery.tar.gz','w:gz') as archive:
        for p in files+[NEW/'manifest.json']:archive.add(p,arcname=str(p.relative_to(NEW)))
    exists=subprocess.run(['ssh','-o','ConnectTimeout=15','idark','test','-e',REMOTE+'/manifest.json'])
    if exists.returncode!=1:raise RuntimeError('Remote root exists or connectivity failed; refusing overwrite')
    subprocess.run(['ssh','idark','mkdir','-p',REMOTE],check=True)
    subprocess.run(['scp',str(NEW/'recovery.tar.gz'),'idark:'+REMOTE+'/recovery.tar.gz'],check=True)
    subprocess.run(['ssh','idark','tar','-xzf',REMOTE+'/recovery.tar.gz','-C',REMOTE],check=True)
    result=subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=remote_code,text=True)
    (ROOT/'results/recovery_submission.json').write_text(result);print(result)

if __name__=='__main__':main()
