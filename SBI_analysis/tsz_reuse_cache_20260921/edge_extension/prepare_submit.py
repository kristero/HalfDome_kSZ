"""Freeze and queue the derivative-continuity hypothesis as a separate test."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/'+ROOT.parent.name+'/edge_extension'
nodes,weights=np.polynomial.legendre.leggauss(16)
np.savetxt(ROOT/'legendre16.csv',np.c_[(nodes+1)/2,weights/2],delimiter=',',fmt='%.18e')
source=json.loads((ROOT.parent/'plan.json').read_text())['tasks'][2]
plan=[]
for i,n in enumerate(([512,256,128],[256,128,64],[1024,512,256])):
    task=dict(source,id=i,nodes=n,label='smooth_exterior_'+str(i),pixel_targets=[])
    plan.append(task)
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for name,queue,cpus,mem,command,wall in [
    ('gate','mini2',2,8,'run.py --gate','00:20:00'),
    ('worker','mini',26,64,'run.py','23:59:00')]:
    (ROOT/f'{name}.pbs').write_bytes(f'''#!/bin/bash
#PBS -q {queue}
#PBS -l select=1:ncpus={cpus}:mem={mem}gb
#PBS -l walltime={wall}
#PBS -j oe
set -euo pipefail
cd {REMOTE}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS={cpus} MKL_NUM_THREADS=1
/home/anaconda3/bin/python3 {command}
'''.encode())
files=sorted(p for p in ROOT.iterdir() if p.suffix in ('.py','.jl','.pbs','.csv','.json') and p.name not in ('manifest.json','submission.json'))
(ROOT/'manifest.json').write_text(json.dumps(dict(sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
with tarfile.open(ROOT/'source.tar.gz','w:gz') as out:
    for p in files+[ROOT/'manifest.json']:out.add(p,arcname=p.name)
check=subprocess.run(['ssh','idark','test','-e',REMOTE+'/manifest.json'])
if check.returncode!=1:raise RuntimeError('Remote root exists or connection failed')
subprocess.run(['ssh','idark','mkdir','-p',REMOTE],check=True)
subprocess.run(['scp',str(ROOT/'source.tar.gz'),'idark:'+REMOTE+'/source.tar.gz'],check=True)
subprocess.run(['ssh','idark','tar','-xzf',REMOTE+'/source.tar.gz','-C',REMOTE],check=True)
state={}
for name in ['gate','worker']:
    command=['ssh','idark','qsub','-N','tsz_edge_'+name,'-o',REMOTE+'/'+name+'.log']
    if name=='worker':command+=['-W','depend=afterok:'+state['gate']]
    job=subprocess.check_output(command+[REMOTE+'/'+name+'.pbs'],text=True).strip()
    assert job.split('.')[0].isdigit()
    state[name]=job;(ROOT/'submission.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
