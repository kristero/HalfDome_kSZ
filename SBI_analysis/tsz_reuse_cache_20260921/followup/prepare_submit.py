"""Deploy an append-only follow-up audit; preserve frozen benchmark sources."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/'+ROOT.parent.name+'/followup'
plan=json.loads((ROOT.parent/'plan.json').read_text())
old=json.loads((ROOT.parent.parent/'tsz_8192_validation_recovery_20260921/plan.json').read_text())
design=np.load(ROOT.parent.parent/'tsz_diagnostic_256_8192_prepared_20260921/diagnostic_256/theta_design.npy')
anchors=[c['theta'] for c in plan['tasks'][0]['cases']]
tails=[next(t for t in old['tasks'] if t['label']==label)['theta'] for label in ['combined_tails','high_amplitude_tails']]
np.savetxt(ROOT/'theta_probe.csv',np.vstack([anchors,tails,design[:16]]),delimiter=',',fmt='%.18e')
jobs=[('probe','mini',26,16,'launch.py','02:00:00'),
      ('baseline','mini2',2,8,'baseline_cache.py','00:30:00')]
for name,queue,cpus,mem,script,wall in jobs:
    (ROOT/f'{name}.pbs').write_bytes(f'''#!/bin/bash
#PBS -q {queue}
#PBS -l select=1:ncpus={cpus}:mem={mem}gb
#PBS -l walltime={wall}
#PBS -j oe
set -euo pipefail
cd {REMOTE}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS={cpus} MKL_NUM_THREADS=1
/home/anaconda3/bin/python3 {script}
'''.encode())
files=sorted(p for p in ROOT.iterdir() if p.suffix in ('.py','.jl','.csv','.pbs'))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT/'manifest.json').write_text(json.dumps(dict(sha256=hashes),indent=2)+'\n')
with tarfile.open(ROOT/'source.tar.gz','w:gz') as archive:
    for path in files+[ROOT/'manifest.json']:archive.add(path,arcname=path.name)
check=subprocess.run(['ssh','idark','test','-e',REMOTE+'/manifest.json'])
if check.returncode != 1:raise RuntimeError('Remote audit exists or SSH check failed')
subprocess.run(['ssh','idark','mkdir','-p',REMOTE],check=True)
subprocess.run(['scp',str(ROOT/'source.tar.gz'),'idark:'+REMOTE+'/source.tar.gz'],check=True)
subprocess.run(['ssh','idark','tar','-xzf',REMOTE+'/source.tar.gz','-C',REMOTE],check=True)
state={}
for name,*_ in jobs:
    job=subprocess.check_output(['ssh','idark','qsub','-N','tsz_cache_'+name,
        '-o',REMOTE+'/'+name+'.log',REMOTE+'/'+name+'.pbs'],text=True).strip()
    assert job.split('.')[0].isdigit()
    state[name]=job
    (ROOT/'submission.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
