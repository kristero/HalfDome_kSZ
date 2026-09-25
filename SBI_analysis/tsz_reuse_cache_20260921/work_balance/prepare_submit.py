"""Isolate task scheduling from catalogue reuse, interpolation and physics."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
REMOTE='/lustre/work/kristero10/'+PARENT.name+'/work_balance'
source=(PARENT/'benchmark.jl').read_text()
first=source.index('function paint_shared!(')
last=source.index('\n"""Read only one original chunk',first)
kernel=source[first:last]
old='    Threads.@threads :static for i in eachindex(masses)\n'
new='''    # Fine-grained greedy scheduling prevents nearby catalogue tails from
    # remaining on one thread. Block size is a performance knob only.
    block_size = parse(Int,get(ENV,"HALO_BLOCK_SIZE","256"))
    @assert block_size > 0
    Threads.@threads :greedy for first in 1:block_size:length(masses)
        for i in first:min(first+block_size-1,length(masses))
'''
assert kernel.count(old)==1
kernel=kernel.replace(old,new)
old='    return length(masses)\nend'
assert kernel.count(old)==1
kernel=kernel.replace(old,'    end # greedy block loop\n'+old)
(ROOT/'balanced_painter.jl').write_text('# Same arithmetic and locks as frozen paint_shared!, with small scheduled blocks.\n'+kernel)
task=json.loads((PARENT/'plan.json').read_text())['tasks'][2]
task=dict(task,id=0,label='shared_geometry_greedy256',pixel_targets=[])
(ROOT/'plan.json').write_text(json.dumps([task],indent=2)+'\n')
# Reuse the generic immutable launcher; imports and relative paths remain valid.
(ROOT/'run.py').write_bytes((PARENT/'edge_extension/run.py').read_bytes())
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
files=sorted(p for p in ROOT.iterdir() if p.suffix in ('.py','.jl','.pbs','.json') and p.name not in ('manifest.json','submission.json'))
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
    command=['ssh','idark','qsub','-N','tsz_balance_'+name,'-o',REMOTE+'/'+name+'.log']
    if name=='worker':command+=['-W','depend=afterok:'+state['gate']]
    job=subprocess.check_output(command+[REMOTE+'/'+name+'.pbs'],text=True).strip()
    assert job.split('.')[0].isdigit()
    state[name]=job;(ROOT/'submission.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2))
