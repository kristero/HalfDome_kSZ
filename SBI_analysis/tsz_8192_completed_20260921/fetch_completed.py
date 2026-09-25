"""Fetch completed validation products, excluding large maps and caches."""
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = r'''
import datetime,json,pathlib,subprocess,sys,tarfile,numpy as np
base=pathlib.Path('/lustre/work/kristero10')
names=['tsz_8192_validation_recovery_20260921','tsz_8192_noise_extremes_20260921']
snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'queue':subprocess.run(['qstat','-u','kristero10'],stdout=subprocess.PIPE,text=True).stdout,
 'jobs':{},'roots':{},'noise_checks':{}}
for job in [598526,598527,598528,598529,598530,598533,598534,598535]:
 try:
  p=subprocess.run(['qstat','-xf',str(job)+'.idark'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=5)
  snapshot['jobs'][str(job)]={'returncode':p.returncode,'qstat':p.stdout}
 except subprocess.TimeoutExpired:
  snapshot['jobs'][str(job)]={'returncode':None,'qstat':'PBS history query timed out; inspect saved task and analysis records.'}
for name in names:
 root=base/name
 snapshot['roots'][name]=[json.loads(p.read_text()) for p in sorted((root/'controls').glob('*/status.json'))]
 checks={}
 for task in json.loads((root/'plan.json').read_text())['tasks']:
  directory=root/'controls'/str(task['id']).zfill(3)/'noise'
  spectra=list(directory.glob('*.npy'))
  assert all(np.load(p).shape==(7980,) and np.isfinite(np.load(p)).all() for p in spectra)
  assert len(list(directory.glob('[0-9][0-9][0-9].npy')))==task.get('noise_draws',0)
  checks[str(task['id'])]={'finite_spectra':len(spectra),'draws':task.get('noise_draws',0)}
 snapshot['noise_checks'][name]=checks
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in names:
  root=base/name
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p.stat().st_size>5000000:continue
   relative=p.relative_to(root)
   if '__pycache__' in relative.parts or 'local_campaign' in relative.parts:continue
   if 'noise' in relative.parts and p.suffix=='.npy':continue
   if p.suffix not in ('.json','.toml','.npy','.npz','.txt','.log','.png','.pdf','.md','.py','.jl','.pbs'):continue
   archive.add(p,arcname=name+'/'+relative.as_posix())
 payload=json.dumps(snapshot,indent=2).encode()
 item=tarfile.TarInfo('tsz_8192_completed_20260921/cluster_snapshot.json');item.size=len(payload)
 import io
 archive.addfile(item,io.BytesIO(payload))
'''

if __name__ == '__main__':
    data = subprocess.check_output(['ssh','-o','ConnectTimeout=15','idark',
                                   '/home/anaconda3/bin/python3','-'],input=CODE.encode())
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        archive.extractall(ROOT.parent,filter='data')
    snapshot=json.loads((ROOT/'cluster_snapshot.json').read_text())
    print('UTC:',snapshot['utc'],'; transfer bytes:',len(data))
    for name, rows in snapshot['roots'].items():
        print(name, 'finished:',len(rows),'failed:',sum(r['returncode']!=0 for r in rows))
    for job, row in snapshot['jobs'].items():
        print(job, [line.strip() for line in row['qstat'].splitlines()
                    if any(key in line for key in ['job_state =','Exit_status =','resources_used.walltime ='])])
