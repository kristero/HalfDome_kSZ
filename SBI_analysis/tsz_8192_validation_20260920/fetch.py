"""Retrieve small products and scheduler state; never retrieve maps or caches."""
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
code='''import pathlib,sys,tarfile
root=pathlib.Path('/lustre/work/kristero10/tsz_8192_validation_20260920')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for p in root.rglob('*'):
        if not p.is_file() or p.stat().st_size>2000000:continue
        if p.suffix not in ('.json','.toml','.npy','.txt','.log','.png','.pdf'):continue
        if 'inputs' in p.parts or 'local_campaign' in p.parts:continue
        archive.add(p,arcname=str(p.relative_to(root)))
'''
data=subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=code.encode())
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
    archive.extractall(ROOT,filter='data')
snapshot=subprocess.check_output(['ssh','idark','qstat','-u','kristero10'],text=True)
(ROOT/'results/scheduler.txt').write_text(snapshot)
print(snapshot)
