"""Fetch only small test results, never full-sky maps or interpolation caches."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/tsz_spherical_preflight_20260920'
code='''import pathlib,sys,tarfile
root=pathlib.Path("/lustre/work/kristero10/tsz_spherical_preflight_20260920")
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz") as archive:
    for p in (root/"fullsky").rglob("*"):
        if p.is_file() and p.suffix in (".json",".toml",".npy",".txt",".log") and p.stat().st_size<2000000:
            archive.add(p,arcname=str(p.relative_to(root)))
'''
blob=subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=code.encode())
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
    archive.extractall(ROOT,filter='data')
for p in (ROOT/'fullsky').rglob('status.json'):
    d=json.loads(p.read_text());print(p.parent.relative_to(ROOT),d['returncode'],round(d['seconds'],1))
