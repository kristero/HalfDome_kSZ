"""Fetch compact evidence, never catalogue files, caches or full-sky maps."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/'+ROOT.name
script=r'''
from pathlib import Path
import hashlib,io,json,tarfile
root=Path(ROOT_PLACEHOLDER)
paths=[]
for directory in ('gate','controls','results','plots','followup/probe',
                  'edge_extension/gate','edge_extension/controls','work_balance/gate','work_balance/controls'):
    for p in (root/directory).rglob('*'):
        if p.is_file() and p.suffix in ('.json','.toml','.txt','.npy','.png','.log'):
            if p.stat().st_size<5_000_000:paths.append(p)
for name in ('submission.json','followup/baseline_cache.json','followup/compression_context.npz',
             'edge_extension/report.json','work_balance/report.json',
             'edge_extension/submission.json','work_balance/submission.json'):
    p=root/name
    if p.exists():paths.append(p)
snapshot={p.relative_to(root).as_posix():p.read_bytes() for p in paths}
manifest={name:hashlib.sha256(data).hexdigest() for name,data in snapshot.items()}
snapshot['fetch_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
with tarfile.open(root/'evidence.tar.gz','w:gz') as out:
    for name,data in snapshot.items():
        item=tarfile.TarInfo(name);item.size=len(data)
        out.addfile(item,io.BytesIO(data))
print(json.dumps({'files':len(paths),'bytes':(root/'evidence.tar.gz').stat().st_size}))
'''.replace('ROOT_PLACEHOLDER',repr(REMOTE))
result=subprocess.run(['ssh','idark','/home/anaconda3/bin/python3','-'],input=script,
                      text=True,capture_output=True,check=True)
print(result.stdout)
subprocess.run(['scp','idark:'+REMOTE+'/evidence.tar.gz',str(ROOT/'evidence.tar.gz')],check=True)
with tarfile.open(ROOT/'evidence.tar.gz') as source:
    for member in source.getmembers():
        target=(ROOT/member.name).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not member.isfile():
            raise RuntimeError('Unexpected tar member: '+member.name)
    source.extractall(ROOT)
manifest=json.loads((ROOT/'fetch_manifest.json').read_text())
for name,sha in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
print('Verified',len(manifest),'retrieved products.')
