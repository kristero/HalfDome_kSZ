"""Read cluster progress and fetch completed products without changing jobs.

Run from Windows Python, where the idark SSH alias is configured. A snapshot is
unpacked separately, so fetching never overwrites local code under development.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/'+ROOT.name


def remote_python(source,timeout=45):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','idark',
        '/home/anaconda3/bin/python3','-'],input=source,text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr.strip())
    return result.stdout


def status():
    source=f'''
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
r=Path({REMOTE!r})
result={{'utc':datetime.now(timezone.utc).isoformat(),'rows':len(list((r/'diagnostic_256/rows').glob('*/status.json')))}}
for name in ['test_submission.json','spots_submission.json','production_submission.json','analysis_submission.json','forward_submission.json','results/final_gate.json']:
 if (r/name).exists():result[name]=json.loads((r/name).read_text())
for name in ['gate','anchors','spot0','spot1']:
 p=r/'tests'/name/'status.json'
 if p.exists():result['test_'+name]=json.loads(p.read_text())
result['complete']=(r/'diagnostic_256/analysis/artifact_manifest.json').exists()
result['queue']=subprocess.check_output(['qstat','-u','kristero10'],text=True)
print(json.dumps(result))
'''
    value=json.loads(remote_python(source))
    (ROOT/'cluster_state.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def fetch():
    source=f'''
from pathlib import Path
import tarfile,hashlib,json
r=Path({REMOTE!r});archive=r/'export_snapshot.tar.gz'
paths=[]
for p in r.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 relative=p.relative_to(r)
 if any(part in ['tmp','__pycache__','claims'] for part in relative.parts):continue
 if p.suffix=='.gz' or p==archive:continue
 # Row spectra are stored once under rows; batch metadata/logs remain available.
 if len(relative.parts)>3 and relative.parts[:2]==('diagnostic_256','batches') and len(relative.parts)>4:continue
 paths.append((p,str(relative)))
# pathlib does not recursively follow the absolute row symlinks.
for row in (r/'diagnostic_256/rows').glob('*'):
 if row.is_dir():
  for p in row.iterdir():
   if p.is_file():paths.append((p,str(p.relative_to(r))))
with tarfile.open(archive,'w:gz',dereference=True) as tar:
 seen=set()
 for p,name in paths:
  if name in seen:continue
  tar.add(p,arcname=name,recursive=False);seen.add(name)
print(json.dumps(dict(files=len(seen),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''
    receipt=json.loads(remote_python(source,timeout=300))
    archive=ROOT/'cluster_snapshot.tar.gz'
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'idark:'+REMOTE+'/export_snapshot.tar.gz',str(archive)],check=True)
    import hashlib
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    target=ROOT/'cluster_snapshot';target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        # All entries were added as regular files; reject path traversal anyway.
        for member in tar.getmembers():
            destination=(target/member.name).resolve()
            if not destination.is_relative_to(target.resolve()) or not member.isfile():
                raise RuntimeError('Unexpected archive member: '+member.name)
        tar.extractall(target)
    (ROOT/'fetch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(fetched=str(target),**receipt)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    parser.add_argument('--fetch',action='store_true');parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args()
    while True:
        try:
            value=status()
            print(json.dumps(dict(utc=value['utc'],rows=value['rows'],complete=value['complete'],
                production=value.get('production_submission.json'))),flush=True)
            if args.fetch and (value['complete'] or not args.watch):fetch()
            if value['complete'] or not args.watch:break
        except (OSError,RuntimeError,subprocess.SubprocessError) as error:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),connection_error=str(error))),flush=True)
            if not args.watch:raise
        time.sleep(max(10,args.interval))
