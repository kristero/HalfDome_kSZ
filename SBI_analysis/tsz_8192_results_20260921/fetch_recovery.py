"""Fetch recovery metadata/new controls without recopying old noise arrays."""
import io,json,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LOCAL=ROOT.parent/'tsz_8192_validation_recovery_20260921'
code=r'''
import pathlib,sys,tarfile,json,subprocess,datetime
root=pathlib.Path('/lustre/work/kristero10/tsz_8192_validation_recovery_20260921')
status=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scheduler=subprocess.check_output(['qstat','-u','kristero10'],text=True),controls=[])
for p in sorted((root/'controls').glob('*')):
 if not p.is_dir():continue
 marker=p/'status.json'
 row=json.loads(marker.read_text()) if marker.exists() else dict(task_id=int(p.name),returncode=None)
 status['controls'].append(row)
(root/'live_snapshot.json').write_text(json.dumps(status,indent=2))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for p in root.rglob('*'):
  if not p.is_file() or p.stat().st_size>2000000:continue
  relative=p.relative_to(root)
  if relative.parts[0]=='inputs':continue
  if relative.parts[0]=='controls' and int(relative.parts[1])<78:continue
  if p.suffix not in ['.npy','.json','.toml','.txt','.log','.pbs']:continue
  archive.add(p,arcname=str(relative))
'''
blob=subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=code.encode())
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:archive.extractall(LOCAL,filter='data')
snapshot=json.loads((LOCAL/'live_snapshot.json').read_text())
(ROOT/'results/recovery_live.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print(snapshot['scheduler'])
for row in snapshot['controls']:
    if row['task_id']>=78:print(row)
