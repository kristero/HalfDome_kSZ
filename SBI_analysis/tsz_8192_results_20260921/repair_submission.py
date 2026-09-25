"""Correct CRLF before any recovery worker ran, retaining failed job evidence."""
import json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
code=r'''
import pathlib,json,hashlib,subprocess,shutil
root=pathlib.Path('/lustre/work/kristero10/tsz_8192_validation_recovery_20260921')
manifest=json.loads((root/'manifest.json').read_text())
for name in ['partition.pbs','analyze.pbs']:
 p=root/name;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 assert b'\r' not in p.read_bytes()
 subprocess.run(['bash','-n',str(p)],check=True)
 manifest['source_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
journal=root/'submission_recovery.json'
previous=json.loads(journal.read_text());assert previous['workers'][0]=='598520.idark'
shutil.copy2(journal,root/'submission_recovery_failed_crlf.json')
for p in root.glob('*.log'):shutil.copy2(p,p.with_suffix('.failed_crlf.log'))
state=dict(workers=[],analysis=None,retained=previous['retained'],diagnostic_256_submitted=False)
for i in range(4):
 job=subprocess.check_output(['qsub','-N',f'tszfix{i}','-v',f'TSZ_SHARD={i}',
  '-o',str(root/f'shard_{i}.log'),str(root/'partition.pbs')],text=True).strip()
 state['workers'].append(job);journal.write_text(json.dumps(state,indent=2))
state['analysis']=subprocess.check_output(['qsub','-N','tszfix_report','-W','depend=afterany:'+':'.join(state['workers']),
  '-o',str(root/'analysis.log'),str(root/'analyze.pbs')],text=True).strip()
journal.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
'''
result=subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=code,text=True)
(ROOT/'results/recovery_submission.json').write_text(result);print(result)
