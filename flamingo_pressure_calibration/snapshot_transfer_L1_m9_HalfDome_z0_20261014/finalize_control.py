"""Inspect, aggregate and retrieve only compact products of this study."""
import argparse
import json
from cluster_control import ROOT, IDARK_ROOT, STRW_ROOT, remote, submit

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['submit', 'retry-plotting', 'status', 'metadata'])
args = parser.parse_args()

if args.action in ['submit', 'retry-plotting']:
    program = f'''
from pathlib import Path
import json,hashlib
r=Path({IDARK_ROOT!r})
for filename,receipt in [('measured_flamingo_compact.h5','compact_complete.json'),('measured_halo_union_compact.h5','halo_union_complete.json')]:
 expected=json.loads((r/receipt).read_text())
 assert hashlib.sha256((r/filename).read_bytes()).hexdigest()==expected['sha256'],filename
 print(filename,expected['bytes'],'SHA256 verified')
'''
    print(remote('idark',program).decode())
    if args.action == 'submit':
        submit('run_aggregate.pbs')
    else:
        program=f'''
from pathlib import Path
import subprocess,json
r=Path({IDARK_ROOT!r})
old=json.loads((r/'run_aggregate.pbs.job.json').read_text())
assert old['job_id']=='603000.idark'
status=subprocess.run(['qstat','-f',old['job_id']],capture_output=True,text=True)
assert status.returncode or 'job_state = F' in status.stdout
(r/'aggregate_initial.log').write_bytes((r/'aggregate.log').read_bytes())
job=subprocess.check_output(['qsub','run_aggregate.pbs'],cwd=r,text=True).strip()
receipt=dict(job_id=job,previous_job=old['job_id'],script='run_aggregate.pbs',queue='mini',ncpus=26,mem_gb=32)
(r/'run_aggregate_retry.job.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
        result=remote('idark',program)
        (ROOT/'run_aggregate_retry.job.json').write_bytes(result)
        print(result.decode())
elif args.action == 'status':
    program=f'''
from pathlib import Path
import subprocess
r=Path({IDARK_ROOT!r})
for n in ['run_aggregate.pbs.job.json','cluster_analysis_complete.json']:
 p=r/n
 print(n,p.read_text()[:800] if p.exists() else 'PENDING')
p=r/'aggregate.log'
print('LOG',p.read_text()[-4000:] if p.exists() else 'PENDING')
print('figures',len(list((r/'figures').glob('*.png'))))
'''
    print(remote('idark',program).decode())
else:
    names=['mass_mapping.json','selection_summary.json','physics_gate.json','painting_gate.json',
           'fits_sd_ready.json','available_resources_live.json','fit_legacy_parity.json',
           'strw_execution_metadata.json']
    program=f'''
from pathlib import Path
import json
r=Path({IDARK_ROOT!r})
names={names!r}
print(json.dumps({{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}))
'''
    values=json.loads(remote('idark',program))
    for name,value in values.items():
        (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
    print('Retrieved',list(values))
