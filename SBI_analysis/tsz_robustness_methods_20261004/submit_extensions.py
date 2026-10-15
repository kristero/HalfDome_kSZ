#!/usr/bin/env python3
"""Submit the six matched Merlin controls and dependent final comparison within the 20-job cap."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common import ROOT, write_json
from dispatch import command, available


def submit(root):
    primary=json.loads((root/'jobs.json').read_text())['jobs']
    path=root/'extension_jobs.json'
    ledger=json.loads(path.read_text()) if path.exists() else dict(jobs=[])
    existing={job['task']:job['id'] for job in ledger['jobs']}
    parent=next(job['id'] for job in primary if job['task']=='pipeline')
    if len(primary)+len(ledger['jobs'])+2-len(existing)>20:
        raise RuntimeError('These submissions would exceed the authorized 20-job limit')
    if 'merlin_checks' not in existing:
        _,counts,slots=available()
        queue='mini' if counts['mini']<6 and slots['GroupA']>0 else 'mini_B'
        job=command(['qsub','-q',queue,'-v',f'RUN_ROOT={root}','-o',str(root/'logs'),str(root/'code/merlin_checks.pbs')]).strip()
        ledger['jobs'].append(dict(id=job,task='merlin_checks',queue=queue,utc=datetime.now(timezone.utc).isoformat()))
        write_json(path,ledger);existing['merlin_checks']=job
        print('SUBMITTED',job,queue,'matched Merlin controls',flush=True)
    if 'comparison' not in existing:
        dependency=f'afterok:{parent}:{existing["merlin_checks"]}'
        job=command(['qsub','-q','mini2','-W',f'depend={dependency}','-v',f'RUN_ROOT={root}',
                     '-o',str(root/'logs'),str(root/'code/combine.pbs')]).strip()
        ledger['jobs'].append(dict(id=job,task='comparison',queue='mini2',dependency=dependency,
                                  utc=datetime.now(timezone.utc).isoformat()))
        write_json(path,ledger)
        print('SUBMITTED',job,'dependent final comparison',flush=True)
    print('TOTAL SUBMITTED FOR THIS STUDY',len(primary)+len(ledger['jobs']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    submit(p.parse_args().root)
