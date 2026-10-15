#!/usr/bin/env python3
"""Freeze the reviewed inference code before its PBS job starts; record the change from preflight."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common import ROOT, sha, write_json


def freeze(root):
    jobs=json.loads((root/'jobs.json').read_text()) if (root/'jobs.json').exists() else dict(jobs=[])
    if any(j['task']=='pipeline' for j in jobs['jobs']):
        raise RuntimeError('Cannot change the inference code freeze after submission')
    path=root/'input_sha256.json';before=json.loads(path.read_text())
    after=dict(before)
    after['code']={str(p.relative_to(root)):sha(p) for p in (root/'code').iterdir() if p.is_file()}
    # The only code used by the running painting jobs must remain byte-identical.
    assert before['code']['code/paint_seed.jl']==after['code']['code/paint_seed.jl']
    audit_path=root/'code_freeze.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else dict(initial_code=before['code'],revisions=[])
    audit.setdefault('revisions',[]).append(dict(utc=datetime.now(timezone.utc).isoformat(),final_code=after['code']))
    audit.update(final_code=after['code'],reason='Complete inference/report review before dependent job submission')
    write_json(audit_path,audit)
    write_json(path,after)
    print('Frozen',len(after['code']),'code files; physical painter unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    freeze(p.parse_args().root)
