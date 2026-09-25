"""Check PBS portability, source identity and no duplicate task ownership."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent/'tsz_8192_validation_recovery_20260921'
manifest=json.loads((NEW/'manifest.json').read_text())
for name,expected in manifest['source_sha256'].items():
    assert hashlib.sha256((NEW/name).read_bytes()).hexdigest()==expected,name
for name in ['partition.pbs','analyze.pbs']:
    content=(NEW/name).read_bytes()
    assert content.startswith(b'#!/bin/bash\n') and b'\r' not in content,name
probes=[json.loads(p.read_text()) for p in NEW.glob('claim_probe_*.json')]
assert len(probes)==4 and sum(p['won'] for p in probes)==1
assert len(set(p['host'] for p in probes))==4
print('PASS source identities, LF PBS scripts and four-node exclusion')
