"""Read-only cluster audit and compact experiment storage inventory.

Run with Windows Python so the existing idark SSH alias is available.
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SCRIPT = r'''
from datetime import datetime, timezone
import hashlib, json, subprocess
from pathlib import Path
root=Path('/lustre/work/kristero10/tsz_reuse_cache_20260921')
checks={}
for rel in ('manifest.json','edge_extension/manifest.json','work_balance/manifest.json'):
    manifest=root/rel
    values=json.loads(manifest.read_text())['sha256']
    for name,digest in values.items():
        assert hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest()==digest,name
    checks[rel]=len(values)
products=EXPECTED_PRODUCTS
for name,digest in products.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
checks['products']=len(products)
paths=[p for p in root.rglob('*') if p.is_file()]
qstat=subprocess.run(['qstat','-u','kristero10'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),root=str(root),verified=checks,
    qstat_returncode=qstat.returncode,qstat_stdout=qstat.stdout,qstat_stderr=qstat.stderr,
    total_files=len(paths),total_bytes=sum(p.stat().st_size for p in paths),
    largest_files=sorted([dict(path=str(p.relative_to(root)),bytes=p.stat().st_size) for p in paths],
        key=lambda p:p['bytes'],reverse=True)[:25],
    archives=[str(p.relative_to(root)) for p in paths if p.name.endswith('.tar.gz')]),indent=2))
'''.replace('EXPECTED_PRODUCTS', repr(json.loads((ROOT.parent / 'tsz_reuse_cache_20260921/fetch_manifest.json').read_text())))
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'idark',
                         '/home/anaconda3/bin/python3', '-'], input=SCRIPT,
                        text=True, capture_output=True)
if result.returncode:
    raise RuntimeError(result.stderr)
data = json.loads(result.stdout)
(ROOT / 'cluster_audit.json').write_text(json.dumps(data, indent=2) + '\n')
print(result.stdout)
