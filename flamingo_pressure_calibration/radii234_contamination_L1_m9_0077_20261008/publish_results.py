"""Verify frozen compact inputs, remove redundant products, and mirror delivery.

Only this explicitly named isolated research directory can be written/cleaned.
Raw snapshots, earlier studies, authentication files and production are untouched.
"""
import base64
import hashlib
import json
from pathlib import Path
from remote_control import remote,REMOTE
from control import PYTHON

HERE=Path(__file__).resolve().parent
EXCLUDED={'delivery_manifest.json','delivery_verification.json','latest_status.json','synthetic_gate_render.png'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((HERE/'complete.json').read_text())['completed']
    assert json.loads((HERE/'scientific_checks.json').read_text())['passed']
    assert json.loads((HERE/'report_visual_qa.json').read_text())['visual_review_complete']
    assert json.loads((HERE/'browser_qa.json').read_text())['passed']
    frozen={name:sha(HERE/name) for name in ('selected_halos.npz','measurements.npz','projection_stacks.npz',
            'swiftsimio_profiles.npz','swift_checks.json','complete.json')}
    # Clean only after independent local verification and live remote hashes.
    program=f'''import hashlib,json,psutil
from pathlib import Path
root=Path({REMOTE!r});expected={frozen!r}
assert root.name=='radii234_contamination_L1_m9_0077_20261008'
assert json.loads((root/'complete.json').read_text())['completed']
assert not psutil.pid_exists(json.loads((root/'launch_receipt.json').read_text())['pid'])
for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
directory=root/'file_products';removed=[]
if directory.exists():
    assert directory.resolve().parent==root.resolve()
    for path in directory.iterdir():
        assert path.is_file() and path.name.startswith('file_') and path.suffix in ('.npz','.json')
        removed.append({{'name':str(path.relative_to(root)),'bytes':path.stat().st_size}});path.unlink()
    directory.rmdir()
print(json.dumps({{'frozen_compact_inputs_SHA256_verified':expected,'removed_redundant_files':removed,
    'reclaimed_bytes':sum(x['bytes'] for x in removed),'raw_snapshot_or_prior_outputs_modified':False}}))
'''
    cleanup=json.loads(remote(program,python=PYTHON,timeout=180))
    previous_cleanup=HERE/'cleanup_receipt.json'
    if previous_cleanup.exists() and not cleanup['removed_redundant_files']:
        previous=json.loads(previous_cleanup.read_text())
        previous['frozen_compact_inputs_SHA256_verified']=cleanup['frozen_compact_inputs_SHA256_verified']
        cleanup=previous
    (HERE/'cleanup_receipt.json').write_text(json.dumps(cleanup,indent=2)+'\n')
    test_image=HERE/'synthetic_gate_render.png'
    if test_image.exists():
        cleanup['removed_local_synthetic_render']={'bytes':test_image.stat().st_size,'sha256':sha(test_image)}
        test_image.unlink();(HERE/'cleanup_receipt.json').write_text(json.dumps(cleanup,indent=2)+'\n')
    sources={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*')
        if p.is_file() and (p.suffix in ('.py','.mjs') or p.name in ('viewer_template.html','study_config.json','README.txt'))
        and '__pycache__' not in p.parts}
    (HERE/'final_source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    files={str(p.relative_to(HERE)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in HERE.rglob('*')
        if p.is_file() and p.name not in EXCLUDED and '__pycache__' not in p.parts and '.transfer' not in p.suffix}
    inventory={'root':str(HERE),'preexisting_files_modified':[], 'created_sources':sorted(sources),'files':files,
               'total_bytes':sum(v['bytes'] for v in files.values())}
    (HERE/'delivery_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    files['delivery_manifest.json']={'bytes':(HERE/'delivery_manifest.json').stat().st_size,'sha256':sha(HERE/'delivery_manifest.json')}
    actual=json.loads(remote(f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
print(json.dumps({{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}}))
''',timeout=180))
    pending=[name for name,info in files.items() if actual.get(name)!=info['sha256']]
    # Batch small files to avoid one SSH login for each figure.
    batches=[];batch=[];size=0
    for name in pending:
        if size+files[name]['bytes']>8*1024**2 and batch:batches.append(batch);batch=[];size=0
        batch.append(name);size+=files[name]['bytes']
    if batch:batches.append(batch)
    for i,names in enumerate(batches):
        payload={name:base64.b64encode((HERE/name).read_bytes()).decode() for name in names}
        encoded=base64.b64encode(json.dumps(payload).encode()).decode()
        remote(f'''import base64,json
from pathlib import Path
root=Path({REMOTE!r})
for name,data in json.loads(base64.b64decode({encoded!r})).items():
    path=root/name;assert path.resolve().is_relative_to(root.resolve())
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.upload')
    temporary.write_bytes(base64.b64decode(data));temporary.replace(path)
print('ok')
''',timeout=240)
        print(json.dumps({'uploaded_batch':i+1,'total_batches':len(batches),'files':len(names)}),flush=True)
    expected={name:info['sha256'] for name,info in files.items()}
    payload=base64.b64encode(json.dumps(expected).encode()).decode()
    verification=json.loads(remote(f'''import base64,hashlib,json
from pathlib import Path
root=Path({REMOTE!r});expected=json.loads(base64.b64decode({payload!r}))
for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
result={{'passed':True,'verified_files':len(expected),'remote_root':str(root),
    'delivery_manifest_sha256':expected['delivery_manifest.json'],'frozen_processed_inputs_verified':True}}
(root/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
''',timeout=240))
    (HERE/'delivery_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
