"""Deploy, gate, launch, inspect and fetch this isolated STRW study."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from remote_control import remote,REMOTE

HERE=Path(__file__).resolve().parent
PYTHON='/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['archive_failed','deploy','gate','launch','status','fetch'])
    args=parser.parse_args()
    if args.action=='archive_failed':
        result=remote(f'''import json,psutil,shutil
from pathlib import Path
root=Path({REMOTE!r});receipt=json.loads((root/'launch_receipt.json').read_text())
assert not psutil.pid_exists(receipt['pid']), 'Previous process still exists'
assert (root/'failed.json').exists() and not (root/'complete.json').exists()
archive=root/'attempts'/('attempt_%02d'%(1+len(list((root/'attempts').glob('attempt_*')))))
archive.mkdir(parents=True)
for name in ('launch_receipt.json','failed.json','progress.json','extraction.log','executed_sources.json','one_cell_gate.json'):
    path=root/name
    if path.exists():shutil.move(str(path),str(archive/name))
print(json.dumps({{'archive':str(archive),'retained_file_products':len(list((root/'file_products').glob('*.npz')))}}))
''',python=PYTHON)
        info=json.loads(result);archive=HERE/'attempts'/Path(info['archive']).name;archive.mkdir(parents=True,exist_ok=True)
        for name in ('launch_receipt.json','executed_sources.json','one_cell_gate.json'):
            if (HERE/name).exists():(HERE/name).replace(archive/name)
        # Fetch the compact failure evidence; resumable particle products stay remote.
        evidence=remote(f'''import base64,json
from pathlib import Path
root=Path({info['archive']!r})
print(json.dumps({{p.name:base64.b64encode(p.read_bytes()).decode() for p in root.iterdir() if p.is_file()}}))
''')
        for name,data in json.loads(evidence).items():(archive/name).write_bytes(base64.b64decode(data))
        print(result)
    elif args.action=='deploy':
        files=[p for p in HERE.rglob('*.py') if '__pycache__' not in p.parts and 'attempts' not in p.parts]
        content={str(p.relative_to(HERE)):base64.b64encode(p.read_bytes()).decode() for p in files}
        content['study_config.json']=base64.b64encode((HERE/'study_config.json').read_bytes()).decode()
        payload=base64.b64encode(json.dumps(content).encode()).decode()
        program=f'''import base64,hashlib,json
from pathlib import Path
root=Path({REMOTE!r});assert not (root/'launch_receipt.json').exists()
hashes={{}}
for name,data in json.loads(base64.b64decode({payload!r})).items():
    path=root/name;assert path.resolve().is_relative_to(root.resolve())
    data=base64.b64decode(data);path.parent.mkdir(exist_ok=True);path.write_bytes(data)
    hashes[name]=hashlib.sha256(data).hexdigest()
(root/'executed_sources.json').write_text(json.dumps(hashes,indent=2)+'\\n')
print(json.dumps(hashes))
'''
        hashes=json.loads(remote(program));(HERE/'executed_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
        print(json.dumps({'deployed_sources':len(hashes)}))
    elif args.action=='gate':
        result=remote(f'''import sys,json,os
sys.dont_write_bytecode=True
sys.path.insert(0,{REMOTE!r})
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import extract_study as E
E.initialize()
snap=E.F.Snapshot(E.F.Store(E.ROOT),E.SNAPSHOT,77,read_from='chunks',chunk_cache_bytes=1024**2)
file_id=int(snap.cell_file[E.H['cells'][0]]);snap.close()
result=E.read_file(file_id,gate_only=True)
from pathlib import Path
(Path({REMOTE!r})/'one_cell_gate.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result,indent=2))
''',python=PYTHON,timeout=180)
        (HERE/'one_cell_gate.json').write_text(result);print(result)
    elif args.action=='launch':
        program=f'''import json,os,subprocess
from pathlib import Path
root=Path({REMOTE!r});assert not (root/'launch_receipt.json').exists()
assert json.loads((root/'one_cell_gate.json').read_text())['all_shell_estimators_and_original_maps_match_reference']
assert json.loads((root/'one_cell_gate.json').read_text())['empty_then_populated_cell_merge_passed']
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):env[key]='1'
with (root/'extraction.log').open('wb') as log:
    p=subprocess.Popen([{PYTHON!r},'-B','-u','extract_study.py'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={{'pid':p.pid,'host':'maas.strw.leidenuniv.nl','scheduler_job':False,'script':'extract_study.py','workers':8,'RAM_limit_GiB':40,'wall_limit_minutes':180,'scientific_process_launches':1+len(list((root/'attempts').glob('*/launch_receipt.json')))}}
(root/'launch_receipt.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result,indent=2))
'''
        result=remote(program);(HERE/'launch_receipt.json').write_text(result);print(result)
    elif args.action=='status':
        result=remote(f'''import json,psutil,time,os
from pathlib import Path
root=Path({REMOTE!r});result={{}}
for name in ('progress.json','complete.json','failed.json'):
    if (root/name).exists():result[name]=json.loads((root/name).read_text())
pid=json.loads((root/'launch_receipt.json').read_text())['pid']
result['pid_exists']=psutil.pid_exists(pid)
if result['pid_exists']:
    p=psutil.Process(pid);tree=[p]+p.children(recursive=True)
    result['tree_RSS_GiB']=sum(x.memory_info().rss for x in tree if x.is_running())/1024**3
    result['elapsed_minutes']=(time.time()-p.create_time())/60
    result['live_tree_CPU_seconds']=sum(x.cpu_times().user+x.cpu_times().system for x in tree if x.is_running())
result['completed_file_products']=len(list((root/'file_products').glob('*.npz')))
result['host_available_RAM_GiB']=psutil.virtual_memory().available/1024**3
result['host_logical_CPUs']=psutil.cpu_count();result['host_load_average']=os.getloadavg()
if (root/'failed.json').exists():result['log_tail']=(root/'extraction.log').read_text()[-4000:]
print(json.dumps(result,indent=2))
''',python=PYTHON)
        print(result)
    else:
        listing=json.loads(remote(f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE!r})
files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.npz')]
print(json.dumps({{p.name:{{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}} for p in files}}))
'''))
        for name,info in listing.items():
            target=HERE/name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==info['sha256']:continue
            with target.with_suffix('.transfer').open('wb') as output:
                for offset in range(0,info['bytes'],16*1024**2):
                    encoded=remote(f'''import base64
from pathlib import Path
with (Path({REMOTE!r})/{name!r}).open('rb') as f:
    f.seek({offset});print(base64.b64encode(f.read(16*1024**2)).decode())
''',timeout=180)
                    output.write(base64.b64decode(encoded))
            temporary=target.with_suffix('.transfer')
            assert hashlib.sha256(temporary.read_bytes()).hexdigest()==info['sha256']
            temporary.replace(target)
        (HERE/'remote_verified_inputs.json').write_text(json.dumps(listing,indent=2)+'\n')
        print(json.dumps({'verified_input_files':len(listing),'total_MiB':sum(v['bytes'] for v in listing.values())/1024**2}))


if __name__=='__main__':main()
