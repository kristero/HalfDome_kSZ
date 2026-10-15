"""Deploy, probe, census, prepare, gate, gate_file, launch, status, fetch and archive this study on STRW maas.

Transport: remote_control.remote (frozen; Windows OpenSSH alias strw-maas with ProxyJump strw). The remote
login shell is tcsh, so every remote action is a Python program sent on stdin (no shell redirections).
Order after Checkpoint 1 (see STEP1_README.md):
  deploy -> (probe) -> census -> [set enlargement.apply, deploy] -> prepare -> gate -> gate_file -> launch
  -> status ... -> fetch -> python -B verify_study.py
census, prepare, gate and gate_file run as detached remote processes (receipt + log) and are polled until
they finish; launch returns immediately. Only DEPLOY files are ever sent; analysis/, kernels/, quicklook/,
C_sigmoid/, census_dryrun/, tests/ and attempts/ never are.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from remote_control import remote, REMOTE

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'study_config.json').read_text())
PYTHON = CFG['remote_python']
POLITE = CFG['politeness']
FROZEN = ['pilot_physics.py', 'measure.py', 'models.py', 'projection_model.py', 'swift_reader.py', 'check_fresh_mask.py',
          'remote_control.py', 'dependencies/common.py', 'dependencies/flamingo_io.py', 'dependencies/catalogue_stage.py']
NEW = ['study_config.json', 'grids.py', 'census.py', 'prepare_study.py', 'measure_overlap.py', 'extract_study.py',
       'geometry.py', 'verify_study.py', 'control.py', 'probe_g4.py', 'probe_g4.json', 'input_hashes.json']
INPUTS = ['inputs/radii234_selected_halos.npz', 'inputs/radii234_metadata_plan.json', 'inputs/radii234_particle_hashes.json']
DEPLOY = FROZEN + NEW + INPUTS
NEVER = ('analysis', 'kernels', 'quicklook', 'C_sigmoid', 'census_dryrun', 'tests', 'attempts')
JOBS = {'census': ['census.py'], 'prepare': ['prepare_study.py'], 'gate': ['extract_study.py', '--gate-cell'],
        'gate_file': ['extract_study.py', '--gate-file', str(CFG['gate_file'])], 'launch': ['extract_study.py']}
# A job may start only when these remote files exist (and, for *.json gates, report passed).
REQUIRES = {'census': ['study_config.json', 'inputs/radii234_selected_halos.npz'],
            'prepare': ['census_pairs.npz', 'census_population.npz', 'census_summary.json', 'census_g3.json', 'probe_g4.json'],
            'gate': ['metadata_plan.json', 'selected_halos.npz', 'overlap_tables.npz'],
            'gate_file': ['one_cell_gate.json'], 'launch': ['one_cell_gate.json', 'gate_file.json']}
GATES = {'gate_file': ['one_cell_gate.json'], 'launch': ['one_cell_gate.json', 'gate_file.json']}
SMALL_OUTPUTS = {'census': ['census_summary.json', 'census_g3.json'], 'prepare': ['metadata_plan.json'],
                 'gate': ['one_cell_gate.json'], 'gate_file': ['gate_file.json']}
# a failed job archives only its own outputs (a failed gate_file keeps the passed one-cell gate)
OWN_OUTPUTS = {'gate': ('one_cell_gate.json',), 'gate_file': ('gate_file.json',),
               'launch': ('failed.json', 'progress.json')}
TIMEOUT_MINUTES = {'census': 60, 'prepare': 60, 'gate': 30, 'gate_file': 60}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def deploy():
    for name in DEPLOY:
        assert not any(part in NEVER for part in Path(name).parts), name
        assert (HERE / name).is_file(), f'missing {name}'
    content = {name: base64.b64encode((HERE / name).read_bytes()).decode() for name in DEPLOY}
    payload = base64.b64encode(json.dumps(content).encode()).decode()
    program = f'''import base64,hashlib,json
from pathlib import Path
root=Path({REMOTE!r});root.mkdir(parents=True,exist_ok=True)
assert not (root/'launch_receipt.json').exists(), 'the particle pass was launched: archive it before redeploying'
hashes={{}}
for name,data in json.loads(base64.b64decode({payload!r})).items():
    path=root/name;assert path.resolve().is_relative_to(root.resolve())
    data=base64.b64decode(data);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    hashes[name]=hashlib.sha256(data).hexdigest()
(root/'executed_sources.json').write_text(json.dumps(hashes,indent=2)+'\\n')
print(json.dumps(hashes))
'''
    hashes = json.loads(remote(program, timeout=300))
    local = {name: sha256((HERE / name).read_bytes()) for name in DEPLOY}
    assert hashes == local, 'remote hashes differ from the local files'
    (HERE / 'executed_sources.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'deployed_files': len(hashes), 'remote': REMOTE}))


def start(job):
    args = JOBS[job]
    program = f'''import json,os,subprocess,time
from pathlib import Path
root=Path({REMOTE!r})
missing=[n for n in {REQUIRES[job]!r} if not (root/n).exists()]
assert not missing, 'missing prerequisites: '+str(missing)
for gate in {GATES.get(job, [])!r}:
    assert json.loads((root/gate).read_text()).get('passed') is True, gate+' did not pass'
import psutil
for receipt in root.glob('*_receipt.json'):
    pid=json.loads(receipt.read_text()).get('pid')
    try:
        command=' '.join(psutil.Process(pid).cmdline()) if pid and psutil.pid_exists(pid) else ''
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        command=''
    if any(s in command for s in ('extract_study.py','census.py','prepare_study.py')):
        raise SystemExit('job still running: '+receipt.name)
assert not (root/{job + '_receipt.json'!r}).exists(), 'receipt exists: archive_failed first'
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['HDF5_USE_FILE_LOCKING']='FALSE'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):env[key]='1'
env.update({CFG.get('malloc_env', {})!r})   # glibc: freed HDF5 buffers go back to the OS
# shared host: start only with enough free memory, and run at the lowest CPU and I/O priority
need={POLITE['min_available_GiB_start'][job if job in POLITE['min_available_GiB_start'] else 'launch']!r}
available=psutil.virtual_memory().available/2**30;load=os.getloadavg()
assert available>=need,'only %.0f GiB available on maas; %s needs %s GiB' % (available,{job!r},need)
def polite():
    os.nice({POLITE['nice']!r})
    try:
        psutil.Process().ionice(psutil.IOPRIO_CLASS_BE,{POLITE['ionice_level']!r})
    except Exception:
        pass
with (root/{job + '.log'!r}).open('wb') as log:
    p=subprocess.Popen([{PYTHON!r},'-B','-u']+{args!r},cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                       start_new_session=True,preexec_fn=polite)
receipt={{'job':{job!r},'pid':p.pid,'args':{args!r},'host':'maas.strw.leidenuniv.nl','scheduler_job':False,
         'started_unix':time.time(),'attempt':1+len(list((root/'attempts').glob('attempt_*'))),
         'nice':{POLITE['nice']!r},'ionice':'best-effort {POLITE['ionice_level']}','available_GiB_at_start':available,
         'load_at_start':load}}
(root/{job + '_receipt.json'!r}).write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    receipt = json.loads(remote(program, python=PYTHON))
    (HERE / f'{job}_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return receipt


def job_state(job):
    program = f'''import json,psutil
from pathlib import Path
root=Path({REMOTE!r});r=json.loads((root/{job + '_receipt.json'!r}).read_text())
alive=psutil.pid_exists(r['pid'])
log=root/{job + '.log'!r}
print(json.dumps({{'alive':alive,'log_tail':log.read_text()[-3000:] if log.exists() else ''}}))
'''
    return json.loads(remote(program, python=PYTHON))


def wait(job):
    deadline = time.time() + 60 * TIMEOUT_MINUTES[job]
    while True:
        state = job_state(job)
        if not state['alive']:
            break
        if time.time() > deadline:
            raise TimeoutError(f'{job} still running after {TIMEOUT_MINUTES[job]} min; use status')
        time.sleep(30)
    print(state['log_tail'][-2000:])
    fetch(SMALL_OUTPUTS.get(job))


def status():
    program = f'''import json,psutil,time,os
from pathlib import Path
root=Path({REMOTE!r});result={{}}
for name in ('progress.json','complete.json','failed.json'):
    if (root/name).exists():result[name]=json.loads((root/name).read_text())
for receipt in sorted(root.glob('*_receipt.json')):
    r=json.loads(receipt.read_text());alive=psutil.pid_exists(r['pid'])
    item={{'pid':r['pid'],'alive':alive}}
    if alive:
        p=psutil.Process(r['pid']);tree=[p]+p.children(recursive=True)
        item['tree_RSS_GiB']=sum(x.memory_info().rss for x in tree if x.is_running())/1024**3
        item['elapsed_minutes']=(time.time()-p.create_time())/60
    result[receipt.stem]=item
result['completed_file_products']=len(list((root/'file_products').glob('*.npz'))) if (root/'file_products').exists() else 0
result['host_available_RAM_GiB']=psutil.virtual_memory().available/1024**3
result['host_load_average']=os.getloadavg()
for log in root.glob('*.log'):
    result['log_tail_'+log.stem]=log.read_text()[-1500:]
print(json.dumps(result,indent=2))
'''
    result = remote(program, python=PYTHON)
    print(result)
    return result


def fetch(names=None):
    """Download changed top-level *.json / *.npz files (or `names`), verified by SHA256."""
    listing = json.loads(remote(f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE!r});wanted={names!r}
files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.npz','.log') and (wanted is None or p.name in wanted)]
print(json.dumps({{p.name:{{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}} for p in files}}))
''', timeout=600))
    for name, info in listing.items():
        target = HERE / name
        if name in DEPLOY:
            assert sha256(target.read_bytes()) == info['sha256'], f'deployed file {name} changed remotely'
            continue
        if target.exists() and sha256(target.read_bytes()) == info['sha256']:
            continue
        temporary = target.with_suffix(target.suffix + '.transfer')
        with temporary.open('wb') as output:
            for offset in range(0, info['bytes'], 16 * 1024 ** 2):
                encoded = remote(f'''import base64
from pathlib import Path
with (Path({REMOTE!r})/{name!r}).open('rb') as f:
    f.seek({offset});print(base64.b64encode(f.read(16*1024**2)).decode())
''', timeout=300)
                output.write(base64.b64decode(encoded))
        assert sha256(temporary.read_bytes()) == info['sha256'], name
        temporary.replace(target)
    record = HERE / 'remote_verified_inputs.json'
    previous = json.loads(record.read_text()) if record.exists() else {}
    previous.update(listing)
    record.write_text(json.dumps(previous, indent=2) + '\n')
    print(json.dumps({'verified_files': len(listing), 'MiB': sum(v['bytes'] for v in listing.values()) / 1024 ** 2}))


def archive_failed(job):
    result = remote(f'''import json,psutil,shutil
from pathlib import Path
root=Path({REMOTE!r});receipt=json.loads((root/{job + '_receipt.json'!r}).read_text())
assert not psutil.pid_exists(receipt['pid']), 'process still exists'
archive=root/'attempts'/('attempt_%02d'%(1+len(list((root/'attempts').glob('attempt_*')))))
archive.mkdir(parents=True)
for name in ({job + '_receipt.json'!r},{job + '.log'!r})+{OWN_OUTPUTS.get(job, ())!r}:
    if (root/name).exists():shutil.move(str(root/name),str(archive/name))
print(json.dumps({{'archive':str(archive)}}))
''', python=PYTHON)
    info = json.loads(result)
    archive = HERE / 'attempts' / Path(info['archive']).name
    archive.mkdir(parents=True, exist_ok=True)
    for name in (f'{job}_receipt.json',) + OWN_OUTPUTS.get(job, ()):
        if (HERE / name).exists():
            (HERE / name).replace(archive / name)
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['deploy', 'probe', 'census', 'prepare', 'gate', 'gate_file', 'launch', 'status',
                                           'fetch', 'archive_failed'])
    parser.add_argument('--job', choices=sorted(JOBS), help='job for archive_failed')
    parser.add_argument('--no-wait', action='store_true')
    args = parser.parse_args()
    if args.action == 'deploy':
        deploy()
    elif args.action == 'probe':
        import probe_g4
        probe_g4.main()
    elif args.action in JOBS:
        start(args.action)
        if args.action != 'launch' and not args.no_wait:
            wait(args.action)
    elif args.action == 'status':
        status()
    elif args.action == 'fetch':
        fetch()
    else:
        archive_failed(args.job or 'launch')


if __name__ == '__main__':
    main()
