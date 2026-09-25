"""Locked, resumable, finite preflight tasks. Never generate diagnostic rows."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=Path('/lustre/work/kristero10/flamingo_tsz_comparison_20260914')
KEYS=['P0_amp','x_c_amp','beta_amp','P0_alpha_m','x_c_alpha_m',
      'beta_alpha_m','P0_alpha_z','x_c_alpha_z','beta_alpha_z']

def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['control.jl','fullsky_test.jl','spherical_truncation_profiles.jl','worker.py']}

def run_task(task):
    out=ROOT/'controls'/f"{task['id']:03d}";out.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,source_sha256=source_hashes())
    request=out/'request.json'
    if request.exists() and json.loads(request.read_text())!=identity:
        raise RuntimeError('Source or settings changed: require fresh output root')
    atomic_json(request,identity)
    command=json.loads((CAMPAIGN/'preflight/metadata_manifest.json').read_text())['halfdome_reference']['simulation_request']['command']
    settings=dict(arg.split('=',1) for arg in command if '=' in arg and not arg.startswith('--'))
    noise=CAMPAIGN/'code/halfdome/other_sims/SO'
    settings.update(output_dir=str(out/'raw'),cache_dir=str(out/'cache'),nside=str(task['nside']),
        interpolator_pad='256',enforce_battaglia_guardrails='false',model_exists='false',reuse_existing_cache='false',
        halfdome_path='/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5',
        baseline_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt'),
        goal_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt'))
    settings.update({'battaglia_'+key:format(value,'.17g') for key,value in zip(KEYS,task['theta'])})
    runtime=CAMPAIGN/'runtime';threads=task['threads']
    env=dict(os.environ,FLAMINGO_CAMPAIGN=str(CAMPAIGN),HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'),
        PREFLIGHT_OUTPUT=str(out),PREFLIGHT_GRID_FACTOR=str(task['grid_factor']),
        VALIDATION_TASK_ID=str(task['id']),VALIDATION_NOISE_DRAWS=str(task['noise_draws']),
        VALIDATION_PAIRED_4096=str(int(task.get('paired_4096',False))),
        VALIDATION_LEAN_MAPS=str(int(task['lean_maps'])),
        JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
        LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS='1',HDF5_USE_FILE_LOCKING='FALSE')
    cmd=[str(runtime/'julia-1.12.2/bin/julia'),'--startup-file=no','--threads='+str(threads),
         '--project='+str(runtime/'julia_env'),str(ROOT/'control.jl')]+[k+'='+v for k,v in settings.items()]
    started=time.monotonic()
    # Keep failed attempts rather than overwriting their evidence.
    attempt=len(list(out.glob('attempt_*.json')))
    with (out/f'run_{attempt:02d}.log').open('w') as log:
        code=subprocess.call(['/usr/bin/time','-v','-o',str(out/f'time_{attempt:02d}.txt')]+cmd,
            env=env,stdout=log,stderr=subprocess.STDOUT)
    status=dict(task_id=task['id'],returncode=code,seconds=time.monotonic()-started,
        attempt=attempt,job=os.environ.get('PBS_JOBID'),utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/f'attempt_{attempt:02d}.json',status);atomic_json(out/'status.json',status)
    print(task['id'],task['label'],code,status['seconds'],flush=True)

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--budget-seconds',type=float,default=22.5*3600);args=parser.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text());started=time.monotonic()
    sources=source_hashes()
    manifest_path=ROOT/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for name,actual in sources.items():
            if manifest['source_sha256'][name]!=actual:raise RuntimeError('Frozen producer changed: '+name)
    external=ROOT/'external_dependencies.json'
    if external.exists():
        for name,expected in json.loads(external.read_text())['sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Archived external dependency changed: '+name)
    (ROOT/'locks').mkdir(exist_ok=True)
    for task in plan['tasks']:
        if time.monotonic()-started>args.budget_seconds:break
        with (ROOT/'locks'/f"{task['id']:03d}.lock").open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            marker=ROOT/'controls'/f"{task['id']:03d}"/'status.json'
            request=marker.parent/'request.json'
            if request.exists() and json.loads(request.read_text())!=dict(task=task,source_sha256=sources):
                raise RuntimeError('Refusing resume or reuse across source/settings changes')
            if marker.exists():
                status=json.loads(marker.read_text())
                if status['returncode']==0 or not args.retry_failed:continue
            try:run_task(task)
            except Exception as error:
                out=marker.parent;out.mkdir(parents=True,exist_ok=True)
                atomic_json(marker,dict(task_id=task['id'],returncode=-1,error=str(error)))
    statuses=[]
    for task in plan['tasks']:
        marker=ROOT/'controls'/f"{task['id']:03d}"/'status.json'
        statuses.append(json.loads(marker.read_text()) if marker.exists() else dict(task_id=task['id'],returncode=None))
    atomic_json(ROOT/f"worker_{os.environ.get('PBS_JOBID','local')}.json",dict(statuses=statuses))

if __name__=='__main__':main()
