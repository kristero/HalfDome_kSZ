"""Finite benchmark launcher with immutable requests and atomic task claims."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import toml

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/lustre/work/kristero10/flamingo_tsz_comparison_20260914')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_sources():
    sources = json.loads((ROOT / 'manifest.json').read_text())['sha256']
    external = json.loads((ROOT / 'external_dependencies.json').read_text())['sha256']
    for name, expected in list(sources.items()) + list(external.items()):
        path = Path(name) if name.startswith('/') else ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source changed: ' + name)


def launch(task, gate=False):
    folder = ROOT / ('gate' if gate else 'controls/' + f"{task['id']:03d}")
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / 'claim'
    try:
        claim.mkdir()
    except FileExistsError:
        raise RuntimeError('Task already claimed; inspect status rather than overwrite: ' + str(folder))
    write_json(folder / 'request.json', task)
    (folder / 'task.toml').write_text(toml.dumps(task))
    original = json.loads((CAMPAIGN / 'preflight/metadata_manifest.json').read_text())
    command = original['halfdome_reference']['simulation_request']['command']
    settings = dict(arg.split('=',1) for arg in command if '=' in arg and not arg.startswith('--'))
    noise = CAMPAIGN / 'code/halfdome/other_sims/SO'
    settings.update(output_dir=str(folder/'raw'), cache_dir=str(folder/'cache'),
        nside=str(task['nside']), interpolator_pad='256', enforce_battaglia_guardrails='false',
        model_exists='false', reuse_existing_cache='false',
        halfdome_path='/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5',
        baseline_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt'),
        goal_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt'))
    runtime = CAMPAIGN / 'runtime'
    threads = 2 if gate else task['threads']
    env = dict(os.environ, FLAMINGO_CAMPAIGN=str(CAMPAIGN),
        HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'), PREFLIGHT_OUTPUT=str(folder),
        BENCHMARK_TASK=str(folder/'task.toml'),
        JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
        LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS='1',
        HDF5_USE_FILE_LOCKING='FALSE')
    cmd = [str(runtime/'julia-1.12.2/bin/julia'), '--startup-file=no', '--threads='+str(threads),
           '--project='+str(runtime/'julia_env'), str(ROOT/('gate.jl' if gate else 'benchmark.jl'))]
    cmd.extend(key+'='+value for key,value in settings.items())
    write_json(folder/'command.json', dict(argv=cmd, job=os.environ.get('PBS_JOBID')))
    started = time.monotonic()
    with (folder/'run.log').open('w') as log:
        code = subprocess.call(['/usr/bin/time','-v','-o',str(folder/'time.txt')]+cmd,
                               env=env, stdout=log, stderr=subprocess.STDOUT)
    status = dict(returncode=code, seconds=time.monotonic()-started,
                  utc=datetime.now(timezone.utc).isoformat(), job=os.environ.get('PBS_JOBID'))
    write_json(folder/'status.json',status)
    print(json.dumps(dict(task=task['label'], **status)), flush=True)
    if code:
        raise RuntimeError('Benchmark failed; original parameters retained: '+str(folder))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--shard',type=int)
    parser.add_argument('--references',action='store_true')
    args = parser.parse_args()
    verify_sources()
    tasks = json.loads((ROOT/'plan.json').read_text())['tasks']
    if args.gate:
        launch(tasks[0], gate=True)
    else:
        for task in tasks:
            reference = task.get('reference_only',False)
            if reference == args.references and (args.references or task['id'] % 4 == args.shard):
                launch(task)


if __name__ == '__main__':
    main()
