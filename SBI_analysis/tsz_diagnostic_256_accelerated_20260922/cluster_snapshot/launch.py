"""Immutable cluster requests, source verification and explicit failure records."""
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


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_sources():
    for manifest in ['source_manifest.json', 'external_dependencies.json']:
        for name, expected in json.loads((ROOT/manifest).read_text())['sha256'].items():
            path = Path(name) if name.startswith('/') else ROOT/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen input changed: '+str(path))


def launch(task, relative, threads):
    verify_sources()
    folder = ROOT/relative
    folder.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(dict(task=task,
        sources=json.loads((ROOT/'source_manifest.json').read_text())),sort_keys=True).encode()).hexdigest()
    marker = folder/'status.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['identity_sha256'] != identity:
            raise RuntimeError('Refusing changed request identity: '+str(folder))
        if previous['returncode'] == 0:
            return
        raise RuntimeError('Previous failure requires inspection: '+str(folder))
    claim = folder/'claim'
    claim.mkdir()
    try:
        atomic_json(folder/'request.json', task)
        (folder/'task.toml').write_text(toml.dumps(task))
        cmd = json.loads((ROOT/'reference/base_command.json').read_text())['argv']
        cmd = [str(ROOT/'engine.jl') if value.endswith('/gate.jl') else
               '--threads='+str(threads) if value.startswith('--threads=') else
               'output_dir='+str(folder/'raw') if value.startswith('output_dir=') else
               'cache_dir='+str(folder/'cache') if value.startswith('cache_dir=') else value for value in cmd]
        runtime = CAMPAIGN/'runtime'
        env = dict(os.environ,FLAMINGO_CAMPAIGN=str(CAMPAIGN),
            HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'),PREFLIGHT_OUTPUT=str(folder),
            BENCHMARK_TASK=str(folder/'task.toml'),HALO_BLOCK_SIZE='256',
            JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
            LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS='1',
            HDF5_USE_FILE_LOCKING='FALSE')
        atomic_json(folder/'command.json',dict(argv=cmd,job=os.environ.get('PBS_JOBID')))
        started=time.monotonic()
        with (folder/'run.log').open('w') as log:
            code=subprocess.call(['/usr/bin/time','-v','-o',str(folder/'time.txt')]+cmd,
                env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(marker,dict(returncode=code,seconds=time.monotonic()-started,
            identity_sha256=identity,utc=datetime.now(timezone.utc).isoformat(),job=os.environ.get('PBS_JOBID')))
        if code:
            raise RuntimeError('Task failed; all identities retained: '+str(folder))
    finally:
        claim.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',type=Path)
    parser.add_argument('--out',required=True)
    parser.add_argument('--threads',type=int,default=26)
    args=parser.parse_args()
    launch(json.loads(args.task.read_text()),args.out,args.threads)
