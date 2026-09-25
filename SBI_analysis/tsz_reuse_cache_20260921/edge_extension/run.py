"""Bounded cache-edge experiment, no dataset submission or parameter rejection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import toml

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
CAMPAIGN=Path('/lustre/work/kristero10/flamingo_tsz_comparison_20260914')


def main():
    gate='--gate' in sys.argv
    for base in [PARENT,ROOT]:
        for name,sha in json.loads((base/'manifest.json').read_text())['sha256'].items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest()==sha,name
    for name,sha in json.loads((PARENT/'external_dependencies.json').read_text())['sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    tasks=json.loads((ROOT/'plan.json').read_text())
    if gate:tasks=tasks[:1]
    for task in tasks:
        folder=ROOT/('gate' if gate else f"controls/{task['id']:03d}")
        folder.mkdir(parents=True,exist_ok=True);(folder/'claim').mkdir()
        (folder/'task.toml').write_text(toml.dumps(task))
        cmd=json.loads((PARENT/'gate/command.json').read_text())['argv']
        script='gate.jl' if gate else 'entrypoint.jl'
        threads=2 if gate else 26
        cmd=[str(ROOT/script) if v.endswith('/gate.jl') else
             '--threads='+str(threads) if v.startswith('--threads=') else
             'output_dir='+str(folder/'raw') if v.startswith('output_dir=') else
             'cache_dir='+str(folder/'cache') if v.startswith('cache_dir=') else v for v in cmd]
        runtime=CAMPAIGN/'runtime'
        env=dict(os.environ,FLAMINGO_CAMPAIGN=str(CAMPAIGN),HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'),
            PREFLIGHT_OUTPUT=str(folder),BENCHMARK_TASK=str(folder/'task.toml'),
            JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
            LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS='1',HDF5_USE_FILE_LOCKING='FALSE')
        start=time.monotonic()
        with (folder/'run.log').open('w') as log:
            code=subprocess.call(['/usr/bin/time','-v','-o',str(folder/'time.txt')]+cmd,
                env=env,stdout=log,stderr=subprocess.STDOUT)
        (folder/'status.json').write_text(json.dumps(dict(returncode=code,seconds=time.monotonic()-start,
                                                         job=os.environ.get('PBS_JOBID')),indent=2)+'\n')
        if code:raise RuntimeError('Cache-edge test failed; inspect '+str(folder))
    if not gate:
        subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)


if __name__=='__main__':main()
