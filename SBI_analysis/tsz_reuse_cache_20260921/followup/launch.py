"""Launch only the scalar cache audit, reusing frozen cluster configuration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from run import verify_sources, write_json, CAMPAIGN

verify_sources()
out = ROOT/'probe'
out.mkdir(exist_ok=True)
(out/'claim').mkdir()  # never silently rerun or overwrite
cmd = json.loads((ROOT.parent/'gate/command.json').read_text())['argv']
cmd = [str(ROOT/'cache_probe.jl') if v.endswith('/gate.jl') else
       '--threads=26' if v.startswith('--threads=') else v for v in cmd]
runtime=CAMPAIGN/'runtime'
env=dict(os.environ,FLAMINGO_CAMPAIGN=str(CAMPAIGN),HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'),
    PREFLIGHT_OUTPUT=str(out),JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
    LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='26',MKL_NUM_THREADS='1',HDF5_USE_FILE_LOCKING='FALSE')
start=time.monotonic()
with (out/'run.log').open('w') as log:
    code=subprocess.call(['/usr/bin/time','-v','-o',str(out/'time.txt')]+cmd,
                         env=env,stdout=log,stderr=subprocess.STDOUT)
write_json(out/'status.json',dict(returncode=code,seconds=time.monotonic()-start,job=os.environ.get('PBS_JOBID')))
raise SystemExit(code)
