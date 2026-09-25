"""Read-only verification of the unsubmitted design and its cluster runtime."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/lustre/work/kristero10/tsz_spherical_preflight_20260920'
RUN = '/lustre/work/kristero10/tsz_spherical_diagnostic_256_prepared_20260920'

probe = '''
import hashlib, inspect, json, pathlib, subprocess
import numpy, scipy, sklearn, torch, sbi
from sbi.inference import SNPE
torch.set_num_threads(1)
root = pathlib.Path(%r)
run = pathlib.Path(%r)
manifest = json.loads((run/'manifest.json').read_text())
source = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
          for name in manifest['source_sha256']}
assert source == manifest['source_sha256'], 'Producer source drift'
theta = numpy.load(run/'theta_design.npy')
seeds = numpy.load(run/'noise_seeds.npy')
heldout = numpy.load(run/'heldout_design.npy')
assert theta.shape == (256, 9) and seeds.shape == (256, 2)
assert len(numpy.unique(seeds)) == 512 and heldout.sum() == 64
assert manifest['moped_input'] == dict(ell_min=80,ell_max=7979,features=7900,binned=False)
observations={}
for path in (run/'observations').glob('*.npz'):
    with numpy.load(path) as item:
        numpy.testing.assert_array_equal(item['ell_unbinned'],numpy.arange(80,7980))
        assert item['noisy_dl_unbinned'].shape==(7900,)
        observations[path.stem]=7900
rows = list((run/'rows').glob('*')) if (run/'rows').exists() else []
assert not rows, 'Diagnostic generation has started'
jobs = json.loads(subprocess.check_output(['qstat','-f','-F','json']))['Jobs']
diagnostic_jobs = {key: value['job_state'] for key, value in jobs.items()
                   if value.get('Job_Name','').startswith('sphere_diag')}
assert not diagnostic_jobs, 'Diagnostic submitted'
print(json.dumps(dict(
    count=256, training_pool=192, heldout=64, unique_split_seeds=512,
    generated_rows=len(rows), diagnostic_jobs=diagnostic_jobs,
    source_sha256=source,
    analysis_sha256=hashlib.sha256((root/'diagnostic_analysis.py').read_bytes()).hexdigest(),
    moped_module_sha256=hashlib.sha256((root/'unbinned_moped.py').read_bytes()).hexdigest(),
    moped_input=manifest['moped_input'],observation_multipoles=observations,
    runtime={name: getattr(module,'__version__','unknown') for name,module in
             [('numpy',numpy),('scipy',scipy),('sklearn',sklearn),('torch',torch),('sbi',sbi)]},
    loader_signature=str(inspect.signature(SNPE.get_dataloaders)),
    scope='Read-only preparation verification; no physical row or inference run')))
''' % (REMOTE, RUN)

if __name__ == '__main__':
    result = subprocess.run(
        ['ssh','idark','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
         'CUDA_VISIBLE_DEVICES=-1','/home/anaconda3/bin/python3','-'],
        input=probe, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout)
    assert record['analysis_sha256'] == hashlib.sha256(
        (ROOT/'diagnostic_analysis.py').read_bytes()).hexdigest()
    assert record['moped_module_sha256'] == hashlib.sha256(
        (ROOT/'unbinned_moped.py').read_bytes()).hexdigest()
    record['utc'] = datetime.now(timezone.utc).isoformat()
    (ROOT/'results/prepared_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
