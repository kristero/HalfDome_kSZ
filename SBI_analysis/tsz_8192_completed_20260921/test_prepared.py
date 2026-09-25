"""Check the prepared launcher without generating any catalogue maps."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent/'tsz_diagnostic_256_8192_prepared_20260921'
OLD=ROOT.parent/'tsz_spherical_preflight_20260920'
VALIDATED=ROOT.parent/'tsz_8192_validation_recovery_20260921'
sys.path.insert(0,str(NEW))
import diagnostic

manifest=json.loads((NEW/'diagnostic_256/manifest.json').read_text())
assert manifest['raw_nside']==8192 and manifest['output_nside']==4096
assert manifest['moped_input']['features']==7900 and not manifest['submission_authorized']
for name,expected in manifest['source_sha256'].items():
    assert hashlib.sha256((NEW/name).read_bytes()).hexdigest()==expected,name
for name,expected in manifest['design_sha256'].items():
    assert hashlib.sha256((NEW/'diagnostic_256'/name).read_bytes()).hexdigest()==expected
    assert (NEW/'diagnostic_256'/name).read_bytes()==(OLD/'diagnostic_256'/name).read_bytes()
for name in ['fullsky_test.jl','spherical_truncation_profiles.jl']:
    assert (NEW/name).read_bytes()==(VALIDATED/name).read_bytes()
for path in NEW.glob('*.pbs'):
    assert b'\r' not in path.read_bytes()
    subprocess.run(['bash','-n',str(path)],check=True)

theta=np.load(NEW/'diagnostic_256/theta_design.npy')
seeds=np.load(NEW/'diagnostic_256/noise_seeds.npy')
assert theta.shape==(256,9) and len(np.unique(seeds))==512
assert sorted(i for w in range(4) for i in range(256) if i%4==w)==list(range(256))
with tempfile.TemporaryDirectory(prefix='tsz_prepared_test_') as directory:
    temp=Path(directory);diagnostic.RUN=temp/'run';diagnostic.RUN.mkdir()
    diagnostic.CAMPAIGN=temp/'campaign';(diagnostic.CAMPAIGN/'preflight').mkdir(parents=True)
    (diagnostic.CAMPAIGN/'preflight/metadata_manifest.json').write_text(json.dumps(
        {'halfdome_reference':{'simulation_request':{'command':[]}}}))
    np.save(diagnostic.RUN/'theta_design.npy',theta[:2]);np.save(diagnostic.RUN/'noise_seeds.npy',seeds[:2])
    calls=[]
    def fake_call(command,env,**kwargs):
        assert 'nside=8192' in command and env['VALIDATION_LEAN_MAPS']=='1'
        assert env['DIAGNOSTIC_SPLIT_SEEDS'] in [','.join(map(str,s)) for s in seeds[:2]]
        out=Path(env['PREFLIGHT_OUTPUT'])
        for name in ['masked_clean_cl.npy','masked_noisy_cross_cl.npy']:
            np.save(out/name,np.ones(7980)*1e-16)
        calls.append(command)
        return 0
    diagnostic.subprocess.call=fake_call
    for i in range(2):diagnostic.run_row(i,theta[i],26)
    diagnostic.summarize(2)
    assert json.loads((diagnostic.RUN/'summary.json').read_text())['completed']==2
    diagnostic.run_row(0,theta[0],26)
    assert len(calls)==2  # identical completed row reused, never regenerated
    claim=diagnostic.RUN/'claims/00000';claim.mkdir()
    try:
        diagnostic.run_row(0,theta[0],26)
        raise AssertionError('Duplicate claim accepted')
    except FileExistsError:pass
    claim.rmdir()
    try:
        diagnostic.summarize(3)
        raise AssertionError('Incomplete dataset accepted')
    except RuntimeError:pass

result=dict(passed=True,design_rows=256,unique_split_seeds=512,
    checks=['source identities','unchanged saved design','raw8192 argument','lean allocation flag',
            'noise seed routing','disjoint four-worker partitions','exclusive claims',
            'completed-row reuse','incomplete-dataset rejection','LF PBS and shell syntax'],
    scope='Launcher and artifact checks with a mocked subprocess; no new catalogue row or SBI training executed')
(ROOT/'results/prepared_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
