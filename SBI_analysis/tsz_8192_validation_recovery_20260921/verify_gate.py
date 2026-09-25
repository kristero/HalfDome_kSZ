"""Refuse a production-sized test unless the small runtime checks passed."""
import json
from pathlib import Path
import numpy as np
try:
    import tomllib
except ImportError:
    import toml as tomllib

ROOT=Path(__file__).resolve().parent
results=ROOT/'results'
quadrature=tomllib.loads((results/'quadrature.toml').read_text())
assert quadrature['reference_points']==10180
assert quadrature['corner_columns']==36864 and quadrature['failure_path_passed']
assert quadrature['max_relative_error']<1e-7
assert tomllib.loads((results/'noise_algebra.toml').read_text())['relative_error']<1e-12
assert json.loads((results/'analysis_software_test.json').read_text())['passed']
workflow=json.loads((results/'workflow.json').read_text())
assert workflow['interruption_resume_passed'] and workflow['changed_identity_refused']
standard=np.load(results/'load_standard/allocation_test_map.npy')
lean=np.load(results/'load_lean/allocation_test_map.npy')
np.testing.assert_allclose(standard,lean,rtol=1e-14,atol=0)
assert np.count_nonzero(standard)>0
result=dict(passed=True,quadrature=quadrature,
    allocation_relative_error=float(np.linalg.norm(standard-lean)/np.linalg.norm(standard)),
    scope='Small runtime/software tests only; not full-sky validation')
(results/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
