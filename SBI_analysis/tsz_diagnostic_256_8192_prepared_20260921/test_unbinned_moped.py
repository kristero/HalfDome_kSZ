"""Independent dense-covariance and within-bin information regression tests."""
import json
from pathlib import Path
import time
import numpy as np
from sklearn.covariance import OAS
from unbinned_moped import oas_inverse_action
from diagnostic_analysis import transform_fit, project, rebin_unbinned
from design_tests import LOW, HIGH, design

ROOT=Path(__file__).resolve().parent
started=time.monotonic()
rng=np.random.default_rng(20260920)
comparisons=[]
for n,p,scale in [(70,32,1.),(30,80,1.),(30,80,1e-12)]:
    residuals=rng.normal(size=(n,p))*scale
    derivatives=rng.normal(size=(p,9))*scale
    actual,action,meta=oas_inverse_action(residuals,derivatives)
    oracle=OAS().fit(residuals)
    covariance=oracle.covariance_+np.eye(p)*meta['covariance_ridge']
    expected=np.linalg.solve(covariance,derivatives)
    error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    assert error<1e-10, error
    np.testing.assert_allclose(action(derivatives),covariance@derivatives,rtol=1e-10,atol=1e-50)
    np.testing.assert_allclose(meta['oas_shrinkage'],oracle.shrinkage_,rtol=1e-12)
    comparisons.append(dict(rows=n,features=p,scale=scale,relative_inverse_error=float(error)))

# Nine shape modes live inside the first bin and cancel under its exact weights.
# A 40-bin-only implementation cannot retain them; the unbinned path must.
theta=design(64);u=(theta-LOW)/(HIGH-LOW)
ell=np.arange(80,7980);weights=2*ell+1
shape=np.zeros((9,len(ell)))
for j in range(9):
    shape[j,2*j]=1
    shape[j,2*j+1]=-weights[2*j]/weights[2*j+1]
clean=(1+.5*(u-.5)@shape)*1e-13
noisy=clean+rng.normal(0,1e-15,clean.shape)
binned=rebin_unbinned(clean,ell)
assert np.max(np.ptp(binned,axis=0))<1e-27
state,metadata=transform_fit(clean,noisy,theta,LOW,HIGH,'moped_unbinned')
assert state['matrix'].shape==(7900,9)
assert np.min(np.std(project(clean,state),axis=0))>.1
result=dict(dense_covariance_comparisons=comparisons,
    unbinned_input_features=7900,retained_directions=metadata['retained'],
    within_bin_shape_test='Nine modes invisible to 40 bins remain in unbinned MOPED',
    covariance_identity_error=metadata['compression_covariance_identity_error'],
    fisher_relative_error=metadata['retained_fisher_relative_error'],
    seconds=time.monotonic()-started,scope='Software and linear algebra, not physical SBI validation')
(ROOT/'results/unbinned_moped_tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
