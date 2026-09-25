"""Exercise the installed SBI API and split/sampling contract, not science accuracy."""
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn
from diagnostic_analysis import bounded_samples,transform_fit,project

torch.set_num_threads(2)
torch.manual_seed(20260922)
class FixedSplitSNPE(SNPE):
    def get_dataloaders(self,*args,**kwargs):
        parent=super().get_dataloaders
        bound=inspect.signature(parent).bind_partial(*args,**kwargs)
        bound.arguments['resume_training']=True
        return parent(*bound.args,**bound.kwargs)

low=np.zeros(9);high=np.ones(9)
prior=BoxUniform(torch.zeros(9),torch.ones(9))
theta=prior.sample((64,));x=theta+.05*torch.randn(64,9)
inference=FixedSplitSNPE(prior=prior,density_estimator=posterior_nn('maf',hidden_features=32,
    num_transforms=3,z_score_x='none'),show_progress_bars=False)
inference.append_simulations(theta,x)
inference.train_indices=torch.arange(54);inference.val_indices=torch.arange(54,64)
estimator=inference.train(max_num_epochs=3,training_batch_size=32,validation_fraction=10/64,
    show_train_summary=False)
assert torch.equal(inference.train_indices,torch.arange(54))
assert torch.equal(inference.val_indices,torch.arange(54,64))
samples,draws=bounded_samples(estimator,x[0].numpy(),low,high,32,100000)
assert samples.shape==(32,9) and np.isfinite(samples).all()
root=Path(__file__).resolve().parent
rng=np.random.default_rng(17)
raw=rng.normal(size=(32,7900))*1e-14
state,meta=transform_fit(raw,raw,theta[:32].numpy(),low,high,'moped_fixed')
actual=project(raw,state)
assert actual.shape==(32,9) and np.isfinite(actual).all()
np.testing.assert_allclose(actual.mean(0),0,atol=1e-6)
summary=inference.summary
result=dict(passed=True,split_preserved=True,samples=len(samples),proposals=draws,
    best_validation_log_prob=float(inference._best_val_log_prob),
    validation_summary_keys=list(summary),fixed_unbinned_features=meta['input_features'],
    scope='Software-only smoke test; no posterior accuracy claim')
(root/'results').mkdir(exist_ok=True)
(root/'results/training_software_test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
