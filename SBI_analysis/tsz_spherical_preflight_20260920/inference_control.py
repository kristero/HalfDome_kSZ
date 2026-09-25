"""P0-only positive control for compression/SBI; deliberately not a new sky run.

The exact P0^2 scaling of a saved B12 spectrum gives a known likelihood. A
fixed, saved noise covariance is used only for this software control; it is
not a candidate-dependent physical SO covariance or nine-parameter validation.
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import trapezoid
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform
try:
    from sbi.neural_nets import posterior_nn
except ImportError:
    from sbi.utils.get_nn_models import posterior_nn

ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent/'tsz_guardrail_study'


def main():
    torch.set_num_threads(2);torch.manual_seed(20260920)
    rng=np.random.default_rng(20260920)
    ell=np.arange(7980);edges=np.r_[np.arange(80,7881,200),7980]
    cl=np.load(STUDY/'maps/Battaglia12/pixel16384_2/masked_clean_cl.npy')
    dl=cl[:7980]*ell*(ell+1)/(2*np.pi)
    template=np.array([np.average(dl[a:b],weights=2*ell[a:b]+1) for a,b in zip(edges[:-1],edges[1:])])
    covariance=np.load(STUDY/'audit/noise_covariance.npz')['covariance']
    derivative=2*template/18.1
    weight=np.linalg.solve(covariance,derivative)
    weight/=np.sqrt(weight@covariance@weight)
    scale=weight@template
    theta=rng.uniform(1,60,(1024,1))
    # This compression is exactly sufficient for the one-amplitude Gaussian
    # likelihood; use signed asinh so cross-spectrum sign is never discarded.
    compressed=scale*(theta/18.1)**2+rng.normal(size=(len(theta),1))
    x=np.arcsinh(compressed/scale).astype('float32')
    prior=BoxUniform(torch.tensor([1.]),torch.tensor([60.]))
    truth=np.array([5.,18.1,40.,55.])
    obs=scale*(truth/18.1)**2
    grids=np.linspace(1,60,30001)
    exact=[]
    for value in obs:
        l=-.5*(value-scale*(grids/18.1)**2)**2;l-=l.max()
        prob=np.exp(l);prob/=trapezoid(prob,grids)
        exact.append(float(trapezoid(grids*prob,grids)))
    records=[]
    started=time.monotonic()
    for shuffled in [False,True]:
        torch.manual_seed(20260920)
        use_x=x[rng.permutation(len(x))] if shuffled else x
        builder=posterior_nn(model='maf',hidden_features=32,num_transforms=3,
                             z_score_x='independent',z_score_theta='independent')
        inference=SNPE(prior=prior,density_estimator=builder,device='cpu',show_progress_bars=False)
        estimator=inference.append_simulations(torch.tensor(theta,dtype=torch.float32),torch.tensor(use_x)).train(
            training_batch_size=128,max_num_epochs=300,stop_after_epochs=30,learning_rate=5e-4,
            validation_fraction=.15,show_train_summary=False)
        posterior=inference.build_posterior(estimator)
        predictions=[]
        for value in obs:
            context=torch.tensor([np.arcsinh(value/scale)],dtype=torch.float32)
            samples=posterior.sample((2048,),x=context,show_progress_bars=False).numpy()[:,0]
            predictions.append(dict(mean=float(samples.mean()),q16=float(np.quantile(samples,.16)),
                q84=float(np.quantile(samples,.84))))
        summary=inference.summary
        records.append(dict(shuffled=shuffled,predictions=predictions,
            normalized_mean_error=float(np.sqrt(np.mean((np.array([p['mean'] for p in predictions])-exact)**2))/59),
            epochs=summary.get('epochs_trained'),validation_log_probs=summary.get('validation_log_probs',[])[-5:]))
    # Exact noiseless amplitude inversion also checks labels and units.
    recovered=18.1*np.sqrt(obs/scale)
    assert np.max(abs(recovered-truth))<1e-12
    result=dict(scope=__doc__,truth=truth.tolist(),exact_posterior_means=exact,
        controls=records,seconds=time.monotonic()-started,
        noiseless_recovery_max_error=float(np.max(abs(recovered-truth))),
        moped_variance=float(weight@covariance@weight),n_train_total=len(theta))
    (ROOT/'results/inference_control.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
