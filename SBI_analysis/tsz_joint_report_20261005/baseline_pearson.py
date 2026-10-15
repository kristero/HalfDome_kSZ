#!/usr/bin/env python3
"""Posterior Pearson matrices for the fixed seed-100/mask-12345 observation.

The original trained networks remain frozen. Smoothing the observation is a
diagnostic perturbation, not a new calibrated inference method. The controls are
the exact saved smooth-change inputs, not smoothing of the baseline or noise.
Each posterior uses physical parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MERLIN_PARAM_SPACE', 'linear')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter1d

PARENT=Path('/lustre/work/kristero10/tsz_robustness_methods_20261004')
OLD=Path('/lustre/work/kristero10/tsz_seed_mask_variation_20261002')
ROOT=Path('/lustre/work/kristero10/tsz_joint_report_20261005')
DIAG=Path('/lustre/work/kristero10/tsz_seed_offset_diagnosis_20261005')
PARAMETERS=['P0','xc','beta','alpha_m_P0','alpha_m_xc','alpha_m_beta',
            'alpha_z_P0','alpha_z_xc','alpha_z_beta']
TRUTH=np.array([18.1,.497,4.35,.154,-.00865,.0393,-.758,.731,.415])


def write_json(path, value):
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(value,indent=2,default=encode)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def correlation_and_mcse(draws, blocks=90):
    """Correlation of all draws, plus a batch estimate of Monte Carlo precision.

    Batch uncertainty describes posterior sampling only, not uncertainty from
    training a flow, changing the simulation or selecting a smoothing kernel.
    """
    correlation=np.corrcoef(draws,rowvar=False)
    assert np.allclose(correlation,correlation.T,atol=1e-12)
    assert np.allclose(np.diag(correlation),1)
    assert np.max(np.abs(correlation))<=1+1e-12
    batches=np.stack([np.corrcoef(part,rowvar=False) for part in np.array_split(draws,blocks)])
    mcse=batches.std(axis=0,ddof=1)/np.sqrt(blocks)
    return correlation,mcse


def prepare(root,target):
    assert target=='relative_change'
    ell=np.arange(80,7980)
    dl_factor=ell*(ell+1)/(2*np.pi)
    clean=np.load(PARENT/'results/seed_100/mask_12345/clean_cl.npy')[80:7980]*dl_factor
    noisy=np.load(PARENT/'data/grid.npz')['full'][0]
    residual=noisy-clean
    saved_spectra=dict(np.load(DIAG/'arrays/spectra.npz'))
    cases=json.loads((DIAG/'counterfactuals.json').read_text())
    case_ids={case['label']:i for i,case in enumerate(cases)}
    assert np.array_equal(noisy,saved_spectra['noisy'][0])
    assert np.array_equal(clean,saved_spectra['clean'][0])
    # This is the SAME mean-simulation control in the actual/clean/smooth/fine
    # panels. It changes the baseline by a smooth part of the other skies'
    # clean difference; it does not smooth the baseline itself or its noise.
    mean_change=saved_spectra['clean'][10::10].mean(axis=0)-clean
    spectra=[noisy]
    definitions=[dict(label='original',target='unchanged',sigma_ell=0)]
    control_errors=[]
    for sigma in [5,50]:
        smooth_change=clean*gaussian_filter1d(mean_change/clean,sigma,mode='nearest')
        reconstructed=noisy+smooth_change
        source_label=f'smooth_clean_{sigma}'
        spectrum=saved_spectra['counterfactual'][case_ids[source_label]]
        error=float(np.max(np.abs(reconstructed-spectrum))/np.max(np.abs(spectrum)))
        assert error<1e-12,error
        control_errors.append(error)
        spectra.append(spectrum)
        definitions.append(dict(label=f'smooth_sigma{sigma}',target=target,sigma_ell=sigma,
                                source_control=source_label))
    spectra=np.asarray(spectra)
    with np.load(OLD/'reference/whitening_pca.npz') as w:
        projected=solve_triangular(w['L'],spectra.T,lower=True).T@w['V']
    f=dict(np.load(OLD/'models/linear/features_merlin.npz'))
    z=np.arcsinh((projected-f['s_center'])/f['s_scale'])
    x=((z-f['z_mean'])/f['z_sd']).astype(np.float32)
    saved=np.load(PARENT/'data/original_merlin_observations.npy')[0]
    error=float(np.max(np.abs(saved-x[0])))
    assert error<1e-5,error
    np.savez(root/'data/baseline_spectra.npz',ell=ell,clean=clean,noisy=noisy,residual=residual,
             mean_clean_change=mean_change,spectra=spectra,features=x,low=f['low'],high=f['high'])
    write_json(root/'control_definition.json',dict(simulation_seed=100,mask_seed=12345,
        target=target,cases=definitions,original_input_max_error=error,
        saved_control_reconstruction_relative_error=control_errors,
        change_definition='Mean clean difference of seeds 102..118 at fixed mask 12345 relative to seed100; fractional Gaussian smoothing; baseline SO residual unchanged',
        primary_sigma_ell=5,secondary_sigma_ell=50,
        kernel='Gaussian in integer multipole; scipy gaussian_filter1d, nearest edges, default truncate=4',
        observation='Same trained beam correction, noise-only whitening, PCA and asinh standardization',
        interpretation='Diagnostic changed input; the saved model is not retrained or recalibrated.'))
    return x


def evaluate(root,x,space,threads,target):
    import torch
    sys.path.insert(0,str(PARENT/'snapshot/inference'))
    import merlin_compression as mc
    import train_ensemble as te
    ens=mc.ensemble(OLD/f'models/{space}','merlin',threads)
    torch.manual_seed(20261006)
    rng=np.random.default_rng(20261006)
    u=ens.sample_u(x,90000,rng=rng)
    draws=te.from_u(u,ens.low,ens.high)
    # Reuse the fully verified original linear baseline samples for exact parity.
    if space=='linear':
        saved=np.load(PARENT/'candidates/original_merlin/posterior_final.npy',mmap_mode='r')
        draws[:,0]=np.asarray(saved[:,0],dtype=np.float64)
    truth_u=te.to_u(TRUTH[None],ens.low,ens.high)
    truth_lp=ens.log_prob_u(np.repeat(truth_u[None],len(x),axis=1),x)[0]
    truth_lp-=te.log_jacobian_theta_per_u(truth_u,ens.low,ens.high)[0]
    # Density check uses a small sample subset; Pearson uses every retained draw.
    density_u=te.to_u(draws[::15],ens.low,ens.high)
    lp=ens.log_prob_u(density_u,x)-te.log_jacobian_theta_per_u(density_u,ens.low,ens.high)
    hpd=np.mean(lp>truth_lp[None],axis=0)
    correlations=[]
    uncertainty=[]
    rows=[]
    for i,name in enumerate(['original','smooth_sigma5','smooth_sigma50']):
        sample=draws[:,i]
        corr,se=correlation_and_mcse(sample)
        correlations.append(corr)
        uncertainty.append(se)
        inside=np.all((sample>=ens.low)&(sample<=ens.high),axis=1)
        q=np.quantile(sample,[.025,.16,.5,.84,.975],axis=0)
        rows.append(dict(label=name,n=len(sample),q025=q[0],q16=q[1],median=q[2],q84=q[3],q975=q[4],
                         width68=q[3]-q[1],outside_any_prior_fraction=1-inside.mean(),
                         inside_prior_samples=int(inside.sum()),
                         inside_prior_correlation=np.corrcoef(sample[inside],rowvar=False),
                         truth_hpd_theta=hpd[i],log_prob_truth=truth_lp[i]))
        np.save(root/f'data/{space}_{name}_posterior.npy',sample.astype(np.float32))
    correlations=np.asarray(correlations)
    uncertainty=np.asarray(uncertainty)
    delta=correlations-correlations[:1]
    delta_se=np.sqrt(uncertainty**2+uncertainty[:1]**2)
    delta_se[0]=0
    triangles=np.triu_indices(9,1)
    changes=[]
    for i in range(1,3):
        order=np.argsort(np.abs(delta[i][triangles]))[::-1]
        changes.append([dict(first=PARAMETERS[triangles[0][k]],second=PARAMETERS[triangles[1][k]],
                             original=correlations[0][triangles[0][k],triangles[1][k]],
                             smooth=correlations[i][triangles[0][k],triangles[1][k]],
                             delta=delta[i][triangles[0][k],triangles[1][k]],
                             delta_mcse=delta_se[i][triangles[0][k],triangles[1][k]]) for k in order])
    result=dict(parameter_order=PARAMETERS,param_space=space,smoothing_target=target,
                correlation=correlations,correlation_mcse=uncertainty,delta=delta,delta_mcse=delta_se,
                largest_changes=changes,cases=rows,model_seed_ids=ens.seeds,
                mcse_definition='SD of 90 independent draw-batch correlations / sqrt(90); approximate sampling precision only',
                correlation_definition='Pearson among physical theta coordinates, all posterior draws; no prior clipping',
                caveat='Changed-input diagnostics do not establish calibrated posterior constraints.')
    write_json(root/f'pearson_{space}.json',result)
    np.savez(root/f'data/{space}_plot_draws.npz',draws=draws[::20])
    print('PEARSON',space,'P0-beta',correlations[:,0,2],'P0-xc',correlations[:,0,1],flush=True)


def main(args):
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    root=args.root
    (root/'data').mkdir(parents=True,exist_ok=True)
    start=time.time()
    if args.bounded:
        x=np.load(root/'data/baseline_spectra.npz')['features']
        evaluate(root,x,'logit',args.threads,args.target)
        return
    x=prepare(root,args.target)
    evaluate(root,x,'linear',args.threads,args.target)
    env=os.environ.copy();env['MERLIN_PARAM_SPACE']='logit'
    command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--target',args.target,
             '--threads',str(args.threads),'--bounded']
    subprocess.run(command,env=env,check=True)
    inputs=[PARENT/'comparison/COMPLETE.json',PARENT/'data/grid.npz',
            PARENT/'candidates/original_merlin/posterior_final.npy',
            PARENT/'data/original_merlin_observations.npy',
            PARENT/'results/seed_100/mask_12345/clean_cl.npy',OLD/'reference/whitening_pca.npz',
            OLD/'models/linear/features_merlin.npz',OLD/'models/logit/features_merlin.npz']
    inputs.extend([DIAG/'arrays/spectra.npz',DIAG/'counterfactuals.json',DIAG/'FINAL_COMPLETE.json'])
    for directory in [OLD/'models/linear',OLD/'models/logit']:
        inputs.extend(directory.glob('merlin_seed*.pt'))
        inputs.extend(directory.glob('merlin_seed*.json'))
    inputs.extend((PARENT/'snapshot/inference').glob('*.py'))
    inputs.extend([Path(__file__).resolve(),root/'pearson.pbs'])
    outputs=list((root/'data').glob('*'))+[root/'pearson_linear.json',root/'pearson_logit.json',root/'control_definition.json']
    write_json(root/'PEARSON_COMPLETE.json',dict(seconds=time.time()-start,
        inputs={str(p):sha(p) for p in inputs},outputs={str(p.relative_to(root)):sha(p) for p in outputs},
        observations=3,posterior_samples_per_observation=90000,baseline=[100,12345],smoothing_target=args.target))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--target',choices=['relative_change'],default='relative_change')
    parser.add_argument('--threads',type=int,default=26)
    parser.add_argument('--bounded',action='store_true')
    main(parser.parse_args())
