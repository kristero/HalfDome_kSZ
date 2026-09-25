"""Held-out 256-row diagnostics, with optional explicitly small-sample SBI.

MOPED takes all 7900 individual multipoles, ell=80..7979, before compression.
It uses locally regressed derivatives and is compared with 40 bins and PCA.
Nothing here authorizes a larger production run.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from unbinned_moped import regression_moped
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def transform_fit(clean,noisy,theta,low,high,method):
    if method=='moped_fixed':
        # Independent finite-difference B12 derivatives and noise calibration.
        # All 7900 raw multipoles enter the linear projection before any asinh.
        context=np.load(Path(__file__).parent/'reference/compression_context.npz')
        matrix=context['Battaglia12_rcond1e-06']
        assert matrix.shape==(7900,9)
        raw=noisy@matrix
        scale=np.maximum(np.median(abs(raw),axis=0),1e-30)
        y=np.arcsinh(raw/scale)
        state=dict(matrix=matrix,scale=scale,center=y.mean(0),
            spread=np.maximum(y.std(0),1e-8),project_first=np.array(True))
        return state,dict(input_features=7900,retained=9,
            derivative_method='Frozen B12 finite differences, step 0.0025 prior width; P0 analytic',
            coordinate_transform='Linear raw unbinned MOPED, then training-only signed asinh',
            caveat='Local B12 information compression; not globally sufficient throughout the extended prior')
    scale=np.maximum(np.median(abs(clean),axis=0),1e-30)
    y=np.arcsinh(noisy/scale);yc=np.arcsinh(clean/scale)
    center=y.mean(0);spread=np.maximum(y.std(0),1e-8)
    z=(y-center)/spread;zc=(yc-center)/spread
    meta={}
    if method=='bins40':matrix=np.eye(40)
    elif method=='pca':
        _,s,v=np.linalg.svd(z,full_matrices=False);matrix=v[:9].T
        meta['singular_values']=s.tolist()
    elif method=='moped_unbinned':
        matrix,meta=regression_moped(zc,z,theta,low,high)
    else:raise ValueError('Unknown compression method: '+method)
    meta['input_features']=clean.shape[1]
    projected=z@matrix
    mean=projected.mean(0);std=np.maximum(projected.std(0),1e-8)
    state=dict(scale=scale,center=center,spread=spread,matrix=matrix,mean=mean,std=std)
    return state,meta


def project(noisy,state):
    if bool(state.get('project_first',False)):
        if np.asarray(noisy).shape[-1]!=state['matrix'].shape[0]:
            raise ValueError('Unbinned multipole count changed')
        return ((np.arcsinh(np.asarray(noisy)@state['matrix']/state['scale'])-
            state['center'])/state['spread']).astype('float32')
    if np.asarray(noisy).shape[-1]!=len(state['scale']):
        raise ValueError('Spectrum and fitted transform have different feature counts')
    z=(np.arcsinh(noisy/state['scale'])-state['center'])/state['spread']
    return ((z@state['matrix']-state['mean'])/state['std']).astype('float32')


def bounded_samples(estimator,context,low,high,count,max_proposals):
    """Sample the learned density restricted to the declared box, with a work cap."""
    import inspect
    import torch
    signature=inspect.signature(estimator.sample).parameters
    accepted=[];draws=0;total=0
    condition=torch.tensor(np.asarray(context).reshape(1,-1),dtype=torch.float32)
    while total<count and draws<max_proposals:
        batch=min(2048,max_proposals-draws)
        with torch.no_grad():
            raw=(estimator.sample((batch,),condition=condition) if 'condition' in signature
                 else estimator.sample(batch,context=condition)).cpu().numpy().reshape(-1,9)
        good=raw[np.all((raw>=low)&(raw<=high),axis=1)]
        accepted.append(good);total+=len(good);draws+=len(raw)
    return np.concatenate(accepted)[:count],draws


def load_unbinned(root,ids):
    """Read retained per-row C_ell directly; never reconstruct from 40 bins."""
    ell=np.arange(80,7980,dtype=np.int64)
    factor=ell*(ell+1)/(2*np.pi)
    clean=[];noisy=[]
    for row in ids:
        folder=root/'rows'/f'{row:05d}'
        for name,target in [('masked_clean_cl.npy',clean),('masked_noisy_cross_cl.npy',noisy)]:
            spectrum=np.load(folder/name)
            if spectrum.ndim!=1 or len(spectrum)<7980:
                raise ValueError('Unbinned spectrum missing multipoles: '+str(folder/name))
            target.append(spectrum[ell]*factor)
    return ell,np.asarray(clean),np.asarray(noisy)


def rebin_unbinned(values,ell):
    edges=np.r_[np.arange(80,7881,200),7980]
    return np.stack([np.average(values[:,(ell>=a)&(ell<b)],axis=1,
        weights=2*ell[(ell>=a)&(ell<b)]+1) for a,b in zip(edges[:-1],edges[1:])],axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--train',action='store_true');p.add_argument('--threads',type=int,default=26)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--posterior-samples',type=int,default=512)
    p.add_argument('--max-proposals',type=int,default=100000)
    p.add_argument('--seeds',nargs='+',type=int,default=[20260920,20260921])
    p.add_argument('--methods',nargs='+',default=['moped_fixed','moped_unbinned','bins40'])
    args=p.parse_args()
    root=args.run_root;out=root/'analysis';out.mkdir(exist_ok=True)
    data=np.load(root/'dataset.npz');manifest=json.loads((root/'manifest.json').read_text())
    ids=data['row_id'].astype(int);theta=data['theta'];clean=data['clean_dl'];noisy=data['noisy_dl']
    if len(ids)!=manifest['count']:
        (out/'incomplete.json').write_text(json.dumps(dict(completed=len(ids),requested=manifest['count'],
            decision='Do not train on a silently filtered subset; resolve and record failures first'),indent=2))
        raise RuntimeError('Dataset incomplete: analysis stopped before training')
    assert np.array_equal(theta,np.load(root/'theta_design.npy')[ids])
    assert np.isfinite(clean).all() and np.isfinite(noisy).all()
    if not np.array_equal(np.sort(ids),np.arange(manifest['count'])):
        raise ValueError('Dataset has duplicate or missing row identities')
    ell,clean_unbinned,noisy_unbinned=load_unbinned(root,ids)
    assert np.isfinite(clean_unbinned).all() and np.isfinite(noisy_unbinned).all()
    for raw,binned in [(clean_unbinned,clean),(noisy_unbinned,noisy)]:
        np.testing.assert_allclose(rebin_unbinned(raw,ell),binned,rtol=1e-10,atol=1e-28)
    test=np.load(root/'heldout_design.npy')[ids];train=np.flatnonzero(~test);heldout=np.flatnonzero(test)
    low,high=data['lower'],data['upper'];width=high-low
    baseline=np.sqrt(np.mean(((theta[heldout]-(low+high)/2)/width)**2,axis=0))
    quality=dict(count=len(ids),heldout=len(heldout),training_pool=len(train),
        negative_cross_bins=int(np.sum(noisy<0)),negative_cross_multipoles=int(np.sum(noisy_unbinned<0)),
        moped_input=dict(features=len(ell),ell_min=int(ell[0]),ell_max=int(ell[-1]),binned=False),
        prior_mean_rmse_over_width=baseline.tolist(),
        clean_dl_range=[float(clean.min()),float(clean.max())],
        scope='Numerically gated spherical point painter; 256-row inference diagnostic')
    (out/'data_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    plt.rcParams.update({'font.size':14,'axes.labelsize':16})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(clean.T,alpha=.15);axes[0].set(yscale='log',xlabel='Bandpower index',ylabel=r'Clean $D_b$')
    axes[1].plot(noisy.T,alpha=.15)
    axes[1].set_yscale('symlog',linthresh=max(np.median(abs(noisy)),1e-30))
    axes[1].set(xlabel='Bandpower index',ylabel=r'Split-cross $D_b$')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('spectra.'+ext),dpi=180,bbox_inches='tight')
    external_observations={}
    for path in (root/'observations').glob('*.npz'):
        with np.load(path) as observation:
            np.testing.assert_array_equal(observation['ell_unbinned'],ell)
            external_observations[path.stem]={name:observation[name].copy()
                for name in ['noisy_dl','noisy_dl_unbinned']}
            np.testing.assert_allclose(rebin_unbinned(observation['noisy_dl_unbinned'][None,:],ell)[0],
                observation['noisy_dl'],rtol=1e-10,atol=1e-28)
    if not args.train:return
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    try:from sbi.neural_nets import posterior_nn
    except ImportError:from sbi.utils.get_nn_models import posterior_nn
    class FixedSplitSNPE(SNPE):
        def get_dataloaders(self,*positional,**keywords):
            # Only the loader reuses indices; train() still initializes a fresh
            # network/optimizer. This prevents preprocessing/validation leakage.
            import inspect
            parent=super().get_dataloaders
            bound=inspect.signature(parent).bind_partial(*positional,**keywords)
            bound.arguments['resume_training']=True
            return parent(*bound.args,**bound.kwargs)
    torch.set_num_threads(args.threads)
    prior=BoxUniform(torch.tensor(low,dtype=torch.float32),torch.tensor(high,dtype=torch.float32))
    results=json.loads((out/'inference_metrics.json').read_text()) if (out/'inference_metrics.json').exists() else []
    for size in sorted(set(n for n in [64,128,len(train)] if n<=len(train))):
        # Fit preprocessing on a deterministic optimization subset. Neural
        # validation and held-out rows are excluded from this fitting step.
        pool=train[:size];opt=pool[:max(20,int(.85*size))]
        for method in args.methods:
            is_unbinned=method in ['moped_unbinned','moped_fixed']
            clean_input=clean_unbinned if is_unbinned else clean
            noisy_input=noisy_unbinned if is_unbinned else noisy
            state,meta=transform_fit(clean_input[opt],noisy_input[opt],theta[opt],low,high,method)
            if is_unbinned:
                state['ell']=ell
                meta.update(ell_min=int(ell[0]),ell_max=int(ell[-1]),binned=False)
            np.savez(out/f'transform_{method}_{size}.npz',**state)
            x=project(noisy_input,state)
            for shuffled in [False,True]:
                # Shuffled controls at largest size suffice for this small pilot.
                if shuffled and size!=len(train):continue
                for seed in args.seeds:
                    if any(r['size']==size and r['method']==method and r['shuffled']==shuffled and r['seed']==seed for r in results):
                        continue
                    torch.manual_seed(seed);rng=np.random.default_rng(seed)
                    observations=x[pool].copy()
                    if shuffled:observations=observations[rng.permutation(size)]
                    builder=posterior_nn(model='maf',hidden_features=32,num_transforms=3,
                        z_score_x='none',z_score_theta='independent')
                    inference=FixedSplitSNPE(prior=prior,density_estimator=builder,device='cpu',show_progress_bars=False)
                    started=time.monotonic()
                    # Restrict the trainer's split so preprocessing sees only
                    # optimization indices, independently of its random split.
                    inference.append_simulations(torch.tensor(theta[pool],dtype=torch.float32),torch.tensor(observations))
                    train_local=torch.arange(len(opt));valid_local=torch.arange(len(opt),size)
                    inference.train_indices=train_local;inference.val_indices=valid_local
                    estimator=inference.train(training_batch_size=min(64,len(opt)),max_num_epochs=args.epochs,
                        stop_after_epochs=30,learning_rate=5e-4,validation_fraction=1-len(opt)/size,
                        show_train_summary=False)
                    assert torch.equal(inference.train_indices,train_local)
                    assert torch.equal(inference.val_indices,valid_local)
                    means=[];coverage=[];coverage95=[];ranks=[];widths=[];sampling=[];posterior_arrays=[]
                    for i in heldout:
                        samples,draws=bounded_samples(estimator,x[i],low,high,args.posterior_samples,args.max_proposals)
                        sampling.append(dict(row=int(ids[i]),accepted=len(samples),proposals=draws))
                        if len(samples)==args.posterior_samples:
                            means.append(samples.mean(0));a,b=np.quantile(samples,[.16,.84],axis=0)
                            coverage.append((theta[i]>=a)&(theta[i]<=b))
                            a95,b95=np.quantile(samples,[.025,.975],axis=0)
                            coverage95.append((theta[i]>=a95)&(theta[i]<=b95))
                            ranks.append(np.mean(samples<theta[i],axis=0))
                            widths.append((b-a)/width)
                            posterior_arrays.append(samples.astype('float32'))
                        else:
                            means.append(np.full(9,np.nan));coverage.append(np.full(9,np.nan))
                            coverage95.append(np.full(9,np.nan));ranks.append(np.full(9,np.nan))
                            widths.append(np.full(9,np.nan))
                    means=np.array(means)
                    complete=all(r['accepted']==args.posterior_samples for r in sampling)
                    result=dict(size=size,method=method,shuffled=shuffled,seed=seed,
                        rmse_over_width=np.sqrt(np.mean(((means-theta[heldout])/width)**2,axis=0)).tolist() if complete else None,
                        marginal_68_coverage=np.mean(coverage,axis=0).tolist() if complete else None,
                        marginal_95_coverage=np.mean(coverage95,axis=0).tolist() if complete else None,
                        median_68_width_over_prior_width=np.median(widths,axis=0).tolist() if complete else None,
                        best_validation_log_prob=float(inference._best_val_log_prob),
                        optimization_rows=ids[opt].tolist(),validation_rows=ids[pool[len(opt):]].tolist(),
                        heldout_count=len(heldout),sampling=sampling,sampling_complete=complete,
                        compression=meta,seconds=time.monotonic()-started,
                        limitation='64 held-out rows do not establish precise coverage or SBC')
                    if size==len(train) and not shuffled:
                        external={}
                        for label,observation in external_observations.items():
                            key='noisy_dl_unbinned' if is_unbinned else 'noisy_dl'
                            context=project(observation[key][None,:],state)
                            # Common posterior Monte Carlo numbers improve the
                            # resolution-sensitivity comparison. This does not
                            # change or correlate the simulated SO noise seeds.
                            torch.manual_seed(seed+100000)
                            # Explicitly bound rejection work for an out-of-family
                            # observation. No new physical prior cut is introduced.
                            samples,draws=bounded_samples(estimator,context,low,high,args.posterior_samples,args.max_proposals)
                            external[label]=dict(accepted=len(samples),proposals=draws,
                                posterior_sampling_seed=seed+100000,
                                mean=samples.mean(0).tolist() if len(samples) else None,
                                nearest_training_feature_distance=float(np.min(np.linalg.norm(x[pool]-context,axis=1))),
                                interpretation=('B12 clean-resolution perturbation sensitivity, noise residual fixed' if 'sensitivity' in label else
                                    'Known in-family B12 control' if label=='Battaglia12' else
                                    'Effective out-of-family posterior, not a FLAMINGO parameter truth'))
                            np.save(out/f'flamingo_{label}_{method}_{seed}.npy',samples)
                        result['flamingo']=external
                    results.append(result)
                    tag=f'{method}_{size}_{seed}_{int(shuffled)}'
                    np.savez_compressed(out/f'heldout_{tag}.npz',row_id=ids[heldout],truth=theta[heldout],
                        mean=means,ranks=ranks,width68=widths,
                        posterior=np.stack(posterior_arrays) if complete else np.empty((0,0,9)))
                    # Preserve the actual validation history, not the training loss.
                    summary=inference.summary
                    validation={k:np.asarray(v).tolist() for k,v in summary.items() if 'validation' in k or 'epoch' in k}
                    validation['best_validation_log_prob']=float(inference._best_val_log_prob)
                    (out/f'validation_{tag}.json').write_text(json.dumps(validation,indent=2)+'\n')
                    torch.save(estimator.state_dict(),out/f'estimator_{method}_{size}_{seed}_{int(shuffled)}.pt')
                    (out/'inference_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
                    print(size,method,shuffled,seed,result['seconds'],flush=True)


if __name__=='__main__':main()
