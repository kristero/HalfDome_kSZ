#!/usr/bin/env python3
"""Check the fingerprint result per seed and factor mask changes into three terms.

For masks, the noise-noise spectrum changes as well as the signal-noise term.
These diagnostic perturbations are not substituted for physical observations.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MERLIN_PARAM_SPACE', 'linear')
import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter1d
import diagnose as d


def make_controls(clean, noisy):
    spectra = []
    cases = []
    def add(label, spectrum, family, seed=None):
        cases.append(dict(label=label, family=family, seed=seed))
        spectra.append(spectrum)
    base = noisy[0]
    add('baseline', base, 'baseline')
    for i, seed in enumerate(d.SEEDS[1:], 1):
        delta = clean[10*i] - clean[0]
        smooth = clean[0] * gaussian_filter1d(delta/clean[0], 5, mode='nearest')
        add(f'smooth_seed_{seed}', base+smooth, 'smooth_simulation', int(seed))
        add(f'fine_seed_{seed}', base+delta-smooth, 'fine_simulation', int(seed))
    mean = clean[10::10].mean(0) - clean[0]
    # Absolute smoothing provides an independent version of the ratio control.
    add('mean_absolute_smooth5', base+gaussian_filter1d(mean, 5, mode='nearest'), 'smooth_absolute')
    add('mean_absolute_smooth50', base+gaussian_filter1d(mean, 50, mode='nearest'), 'smooth_absolute')
    for sign in [1, -1]:
        smooth = clean[0]*gaussian_filter1d(mean/clean[0], 50, mode='nearest')
        add(f'mean_fine_sign{sign}', base+sign*(mean-smooth), 'fine_sign')
    # A high-ell amplitude-only change cannot encode catalogue phases.
    add('amplitude_plus_0p742percent', base+clean[0]*.007422065679770441, 'amplitude')
    beam=np.load(d.PARENT/'results/seed_100/beam_bl.npy')[80:7980]
    nbase=np.load(d.PARENT/'results/seed_100/mask_12345/noise_cl.npy')[80:7980]*d.DL*beam**2
    mask_terms=[]
    for i in range(1,10):
        mask=12345+i
        nn=np.load(d.PARENT/f'results/seed_100/mask_{mask}/noise_cl.npy')[80:7980]*d.DL*beam**2
        delta_clean=clean[i]-clean[0]
        delta_nn=nn-nbase
        delta_cross=noisy[i]-base-delta_clean-delta_nn
        assert np.allclose(base+delta_clean+delta_nn+delta_cross,noisy[i],rtol=1e-12,atol=1e-25)
        smooth=clean[0]*gaussian_filter1d(delta_clean/clean[0],5,mode='nearest')
        for family,delta in [('actual',noisy[i]-base),('clean',delta_clean),('nn',delta_nn),
                             ('cross',delta_cross),('smooth',smooth),('fine',delta_clean-smooth)]:
            add(f'mask_{family}_{mask}',base+delta,f'mask_{family}',mask)
        mask_terms.append((delta_clean,delta_nn,delta_cross))
    mask_mean=np.mean(mask_terms,axis=0)
    for name,delta in zip(['clean','nn','cross'],mask_mean):
        add(f'mask_mean_{name}',base+delta,'mask_mean')
    add('mask_mean_actual',base+mask_mean.sum(0),'mask_mean')
    # The joint-only arm consists of non-training simulations and non-training masks.
    ids=np.array([10*s+m for s in range(1,10) for m in range(1,10)])
    joint_clean=clean[ids].mean(0)-clean[0]
    joint_noisy=noisy[ids].mean(0)-base
    joint_smooth=clean[0]*gaussian_filter1d(joint_clean/clean[0],5,mode='nearest')
    for name,delta in [('actual',joint_noisy),('clean',joint_clean),
                       ('residual',joint_noisy-joint_clean),('smooth',joint_smooth),
                       ('fine',joint_clean-joint_smooth)]:
        add(f'joint_mean_{name}',base+delta,'joint_mean')
    # Append these to preserve the ordering of the already completed controls.
    mask_smooth=clean[0]*gaussian_filter1d(mask_mean[0]/clean[0],5,mode='nearest')
    add('mask_mean_smooth',base+mask_smooth,'mask_mean')
    add('mask_mean_fine',base+mask_mean[0]-mask_smooth,'mask_mean')
    return np.asarray(spectra),cases


def evaluate(folder, x, root, label, threads, append=False):
    import torch
    import merlin_compression as mc
    import train_ensemble as te
    ens=mc.ensemble(folder,'merlin',threads)
    torch.manual_seed(20261007)
    u=ens.sample_u(x,9000,rng=np.random.default_rng(20261007))
    theta=te.from_u(u,ens.low,ens.high)
    result=d.summarize(theta,ens.low,ens.high)
    truth=te.to_u(d.TRUTH[None],ens.low,ens.high)
    lp=ens.log_prob_u(u,x)-te.log_jacobian_theta_per_u(u,ens.low,ens.high)
    lpt=ens.log_prob_u(np.repeat(truth[None],len(x),axis=1),x)[0]
    lpt-=te.log_jacobian_theta_per_u(truth,ens.low,ens.high)[0]
    result.update(log_prob_truth=lpt,truth_hpd_theta=np.mean(lp>lpt[None],axis=0),samples=len(u))
    if append:
        import json
        previous=json.loads((root/f'extended_{label}.json').read_text())
        for key in result:
            if key != 'samples':
                result[key]=np.concatenate([np.asarray(previous[key]),result[key]],axis=0)
    d.write_json(root/f'extended_{label}.json',result)
    print('EXTENDED',label,flush=True)


def posterior_correlations(root):
    """Use joint draws; a vector of marginal medians need not be a plausible profile."""
    samples=np.load(d.PARENT/'candidates/original_merlin/posterior_final.npy',mmap_mode='r')
    results={}
    for i,seed in enumerate(d.SEEDS):
        draws=samples[::9,10*i,:]
        corr=np.corrcoef(draws,rowvar=False)
        results[str(seed)]=dict(correlation=corr,draws=len(draws))
    d.write_json(root/'posterior_correlations.json',results)


def plot(root):
    import json
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16,'axes.labelsize':18,'axes.titlesize':18,
                         'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':12})
    cases=json.loads((root/'extended_cases.json').read_text())
    index={c['label']:i for i,c in enumerate(cases)}
    r=json.loads((root/'extended_original_linear.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for j,ax in zip([0,2,3],axes):
        for prefix,label,color in [('smooth_seed','Smooth change','C2'),('fine_seed','Fine change','C3')]:
            ids=[index[f'{prefix}_{s}'] for s in d.SEEDS[1:]]
            med=np.asarray(r['median'])[ids,j]
            lo=np.asarray(r['q16'])[ids,j];hi=np.asarray(r['q84'])[ids,j]
            ax.errorbar(d.SEEDS[1:],med,yerr=[med-lo,hi-med],fmt='o-',color=color,label=label,markersize=4,alpha=.8)
        ax.axhline(d.TRUTH[j],color='k',ls='--',label='Truth')
        ax.axhline(r['median'][0][j],color='0.4',ls=':',label='Seed 100')
        ax.set_xlabel('Simulation seed');ax.set_ylabel(d.LABELS[j])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.05),ncol=4,frameon=False,fontsize=13)
    for ext in ['png','pdf']:fig.savefig(root/f'plots/fingerprint_per_seed.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for j,ax in zip([0,2,3],axes):
        for family,label,color in [('actual','Actual mask','C0'),('clean','Clean only','C1'),
                                   ('nn','Noise–noise','C2'),('cross','Signal–noise','C3')]:
            ids=[index[f'mask_{family}_{m}'] for m in range(12346,12355)]
            ax.plot(np.arange(1,10),np.asarray(r['median'])[ids,j],'o-',color=color,label=label,markersize=4)
        ax.axhline(d.TRUTH[j],color='k',ls='--');ax.axhline(r['median'][0][j],color='0.4',ls=':')
        ax.set_xlabel('Mask index');ax.set_ylabel(d.LABELS[j]);ax.set_xticks([1,5,9])
    axes[0].legend(fontsize=11)
    for ext in ['png','pdf']:fig.savefig(root/f'plots/mask_term_controls.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/lustre/work/kristero10/tsz_seed_offset_diagnosis_20261005'))
    parser.add_argument('--threads',type=int,default=26)
    parser.add_argument('--bounded',action='store_true')
    parser.add_argument('--append-mask-means',action='store_true',
                        help='Evaluate only two newly appended mean-mask controls; preserve existing draws.')
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    sys.path.insert(0,str(d.PARENT/'snapshot/inference'))
    if args.bounded:
        data=np.load(args.root/'arrays/extended_inputs.npz')
        original=data['original'][-2:] if args.append_mask_means else data['original']
        banded=data['banded'][-2:] if args.append_mask_means else data['banded']
        evaluate(d.OLD/'models/logit',original,args.root,'original_bounded',args.threads,args.append_mask_means)
        evaluate(d.PARENT/'candidates/merlin_band50_full/bounded',banded,args.root,'band50_bounded',args.threads,args.append_mask_means)
    else:
        saved=np.load(args.root/'arrays/spectra.npz')
        spectra,cases=make_controls(saved['clean'],saved['noisy'])
        if args.append_mask_means:
            import json
            previous=json.loads((args.root/'extended_cases.json').read_text())
            if cases==previous:
                print('Mean-mask controls already complete; preserving existing posterior draws.')
                sys.exit(0)
            assert cases[:-2]==previous
        d.write_json(args.root/'extended_cases.json',cases)
        w=np.load(d.OLD/'reference/whitening_pca.npz')
        p=np.load(d.OLD/'models/linear/features_merlin.npz')
        original=d.transform(solve_triangular(w['L'],spectra.T,lower=True).T@w['V'],p)
        a=np.load(d.PARENT/'data/band_operator.npy')
        comp=np.load(d.PARENT/'candidates/merlin_band50_full/compression.npz')
        banded=d.transform(((spectra@a.T)/comp['fid'])@comp['weights'],comp)
        np.savez(args.root/'arrays/extended_inputs.npz',original=original,banded=banded)
        if args.append_mask_means:
            evaluate(d.OLD/'models/linear',original[-2:],args.root,'original_linear',args.threads,True)
            evaluate(d.PARENT/'candidates/merlin_band50_full',banded[-2:],args.root,'band50_linear',args.threads,True)
        else:
            evaluate(d.OLD/'models/linear',original,args.root,'original_linear',args.threads)
            evaluate(d.PARENT/'candidates/merlin_band50_full',banded,args.root,'band50_linear',args.threads)
            posterior_correlations(args.root)
            plot(args.root)
        env=os.environ.copy();env['MERLIN_PARAM_SPACE']='logit'
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(args.root),
                 '--threads',str(args.threads),'--bounded']
        if args.append_mask_means:command.append('--append-mask-means')
        subprocess.run(command,env=env,check=True)
        outputs=[p for p in args.root.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.npy','.png','.pdf']
                 and 'logs' not in p.parts and p.name not in ['COMPLETE.json','EXTENDED_COMPLETE.json']]
        d.write_json(args.root/'EXTENDED_COMPLETE.json',dict(cases=len(cases),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     sha256={str(p.relative_to(args.root)):d.digest(p) for p in outputs}))
