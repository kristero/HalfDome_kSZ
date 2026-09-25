"""Fisher and saved 8k SBI under one physical model and the same joint prior.

Stages are resumable but completed scientific results are never overwritten.
Fisher is the local Gaussian, frozen-covariance mean-information approximation.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

from calibration import FIDUCIAL, EDGES, digest, write_json

NAMES=['P0','xc','beta','alpha_m_P0','alpha_m_xc','alpha_m_beta',
       'alpha_z_P0','alpha_z_xc','alpha_z_beta']
LABELS=[r'P_0',r'x_c',r'\beta',r'\alpha_{m,P_0}',r'\alpha_{m,x_c}',
        r'\alpha_{m,\beta}',r'\alpha_{z,P_0}',r'\alpha_{z,x_c}',r'\alpha_{z,\beta}']
METHODS=('bins40','pca','moped')


def setup(args):
    config=json.loads((args.root/'manifest.json').read_text())
    dataset=Path(config['dataset'])
    sys.path.insert(0,str(dataset/'code'))
    sys.path.insert(0,str(args.common))
    from prior import JointPrior
    prior=JointPrior(config['prior'])
    assert prior.contains(FIDUCIAL)
    return config,dataset,prior


def combine(args):
    from so_nine_fisher import conditional_covariance,fisher_matrix,fisher_modes,moped_weights
    config,dataset,prior=setup(args)
    out=args.root/'comparison'
    out.mkdir(exist_ok=True)
    if (out/'calibration_complete.json').exists():
        raise ValueError('Completed calibration exists')
    derivatives={}
    ensemble=np.empty((config['noise_count'],40))
    fiducials=[]
    operators=[]
    hashes={}
    for i,task in enumerate(config['tasks']):
        folder=args.root/'tasks'/f'task{i:03d}'
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['manifest_sha256']==digest(args.root/'manifest.json')
        assert complete['spectra_sha256']==digest(folder/'spectra.npz')
        values=dict(np.load(folder/'spectra.npz'))
        np.testing.assert_array_equal(values['theta'],task['theta'])
        operators.append(complete['operator'])
        hashes[str(folder/'spectra.npz')]=complete['spectra_sha256']
        if task['kind']=='derivative':
            derivatives[task['parameter'],task['multiple']]=values['clean']
        else:
            fiducials.append(values['clean'])
            if task['kind']=='noise':
                ensemble[task['noise_indices']]=values['noisy']
            else:
                observation=values['noisy'][0]
    for key in ('mask_pixel_sha256','noise_table_sha256','stable_los_sha256'):
        assert len({o[key] for o in operators})==1,key
    all_hashes=[h for o in operators for h in o['noise_pixel_hashes']]
    assert len(set(all_hashes))==len(all_hashes)==2*(config['noise_count']+1)
    mean=fiducials[0]
    # A fresh calibration must reproduce the actual training operator, not
    # merely reproduce itself across its worker processes.
    archived=np.load(dataset/'observations/HalfDome.npz')
    np.testing.assert_array_equal(archived['bin_edges'],EDGES)
    reference=archived['masked_clean_dl40']
    fiducial_relative_error=float(np.max(np.abs(mean/reference-1)))
    if fiducial_relative_error>1e-7:
        raise RuntimeError('Fresh Battaglia12 spectrum differs from the frozen 8k observation')
    covariance,shrinkage=conditional_covariance(ensemble)
    chol=np.linalg.cholesky(covariance)
    repeat_error=max(np.linalg.norm(solve_triangular(chol,x-mean,lower=True)) for x in fiducials)
    assert repeat_error<1e-4,repeat_error
    steps=np.array(config['steps'])
    small=np.column_stack([(derivatives[j,1]-derivatives[j,-1])/(2*steps[j]) for j in range(9)])
    large=np.column_stack([(derivatives[j,2]-derivatives[j,-2])/(4*steps[j]) for j in range(9)])
    jacobian=(4*small-large)/3
    width=prior.high-prior.low
    fisher=fisher_matrix(jacobian*width,covariance)
    eigenvalues,vectors,resolved=fisher_modes(fisher)
    weights,singular,error=moped_weights(jacobian*width,covariance)
    derivative_error=np.linalg.norm(solve_triangular(chol,small-jacobian,lower=True),axis=0)
    derivative_error/=np.linalg.norm(solve_triangular(chol,jacobian,lower=True),axis=0)
    if derivative_error.max()>.01:
        raise RuntimeError('Derivative step convergence exceeds 1%; inspect before using the forecast')
    np.savez(out/'calibration.npz',mean=mean,observation=observation,ensemble=ensemble,
        covariance=covariance,jacobian=jacobian,derivative_small=small,derivative_large=large,
        fisher=fisher,eigenvalues=eigenvalues,eigenvectors=vectors,resolved=resolved,
        optimal_moped_weights=weights,low=prior.low,high=prior.high,edges=EDGES)
    sensitivity=[]
    reference=np.sqrt(np.diag(np.linalg.inv(fisher+12*np.eye(9))))
    for n in sorted({len(ensemble)//2,len(ensemble)}):
        subcov,shrink=conditional_covariance(ensemble[:n])
        subf=fisher_matrix(jacobian*width,subcov)
        ratio=np.sqrt(np.diag(np.linalg.inv(subf+12*np.eye(9))))/reference
        sensitivity.append(dict(n_noise=n,shrinkage=shrink,
            sigma_ratio_gaussian_prior_moment_diagnostic=ratio.tolist(),
            bin_sigma_ratio=(np.sqrt(np.diag(subcov)/np.diag(covariance))).tolist()))
    write_json(out/'calibration_complete.json',dict(calibration_sha256=digest(out/'calibration.npz'),
        noise_count=len(ensemble),oas_shrinkage=shrinkage,fiducial_repeat_noise_norm=repeat_error,
        archived_fiducial_max_relative_error=fiducial_relative_error,
        derivative_small_relative_error=derivative_error.tolist(),fisher_rank=int(resolved.sum()),
        optimal_moped_fisher_relative_error=error,noise_pixel_hashes_unique=True,
        independent_of_training_and_observation=True,source_sha256=hashes,
        covariance_sensitivity=sensitivity,
        covariance_derivative_term_computed=False,changing_sky_variance_included=False))
    print('Calibration combined; rank',int(resolved.sum()),'shrinkage',shrinkage,flush=True)


def joint_fisher_samples(jacobian,covariance,residual,prior,draws=2000000,count=50000,seed=42):
    """Importance integration of the linear Gaussian likelihood times joint prior.

    The proposal penalty is cancelled analytically. The actual support is the
    frozen JointPrior.contains function, including all coupled restrictions.
    """
    assert not prior.config['log_uniform_indices'],'This integrator requires the physical-uniform base'
    width=prior.high-prior.low
    chol=np.linalg.cholesky(covariance)
    a=solve_triangular(chol,jacobian*width,lower=True)
    y=solve_triangular(chol,residual,lower=True)
    middle=((prior.low+prior.high)/2-FIDUCIAL)/width
    precision=a.T@a+12*np.eye(9)
    pchol=np.linalg.cholesky(precision)
    center=np.linalg.solve(precision,a.T@y+12*middle)
    rng=np.random.default_rng(seed)
    accepted=[]
    for first in range(0,draws,25000):
        n=min(25000,draws-first)
        u=center+solve_triangular(pchol.T,rng.normal(size=(9,n)),lower=False).T
        theta=FIDUCIAL+u*width
        accepted.append(u[prior.contains(theta)])
    u=np.concatenate(accepted)
    if not len(u):
        raise RuntimeError('No accepted Fisher proposals in the joint prior')
    logw=6*np.sum((u-middle)**2,axis=1)
    weights=np.exp(logw-logw.max());weights/=weights.sum()
    ess=1/np.sum(weights**2)
    if ess<5000:
        raise RuntimeError(f'Insufficient importance ESS {ess:.0f}; increase draws')
    mean=weights@u
    std=np.sqrt(weights@(u-mean)**2)
    cdf=np.cumsum(weights);cdf[-1]=1.
    positions=(np.arange(count)+rng.random())/count
    samples=FIDUCIAL+u[np.searchsorted(cdf,positions)]*width
    return samples,dict(draws=draws,in_support=len(u),importance_ess=float(ess),
        weighted_mean=(FIDUCIAL+mean*width).tolist(),weighted_std=(std*width).tolist(),
        prior='Exact frozen 8k joint support with physical-uniform base')


def fisher(args):
    from so_fisher_compression import project_likelihood
    from so_sbi_compression import project
    config,dataset,prior=setup(args)
    out=args.root/'comparison'
    marker=json.loads((out/'calibration_complete.json').read_text())
    assert marker['calibration_sha256']==digest(out/'calibration.npz')
    cal=dict(np.load(out/'calibration.npz'))
    width=prior.high-prior.low
    j,c,mean=cal['jacobian'],cal['covariance'],cal['mean']
    forecast=np.zeros(40)
    projections={'full':np.eye(40),'optimal_moped':cal['optimal_moped_weights']}
    transforms={}
    metadata={}
    for method in METHODS:
        path=args.sbi/'features'/f'{method}_N7349_transform.pkl'
        with path.open('rb') as stream:
            transform=pickle.load(stream)
        transforms[method]=transform
        # Tangent of the ACTUAL saved nonlinear asinh compression at B12.
        # This is a local delta-method covariance, not an exact global Gaussian
        # model in compressed space. bins40 retains all coordinates invertibly.
        gradient=1/(np.sqrt(transform['scale']**2+mean**2)*transform['std'])
        matrix=gradient[:,None]*transform['matrix']/transform['output_std']
        projections[method]=matrix
        linear=(cal['ensemble']-mean)@matrix
        exact=project(cal['ensemble'],transform).astype(float)-project(mean,transform)
        metadata[method]=dict(transform_sha256=digest(path),
            tangent_rms_error_over_fluctuation=float(np.linalg.norm((exact-linear)-(exact-linear).mean(0))/
                np.linalg.norm(exact-exact.mean(0))))
        full=cal['fisher']
        compressed=project_likelihood(j*width,c,forecast,matrix)['fisher']
        vectors=cal['eigenvectors'][:,cal['resolved']]
        eigenvalues=cal['eigenvalues'][cal['resolved']]
        ratio=(vectors.T@compressed@vectors)/np.sqrt(np.outer(eigenvalues,eigenvalues))
        metadata[method]['resolved_mode_information_fractions']=np.linalg.eigvalsh((ratio+ratio.T)/2).tolist()
        metadata[method]['relative_fisher_change']=float(np.linalg.norm(compressed-full)/np.linalg.norm(full))
    for name,matrix in projections.items():
        for case,residual in (('forecast',forecast),('observed',cal['observation']-mean)):
            result=project_likelihood(j,c,residual,matrix)
            np.savez(out/f'likelihood_{case}_{name}.npz',**result)
            if name not in ('full','pca','moped'):
                continue
            destination=out/f'fisher_{case}_{name}.npz'
            if destination.exists():
                continue
            samples,info=joint_fisher_samples(result['derivatives'],result['covariance'],result['residual'],
                prior,draws=args.draws,seed=20260920+(case=='observed')+10*list(projections).index(name))
            _,repeat=joint_fisher_samples(result['derivatives'],result['covariance'],result['residual'],
                prior,draws=args.draws,count=1,seed=20261020+(case=='observed')+10*list(projections).index(name))
            shift=np.abs(np.array(info['weighted_mean'])-repeat['weighted_mean'])/info['weighted_std']
            change=np.abs(np.array(repeat['weighted_std'])/info['weighted_std']-1)
            if shift.max()>.05 or change.max()>.05:
                raise RuntimeError('Importance integration is not repeatable; increase draws')
            info.update(repeat_ess=repeat['importance_ess'],repeat_max_mean_shift_over_std=float(shift.max()),
                        repeat_max_std_fractional_change=float(change.max()))
            np.savez(destination,samples=samples,info_json=np.asarray(json.dumps(info)))
            print('Fisher',case,name,'ESS',round(info['importance_ess']),flush=True)
    if not (out/'prior_samples.npy').exists():
        np.save(out/'prior_samples.npy',prior.sample(30000,np.random.default_rng(20260920)))
    write_json(out/'compression_audit.json',metadata)


def infer(args):
    import torch
    from so_sbi_compression import project
    config,dataset,prior=setup(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out=args.root/'comparison'
    cal=dict(np.load(out/'calibration.npz'))
    records={}
    for method in METHODS:
        model_dir=args.sbi/method/'N7349_maf'
        training=json.loads((model_dir/'complete.json').read_text())
        assert training['N']==7349 and training['method']==method
        with (model_dir/'estimator.pkl').open('rb') as stream:
            model=pickle.load(stream).cpu().eval()
        with (args.sbi/'features'/f'{method}_N7349_transform.pkl').open('rb') as stream:
            transform=pickle.load(stream)
        for case,x in (('forecast',cal['mean']),('observed',cal['observation'])):
            path=out/f'sbi_{case}_{method}.npz'
            if path.exists():
                values=dict(np.load(path)); records[f'{case}/{method}']=json.loads(str(values['info_json']))
                continue
            context=project(x,transform)
            torch.manual_seed(20260920+list(METHODS).index(method)+(case=='observed')*10)
            pieces=[];proposals=0;accepted=0;started=time.monotonic()
            while accepted<10000:
                if proposals>=2000000 or time.monotonic()-started>600:
                    raise RuntimeError(f'SBI sampling incomplete: {case}/{method}, {accepted}/{proposals}')
                with torch.no_grad():
                    raw=model.sample(4096,context=torch.tensor(context,dtype=torch.float32).reshape(1,-1))
                values=raw.detach().cpu().numpy().reshape(-1,9)
                good=values[prior.contains(values)]
                pieces.append(good);accepted+=len(good);proposals+=len(values)
            samples=np.concatenate(pieces)[:10000]
            info=dict(method=method,case=case,training_rows=7349,total_dataset_rows=8192,
                accepted_count=10000,acceptance=accepted/proposals,proposals=proposals,
                model_sha256=digest(model_dir/'estimator.pkl'),training=training,
                calibration_sha256=digest(out/'calibration.npz'),prior_support='exact frozen joint prior')
            np.savez(path,samples=samples,context=context,observation=x,truth=FIDUCIAL,
                     info_json=np.asarray(json.dumps(info)))
            records[f'{case}/{method}']=info
            print('SBI',case,method,'acceptance',round(accepted/proposals,3),flush=True)
    write_json(out/'sbi_sampling.json',records)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from getdist import MCSamples,plots
    config,dataset,prior=setup(args)
    out=args.root/'comparison'
    if (out/'complete.json').exists():
        raise ValueError('Completed result exists; preserve it')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':15})
    cal=dict(np.load(out/'calibration.npz'))
    prior_draws=np.load(out/'prior_samples.npy')
    ranges={n:[prior.low[j],prior.high[j]] for j,n in enumerate(NAMES)}
    prior_root=MCSamples(samples=prior_draws,names=NAMES,labels=LABELS,ranges=ranges)
    colors={'full':'#222222','bins40':'#0072B2','pca':'#CC79A7','moped':'#D55E00'}
    labels={'full':'Fisher + joint prior','bins40':'40-bin SBI','pca':'PCA SBI','moped':'MOPED SBI'}
    rows=[];summary={}
    for case in ('forecast','observed'):
        arrays={'full':np.load(out/f'fisher_{case}_full.npz')['samples']}
        arrays.update({m:np.load(out/f'sbi_{case}_{m}.npz')['samples'] for m in METHODS})
        for method in ('pca','moped'):
            arrays['fisher_'+method]=np.load(out/f'fisher_{case}_{method}.npz')['samples']
        summary[case]={}
        for name,samples in arrays.items():
            assert np.isfinite(samples).all() and prior.contains(samples).all()
            quantiles=np.quantile(samples,[.025,.16,.5,.84,.975],axis=0)
            std=samples.std(0,ddof=1)
            summary[case][name]=dict(mean=samples.mean(0).tolist(),std=std.tolist())
            for j,param in enumerate(NAMES):
                rows.append(dict(case=case,method=name,parameter=param,truth=FIDUCIAL[j],mean=samples[:,j].mean(),
                    std=std[j],q025=quantiles[0,j],q16=quantiles[1,j],median=quantiles[2,j],q84=quantiles[3,j],q975=quantiles[4,j]))
        comparisons=[('all',['full']+list(METHODS))]
        if case=='forecast':
            comparisons += [(m,['fisher_'+m,m]) for m in ('pca','moped')]
        for tag,methods in comparisons:
            roots=[MCSamples(samples=arrays[m][::max(1,len(arrays[m])//20000)],names=NAMES,labels=LABELS,ranges=ranges,
                settings={'smooth_scale_1D':.35,'smooth_scale_2D':.45,'fine_bins_2D':128}) for m in methods]
            g=plots.get_subplot_plotter(width_inch=19)
            g.settings.scaling=False;g.settings.axes_fontsize=13;g.settings.lab_fontsize=17
            g.settings.legend_fontsize=16;g.settings.figure_legend_frame=False
            method_colors=[colors[m.replace('fisher_','')] for m in methods]
            method_labels=[labels.get(m,m.replace('fisher_','').upper()+' Fisher + prior') for m in methods]
            g.triangle_plot(roots,filled=[m in METHODS for m in methods],contour_colors=method_colors,
                legend_labels=method_labels,markers=FIDUCIAL,
                contour_ls=['--' if m=='full' or m.startswith('fisher_') else '-' for m in methods],
                contour_lws=[2.0 if m=='full' or m.startswith('fisher_') else 1.3 for m in methods],
                contour_args=[dict(zorder=20) if m=='full' or m.startswith('fisher_') else dict(alpha=.35)
                              for m in methods],
                line_args=[dict(color=c,ls='--' if m=='full' or m.startswith('fisher_') else '-') for c,m in zip(method_colors,methods)],
                marker_args={'color':'black','ls':':','lw':.9})
            for j,name in enumerate(NAMES):
                density=prior_root.get1DDensity(name)
                g.subplots[j,j].plot(density.x,density.P/density.P.max(),color='.55',ls=':',lw=1.2)
            title=('Synthetic software test' if config.get('synthetic_test',False) else
                   'Battaglia12 · '+('mean-spectrum forecast' if case=='forecast' else 'independent-noise observation'))
            g.fig.suptitle(title,y=1.01,fontsize=21)
            for ext in ('png','pdf'):
                g.fig.savefig(out/f'battaglia12_{case}_{tag}.{ext}',dpi=180,bbox_inches='tight')
            plt.close(g.fig)
        fig,axes=plt.subplots(3,3,figsize=(15,11))
        for j,ax in enumerate(axes.flat):
            for i,m in enumerate(['full']+list(METHODS)):
                q=np.quantile(arrays[m][:,j],[.025,.16,.5,.84,.975])
                ax.plot(q[[0,4]],[i,i],color=colors[m],lw=1.8)
                ax.plot(q[[1,3]],[i,i],color=colors[m],lw=5)
                ax.plot(q[2],i,'o',color=colors[m],ms=6)
            ax.axvline(FIDUCIAL[j],color='black',ls=':',lw=1.2)
            ax.set(yticks=range(4),yticklabels=[labels[m] for m in ['full']+list(METHODS)] if j%3==0 else ['']*4,
                   xlabel='$'+LABELS[j]+'$',ylim=(-.5,3.5))
            ax.tick_params(labelsize=12);ax.grid(axis='x',alpha=.15)
        fig.suptitle('Synthetic software test' if config.get('synthetic_test',False) else 'Battaglia12 · '+case,fontsize=21)
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(out/f'battaglia12_{case}_intervals.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    with (out/'constraints.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'constraint_summary.json',summary)
    write_json(out/'complete.json',dict(complete=True,synthetic_test=config.get('synthetic_test',False),
        physical_dataset='synthetic software test' if config.get('synthetic_test',False) else '8192 independent-noise rows',
        same_joint_prior=True,same_observable=True,same_observation_per_comparison=True,
        forecast_observation='fiducial clean spectrum; expected-data (Asimov) forecast, not noise-averaged posterior',
        fisher_scope='linear mean, frozen covariance, no covariance-derivative or changing-sky term',
        compression_fisher_scope='local tangent of the actual saved asinh compression',
        analysis_source_sha256={str(p):digest(p) for p in sorted(set(
            list(Path(__file__).parent.glob('*.py'))+list(args.common.glob('*.py'))))},
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('combine','fisher','infer','plot'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sbi',type=Path,default=Path('/lustre/work/kristero10/sbi_linear_8k_20260918/analysis'))
    p.add_argument('--common',type=Path,required=True)
    p.add_argument('--draws',type=int,default=2000000)
    p.add_argument('--threads',type=int,default=26)
    args=p.parse_args()
    sys.path.insert(0,str(args.common))
    setup(args)
    globals()[args.stage](args)


if __name__=='__main__':
    main()
