#!/usr/bin/env python3
"""Matched-compression SBI training and evaluation, running in one checkpointed PBS job."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import solve_triangular

from common import (ROOT, OLD, BEAMFIX, FISHER, ELL, DL, EDGES, THETA, SEEDS, MASKS,
                    band_operator, noise_covariance, moped_weights, safe_cholesky,
                    fit_transform, transform, add_sky_variation, arm_indices,
                    sha, write_json)


def band_training_data(root, plan):
    """Read each uploaded catalogue-spectrum batch once; keep only 158 bands on disk."""
    out = root/'data'
    if (out/'training_complete.json').exists():
        return
    a = band_operator()
    row_ids = np.load(BEAMFIX/'inputs/row_id.npy')
    positions = {int(row):i for i,row in enumerate(row_ids)}
    n = len(row_ids)
    clean = np.full((n,158), np.nan)
    noisy = np.full_like(clean, np.nan)
    beam = np.load(OLD/'reference/beam_bl.npy')[ELL]
    cached = {}
    for stage in sorted((BEAMFIX/'stages').glob('stage*')):
        if (stage/'beam_dl_unbinned.npy').exists():
            cached[stage.name] = (dict(zip(np.load(stage/'rows.npy').tolist(),
                                          range(len(np.load(stage/'rows.npy'))))),
                                   np.load(stage/'beam_dl_unbinned.npy',mmap_mode='r'))
    audit_positions=set(np.random.default_rng(5704).choice(n,size=32,replace=False).tolist())
    for lookup,_ in cached.values():
        audit_positions.update(positions[row] for row in list(lookup)[:4] if row in positions)
    audit_full={}
    rebuilt = 0
    for batch_name in plan['clean_batches']:
        batch=Path(batch_name); rows=np.load(batch/'rows.npy')
        c=np.load(batch/'clean_dl_unbinned.npy')
        assert c.shape==(len(rows),7900)
        eligible=[i for i,row in enumerate(rows) if int(row) in positions]
        if not eligible:
            continue
        target=np.array([positions[int(rows[i])] for i in eligible])
        clean[target]=c[eligible]@a.T
        cache=cached.get(batch.parent.name)
        beam_rows=np.empty((len(eligible),7900))
        for j,i in enumerate(eligible):
            row=int(rows[i])
            if cache is not None and row in cache[0]:
                beam_rows[j]=cache[1][cache[0][row]]
            else:
                production=np.load(batch/'noisy'/f'{row:05d}.npy')[ELL]*DL
                nn=np.load(BEAMFIX/'nn'/f'r{row:05d}.npy')[ELL]*DL
                beam_rows[j]=c[i]+beam*(production-c[i]-nn)+beam**2*nn
                rebuilt+=1
            if target[j] in audit_positions:
                audit_full[int(target[j])]=beam_rows[j].copy()
        noisy[target]=beam_rows@a.T
        print('DATA',batch.parent.name,batch.name,'rows',len(target),flush=True)
    assert np.isfinite(clean).all() and np.isfinite(noisy).all(), 'Missing training spectra'
    with np.load(OLD/'models/linear/features_merlin.npz') as f:
        theta=np.load(BEAMFIX/'inputs/theta.npy')
        pool,held=f['pool'],f['held']
        np.testing.assert_array_equal(theta[pool],f['theta_pool'])
        np.testing.assert_array_equal(theta[held],f['theta_held'])
        np.savez(out/'design.npz',theta=theta,pool=pool,held=held,low=f['low'],high=f['high'])
        expected_by_position=dict(zip(pool.tolist(),f['x_pool']))
        expected_by_position.update(zip(held.tolist(),f['x_held']))
        ordered=sorted(audit_positions)
        assert set(ordered)==set(audit_full), 'Training-feature audit missed source rows'
        with np.load(OLD/'reference/whitening_pca.npz') as w:
            s=solve_triangular(w['L'],np.array([audit_full[i] for i in ordered]).T,lower=True).T@w['V']
        reproduced=transform(s,{key:f[key] for key in ['s_center','s_scale','z_mean','z_sd']})
        feature_error=float(np.max(np.abs(reproduced-np.array([expected_by_position[i] for i in ordered]))))
        assert feature_error<1e-4, f'Training inputs differ from saved Merlin: {feature_error}'
    np.save(out/'clean_bands.npy',clean);np.save(out/'noisy_bands.npy',noisy)
    np.save(out/'band_operator.npy',a)
    write_json(out/'training_complete.json',dict(rows=n,bands=158,rebuilt_beam_rows=rebuilt,
        saved_merlin_feature_audit_rows=len(audit_positions),saved_merlin_feature_max_error=feature_error,
        sha256={name:sha(out/name) for name in ['clean_bands.npy','noisy_bands.npy','design.npz']}))


def grid_data(root, plan):
    import tomllib
    a=band_operator()
    clean,noisy,full=[],[],[]
    reference=tomllib.loads((OLD/'results/seed_100/complete.toml').read_text())
    for seed in SEEDS:
        folder=root/f'results/seed_{seed}'
        assert (folder/'VALIDATED.json').exists(),folder
        meta=tomllib.loads((folder/'complete.toml').read_text())
        assert meta['noise_sha256']==reference['noise_sha256']
    for case in plan['cases']:
        folder=root/case['path'];meta=tomllib.loads((folder/'complete.toml').read_text())
        for name,expected in meta['spectrum_sha256'].items():
            assert sha(folder/name)==expected,folder/name
        c=np.load(folder/'clean_cl.npy')[ELL]*DL
        x=np.load(folder/'beam_corrected_cl.npy')[ELL]*DL
        clean.append(c@a.T);noisy.append(x@a.T);full.append(x)
    clean,noisy,full=map(np.array,(clean,noisy,full))
    np.savez(root/'data/grid.npz',clean=clean,noisy=noisy,full=full)
    return clean,noisy,full


def build_features(root, plan):
    if (root/'data/features_complete.json').exists():
        return
    band_training_data(root,plan)
    clean_grid,noisy_grid,full_grid=grid_data(root,plan)
    a=band_operator()
    f=np.load(FISHER)
    fid=a@f['d0']
    old_fid=np.load(OLD/'results/seed_100/mask_12345/clean_cl.npy')[ELL]*DL
    np.testing.assert_allclose(f['d0'],old_fid,rtol=1e-6,atol=1e-25)
    assert np.all(fid>0)
    cov=noise_covariance(f['sd'],f['rho'],a)/np.outer(fid,fid)
    derivatives=(a@f['J'])/fid[:,None]
    with np.load(OLD/'reference/whitening_pca.npz') as w:
        projected=a@w['L']
        nn_cov=(projected@projected.T)/np.outer(fid,fid)
    design=np.load(root/'data/design.npz')
    pool,held=design['pool'],design['held']
    clean=np.load(root/'data/clean_bands.npy')/fid
    noisy=np.load(root/'data/noisy_bands.npy')/fid
    discovery=[i for i,c in enumerate(plan['cases']) if c['simulation_seed'] in plan['discovery_seeds']
               and c['mask_seed'] in plan['discovery_masks']]
    ratios=clean_grid[discovery]/clean_grid[0]
    covariances={}
    for arm in ['simulation','mask','combined']:
        if arm=='simulation':
            ids=[i for i,c in enumerate(plan['cases']) if c['simulation_seed'] in plan['discovery_seeds']
                 and c['mask_seed']==MASKS[0]]
        elif arm=='mask':
            ids=[i for i,c in enumerate(plan['cases']) if c['mask_seed'] in plan['discovery_masks']
                 and c['simulation_seed']==SEEDS[0]]
        else:ids=discovery
        residual=(clean_grid[ids]-clean_grid[0])/fid
        covariances[arm]=np.cov(residual,rowvar=False,ddof=1)
    np.savez(root/'data/sky_discovery.npz',ratios=ratios,indices=discovery,**covariances)
    np.savez(root/'data/local_model.npz',fid=fid,J=derivatives,C=cov,C_nn=nn_cov)
    centers=a@ELL
    checks={}
    for recipe in plan['candidates']:
        name=recipe['name'];folder=root/'candidates'/name;folder.mkdir(exist_ok=True)
        keep=(EDGES[:-1]>=recipe['lmin'])&(EDGES[1:]<=recipe['lmax'])
        # A requested boundary is rounded inward to a complete band and documented.
        keep_ids=np.flatnonzero(keep)
        c=(nn_cov if recipe['covariance']=='noiseonly' else cov).copy()
        if recipe.get('nuisance'):
            c+=recipe['strength']*covariances[recipe['nuisance']]
        if recipe.get('taper'):
            scale=np.sqrt(1+9*np.clip((centers-4000)/3980,0,1)**2)
            c*=np.outer(scale,scale)
        c=c[np.ix_(keep,keep)]
        if recipe['network']=='merlin':
            # Keep the original noise-only whitening convention for the Merlin band control.
            l,jitter=safe_cholesky(nn_cov[np.ix_(keep,keep)])
            partial=solve_triangular(l.T,np.eye(len(l)),lower=False)
            audit=dict(jitter=jitter,directions=len(l),compression='whitened bands, learned Merlin head')
        else:
            partial,audit=moped_weights(derivatives[keep],c,design['high']-design['low'])
        weights=np.zeros((158,partial.shape[1]));weights[keep]=partial
        training_noisy=noisy
        if recipe.get('augmentation'):
            choice=np.random.default_rng(20261004).integers(len(ratios),size=len(noisy))
            training_noisy=add_sky_variation(clean,noisy,ratios,choice)
        projected_clean=clean@weights;projected_noisy=training_noisy@weights
        params=fit_transform(projected_noisy,projected_clean,pool)
        x=transform(projected_noisy,params)
        obs=transform((noisy_grid/fid)@weights,params)
        fixed_test=transform(noisy[held]@weights,params)
        np.savez(folder/f"features_{recipe['network']}.npz",x_pool=x[pool],x_held=x[held],
                 x_obs=obs,x_draws=obs[:1],theta_pool=design['theta'][pool],
                 theta_held=design['theta'][held],pool=pool,held=held,low=design['low'],high=design['high'],
                 s_mean=np.zeros(x.shape[1]),s_sd=np.ones(x.shape[1]),input_transform=np.array('asinh'),**params)
        np.save(folder/'fixed_sky_test.npy',fixed_test)
        np.savez(folder/'compression.npz',weights=weights,fid=fid,keep=keep,**params)
        effective=dict(recipe,n_input_bands=len(keep_ids),summary_dimension=weights.shape[1],
                       effective_lmin=int(EDGES[keep_ids[0]]),effective_lmax=int(EDGES[keep_ids[-1]+1]-1),
                       scale_fraction_retained=float(50*len(keep_ids)/7900),audit=audit)
        write_json(folder/'prepare.json',effective)
        checks[name]=audit
        print('FEATURES',name,weights.shape,'retained ell',effective['effective_lmin'],effective['effective_lmax'],flush=True)
    # Verify exact saved-network input reconstruction for the original baseline.
    with np.load(OLD/'reference/whitening_pca.npz') as w:
        s=solve_triangular(w['L'],full_grid.T,lower=True).T@w['V']
    with np.load(OLD/'models/linear/features_merlin.npz') as features:
        old_params={key:features[key] for key in ['s_center','s_scale','z_mean','z_sd']}
        obs=transform(s,old_params)
        error=float(np.max(np.abs(obs[:1]-features['x_obs'][:1])))
        assert error<1e-4,error
    np.save(root/'data/original_merlin_observations.npy',obs)
    write_json(root/'data/features_complete.json',dict(methods=len(checks),checks=checks,
               original_baseline_feature_error=error,discovery_cases=len(discovery),
               held_out_joint_cases=len(arm_indices(plan['cases'],'combined',True))))


def train_one(root_string,name,variant,seed,threads,param_space='linear',bounded=False):
    os.environ.update(MERLIN_PARAM_SPACE=param_space,MERLIN_PATIENCE='30',MERLIN_LR_PATIENCE='10',
                      OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
    os.environ.pop('MERLIN_TRAIN_ROWS',None)
    from threadpoolctl import threadpool_limits
    # Spawn imports numpy before entering this function; env alone cannot resize
    # an already initialized BLAS pool in the worker process.
    threadpool_limits(limits=threads)
    root=Path(root_string);sys.path.insert(0,str(root/'snapshot/inference'))
    import merlin_compression as mc
    folder=root/'candidates'/name
    if bounded:
        folder=folder/'bounded'
    return mc.train_one(folder,variant,seed,threads)


def train(root,plan):
    jobs=[]
    for recipe in plan['candidates']:
        folder=root/'candidates'/recipe['name'];variant=recipe['network']
        for seed in plan['training_seeds']:
            if not all((folder/f'{variant}_seed{seed}.{ext}').exists() for ext in ['json','pt']):
                jobs.append((str(root),recipe['name'],variant,seed,4))
    # Six independent processes x four application/BLAS threads = 24 requested CPUs.
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=context) as pool:
        futures=[pool.submit(train_one,*job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):print('TRAIN',future.result(),flush=True)


def posterior_metrics(ens,x,truth,samples,seed=2):
    import torch
    from train_ensemble import from_u,to_u,log_jacobian_theta_per_u
    torch.manual_seed(seed)
    u=ens.sample_u(x,samples,rng=np.random.default_rng(seed))
    theta=from_u(u,ens.low,ens.high)
    assert np.isfinite(theta).all()
    q=np.quantile(theta,[.025,.16,.5,.84,.975],axis=0)
    truth=np.broadcast_to(truth,q[2].shape)
    ut=to_u(truth,ens.low,ens.high)
    lp=ens.log_prob_u(ut[None],x)[0]-log_jacobian_theta_per_u(ut,ens.low,ens.high)
    result=dict(q025=q[0].tolist(),q16=q[1].tolist(),median=q[2].tolist(),q84=q[3].tolist(),
                q975=q[4].tolist(),width68=(q[3]-q[1]).tolist(),mean=theta.mean(0).tolist(),
                truth_in68=((truth>=q[1])&(truth<=q[3])).tolist(),
                truth_in95=((truth>=q[0])&(truth<=q[4])).tolist(),
                log_prob_truth=lp.tolist(),samples=int(len(theta)),
                outside_prior_per_parameter=np.mean((theta<ens.low)|(theta>ens.high),axis=0).tolist())
    # Independent Monte Carlo blocks quantify sample precision, not sky/model uncertainty.
    if len(theta)>=3000:
        blocks=np.array_split(theta,10)
        block_q=np.array([np.quantile(b,[.16,.5,.84],axis=0) for b in blocks])
        result['median_mcse']=(block_q[:,1].std(0,ddof=1)/np.sqrt(10)).tolist()
        result['width68_mcse']=((block_q[:,2]-block_q[:,0]).std(0,ddof=1)/np.sqrt(10)).tolist()
    return result,theta.astype(np.float32)


def evaluate_one(root,plan,name,variant,folder,x,threads,final=False):
    import merlin_compression as mc
    output=root/'candidates'/name;output.mkdir(exist_ok=True)
    target=output/('evaluation_final.json' if final else 'evaluation.json')
    if target.exists():return
    ens=mc.ensemble(folder,variant,threads)
    count=plan['posterior_samples_final'] if final else plan['posterior_samples_screen']
    metrics,samples=posterior_metrics(ens,x,THETA,count)
    np.save(output/('posterior_final.npy' if final else 'posterior_screen.npy'),samples)
    del samples
    if not final:
        selected=np.arange(len(ens.f['x_held']))[::2][:plan['prior_test_rows']]
        prior,prior_samples=posterior_metrics(ens,ens.f['x_held'][selected],ens.f['theta_held'][selected],
                                  plan['prior_test_samples'],seed=13)
        metrics['prior_validation']=dict(rows=len(selected),
           marginal_coverage68=np.mean(prior['truth_in68'],axis=0).tolist(),
           marginal_coverage95=np.mean(prior['truth_in95'],axis=0).tolist(),
           mean_log_prob_truth=float(np.mean(prior['log_prob_truth'])),
           mean_width_over_prior=(np.mean(prior['width68'],axis=0)/(ens.high-ens.low)).tolist())
        # Random-reference distance ranks (TARP): a joint posterior diagnostic over held-out prior rows.
        rng=np.random.default_rng(4701)
        normalized=(prior_samples-ens.low)/(ens.high-ens.low)
        normalized_truth=(ens.f['theta_held'][selected]-ens.low)/(ens.high-ens.low)
        references=rng.uniform(0,1,normalized_truth.shape)
        sample_distances=np.linalg.norm(normalized-references,axis=-1)
        true_distances=np.linalg.norm(normalized_truth-references,axis=-1)
        ranks=np.mean(sample_distances<true_distances,axis=0)
        metrics['prior_validation']['random_reference_rank68']=float(np.mean(ranks<=.68))
        metrics['prior_validation']['random_reference_rank95']=float(np.mean(ranks<=.95))
        np.save(output/'prior_distance_ranks.npy',ranks)
        del prior_samples
        fixed=output/'fixed_sky_test.npy'
        if fixed.exists() and json.loads((output/'prepare.json').read_text()).get('augmentation'):
            test,_=posterior_metrics(ens,np.load(fixed)[selected],ens.f['theta_held'][selected],
                                    plan['prior_test_samples'],seed=13)
            metrics['fixed_sky_prior_validation']=dict(
                marginal_coverage68=np.mean(test['truth_in68'],axis=0).tolist(),
                marginal_coverage95=np.mean(test['truth_in95'],axis=0).tolist(),
                mean_log_prob_truth=float(np.mean(test['log_prob_truth'])))
    metrics.update(name=name,network=variant,case_labels=[c['label'] for c in plan['cases']])
    write_json(target,metrics)
    print('EVALUATED',name,'samples',count,flush=True)


def evaluate(root,plan,threads,final_names=None):
    os.environ['MERLIN_PARAM_SPACE']='linear';os.environ.pop('MERLIN_TRAIN_ROWS',None)
    sys.path.insert(0,str(root/'snapshot/inference'))
    import merlin_compression  # Patches the inherited parameter transforms before importing them.
    for recipe in plan['candidates']:
        name=recipe['name']
        if final_names is not None and name not in final_names:continue
        folder=root/'candidates'/name
        with np.load(folder/f"features_{recipe['network']}.npz") as f:x=f['x_obs']
        evaluate_one(root,plan,name,recipe['network'],folder,x,threads,final=final_names is not None)
    if final_names is None or 'original_merlin' in final_names:
        evaluate_one(root,plan,'original_merlin','merlin',OLD/'models/linear',
                     np.load(root/'data/original_merlin_observations.npy'),threads,final=final_names is not None)


def main(root,threads):
    plan=json.loads((root/'plan.json').read_text())
    frozen=json.loads((root/'input_sha256.json').read_text())
    for filename,expected in frozen['references'].items():
        assert sha(filename)==expected, f'Frozen scientific input changed: {filename}'
    for filename,expected in frozen['code'].items():
        assert sha(root/filename)==expected, f'Frozen experiment code changed: {filename}'
    build_features(root,plan)
    train(root,plan)
    evaluate(root,plan,threads)
    from report import report
    winners=report(root,plan,final=False)
    evaluate(root,plan,threads,final_names=set(winners)|{'original_merlin'})
    report(root,plan,final=True)
    # A fresh interpreter is essential: the inherited module patches transforms at import time.
    subprocess.run([sys.executable,str(root/'code/bounded.py'),'--root',str(root),
                    '--threads',str(threads)],check=True)
    files={str(p.relative_to(root)):sha(p) for p in (root/'report').rglob('*') if p.is_file()}
    write_json(root/'COMPLETE.json',dict(observations=len(plan['cases']),methods=len(plan['candidates'])+1,
                 primary_param_space='linear',report_sha256=files,
                 plan_sha256=sha(root/'plan.json'),inputs_sha256=sha(root/'input_sha256.json'),
                 finished_utc=datetime.now(timezone.utc).isoformat()))
    print('COMPLETE',root/'report',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--threads',type=int,default=26)
    args=p.parse_args();main(args.root,args.threads)
