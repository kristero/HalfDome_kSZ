#!/usr/bin/env python3
"""Matched Merlin scale-cut controls and joint final comparison; reuse the validated full grid."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

from common import ROOT, EDGES, fit_transform, transform, safe_cholesky, add_sky_variation, sha, write_json
from pipeline import train, evaluate
from final_report import report


def links(root, source):
    root.mkdir(parents=True,exist_ok=True)
    for name in ['data','snapshot','results']:
        path=root/name
        if not path.exists():
            path.symlink_to(source/name,target_is_directory=True)
    (root/'candidates').mkdir(exist_ok=True)
    (root/'report').mkdir(exist_ok=True)


def prepare(parent):
    root=parent/'merlin_checks'
    links(root,parent)
    base=json.loads((parent/'plan.json').read_text())
    candidates=[]
    for low in [280,580,1080]:
        candidates.append(dict(name=f'merlin_lmin{low}',lmin=low))
    for high in [5000,4000]:
        candidates.append(dict(name=f'merlin_lmax{high}',lmax=high))
    candidates.append(dict(name='merlin_sky_augmented',augmentation=True))
    for recipe in candidates:
        recipe.update(network='merlin',covariance='noiseonly',strength=1.)
        recipe.setdefault('lmin',80);recipe.setdefault('lmax',7980)
    plan=dict(base,root=str(root),candidates=candidates,
              additional_control='Same learned Merlin head, noise-only whitening, with cuts or approximate signal augmentation')
    write_json(root/'plan.json',plan)
    if (root/'FEATURES_COMPLETE.json').exists():
        return root,plan
    model=np.load(parent/'data/local_model.npz')
    design=np.load(parent/'data/design.npz')
    pool,held=design['pool'],design['held'];fid=model['fid']
    clean=np.load(parent/'data/clean_bands.npy')/fid
    noisy=np.load(parent/'data/noisy_bands.npy')/fid
    grid=np.load(parent/'data/grid.npz')
    ratios=np.load(parent/'data/sky_discovery.npz')['ratios']
    for recipe in candidates:
        folder=root/'candidates'/recipe['name'];folder.mkdir(exist_ok=True)
        keep=(EDGES[:-1]>=recipe['lmin'])&(EDGES[1:]<=recipe['lmax'])
        ids=np.flatnonzero(keep)
        l,jitter=safe_cholesky(model['C_nn'][np.ix_(keep,keep)])
        weights=np.zeros((158,len(ids)))
        weights[keep]=solve_triangular(l.T,np.eye(len(ids)),lower=False)
        changed_clean=clean
        changed_noisy=noisy
        if recipe.get('augmentation'):
            choice=np.random.default_rng(20261004).integers(len(ratios),size=len(noisy))
            changed_clean=clean*ratios[choice]
            changed_noisy=add_sky_variation(clean,noisy,ratios,choice)
        # Subtract the augmented clean signal: the asinh scale remains the SO residual
        # scale, preserving the requested noise-only Merlin preprocessing convention.
        params=fit_transform(changed_noisy@weights,changed_clean@weights,pool)
        x=transform(changed_noisy@weights,params)
        obs=transform((grid['noisy']/fid)@weights,params)
        np.savez(folder/'features_merlin.npz',x_pool=x[pool],x_held=x[held],x_obs=obs,x_draws=obs[:1],
            theta_pool=design['theta'][pool],theta_held=design['theta'][held],pool=pool,held=held,
            low=design['low'],high=design['high'],s_mean=np.zeros(x.shape[1]),s_sd=np.ones(x.shape[1]),
            input_transform=np.array('asinh'),**params)
        np.save(folder/'fixed_sky_test.npy',transform(noisy[held]@weights,params))
        np.savez(folder/'compression.npz',weights=weights,fid=fid,keep=keep,**params)
        write_json(folder/'prepare.json',dict(recipe,n_input_bands=len(ids),summary_dimension=len(ids),
            effective_lmin=int(EDGES[ids[0]]),effective_lmax=int(EDGES[ids[-1]+1]-1),
            scale_fraction_retained=float(keep.mean()),
            audit=dict(jitter=jitter,whitening='SO noise-only covariance',asinh_scale='SO residual MAD')))
    sources={str(parent/'data'/name):sha(parent/'data'/name) for name in
             ['design.npz','noisy_bands.npy','clean_bands.npy','grid.npz','local_model.npz','sky_discovery.npz']}
    write_json(root/'FEATURES_COMPLETE.json',dict(source_sha256=sources,
                                               code_sha256=sha(Path(__file__)),methods=len(candidates)))
    return root,plan


def screen(parent,threads):
    root,plan=prepare(parent)
    train(root,plan)
    evaluate(root,plan,threads)
    write_json(root/'COMPLETE.json',dict(methods=len(plan['candidates']),observations=100,
        screen_sha256={str(p.relative_to(root)):sha(p) for p in (root/'candidates').glob('*/evaluation.json')},
        finished_utc=datetime.now(timezone.utc).isoformat()))


def combine(parent,threads):
    assert (parent/'COMPLETE.json').exists() and (parent/'merlin_checks/COMPLETE.json').exists()
    root=parent/'comparison';links(root,parent)
    primary=json.loads((parent/'plan.json').read_text())
    extra=json.loads((parent/'merlin_checks/plan.json').read_text())
    recipes=primary['candidates']+extra['candidates']
    plan=dict(primary,root=str(root),candidates=recipes,methods_total_including_original=len(recipes)+1,
              base_complete_sha256=sha(parent/'COMPLETE.json'),
              merlin_controls_complete_sha256=sha(parent/'merlin_checks/COMPLETE.json'))
    write_json(root/'plan.json',plan)
    for name in ['original_merlin']+[r['name'] for r in primary['candidates']]:
        target=root/'candidates'/name
        if not target.exists():target.symlink_to(parent/'candidates'/name,target_is_directory=True)
    for recipe in extra['candidates']:
        target=root/'candidates'/recipe['name']
        if not target.exists():target.symlink_to(parent/'merlin_checks/candidates'/recipe['name'],target_is_directory=True)
    winners=report(root,plan,False)
    evaluate(root,plan,threads,final_names=set(winners)|{'original_merlin'})
    report(root,plan,True)
    import subprocess
    subprocess.run([sys.executable,str(parent/'code/bounded.py'),'--root',str(root),'--threads',str(threads)],check=True)
    for space in ['linear','logit']:
        subprocess.run([sys.executable,str(parent/'code/joint_truth.py'),'--root',str(root),
                        '--space',space,'--threads',str(threads)],check=True)
        subprocess.run([sys.executable,str(parent/'code/final_prior_test.py'),'--root',str(root),
                        '--space',space,'--threads',str(threads)],check=True)
    provenance=dict(code={str(p):sha(p) for p in (parent/'code').glob('*') if p.is_file()},
                    parent_inputs_sha256=sha(parent/'input_sha256.json'))
    write_json(root/'input_sha256.json',provenance)
    write_json(root/'COMPLETE.json',dict(observations=100,methods=len(recipes)+1,primary_param_space='linear',
        plan_sha256=sha(root/'plan.json'),inputs_sha256=sha(root/'input_sha256.json'),
        report_sha256={str(p.relative_to(root)):sha(p) for p in (root/'report').rglob('*') if p.is_file()},
        finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['screen','combine']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--threads',type=int,default=26)
    args=p.parse_args()
    (screen if args.mode=='screen' else combine)(args.root,args.threads)
