#!/usr/bin/env python3
"""Joint theta-space credibility of B12 truth: distinguish median shifts from density exclusion."""
import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from common import ROOT, OLD, THETA, arm_indices, transform, write_json
from report import save_figure, table, ARMS, COLORS


def run(root,space,threads):
    os.environ['MERLIN_PARAM_SPACE']=space
    sys.path.insert(0,str(root/'snapshot/inference'))
    import torch
    import merlin_compression as mc
    from train_ensemble import to_u, log_jacobian_theta_per_u
    plan=json.loads((root/'plan.json').read_text())
    selected=json.loads((root/'report/selection.json').read_text())['winners']
    recipes={r['name']:r for r in plan['candidates']}
    names=list(dict.fromkeys(['original_merlin']+list(selected.values())))
    rows=[];levels={}
    for name in names:
        if name=='original_merlin':
            folder=OLD/'models'/space;variant='merlin'
            full=np.load(root/'data/grid.npz')['full']
            with np.load(OLD/'reference/whitening_pca.npz') as w:
                s=solve_triangular(w['L'],full.T,lower=True).T@w['V']
            with np.load(folder/'features_merlin.npz') as f:
                x=transform(s,{k:f[k] for k in ['s_center','s_scale','z_mean','z_sd']})
        else:
            folder=root/'candidates'/name
            if space=='logit':folder=folder/'bounded'
            variant=recipes[name]['network'];x=np.load(folder/f'features_{variant}.npz')['x_obs']
        ens=mc.ensemble(folder,variant,threads)
        torch.manual_seed(7103)
        u=ens.sample_u(x,6000,rng=np.random.default_rng(7103))
        # Compare densities in the SAME physical parameter coordinates, not logit density.
        log_samples=ens.log_prob_u(u,x)-log_jacobian_theta_per_u(u,ens.low,ens.high)
        truth=to_u(np.broadcast_to(THETA,(len(x),9)),ens.low,ens.high)
        log_truth=ens.log_prob_u(truth[None],x)[0]-log_jacobian_theta_per_u(truth,ens.low,ens.high)
        hpd=np.mean(log_samples>log_truth,axis=0)
        levels[name]=hpd
        for i,case in enumerate(plan['cases']):
            rows.append(dict(method=name,param_space=space,case=case['label'],
                truth_hpd_level=float(hpd[i]),truth_inside_joint68=bool(hpd[i]<=.68),
                truth_inside_joint95=bool(hpd[i]<=.95),
                rank_mcse=float(np.sqrt(hpd[i]*(1-hpd[i])/len(u))),samples=len(u)))
    table(root/'report'/f'joint_truth_{space}.csv',rows)
    write_json(root/'report'/f'joint_truth_{space}.json',dict(coordinates='Physical theta; correct transform Jacobian',
        interpretation='Posterior mass at density higher than truth. Small rank means truth is in a high-density region.',
        samples=6000,levels={n:v.tolist() for n,v in levels.items()}))
    plt.rcParams.update({'font.size':17,'axes.labelsize':19,'axes.titlesize':19,
                         'xtick.labelsize':15,'ytick.labelsize':15,'legend.fontsize':13,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(21,6),sharey=True)
    for ax,arm in zip(axes,ARMS):
        ids=arm_indices(plan['cases'],arm)
        for k,name in enumerate(list(dict.fromkeys(['original_merlin',selected[arm]]))):
            ax.plot(np.arange(len(ids)),levels[name][ids],'.-',ms=5,lw=.9,
                    color=COLORS[k],label='Original' if k==0 else 'Selected')
        ax.axhline(.68,color='#777777',ls=':',lw=1.3);ax.axhline(.95,color='k',ls='--',lw=1.3)
        ax.set(xlabel='Sky index',title=arm.capitalize(),ylim=(0,1.02));ax.grid(alpha=.15)
    axes[0].set_ylabel('Joint HPD level of truth');axes[0].legend()
    fig.tight_layout();save_figure(fig,root/'report',f'joint_truth_{space}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'comparison');p.add_argument('--space',choices=['linear','logit'],required=True)
    p.add_argument('--threads',type=int,default=26)
    args=p.parse_args();run(args.root,args.space,args.threads)
