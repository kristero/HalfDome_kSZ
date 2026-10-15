#!/usr/bin/env python3
"""Prior-bounded companions for the methods selected by the primary unbounded test."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

os.environ['MERLIN_PARAM_SPACE']='logit'
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

from common import ROOT, OLD, THETA, transform, write_json
from pipeline import train_one, posterior_metrics
from report import metrics, table, ARMS, save_figure, COLORS
from common import LABELS
from common import arm_indices


def run(root, threads):
    plan=json.loads((root/'plan.json').read_text())
    selection=json.loads((root/'report/selection.json').read_text())
    names=list(dict.fromkeys(selection['winners'].values()))
    recipes={r['name']:r for r in plan['candidates']}
    jobs=[]
    for name in names:
        if name=='original_merlin':
            continue
        folder=root/'candidates'/name/'bounded';folder.mkdir(exist_ok=True)
        variant=recipes[name]['network']
        if not (folder/f'features_{variant}.npz').exists():
            shutil.copy2(root/'candidates'/name/f'features_{variant}.npz',folder/f'features_{variant}.npz')
        for seed in plan['training_seeds']:
            if not all((folder/f'{variant}_seed{seed}.{ext}').exists() for ext in ['json','pt']):
                jobs.append((str(root),name,variant,seed,4,'logit',True))
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=context) as pool:
        futures=[pool.submit(train_one,*job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):print('BOUNDED TRAIN',future.result(),flush=True)
    sys.path.insert(0,str(root/'snapshot/inference'))
    import merlin_compression as mc
    records={}
    for name in list(dict.fromkeys(['original_merlin']+names)):
        output=root/'candidates'/name/'bounded';output.mkdir(exist_ok=True)
        target=output/'evaluation_final.json'
        if target.exists():
            records[name]=json.loads(target.read_text());continue
        if name=='original_merlin':
            folder=OLD/'models/logit';variant='merlin'
            full=np.load(root/'data/grid.npz')['full']
            with np.load(OLD/'reference/whitening_pca.npz') as w:
                summary=solve_triangular(w['L'],full.T,lower=True).T@w['V']
            with np.load(folder/'features_merlin.npz') as f:
                x=transform(summary,{k:f[k] for k in ['s_center','s_scale','z_mean','z_sd']})
                np.testing.assert_allclose(x[:1],f['x_obs'][:1],rtol=1e-4,atol=1e-4)
        else:
            folder=output;variant=recipes[name]['network']
            x=np.load(folder/f'features_{variant}.npz')['x_obs']
        ens=mc.ensemble(folder,variant,threads)
        result,samples=posterior_metrics(ens,x,THETA,plan['posterior_samples_final'])
        result.update(name=name,network=variant,param_space='logit')
        np.save(output/'posterior_final.npy',samples);del samples
        write_json(target,result);records[name]=result
    rows=[]
    for arm in ARMS:
        for subset,ids in [('held_out',arm_indices(plan['cases'],arm,True)),('all',arm_indices(plan['cases'],arm))]:
            for name in list(dict.fromkeys(['original_merlin',selection['winners'][arm]])):
                record=metrics(records[name],records['original_merlin'],ids)
                rows.append(dict(arm=arm,subset=subset,method=name,
                                 **{k:v for k,v in record.items() if not isinstance(v,list)}))
    table(root/'report/bounded_comparison.csv',rows)
    write_json(root/'report/bounded_summary.json',dict(selection='Primary unbounded winners, without re-ranking',
                                                      results=rows))
    plt.rcParams.update({'font.size':17,'axes.labelsize':19,'axes.titlesize':19,
                         'xtick.labelsize':15,'ytick.labelsize':15,'pdf.fonttype':42})
    for arm in ARMS:
        name=selection['winners'][arm]
        unbounded=json.loads((root/'candidates'/name/'evaluation_final.json').read_text())
        bounded=records[name]
        ids=arm_indices(plan['cases'],arm)
        fig,axes=plt.subplots(3,3,figsize=(18,12))
        for j,ax in enumerate(axes.flat):
            for k,record in enumerate([unbounded,bounded]):
                med=np.array(record['median'])[ids,j]
                lo=np.array(record['q16'])[ids,j];hi=np.array(record['q84'])[ids,j]
                ax.errorbar(np.arange(len(ids))+(-.15 if k==0 else .15),med,yerr=[med-lo,hi-med],
                            fmt='o',ms=3 if len(ids)>20 else 5,lw=.7,color=COLORS[k],alpha=.75,
                            label='Unbounded' if k==0 else 'Prior bounded')
            ax.axhline(THETA[j],color='k',ls='--',lw=1.3)
            ax.set_title(LABELS[j]);ax.set_xlabel('Sky index');ax.grid(alpha=.15)
        axes.flat[0].legend(fontsize=13)
        fig.tight_layout();save_figure(fig,root/'report',f'bounded_vs_unbounded_{arm}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--threads',type=int,default=26)
    args=p.parse_args();run(args.root,args.threads)
