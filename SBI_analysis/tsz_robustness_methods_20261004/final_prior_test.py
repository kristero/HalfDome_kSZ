#!/usr/bin/env python3
"""Independent parameter-row test of selected methods; odd held-out rows never enter selection."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from common import ROOT, OLD, write_json
from pipeline import posterior_metrics
from report import table


def statistics(ens,x,truth):
    result,samples=posterior_metrics(ens,x,truth,900,seed=1313)
    reference=np.random.default_rng(4702).uniform(0,1,truth.shape)
    unit=(samples-ens.low)/(ens.high-ens.low)
    true_unit=(truth-ens.low)/(ens.high-ens.low)
    ranks=np.mean(np.linalg.norm(unit-reference,axis=-1)
                  <np.linalg.norm(true_unit-reference,axis=-1),axis=0)
    return dict(rows=len(x),posterior_samples=900,
        marginal_coverage68=np.mean(result['truth_in68'],axis=0).tolist(),
        marginal_coverage95=np.mean(result['truth_in95'],axis=0).tolist(),
        joint_rank68=float(np.mean(ranks<=.68)),joint_rank95=float(np.mean(ranks<=.95)),
        mean_log_prob_truth=float(np.mean(result['log_prob_truth'])),
        mean_log_prob_row_sem=float(np.std(result['log_prob_truth'],ddof=1)/np.sqrt(len(x))))


def run(root,space,threads):
    os.environ['MERLIN_PARAM_SPACE']=space
    sys.path.insert(0,str(root/'snapshot/inference'))
    import merlin_compression as mc
    plan=json.loads((root/'plan.json').read_text())
    selected=json.loads((root/'report/selection.json').read_text())['winners']
    recipes={r['name']:r for r in plan['candidates']}
    records={}
    for name in list(dict.fromkeys(['original_merlin']+list(selected.values()))):
        if name=='original_merlin':folder=OLD/'models'/space;variant='merlin'
        else:
            folder=root/'candidates'/name
            if space=='logit':folder=folder/'bounded'
            variant=recipes[name]['network']
        ens=mc.ensemble(folder,variant,threads)
        rows=np.arange(len(ens.f['x_held']))[1::2]
        assert len(rows)==1500
        result=statistics(ens,ens.f['x_held'][rows],ens.f['theta_held'][rows])
        fixed=root/'candidates'/name/'fixed_sky_test.npy'
        if name!='original_merlin' and recipes[name].get('augmentation'):
            result['fixed_sky_test']=statistics(ens,np.load(fixed)[rows],ens.f['theta_held'][rows])
        records[name]=result
    write_json(root/'report'/f'prior_final_test_{space}.json',dict(selection_uses='Even held-out rows only',
        final_test_uses='Odd held-out rows, no re-ranking',methods=records))
    output=[]
    for name,record in records.items():
        for j,parameter in enumerate(plan['parameters']):
            output.append(dict(method=name,parameter=parameter,coverage68=record['marginal_coverage68'][j],
                coverage95=record['marginal_coverage95'][j],joint_rank68=record['joint_rank68'],
                joint_rank95=record['joint_rank95'],mean_log_prob_truth=record['mean_log_prob_truth']))
    table(root/'report'/f'prior_final_test_{space}.csv',output)
    if space == 'logit':
        # The final process runs after both inference spaces have been checked.
        # This only formats figures; it cannot change selection or posteriors.
        from publication_plots import run as publication_plots
        publication_plots(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'comparison');p.add_argument('--space',choices=['linear','logit'],required=True)
    p.add_argument('--threads',type=int,default=26)
    args=p.parse_args();run(args.root,args.space,args.threads)
