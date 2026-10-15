#!/usr/bin/env python3
"""Small, descriptive full-grid signal audit; no inference or method tuning."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, ELL, band_operator, arm_indices, write_json
from report import save_figure, table, ARMS, COLORS


def audit(root):
    plan=json.loads((root/'plan.json').read_text())
    clean=np.load(root/'data/grid.npz')['clean']
    centers=band_operator()@ELL
    folder=root/'signal_audit';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':17,'axes.labelsize':19,'axes.titlesize':19,
                         'xtick.labelsize':15,'ytick.labelsize':15,'pdf.fonttype':42})
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(21,6),sharey=True)
    for ax,arm in zip(axes,ARMS):
        ids=arm_indices(plan['cases'],arm)
        relative=clean[ids]/clean[0]-1
        if arm=='combined':
            lo,mid,hi=np.quantile(relative,[.16,.5,.84],axis=0)
            ax.fill_between(centers,100*lo,100*hi,color=COLORS[0],alpha=.2)
            ax.plot(centers,100*mid,color=COLORS[0],lw=2)
        else:
            for curve in relative:ax.plot(centers,100*curve,lw=1.3,alpha=.8)
        ax.axhline(0,color='k',lw=1);ax.set_xscale('log')
        ax.set(xlabel=r'Multipole $\ell$',title=arm.capitalize());ax.grid(alpha=.15)
        for low,high in [(80,580),(580,2980),(2980,7980)]:
            keep=(centers>=low)&(centers<high)
            values=relative[:,keep]
            rows.append(dict(arm=arm,ell_min=low,ell_max=high-1,cases=len(ids),
                mean_offset_percent=float(100*values.mean()),
                rms_about_baseline_percent=float(100*np.sqrt(np.mean(values**2))),
                rms_scatter_about_ensemble_percent=float(100*np.sqrt(np.mean(values.var(0))))))
    axes[0].set_ylabel('Signal change from baseline [%]')
    fig.tight_layout();save_figure(fig,folder,'signal_variation')
    table(folder/'variation_by_scale.csv',rows)
    write_json(folder/'audit.json',dict(observations=100,statistics=rows,
        meaning='Descriptive clean-spectrum variation, equal-weighted bands. Not independent-mock covariance or inference results.',
        methods='All method recipes fixed before this audit; no tuning on validation skies.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    audit(p.parse_args().root)
