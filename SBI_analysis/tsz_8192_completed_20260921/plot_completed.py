"""Publication-style figures from the completed cluster analysis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'plots'
OUT.mkdir(exist_ok=True)
REPORT = json.loads((ROOT / 'cluster_report.json').read_text())
EXTREMES = json.loads((ROOT / 'extreme_noise.json').read_text())
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':17,
                     'legend.fontsize':11,'xtick.labelsize':13,'ytick.labelsize':13})
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7']


def save(fig, name):
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(OUT / f'{name}.{suffix}', dpi=190, bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax, record in zip(axes,REPORT['anchors']):
    cuts=record['cuts'];x=[r['ell_max'] for r in cuts]
    ax.semilogy(x,[r['spherical_moped_distance'] for r in cuts],'-o',color=COLORS[0],label='4096 / 8192: local MOPED')
    ax.semilogy(x,[r['reference16384_spherical_moped_distance'] for r in cuts],'-s',color=COLORS[2],label='8192 / 16384: local MOPED')
    ax.semilogy(x,[r['frozen_moped_distance'] for r in cuts],'--',color=COLORS[1],label='4096 / 8192: old MOPED')
    ax.axhline(.1,color='.5',ls=':',lw=1)
    ax.set(title='Battaglia12' if record['anchor']=='Battaglia12' else 'FLAMINGO-fit anchor',xlabel=r'$\ell_{\max}$')
    ax.grid(alpha=.2);ax.legend(loc='upper left')
axes[0].set_ylabel('Compressed shift / noise scale')
save(fig,'resolution_moped')

fig, axes=plt.subplots(1,2,figsize=(12,4.7))
for color,record in zip(COLORS,REPORT['anchors']):
    cuts=record['cuts'];x=[r['ell_max'] for r in cuts]
    label='Battaglia12' if record['anchor']=='Battaglia12' else 'FLAMINGO-fit anchor'
    axes[0].plot(x,[100*r['information_trace_fraction'] for r in cuts],'-o',color=color,label=label)
    axes[1].semilogy(x,[100*min(r['identified_mode_information_fractions']) for r in cuts],'-o',color=color,label=label)
axes[0].set(title='Prior-scaled Fisher trace',ylabel='Retained fraction [%]')
axes[1].set(title='Least retained direction',ylabel='Retained fraction [%]')
for ax in axes:
    ax.set_xlabel(r'$\ell_{\max}$');ax.grid(alpha=.2);ax.legend()
save(fig,'information_cutoffs')

fig, axes=plt.subplots(1,2,figsize=(12,4.7))
labels=[r'$P_0$',r'$x_c$',r'$\beta$',r'$a_{m,P}$',r'$a_{m,x}$',r'$a_{m,\beta}$',r'$a_{z,P}$',r'$a_{z,x}$',r'$a_{z,\beta}$']
for color,record in zip(COLORS,REPORT['anchors']):
    label='Battaglia12' if record['anchor']=='Battaglia12' else 'FLAMINGO-fit anchor'
    # P0 is analytic at both steps; plot the eight numerical derivative changes.
    axes[0].semilogy(np.arange(1,9),100*np.array(record['derivative_step_relative_change'][1:]),'-o',color=color,label=label)
    singular=np.array(record['cuts'][-1]['spherical_moped']['singular_values'])
    axes[1].semilogy(np.arange(1,10),singular/singular[0],'-o',color=color,label=label)
axes[0].set(xticks=np.arange(1,9),xticklabels=labels[1:],ylabel='Derivative step change [%]',title='Step halving')
axes[0].tick_params(axis='x',rotation=35)
axes[1].axhline(1e-3,color='.45',ls='--',label='Default relative threshold')
axes[1].set(xlabel='Ordered parameter direction',ylabel='Singular value / largest',title='Local sensitivity spectrum',xticks=np.arange(1,10))
for ax in axes:ax.grid(alpha=.2);ax.legend()
save(fig,'derivatives_and_modes')

fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharex=True)
for ax,record,color in zip(axes,EXTREMES['records'],[COLORS[3],COLORS[1]]):
    x=[r['ell_max'] for r in record['cuts']]
    ax.semilogy(x,[r['distance'] for r in record['cuts']],'-o',color=color,label='First covariance half')
    ax.semilogy(x,[r['alternate_covariance_distance'] for r in record['cuts']],'--',color='.35',label='Second covariance half')
    ax.set(xlabel=r'$\ell_{\max}$',ylabel='4096-8192 shift / noise scale',
           title='Compact, faint' if record['label']=='compact' else 'Extended, bright')
    ax.grid(alpha=.2);ax.legend()
save(fig,'extreme_noise')

selected=[next(r for r in REPORT['benchmarks'] if r['task_id']==i) for i in [80,78,79]]
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
x=np.arange(3);names=['Standard\n26 threads','Lean\n26 threads','Lean\n13 threads']
seconds=np.array([r['candidate_signal_seconds'] for r in selected])
axes[0].bar(x,seconds/60,color=[COLORS[1],COLORS[0],COLORS[2]])
axes[1].bar(x,seconds*np.array([r['threads'] for r in selected])/3600,color=[COLORS[1],COLORS[0],COLORS[2]])
axes[0].set(ylabel='Signal preparation [min]',title='Time per map')
axes[1].set(ylabel='Signal preparation [CPU hours]',title='CPU cost per map')
for ax in axes:
    ax.set_xticks(x,names);ax.grid(axis='y',alpha=.2)
save(fig,'performance')
print('Saved five figure pairs to',OUT)
