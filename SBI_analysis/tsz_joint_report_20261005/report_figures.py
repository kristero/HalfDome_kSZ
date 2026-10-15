#!/usr/bin/env python3
"""Publication figures drawn from the frozen studies and new Pearson outputs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
STUDY = ROOT / 'evidence' / 'study'
PARAMS = ['P0','xc','beta','alpha_m_P0','alpha_m_xc','alpha_m_beta',
          'alpha_z_P0','alpha_z_xc','alpha_z_beta']
LABELS = [r'$P_0$',r'$x_c$',r'$\beta$',r'$\alpha_{m,P}$',r'$\alpha_{m,x}$',
          r'$\alpha_{m,\beta}$',r'$\alpha_{z,P}$',r'$\alpha_{z,x}$',r'$\alpha_{z,\beta}$']
COLORS = ['#2765a4','#21917c','#8854a6']
TRUTH = np.array([18.1,.497,4.35,.154,-.00865,.0393,-.758,.731,.415])
plt.rcParams.update({'font.size':16,'axes.labelsize':18,'axes.titlesize':19,
                    'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
                    'savefig.dpi':230,'pdf.fonttype':42,'font.family':'DejaVu Sans',
                    'axes.spines.top':False,'axes.spines.right':False})


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / (name+'.png'), bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / (name+'.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def grid_design():
    grid = np.full((10,10),4)
    grid[:6,:5]=0
    grid[6:,0]=1
    grid[0,5:]=2
    grid[6:,5:]=3
    colors=['#aec9e6','#5b91b2','#c0a65a','#65a88d','#edf0f4']
    fig,ax=plt.subplots(figsize=(10.4,6))
    ax.imshow(grid,cmap=ListedColormap(colors),vmin=0,vmax=4)
    ax.set_xticks(np.arange(10),np.arange(12345,12355),rotation=45,ha='right')
    ax.set_yticks(np.arange(10),np.arange(100,120,2))
    ax.set_xlabel('Mask seed');ax.set_ylabel('Simulation seed')
    ax.add_patch(Rectangle((-.5,-.5),1,1,fill=False,lw=3,color='black'))
    ax.set_xticks(np.arange(-.5,10,.5),minor=True)
    ax.tick_params(which='minor',length=0)
    handles=[Rectangle((0,0),1,1,color=c) for c in colors]
    ax.legend(handles,['Discovery: 30','Simulation test: 4','Mask test: 5',
                       'Joint test: 20','Mixed: 41'],bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout();save(fig,'grid_design')


def metric_tradeoffs(summary):
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
    for ax,arm,title in zip(axes,['simulation','mask','combined'],['Simulation','Mask','Combined']):
        methods=summary['subsets'][arm]['held_out']
        for name,metrics in methods.items():
            if name=='original_merlin':continue
            ax.scatter(metrics['width_cost'],np.hypot(metrics['truth_rms'],metrics['agreement_rms']),
                       color='#477fa8' if name.startswith('moped') else '#35947f',s=45,alpha=.65)
        selected=methods['merlin_band50_full'];original=methods['original_merlin']
        ax.scatter(1,np.hypot(original['truth_rms'],original['agreement_rms']),c='black',s=65,label='Original')
        ax.scatter(selected['width_cost'],np.hypot(selected['truth_rms'],selected['agreement_rms']),
                   c='#dc7138',marker='*',s=220,label='Band 50')
        ax.axvline(1.25,color='gray',ls='--');ax.set_title(title)
        ax.set_xlabel('Mean width ratio');ax.grid(alpha=.15)
    axes[0].set_ylabel('Truth + agreement cost')
    axes[2].legend(loc='upper right')
    fig.tight_layout();save(fig,'held_tradeoffs')


def variance_and_width(summary):
    fig,axes=plt.subplots(1,2,figsize=(13.8,4.5))
    for n,col,x in [('original_merlin','#2765a4',-.18),('merlin_band50_full','#dc7138',.18)]:
        part=np.asarray(summary['median_variance_parts'][n]).mean(axis=1)
        axes[0].bar(np.arange(3)+x,part,width=.35,color=col,label='Original' if x<0 else 'Band 50')
    axes[0].set_xticks(np.arange(3),['Simulation','Mask','Interaction'])
    axes[0].set_ylabel('Mean normalized median variance')
    axes[0].legend();axes[0].grid(axis='y',alpha=.15)
    for arm,c in zip(['simulation','mask','combined'],['#2765a4','#21917c','#8854a6']):
        p=summary['subsets'][arm]['held_out']['merlin_band50_full']['width_cost_per_parameter']
        axes[1].plot(np.arange(9),p,'o-',label=arm.title(),color=c)
    axes[1].axhline(1.25,ls='--',color='gray')
    axes[1].set_xticks(np.arange(9),LABELS,rotation=45,ha='right')
    axes[1].set_ylabel('Per-parameter width ratio')
    axes[1].legend();axes[1].grid(alpha=.15)
    fig.tight_layout();save(fig,'variance_width')


def calibration():
    linear=read(STUDY/'prior_final_test_linear.json')['methods']
    bounded=read(STUDY/'prior_final_test_logit.json')['methods']
    fig,axes=plt.subplots(1,2,figsize=(14,4.2))
    for ax,level in zip(axes,[68,95]):
        for data,name,label,col in [(linear,'original_merlin','Original, linear','#2765a4'),
                                   (linear,'merlin_band50_full','Band 50, linear','#dc7138'),
                                   (bounded,'merlin_band50_full','Band 50, bounded','#21917c')]:
            ax.plot(np.arange(9),np.asarray(data[name][f'marginal_coverage{level}'])*100,'o-',label=label,color=col)
        ax.axhline(level,color='black',ls='--')
        ax.set_xticks(np.arange(9),LABELS,rotation=45,ha='right')
        ax.set_ylabel('Contained truths (%)');ax.set_title(f'{level}% marginal intervals')
        ax.grid(alpha=.15)
    axes[1].legend(loc='lower right');fig.tight_layout();save(fig,'prior_calibration')


def combined_grid():
    rows=list(csv.DictReader((STUDY/'posterior_intervals.csv').open()))
    lookup={(r['method'],r['case']):float(r['median']) for r in rows if r['parameter']=='P0'}
    arrays=[]
    for name in ['original_merlin','merlin_band50_full']:
        arrays.append(np.array([[lookup[(name,f's{s}_m{m}')] for m in range(12345,12355)]
                                for s in range(100,120,2)]))
    fig,axes=plt.subplots(1,2,figsize=(13.5,3.7),layout='constrained')
    for ax,a,title in zip(axes,arrays,['Original','Band 50']):
        im=ax.imshow(a,vmin=10,vmax=60,cmap='viridis',aspect='auto')
        ax.set_xticks(np.arange(0,10,2),np.arange(12345,12355,2))
        ax.set_yticks(np.arange(0,10,2),np.arange(100,120,4))
        ax.set_xlabel('Mask seed');ax.set_ylabel('Simulation seed');ax.set_title(title)
    fig.colorbar(im,ax=list(axes),label=r'$P_0$ median',shrink=.9)
    save(fig,'combined_P0_grid')


def mechanism():
    diag=ROOT/'evidence/diagnosis'
    old=read(diag/'original_linear.json');extra=read(diag/'extended_original_linear.json')
    old_labels={d['label']:i for i,d in enumerate(read(diag/'counterfactuals.json'))}
    ext_labels={d['label']:i for i,d in enumerate(read(diag/'extended_cases.json'))}
    groups=[(old,old_labels,['mean_noisy','mean_clean','smooth_clean_5','fine_only_5']),
            (extra,ext_labels,['mask_mean_actual','mask_mean_clean','mask_mean_smooth','mask_mean_fine']),
            (extra,ext_labels,['joint_mean_actual','joint_mean_clean','joint_mean_smooth','joint_mean_fine'])]
    fig,axes=plt.subplots(3,3,figsize=(15,6.6),sharey='row')
    for col,(data,lookup,labels) in enumerate(groups):
        ids=[lookup[n] for n in labels]
        for row,param in enumerate([0,2,3]):
            ax=axes[row,col]
            med=np.array(data['median'])[ids,param];lo=np.array(data['q16'])[ids,param];hi=np.array(data['q84'])[ids,param]
            ax.errorbar(np.arange(4),med,yerr=[med-lo,hi-med],fmt='o',c=COLORS[0],lw=1.5)
            ax.axhline(TRUTH[param],color='black',ls='--',lw=1.3)
            ax.axhline(old['median'][old_labels['baseline']][param],color='gray',ls=':',lw=1.3)
            ax.set_xticks(np.arange(4),['Actual','Clean','Smooth','Fine'])
            if col==0:ax.set_ylabel(LABELS[param])
            if row==0:ax.set_title(['Simulation','Mask','Combined'][col])
            ax.grid(alpha=.12)
    fig.tight_layout();save(fig,'mechanism_summary')


def posterior_subset(arm):
    rows=list(csv.DictReader((STUDY/'posterior_intervals.csv').open()))
    seeds=list(range(100,120,2));masks=list(range(12345,12355))
    cases=[f's{s}_m12345' for s in seeds] if arm=='simulation' else [f's100_m{m}' for m in masks]
    fig,axes=plt.subplots(3,3,figsize=(15,7.2))
    lookup={(r['method'],r['case'],r['parameter']):r for r in rows}
    for p,ax in enumerate(axes.flat):
        for method,col,offset in [('original_merlin','#2765a4',-.13),('merlin_band50_full','#dc7138',.13)]:
            values=[lookup[(method,c,PARAMS[p])] for c in cases]
            mid=np.array([float(v['median']) for v in values]);lo=np.array([float(v['q16']) for v in values]);hi=np.array([float(v['q84']) for v in values])
            ax.errorbar(np.arange(10)+offset,mid,yerr=[mid-lo,hi-mid],fmt='o',ms=4,c=col,lw=1.3,
                        label='Original' if offset<0 else 'Band 50')
        ax.axhline(TRUTH[p],color='black',ls='--',lw=1.2);ax.set_title(LABELS[p])
        ax.set_xticks(np.arange(0,10,2),seeds[::2] if arm=='simulation' else masks[::2])
        ax.grid(alpha=.12)
        if p>=6:ax.set_xlabel('Simulation seed' if arm=='simulation' else 'Mask seed')
    axes[0,0].legend(fontsize=11);fig.tight_layout();save(fig,'posterior_'+arm)


def pearson_figures():
    if not (ROOT/'pearson_linear.json').exists():return
    for space in ['linear','logit']:
        d=read(ROOT/f'pearson_{space}.json');r=np.asarray(d['correlation']);delta=np.asarray(d['delta'])
        fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),layout='constrained')
        for ax,mat,title in zip(axes,r[:2],['Original',r'Smooth change $\sigma_\ell=5$']):
            im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
            ax.set_title(title)
            ax.set_xticks(np.arange(9),LABELS,rotation=60,ha='right',fontsize=14)
            ax.set_yticks(np.arange(9),LABELS,fontsize=14)
        fig.colorbar(im,ax=list(axes),shrink=.86,label='Pearson r')
        save(fig,f'pearson_{space}_primary')
        for group,mats,limit,names in [('r',r,1,['Original',r'Smooth $\sigma_\ell=5$',r'Smooth $\sigma_\ell=50$']),
                                     ('delta',delta[1:],max(.1,float(np.max(np.abs(delta[1:])))),[r'$\Delta r$: $\sigma_\ell=5$',r'$\Delta r$: $\sigma_\ell=50$'])]:
            fig,axes=plt.subplots(1,len(mats),figsize=(16,5.3) if group=='r' else (12,5.3),layout='constrained')
            axes=np.atleast_1d(axes)
            for ax,mat,title in zip(axes,mats,names):
                im=ax.imshow(mat,vmin=-limit,vmax=limit,cmap='RdBu_r')
                ax.set_title(title)
                ax.set_xticks(np.arange(9),LABELS,rotation=60,ha='right',fontsize=14 if group=='delta' else 12)
                ax.set_yticks(np.arange(9),LABELS,fontsize=14 if group=='delta' else 12)
                if group=='delta':
                    for i in range(9):
                        for j in range(9):
                            if j<i:ax.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',fontsize=10,
                                           color='white' if abs(mat[i,j])>limit*.65 else 'black')
            fig.colorbar(im,ax=list(axes),shrink=.86,label='Pearson r' if group=='r' else 'Change in r')
            save(fig,f'pearson_{space}_{group}')
    d=read(ROOT/'pearson_linear.json');samples=np.load(ROOT/'data/linear_plot_draws.npz')['draws']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    for ax,(a,b) in zip(axes,[(0,2),(1,2),(0,6)]):
        for i,color in enumerate(COLORS):
            ax.scatter(samples[::3,i,a],samples[::3,i,b],s=4,alpha=.18,color=color,rasterized=True)
        ax.scatter(TRUTH[a],TRUTH[b],marker='*',s=150,color='black',zorder=5)
        ax.set_xlabel(LABELS[a]);ax.set_ylabel(LABELS[b]);ax.grid(alpha=.12)
    handles=[Line2D([0],[0],marker='o',ls='',color=c,label=n) for c,n in zip(COLORS,['Original',r'Smooth change $\sigma_\ell=5$',r'Smooth change $\sigma_\ell=50$'])]
    handles.append(Line2D([0],[0],marker='*',ls='',color='black',label='B12 truth'))
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'pearson_joint_clouds')
    f=np.load(ROOT/'data/baseline_spectra.npz');ell=f['ell'];spec=f['spectra']
    fig,axes=plt.subplots(1,2,figsize=(14,4.1))
    for i,color in enumerate(COLORS):
        axes[0].plot(ell[500:900],spec[i,500:900],color=color,lw=1,label=['Original',r'$\sigma_\ell=5$',r'$\sigma_\ell=50$'][i])
        if i:axes[1].plot(ell,100*(spec[i]-spec[0])/f['clean'],color=color,lw=.8)
    axes[0].set_xlabel(r'$\ell$');axes[0].set_ylabel(r'$D_\ell$');axes[0].legend()
    axes[1].set_xlabel(r'$\ell$');axes[1].set_ylabel(r'$100\,(D^{\rm smooth}-D^{\rm original})/D^{\rm clean}$')
    axes[1].axhline(0,color='gray',ls='--');axes[1].set_xscale('log')
    fig.tight_layout();save(fig,'baseline_smoothing')
    # A full numerical CSV accompanies the heatmaps, including sampling precision.
    with (ROOT/'pearson_pairs.csv').open('w',newline='') as out:
        w=csv.writer(out);w.writerow(['space','control','parameter_a','parameter_b','r_original','r_control','delta_r','delta_mcse'])
        for space in ['linear','logit']:
            d=read(ROOT/f'pearson_{space}.json')
            for case,pairs in zip(['sigma5','sigma50'],d['largest_changes']):
                for p in pairs:w.writerow([space,case,p['first'],p['second'],p['original'],p['smooth'],p['delta'],p['delta_mcse']])


def main():
    summary=read(STUDY/'summary.json')
    grid_design();metric_tradeoffs(summary);variance_and_width(summary);calibration();combined_grid();mechanism()
    posterior_subset('simulation');posterior_subset('mask');pearson_figures()
    print('Figure files:',len(list(FIG.glob('*'))))


if __name__=='__main__':main()
