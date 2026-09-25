"""Publication figures with explicit separation of old and new geometries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from design_tests import LOW,HIGH,NAMES

ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent/'tsz_guardrail_study'
if not STUDY.exists():STUDY=ROOT.parent/'tsz_guardrail_study_20260915'
ELL=np.arange(7980);EDGES=np.r_[np.arange(80,7881,200),7980]
COV=np.load(STUDY/'audit/noise_covariance.npz')
ELLB=COV['ell'];SIGMA=np.sqrt(np.diag(COV['covariance']))
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'legend.fontsize':12,'axes.titlesize':16})


def bins(cl):
    dl=cl[:7980]*ELL*(ELL+1)/(2*np.pi)
    return np.array([np.average(dl[a:b],weights=2*ELL[a:b]+1) for a,b in zip(EDGES[:-1],EDGES[1:])])


def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(ROOT/'plots'/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def spectra():
    for fresh in [False,True]:
        fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
        available=[]
        for j,case in enumerate(['Battaglia12','FL_L1_m9']):
            curves={}
            for nside,color in zip([4096,8192,16384],['#D55E00','#0072B2','#009E73']):
                variant={4096:'logz2',8192:'pixel8192_2',16384:'pixel16384_2'}[nside]
                path=(ROOT/'fullsky'/case/f'nside{nside}_grid1' if fresh else STUDY/'maps'/case/variant)
                if not (path/'status.json').exists():continue
                status=json.loads((path/'status.json').read_text())
                if status.get('returncode',status.get('exit_code'))!=0:continue
                curve=bins(np.load(path/'masked_clean_cl.npy'));curves[nside]=(curve,color)
                axes[0,j].semilogy(ELLB,curve,color=color,label=str(nside))
            available.append(len(curves))
            if 16384 in curves:
                reference=curves[16384][0]
                for nside,(curve,color) in curves.items():
                    if nside==16384:continue
                    axes[1,j].plot(ELLB,100*(curve/reference-1),color=color)
                    axes[2,j].plot(ELLB,(curve-reference)/SIGMA,color=color)
            axes[0,j].set_title('Battaglia12' if j==0 else 'Historical FLAMINGO fit')
            if curves:axes[0,j].legend(title='Raw NSIDE',ncol=3,fontsize=10)
            axes[2,j].set_xlabel(r'$\ell$')
            for ax in axes[:,j]:ax.grid(alpha=.15)
        axes[0,0].set_ylabel(r'Masked $D_\ell^{yy}$')
        axes[1,0].set_ylabel('Difference from 16384 [%]')
        axes[2,0].set_ylabel(r'$\Delta D_b/\sigma_b$')
        fig.suptitle('Spherical projection' if fresh else 'Historical cylindrical projection',fontsize=17)
        if any(available):save(fig,'resolution_spherical' if fresh else 'resolution_historical')
        else:plt.close(fig)


def priors():
    old=ROOT.parent/'flamingo_linear_prior/cluster_results/completed_20260918'
    if not old.exists():return
    theta=np.load(old/'dataset/theta.npy')
    manifest=json.loads((old/'manifest.json').read_text())['prior']
    oldlow=np.array(manifest['lower']);oldhigh=np.array(manifest['upper'])
    cases=json.loads((ROOT/'inputs/cases.json').read_text())
    labels=[r'$P_0$',r'$x_c$',r'$\beta_0$',r'$\alpha_{m,P_0}$',r'$\alpha_{m,x_c}$',
            r'$\alpha_{m,\beta}$',r'$\alpha_{z,P_0}$',r'$\alpha_{z,x_c}$',r'$\alpha_{z,\beta}$']
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for i,ax in enumerate(axes.flat):
        ax.hist(theta[:,i],bins=35,density=True,color='.65',alpha=.45,label='Completed 8k')
        ax.plot([LOW[i],HIGH[i]],[1/(HIGH[i]-LOW[i])]*2,color='#0072B2',lw=2.5,label='Flat candidate')
        for value in [oldlow[i],oldhigh[i]]:ax.axvline(value,color='.3',ls=':',lw=1.2)
        ax.axvline(cases['Battaglia12'][i],color='black',lw=1.6,label='Battaglia12')
        ax.axvline(cases['FL_L1_m9'][i],color='#D55E00',ls='--',lw=1.7,label='Historical FL fit')
        ax.set_xlabel(labels[i]);ax.set_xlim(LOW[i]-.03*(HIGH[i]-LOW[i]),HIGH[i]+.03*(HIGH[i]-LOW[i]))
        ax.grid(alpha=.12)
        if i%3==0:ax.set_ylabel('Probability density')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=12)
    fig.subplots_adjust(top=.93,hspace=.38,wspace=.27)
    for ext in ['pdf','png']:fig.savefig(ROOT/'plots'/('flat_prior_comparison.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    (ROOT/'plots').mkdir(exist_ok=True)
    spectra();priors()
