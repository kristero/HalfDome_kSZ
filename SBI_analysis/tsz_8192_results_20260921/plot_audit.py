"""Publication-size figures from original-code probes and resolution controls."""
import json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:import tomllib
except ImportError:import toml as tomllib

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'tsz_spherical_preflight_20260920'
ELL=np.arange(80,7980);FACTOR=ELL*(ELL+1)/(2*np.pi)

def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,list):return [safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value

def save(fig,name):
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(ROOT/'plots'/f'{name}.{suffix}',dpi=180)
    plt.close(fig)

def main():
    summary={p.stem:tomllib.loads(p.read_text()) for p in (ROOT/'results').glob('stock_*.toml')}
    (ROOT/'results/stock_summary.json').write_text(json.dumps(safe(summary),indent=2)+'\n')
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'legend.fontsize':11,'axes.titlesize':15})
    curves=np.loadtxt(ROOT/'results/stock_profile_curves.csv',delimiter=',')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    labels=['Battaglia12 shape','Compact pivot shape','Steep evolved shape','Shallow evolved shape']
    for i,ax in enumerate(axes.ravel(),1):
        data=curves[curves[:,0]==i];x=data[:,1]
        ax.loglog(x,np.maximum(data[:,2],1e-300),label='Sphere, stable',color='#0072B2')
        ax.loglog(x,np.maximum(data[:,3],1e-300),'--',label='Cylinder, stable',color='#D55E00')
        nonzero=data[:,4]>0
        ax.loglog(x[nonzero][::8],data[:,4][nonzero][::8],'o',mfc='none',color='#222222',label='Original LOS')
        if not nonzero.any():ax.text(.97,.92,'Original LOS: all zero',transform=ax.transAxes,ha='right',color='#B34030',fontsize=12)
        ax.set(title=labels[i-1]+r', $\beta$='+f'{data[0,5]:.3g}',xlabel=r'$x=b/R_{200c}$',ylabel=r'$J(x)$')
        ax.set_ylim(max(data[:,3].max()*1e-9,1e-15),data[:,3].max()*3)
        ax.grid(alpha=.2);ax.legend(loc='lower left')
    save(fig,'original_profile_comparison')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    groups=np.array_split(np.arange(len(ELL)),79);centres=np.array([ELL[g].mean() for g in groups])
    for j,anchor in enumerate(['Battaglia12','FL_L1_m9']):
        spectra={n:np.load(REFERENCE/'fullsky'/anchor/f'nside{n}_grid1/masked_clean_cl.npy')[ELL]*FACTOR for n in [4096,8192,16384]}
        reference=np.array([spectra[16384][g].mean() for g in groups])
        for n,color in zip([4096,8192,16384],['#D55E00','#0072B2','#009E73']):
            binned=np.array([spectra[n][g].mean() for g in groups])
            axes[0,j].semilogy(centres,binned,color=color,label=str(n)+(' ref.' if n==16384 else ''))
            if n!=16384:axes[1,j].plot(centres,100*(binned/reference-1),color=color)
        axes[0,j].set_title('Battaglia12' if j==0 else 'Historical FLAMINGO fit');axes[0,j].legend()
        axes[1,j].set_xlabel(r'$\ell$');axes[1,j].axhline(0,color='k',lw=.7)
    axes[0,0].set_ylabel(r'$D_\ell^{yy}$');axes[1,0].set_ylabel('Difference from 16384 [%]')
    for ax in axes.ravel():ax.grid(alpha=.2)
    save(fig,'resolution_reference')

if __name__=='__main__':main()
