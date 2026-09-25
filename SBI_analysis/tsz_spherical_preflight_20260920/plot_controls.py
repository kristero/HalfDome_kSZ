"""CLASS-SZ overlays and exact-likelihood comparison for the SBI control."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from plot_results import ROOT,STUDY,bins,COV,save


def inference():
    d=json.loads((ROOT/'results/inference_control.json').read_text())
    template=bins(np.load(STUDY/'maps/Battaglia12/pixel16384_2/masked_clean_cl.npy'))
    cov=COV['covariance'];weight=np.linalg.solve(cov,2*template/18.1)
    weight/=np.sqrt(weight@cov@weight);scale=weight@template
    grids=np.linspace(1,60,60001);records=[]
    for t,learned in zip(d['truth'],d['controls'][0]['predictions']):
        likelihood=np.exp(-.5*(scale*(grids/18.1)**2-scale*(t/18.1)**2)**2)
        cdf=np.r_[0,cumulative_trapezoid(likelihood,grids)];cdf/=cdf[-1]
        lo,hi=np.interp([.16,.84],cdf,grids)
        records.append(dict(truth=t,exact_q16=float(lo),exact_q84=float(hi),
            learned_width_over_exact=float((learned['q84']-learned['q16'])/(hi-lo))))
    (ROOT/'results/inference_width_check.json').write_text(json.dumps(records,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,6))
    truth=np.array(d['truth']);ax.plot([1,60],[1,60],color='.5',ls=':')
    for control,color,label in zip(d['controls'],['#0072B2','#D55E00'],['Paired training','Shuffled training']):
        means=np.array([r['mean'] for r in control['predictions']])
        errors=np.array([[r['mean']-r['q16'],r['q84']-r['mean']] for r in control['predictions']]).T
        ax.errorbar(truth,means,yerr=errors,fmt='o',capsize=4,color=color,label=label)
    ax.set(xlabel=r'True $P_0$',ylabel=r'Posterior $P_0$',xlim=(0,60),ylim=(0,60))
    ax.legend();ax.grid(alpha=.15);save(fig,'sbi_amplitude_control')
    print(json.dumps(records,indent=2))


def classsz():
    p=ROOT/'results/classsz_pressure_profiles.npz'
    if not p.exists():return
    data=np.load(p);meta=json.loads(str(data['metadata_json']))
    fig,ax=plt.subplots(figsize=(8,6))
    for row,profile in zip(meta,data['profiles']):
        if row['mass_msun']!=1e14 or row['z']!=.5:continue
        line,=ax.loglog(profile[:,0],profile[:,2],label=row['case'])
        ax.loglog(profile[::8,0],profile[::8,1],'o',mfc='none',color=line.get_color())
    ax.set(xlabel=r'$r/R_{200c}$',ylabel=r'$P/P_{200c}$',title=r'$M_{200c}=10^{14}M_\odot,\ z=0.5$')
    ax.legend();ax.grid(alpha=.15);save(fig,'classsz_pressure_comparison')
    sphere=ROOT/'fullsky/Battaglia12/nside4096_grid1/unmasked_clean_cl.npy'
    if sphere.exists():
        halo=np.load(ROOT/'results/classsz_sphere4_h068.npz');ell=halo['ell']
        sigma=np.deg2rad(2/60)/np.sqrt(8*np.log(2));beam=np.exp(-ell*(ell+1)*sigma*sigma)
        fig,ax=plt.subplots(figsize=(8,6))
        ax.loglog(COV['ell'],bins(np.load(sphere)),label='HalfDome spherical 4096',color='black')
        ax.loglog(ell,(halo['dl_1h']+halo['dl_2h'])*beam,label='CLASS-SZ 1h + 2h',color='#0072B2')
        ax.set(xlabel=r'$\ell$',ylabel=r'$D_\ell^{yy}$',xlim=(80,7980))
        ax.legend();ax.grid(alpha=.15);save(fig,'classsz_spectrum_comparison')


if __name__=='__main__':inference();classsz()
