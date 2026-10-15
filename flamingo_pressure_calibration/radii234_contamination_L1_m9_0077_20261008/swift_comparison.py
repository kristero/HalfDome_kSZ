"""Matched spatial cutout profiles and coefficient checks, using saved audits."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pilot_physics as P
from fit_study import fit_one,plain,accepted

HERE=Path(__file__).resolve().parent


def main():
    h=dict(np.load(HERE/'selected_halos.npz'))
    sw=dict(np.load(HERE/'swiftsimio_profiles.npz'));rows=sw['halo_rows']
    with np.load(HERE/'measurements.npz') as d:full=d['sums'][rows].sum(axis=1)
    subset={k:v[rows] for k,v in h.items() if isinstance(v,np.ndarray) and len(v)==len(h['halo_index'])}
    fb=json.loads((HERE/'metadata_plan.json').read_text())['baryon_fraction']
    profiles=[]
    for sums in (full,sw['sums']):
        profiles.append(P.physical_profiles({name:sums[:,i] for i,name in enumerate(P.SUM_NAMES)},subset,fb))
    direct,swift=profiles
    cases=json.loads((HERE/'products/study_summary.json').read_text())['bins'];receipts=[]
    x=(P.EDGES[:-1]+P.EDGES[1:])/2
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,(field,family,key,ylabel) in enumerate([
            ('pressure','b12','pressure_ratio',r'$P_e/P_{200}$'),
            ('electrons','b16','tau_density_ratio',r'$\rho_\tau/(f_b\rho_c)$')]):
        for b,case in enumerate(cases):
            take=subset['mass_bin']==b
            assert take.sum()==3
            a=direct[key][take];c=swift[key][take]
            am=a.mean(axis=0);cm=c.mean(axis=0)
            valid=(am>0)&(x>=.04)
            axes[1,col].plot(x[valid],1e6*(cm[valid]/am[valid]-1),color=plt.cm.viridis(b/14),lw=1)
            for ri,entry in enumerate(case['selections']['all'][field]):
                if entry['fit'] is None:continue
                mask=np.array(entry['fit_mask'],dtype=bool);start=entry['fit']['parameters']
                fits=[fit_one(family,v.mean(axis=0),v.std(axis=0,ddof=1),mask,start=start) for v in (a,c)]
                if any(v is None for v in fits):continue
                delta=np.array(fits[1]['parameters'])/np.array(fits[0]['parameters'])-1
                receipts.append({'bin':case['label'],'field':field,'R':ri+2,'matched_halos':3,
                    'direct_fit':fits[0],'SWIFT_fit':fits[1],'fractional_coefficient_difference':delta,
                    'both_numerically_stable':accepted(fits[0]) and accepted(fits[1]),
                    'population_strict_resolution_supported':case['strict_resolution_supported']})
            if b==10:
                stats=dict(np.load(HERE/'products'/(case['label']+'_profiles.npz')))
                axes[0,col].plot(x,stats[f'all_{field}_mean'],color='0.65',lw=2,label=f'full selected stack (N={case["halos"]})')
                axes[0,col].plot(x,am,color='black',lw=2,label='direct, matched N=3')
                axes[0,col].plot(x,cm,color='#0072B2',lw=2,ls='--',label='SWIFT, same N=3')
                axes[0,col].set_title(field+': 14.5-14.7')
        axes[0,col].set_yscale('log');axes[0,col].set_ylabel(ylabel);axes[0,col].legend(fontsize=10)
        axes[1,col].axhline(0,color='0.7',lw=.7);axes[1,col].set_ylabel('SWIFT / matched direct - 1 (ppm)')
        axes[1,col].set_yscale('symlog',linthresh=1e-6)
        for ax in axes[:,col]:ax.set_xscale('log');ax.set_xlim(.04,4.1);ax.set_xlabel(r'$r/R_{200c}$')
    color=plt.cm.ScalarMappable(norm=Normalize(12.6,15.4),cmap='viridis')
    fig.colorbar(color,ax=axes[1,:],label=r'$\log_{10}(M_{200c}/M_\odot)$',shrink=.85)
    for suffix in ('png','svg'):fig.savefig(HERE/'figures'/('SWIFT_matched_comparison.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)
    result={'matched_halos_per_bin':3,'fit_comparisons':receipts,
            'scope':'The same particles loaded independently; the three representatives are not the full selected population.',
            'max_absolute_fractional_coefficient_difference':float(np.max(np.abs([r['fractional_coefficient_difference'] for r in receipts]))) if receipts else None}
    stable=[r['fractional_coefficient_difference'] for r in receipts if r['both_numerically_stable'] and r['population_strict_resolution_supported']]
    result['max_stable_resolved_fractional_coefficient_difference']=float(np.max(np.abs(stable))) if stable else None
    (HERE/'products/SWIFT_matched_fit_comparison.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    print(json.dumps({'SWIFT_matched_fit_comparisons':len(receipts),'maximum_parameter_difference':result['max_absolute_fractional_coefficient_difference']}),flush=True)


if __name__=='__main__':main()
