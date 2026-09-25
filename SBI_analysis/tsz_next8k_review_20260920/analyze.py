"""Review saved, matched cluster experiments without altering any simulation.

The covariance is conditional on one B12 sky and the existing idealized noise
prescription. No results below are a new full-sky benchmark or SO forecast.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
SBI = ROOT.parent
REPO = SBI.parent
STUDY = SBI/'tsz_guardrail_study'
EDGES = np.append(np.arange(80, 7881, 200), 7980)
ELL = np.arange(7980)
LABELS = ['P0','xc','beta','alpha_m_P0','alpha_m_xc','alpha_m_beta',
          'alpha_z_P0','alpha_z_xc','alpha_z_beta']


def save(name, value):
    (ROOT/'results'/name).write_text(json.dumps(value,indent=2)+'\n')


def bin_cl(cl):
    dl = cl[:7980]*ELL*(ELL+1)/(2*np.pi)
    return np.array([np.average(dl[a:b],weights=2*ELL[a:b]+1)
                     for a,b in zip(EDGES[:-1],EDGES[1:])])


def figure(fig, name):
    for suffix in ['png','pdf']:
        fig.savefig(ROOT/'plots'/(name+'.'+suffix),bbox_inches='tight',dpi=180)
    plt.close(fig)


def norm(vector, covariance):
    return float(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(covariance), vector)))


def beam_and_resolution():
    bundle=np.load(STUDY/'audit/noise_covariance.npz')
    covariance=bundle['covariance']; sigma=np.sqrt(np.diag(covariance)); ell=bundle['ell']
    beam_sigma=np.deg2rad(2/60)/np.sqrt(8*np.log(2))
    beam_power=np.exp(-ELL*(ELL+1)*beam_sigma**2)
    noise_table=np.loadtxt(REPO/'other_sims/SO/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt')
    noise_cl=np.interp(ELL,noise_table[:,0],noise_table[:,1])
    noise_dl=bin_cl(noise_cl)
    result={'beam_fwhm_arcmin':2.,'beam_power':{str(l):float(beam_power[l]) for l in [1000,2000,3000,4000,5000,6000,7000,7979]},
            'covariance_scope':'64 independent split-noise realizations, fixed B12 sky, original beam/mask; excludes cosmic variance and model discrepancy',
            'cases':{}}
    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    axs[0,0].semilogy(ELL[80:],beam_power[80:],color='black')
    axs[0,0].set(ylabel=r'$B_\ell^2$',xlabel=r'$\ell$',title="2 arcmin beam")
    axs[0,1].loglog(ell,noise_dl,'--',color='.4',label=r'Tabulated $N_\ell$')
    output_rows=[]
    for case,label,color in [('Battaglia12','Battaglia12','black'),('FL_L1_m9','FLAMINGO fit','#0072B2')]:
        variants={name:bin_cl(np.load(STUDY/'maps'/case/name/'masked_clean_cl.npy'))
                  for name in ['logz2','pixel8192_2','pixel16384_2']}
        signal=variants['pixel16384_2']
        delta=variants['logz2']-signal
        refined=variants['pixel8192_2']-signal
        unmasked=bin_cl(np.load(STUDY/'maps'/case/'pixel16384_2/unmasked_clean_cl.npy'))
        cumulative=np.array([norm(delta[:k],covariance[:k,:k]) for k in range(1,41)])
        cumulative_refined=np.array([norm(refined[:k],covariance[:k,:k]) for k in range(1,41)])
        information=np.array([norm(2*signal[:k],covariance[:k,:k])**2 for k in range(1,41)])
        cuts=[]
        for k in [5,10,20,30,40]:
            cuts.append(dict(ell_max=int(EDGES[k]-1),error4096=float(cumulative[k-1]),
                error8192=float(cumulative_refined[k-1]),amplitude_information_fraction=float(information[k-1]/information[-1])))
        result['cases'][case]=dict(cuts=cuts,last_bin_fractional_error=float(delta[-1]/signal[-1]),
            last_bin_error_over_sigma=float(delta[-1]/sigma[-1]),
            maximum_error_over_sigma=float(np.max(abs(delta)/sigma)),
            max_error_bin_ell=float(ell[np.argmax(abs(delta)/sigma)]),
            signal_to_noise_power_at_ell4000=float(unmasked[19]/noise_dl[19]),
            signal_to_noise_power_last_bin=float(unmasked[-1]/noise_dl[-1]),
            amplitude_information_scope='Only ln(P0), other parameters fixed, existing fixed B12 covariance; not nine-parameter information retention')
        axs[0,1].loglog(ell,unmasked,color=color,label=label)
        axs[1,0].plot(ell,delta/sigma,color=color,label=label+' 4096')
        axs[1,0].plot(ell,refined/sigma,color=color,ls='--',label=label+' 8192')
        short_label = 'B12' if case == 'Battaglia12' else 'FL fit'
        axs[1,1].plot(EDGES[1:]-1,cumulative,color=color,label=short_label+' 4096')
        axs[1,1].plot(EDGES[1:]-1,cumulative_refined,color=color,ls='--',label=short_label+' 8192')
        for i in range(40):
            output_rows.append(dict(case=case,ell=float(ell[i]),ell_upper=int(EDGES[i+1]-1),
                signal_masked=float(signal[i]),noise_sigma=float(sigma[i]),delta4096=float(delta[i]),
                delta8192=float(refined[i]),cumulative_error4096=float(cumulative[i]),
                cumulative_error8192=float(cumulative_refined[i]),
                amplitude_information_fraction=float(information[i]/information[-1])))
    axs[0,1].set(xlabel=r'$\ell$',ylabel=r'$D_\ell^{yy}$',title='Unmasked signal and noise power')
    axs[0,1].set_xticks([200,1000,4000,8000])
    axs[0,1].xaxis.set_major_formatter(ScalarFormatter())
    axs[0,1].xaxis.set_minor_formatter(NullFormatter())
    axs[0,1].legend(fontsize=12)
    axs[1,0].set(xlabel=r'$\ell$',ylabel=r'$\Delta D_b/\sigma_b$',title='After beam and mask')
    axs[1,0].axhline(0,color='.6',lw=.8)
    axs[1,1].set(xlabel=r'Maximum $\ell$',ylabel=r'$\sqrt{\Delta D^T C^{-1}\Delta D}$',title='Cumulative numerical discrepancy')
    axs[1,1].axhline(.1,color='#D55E00',ls=':',label='0.1 budget')
    axs[1,1].legend(fontsize=12)
    for ax in [axs[0,0], axs[1,0], axs[1,1]]:
        ax.set_xticks([0,2000,4000,6000,8000])
    for ax in axs.flat:ax.grid(alpha=.15)
    fig.tight_layout();figure(fig,'beam_noise_resolution')
    save('beam_noise_resolution.json',result)
    with (ROOT/'results/bandpower_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output_rows[0]));writer.writeheader();writer.writerows(output_rows)
    return result


def costs():
    results=[]
    for case in ['Battaglia12','FL_L1_m9']:
        rows=[]
        for variant,nside,grid in [('historical1',4096,1),('logz2',4096,2),('pixel8192_2',8192,2),('pixel16384_2',16384,2)]:
            p=STUDY/'maps'/case/variant
            status=json.loads((p/'status.json').read_text())
            rss=float(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',(p/'time.txt').read_text()).group(1))/2**20
            cache=float(re.search(r'END\s+profile_grid threaded evaluation wall=([\d.]+)',(p/'run.log').read_text()).group(1))
            rows.append(dict(variant=variant,nside=nside,grid_refinement=grid,seconds=status['seconds'],cache_seconds=cache,rss_gib=rss))
        results.append(dict(case=case,rows=rows,matched_grid_time_ratio=rows[2]['seconds']/rows[1]['seconds'],
            matched_grid_memory_ratio=rows[2]['rss_gib']/rows[1]['rss_gib']))
    old=json.loads((SBI/'flamingo_linear_prior/cluster_results/completed_20260918/summary.json').read_text())
    save('costs.json',dict(clean_eight_thread_experiments=results,completed_production=old,
        caveat='New spherical/prebeam/noise pipeline has not been benchmarked; do not scale 26-thread production using these ratios as a measured forecast'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    x=np.arange(4)
    for j,record in enumerate(results):
        axes[0].bar(x+(.16 if j else -.16),[r['seconds']/60 for r in record['rows']],width=.3,
                    color=['#555555','#0072B2'][j],label=['Battaglia12','FLAMINGO fit'][j])
        axes[1].bar(x+(.16 if j else -.16),[r['rss_gib'] for r in record['rows']],width=.3,color=['#555555','#0072B2'][j])
    for ax in axes:
        ax.set_xticks(x);ax.set_xticklabels(['4096\nold grid','4096\nfine grid','8192\nfine grid','16384\nfine grid'],fontsize=13)
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Wall time [minutes]');axes[1].set_ylabel('Peak memory [GiB]')
    axes[0].legend(fontsize=12)
    fig.tight_layout();figure(fig,'resolution_costs')
    return results


def prior_baseline():
    base=SBI/'flamingo_linear_prior/cluster_results/completed_20260918'
    theta=np.load(base/'dataset/theta.npy');test=np.load(base/'dataset/validation_split.npy').astype(bool)
    manifest=json.loads((base/'manifest.json').read_text());low=np.array(manifest['prior']['lower']);high=np.array(manifest['prior']['upper']);width=high-low
    truth=theta[test];prediction=np.broadcast_to(theta[~test].mean(0),truth.shape)
    rmse=np.sqrt(np.mean(((prediction-truth)/width)**2,axis=0))
    pooled=float(np.corrcoef(((truth-low)/width).ravel(),((prediction-low)/width).ravel())[0,1])
    with (SBI/'linear_prior_sbi/cluster_results/figures/convergence_metrics.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['N']=='7349' and r['estimator']=='maf']
    record=dict(test_count=int(test.sum()),training_count=int((~test).sum()),
        baseline='Constant training-prior mean; no observation dependence',
        normalized_rmse=dict(zip(LABELS,rmse.tolist())),mean_normalized_rmse=float(rmse.mean()),
        pooled_pearson=pooled,per_parameter_pearson='undefined: predictions are constant',trained=rows)
    save('prior_only_baseline.json',record)
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(9);ax.plot(x,rmse,'o-',color='black',label='Prior-mean predictor')
    for row,color in zip(sorted(rows,key=lambda r:r['method']),['#0072B2','#D55E00','#009E73']):
        label = {'bins40':'40 bins', 'moped':'MOPED', 'pca':'PCA'}[row['method']]
        ax.plot(x,[float(row['rmse_'+name]) for name in LABELS],'o-',color=color,label=label)
    ax.set_xticks(x);ax.set_xticklabels([r'$P_0$',r'$x_c$',r'$\beta_0$',r'$\alpha_{m,P_0}$',r'$\alpha_{m,x_c}$',r'$\alpha_{m,\beta}$',r'$\alpha_{z,P_0}$',r'$\alpha_{z,x_c}$',r'$\alpha_{z,\beta}$'],fontsize=15)
    ax.set_ylabel('RMSE / parameter range');ax.legend(fontsize=13,ncol=2);ax.grid(alpha=.15)
    fig.tight_layout();figure(fig,'prior_only_baseline')
    return record


def snapshot_sources():
    sha='5dd0b57cae243598cef9608de77f6807689db712'
    snapshots={}
    for name in ['profiles_y.jl','profiles.jl']:
        url='https://raw.githubusercontent.com/kristero/XGPaint.jl/'+sha+'/src/'+name
        blob=urllib.request.urlopen(url,timeout=30).read()
        (ROOT/'inputs'/('upstream_'+name)).write_bytes(blob)
        snapshots[name]=dict(url=url,sha256=hashlib.sha256(blob).hexdigest())
    for relative in ['truncation_comparison/spherical_truncation_profiles.jl',
                     'SBI_analysis/flamingo_linear_prior/stable_los.jl',
                     'SBI_analysis/tsz_guardrail_study/map_experiment.jl',
                     'SBI_analysis/tsz_beta_flat_followup_20260920/continuous_flat.py']:
        p=REPO/relative;snapshots[relative]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save('source_review.json',dict(halfdome_commit='10947ce33efadbe8431af54d5fb9b92fe6649fc5',
        spherical_fix_commit='3522b71',public_xgpaint_cluster_commit=sha,
        user_confirmed_halfdome_wrapper=True,cluster_ssh='Two attempts timed out; this review uses fetched Git and preserved cluster artifacts',sources=snapshots))


if __name__=='__main__':
    for directory in ['inputs','results','plots']:(ROOT/directory).mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':15,'pdf.fonttype':42})
    beam=beam_and_resolution();cost=costs();baseline=prior_baseline();snapshot_sources()
    print(json.dumps({'beam':beam,'cost_ratios':[{k:r[k] for k in ['case','matched_grid_time_ratio','matched_grid_memory_ratio']} for r in cost],
        'prior_baseline_mean_rmse':baseline['mean_normalized_rmse'],'prior_baseline_pooled_r':baseline['pooled_pearson']},indent=2))
