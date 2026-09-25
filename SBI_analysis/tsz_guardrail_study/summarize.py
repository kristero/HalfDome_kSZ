"""Build publication figures and measured comparisons; never invent missing tests."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from analytic_prior import AnalyticPrior, B12, fidelity_distance
from audit import save

LABELS = [r'$P_{0,0}$', r'$x_{c,0}$', r'$\beta_0$', r'$\alpha_{m,P_0}$',
          r'$\alpha_{m,x_c}$', r'$\alpha_{m,\beta}$', r'$\alpha_{z,P_0}$',
          r'$\alpha_{z,x_c}$', r'$\alpha_{z,\beta}$']
COLORS = ['#0072B2', '#D55E00', '#009E73']
VARIANTS = ['L1_m9', 'fgas-8sigma', 'Mstar-1sigma']
CASE_LABELS = {'Battaglia12':'Battaglia12', 'FL_L1_m9':'FLAMINGO fiducial fit',
    'xc_low_1':r'$x_{c,0}=0.05$', 'combined_tails':'Combined tails',
    'amP0_low_2':r'$\alpha_{m,P_0}=-0.6$', 'amxc_low_2':r'$\alpha_{m,x_c}=-1$',
    'azP0_low_2':r'$\alpha_{z,P_0}=-6$', 'azxc_high_2':r'$\alpha_{z,x_c}=3$',
    'azbeta_high_2':r'$\alpha_{z,\beta}=2$', 'beta_steep':r'$\beta_0=128$ stress test'}
CASE_COLORS = {'Battaglia12':'black', 'FL_L1_m9':'#0072B2', 'xc_low_1':'#56B4E9',
    'combined_tails':'#7B3294', 'amP0_low_2':'#009E73', 'amxc_low_2':'#E69F00',
    'azP0_low_2':'#D55E00', 'azxc_high_2':'#CC79A7', 'azbeta_high_2':'#999999',
    'beta_steep':'#666666'}
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 19, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 13, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def figure_save(fig, root, name):
    (root/'plots').mkdir(exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(root/'plots'/(name+'.'+extension), dpi=220, bbox_inches='tight')
    plt.close(fig)


def bins(cl):
    ell = np.arange(7980)
    edges = np.append(np.arange(80, 7881, 200), 7980)
    dl = cl*ell*(ell+1)/(2*np.pi)
    return np.array([np.average(dl[a:b], weights=2*ell[a:b]+1) for a,b in zip(edges[:-1],edges[1:])])


def prior_plots(root):
    proposal = AnalyticPrior()
    config = json.loads((root/'code/baseline/prior.json' if (root/'code').exists()
                        else root/'baseline/prior.json').read_text())
    data = np.load(root/'audit/prior_samples.npz')
    production = np.load(root/'inputs/production_theta.npy')
    old = json.loads((root/'inputs/original_sbi_bounds.json').read_text())
    fits = json.loads((root/'inputs/flamingo_cosmology_refits.json').read_text())['corrected_parameters']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        edges = np.linspace(proposal.low[i], proposal.high[i], 35)
        ax.hist(production[:, i], bins=edges, density=True, color='.7', alpha=.8)
        ax.hist(data['analytic_extended'][:, i], bins=edges, density=True,
                histtype='step', color='#7B3294', lw=2)
        ax.axvspan(old['low'][i], old['high'][i], color='#E6AB02', alpha=.12)
        for value in (old['low'][i], old['high'][i]):
            ax.axvline(value, color='#A6761D', ls=':', lw=1.4)
        for value in (config['lower'][i], config['upper'][i]):
            ax.axvline(value, color='.25', ls='--', lw=1.2)
        ax.axvline(B12[i], color='black', lw=1.8)
        for j, variant in enumerate(VARIANTS):
            ax.axvline(fits[variant][i], color=COLORS[j], lw=1.6)
        ax.set_xlabel(LABELS[i]); ax.set_ylabel('Density')
        ax.set_xlim(proposal.low[i], proposal.high[i])
    handles = [Patch(color='.7', label='8k production design'),
        Line2D([],[],color='#7B3294',lw=2,label='Analytic candidate; fidelity pending'),
        Line2D([],[],color='.25',ls='--',label='8k rectangular bounds'),
        Patch(color='#E6AB02',alpha=.3,label='Original SBI bounds'),
        Line2D([],[],color='black',label='Battaglia12')]
    handles += [Line2D([],[],color=c,label='FLAMINGO '+v) for c,v in zip(COLORS,VARIANTS)]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5,1.00))
    fig.tight_layout(rect=(0,0,1,.90))
    figure_save(fig, root, 'prior_comparison_all_parameters')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for ax, (i,j) in zip(axes, [(1,2),(4,7),(5,8)]):
        sample = data['analytic_extended'][::8]
        ax.scatter(sample[:,i],sample[:,j],s=2,color='#7B3294',alpha=.08,rasterized=True)
        ax.scatter(production[:,i],production[:,j],s=2,color='.25',alpha=.15,rasterized=True)
        ax.scatter(B12[i],B12[j],marker='*',s=180,c='black',zorder=4)
        for color, variant in zip(COLORS,VARIANTS):
            ax.scatter(fits[variant][i],fits[variant][j],s=65,color=color,edgecolor='white',zorder=4)
        ax.set_xlabel(LABELS[i]); ax.set_ylabel(LABELS[j])
        ax.set_xlim(proposal.low[i],proposal.high[i]); ax.set_ylim(proposal.low[j],proposal.high[j])
    fig.tight_layout()
    figure_save(fig,root,'joint_support_comparison')


def quadrature_plot(root):
    all_data = json.loads((root/'audit/quadrature.json').read_text())
    data = [row for row in all_data if not row.get('failed')]
    fig, axes = plt.subplots(1,2,figsize=(12,4.7))
    beta=np.array([x['beta'] for x in data]); neval=np.array([x['evaluations'] for x in data])
    radius=np.array([x['radius'] for x in data]); changes=np.array([x['tightened_relative_change'] for x in data])
    axes[0].scatter(beta,neval,s=20,alpha=.4,color='#0072B2')
    axes[0].axvline(50,color='black',ls='--',label='Previous upper cut')
    axes[0].set(xscale='log',xlabel=r'$\beta(M,z)$',ylabel='Quadrature evaluations')
    axes[0].legend()
    tail=np.array([x['doubled_endpoint_relative_change'] for x in data])
    axes[1].scatter(radius,np.maximum(tail,1e-16),s=20,alpha=.35,color='#D55E00')
    axes[1].axvline(4,color='black',ls='--',label=r'Painting radius: $4R_{200}$')
    axes[1].set(xscale='log',yscale='log',xlabel=r'$R/R_{200}$',ylabel=r'$|y_{2L}/y_L-1|$')
    axes[1].legend()
    fig.tight_layout();figure_save(fig,root,'quadrature_and_los')
    save(root/'audit/quadrature_summary.json',dict(tests=len(all_data),failed=len(all_data)-len(data),max_beta=float(beta.max()),
        max_evaluations=int(neval.max()),max_tolerance_change=float(changes.max()),
        max_endpoint_change_inside_painting=float(tail[radius<=4].max()),
        max_endpoint_change_all=float(tail.max()),
        underflows=sum(x['underflows_float64'] for x in data)))


def native_quadrature_audit(root):
    """Compare unique historical-case native probes to the normalized solver.

    A small native error estimate is not proof of correctness: direct adaptive
    integration can miss a narrow positive peak. Preserve that distinction.
    """
    groups=[]
    for directory in sorted((root/'maps').glob('*')):
        paths=[directory/grid/'columns.csv' for grid in ('historical1','logz1','logz2','pixel8192_2','pixel16384_2')]
        path=next((p for p in paths if p.exists()),None)
        if path is None:continue
        with path.open() as stream:
            rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        positive=[d for d in rows if d['stable']>0]
        nominal=[d for d in positive if np.isfinite(d['native_error_ratio'])
                 and d['native_error_ratio']<=5e-10]
        differences=[d['relative_difference'] for d in nominal if np.isfinite(d['relative_difference'])]
        groups.append(dict(case=path.parent.parent.name,source=str(path.relative_to(root)),probes=len(rows),
            positive_stable_columns=len(positive),nominal_native_convergence=len(nominal),
            native_positive_peak_missed=sum(d['native']==0 for d in positive),
            nominal_native_disagrees_over_1e_minus_7=sum(x>1e-7 for x in differences),
            max_relative_disagreement_nominal=max(differences) if differences else None,
            max_native_evaluations=max(d['native_evaluations'] for d in rows)))
    save(root/'audit/native_quadrature_comparison.json',dict(cases=groups,
        interpretation='One set per unique case, excluding repeated grid refinements. The bounded native route is diagnostic, not ground truth; normalized integration was separately checked with QUADPACK.'))


def map_comparisons(root):
    noise=np.load(root/'audit/noise_covariance.npz')
    covariance = noise['covariance']
    ell = noise['ell']
    def bootstrap_interval(a,b):
        if 'bootstrap_whiteners' not in noise:return None
        errors=np.linalg.norm(np.einsum('bij,j->bi',noise['bootstrap_whiteners'],a-b),axis=1)
        return np.percentile(errors,[2.5,50,97.5]).tolist()
    records = []
    expected_cases=json.loads((root/'inputs/cases.json').read_text())
    complete = []
    for directory in sorted((root/'maps').glob('*')):
        spectra = {}
        timing = {}
        variants=['historical1','logz1','logz2']+sorted(p.name for p in directory.glob('pixel*') if p.is_dir())
        for variant in variants:
            p = directory/variant
            if not (p/'status.json').exists(): continue
            status = json.loads((p/'status.json').read_text())
            if status['exit_code'] != 0: continue
            np.testing.assert_array_equal(status['theta'],expected_cases[directory.name])
            spectra[variant] = bins(np.load(p/'masked_clean_cl.npy'))
            time_text=(p/'time.txt').read_text() if (p/'time.txt').exists() else ''
            rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',time_text)
            timing[variant]=dict(seconds=status['seconds'],peak_rss_gib=int(rss[1])/1024**2 if rss else None)
        record=dict(case=directory.name,completed=list(spectra),timing=timing)
        if 'historical1' in spectra and 'logz1' in spectra:
            record['historical_to_logz1_sigma']=fidelity_distance(spectra['historical1'],spectra['logz1'],covariance)
            record['historical_to_logz1_max_relative']=float(np.max(abs(spectra['historical1']/spectra['logz1']-1)))
        if all(key in spectra for key in ('historical1','logz1','logz2')):
            record['historical_to_logz2_sigma']=fidelity_distance(spectra['historical1'],spectra['logz2'],covariance)
            record['historical_to_logz2_covariance_bootstrap_interval']=bootstrap_interval(spectra['historical1'],spectra['logz2'])
            record['logz1_to_logz2_sigma']=fidelity_distance(spectra['logz1'],spectra['logz2'],covariance)
            record['historical_max_relative']=float(np.max(abs(spectra['historical1']/spectra['logz2']-1)))
            record['logz1_max_relative']=float(np.max(abs(spectra['logz1']/spectra['logz2']-1)))
            record['historical_interpolation_budget_pass']={str(b):record['historical_to_logz2_sigma']+record['logz1_to_logz2_sigma']<=b for b in (.05,.1,.3)}
            complete.append((directory.name,spectra))
        if 'logz2' in spectra:
            for variant in spectra:
                if variant.startswith('pixel'):
                    record[variant+'_from_4096_sigma']=fidelity_distance(spectra['logz2'],spectra[variant],covariance)
                    record[variant+'_covariance_bootstrap_interval']=bootstrap_interval(spectra['logz2'],spectra[variant])
                    record[variant+'_from_4096_max_relative']=float(np.max(abs(spectra['logz2']/spectra[variant]-1)))
                    fractional=spectra['logz2']/spectra[variant]-1
                    record[variant+'_max_relative_ell']=float(ell[np.argmax(abs(fractional))])
                    # An identifiable one-amplitude example, conditional on the
                    # other eight parameters and the same fixed covariance.
                    lower=np.linalg.cholesky(covariance)
                    coarse=np.linalg.solve(lower,spectra['logz2'])
                    finer=np.linalg.solve(lower,spectra[variant])
                    scale=float(coarse@finer/(coarse@coarse))
                    record[variant+'_amplitude_only_fit']=dict(
                        power_amplitude=scale,P0_ratio=float(np.sqrt(scale)) if scale>0 else None,
                        signed_power_amplitude_shift_sigma=float((scale-1)*np.linalg.norm(coarse)),
                        residual_sigma=float(np.linalg.norm(scale*coarse-finer)),
                        interpretation='Treat finer sampling as synthetic data and fit only P0 with the coarse template; not a nine-parameter posterior bias or proof of a continuum reference.')
            if 'pixel16384_2' in spectra and 'pixel8192_2' in spectra:
                record['pixel16384_2_from_8192_sigma']=fidelity_distance(spectra['pixel8192_2'],spectra['pixel16384_2'],covariance)
                record['pixel16384_2_from_8192_max_relative']=float(np.max(abs(spectra['pixel8192_2']/spectra['pixel16384_2']-1)))
        records.append(record)
    save(root/'audit/map_comparison.json',dict(cases=records,
        covariance='64-realization conditional B12 SO-noise reference, not each candidate posterior uncertainty',
        interpretation='Paired interpolation and pixel-sampling tests. Same continuous pressure and analysis operator. Two sampling levels do not establish a continuum limit; full-observable LOS convergence remains untested.'))
    if complete:
        fig,axes=plt.subplots(1,2,figsize=(15,6.8))
        for name, spectra in complete:
            axes[0].plot(ell,(spectra['historical1']/spectra['logz2']-1)*100,label=CASE_LABELS.get(name,name),color=CASE_COLORS.get(name))
            axes[1].plot(ell,(spectra['logz1']/spectra['logz2']-1)*100,label=CASE_LABELS.get(name,name),color=CASE_COLORS.get(name))
        for ax in axes:
            ax.axhline(0,color='.4',lw=.7);ax.set_xlabel(r'$\ell$');ax.set_ylabel(r'$\Delta D_\ell/D_\ell$ [%]')
        axes[0].set_title('Historical grid');axes[1].set_title(r'$\log z$ grid')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=5,fontsize=13)
        fig.tight_layout(rect=(0,0,1,.83));figure_save(fig,root,'full_map_interpolation_convergence')
        fig,ax=plt.subplots(figsize=(12,6.5))
        for name,spectra in complete:
            ax.loglog(ell,spectra['logz2'],label=CASE_LABELS.get(name,name),color=CASE_COLORS.get(name),
                      lw=2.5 if name in ('Battaglia12','FL_L1_m9') else 1.8,
                      ls='--' if name=='beta_steep' else '-')
        ax.set_xticks([200,500,1000,2000,5000])
        ax.set_xticklabels(['200','500','1000','2000','5000'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r'$\ell$');ax.set_ylabel(r'$D_\ell^{yy}$');ax.legend(fontsize=14,loc='center left',bbox_to_anchor=(1, .5))
        fig.tight_layout();figure_save(fig,root,'clean_tsz_examples')
        pixel_complete=[(name,spectra) for name,spectra in complete if 'pixel8192_2' in spectra]
        if pixel_complete:
            fig,ax=plt.subplots(figsize=(8,5.5))
            for name,spectra in pixel_complete:
                ax.plot(ell,(spectra['logz2']/spectra['pixel8192_2']-1)*100,label=CASE_LABELS.get(name,name),color=CASE_COLORS.get(name))
            ax.axhline(0,color='.4',lw=.7)
            ax.set(xlabel=r'$\ell$',ylabel=r'$D_\ell^{4096}/D_\ell^{8192}-1$ [%]')
            ax.legend();fig.tight_layout();figure_save(fig,root,'full_map_pixel_convergence')
        refined=[(name,spectra) for name,spectra in complete if 'pixel16384_2' in spectra]
        if refined:
            fig,axes=plt.subplots(1,len(refined),figsize=(7*len(refined),5.5),squeeze=False)
            for ax,(name,spectra) in zip(axes.flat,refined):
                for key,label in [('logz2','4096 / 16384'),('pixel8192_2','8192 / 16384')]:
                    ax.plot(ell,(spectra[key]/spectra['pixel16384_2']-1)*100,label=label)
                ax.axhline(0,color='.4',lw=.7)
                ax.set(xlabel=r'$\ell$',ylabel=r'$\Delta D_\ell/D_\ell$ [%]',title=CASE_LABELS.get(name,name))
                ax.legend()
            fig.tight_layout();figure_save(fig,root,'full_map_pixel_refinement')


def pixel_plot(root):
    path=root/'audit/pixel_summary.json'
    if not path.exists():return
    d=json.loads(path.read_text())
    names=['Battaglia12','FL_L1_m9','xc_low_1','combined_tails','azbeta_high_2','beta_steep']
    selected=[x for x in d['rows'] if x['case'] in names and x['mass']==1e14]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for index,nside in enumerate((4096,8192)):
        for row in selected:
            if row['nside']!=nside:continue
            axes[index].errorbar(d['ell'][1:],row['mean_power'][1:],yerr=row['power_mc_standard_error'][1:],
                label=row['case'],marker='.',capsize=2)
        axes[index].axhline(1,color='black',lw=1,ls='--')
        axes[index].set(xlabel=r'$\ell$',ylabel=r'$\langle|\tilde y_{\rm pix}|^2\rangle/|\tilde y_{\rm ref}|^2$',
                        yscale='log',title=r'$N_{\rm side}='+str(nside)+'$')
    axes[0].legend(fontsize=11);fig.tight_layout();figure_save(fig,root,'pixel_power_sampling')
    if all('effective_pixel_count' in row for row in d['rows']):
        fig,ax=plt.subplots(figsize=(8,5.5))
        for nside,color in zip((4096,8192),COLORS):
            group=[row for row in d['rows'] if row['nside']==nside]
            ax.errorbar([row['effective_pixel_count'] for row in group],
                [row['mean_power'][0] for row in group],
                yerr=[row['power_mc_standard_error'][0] for row in group],
                marker='o',ls='none',alpha=.65,color=color,label=r'$N_{\rm side}='+str(nside)+'$')
        x=np.geomspace(min(row['effective_pixel_count'] for row in d['rows'])/2,
                      max(row['effective_pixel_count'] for row in d['rows'])*2,200)
        ax.plot(x,np.maximum(1,1/x),color='black',ls='--',label='Exact ensemble lower bound')
        ax.set(xscale='log',yscale='log',xlabel=r'$N_{\rm eff}=A_{\rm eff}/\Omega_{\rm pix}$',
               ylabel=r'$\langle F_{\rm pix}^2\rangle/F_{\rm ref}^2$')
        ax.legend();fig.tight_layout();figure_save(fig,root,'effective_area_bound')


def mean_y_plot(root):
    path=root/'audit/mean_y_prior_samples.npz'
    if not path.exists():return
    data=np.load(path)
    fig,ax=plt.subplots(figsize=(9,5.5))
    statistics={}
    all_values=np.r_[data['production_mean_y_lower'],data['analytic_candidate_mean_y_lower']]
    edges=np.linspace(np.floor(np.log10(all_values.min())),np.ceil(np.log10(all_values.max())),91)
    for label,color,title in [('production','.55','8k production design'),
                              ('analytic_candidate','#7B3294','Analytic candidate')]:
        values=data[label+'_mean_y_lower']
        ax.hist(np.log10(values),bins=edges,density=True,
                histtype='step',lw=2,color=color,label=title)
        statistics[label]=dict(count=len(values),lower_bound_above_5p2e6=int(np.sum(values>5.2e-6)),
            lower_bound_above_15e6=int(np.sum(values>15e-6)),
            range=[float(values.min()),float(values.max())])
    for value,color,label in [(5.2e-6,'#D55E00','Fabbian et al.: 95% limit'),
                              (15e-6,'black','Fixsen et al.: 95% limit')]:
        ax.axvline(np.log10(value),color=color,ls='--',label=label)
    all_values=np.r_[data['production_mean_y_lower'],data['analytic_candidate_mean_y_lower']]
    ax.set_xlim(np.floor(np.log10(all_values.min())),np.ceil(np.log10(all_values.max())))
    ax.set_xlabel(r'$\log_{10}(\bar y_{\rm halo,lower})$');ax.set_ylabel('Probability per dex')
    ax.legend();fig.tight_layout();figure_save(fig,root,'independent_mean_y_check')
    save(root/'audit/mean_y_prior_summary.json',dict(groups=statistics,used_as_prior=False,
        interpretation='Approximate conservative halo contribution, not total cosmic mean y. Exceeding the independent data limit warrants rejection only in a chosen data-informed prior or likelihood; not silently applied here.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    prior_plots(args.root);quadrature_plot(args.root);map_comparisons(args.root);pixel_plot(args.root);mean_y_plot(args.root)
