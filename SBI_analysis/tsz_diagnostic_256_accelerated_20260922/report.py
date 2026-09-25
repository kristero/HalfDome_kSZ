"""Publication-sized diagnostic figures and a self-contained PDF/Markdown report.

This script requires completed simulations AND completed inference results. It
never labels submission, compilation or a synthetic smoke test as science data.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import textwrap
import numpy as np
import toml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binom
from launch import ROOT,atomic_json

RUN=ROOT/'diagnostic_256'
OUT=RUN/'analysis'
LABELS=[r'$P_0$',r'$x_c$',r'$\beta$',r'$\alpha_{m,P_0}$',r'$\alpha_{m,x_c}$',
        r'$\alpha_{m,\beta}$',r'$\alpha_{z,P_0}$',r'$\alpha_{z,x_c}$',r'$\alpha_{z,\beta}$']
METHODS={'moped_fixed':'Local unbinned MOPED','moped_unbinned':'Regression unbinned MOPED','bins40':'40-bin control'}
COLORS={'moped_fixed':'#0072B2','moped_unbinned':'#D55E00','bins40':'#009E73'}
B12=np.array([18.1,.497,4.35,.154,-.00865,.0393,-.758,.731,.415])
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':16,
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,
    'figure.dpi':130,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name,pdf):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    pdf.savefig(fig,bbox_inches='tight')
    plt.close(fig)


def prose_page(pdf,title,paragraphs):
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.09,.94,title,fontsize=22,weight='bold',va='top')
    y=.875
    for paragraph in paragraphs:
        lines=textwrap.wrap(paragraph,width=83,break_long_words=False)
        text='\n'.join(lines)
        fig.text(.09,y,text,fontsize=11.5,va='top',linespacing=1.5)
        y-=len(lines)*.022+.023
    if y<.04:raise RuntimeError('Report prose page overflows: '+title)
    pdf.savefig(fig);plt.close(fig)


def main():
    data=np.load(RUN/'dataset.npz');theta=data['theta'];low=data['lower'];high=data['upper'];width=high-low
    ell=data['ell_unbinned'];clean=data['clean_dl_unbinned'];noisy=data['noisy_dl_unbinned']
    gate=json.loads((ROOT/'results/final_gate.json').read_text())
    quality=json.loads((OUT/'data_quality.json').read_text())
    metrics=json.loads((OUT/'inference_metrics.json').read_text())
    forward=json.loads((OUT/'forward_summary.json').read_text())
    expected={(method,size,shuffle,seed) for method in METHODS for size in [64,128,192]
        for shuffle in ([False,True] if size==192 else [False]) for seed in [20260920,20260921]}
    actual={(r['method'],r['size'],r['shuffled'],r['seed']) for r in metrics}
    assert actual==expected,'Inference jobs are incomplete or have unexpected settings'
    failures=[{k:r[k] for k in ['method','size','shuffled','seed']} for r in metrics if not r['sampling_complete']]
    baseline=np.array(quality['prior_mean_rmse_over_width'])
    primary=[r for r in metrics if r['method']=='moped_fixed' and r['size']==192 and not r['shuffled']]
    complete_primary=all(r['sampling_complete'] for r in primary)
    ratios=(np.mean([r['rmse_over_width'] for r in primary],axis=0)/baseline if complete_primary else np.full(9,np.nan))
    batch_times=[]
    for folder in sorted((RUN/'batches').glob('*')):
        if not (folder/'status.json').exists():continue
        status=json.loads((folder/'status.json').read_text());stages=toml.load(folder/'batch.toml')['timings']
        match=re.search(r'Maximum resident set size \(kbytes\):\s+(\d+)',(folder/'time.txt').read_text())
        batch_times.append(dict(batch=folder.name,seconds=status['seconds'],stages=stages,
            peak_RSS_GiB=int(match[1])/2**20 if match else None,job=status['job']))
    result=dict(count=len(theta),gate_passed=gate['passed'],all_inference_runs_complete=True,
        sampling_failures=failures,primary_rmse_relative_to_prior=ratios.tolist(),
        actual_posterior_forward_checks=forward,
        batch_timings=batch_times,utc=datetime.now(timezone.utc).isoformat(),
        decision=('Numerical pilot complete; inspect recovery and calibration before authorizing a larger run'
                  if not failures else 'Sampling failures require investigation before a larger run'),
        limitations=['Conditional SO noise on one fixed catalogue, not a full observational covariance',
            '64 held-out Sobol rows provide a coarse coverage diagnostic, not a precise SBC certificate',
            'The NSIDE8192 point painter retains measured resolution systematics',
            'FLAMINGO posteriors are effective parameters at unmatched cosmology and gas physics'])
    atomic_json(OUT/'final_summary.json',result)
    pdfpath=OUT/'tsz_256_diagnostic_report.pdf'
    with PdfPages(pdfpath) as pdf:
        prose_page(pdf,'256-row tSZ diagnostic',[
            f"Completed {len(theta)} saved parameter draws, with no prior rejection. The primary inference uses all 7,900 individual multipoles, ell = 80...7979, before local nine-direction MOPED compression. The split is 192 training-pool rows and 64 held-out rows.",
            "Physics: spherical gas support at 4 R200c, exact finite line-of-sight chord, fixed painter cosmology h=0.68, Omega_b=0.049, Omega_c=0.261, and the selected 85,224,251-halo catalogue. A 2 arcmin Gaussian smooths the signal once. The f_sky=0.4 cap, 60 arcmin apodization and mask seed 12345 are shared. Each row has two independent SO noise splits.",
            f"The accelerated-versus-reference numerical gate passed: {gate['passed']}. Cache choices: {gate['cache_counts']}. Every row was checked at 2,496 direct spherical-column points; four selected prior rows also received a finer-cache full-sky comparison. The clean per-multipole target is 1%. These sampled tests do not prove a uniform bound over all continuous halo coordinates.",
            f"There were {len(failures)} incomplete held-out posterior-sampling runs. The following pages show the prior baseline, shuffled-data baseline, learning curves and approximate marginal coverage. No larger dataset is submitted by this pipeline.",
            "A percentage accuracy target and an inference error are different. Existing 8192-versus-16384 references give B12 a 0.091% spectrum-norm residual, a 1.06% worst-multipole residual and about 0.39 conditional-noise units in local nine-direction MOPED. The deliberately bright/shallow reference has a 0.0042% norm residual but about 2.33 conditional-noise units. Those discretization errors remain even when cache errors are negligible."
        ])
        # The true design is shown, not a Gaussian KDE of a rectangular prior.
        fig,axes=plt.subplots(3,3,figsize=(13,10))
        for j,ax in enumerate(axes.flat):
            ax.hist(theta[:,j],bins=np.linspace(low[j],high[j],17),density=True,
                histtype='step',lw=2,color='#0072B2',label='256 draws')
            ax.axhline(1/width[j],color='black',ls='--',label='Uniform density')
            ax.axvline(B12[j],color='#D55E00',lw=1.6,label='Battaglia12')
            ax.set(xlabel=LABELS[j],ylabel='Density',xlim=(low[j],high[j]))
        axes.flat[0].legend();fig.tight_layout();save(fig,'prior_coverage',pdf)
        curves=np.load(ROOT/'results/final_gate_spectra.npz')
        anchors=json.loads((ROOT/'anchors.json').read_text())
        fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
        for ax,case in zip(axes.flat,anchors):
            label=case['label'];c=curves[label+'_candidate']
            for suffix,name,color in [('cache_reference','Cache refinement','#0072B2'),('output8192','Output refinement','#D55E00')]:
                ref=curves[label+'_'+suffix]
                ax.plot(ell,100*(c/ref-1),color=color,lw=.8,label=name)
            ax.set(title=label.replace('_',' '),xlabel=r'$\ell$',ylabel=r'$\Delta D_\ell / D_\ell$ [%]')
            ax.axhline(0,color='.5',lw=.7)
        axes.flat[0].legend();fig.tight_layout();save(fig,'final_numerical_comparison',pdf)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for row in clean:axes[0].plot(ell[::8],row[::8],color='#0072B2',alpha=.07,lw=.6)
        for row in noisy:axes[1].plot(ell[::8],row[::8],color='#0072B2',alpha=.04,lw=.5)
        observations={}
        for path in sorted((RUN/'observations').glob('*.npz')):
            observations[path.stem]=np.load(path)
            if 'sensitivity' not in path.stem:
                axes[0].plot(ell,observations[path.stem]['clean_dl_unbinned'],lw=1.5,label=path.stem)
        axes[0].set(yscale='log',xlabel=r'$\ell$',ylabel=r'Clean $D_\ell^{yy}$')
        axes[1].set_yscale('symlog',linthresh=max(float(np.median(abs(noisy))),1e-30))
        axes[1].set(xlabel=r'$\ell$',ylabel=r'Split-cross $D_\ell^{yy}$')
        axes[0].legend();fig.tight_layout();save(fig,'spectra_and_observations',pdf)
        theory=np.load(ROOT/'reference/classsz_sphere4_h068.npz')
        theory_ell=theory['ell']
        sigma=np.deg2rad(2/60)/np.sqrt(8*np.log(2))
        theory_dl=(theory['dl_1h']+theory['dl_2h'])*np.exp(-theory_ell*(theory_ell+1)*sigma**2)
        unmasked=np.load(ROOT/'tests/anchors/Battaglia12/output4096/unmasked_clean_cl.npy')[ell]*ell*(ell+1)/(2*np.pi)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        axes[0].loglog(ell,unmasked,label='B12 catalogue',lw=1.4)
        axes[0].loglog(theory_ell,theory_dl,label='CLASS-SZ: 1h + 2h',lw=1.7)
        axes[0].set(xlabel=r'$\ell$',ylabel=r'Beam-smoothed $D_\ell^{yy}$',xlim=(80,7979))
        interp=np.exp(np.interp(np.log(ell),np.log(theory_ell),np.log(theory_dl)))
        axes[1].plot(ell,unmasked/interp,color='#D55E00',lw=1)
        axes[1].axhline(1,color='black',ls='--')
        axes[1].set(xlabel=r'$\ell$',ylabel='Catalogue / halo model')
        axes[0].legend();fig.tight_layout();save(fig,'classsz_context',pdf)
        fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
        for j,ax in enumerate(axes.flat):
            for method,label in METHODS.items():
                for seed in [20260920,20260921]:
                    rows=sorted([r for r in metrics if r['method']==method and r['seed']==seed and not r['shuffled']
                        and r['sampling_complete']],key=lambda r:r['size'])
                    ax.plot([r['size'] for r in rows],[r['rmse_over_width'][j] for r in rows],
                        'o-',color=COLORS[method],alpha=1 if seed==20260920 else .45,
                        label=label if seed==20260920 else None)
            shuffle=[r['rmse_over_width'][j] for r in metrics if r['method']=='moped_fixed' and r['shuffled'] and r['sampling_complete']]
            ax.axhline(baseline[j],color='black',ls='--',label='Prior mean')
            if shuffle:ax.scatter([192]*len(shuffle),shuffle,color='black',marker='x',s=45,label='Shuffled')
            ax.set(title=LABELS[j],xlabel='Training-pool rows',ylabel='RMSE / prior width',xticks=[64,128,192])
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.03))
        fig.tight_layout();save(fig,'learning_curves',pdf)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,target,key in zip(axes,[.68,.95],['marginal_68_coverage','marginal_95_coverage']):
            lo,hi=binom.interval(.95,64,target)
            ax.axhspan(lo/64,hi/64,color='.88',label='Binomial reference band')
            ax.axhline(target,color='black',ls='--')
            for seed,r in enumerate(primary):
                if r['sampling_complete']:ax.plot(np.arange(9),r[key],'o-',label='Training seed '+str(seed+1))
            ax.set(xticks=np.arange(9),xticklabels=LABELS,ylim=(0,1),ylabel='Held-out marginal coverage',title=f'{100*target:.0f}% intervals')
            ax.tick_params(axis='x',rotation=70)
        axes[0].legend();fig.tight_layout();save(fig,'heldout_coverage',pdf)
        for label in observations:
            if 'sensitivity' in label:continue
            fig,axes=plt.subplots(3,3,figsize=(13,10))
            for j,ax in enumerate(axes.flat):
                for seed,color in zip([20260920,20260921],['#0072B2','#D55E00']):
                    path=OUT/f'flamingo_{label}_moped_fixed_{seed}.npy'
                    samples=np.load(path)
                    if len(samples):ax.hist(samples[:,j],bins=np.linspace(low[j],high[j],21),density=True,
                        histtype='step',lw=1.8,color=color,label='Training seed '+str(seed-20260919))
                ax.axhline(1/width[j],color='black',ls='--',label='Uniform prior')
                if label=='Battaglia12':ax.axvline(B12[j],color='#009E73',lw=1.8,label='Known truth')
                ax.set(xlabel=LABELS[j],ylabel='Density',xlim=(low[j],high[j]))
            axes.flat[0].legend();fig.suptitle(label,y=1.01,fontsize=20)
            fig.tight_layout();save(fig,'posterior_'+label,pdf)
        fig,axes=plt.subplots(3,3,figsize=(13,10))
        for j,ax in enumerate(axes.flat):
            for label,name,color in [('Battaglia12','Raw 8192','#0072B2'),
                ('Battaglia12_raw4096_sensitivity','4096 perturbation','#D55E00'),
                ('Battaglia12_raw16384_sensitivity','16384 perturbation','#009E73')]:
                samples=np.load(OUT/f'flamingo_{label}_moped_fixed_20260920.npy')
                if len(samples):ax.hist(samples[:,j],bins=np.linspace(low[j],high[j],21),density=True,
                    histtype='step',lw=1.6,color=color,label=name)
            ax.axhline(1/width[j],color='black',ls='--',label='Uniform prior')
            ax.axvline(B12[j],color='.5',lw=1.2)
            ax.set(xlabel=LABELS[j],ylabel='Density',xlim=(low[j],high[j]))
        axes.flat[0].legend(fontsize=9)
        fig.tight_layout();save(fig,'trained_SBI_resolution_sensitivity',pdf)
        if forward['completed']:
            predictions=np.load(OUT/'forward_spectra.npz')
            fig,axes=plt.subplots(2,2,figsize=(13,9))
            for ax,label in zip(axes.flat,['Battaglia12','L1_m9','fgas-8sigma','Mstar-1sigma']):
                ax.plot(ell,observations[label]['clean_dl_unbinned'],color='black',lw=2,label='Observation signal')
                for number,case in enumerate([r for r in forward['cases'] if r['observation']==label]):
                    ax.plot(ell,predictions[case['label']+'_clean'],lw=1.4,label='Posterior draw '+str(number+1))
                ax.set(yscale='log',xlabel=r'$\ell$',ylabel=r'Clean $D_\ell^{yy}$',title=label)
                ax.legend()
            fig.tight_layout();save(fig,'actual_posterior_forward_checks',pdf)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        axes[0].hist(np.array([r['seconds'] for r in batch_times])/60,bins=12,color='#0072B2')
        axes[0].set(xlabel='Minutes per four-row batch',ylabel='Batches')
        axes[1].plot([r['peak_RSS_GiB'] for r in batch_times],'o',color='#D55E00')
        axes[1].set(xlabel='Batch index',ylabel='Peak resident memory [GiB]')
        fig.tight_layout();save(fig,'runtime_and_memory',pdf)
        prose_page(pdf,'Interpretation and limits',[
            "Flat prior density does not guarantee informative data. A posterior resembling the prior can be correct for a weakly constrained parameter. It can also expose insufficient simulations, an unsuitable compression or a failed density estimator. The held-out RMSE, shuffled-data control, training-size trend and known B12 observation distinguish these possibilities better than posterior appearance alone.",
            "The primary compressor multiplies all 7,900 raw D_ell values by the frozen nine-direction B12 finite-difference MOPED matrix. A signed asinh and standardization are then fitted using only optimization rows. A separate regression-based unbinned compressor transforms individual multipoles first and estimates derivatives from nearby training rows. It is an approximate alternative, not a global sufficiency certificate.",
            "FLAMINGO is an external, out-of-family observation. Its posterior represents effective gNFW parameters for this fixed HalfDome painter, cosmology, radial truncation and estimator. It is not a measurement of a uniquely defined FLAMINGO gNFW truth. Observations use the saved fixed noise; training rows use independent noise realizations.",
            "The coverage band is a binomial sampling reference for 64 marginal events. A Sobol design and correlations between parameters mean this is not an exact independent-binomial test of simulation-based calibration. A precise coverage study, field-to-field catalogue variance, foregrounds and a full survey covariance require additional work.",
            "Smooth boundary means smooth continuation of the cached chord-mean pressure across the spherical edge. The physical gas still stops exactly at 4 R200c. Halving the cache changes interpolation node density, not the physical model. Greedy scheduling changes which thread paints a halo block, while shared geometry reuses positions, radii and intersected pixels across four different pressure fields.",
            "The noise split convention is two independent maps each with the specified SO N_ell. If that curve describes a full-depth survey, two half-exposure observational splits ordinarily need twice that N_ell each. The current convention is retained and explicitly reported; this diagnostic is not a validation of survey split-depth modelling."
        ])
        prose_page(pdf,'Independent physics checks',[
            "The complete 256-row prior also receives an independent SciPy LOS check at 4,096 selected points. This integrates directly along LOS distance using geometric subintervals and a cusp-removing substitution at the centre, instead of the Julia sinh/log-radius integration. Ratios to the central column test the pressure shape and spherical projection; this ratio check does not independently determine the absolute electron-pressure amplitude.",
            f"Measured independent normalized-column discrepancy: {gate['independent_projection']['max_relative_visible']:.3g} maximum relative error where the normalized column exceeds 1e-10. The maximum absolute normalized-column error is {gate['independent_projection']['max_absolute_ratio_error']:.3g}.",
            "The CLASS-SZ comparison reuses the independently computed sphere-4, matched-painter-cosmology halo-model reference from September 20. It applies the same 2 arcmin signal beam and compares against the new unmasked B12 catalogue spectrum. This is a normalization/shape context check, not a 1% equality test: a theoretical halo mass function and bias model differ from a particular finite lightcone catalogue.",
            "The earlier CLASS-SZ three-dimensional pressure evaluations matched the B12-family formula to about 4.3e-15 across the tested ordinary, FL-fit and shallow profiles. Earlier independent Python/Julia columns agreed to about 7.3e-13, and volume versus projected pressure integrals agreed to about 5e-14. Those are retained reference tests; they are not newly claimed all-catalogue CLASS-SZ measurements.",
            "The original absolute amplitude is obtained from XGPaint's prepared thermal-SZ slice. The normalized quadrature divides by a convenient integrand scale only during numerical integration, and restores the scale in the returned integral. The 1e-10 relative quadrature tolerance, mass convention, cosmology, fixed gamma=-0.3 and alpha=1, and physical outer radius are unchanged by the speed improvements."
        ])
    # Keep the text conclusion usable without opening a figure book.
    lines=['# Completed 256-row diagnostic','',result['decision'],'',
        f"Completed rows: {len(theta)}. Numerical gate: {gate['passed']}. Inference runs: {len(metrics)}.",
        f"Incomplete posterior-sampling runs: {len(failures)}.",'',
        'Primary MOPED RMSE / prior-mean RMSE by parameter:',
        json.dumps(dict(zip([str(n) for n in data['parameter_order']],ratios.tolist())),indent=2),'',
        'Numbers below one indicate improved held-out point prediction over the prior mean. They do not alone establish posterior calibration.',
        '', 'Remaining limits:','']+['- '+v for v in result['limitations']]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
        if p.is_file() and p.name!='artifact_manifest.json'}
    atomic_json(OUT/'artifact_manifest.json',dict(sha256=outputs,completed_utc=result['utc']))
    export=ROOT/'output/pdf';export.mkdir(parents=True,exist_ok=True)
    shutil.copy2(pdfpath,export/pdfpath.name)
    print(str(pdfpath),flush=True)


if __name__=='__main__':main()
