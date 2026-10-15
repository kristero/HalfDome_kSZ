#!/usr/bin/env python3
"""Build portable tables, a physics report and a checked artifact manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
ORDER = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta',
         'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
TRUTH = [18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415]


def load(root, name):
    return json.loads((root/name).read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def posterior_table(root, definitions, prefix, filename):
    with (root/filename).open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['model','case','family','parameter','truth','q025','q16','median','q84','q975',
                         'truth_hpd_theta','log_prob_truth'])
        for model in ['original_linear','band50_linear','original_bounded','band50_bounded']:
            result=load(root,f'{prefix}{model}.json')
            for i,case in enumerate(definitions):
                for j,name in enumerate(ORDER):
                    writer.writerow([model,case['label'],case['family'],name,TRUTH[j],
                                     *[result[k][i][j] for k in ['q025','q16','median','q84','q975']],
                                     result['truth_hpd_theta'][i],result['log_prob_truth'][i]])


def summary_figure(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16,'axes.labelsize':19,'axes.titlesize':19,
                         'xtick.labelsize':13,'ytick.labelsize':14,'legend.fontsize':12})
    definitions=load(root,'counterfactuals.json')
    extended=load(root,'extended_cases.json')
    old=load(root,'original_linear.json')
    extra=load(root,'extended_original_linear.json')
    lookup={case['label']:i for i,case in enumerate(definitions)}
    xlookup={case['label']:i for i,case in enumerate(extended)}
    series=[(old,lookup,['mean_noisy','mean_clean','smooth_clean_5','fine_only_5'],'Simulations'),
            (extra,xlookup,['mask_mean_actual','mask_mean_clean',None,None],'Masks'),
            (extra,xlookup,['joint_mean_actual','joint_mean_clean','joint_mean_smooth','joint_mean_fine'],'Combined')]
    # The mean mask smooth/fine cases are averaged spectra, not averaged posterior medians.
    # They are supplied by the final small controls added to the extension below.
    series[1]=(extra,xlookup,['mask_mean_actual','mask_mean_clean','mask_mean_smooth','mask_mean_fine'],'Masks')
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained',sharey='row')
    for col,(result,index,labels,title) in enumerate(series):
        ids=[index[label] for label in labels]
        for row,(j,ylabel) in enumerate([(0,r'$P_0$'),(2,r'$\beta$'),(3,r'$\alpha_{m,P_0}$')]):
            ax=axes[row,col]
            med=np.asarray(result['median'])[ids,j]
            lo=np.asarray(result['q16'])[ids,j];hi=np.asarray(result['q84'])[ids,j]
            ax.errorbar(np.arange(4),med,yerr=[med-lo,hi-med],fmt='o',color='C0',capsize=3,label='68% interval')
            ax.axhline(TRUTH[j],color='k',ls='--',label='Truth')
            ax.axhline(old['median'][0][j],color='0.5',ls=':',label='Seed 100')
            ax.set_xticks(np.arange(4),['Actual','Clean','Smooth','Fine'])
            if row==0:ax.set_title(title)
            if col==0:ax.set_ylabel(ylabel)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.035),ncol=3,frameon=False,fontsize=14)
    for extension in ['png','pdf']:
        fig.savefig(root/f'plots/mechanism_summary.{extension}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def report(root):
    diag=load(root,'diagnostics.json')
    old=load(root,'original_linear.json')
    extra=load(root,'extended_original_linear.json')
    definitions=load(root,'counterfactuals.json')
    extended=load(root,'extended_cases.json')
    index={case['label']:i for i,case in enumerate(definitions)}
    xindex={case['label']:i for i,case in enumerate(extended)}
    def values(result,i):
        return ' | '.join(f'{result["median"][i][j]:.2f}' for j in [0,2,3])
    mean=diag['directional_offsets']['other_seed_median_mean']
    sd=diag['directional_offsets']['other_seed_median_sd']
    selected=load(root,'band50_linear.json')
    bounded=load(root,'original_bounded.json')
    correlations=load(root,'posterior_correlations.json')
    corr=np.array([correlations[str(seed)]['correlation'] for seed in range(102,120,2)])
    noise=np.asarray(old['training_sky_noise_draws']['median'])
    smooth_ids=[i for i,case in enumerate(extended) if case['family']=='smooth_simulation']
    smooth=np.asarray(extra['median'])[smooth_ids,0]
    text=fr'''# Why the non-training seeds share a pressure-parameter offset

The dominant diagnosed effect is inference conditional on the fixed training sky, amplified by fine multipole structure. It is not an approximately 3× increase in physical pressure in the other simulations. All ten maps were painted with exactly the same Battaglia12 parameters. The other nine seeds' original unbounded posterior medians have P0 = {mean[0]:.2f} ± {sd[0]:.2f} and beta = {mean[2]:.2f} ± {sd[2]:.2f}, where ± here describes the scatter of medians across nine skies, not a posterior uncertainty. Their offsets are similar, rather than exactly constant.

![Controlled spectral tests](plots/mechanism_summary.png)

## Evidence from controlled changes

These tests perturb the saved seed-100/mask-12345 observation. They are diagnostics, not new physical mock skies. The fixed Gaussian beam, SO baseline deproj0 cross-spectrum definition and trained feature transformation are preserved.

For each multipole ell, D_ell = ell(ell+1)C_ell/(2 pi), where C_ell is the masked angular cross-power. The saved training correction is

$$D_{{\mathrm{{obs}},\ell}}=D_{{\mathrm{{ss}},\ell}}
+B_\ell(D_{{\mathrm{{prod}},\ell}}-D_{{\mathrm{{ss}},\ell}}-D_{{\mathrm{{nn}},\ell}})
+B_\ell^2D_{{\mathrm{{nn}},\ell}}.$$

Here ss denotes the beam-smoothed clean signal, prod the original signal-plus-two-noise-draw cross-spectrum, nn the unfiltered noise–noise cross-spectrum, and B_ell the two-arcminute Gaussian harmonic beam. The middle term is the corrected signal–noise contribution. This is the trained harmonic correction approximation; it is not an exact commutation of masking and beam smoothing. All controls preserve this operator. At a fixed mask the nn term is identical for all simulation seeds, so its differences cancel; for changed masks we subtract its saved B_ell squared contribution separately.

| Change to seed 100 | P0 median | beta median | alpha_m,P0 median |
|---|---:|---:|---:|
| None | {values(old,0)} |
| Mean of the nine other observations | {values(old,index['mean_noisy'])} |
| Change only the clean signal | {values(old,index['mean_clean'])} |
| Change only the signal–noise term | {values(old,index['mean_cross'])} |
| Smooth the fractional clean-signal change, sigma_ell = 5 | {values(old,index['smooth_clean_5'])} |
| Keep only its removed fine residual, sigma_ell = 5 | {values(old,index['fine_only_5'])} |
| Keep only its removed fine residual, sigma_ell = 50 | {values(old,index['fine_only_50'])} |
| Reverse the clean-signal change | {values(old,index['mean_reversed_clean'])} |
| Change amplitude alone by +0.742% in clean power | {values(extra,xindex['amplitude_plus_0p742percent'])} |
| Generating Battaglia12 parameters | 18.10 | 4.35 | 0.154 |

The smooth sigma_ell = 5 control returns all nine individual simulations to P0 medians between {smooth.min():.2f} and {smooth.max():.2f}. The complementary fine residual retains the upward shift. Smoothing the absolute change independently gives P0 = {extra['median'][xindex['mean_absolute_smooth50']][0]:.2f} at sigma_ell = 50, confirming that the result does not depend on smoothing a fractional ratio.

The mean of nine other skies still produces the shift. Averaging away their independent fluctuations cannot restore the particular training realization. Reversing the fine residual instead moves P0 to {extra['median'][xindex['mean_fine_sign-1']][0]:.2f}. This is a directional response, not an input-independent high-P0 prediction.

Both parameter-space implementations show the mechanism: the original bounded model gives P0 = {bounded['median'][index['mean_noisy']][0]:.2f} for the mean new observation, {bounded['median'][index['smooth_clean_50']][0]:.2f} for the smooth control and {bounded['median'][index['fine_only_50']][0]:.2f} for the fine residual. All three independently trained ensemble members give elevated P0 on the other simulations. All loaded modules are in evaluation mode, and the seed-100 log density is invariant to evaluation in a different batch to within {old['cohort_log_prob_error']:.3g}; BatchNorm updating is not responsible.

## Why different random skies can shift in the same direction

The network learns a posterior conditional on one fixed halo realization H_100 and one fixed training window W_12345:

$$q_\phi(\theta\mid D)\simeq p(\theta\mid D,H_{{100}},W_{{12345}}).$$

Here theta is the nine pressure parameters, D is the signed beam-corrected pseudo-power cross-spectrum, H is the halo lightcone and W is the apodized cap mask. Training varies theta and the SO noise, but does not marginalize over independent H or W. The intended general-sky inference instead requires integrating over these nuisance realizations. This differs from fitting a truly seed-marginalized simulator.

For independent sky fluctuations epsilon_i about an ensemble spectrum, every difference used in the comparison contains the same reference:

$$\Delta_i=\epsilon_i-\epsilon_{{100}},\qquad
\mathbb E[\Delta_i\mid\epsilon_{{100}}]=-\epsilon_{{100}}.$$

Thus all differences share the negative of seed 100's particular fluctuation. Their signs are not independent tests of a zero-mean parameter bias. Under independent, equal-covariance skies, distinct differences have unconditional cross-covariance equal to the covariance of the reference sky. Nonlinear inference can map their common spectral component into a common pressure-parameter direction.

The controlled smooth/fine tests identify that component in the rapidly varying multipole structure. A fixed catalogue and mask leave a repeatable pattern of halo positions, masses, redshifts and mask mixing in the clean spectrum. The trained fine-spectrum model uses that pattern. A different sky or window changes it, even though the actual pressure law is identical. Noise-only whitening does not model this additional sky/window variation. In the original retained whitened PCA coordinates, clean differences have norms {np.min(np.asarray(diag['spectral_metrics']['retained_pca_clean_distance'])[10::10]):.1f}–{np.max(np.asarray(diag['spectral_metrics']['retained_pca_clean_distance'])[10::10]):.1f}. These are reference noise distances, not formal sigma significances or chi-square probabilities.

This is a conditional mismatch, not proof that all new observations lie outside global training support. Nearest-neighbour distances to the broad training pool sit at roughly the 22nd–30th percentiles of held-out training distances. The parameter prior spans alternative spectra that can resemble the new observation at incorrect parameters; global proximity alone does not ensure recovery of the true pressure law. General neural-SBI misspecification failures are discussed by [Cannon, Ward and Schmon](https://arxiv.org/abs/2209.01845) and explicit discrepancy modelling by [Ward et al.](https://arxiv.org/abs/2210.06564).

## Why P0, beta and the alpha parameters move together

The implemented Battaglia pressure shape is

$$\frac{{P_e(r\mid M,z)}}{{P_{{200}}(M,z)}}=
P_0(M,z)\left[\frac{{x}}{{x_c(M,z)}}\right]^{{-0.3}}
\left[1+\frac{{x}}{{x_c(M,z)}}\right]^{{-\beta(M,z)}},\quad x=\frac{{r}}{{R_{{200c}}}},$$

$$A(M,z)=A_0\left(\frac{{M}}{{10^{{14}}M_\odot}}\right)^{{\alpha_{{m,A}}}}
\,(1+z)^{{\alpha_{{z,A}}}},\qquad A\in\{{P_0,x_c,\beta\}}.$$

In the last equation the two scaling factors multiply. P_e is electron pressure; P_200 is the common mass/redshift-dependent pressure scale; x_c is a dimensionless radius scale; beta controls the outer decline. The alpha_m and alpha_z coordinates control how each of these three quantities changes with halo mass and redshift. The fixed inner slope is -0.3, the transition exponent is 1, and projection is inside a finite 4R_200c sphere, matching training. See [Battaglia et al.](https://arxiv.org/abs/1109.3711).

The painted Compton-y contribution along a sky direction is the physical pressure integral

$$y=\frac{{\sigma_T}}{{m_ec^2}}\int P_e\,dl_{{\rm phys}}.$$

Here sigma_T is the Thomson scattering cross section, m_e the electron mass, c the speed of light and dl_phys physical line-of-sight length. The integral is restricted to the halo's finite sphere and summed over the catalogue. M is physical M200c, R_200c encloses mean density 200 times the critical density, and z is redshift. Catalogue masses in solar masses divided by h are converted using h = 0.68, and the physical mass threshold is 10^12 solar masses. All available finite nonnegative redshifts are used; the interpolation cache spans z = 0.001–5. No low-redshift-only tSZ selection is applied.

At fixed other parameters, raising P0 increases pressure and raises the clean tSZ power as P0 squared. Raising beta suppresses pressure at finite radius, while raising x_c increases it. They can compensate. The mass and redshift exponents provide further compensations by changing which halo populations contribute most to each angular scale. Consequently a high inferred P0 alongside high beta and altered alpha values need not mean a proportionally larger observable tSZ amplitude.

For example, at fixed halo mass, redshift and radius, set u = x/x_c. Then

$$d\ln P_e=d\ln P_0+\left(0.3+\frac{{\beta u}}{{1+u}}\right)d\ln x_c
-\ln(1+u)\,d\beta.$$

The equation shows the compensating directions locally; it does not assert a positive P0–beta correlation in the full posterior. In the stored joint samples from non-training skies, the average x_c–beta correlation is {corr[:,1,2].mean():.2f}, while P0–x_c is {corr[:,0,1].mean():.2f}. The actual nine-dimensional correlations differ from comparing the directions of marginal-median offsets across observations.

This is why a pure-amplitude explanation fails. Compared with seed 100, other simulations average {100*diag['spectral_blocks']['low']['mean_other_relative']:.2f}% in the low-ell block, {100*diag['spectral_blocks']['middle']['mean_other_relative']:.3f}% in the middle block and {100*diag['spectral_blocks']['high']['mean_other_relative']:.3f}% at high ell. One seed has the opposite low-ell shift yet shows the same large P0 increase. A +0.742% clean-power amplitude change alone barely shifts P0. The local derivative matrix has a prior-scaled singular-value ratio of about {diag['linear_scores']['relative_singular_values'][-1]:.2g}, so the full nine-coordinate linear inverse is ill-conditioned; its least-squares coordinate shifts are not reliable nonlinear estimates.

The alpha offsets also have different signs: alpha_m,P0 and alpha_m,beta increase, whereas alpha_m,xc, alpha_z,P0 and alpha_z,xc decrease. These coordinated shifts reflect a coupled fit, not a uniform rescaling of every exponent. The joint-draw correlation matrices are provided separately; marginal medians should not be combined and interpreted as a best-fitting physical profile.

## Mask and combined controls

Mask changes alter the clean spectrum, the signal–noise cross term, and the noise–noise term. The extension separates all three using the actual saved beam operator, rather than attributing all residual changes to signal–noise. For seed 100, clean-only mask changes give most of the P0 shift; noise-only changes do not. The smooth/fine result also holds for changed masks and for the mean of the 81 joint-only cases (nine non-training simulations crossed with nine non-training masks).

| Mean change | Actual P0 | Clean-only P0 | Smooth-only P0 | Fine-only P0 |
|---|---:|---:|---:|---:|
| Other simulations, fixed mask | {old['median'][index['mean_noisy']][0]:.2f} | {old['median'][index['mean_clean']][0]:.2f} | {old['median'][index['smooth_clean_5']][0]:.2f} | {old['median'][index['fine_only_5']][0]:.2f} |
| Other masks, fixed seed 100 | {extra['median'][xindex['mask_mean_actual']][0]:.2f} | {extra['median'][xindex['mask_mean_clean']][0]:.2f} | {extra['median'][xindex['mask_mean_smooth']][0]:.2f} | {extra['median'][xindex['mask_mean_fine']][0]:.2f} |
| Both changed | {extra['median'][xindex['joint_mean_actual']][0]:.2f} | {extra['median'][xindex['joint_mean_clean']][0]:.2f} | {extra['median'][xindex['joint_mean_smooth']][0]:.2f} | {extra['median'][xindex['joint_mean_fine']][0]:.2f} |

## What this diagnosis does and does not solve

The existing retrained Delta-ell = 50 model is insensitive to these fine residual controls: P0 stays near {selected['median'][0][0]:.1f} for the baseline and {selected['median'][index['mean_noisy']][0]:.1f} for the mean new observation. This supports the earlier finding that banding reduces realization sensitivity. It does not establish unbiased marginal medians: even the original seed-100 beta median is about 9.7 instead of the generating 4.35, and the banded P0 median remains above 18.1. Broad nonlinear degeneracies and prior volume can displace medians even when the truth is in a high-density joint region.

For the original models, the mean new observation and fine-only control exclude truth at joint 95%, while the smooth controls restore the truth to joint 68%. For the banded models the truth remains inside joint 68% for these controls. Joint regions here use densities in physical theta coordinates, including the parameter-transform Jacobian. This distinction between median bias, joint truth containment and actual calibration must be retained.

The principled remedy is to train while varying catalogues and masks, then test on entire held-out catalogues/masks. A nuisance covariance or residual basis can provide an interim approximation, but it must enter the simulator/compression and retraining consistently. A revised MOPED could downweight the nuisance directions in C_eff = C_noise + C_sky, or project them out before optimizing the pressure score. The nuisance response changes with pressure parameters, and ten skies at one B12 point are insufficient to certify it throughout the nine-dimensional prior. Compressing every fine residual away also loses some real pressure information. The previous 25%-width-budget comparison remains the constraint/robustness tradeoff evidence; this follow-up diagnoses its mechanism rather than claiming a newly calibrated method.

Smoothing a change relative to the known B12 reference is a counterfactual test, not a deployable bias subtraction. It restores the seed-100-conditioned result, including its existing marginal offsets. Subtracting an ensemble offset fitted at the truth would not demonstrate unbiased inference for unknown parameters.

## Checks and reproducibility

- Three jobs ran on idark, with 26 CPUs each: two diagnostic jobs and one small completion job adding two mean-mask controls and generating the report. No repainting or retraining was needed. The source experiment and its trained model files remain frozen.
- The completion job finished the added inference controls but initially failed in report formatting. The formatting bug was corrected and report generation rerun successfully; existing posterior draws were preserved.
- Every catalogue uses the same physical M200c cut, h conversion, redshift selection, B12 parameters, 4R_200c truncation, beam and noise-map hashes. Catalogue selected counts are recorded in diagnostics.json; physical equivalence of raw simulation provenance beyond these stored contracts was not re-established here.
- Existing clean bandpowers reconstruct to {diag['exact_band_reconstruction_error']:.2g} maximum relative error; original input features reconstruct exactly for all ten fixed-mask seeds.
- Spectral factorization is checked algebraically; the same-mask noise–noise spectra are identical across simulations. Each posterior uses 9,000 equal-mixture draws from three trained networks. Monte Carlo fluctuations of diagnostic medians are small relative to the observed tens-of-units P0 changes; these runs are not precise estimates of sub-percent effects.
- The old 21 B12 noise draws give P0 medians {noise[:,0].min():.1f}–{noise[:,0].max():.1f}, far below the common ~50 on new skies. Twenty of those draws built whitening and are conditional diagnostics, not independent calibration data.
- Empirical support tests use 600 reserved parameter/noise rows. They do not test calibration across independent halo skies. Posterior correlations use 10,000 stored joint samples per simulation.
- Checksums cover the consumed spectra, whitening, features, models and relevant source files; raw halo catalogues were not rehashed or reread, since their saved spectra were reused. Completion markers, job resource records, CSV tables and all PDF/PNG plots are included. No existing HalfDome maps or prior reports were overwritten.

Code: diagnose.py, extended_controls.py, make_report.py, audit_inputs.py, diagnose.pbs, extended_controls.pbs and finalize.pbs. All were created for this follow-up. Full generated file inventory is in FILES_CREATED.md.
'''
    (root/'REPORT.md').write_text(text)


def main(root):
    definitions=load(root,'counterfactuals.json')
    extended=load(root,'extended_cases.json')
    posterior_table(root,definitions,'','counterfactual_posteriors.csv')
    posterior_table(root,extended,'extended_','extended_control_posteriors.csv')
    diag=load(root,'diagnostics.json')
    with (root/'seed_offset_summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['parameter','truth','seed100_median','other_seed_mean_median','other_seed_sd_median'])
        for j,name in enumerate(ORDER):
            offsets=diag['directional_offsets']
            writer.writerow([name,TRUTH[j],offsets['seed100_median'][j],offsets['other_seed_median_mean'][j],offsets['other_seed_median_sd'][j]])
    report(root)
    summary_figure(root)
    paths=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name not in ['VERIFIED.json','FINAL_COMPLETE.json','FILES_CREATED.md']]
    (root/'FILES_CREATED.md').write_text('# Files created for the offset diagnosis\n\n'
        +'All files are new; no source experiment or existing model was edited.\n\n'
        +'\n'.join(f'- `{p.relative_to(root)}`' for p in sorted(paths))+'\n')
    hashes={str(p.relative_to(root)):sha(p) for p in paths+[root/'FILES_CREATED.md']}
    result=dict(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                jobs=load(root,'jobs.json'),primary_counterfactuals=len(definitions),extended_counterfactuals=len(extended),
                models=['original_linear','original_bounded','band50_linear','band50_bounded'],sha256=hashes,
                status='diagnosis_complete; general-sky posterior calibration is not established')
    (root/'FINAL_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    # A second read checks the saved bytes against the manifest before delivery.
    assert all(sha(root/name)==expected for name,expected in hashes.items())
    (root/'VERIFIED.json').write_text(json.dumps(dict(files=len(hashes),all_hashes_match=True,
        final_complete_sha256=sha(root/'FINAL_COMPLETE.json')),indent=2)+'\n')
    print('REPORT VERIFIED',len(hashes),'files')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    main(parser.parse_args().root)
