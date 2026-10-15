#!/usr/bin/env python3
"""Diagnose directional seed bias without repainting skies or changing trained models.

Counterfactual spectra are controlled perturbations of the saved seed-100
observation, not additional physical skies. Existing observations remain frozen.
The clean signal and its noise cross term can therefore be isolated exactly.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib

os.environ.setdefault('MERLIN_PARAM_SPACE', 'linear')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

PARENT = Path('/lustre/work/kristero10/tsz_robustness_methods_20261004')
OLD = Path('/lustre/work/kristero10/tsz_seed_mask_variation_20261002')
TRUTH = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta',
         'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
LABELS = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_{m,P_0}$',
          r'$\alpha_{m,x_c}$', r'$\alpha_{m,\beta}$', r'$\alpha_{z,P_0}$',
          r'$\alpha_{z,x_c}$', r'$\alpha_{z,\beta}$']
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
SEEDS = np.arange(100, 120, 2)


def write_json(path, obj):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value))
    path.write_text(json.dumps(obj, indent=2, default=encode) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def transform(spectra, params):
    z = np.arcsinh((spectra - params['s_center']) / params['s_scale'])
    return ((z - params['z_mean']) / params['z_sd']).astype(np.float32)


def nearest_distance(x, pool, chunk=80):
    """Euclidean feature distance; compare to held-out rows, not a chi-square law."""
    distances = []
    neighbors = []
    for start in range(0, len(x), chunk):
        d = cdist(x[start:start+chunk], pool, metric='sqeuclidean')
        index = np.argmin(d, axis=1)
        distances.extend(np.sqrt(d[np.arange(len(index)), index]))
        neighbors.extend(index)
    return np.asarray(distances), np.asarray(neighbors)


def load_signals():
    plan = json.loads((PARENT / 'plan.json').read_text())
    clean = []
    unmasked = []
    meta = []
    for case in plan['cases']:
        directory = PARENT / case['path']
        clean.append(np.load(directory / 'clean_cl.npy')[80:7980] * DL)
        meta.append(tomllib.loads((directory / 'complete.toml').read_text()))
    for seed in SEEDS:
        unmasked.append(np.load(PARENT / f'results/seed_{seed}/unmasked_clean_cl.npy')[80:7980]*DL)
    clean = np.asarray(clean)
    noisy = np.load(PARENT / 'data/grid.npz')['full']
    a = np.load(PARENT / 'data/band_operator.npy')
    saved = np.load(PARENT / 'data/grid.npz')['clean']
    error = np.max(np.abs(clean @ a.T / saved - 1))
    assert error < 1e-12, error
    # The same mask must give the same noise-noise spectrum for every sky.
    noise = [np.load(PARENT / c['path'] / 'noise_cl.npy') for c in plan['cases'][::10]]
    assert all(np.array_equal(noise[0], n) for n in noise)
    assert len({m['mask_sha256'] for m in meta[::10]}) == 1
    assert len({tuple(m['noise_seeds']) for m in meta}) == 1
    contracts = [tomllib.loads((PARENT / f'results/seed_{s}/complete.toml').read_text()) for s in SEEDS]
    equal_keys = ['theta', 'mass_convention', 'redshift_selection', 'sphere_R200c',
                  'beam_fwhm_arcmin', 'raw_nside', 'output_nside', 'noise_sha256']
    for key in equal_keys:
        assert all(c[key] == contracts[0][key] for c in contracts), key
    return plan, clean, noisy, np.asarray(unmasked), a, contracts, error


def counterfactuals(clean, noisy):
    """Factor sky changes into clean power and signal-noise cross power.

    D_noisy - D_clean contains the signal-noise and noise-noise terms under
    the exact saved beam operator. The noise-noise term is identical at fixed
    mask, so its difference across simulations cancels exactly.
    """
    rows = []
    definitions = []
    def add(label, spectrum, family, seed=None, fraction=None, note=''):
        definitions.append(dict(label=label, family=family, seed=seed,
                                fraction=fraction, note=note))
        rows.append(spectrum)
    base = noisy[0]
    add('baseline', base, 'baseline', seed=100)
    for i, seed in enumerate(SEEDS):
        add(f'actual_{seed}', noisy[10*i], 'actual', seed=int(seed))
    # These three paired cases isolate the exact terms for every other seed.
    for i, seed in enumerate(SEEDS[1:], 1):
        delta_clean = clean[10*i] - clean[0]
        delta_noise = noisy[10*i] - base - delta_clean
        add(f'clean_only_{seed}', base + delta_clean, 'clean_only', seed=int(seed))
        add(f'cross_only_{seed}', base + delta_noise, 'cross_only', seed=int(seed),
            note='Only the changed signal-noise cross term; identical noise-noise cancels.')
        add(f'reversed_clean_{seed}', base - delta_clean, 'reversed_clean', seed=int(seed))
        for fraction in [.1, .25, .5, .75]:
            add(f'interpolate_{seed}_{fraction}', base + fraction*(noisy[10*i]-base),
                'interpolation', seed=int(seed), fraction=fraction)
    mean_clean = clean[10::10].mean(axis=0) - clean[0]
    mean_noisy = noisy[10::10].mean(axis=0) - base
    mean_cross = mean_noisy - mean_clean
    add('mean_clean', base + mean_clean, 'mean')
    add('mean_noisy', base + mean_noisy, 'mean')
    add('mean_cross', base + mean_cross, 'mean')
    add('mean_reversed_clean', base - mean_clean, 'mean')
    # Preserve broad shape while suppressing the fine multipole fingerprint.
    for sigma in [5, 15, 50, 100]:
        smooth = clean[0] * gaussian_filter1d(mean_clean / clean[0], sigma, mode='nearest')
        add(f'smooth_clean_{sigma}', base + smooth, 'smoothing', fraction=sigma)
        add(f'fine_only_{sigma}', base + mean_clean - smooth, 'fine_structure', fraction=sigma)
    for lo, hi, name in [(80, 580, 'low'), (580, 2980, 'middle'), (2980, 7980, 'high')]:
        select = (ELL >= lo) & (ELL < hi)
        for kind, delta in [('clean', mean_clean), ('cross', mean_cross), ('noisy', mean_noisy)]:
            add(f'{kind}_{name}_only', base + delta*select, 'scale_only', note=f'{lo} <= ell < {hi}')
            add(f'{kind}_remove_{name}', base + delta*(~select), 'scale_remove', note=f'Remove {lo} <= ell < {hi}')
    spectra = np.asarray(rows)
    assert np.all(np.isfinite(spectra))
    # Algebraic checks are independent of posterior sampling.
    for i, seed in enumerate(SEEDS[1:], 1):
        labels = {d['label']: j for j, d in enumerate(definitions)}
        reconstructed = spectra[labels[f'clean_only_{seed}']] + spectra[labels[f'cross_only_{seed}']] - base
        assert np.allclose(reconstructed, noisy[10*i], rtol=1e-12, atol=1e-25)
    return spectra, definitions


def support_audit(features, cf_x, original_x, root):
    pool = features['x_pool']
    held = features['x_held']
    # Evenly spaced reserved parameter rows provide a reference distance distribution.
    ids = np.arange(0, len(held), 5)
    reference, _ = nearest_distance(held[ids], pool)
    cf_distance, cf_neighbor = nearest_distance(cf_x, pool)
    grid_distance, grid_neighbor = nearest_distance(original_x, pool)
    def percentile(values):
        return np.mean(reference[:, None] <= values[None, :], axis=0)
    result = dict(metric='Euclidean distance in the trained standardized asinh/PCA inputs',
                  reference_held_rows=ids, reference_median=np.median(reference),
                  reference_q95=np.quantile(reference, .95), reference_max=reference.max(),
                  counterfactual_distance=cf_distance, counterfactual_percentile=percentile(cf_distance),
                  grid_distance=grid_distance, grid_percentile=percentile(grid_distance),
                  nearest_grid_theta=features['theta_pool'][grid_neighbor],
                  caveat='This empirical support diagnostic is not a formal posterior calibration test.')
    np.savez(root / 'arrays/support.npz', reference_distance=reference, grid_distance=grid_distance,
             cf_distance=cf_distance, cf_neighbor=cf_neighbor, grid_neighbor=grid_neighbor)
    return result


def summarize(samples, low, high):
    q = np.quantile(samples, [.025, .16, .5, .84, .975], axis=0)
    return dict(q025=q[0], q16=q[1], median=q[2], q84=q[3], q975=q[4],
                mean=samples.mean(axis=0), width68=q[3]-q[1],
                outside_prior=np.mean((samples<low)|(samples>high), axis=0))


def evaluate_model(folder, x, root, name, threads, samples, noise_draws=False):
    import torch
    import merlin_compression as mc
    import train_ensemble as te
    start = time.time()
    ens = mc.ensemble(folder, 'merlin', threads)
    torch.manual_seed(20261005)
    u = ens.sample_u(x, samples, rng=np.random.default_rng(20261005))
    theta = te.from_u(u, ens.low, ens.high)
    result = summarize(theta, ens.low, ens.high)
    truth_u = te.to_u(TRUTH[None], ens.low, ens.high)
    result['log_prob_truth'] = ens.log_prob_u(np.repeat(truth_u[None], len(x), axis=1), x)[0]
    result['log_prob_truth'] -= te.log_jacobian_theta_per_u(truth_u, ens.low, ens.high)[0]
    # Physical-theta HPD: apply the parameter-space Jacobian also to samples.
    lp = ens.log_prob_u(u, x) - te.log_jacobian_theta_per_u(u, ens.low, ens.high)
    result['truth_hpd_theta'] = np.mean(lp > result['log_prob_truth'][None], axis=0)
    result['samples_per_case'] = len(u)
    result['param_space'] = os.environ['MERLIN_PARAM_SPACE']
    result['seconds'] = time.time() - start
    # A deterministic evaluation-mode check: conditioning cannot depend on the batch cohort.
    lp_one = ens.log_prob_u(truth_u[None], x[:1])[0, 0]
    lp_batch = ens.log_prob_u(np.repeat(truth_u[None], 2, axis=1), x[:2])[0, 0]
    result['cohort_log_prob_error'] = abs(float(lp_one-lp_batch))
    assert result['cohort_log_prob_error'] < .002
    result['all_modules_eval'] = all(not m.training for net in ens.nets for m in net.modules())
    assert result['all_modules_eval']
    # Compare ensemble members to exclude a single pathological trained network.
    members = []
    actual_ids = [1+i for i in range(10)]
    for index, net in enumerate(ens.nets):
        torch.manual_seed(20261005+index)
        with torch.no_grad():
            member_u = net.sample((samples,), condition=torch.as_tensor(x[actual_ids], dtype=torch.float32)).numpy()
        members.append(summarize(te.from_u(member_u, ens.low, ens.high), ens.low, ens.high))
    result['member_seed_ids'] = ens.seeds
    result['members_actual'] = members
    if noise_draws:
        # These 21 original seed-100 B12 draws are conditional diagnostics. Twenty
        # entered the original whitening covariance; they are not held-out skies.
        torch.manual_seed(20261006)
        nu = ens.sample_u(ens.f['x_draws'], samples, rng=np.random.default_rng(20261006))
        result['training_sky_noise_draws'] = summarize(te.from_u(nu, ens.low, ens.high), ens.low, ens.high)
        result['noise_draws_caveat'] = '20 of 21 draws entered the whitening covariance; conditional on seed 100.'
        # Locate the actual learned embedding rather than relying on an sbi private attribute.
        head = next(m for m in ens.nets[0].modules() if m.__class__.__name__ == 'Head')
        with torch.no_grad():
            h_pool = head(torch.as_tensor(ens.f['x_pool'], dtype=torch.float32)).numpy()
            h_grid = head(torch.as_tensor(np.load(PARENT/'data/original_merlin_observations.npy'), dtype=torch.float32)).numpy()
            h_held = head(torch.as_tensor(ens.f['x_held'][::5], dtype=torch.float32)).numpy()
        hd, _ = nearest_distance(h_grid, h_pool)
        href, _ = nearest_distance(h_held, h_pool)
        result['head_support'] = dict(grid_distance=hd, reference_median=np.median(href),
                                     reference_q95=np.quantile(href, .95),
                                     grid_percentile=np.mean(href[:, None] <= hd[None], axis=0))
        np.savez(root/'arrays/head_features.npz', pool=h_pool, grid=h_grid, held=h_held)
    write_json(root / f'{name}.json', result)
    print('MODEL', name, 'seconds', round(time.time()-start, 1), flush=True)
    return result


def linear_scores(clean_bands, model, low, high):
    """Local finite-derivative geometry, not nonlinear parameter estimates.

    Normalize parameters by their prior widths. Report truncation sensitivity:
    near-null pressure-profile directions make a full nine-dimensional inverse
    unstable. Residuals are mean-spectrum differences at the B12 parameters.
    """
    l = np.linalg.cholesky(model['C'])
    derivative = solve_triangular(l, model['J']*(high-low), lower=True)
    u, s, vt = np.linalg.svd(derivative, full_matrices=False)
    r = (clean_bands[::10]-clean_bands[0])/model['fid']
    wr = solve_triangular(l, r.T, lower=True).T
    out = dict(singular_values=s, relative_singular_values=s/s[0], whitened_clean_norm=np.linalg.norm(wr, axis=1))
    for threshold in [1e-2, 1e-3, 1e-4]:
        keep = s/s[0] > threshold
        shifts = (wr@u[:, keep]/s[keep])@vt[keep]
        leftover = wr - shifts@derivative.T
        out[str(threshold)] = dict(rank=int(keep.sum()), shifts_prior_units=shifts,
                                  residual_norm=np.linalg.norm(leftover, axis=1))
    return out


def figures(root, clean, noisy, unmasked, a, definitions, original, banded, results):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12,
                         'lines.linewidth': 2, 'savefig.bbox': 'tight'})
    out = root/'plots'
    def save(fig, name):
        fig.savefig(out/f'{name}.pdf')
        fig.savefig(out/f'{name}.png', dpi=180)
        plt.close(fig)
    old = json.loads((PARENT/'candidates/original_merlin/evaluation_final.json').read_text())
    band = json.loads((PARENT/'candidates/merlin_band50_full/evaluation_final.json').read_text())
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout='constrained')
    for j, ax in enumerate(axes.flat):
        for record, color, label, dx in [(old, 'C0', 'Original', -.22), (band, 'C1', r'$\Delta\ell=50$', .22)]:
            med=np.asarray(record['median'])[::10, j];lo=np.asarray(record['q16'])[::10, j];hi=np.asarray(record['q84'])[::10,j]
            ax.errorbar(SEEDS+dx, med, yerr=[med-lo, hi-med], fmt='o', color=color, alpha=.8, markersize=4, label=label)
        ax.axhline(TRUTH[j], color='k', ls='--', label='Truth')
        ax.set_title(LABELS[j]);ax.set_xticks([100, 106, 112, 118])
        if j>=6: ax.set_xlabel('Simulation seed')
    axes[0,0].legend()
    save(fig, 'posterior_seed_offsets')
    centers=a@ELL
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    for ax, (lo, hi) in zip(axes, [(80,580),(580,2980),(2980,7980)]):
        keep=(centers>=lo)&(centers<hi)
        ratios=(clean[::10]@a.T)/(clean[0]@a.T)-1
        for row in ratios[1:]: ax.plot(centers[keep],100*row[keep],color='C0',alpha=.25,lw=1)
        ax.plot(centers[keep],100*ratios[1:].mean(0)[keep],color='C0',label='Other seeds: mean')
        fullratios=(unmasked@a.T)/(unmasked[0]@a.T)-1
        ax.plot(centers[keep],100*fullratios[1:].mean(0)[keep],color='C2',ls=':',label='Full sky: mean')
        ax.axhline(0,color='k',ls='--');ax.set_xlabel(r'$\ell$');ax.set_title(f'{lo}–{hi-1}')
    axes[0].set_ylabel(r'$100(D_\ell/D_\ell^{100}-1)$');axes[2].legend()
    save(fig, 'spectra_relative_to_training_sky')
    labels={d['label']:i for i,d in enumerate(definitions)}
    comparisons=[('Actual','actual'),('Clean only','clean_only'),('Cross only','cross_only'),('Reversed clean','reversed_clean')]
    fig,axes=plt.subplots(3,3,figsize=(16,11),layout='constrained')
    for j,ax in enumerate(axes.flat):
        for k,(label,family) in enumerate(comparisons):
            med=np.asarray(original['median'])[[labels[f'{family}_{s}'] for s in SEEDS[1:]],j]
            ax.plot(SEEDS[1:],med,'o-',color=f'C{k}',label=label,markersize=4)
        ax.axhline(original['median'][0,j],color='0.4',ls=':',label='Seed 100')
        ax.axhline(TRUTH[j],color='k',ls='--',label='Truth');ax.set_title(LABELS[j]);ax.set_xticks([102,110,118])
        if j>=6:ax.set_xlabel('Simulation seed')
    axes[0,0].legend(ncol=2,fontsize=10)
    save(fig,'signal_and_cross_term_controls')
    fig,axes=plt.subplots(3,3,figsize=(16,11),layout='constrained')
    fractions=[0,.1,.25,.5,.75,1]
    for j,ax in enumerate(axes.flat):
        for seed in SEEDS[1:]:
            ids=[0]+[labels[f'interpolate_{seed}_{f}'] for f in fractions[1:-1]]+[labels[f'actual_{seed}']]
            ax.plot(fractions,np.asarray(original['median'])[ids,j],color='C0',alpha=.4,lw=1.5)
        ax.axhline(TRUTH[j],color='k',ls='--');ax.set_title(LABELS[j]);ax.set_xticks([0,.5,1])
        if j>=6:ax.set_xlabel('Fraction of sky change')
    save(fig,'posterior_response_paths')
    control_labels=['baseline','mean_noisy','mean_clean','mean_cross','mean_reversed_clean',
                    'smooth_clean_50','fine_only_50','clean_low_only','clean_middle_only','clean_high_only']
    ticklabels=['Seed 100','Mean sky','Clean','Cross term','Reverse','Smooth','Fine residual','Low ell','Mid ell','High ell']
    fig,axes=plt.subplots(3,1,figsize=(14,12),layout='constrained',sharex=True)
    for j,ax in zip([0,2,3],axes):
        ids=[labels[label] for label in control_labels]
        for rec,color,label,dx in [(original,'C0','Original',-.13),(banded,'C1',r'$\Delta\ell=50$',.13)]:
            med=np.asarray(rec['median'])[ids,j];lo=np.asarray(rec['q16'])[ids,j];hi=np.asarray(rec['q84'])[ids,j]
            ax.errorbar(np.arange(len(ids))+dx,med,yerr=[med-lo,hi-med],fmt='o',color=color,label=label,capsize=2)
        ax.axhline(TRUTH[j],color='k',ls='--');ax.set_ylabel(LABELS[j])
    axes[0].legend();axes[-1].set_xticks(np.arange(len(ids)),ticklabels,rotation=35,ha='right')
    save(fig,'counterfactual_posterior_response')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    q=np.asarray(results['support']['grid_percentile'])[::10]
    axes[0].plot(SEEDS,q*100,'o-',label='Input nearest neighbour')
    q=np.asarray(original['head_support']['grid_percentile'])[::10]
    axes[0].plot(SEEDS,q*100,'s-',label='Learned head')
    axes[0].set_xlabel('Simulation seed');axes[0].set_ylabel('Held-out distance percentile');axes[0].legend()
    eps=np.asarray(results['spectral_metrics']['retained_pca_clean_distance'])[::10]
    axes[1].plot(SEEDS,eps,'o-');axes[1].set_xlabel('Simulation seed')
    axes[1].set_ylabel(r'$\|V^T L^{-1}(D_{\rm clean}-D_{\rm clean}^{100})\|$')
    save(fig,'feature_support_and_noise_distance')


def main(args):
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    root=args.root
    for directory in ['arrays','plots','logs']:(root/directory).mkdir(parents=True,exist_ok=True)
    start=time.time()
    plan,clean,noisy,unmasked,a,contracts,error=load_signals()
    spectra,definitions=counterfactuals(clean,noisy)
    write_json(root/'counterfactuals.json',definitions)
    np.savez(root/'arrays/spectra.npz',clean=clean,noisy=noisy,unmasked=unmasked,counterfactual=spectra)
    print('SPECTRA',len(spectra),flush=True)
    params=dict(np.load(OLD/'models/linear/features_merlin.npz'))
    with np.load(OLD/'reference/whitening_pca.npz') as whitener:
        l=whitener['L'];v=whitener['V']
    whitened=solve_triangular(l,(spectra-noisy[0]).T,lower=True).T
    # Project absolute spectra for exact reconstruction of the existing observation.
    projected=solve_triangular(l,spectra.T,lower=True).T@v
    x=transform(projected,params)
    original_x=np.load(PARENT/'data/original_merlin_observations.npy')
    assert np.max(abs(x[1:11]-original_x[::10]))<1e-5
    clean_white=solve_triangular(l,(clean-clean[0]).T,lower=True).T
    spectral_metrics=dict(full_noiseonly_clean_distance=np.linalg.norm(clean_white,axis=1),
                          retained_pca_clean_distance=np.linalg.norm(clean_white@v,axis=1),
                          counterfactual_whitened_distance=np.linalg.norm(whitened,axis=1),
                          counterfactual_pca_distance=np.linalg.norm(whitened@v,axis=1))
    print('FEATURES exact error',np.max(abs(x[1:11]-original_x[::10])),flush=True)
    support=support_audit(params,x,original_x,root)
    print('SUPPORT seed percentiles',np.array(support['grid_percentile'])[::10],flush=True)
    model=dict(np.load(PARENT/'data/local_model.npz'))
    scores=linear_scores(clean@a.T,model,params['low'],params['high'])
    original_eval=json.loads((PARENT/'candidates/original_merlin/evaluation_final.json').read_text())
    med=np.asarray(original_eval['median'])[::10]
    relatives=(clean[::10]@a.T)/(clean[0]@a.T)-1
    blocks={}
    centers=a@ELL
    for name,lo,hi in [('low',80,580),('middle',580,2980),('high',2980,7980)]:
        keep=(centers>=lo)&(centers<hi)
        r=relatives[:,keep]
        # Baseline rank among ten realizations: this is descriptive, with n=10.
        b=(clean[::10]@a.T)[:,keep]
        full=(unmasked@a.T)[:,keep]
        blocks[name]=dict(mean_other_relative=r[1:].mean(),other_scatter=np.std(r[1:],axis=0,ddof=1).mean(),
                          baseline_above_other_fraction=np.mean(b[0]>b[1:]),
                          baseline_above_fullsky_other_fraction=np.mean(full[0]>full[1:]),
                          per_seed_mean_relative=r.mean(axis=1))
    # A common fixed reference invalidates independent random-sign reasoning.
    mean_shift=relatives[1:].mean(axis=0)
    common_norm=np.linalg.norm(mean_shift)
    scatter=np.sqrt(np.mean(np.sum((relatives[1:]-mean_shift)**2,axis=1)))
    common_cosines=relatives[1:]@mean_shift/(np.linalg.norm(relatives[1:],axis=1)*common_norm)
    parameter_offsets=med[1:]-TRUTH
    directional=dict(mean_spectral_shift_norm=common_norm,scatter_about_other_mean_norm=scatter,
                     common_spectral_cosines=common_cosines,
                     all_other_median_above_truth=np.all(parameter_offsets>0,axis=0),
                     all_other_median_below_truth=np.all(parameter_offsets<0,axis=0),
                     other_seed_median_mean=med[1:].mean(axis=0),
                     other_seed_median_sd=med[1:].std(axis=0,ddof=1),
                     seed100_median=med[0],
                     pressure_amplitude_only_equivalent=np.sqrt(1+relatives[1:].mean(axis=0)))
    results=dict(parent=PARENT,physics_contract_keys_checked=list(contracts[0]),contracts=contracts,
                 exact_band_reconstruction_error=error,
                 exact_original_feature_error=np.max(abs(x[1:11]-original_x[::10])),
                 spectral_metrics=spectral_metrics,support=support,linear_scores=scores,
                 spectral_blocks=blocks,directional_offsets=directional,
                 common_reference_caveat='All differences share the same seed-100 reference; they are correlated.',
                 parameter_order=NAMES,truth=TRUTH,prior_low=params['low'],prior_high=params['high'])
    write_json(root/'diagnostics.json',results)
    np.savez(root/'arrays/model_inputs.npz',original=x,definitions=np.array([d['label'] for d in definitions]))
    if args.features_only:return
    sys.path.insert(0,str(PARENT/'snapshot/inference'))
    original=evaluate_model(OLD/'models/linear',x,root,'original_linear',args.threads,args.samples,noise_draws=True)
    comp=dict(np.load(PARENT/'candidates/merlin_band50_full/compression.npz'))
    bx=transform(((spectra@a.T)/comp['fid'])@comp['weights'],comp)
    banded=evaluate_model(PARENT/'candidates/merlin_band50_full',bx,root,'band50_linear',args.threads,args.samples)
    np.save(root/'arrays/band50_inputs.npy',bx)
    figures(root,clean,noisy,unmasked,a,definitions,original,banded,results)
    # Bounded inference needs a fresh interpreter because its transformation is
    # installed at import time by the original frozen training code.
    import subprocess
    env=os.environ.copy();env['MERLIN_PARAM_SPACE']='logit'
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),
                    '--threads',str(args.threads),'--samples',str(args.samples),'--bounded'],
                   check=True,env=env)
    outputs=[p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.npy','.csv','.pdf','.png']
             and 'logs' not in p.parts and p.name!='COMPLETE.json']
    write_json(root/'COMPLETE.json',dict(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               seconds=time.time()-start,counterfactuals=len(definitions),original_observations=100,
               sha256={str(p.relative_to(root)):digest(p) for p in outputs},
               parent_complete_sha256=digest(PARENT/'comparison/COMPLETE.json')))
    print('COMPLETE',time.time()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/lustre/work/kristero10/tsz_seed_offset_diagnosis_20261005'))
    parser.add_argument('--threads',type=int,default=26)
    parser.add_argument('--samples',type=int,default=9000)
    parser.add_argument('--features-only',action='store_true')
    parser.add_argument('--bounded',action='store_true')
    args=parser.parse_args()
    if args.bounded:
        sys.path.insert(0,str(PARENT/'snapshot/inference'))
        x=np.load(args.root/'arrays/model_inputs.npz')['original']
        evaluate_model(OLD/'models/logit',x,args.root,'original_bounded',args.threads,args.samples)
        bx=np.load(args.root/'arrays/band50_inputs.npy')
        evaluate_model(PARENT/'candidates/merlin_band50_full/bounded',bx,args.root,'band50_bounded',args.threads,args.samples)
    else:
        main(args)
