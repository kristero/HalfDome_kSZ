#!/usr/bin/env python3
"""Apply frozen Merlin inference to all seed/mask observations and compare posteriors.

The inference input is the signed, beam-corrected pseudo-D_ell at ell=80..7979,
with the SAVED noise whitener, PCA and asinh transform. Plotting bins are never
fed to the network. No model, compression or whitening is fitted on these mocks.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import solve_triangular

ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
LABELS = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_{P,M}$', r'$\alpha_{x,M}$',
          r'$\alpha_{\beta,M}$', r'$\alpha_{P,z}$', r'$\alpha_{x,z}$', r'$\alpha_{\beta,z}$']


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cl(path):
    value = np.load(path)
    assert value.shape == (7980,) and np.isfinite(value).all(), path
    return value


def correction(clean, noisy, nn, beam):
    return clean + beam * (noisy - clean - nn) + beam**2 * nn


def transform(root, spectra):
    """Uncentered L^-1 D projection, then the stored asinh transform, exactly as training."""
    with np.load(root / 'reference/whitening_pca.npz') as w:
        white = solve_triangular(w['L'], np.atleast_2d(spectra).T, lower=True).T
        projections = white @ w['V']
    with np.load(root / 'models/linear/features_merlin.npz') as f:
        z = (np.arcsinh((projections - f['s_center']) / f['s_scale']) - f['z_mean']) / f['z_sd']
    assert np.isfinite(z).all()
    return z.astype(np.float32), projections, white


def baseline(root, fresh=False):
    """Reproduce the stored training observation before accepting the new experiment."""
    ref = root / 'reference'
    clean, noisy, nn, beam = [cl(ref / n) for n in (
        'masked_clean_cl.npy', 'noisy_cross_39001001_39001002.npy', 'noise_cl.npy', 'beam_bl.npy')]
    spectrum = correction(clean, noisy, nn, beam)[ELL] * DL
    z, s, _ = transform(root, spectrum)
    with np.load(ref / 'b12.npz') as f:
        projection_error = float(np.max(np.abs(s - f['s_obs'][:1])) / np.max(np.abs(f['s_obs'][:1])))
    errors = dict(stored_projection_relative_error=projection_error)
    assert projection_error < 1e-7, errors
    for space in ('linear', 'logit'):
        with np.load(root / f'models/{space}/features_merlin.npz') as f:
            errors[f'{space}_stored_feature_max_error'] = float(np.max(np.abs(z - f['x_obs'][:1])))
            assert errors[f'{space}_stored_feature_max_error'] < 1e-4, errors
    if fresh:
        target = root / 'results/seed_100/mask_12345'
        assert (target / 'complete.toml').exists()
        for name, reference in [('clean_cl', clean), ('production_cl', noisy), ('noise_cl', nn)]:
            value = cl(target / f'{name}.npy')
            error = float(np.max(np.abs(value[ELL] - reference[ELL])) / np.max(np.abs(reference[ELL])))
            errors[f'new_{name}_relative_error'] = error
            assert error < 1e-6, errors
        new_z, _, _ = transform(root, cl(target / 'beam_corrected_cl.npy')[ELL] * DL)
        errors['new_feature_max_error'] = float(np.max(np.abs(new_z - z)))
        assert errors['new_feature_max_error'] < 1e-3, errors
    return errors


def preflight(root):
    frozen = json.loads((root / 'input_sha256.json').read_text())
    for name, expected in frozen.items():
        assert sha(root / name) == expected, f'Frozen input changed: {name}'
    checks = baseline(root)
    checks['frozen_files_verified'] = len(frozen)
    write_json(root / 'analysis/preflight.json', checks)
    print(json.dumps(checks), flush=True)


def collect(root, plan):
    spectra = {}
    noise_hashes = []
    import tomllib
    for seed in plan['simulation_seeds']:
        with (root / f'results/seed_{seed}/complete.toml').open('rb') as f:
            meta = tomllib.load(f)
        assert meta['simulation_seed'] == seed and meta['theta'] == plan['truth']
        noise_hashes.append(meta['noise_sha256'])
    assert all(x == noise_hashes[0] for x in noise_hashes), 'Noise realization changed between simulations'
    for case in plan['cases']:
        folder = root / case['path']
        with (folder / 'complete.toml').open('rb') as f:
            meta = tomllib.load(f)
        assert meta['simulation_seed'] == case['simulation_seed'] and meta['mask_seed'] == case['mask_seed']
        assert meta['noise_seeds'] == plan['noise_seeds']
        for name, expected in meta['spectrum_sha256'].items():
            assert sha(folder / name) == expected, folder / name
        spectra[case['label']] = {name: cl(folder / f'{name}.npy')[ELL] * DL for name in
                                 ('clean_cl', 'noise_cl', 'production_cl', 'beam_corrected_cl', 'exact_beam_cl')}
        beam = cl(folder.parent / 'beam_bl.npy')[ELL]
        s = spectra[case['label']]
        np.testing.assert_allclose(s['beam_corrected_cl'], correction(
            s['clean_cl'], s['production_cl'], s['noise_cl'], beam), rtol=1e-10, atol=1e-25)
    return spectra


def infer(root, space, threads):
    # Parameter-space selection must happen before importing the existing Merlin module.
    os.environ['MERLIN_PARAM_SPACE'] = space
    os.environ.pop('MERLIN_TRAIN_ROWS', None)
    sys.path.insert(0, str(root / 'snapshot/inference'))
    import torch
    import merlin_compression as mc
    from train_ensemble import from_u
    plan = json.loads((root / 'plan.json').read_text())
    features = np.load(root / 'analysis/features.npz')
    ens = mc.ensemble(root / f'models/{space}', 'merlin', threads)
    out = root / f'analysis/{space}'
    out.mkdir(exist_ok=True)
    records = {}
    for i, case in enumerate(plan['cases']):
        path = out / f"{case['label']}.npy"
        torch.manual_seed(plan['posterior_rng_seed'])
        # Same random base draws in each posterior reduce Monte Carlo noise in differences.
        # No need to evaluate density at every draw merely to compute quantiles.
        u = ens.sample_u(features['x'][i:i+1], plan['samples_per_posterior'],
                         rng=np.random.default_rng(plan['posterior_rng_seed']))
        samples = from_u(u[:, 0], ens.low, ens.high).astype(np.float32)
        assert samples.shape == (plan['samples_per_posterior'], 9) and np.isfinite(samples).all()
        np.save(path, samples)
        q = np.quantile(samples, [.025, .16, .5, .84, .975], axis=0)
        inside = (samples >= ens.low) & (samples <= ens.high)
        # Ten disjoint sample batches provide a Monte Carlo precision diagnostic, not a sky covariance.
        batch_q = np.array([np.quantile(a, [.16, .5, .84], axis=0) for a in np.array_split(samples, 10)])
        median_mcse = np.std(batch_q[:, 1], axis=0, ddof=1) / np.sqrt(10)
        width_mcse = np.std(batch_q[:, 2] - batch_q[:, 0], axis=0, ddof=1) / np.sqrt(10)
        records[case['label']] = dict(q025=q[0].tolist(), q16=q[1].tolist(), median=q[2].tolist(),
            q84=q[3].tolist(), q975=q[4].tolist(), width68=(q[3]-q[1]).tolist(),
            mean=samples.mean(0).tolist(), sd=samples.std(0, ddof=1).tolist(),
            outside_training_prior_per_parameter=(1-inside.mean(0)).tolist(),
            outside_training_prior_joint=float(1-inside.all(1).mean()),
            median_mcse=median_mcse.tolist(), width68_mcse=width_mcse.tolist(), sha256=sha(path))
        print(space, case['label'], 'median', np.round(q[2], 4), flush=True)
    write_json(out / 'posteriors.json', dict(space=space, samples=plan['samples_per_posterior'], cases=records))


def rebin(x):
    w = 2 * ELL + 1
    return np.stack([np.average(x[..., a-80:b-80], weights=w[a-80:b-80], axis=-1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=-1)


def plots_and_tables(root, plan, spectra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 17, 'axes.titlesize': 17,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
                         'savefig.dpi': 220, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    out = root / 'analysis'
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    arms = dict(simulation=[c for c in plan['cases'] if c['mask_seed'] == 12345],
                mask=[c for c in plan['cases'] if c['simulation_seed'] == 100])
    labels = dict(simulation='Simulation seed', mask='Mask seed')
    colors = plt.get_cmap('tab10')

    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(figures / f'{name}.{ext}', bbox_inches='tight')
        plt.close(fig)

    ell = rebin(ELL)
    reference = spectra[plan['baseline']]
    for arm, cases in arms.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout='constrained')
        for col, (kind, title) in enumerate([('clean_cl', 'Signal'), ('beam_corrected_cl', 'SO baseline deproj0')]):
            base = rebin(reference[kind])
            scale = rebin(reference['clean_cl'])
            for i, c in enumerate(cases):
                value = rebin(spectra[c['label']][kind])
                label = str(c['simulation_seed'] if arm == 'simulation' else c['mask_seed'])
                axes[0, col].plot(ell, value, color=colors(i), label=label)
                axes[1, col].plot(ell, (value-base)/scale, color=colors(i))
            axes[0, col].set_title(title)
            axes[1, col].axhline(0, color='k', lw=.7)
            axes[1, col].set_xlabel(r'$\ell$')
            axes[0, col].set_yscale('symlog', linthresh=1e-14)
        axes[0, 0].set_ylabel(r'$\widetilde D_\ell^{yy}$')
        axes[1, 0].set_ylabel(r'$\Delta D_\ell / D_{\ell,0}^{\rm signal}$')
        axes[0, 1].legend(title=labels[arm], ncol=2)
        save(fig, f'spectra_{arm}')
    # Unmasked spectra isolate the total-sky variation before the observing window.
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    for i, seed in enumerate(plan['simulation_seeds']):
        value = cl(root / f'results/seed_{seed}/unmasked_clean_cl.npy')[ELL] * DL
        ax.plot(ell, rebin(value), color=colors(i), label=str(seed))
    ax.set(xlabel=r'$\ell$', ylabel=r'$D_\ell^{yy}$', yscale='log')
    ax.legend(title='Simulation seed', ncol=2)
    save(fig, 'spectra_fullsky')

    group_records, rows = {}, []
    for space in plan['spaces']:
        records = json.loads((out / space / 'posteriors.json').read_text())['cases']
        base = records[plan['baseline']]
        base_med, base_width = np.array(base['median']), np.array(base['width68'])
        group_records[space] = {}
        for arm, cases in arms.items():
            med = np.array([records[c['label']]['median'] for c in cases])
            widths = np.array([records[c['label']]['width68'] for c in cases])
            shift = (med-base_med) / (base_width/2)
            ratio = widths/base_width
            group_records[space][arm] = dict(
                median_scatter_over_baseline_halfwidth=(med.std(0, ddof=1)/(base_width/2)).tolist(),
                rms_shift_over_baseline_halfwidth=np.sqrt(np.mean(shift**2, axis=0)).tolist(),
                max_abs_shift_over_baseline_halfwidth=np.max(np.abs(shift), axis=0).tolist(),
                mean_width_ratio=ratio.mean(0).tolist(), width_ratio_sd=ratio.std(0, ddof=1).tolist(),
                width_ratio_min=ratio.min(0).tolist(), width_ratio_max=ratio.max(0).tolist())
            fig, axes = plt.subplots(3, 3, figsize=(17, 13), layout='constrained')
            x = np.arange(len(cases))
            ticks = [str(c['simulation_seed'] if arm == 'simulation' else c['mask_seed']) for c in cases]
            for j, ax in enumerate(axes.flat):
                lo = np.array([records[c['label']]['q16'][j] for c in cases])
                hi = np.array([records[c['label']]['q84'][j] for c in cases])
                ax.axhspan(base['q16'][j], base['q84'][j], color='.85', label='Baseline 68%')
                ax.axhline(plan['truth'][j], color='k', ls='--', label='B12 truth')
                ax.axhline(base_med[j], color='.4', lw=1)
                ax.errorbar(x, med[:, j], yerr=[med[:, j]-lo, hi-med[:, j]], fmt='o', capsize=3, color='#1b6b9c')
                ax.set_title(LABELS[j]); ax.set_xticks(x, ticks, rotation=60)
                if j >= 6: ax.set_xlabel(labels[arm])
            axes[0, 0].legend()
            save(fig, f'parameters_{space}_{arm}')
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
            bound = max(1, float(np.max(np.abs(shift))))
            im = axes[0].imshow(shift.T, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
            fig.colorbar(im, ax=axes[0], label=r'$\Delta\mathrm{median}/(w_{68,0}/2)$')
            im = axes[1].imshow(ratio.T, aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=axes[1], label=r'$w_{68}/w_{68,0}$')
            for ax in axes:
                ax.set_yticks(np.arange(9), LABELS); ax.set_xticks(x, ticks, rotation=60)
                ax.set_xlabel(labels[arm])
            save(fig, f'shifts_widths_{space}_{arm}')
        for c in plan['cases']:
            r = records[c['label']]
            for j, parameter in enumerate(plan['parameters']):
                rows.append(dict(space=space, case=c['label'], simulation_seed=c['simulation_seed'],
                    mask_seed=c['mask_seed'], parameter=parameter, truth=plan['truth'][j],
                    median=r['median'][j], q16=r['q16'][j], q84=r['q84'][j], width68=r['width68'][j],
                    median_shift=r['median'][j]-base_med[j],
                    shift_over_baseline_halfwidth=(r['median'][j]-base_med[j])/(base_width[j]/2),
                    width_ratio=r['width68'][j]/base_width[j], median_mcse=r['median_mcse'][j],
                    width68_mcse=r['width68_mcse'][j],
                    outside_training_prior=r['outside_training_prior_per_parameter'][j]))
    with (out / 'parameter_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out / 'variation_summary.json', group_records)
    primary = group_records['linear']
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, layout='constrained')
    x = np.arange(9)
    for offset, (arm, color) in zip([-.17, .17], [('simulation', '#1b6b9c'), ('mask', '#d16b22')]):
        r = primary[arm]
        axes[0].bar(x+offset, r['median_scatter_over_baseline_halfwidth'], width=.32, color=color, label=labels[arm])
        axes[1].errorbar(x+offset, r['mean_width_ratio'], yerr=r['width_ratio_sd'], fmt='o', capsize=4, color=color)
    axes[0].set_ylabel(r'$\mathrm{SD}(\mathrm{median})/(w_{68,0}/2)$')
    axes[0].legend(); axes[1].set_ylabel(r'$w_{68}/w_{68,0}$'); axes[1].axhline(1, color='.4', lw=1)
    axes[1].set_xticks(x, LABELS)
    save(fig, 'variation_unbounded_primary')
    lines = ['Seed and mask sensitivity of frozen Merlin inference', '',
             'Primary: unbounded model. Companion: prior-bounded model.',
             'Baseline: simulation 100, mask 12345; same SO baseline deproj0 split noise in every observation.',
             'Widths are q84-q16. A normalized shift of one is half the baseline 68% interval.',
             'Mask rotations overlap on the same sky. Their scatter is spatial variation, not independent-mock covariance.',
             'This tests sensitivity conditional on one noise pair; it is not a coverage calibration or a retrained posterior.',
             'Unbounded tail mass outside the training prior is extrapolation; it is tabulated explicitly.', '',
             'parameter, simulation median scatter / baseline half-width, mask median scatter / baseline half-width']
    for j, name in enumerate(plan['parameters']):
        lines.append(f"{name}, {primary['simulation']['median_scatter_over_baseline_halfwidth'][j]:.4f}, "
                     f"{primary['mask']['median_scatter_over_baseline_halfwidth'][j]:.4f}")
    (out / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')


def main(args):
    root = args.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if args.preflight:
        preflight(root); return
    if args.check_seed100:
        checks = baseline(root, fresh=True)
        write_json(root / 'analysis/baseline_check.json', checks)
        print(json.dumps(checks), flush=True); return
    if args.space:
        infer(root, args.space, args.threads); return
    preflight(root)
    checks = baseline(root, fresh=True)
    spectra = collect(root, plan)
    x = np.array([spectra[c['label']]['beam_corrected_cl'] for c in plan['cases']])
    exact = np.array([spectra[c['label']]['exact_beam_cl'] for c in plan['cases']])
    features, _, white = transform(root, x)
    exact_features, _, exact_white = transform(root, exact)
    np.savez(root / 'analysis/features.npz', x=features, exact_beam_x=exact_features,
             labels=[c['label'] for c in plan['cases']], beam_dl=x, exact_beam_dl=exact)
    checks['beam_approximation_noise_whitened_norm'] = np.linalg.norm(exact_white-white, axis=1).tolist()
    checks['beam_approximation_max_feature_change'] = np.max(np.abs(exact_features-features), axis=1).tolist()
    checks['labels'] = [c['label'] for c in plan['cases']]
    write_json(root / 'analysis/validation.json', checks)
    for space in plan['spaces']:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                        '--space', space, '--threads', str(args.threads)], check=True)
    plots_and_tables(root, plan, spectra)
    files = {str(p.relative_to(root)): sha(p) for p in (root / 'analysis').rglob('*')
             if p.is_file() and p.name != 'COMPLETE.json'}
    write_json(root / 'analysis/COMPLETE.json', dict(cases=len(plan['cases']), spaces=plan['spaces'], sha256=files))
    print('COMPLETE', root / 'analysis', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--threads', type=int, default=26)
    p.add_argument('--space', choices=['linear', 'logit'])
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--check-seed100', action='store_true')
    main(p.parse_args())
