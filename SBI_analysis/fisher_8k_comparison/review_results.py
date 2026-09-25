"""Verify the completed cluster comparison and summarize its scientific limits.

The cluster export stays unchanged. This creates a separate review directory,
including a covariance-size check with the exact joint prior used for inference.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze import FIDUCIAL, LABELS, NAMES, joint_fisher_samples
from calibration import digest, write_json
from so_nine_fisher import conditional_covariance, fisher_matrix


def verify_export(root):
    """Verify every final artifact, calibration task and frozen analysis source."""
    comparison = root / 'comparison'
    complete = json.loads((comparison / 'complete.json').read_text())
    assert complete['complete'] and not complete['synthetic_test']
    for name, expected in complete['artifacts'].items():
        assert digest(comparison / name) == expected, name
    for line in (root / 'analysis_sources.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert digest(root / name) == expected, name
    manifest = json.loads((root / 'manifest.json').read_text())
    noise_hashes = []
    for index, task in enumerate(manifest['tasks']):
        folder = root / 'tasks' / f'task{index:03d}'
        marker = json.loads((folder / 'complete.json').read_text())
        assert marker['manifest_sha256'] == digest(root / 'manifest.json')
        assert marker['spectra_sha256'] == digest(folder / 'spectra.npz')
        assert marker['task'] == task
        noise_hashes.extend(marker['operator']['noise_pixel_hashes'])
    assert len(set(noise_hashes)) == len(noise_hashes) == 66
    return manifest, complete


def check_covariance_size(calibration, prior, output, full_info, draws):
    """Compare 16 and 32 draws using the actual joint prior, not Gaussian moments."""
    covariance, shrinkage = conditional_covariance(calibration['ensemble'][:16])
    infos = []
    for seed in (20260921, 20260922):
        _, info = joint_fisher_samples(calibration['jacobian'], covariance, np.zeros(40),
                                      prior, draws=draws, count=1, seed=seed)
        infos.append(info)
    mean_shift = np.abs(np.asarray(infos[0]['weighted_mean']) - infos[1]['weighted_mean'])
    mean_shift /= infos[0]['weighted_std']
    std_change = np.abs(np.asarray(infos[1]['weighted_std']) / infos[0]['weighted_std'] - 1)
    assert mean_shift.max() < .05 and std_change.max() < .05, 'Repeat integration failed'
    ratio = np.asarray(infos[0]['weighted_std']) / full_info['weighted_std']
    result = dict(noise_count=16, oas_shrinkage=shrinkage, integrations=infos,
                  max_repeat_mean_shift_over_std=float(mean_shift.max()),
                  max_repeat_std_fractional_change=float(std_change.max()),
                  sigma16_over_sigma32=ratio.tolist(),
                  maximum_width_change=float(np.max(np.abs(ratio - 1))),
                  scope='Forecast only; exact joint prior; nested 16/32 ensembles; not covariance convergence')
    write_json(output / 'covariance_size_check.json', result)
    return result


def make_summary_plot(output, audit, ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'bins40': '#0072B2', 'pca': '#CC79A7', 'moped': '#D55E00'}
    names = {'bins40': '40-bin SBI', 'pca': 'PCA SBI', 'moped': 'MOPED SBI'}
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 15})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), gridspec_kw={'width_ratios': [1, 1.35]})
    for method in ('pca', 'moped'):
        fractions = audit[method]['resolved_mode_information_fractions']
        axes[0].semilogy(np.arange(1, len(fractions) + 1), fractions, 'o-',
                        color=colors[method], lw=2, label=method.upper())
    axes[0].axhline(1, color='.25', ls='--', lw=1.3, label='Full 40 bins')
    axes[0].set(xlabel='Ordered generalized mode', ylabel='Retained Fisher information',
                xticks=np.arange(1, 7), ylim=(2e-8, 2), title='Local compression')
    axes[0].legend(fontsize=12, loc='lower right')
    axes[0].grid(alpha=.18)
    for method in ('bins40', 'pca', 'moped'):
        axes[1].plot(np.arange(9), ratios[method], 'o-', color=colors[method],
                     lw=2, label=names[method])
    axes[1].axhline(1, color='.25', ls='--', lw=1.3)
    axes[1].set(xticks=np.arange(9), xticklabels=['$' + label + '$' for label in LABELS],
                ylabel=r'$\sigma_{\rm SBI}/\sigma_{\rm Fisher}$', title='Forecast marginal widths')
    axes[1].tick_params(axis='x', rotation=45, labelsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(axis='y', alpha=.18)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(output / f'battaglia12_comparison_summary.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Locally verified cluster export')
    parser.add_argument('--output', type=Path, required=True, help='Fresh review directory')
    parser.add_argument('--draws', type=int, default=1000000)
    args = parser.parse_args()
    assert not (args.output / 'complete.json').exists(), 'Preserve the completed review'
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, complete = verify_export(args.root)
    comparison = args.root / 'comparison'
    frozen_code = HERE.parent / 'flamingo_linear_prior/cluster_results/code'
    assert digest(frozen_code / 'prior.py') == manifest['source_code_sha256']['prior.py']
    sys.path.insert(0, str(frozen_code))
    from prior import JointPrior
    prior = JointPrior(manifest['prior'])
    cal = dict(np.load(comparison / 'calibration.npz'))
    calibration = json.loads((comparison / 'calibration_complete.json').read_text())
    audit = json.loads((comparison / 'compression_audit.json').read_text())
    np.testing.assert_allclose(fisher_matrix(cal['jacobian'] * (prior.high - prior.low), cal['covariance']),
                               cal['fisher'], rtol=1e-10, atol=1e-10)
    summaries, information = {}, {}
    rows = []
    for case in ('forecast', 'observed'):
        summaries[case] = {}
        for method in ('full', 'fisher_pca', 'fisher_moped', 'bins40', 'pca', 'moped'):
            filename = (f'fisher_{case}_{method.removeprefix("fisher_")}.npz'
                        if method == 'full' or method.startswith('fisher_') else f'sbi_{case}_{method}.npz')
            values = np.load(comparison / filename)
            samples = values['samples'].astype(np.float64)
            assert np.isfinite(samples).all() and prior.contains(samples).all(), filename
            info = json.loads(str(values['info_json']))
            information[case + '/' + method] = info
            quantiles = np.quantile(samples, [.025, .16, .5, .84, .975], axis=0)
            summary = dict(mean=samples.mean(0).tolist(), std=samples.std(0, ddof=1).tolist(),
                           q025=quantiles[0].tolist(), q16=quantiles[1].tolist(),
                           median=quantiles[2].tolist(), q84=quantiles[3].tolist(), q975=quantiles[4].tolist(),
                           count=len(samples), truth_in_central95=((FIDUCIAL >= quantiles[0]) &
                                                                  (FIDUCIAL <= quantiles[4])).tolist())
            summaries[case][method] = summary
            for index, parameter in enumerate(NAMES):
                row = dict(case=case, method=method, parameter=parameter, truth=FIDUCIAL[index])
                row.update({key: value[index] for key, value in summary.items() if key != 'count'})
                rows.append(row)
    ratio = {method: (np.asarray(summaries['forecast'][method]['std']) /
                      summaries['forecast']['full']['std']).tolist()
             for method in ('bins40', 'pca', 'moped', 'fisher_pca', 'fisher_moped')}
    covariance_check = check_covariance_size(cal, prior, args.output, information['forecast/full'], args.draws)
    result = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(), source_complete_sha256=digest(comparison / 'complete.json'),
                  verified_artifacts=len(complete['artifacts']), verified_tasks=len(manifest['tasks']),
                  covariance_size_check=covariance_check, summaries=summaries,
                  forecast_sigma_over_full_fisher=ratio, fisher_eigenvalues=cal['eigenvalues'].tolist(),
                  resolved_modes_relative_cutoff=1e-8, numerical_resolved_modes=int(cal['resolved'].sum()),
                  inference_checks=information, compression=audit)
    write_json(args.output / 'review.json', result)
    with (args.output / 'constraints_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_summary_plot(args.output, audit, ratio)
    report = [
        '# Completed Battaglia12 Fisher and independent-noise 8K SBI comparison', '',
        'All 41 map-calibration tasks and the dependent analysis completed. This review verified '
        f"{len(complete['artifacts'])} exported artifacts, the frozen analysis sources and all task checksums.", '',
        '## Main result', '',
        'The saved MOPED compression retains most of the local mean-spectrum Fisher information. '
        'Its projected Fisher forecast is close to the full 40-bin forecast. PCA loses much more local '
        'information. The MOPED SBI posterior still differs substantially from its own Fisher forecast; '
        'compression loss alone does not explain that difference.', '',
        'The primary comparison uses the same Battaglia12 mean spectrum, the same fixed sky, the '
        'same noise prescription and the exact same coupled 8K prior. A second comparison uses the '
        'same independent noisy observation for all methods. These are expected-data and single-observation '
        'comparisons, respectively; they are not coverage tests.', '',
        '## Forecast marginal standard deviations', '',
        'These are standard deviations of the Fisher likelihood times the exact joint prior, or of '
        'the corresponding SBI posterior. They are not diagonal errors from an unregularized inverse Fisher matrix.', '',
        '| Parameter | Full Fisher + prior | MOPED Fisher + prior | 40-bin SBI | PCA SBI | MOPED SBI |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for index, name in enumerate(NAMES):
        entries = [summaries['forecast'][method]['std'][index]
                   for method in ('full', 'fisher_moped', 'bins40', 'pca', 'moped')]
        report.append('| ' + name + ' | ' + ' | '.join(f'{value:.4g}' for value in entries) + ' |')
    report += [
        '', '## What the calculation means physically', '',
        'The nine gNFW parameters change pressure normalization, radial shape, and mass/redshift evolution. '
        'The derivatives measure their effect on the masked, beam-smoothed cross-bandpowers at Battaglia12. '
        'The two independent noise splits give zero mean cross-noise bias while retaining signal-noise '
        'and noise-noise fluctuations in the covariance.', '',
        '`F = J.T C^-1 J` uses one OAS-regularized conditional covariance. The saved asinh/PCA/MOPED '
        'transforms are linearized at the fiducial: `C_y = A.T C A` and `J_y = A.T J`. PCA training '
        'variance is used to select axes; it is not substituted for the conditional noise covariance. '
        'SBI does not receive an explicit covariance matrix; it learns the noisy simulator distribution.', '',
        f"The prior-width-scaled Fisher matrix has {int(cal['resolved'].sum())} numerical modes above the "
        '`1e-8 * lambda_max` cutoff. This is a numerical resolution criterion, not six individually '
        'well-measured parameters. All nine coordinates are retained in prior-restricted sampling.', '',
        'SBI can represent nonlinear, curved degeneracies and parameter-dependent noise that a local '
        'fixed-covariance Fisher likelihood omits. Finite training and density-estimator errors can also '
        'contribute. The present comparison cannot determine their separate contributions or establish '
        'SBI coverage. A narrower SBI marginal is not proof that it exceeds the Fisher information bound.', '',
        '## Checks and limitations', '',
        f"- Maximum finite-difference discrepancy: {100 * max(calibration['derivative_small_relative_error']):.5f}%.",
        f"- Archived fiducial spectrum agreement: {calibration['archived_fiducial_max_relative_error']:.3g} relative.",
        f"- OAS correlation shrinkage: {calibration['oas_shrinkage']:.6f} with 32 noise pairs.",
        f"- Exact-prior forecast widths change by at most {100 * covariance_check['maximum_width_change']:.2f}% "
        'when using the first 16 rather than 32 pairs. The nested ensemble and strong shrinkage do not establish covariance convergence.',
        f"- Freshly optimal MOPED reproduces the full fitted Fisher matrix to {calibration['optimal_moped_fisher_relative_error']:.3g} relative. "
        'This algebra check uses a different basis from the saved SBI transform.',
        '- The saved-transform covariance propagation is a local approximation: noise-ensemble tangent errors '
        f"are {100 * audit['moped']['tangent_rms_error_over_fluctuation']:.2f}% for MOPED and "
        f"{100 * audit['pca']['tangent_rms_error_over_fluctuation']:.2f}% for PCA.",
        '- No covariance-derivative Fisher term, changing-sky variance or model-discrepancy term is included.',
        '- This dataset retains 40 bins over ell=80–7979. The fixed-noise 524K unbinned estimator is a separate experiment.',
        '', '## Figures and numerical products', '',
        '- [Forecast corner](../comparison/battaglia12_forecast_all.pdf)',
        '- [Independent-observation corner](../comparison/battaglia12_observed_all.pdf)',
        '- [MOPED and its own Fisher forecast](../comparison/battaglia12_forecast_moped.pdf)',
        '- [PCA and its own Fisher forecast](../comparison/battaglia12_forecast_pca.pdf)',
        '- [Forecast marginal intervals](../comparison/battaglia12_forecast_intervals.pdf)',
        '- [Information and width summary](battaglia12_comparison_summary.pdf)',
        '- `constraints_review.csv`: both cases, all methods, means, standard deviations and 68/95% marginal intervals.',
        '- `review.json`: sampling checks, eigenvalues, source references and numerical comparisons.',
        '- `covariance_size_check.json`: two independent importance integrations under the actual joint prior.', '',
    ]
    (args.output / 'RESULTS.md').write_text('\n'.join(report))
    write_json(args.output / 'complete.json', dict(complete=True, source=digest(comparison / 'complete.json'),
        review_script_sha256=digest(Path(__file__)),
        artifacts={path.name: digest(path) for path in args.output.iterdir() if path.is_file()}))
    print(json.dumps(dict(verified_tasks=len(manifest['tasks']), verified_artifacts=len(complete['artifacts']),
                         fisher_eigenvalues=cal['eigenvalues'].tolist(), forecast_sigma_ratios=ratio,
                         covariance16_max_width_change=covariance_check['maximum_width_change']), indent=2))


if __name__ == '__main__':
    main()
