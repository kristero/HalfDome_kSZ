#!/usr/bin/env python3
'''Compare prior-normalized RMSE for two and full SBI runs.'''

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TWO_ROOT = Path(
    '/lustre/work/kristero10/'
    'adrian_two_param_nsf_convergence_baseline_deproj0/asinh'
)
FULL_CASE_ROOT = Path(
    '/home/kristero10/HalfDome_kSZ/SBI_analysis/outputs/cluster_outputs/'
    'SBI_Adrian_SO_dataset_size_ell80_7979_dataset_row_metadata_verified_asinh/'
    'masked_baseline_noise_cross_deproj0'
)
FULL_DATASET_DIR = Path(
    '/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/'
    'adrian_so_sbi_cases_ell80_7979_dataset_row_metadata_verified'
)
CASE = 'masked_baseline_noise_cross_deproj0'


def parse_size(name: str, prefix: str) -> int | None:
    match = re.fullmatch(re.escape(prefix) + r'(\d+)', name)
    return int(match.group(1)) if match else None


def two_completed(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    runs: dict[int, Path] = {}
    for run in root.glob('N*'):
        size = parse_size(run.name, 'N')
        evaluation = run / 'evaluation'
        complete = (evaluation / 'evaluation_complete.json').is_file()
        metrics = (evaluation / 'heldout_metrics.csv').is_file()
        if size is not None and complete and metrics:
            runs[size] = run
    if not runs:
        raise FileNotFoundError(
            f'No completed two-parameter evaluations under {root}'
        )
    return dict(sorted(runs.items()))


def saved_full_run(path: Path) -> bool:
    return (path / 'posterior.pkl').is_file() or (
        (path / 'inference.pkl').is_file()
        and (path / 'density_estimator.pkl').is_file()
    )


def full_completed(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    runs: dict[int, Path] = {}
    for job in root.glob('job_N*'):
        size = parse_size(job.name, 'job_N')
        if size is None:
            continue
        candidates = sorted(set([job / f'N{size}', *job.glob(f'**/N{size}')]))
        matches = [
            path for path in candidates
            if path.is_dir() and saved_full_run(path)
        ]
        if len(matches) > 1:
            raise ValueError(
                f'Duplicate completed full runs for N={size}: {matches}'
            )
        if matches:
            runs[size] = matches[0]
    if not runs:
        raise FileNotFoundError(f'No completed full-parameter runs under {root}')
    return dict(sorted(runs.items()))


def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce').dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize_two(size: int, run: Path, holdout: int) -> dict[str, object]:
    source = run / 'evaluation' / 'heldout_metrics.csv'
    frame = pd.read_csv(source)
    required = {'dataset_index', 'param', 'normalized_error_prior'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{source} lacks columns {sorted(missing)}')
    indices = sorted(pd.to_numeric(frame['dataset_index']).astype(int).unique())
    if len(indices) < holdout:
        raise ValueError(
            f'{source} has {len(indices)} profiles; {holdout} are required'
        )
    frame = frame[
        pd.to_numeric(frame['dataset_index']).astype(int).isin(indices[-holdout:])
    ].copy()
    parameters = sorted(frame['param'].astype(str).unique())
    if parameters != ['P0', 'beta']:
        raise ValueError(f'Expected P0 and beta in {source}; found {parameters}')
    counts = frame.groupby('dataset_index')['param'].nunique()
    if not bool((counts == 2).all()):
        raise ValueError(f'Incomplete two-parameter profile rows in {source}')
    frame['normalized_error_squared'] = (
        pd.to_numeric(frame['normalized_error_prior']) ** 2
    )
    per_profile = np.sqrt(
        frame.groupby('dataset_index')['normalized_error_squared'].mean()
    )
    if len(per_profile) != holdout:
        raise ValueError(f'Expected {holdout} profiles in {source}')
    return {
        'series': '2 inferred parameters (P0, beta)',
        'n_train': size,
        'n_test': len(per_profile),
        'n_parameters': 2,
        'mean_rmse_over_prior_range': float(per_profile.mean()),
        'mean_rmse_over_prior_range_sem': sem(per_profile),
        'source': str(source),
    }


def csv_sizes(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if 'case' in frame:
        frame = frame[frame['case'].astype(str) == CASE]
    return set(pd.to_numeric(frame['n_train']).astype(int))


def ensure_full_metrics(
    project: Path,
    case_root: Path,
    dataset_dir: Path,
    sizes: list[int],
    holdout: int,
    samples: int,
    device: str,
    force: bool,
    reuse_only: bool,
) -> tuple[Path, Path, bool, list[str]]:
    tag = f'last{holdout}_two_vs_full'
    output = case_root / f'diagnostics_{tag}'
    profile_csv = output / f'{tag}_profile_metrics.csv'
    param_csv = output / f'{tag}_param_metrics.csv'
    standard_tag = f'last{holdout}'
    standard_output = case_root / f'diagnostics_{standard_tag}'
    candidates = (
        (profile_csv, param_csv),
        (
            standard_output / f'{standard_tag}_profile_metrics.csv',
            standard_output / f'{standard_tag}_param_metrics.csv',
        ),
    )
    if not force:
        for cached_profile, cached_param in candidates:
            complete = set(sizes).issubset(csv_sizes(cached_profile)) and set(
                sizes
            ).issubset(csv_sizes(cached_param))
            if complete:
                print(f'Reusing {cached_profile}')
                return cached_profile, cached_param, False, []
    if reuse_only:
        raise FileNotFoundError(
            f'Cached full metrics do not cover every completed size: {profile_csv}'
        )
    analyzer = project / 'SBI_analysis' / 'analyze_so_sbi_dataset_size.py'
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(analyzer),
        '--run-root', str(case_root.parent),
        '--case-dataset-dir', str(dataset_dir),
        '--cases', CASE,
        '--dataset-sizes', ','.join(map(str, sizes)),
        '--analysis-target', 'last_n',
        '--last-n-test', str(holdout),
        '--analysis-tag', tag,
        '--num-posterior-samples', str(samples),
        '--device', device,
        '--allow-missing',
        '--output-dir', str(output),
    ]
    print('Generating full-parameter metrics:')
    print('  ' + ' '.join(command))
    subprocess.run(command, cwd=project, check=True)
    if not profile_csv.is_file() or not param_csv.is_file():
        raise FileNotFoundError(f'Expected analyzer outputs below {output}')
    return profile_csv, param_csv, True, command


def summarize_full(
    profile_csv: Path,
    param_csv: Path,
    completed_sizes: set[int],
) -> list[dict[str, object]]:
    profiles = pd.read_csv(profile_csv)
    params = pd.read_csv(param_csv)
    profiles = profiles[
        (profiles['case'].astype(str) == CASE)
        & pd.to_numeric(profiles['n_train']).astype(int).isin(completed_sizes)
    ].copy()
    params = params[
        (params['case'].astype(str) == CASE)
        & pd.to_numeric(params['n_train']).astype(int).isin(completed_sizes)
    ].copy()
    found = set(pd.to_numeric(profiles['n_train']).astype(int))
    if found != completed_sizes:
        raise ValueError(f'Full profile metrics missing sizes {sorted(completed_sizes - found)}')

    rows: list[dict[str, object]] = []
    for size in sorted(completed_sizes):
        values = pd.to_numeric(
            profiles[pd.to_numeric(profiles['n_train']).astype(int) == size][
                'rmse_over_prior_range'
            ],
            errors='coerce',
        ).dropna()
        selected_params = params[
            pd.to_numeric(params['n_train']).astype(int) == size
        ]['param'].astype(str)
        n_parameters = selected_params.nunique()
        if n_parameters <= 2:
            raise ValueError(f'Expected full inference for N={size}; got {n_parameters}')
        rows.append(
            {
                'series': f'Full inference ({n_parameters} parameters)',
                'n_train': size,
                'n_test': len(values),
                'n_parameters': n_parameters,
                'mean_rmse_over_prior_range': float(values.mean()),
                'mean_rmse_over_prior_range_sem': sem(values),
                'source': str(profile_csv),
            }
        )
    return rows


def paper_style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': [
                'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'
            ],
            'mathtext.fontset': 'cm',
            'font.size': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7.5,
            'axes.linewidth': 0.7,
            'savefig.bbox': 'tight',
        }
    )


def make_plot(rows: pd.DataFrame, output: Path, dpi: int) -> None:
    paper_style()
    fig, axis = plt.subplots(figsize=(9.2 / 2.54, 6.8 / 2.54))
    styles = (
        ('2 inferred', '#0072B2', 'o'),
        ('Full inference', '#D55E00', 's'),
    )
    for prefix, color, marker in styles:
        selected = rows[rows['series'].str.startswith(prefix)].sort_values('n_train')
        if selected.empty:
            continue
        axis.errorbar(
            selected['n_train'],
            selected['mean_rmse_over_prior_range'],
            yerr=selected['mean_rmse_over_prior_range_sem'],
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.0,
            capsize=0.0,
            label=str(selected.iloc[0]['series']),
        )
    axis.set_xscale('log', base=2)
    axis.set_xlabel('Training set size')
    axis.set_ylabel(
        r'$\left\langle\sqrt{\left\langle[(\bar{\theta}-\theta_{\rm true})/'
        r'\Delta\theta_{\rm prior}]^2\right\rangle_{\theta}}\right\rangle_{\rm test}$'
    )
    axis.grid(True, which='both', alpha=0.25, lw=0.5)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('png', 'jpg'):
        path = output.with_suffix(f'.{suffix}')
        fig.savefig(path, dpi=dpi)
        print(f'Saved {path}')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=project)
    parser.add_argument('--two-param-root', type=Path, default=TWO_ROOT)
    parser.add_argument('--full-case-root', type=Path, default=FULL_CASE_ROOT)
    parser.add_argument('--full-dataset-dir', type=Path, default=FULL_DATASET_DIR)
    parser.add_argument('--holdout-last-n', type=int, default=100)
    parser.add_argument('--num-posterior-samples', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--force-full-analysis', action='store_true')
    parser.add_argument(
        '--reuse-full-metrics-only',
        action='store_true',
        help='Fail instead of resampling when complete cached metrics are absent.',
    )
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.holdout_last_n <= 0:
        raise ValueError('--holdout-last-n must be positive')
    if args.num_posterior_samples <= 1:
        raise ValueError('--num-posterior-samples must exceed one')

    two_runs = two_completed(args.two_param_root)
    full_runs = full_completed(args.full_case_root)
    print(f'Completed two-parameter evaluations: {list(two_runs)}')
    print(f'Completed full-parameter posteriors: {list(full_runs)}')

    two_rows = [
        summarize_two(size, run, args.holdout_last_n)
        for size, run in two_runs.items()
    ]
    profile_csv, param_csv, generated, command = ensure_full_metrics(
        project=args.project_root,
        case_root=args.full_case_root,
        dataset_dir=args.full_dataset_dir,
        sizes=list(full_runs),
        holdout=args.holdout_last_n,
        samples=args.num_posterior_samples,
        device=args.device,
        force=args.force_full_analysis,
        reuse_only=args.reuse_full_metrics_only,
    )
    full_rows = summarize_full(profile_csv, param_csv, set(full_runs))
    rows = pd.DataFrame(two_rows + full_rows).sort_values(['series', 'n_train'])
    full_sample_counts = sorted(
        pd.to_numeric(
            pd.read_csv(param_csv)['num_posterior_samples'], errors='coerce'
        ).dropna().astype(int).unique().tolist()
    )

    output_dir = args.output_dir or (
        args.full_case_root / f'comparison_two_vs_full_last{args.holdout_last_n}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / 'two_vs_full_normalized_rmse_summary.csv'
    rows.to_csv(summary_csv, index=False)
    print(f'Saved {summary_csv}')
    make_plot(rows, output_dir / 'two_vs_full_normalized_rmse', args.dpi)

    provenance = {
        'metric': (
            'Per test profile: sqrt(mean over inferred parameters of '
            '((posterior_mean - theta_true) / prior_width)^2); '
            'the plotted value is the mean over test profiles.'
        ),
        'holdout_last_n': args.holdout_last_n,
        'num_posterior_samples': args.num_posterior_samples,
        'full_metrics_num_posterior_samples': full_sample_counts,
        'two_parameter_root': str(args.two_param_root),
        'full_parameter_case_root': str(args.full_case_root),
        'full_parameter_dataset_dir': str(args.full_dataset_dir),
        'completed_two_parameter_sizes': list(two_runs),
        'completed_full_parameter_sizes': list(full_runs),
        'full_metrics_generated_this_run': generated,
        'full_analysis_command': command,
        'full_profile_metrics': str(profile_csv),
        'full_parameter_metrics': str(param_csv),
        'summary_csv': str(summary_csv),
    }
    provenance_path = output_dir / 'two_vs_full_normalized_rmse_provenance.json'
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(f'Saved {provenance_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
