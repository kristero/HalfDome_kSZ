#!/usr/bin/env python3
"""Rank on discovery skies, test on unused skies, and export readable figures and tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, THETA, PARAMETERS, LABELS, ELL, band_operator, arm_indices, write_json

ARMS = ['simulation', 'mask', 'combined']
COLORS = ['#2868a8', '#cb5a31', '#26805d', '#9865a9']


def save_figure(fig, folder, name):
    for extension in ['pdf', 'png']:
        fig.savefig(folder / f'{name}.{extension}', bbox_inches='tight', dpi=200)
    plt.close(fig)


def table(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discovery_indices(plan, arm):
    selected = arm_indices(plan['cases'], arm)
    return [i for i in selected if plan['cases'][i]['simulation_seed'] in plan['discovery_seeds']
            and plan['cases'][i]['mask_seed'] in plan['discovery_masks']]


def metrics(evaluation, reference, ids):
    """Use a fixed reference scale: broadening a candidate cannot hide its truth displacement."""
    med = np.array(evaluation['median'])[ids]
    width = np.array(evaluation['width68'])[ids]
    scale = np.array(reference['width68'])[0] / 2
    delta = (med - THETA) / scale
    variance = np.mean(((med - med.mean(0)) / scale)**2, axis=0)
    per_parameter_cost = np.exp(np.mean(np.log(width / np.array(reference['width68'])[ids]), axis=0))
    return dict(truth_rms=float(np.sqrt(np.mean(delta**2))),
                agreement_rms=float(np.sqrt(np.mean(variance))),
                width_cost=float(np.exp(np.mean(np.log(per_parameter_cost)))),
                worst_parameter_width_cost=float(np.max(per_parameter_cost)),
                truth_fraction68=float(np.mean(np.array(evaluation['truth_in68'])[ids])),
                truth_fraction95=float(np.mean(np.array(evaluation['truth_in95'])[ids])),
                mean_log_prob_truth=float(np.mean(np.array(evaluation['log_prob_truth'])[ids])),
                truth_rms_per_parameter=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                agreement_per_parameter=np.sqrt(variance).tolist(),
                width_cost_per_parameter=per_parameter_cost.tolist(), cases=len(ids))


def pareto(rows):
    points = np.array([[r['truth_rms'], r['agreement_rms'], r['width_cost']] for r in rows])
    return [not np.any(np.all(points <= p, axis=1) & np.any(points < p, axis=1)) for p in points]


def choose(rows):
    """A stated compromise on the Pareto front, with three explicit width budgets alongside it.

    A 25% increase in every width costs as much as one original-baseline half width
    of combined truth displacement and seed/mask scatter. No validation sky enters this choice.
    """
    frontier = pareto(rows)
    eligible = [r for r, p in zip(rows, frontier) if p and r['prior_calibration_screen_pass']]
    if not eligible:
        eligible = [r for r, p in zip(rows, frontier) if p]
    def score(r):
        return float(np.hypot(r['truth_rms'], r['agreement_rms'])
                     + max(0., np.log(r['width_cost'])) / np.log(1.25))
    best = min(eligible, key=score)
    budgets = {}
    for limit in [1.1, 1.25, 1.5]:
        within = [r for r in eligible if r['width_cost'] <= limit]
        if within:
            budgets[str(limit)] = min(within, key=lambda r: np.hypot(r['truth_rms'], r['agreement_rms']))['method']
        else:
            budgets[str(limit)] = None
    return best['method'], budgets, frontier


def interaction_decomposition(median, scale):
    """Balanced two-factor decomposition of posterior medians; descriptive, no independent-mock assumption."""
    grid = np.asarray(median).reshape(10, 10, 9) / scale
    grand = grid.mean((0, 1))
    simulation = grid.mean(1) - grand
    mask = grid.mean(0) - grand
    residual = grid - grand - simulation[:, None] - mask[None]
    parts = np.array([np.mean(simulation**2, 0), np.mean(mask**2, 0), np.mean(residual**2, (0, 1))])
    total = np.mean((grid - grand)**2, (0, 1))
    np.testing.assert_allclose(parts.sum(0), total, rtol=1e-10, atol=1e-12)
    return parts


def report(root, plan, final=False):
    folder = root / 'report'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 17, 'axes.labelsize': 19, 'axes.titlesize': 19,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 13,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    names = ['original_merlin'] + [r['name'] for r in plan['candidates']]
    screen = {n: json.loads((root / 'candidates' / n / 'evaluation.json').read_text()) for n in names}
    selection_path = folder / 'selection.json'
    rows, selected, all_budgets = [], {}, {}
    for arm in ARMS:
        arm_rows = []
        for name in names:
            evaluation = screen[name]
            prior = evaluation['prior_validation']
            # Broad diagnostic screen, not a proof of calibration. Prior-wide marginal coverage
            # is measured on 1,500 unused parameter rows; overlapping sky cases are not trials.
            coverage68 = np.array(prior['marginal_coverage68'])
            coverage95 = np.array(prior['marginal_coverage95'])
            calibrated = bool(np.max(np.abs(coverage68 - .68)) <= .10
                              and np.max(np.abs(coverage95 - .95)) <= .06)
            arm_rows.append(dict(method=name, arm=arm, subset='discovery',
                                 **metrics(evaluation, screen['original_merlin'], discovery_indices(plan, arm)),
                                 prior_calibration_screen_pass=calibrated))
        winner, budgets, frontier = choose(arm_rows)
        selected[arm] = winner
        all_budgets[arm] = budgets
        for row, front in zip(arm_rows, frontier):
            row['pareto'] = front
            row['selected'] = row['method'] == winner
        rows.extend(arm_rows)
    if final:
        frozen = json.loads(selection_path.read_text())
        assert selected == frozen['winners'], 'Winner selection changed after looking at held-out skies'
    else:
        write_json(selection_path, dict(winners=selected, width_budget_winners=all_budgets,
             selection_uses='Discovery skies only',
             score='hypot(truth_rms, agreement_rms) + max(0, log(width_cost))/log(1.25)',
             truth_scale='original baseline 68% half width, fixed for every method',
             calibration_screen='All 9 prior-test marginal coverages within 0.10 of 0.68 and 0.06 of 0.95; diagnostic only'))
    winners = list(dict.fromkeys(selected.values()))
    data = screen.copy()
    if final:
        for name in winners + ['original_merlin']:
            data[name] = json.loads((root / 'candidates' / name / 'evaluation_final.json').read_text())
    summaries, parameter_rows, case_rows, validation_rows = {}, [], [], []
    for arm in ARMS:
        summaries[arm] = {}
        for subset, ids in [('held_out', arm_indices(plan['cases'], arm, True)),
                            ('all', arm_indices(plan['cases'], arm))]:
            summaries[arm][subset] = {}
            for name in names:
                score = metrics(data[name], data['original_merlin'], ids)
                summaries[arm][subset][name] = score
                for j, parameter in enumerate(PARAMETERS):
                    parameter_rows.append(dict(method=name, arm=arm, subset=subset, parameter=parameter,
                        truth_rms=score['truth_rms_per_parameter'][j],
                        agreement=score['agreement_per_parameter'][j],
                        width_cost=score['width_cost_per_parameter'][j]))
        winner = selected[arm]
        fig, ax = plt.subplots(figsize=(9, 7))
        chosen_rows = [r for r in rows if r['arm'] == arm]
        for row in chosen_rows:
            value = np.hypot(row['truth_rms'], row['agreement_rms'])
            marker = '*' if row['selected'] else ('o' if row['pareto'] else '.')
            color = COLORS[1] if row['selected'] else (COLORS[0] if row['pareto'] else '#969696')
            ax.scatter(row['width_cost'], value, s=180 if row['selected'] else 70, marker=marker, color=color)
            if row['pareto'] or row['method'] == 'original_merlin':
                label = row['method'].replace('moped_', '').replace('merlin_', '')
                ax.annotate(label, (row['width_cost'], value), xytext=(5, 5), textcoords='offset points', fontsize=11)
        ax.axvline(1., color='#777777', lw=1, ls='--')
        ax.set(xlabel='68% width / original Merlin', ylabel='Truth displacement + sky scatter', title=arm.capitalize())
        ax.grid(alpha=.15)
        save_figure(fig, folder, f'tradeoff_{arm}')
        # Raw parameter units preserve the actual change in constraints. Titles are short;
        # scientific caveats and the winning method name belong in the accompanying report.
        ids = arm_indices(plan['cases'], arm)
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        for j, ax in enumerate(axes.flat):
            for k, name in enumerate(['original_merlin', winner]):
                record = data[name]
                med = np.array(record['median'])[ids, j]
                lo = np.array(record['q16'])[ids, j]
                hi = np.array(record['q84'])[ids, j]
                ax.errorbar(np.arange(len(ids)) + (-.15 if k == 0 else .15), med,
                            yerr=[med-lo, hi-med], fmt='o', ms=3 if len(ids)>20 else 5,
                            lw=.6 if len(ids)>20 else 1.2, color=COLORS[k],
                            alpha=.75, label='Original' if k == 0 else 'Selected')
            ax.axhline(THETA[j], color='k', lw=1.3, ls='--')
            ax.set_title(LABELS[j])
            ax.set_xlabel('Sky index' if arm == 'combined' else ('Simulation seed' if arm == 'simulation' else 'Mask seed'))
            if len(ids) == 10:
                key = 'simulation_seed' if arm == 'simulation' else 'mask_seed'
                ax.set_xticks(np.arange(10)[::2], [plan['cases'][ids[i]][key] for i in range(0, 10, 2)])
            ax.grid(alpha=.15)
        axes.flat[0].legend(loc='best')
        fig.tight_layout()
        save_figure(fig, folder, f'posteriors_{arm}')
    for name in names:
        record = data[name]
        for i, case in enumerate(plan['cases']):
            for j, parameter in enumerate(PARAMETERS):
                case_rows.append(dict(method=name, case=case['label'], parameter=parameter,
                    truth=float(THETA[j]), median=record['median'][i][j], q16=record['q16'][i][j],
                    q84=record['q84'][i][j], q025=record['q025'][i][j], q975=record['q975'][i][j],
                    samples=record['samples'], truth_in68=record['truth_in68'][i][j],
                    outside_prior=record['outside_prior_per_parameter'][i][j]))
        prior = screen[name]['prior_validation']
        for j, parameter in enumerate(PARAMETERS):
            validation_rows.append(dict(method=name, parameter=parameter,
                coverage68=prior['marginal_coverage68'][j], coverage95=prior['marginal_coverage95'][j],
                mean_log_prob_truth=prior['mean_log_prob_truth']))
    scale = np.array(data['original_merlin']['width68'])[0] / 2
    parts = {name: interaction_decomposition(data[name]['median'], scale).tolist() for name in names}
    selected_combined = selected['combined']
    for name,short in [('original_merlin','original'),(selected_combined,'selected')]:
        values=((np.array(data[name]['median'])-THETA)/scale).reshape(10,10,9)
        common_limit=max(1.,float(np.max(np.abs((np.array(data['original_merlin']['median'])-THETA)/scale))),
                         float(np.max(np.abs((np.array(data[selected_combined]['median'])-THETA)/scale))))
        fig,axes=plt.subplots(3,3,figsize=(17,14),layout='constrained')
        for j,ax in enumerate(axes.flat):
            picture=ax.imshow(values[:,:,j],origin='lower',vmin=-common_limit,vmax=common_limit,
                              cmap='RdBu_r',aspect='auto')
            ax.set_title(LABELS[j]);ax.set_xlabel('Mask seed');ax.set_ylabel('Simulation seed')
            ax.set_xticks(np.arange(10)[::2],[plan['mask_seeds'][i] for i in range(0,10,2)],rotation=45,ha='right')
            ax.set_yticks(np.arange(10)[::2],[plan['simulation_seeds'][i] for i in range(0,10,2)])
        fig.colorbar(picture,ax=list(axes.flat),shrink=.75,label='Median − truth / reference half width')
        save_figure(fig,folder,f'combined_grid_{short}')
    if (root/'data/grid.npz').exists():
        grid=np.load(root/'data/grid.npz')['clean']
        centers=band_operator()@ELL
        fig,axes=plt.subplots(1,3,figsize=(21,6),sharey=True)
        for ax,arm in zip(axes,ARMS):
            ids=arm_indices(plan['cases'],arm)
            residual=grid[ids]/grid[0]-1
            if arm=='combined':
                lo,mid,hi=np.quantile(residual,[.16,.5,.84],axis=0)
                ax.fill_between(centers,100*lo,100*hi,color=COLORS[0],alpha=.2)
                ax.plot(centers,100*mid,color=COLORS[0],lw=2)
            else:
                for curve in residual:
                    ax.plot(centers,100*curve,lw=1.3,alpha=.8)
            ax.axhline(0,color='k',lw=1);ax.set_xscale('log')
            ax.set_xlabel(r'Multipole $\ell$');ax.set_title(arm.capitalize());ax.grid(alpha=.15)
        axes[0].set_ylabel('Signal change from baseline [%]')
        fig.tight_layout();save_figure(fig,folder,'signal_variation')
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for ax, name, title in zip(axes, ['original_merlin', selected_combined], ['Original', 'Selected']):
        values = np.array(parts[name])
        bottom = np.zeros(9)
        for k, label in enumerate(['Simulation', 'Mask', 'Interaction']):
            ax.bar(np.arange(9), values[k], bottom=bottom, color=COLORS[k], label=label)
            bottom += values[k]
        ax.set_xticks(np.arange(9), LABELS, rotation=45, ha='right')
        ax.set_title(title)
    axes[0].set_ylabel('Median variance / reference half-width²')
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, folder, 'combined_variance')
    # Derivative information accumulated with ell reveals whether the proposed
    # compression still gets most of its local response from small angular scales.
    fig, ax = plt.subplots(figsize=(10, 7))
    model = np.load(root/'data/local_model.npz')
    with np.load(root/'data/design.npz') as design:
        prior_width=design['high']-design['low']
    centers = band_operator() @ ELL
    for k, name in enumerate(list(dict.fromkeys(['moped_full'] + winners))):
        path = root/'candidates'/name/'compression.npz'
        if not path.exists():
            continue
        w = np.load(path)['weights']
        contribution = (model['J']*prior_width)[:, :, None] * w[:, None, :]
        cumulative = np.cumsum(contribution, axis=0)
        curve = np.linalg.norm(cumulative, axis=(1, 2))
        ax.plot(centers, curve / max(curve[-1], 1e-30), lw=2.2,
                label=name.replace('moped_', ''), color=COLORS[k % len(COLORS)])
    ax.set(xlabel=r'Multipole $\ell$', ylabel='Cumulative derivative-response norm')
    ax.legend()
    ax.grid(alpha=.15)
    save_figure(fig, folder, 'compression_scale_response')
    # This response is signed before taking the norm; it need not be monotonic
    # and is not an additive Fisher-information fraction.
    scalar_rows = [{k:v for k,v in row.items() if not isinstance(v, list)} for row in rows]
    table(folder/'discovery_ranking.csv', scalar_rows)
    table(folder/'parameter_comparison.csv', parameter_rows)
    table(folder/'posterior_intervals.csv', case_rows)
    table(folder/'prior_validation.csv', validation_rows)
    contracts=[]
    for recipe in plan['candidates']:
        prepared=json.loads((root/'candidates'/recipe['name']/'prepare.json').read_text())
        contracts.append(dict(method=recipe['name'],ell_min=prepared['effective_lmin'],
            ell_max=prepared['effective_lmax'],scale_fraction_retained=prepared['scale_fraction_retained'],
            input_bands=prepared['n_input_bands'],summary_dimension=prepared['summary_dimension'],
            nuisance=recipe.get('nuisance','none'),strength=recipe['strength'],
            signal_augmentation=recipe.get('augmentation',False)))
    table(folder/'method_contracts.csv',contracts)
    training_rows=[]
    for recipe in plan['candidates']:
        for seed in plan['training_seeds']:
            path=root/'candidates'/recipe['name']/f"{recipe['network']}_seed{seed}.json"
            if path.exists():
                record=json.loads(path.read_text())
                training_rows.append(dict(method=recipe['name'],seed=seed,epochs=record['epochs'],
                    seconds=record['seconds'],best_validation_loss=record['best_validation_loss'],
                    best_epoch=min(record['history'],key=lambda h:h['val_loss'])['epoch']))
    table(folder/'training_summary.csv',training_rows)
    plotted=[n for n in winners if n!='original_merlin']
    if plotted:
        fig,axes=plt.subplots(1,len(plotted),figsize=(7*len(plotted),6),squeeze=False)
        for ax,name in zip(axes.flat,plotted):
            recipe=next(r for r in plan['candidates'] if r['name']==name)
            for k,seed in enumerate(plan['training_seeds']):
                record=json.loads((root/'candidates'/name/f"{recipe['network']}_seed{seed}.json").read_text())
                ax.plot([h['epoch'] for h in record['history']],[h['val_loss'] for h in record['history']],
                        lw=1.8,color=COLORS[k],label=f'Training {seed}')
            ax.set(xlabel='Epoch',ylabel='Validation loss',title=name.replace('moped_',''))
            ax.legend();ax.grid(alpha=.15)
        fig.tight_layout();save_figure(fig,folder,'selected_training_losses')
    summary = dict(final=final, selected=selected, width_budget_winners=all_budgets,
                   subsets=summaries, median_variance_parts=parts,
                   validation_case_counts={arm:len(arm_indices(plan['cases'],arm,True)) for arm in ARMS})
    write_json(folder/'summary.json', summary)
    lines = ['# Scale and sky robustness of the SO Battaglia12 posterior', '',
             'The primary inference uses the unbounded, linear-parameter ensemble. The baseline is simulation 100 '
             'with training mask 12345. Every observation uses the same two split-noise draws of the single SO '
             'deproj0 baseline prescription, with the trained cross-spectrum and Gaussian-beam correction.', '',
             'Winners are selected using 6 simulations × 5 masks only. Validation uses 4 unused simulations '
             'at the baseline mask, 5 unused masks on simulation 100, and the unused 4 × 5 joint grid. '
             'The full 10 × 10 results are descriptive and include discovery cases.', '',
             '| Test | Selected method | Held-out truth RMS | Held-out scatter | Width ratio |',
             '|---|---|---:|---:|---:|']
    for arm in ARMS:
        value=summaries[arm]['held_out'][selected[arm]]
        lines.append(f"| {arm} | {selected[arm]} | {value['truth_rms']:.3f} | "
                     f"{value['agreement_rms']:.3f} | {value['width_cost']:.3f} |")
    lines += ['', 'Truth displacement and scatter use the original baseline 68% half width as a fixed '
              'unit. Width ratio is the geometric mean of the selected/original 68% interval widths for the '
              'same skies and parameters. The per-parameter table must be checked: an average can hide a '
              'poorly recovered parameter. Truth inclusion on these correlated skies is descriptive, '
              'not repeated-noise frequentist coverage.', '',
              'The Pareto front reports the tradeoff without requiring one preferred cost function. The '
              'selected compromise adds the truth/scatter norm to log(width ratio)/log(1.25), with no '
              'reward for narrowing. Alternative winners under 10%, 25%, and 50% width budgets appear '
              'in selection.json. A broad prior-coverage screen is a diagnostic, not certification.', '',
              '## Physics and limitations', '',
              'The thermal SZ spectrum depends on the integrated electron pressure. Changing the simulation '
              'changes halo abundances and structure; changing a large sky cap changes which rare, bright '
              'halos enter and also changes mode coupling. Crossed masks and simulations can interact. '
              'combined_variance shows the exact balanced decomposition of posterior-median variation '
              'into simulation, mask, and interaction terms. Overlapping masks are spatial variations, '
              'not independent sky realizations.', '',
              'Cuts remove complete 50-multipole bands and require retraining. Full-range band averaging '
              'retains every multipole but removes fine spectral structure. MOPED projects nine parameter '
              'derivatives through a chosen covariance. Adding a discovery-sky covariance downweights '
              'directions sensitive to sky variation; a smooth high-ell covariance inflation reduces '
              'small-scale weight without discarding a range. Local Fisher preservation concerns the '
              'chosen covariance and binned inputs only; it does not guarantee a globally calibrated SBI '
              'posterior. See [MOPED](https://arxiv.org/abs/astro-ph/9911102) and '
              '[nuisance-hardened compression](https://arxiv.org/abs/1903.01473).', '',
              'Changing the covariance used to construct weights changes the projection only: the '
              'unaugmented SBI candidates retain the original fixed-sky training distribution. Their '
              'posteriors do not automatically marginalize over new-sky uncertainty. The augmented '
              'candidate is the one that explicitly introduces approximate sky variation into training.', '',
              'The augmented candidate adds discovery B12 signal-ratio templates across the parameter prior. '
              'This is an approximation: it assumes those relative variations transfer across pressure '
              'parameters and leaves the SO noise residual unchanged. Real unused B12 skies and a fixed-sky '
              'prior test check two aspects of this approximation; neither validates arbitrary cosmologies. '
              'The unbounded flow is trained inside the original prior only. Outside-prior mass is reported '
              'per case and parameter and is not calibrated extrapolation.', '',
              'compression_scale_response plots a signed cumulative derivative-response norm. It can be '
              'non-monotonic and must not be interpreted as an additive Fisher-information fraction.', '',
              'All non-winning candidates have 12,000 posterior samples per observation; original and '
              'selected candidates have 90,000 samples when final=true. prior_validation.csv uses 1,500 '
              'unused parameter rows with 600 samples each. input_sha256.json freezes sources; COMPLETE.json '
              'hashes the delivered report. No claim of bias removal is warranted if held-out truth or '
              'agreement fails to improve, or if the width cost is unacceptable.', '']
    (folder/'results.md').write_text('\n'.join(lines))
    return winners


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--final', action='store_true')
    args = p.parse_args()
    report(args.root, json.loads((args.root/'plan.json').read_text()), args.final)
