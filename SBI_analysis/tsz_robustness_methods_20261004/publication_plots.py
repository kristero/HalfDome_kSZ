#!/usr/bin/env python3
"""Readable final tradeoff and parameter-width figures; inference is unchanged."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import LABELS, PARAMETERS, THETA
from report import ARMS, save_figure, table


def export_selected_diagnostics(root, selection):
    """Put both parameter spaces and Monte Carlo precision in a small CSV."""
    plan = json.loads((root / 'plan.json').read_text())
    names = list(dict.fromkeys(['original_merlin'] + list(selection['winners'].values())))
    rows = []
    for space in ['linear', 'logit']:
        for name in names:
            folder = root / 'candidates' / name
            if space == 'logit':
                folder = folder / 'bounded'
            result = json.loads((folder / 'evaluation_final.json').read_text())
            for i, case in enumerate(plan['cases']):
                for j, parameter in enumerate(PARAMETERS):
                    row = dict(method=name, param_space=space, case=case['label'],
                               parameter=parameter, truth=float(THETA[j]))
                    for key in ['q025', 'q16', 'median', 'q84', 'q975', 'width68',
                                'median_mcse', 'width68_mcse', 'truth_in68', 'truth_in95',
                                'outside_prior_per_parameter']:
                        row[key] = result[key][i][j]
                    rows.append(row)
    table(root / 'report' / 'selected_posterior_diagnostics.csv', rows)


def run(root):
    folder = root / 'report'
    selection = json.loads((folder / 'selection.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    with open(folder / 'discovery_ranking.csv', newline='') as handle:
        rows = list(csv.DictReader(handle))

    plt.rcParams.update({
        'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 22,
        'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 15,
        'pdf.fonttype': 42,
    })
    for arm in ARMS:
        cases = [row for row in rows if row['arm'] == arm]
        winner = selection['winners'][arm]
        fig, ax = plt.subplots(figsize=(10, 7))
        for family, color in [('moped', '#2868a8'), ('merlin', '#26805d')]:
            members = [row for row in cases if row['method'].startswith(family + '_')
                       and row['method'] != winner]
            ax.scatter([float(row['width_cost']) for row in members],
                       [np.hypot(float(row['truth_rms']), float(row['agreement_rms']))
                        for row in members],
                       s=70, color=color, alpha=.8, label=family.upper() if family == 'moped' else 'Merlin')
        for name, label, color, marker in [
                ('original_merlin', 'Original', '#333333', 'o'),
                (winner, 'Recommended', '#cb5a31', '*')]:
            row = next(row for row in cases if row['method'] == name)
            point = (float(row['width_cost']),
                     np.hypot(float(row['truth_rms']), float(row['agreement_rms'])))
            ax.scatter(*point, s=220 if marker == '*' else 95, color=color,
                       marker=marker, label=label, zorder=5)
            ax.annotate(label, point,
                        xytext=(9, -17) if label == 'Original' else (-10, 14),
                        ha='left' if label == 'Original' else 'right',
                        textcoords='offset points', fontsize=16)
        ax.axvline(selection['maximum_average_width_ratio'], color='#777777',
                   lw=1.5, ls='--', label='25% width cap')
        ax.axvline(1., color='#aaaaaa', lw=1., ls=':')
        ax.set(xlabel='Geometric mean width ratio', ylabel='Truth and sky discrepancy',
               title=arm.capitalize())
        ax.margins(x=.12, y=.12)
        ax.legend(loc='upper right')
        ax.grid(alpha=.15)
        fig.tight_layout()
        save_figure(fig, folder, f'tradeoff_{arm}')

    # A cap on the average width does not constrain every individual parameter.
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        winner = selection['winners'][arm]
        costs = summary['subsets'][arm]['all'][winner]['width_cost_per_parameter']
        ax.barh(np.arange(9), costs, color='#cb5a31', alpha=.85)
        ax.axvline(1., color='#333333', ls='--', lw=1.4)
        ax.set(yticks=np.arange(9), yticklabels=LABELS,
               xlabel='68% width / original', title=arm.capitalize())
        ax.grid(axis='x', alpha=.15)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save_figure(fig, folder, 'parameter_width_cost')
    if (root / 'candidates').exists():
        export_selected_diagnostics(root, selection)
