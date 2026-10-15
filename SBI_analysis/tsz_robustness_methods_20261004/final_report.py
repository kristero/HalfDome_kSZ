"""Final selection adds a joint posterior diagnostic to the preliminary marginal screen."""
import json
import numpy as np

import report as base
from common import write_json


WIDTH_BUDGET = 1.25  # User-confirmed cap on the geometric mean of width ratios.


def select_with_width_budget(rows):
    """Choose a diagnostic-qualified improvement within the accepted precision cost.

    Both truth displacement and sky scatter must improve over original Merlin.
    A width increase cannot itself reduce either metric: both use the fixed
    original baseline half widths. If no qualifying improvement fits the
    budget, retain the baseline and explicitly report that result.
    """
    original = next(row for row in rows if row['method'] == 'original_merlin')
    qualified = [row for row in rows if row['prior_calibration_screen_pass']]
    improved = [row for row in qualified
                if row['truth_rms'] < original['truth_rms']
                and row['agreement_rms'] < original['agreement_rms']]

    def discrepancy(row):
        return float(np.hypot(row['truth_rms'], row['agreement_rms']))

    budgets = {}
    for limit in [1.1, 1.25, 1.5]:
        within = [row for row in improved if row['width_cost'] <= limit]
        budgets[str(limit)] = min(within, key=discrepancy)['method'] if within else None
    within = [row for row in improved if row['width_cost'] <= WIDTH_BUDGET]
    winner = min(within, key=discrepancy)['method'] if within else 'original_merlin'
    return winner, budgets, base.pareto(rows)


def report(root,plan,final=False):
    diagnostics={}
    names=['original_merlin']+[r['name'] for r in plan['candidates']]
    for name in names:
        prior=json.loads((root/'candidates'/name/'evaluation.json').read_text())['prior_validation']
        joint68=prior['random_reference_rank68'];joint95=prior['random_reference_rank95']
        marginal=bool(np.max(np.abs(np.array(prior['marginal_coverage68'])-.68))<=.10
                      and np.max(np.abs(np.array(prior['marginal_coverage95'])-.95))<=.06)
        joint=abs(joint68-.68)<=.10 and abs(joint95-.95)<=.06
        diagnostics[name]=dict(marginal_screen_pass=marginal,joint_screen_pass=joint,
                               combined_screen_pass=bool(marginal and joint),
                               random_reference_rank68=joint68,random_reference_rank95=joint95)
    original_choose=base.choose
    def choose(rows):
        for row in rows:
            row['joint_calibration_screen_pass']=diagnostics[row['method']]['joint_screen_pass']
            row['prior_calibration_screen_pass']=diagnostics[row['method']]['combined_screen_pass']
        return select_with_width_budget(rows)
    base.choose=choose
    try:
        winners=base.report(root,plan,final)
    finally:
        base.choose=original_choose
    path=root/'report/selection.json';selection=json.loads(path.read_text())
    selection.update(calibration_screen='Marginal and joint random-reference 68/95 diagnostics; tolerances 0.10/0.06',
        selected_diagnostic_pass={arm:diagnostics[name]['combined_screen_pass'] for arm,name in selection['winners'].items()},
        maximum_average_width_ratio=WIDTH_BUDGET,
        score='hypot(truth_rms, agreement_rms) subject to average width ratio <= 1.25',
        improvement_requirement='Both truth_rms and agreement_rms lower than original Merlin on discovery skies',
        no_qualifying_improvement={arm:name=='original_merlin' for arm,name in selection['winners'].items()})
    write_json(path,selection)
    write_json(root/'report/calibration_qualification.json',diagnostics)
    base.table(root/'report/joint_prior_validation.csv',[dict(method=n,**v) for n,v in diagnostics.items()])
    with open(root/'report/results.md','a') as f:
        f.write('\nThe final ranking also screens joint random-reference distance ranks at the 68% and 95% levels, '
                'with the same 0.10 and 0.06 tolerances. The user-confirmed final selection applies a hard '
                '25% cap on the geometric mean of the case- and parameter-wise 68% width ratios. Among '
                'qualified candidates that improve BOTH truth displacement and sky agreement on discovery skies, '
                'it minimizes their quadrature discrepancy within this budget. The preliminary soft width-penalty '
                'score is superseded for this final report. If none qualifies within the budget, the original '
                'method is retained and explicitly flagged as no qualifying improvement in selection.json. '
                'The worst individual-parameter width increase is reported separately. '
                'These two finite diagnostics do not establish accuracy over every parameter or sky.\n')
    return winners
