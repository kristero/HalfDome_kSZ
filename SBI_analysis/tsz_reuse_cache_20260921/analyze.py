"""Assess full-ell runtime and errors, with measured conditional SO covariance.

No truncation of ell, retraining, or prior rejection is hidden in this analysis.
The 0.1-sigma threshold is a numerical-error budget, not a physical guardrail.
"""
import json
from pathlib import Path
import re
import sys
import toml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'tsz_8192_validation_recovery_20260921'
EXTREME = ROOT.parent/'tsz_8192_noise_extremes_20260921'
sys.path.insert(0,str(OLD))
import analyze as previous

ELL = np.arange(80,7980)
FAC = ELL*(ELL+1)/(2*np.pi)


def spectrum(path):
    array = np.load(path)
    assert array.shape == (7980,) and np.isfinite(array).all()
    return array[ELL]*FAC


def build_weights():
    old_plan = json.loads((OLD/'plan.json').read_text())
    weights = {}
    noise = {}
    for label, taskid in [('Battaglia12',0),('FL_L1_m9',1),('compact',86),('extended_shallow',87)]:
        base = (OLD if taskid < 2 else EXTREME)/'controls'/f'{taskid:03d}'
        clean = spectrum(base/'masked_clean_cl.npy')
        draws = []
        for i in range(128):
            path = base/'noise'/f'{i:03d}.npy'
            if not path.exists() and taskid < 2:
                path = ROOT.parent/'tsz_8192_validation_20260920'/'controls'/f'{taskid:03d}'/'noise'/f'{i:03d}.npy'
            draws.append(spectrum(path))
        noise[label] = np.array(draws)-clean
        if taskid > 1:
            continue
        sigma = np.maximum(noise[label][:64].std(0,ddof=1),1e-30)
        widths = np.array(old_plan['upper'])-np.array(old_plan['lower'])
        theta = old_plan['tasks'][taskid]['theta']
        derivatives = np.empty((7900,9))
        derivatives[:,0] = 2*clean/theta[0]*widths[0]
        for j in range(1,9):
            tasks = [t for t in old_plan['tasks'] if t.get('anchor') == label
                     and t.get('derivative_index') == j and t['step_fraction'] == .5]
            plus = next(t for t in tasks if t['sign'] == 1)
            minus = next(t for t in tasks if t['sign'] == -1)
            derivatives[:,j] = (spectrum(OLD/'controls'/f"{plus['id']:03d}"/'masked_clean_cl.npy')-
                spectrum(OLD/'controls'/f"{minus['id']:03d}"/'masked_clean_cl.npy'))/(2*plus['step'])*widths[j]
        for rcond in (1e-3,1e-6):
            matrix,_,metadata = previous.moped_matrix(derivatives/sigma[:,None],noise[label][:64]/sigma,rcond)
            weights[label+f'_rcond{rcond:g}'] = (matrix/sigma[:,None],metadata)
    return weights,noise


def compare(new, reference, label, weights, noise):
    delta = new-reference
    output = dict(relative_Dell_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-300)),
                  maximum_absolute_Dell=float(np.max(np.abs(delta))), moped=[])
    for name,(matrix,meta) in weights.items():
        distance,covariance = previous.covariance_distance(delta@matrix,noise[label][64:]@matrix)
        output['moped'].append(dict(anchor_and_rank=name, distance=distance,
                                    rank=meta['rank'],covariance=covariance))
    output['max_tested_moped_distance'] = max(r['distance'] for r in output['moped'])
    return output


def main():
    plan = json.loads((ROOT/'plan.json').read_text())
    weights,noise = build_weights()
    rows,missing,failures = [],[],[]
    for task in plan['tasks']:
        folder = ROOT/'controls'/f"{task['id']:03d}"
        status_path = folder/'status.json'
        if not status_path.exists():
            missing.append(task['id']);continue
        status = json.loads(status_path.read_text())
        if status['returncode']:
            failures.append(dict(task=task['id'],status=status));continue
        meta = toml.load(folder/'benchmark.toml')
        assert meta['ell_max'] == 7979
        assert meta['mask_sha256'] == 'a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689'
        assert set(meta['selected_halos_per_case']) == {85224251}
        rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',(folder/'time.txt').read_text())
        row = dict(task=task,seconds=status['seconds'],peak_RSS_GiB=int(rss[1])/2**20,
                   timing=meta,comparisons=[])
        for case in task['cases']:
            label = case['label']
            original = spectrum(OLD/'controls'/f"{case['reference_id']:03d}"/'masked_clean_cl.npy')
            current = spectrum(folder/label/'masked_clean_cl.npy')
            comparison = dict(label=label,against_8192_default=compare(current,original,label,weights,noise))
            # A doubled-grid spectrum is a second, independent cache reference.
            old_plan = json.loads((OLD/'plan.json').read_text())
            refined = next(t for t in old_plan['tasks'] if t['label'] == label and t['grid_factor'] == 2)
            fine = spectrum(OLD/'controls'/f"{refined['id']:03d}"/'masked_clean_cl.npy')
            comparison['against_8192_doubled_cache'] = compare(current,fine,label,weights,noise)
            comparison['pixel_averages'] = []
            for target in task.get('pixel_targets',[]):
                for suffix in ('','_dewindow'):
                    path = folder/f'{label}_pixel{target}{suffix}'/'masked_clean_cl.npy'
                    comparison['pixel_averages'].append(dict(nside=target,dewindow=bool(suffix),
                        against_same_fine_map=compare(spectrum(path),current,label,weights,noise)))
            row['comparisons'].append(comparison)
        rows.append(row)
    # Same-process batched variants are compared at painting-only and total cost.
    base = next((r for r in rows if r['task']['id'] == 0),None)
    for row in rows:
        if base:
            row['catalogue_speedup_per_row'] = (base['timing']['catalogue_seconds']/4)/(
                row['timing']['catalogue_seconds']/row['timing']['cases'])
            row['process_speedup_per_row'] = (base['seconds']/4)/(row['seconds']/row['timing']['cases'])
        row['auxiliary_pixel_transforms_in_process_time'] = bool(row['task'].get('pixel_targets'))
    report = dict(ell_min=80,ell_max=7979,rows=rows,missing=missing,failures=failures,
        diagnostic_256_submitted=False,
        scope='Fixed-sky conditional SO noise, local unbinned MOPED at two anchors, not SBI coverage',
        rendering_note='Child-centre averages are finite quadrature. Isotropic pixel-window removal is approximate.',
        cache_accuracy_budget_sigma=plan['accuracy_budget_sigma'],
        unresolved=['Global prior coverage of numerical error','End-to-end inference calibration',
                    'Cross-spectrum noise convention for real SO data products'])
    (ROOT/'results/report.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(rows)
    print(json.dumps(dict(completed=len(rows),missing=missing,failures=failures),indent=2))


def plot(rows):
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':16,
                         'xtick.labelsize':12,'ytick.labelsize':12,'savefig.dpi':180})
    choices = [r for r in rows if r['task']['id'] < 9]
    if not choices:
        return
    fig,axes = plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    labels = [r['task']['label'].replace('_','\n',1) for r in choices]
    x = np.arange(len(choices))
    axes[0].bar(x,[r['timing']['catalogue_seconds']/r['timing']['cases'] for r in choices],color='#326a9f')
    axes[0].set(ylabel='Catalogue + painting [s / sky]',xticks=x,xticklabels=labels)
    axes[0].tick_params(axis='x',labelrotation=55,labelsize=10)
    for case in ['Battaglia12','FL_L1_m9','compact','extended_shallow']:
        values = [next(c for c in r['comparisons'] if c['label']==case)
                  ['against_8192_doubled_cache']['max_tested_moped_distance'] for r in choices]
        axes[1].plot(x,np.maximum(values,1e-12),'o-',label=case)
    axes[1].axhline(.1,color='black',ls='--',lw=1)
    axes[1].set(yscale='log',ylabel='Local MOPED shift / noise',xticks=x,xticklabels=labels)
    axes[1].tick_params(axis='x',labelrotation=55,labelsize=10)
    axes[1].legend(fontsize=10)
    fig.savefig(ROOT/'plots/speed_accuracy.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
