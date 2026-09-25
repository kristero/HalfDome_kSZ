"""Completed full-catalogue four-child averaging test; no convergence claim."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_progress import dl,distance

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'tsz_8192_validation_recovery_20260921'
task=json.loads((ROOT/'plan.json').read_text())['tasks'][2]
folder=ROOT/'controls/002'
assert json.loads((folder/'status.json').read_text())['returncode']==0
context=np.load(ROOT/'followup/compression_context.npz')
rows=[]
for case in task['cases']:
    label=case['label'];identifier=case['reference_id']
    reference=dl(folder/label/'masked_clean_cl.npy')
    original=(OLD/'controls'/f'{identifier:03d}'/'paired4096_clean_cl.npy' if identifier<2 else
              OLD/'controls'/f'{identifier+1:03d}'/'masked_clean_cl.npy')
    variants={'original_4096_centres':original,
              'four_child_mean':folder/f'{label}_pixel4096'/'masked_clean_cl.npy',
              'four_child_mean_divide_parent_window':folder/f'{label}_pixel4096_dewindow'/'masked_clean_cl.npy'}
    for mode,path in variants.items():
        value=dl(path)
        rows.append(dict(label=label,mode=mode,relative_Dell_l2=float(np.linalg.norm(value-reference)/np.linalg.norm(reference)),
            moped={name:distance((value-reference)@context[name],context[label+'__'+name])
                   for name in context.files if '__' not in name}))
report=dict(rows=rows,ell_max=7979,reference='Same NSIDE8192 centre-sampled sky',
    quadrature_samples_per_parent=4,
    interpretation='Accuracy experiment; finite child quadrature and isotropic window removal are both approximate. '
                   'Not evidence that exact 4096 pixel integration fails or succeeds.',
    finite_quadrature_note='Dividing by the parent pixel window is the infinite-quadrature convention. '
        'With finitely many centre samples the response still differs. For a regular one-dimensional grid, '
        'the discrete averaging response equals parent sinc window divided by child sinc window; '
        'HEALPix windows are isotropic approximations. The pending 16384 child test checks convergence.')
(ROOT/'results/pixel_progress.json').write_text(json.dumps(report,indent=2)+'\n')
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
labels=['Battaglia12','FLAMINGO fit','Compact','Bright / shallow']
for mode,color,marker in [('original_4096_centres','#444444','o'),('four_child_mean','#c46b26','s'),
                         ('four_child_mean_divide_parent_window','#277da8','^')]:
    subset=[r for r in rows if r['mode']==mode]
    values=[max(v for k,v in r['moped'].items() if '0.001' in k) for r in subset]
    name={'original_4096_centres':'4096 centres','four_child_mean':'Four-child mean',
          'four_child_mean_divide_parent_window':'Mean / parent window'}[mode]
    ax.plot(np.arange(4),values,marker+'-',color=color,label=name)
ax.axhline(.1,color='black',ls='--',lw=1)
ax.set(xticks=np.arange(4),xticklabels=labels,yscale='log',ylabel='Local MOPED shift / noise')
ax.legend(fontsize=12);ax.grid(alpha=.2)
fig.savefig(ROOT/'plots/pixel_quadrature_progress.png');plt.close(fig)
lines=['# Four-child full-catalogue pixel test', '',
       'Raw NSIDE8192 maps were averaged into NSIDE4096 parent pixels before the same beam and mask. '
       'All results use ell=80..7979. The comparison reference is the same centre-sampled 8192 map.', '',
       '| Profile | 4096 centres | Four-child average | Average / parent window |',
       '| --- | ---: | ---: | ---: |']
for label in [c['label'] for c in task['cases']]:
    subset=[r for r in rows if r['label']==label]
    values=[max(v for k,v in r['moped'].items() if '0.001' in k) for r in subset]
    lines.append('| '+label+' | '+' | '.join(f'{v:.5g}' for v in values)+' |')
lines+=['', 'Values are joint noise units in the five retained local MOPED directions, taking the '
        'larger result from the two anchor compressions. Each profile uses its own held-out split-noise covariance.', '',
        '![Pixel quadrature](plots/pixel_quadrature_progress.png)', '',report['interpretation'], '',
        report['finite_quadrature_note'], '',
        'Pixel averaging changes the response even when total flux is conserved. '
        '[HEALPix documentation](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm) '
        'defines this distinction. Parent-window correction alone must not be described as an exact '
        'correction for this four-point quadrature. The 16384 reference tests are still pending.']
(ROOT/'PIXEL_PROGRESS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([(r['label'],r['mode'],max(r['moped'].values())) for r in rows],indent=2))
