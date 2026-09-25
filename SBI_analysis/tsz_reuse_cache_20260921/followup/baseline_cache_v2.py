"""Correct the module-name collision in the auxiliary analysis, preserving v1."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
# A unique name prevents the old and new analyze.py modules aliasing themselves.
spec = importlib.util.spec_from_file_location('reuse_analysis',ROOT.parent/'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
weights,noise = analysis.build_weights()
plan = json.loads((analysis.OLD/'plan.json').read_text())
results=[]
for label in ['Battaglia12','FL_L1_m9','compact','extended_shallow']:
    default = next(t for t in plan['tasks'] if t['label']==label and t['nside']==8192 and t['grid_factor']==1)
    refined = next(t for t in plan['tasks'] if t['label']==label and t['grid_factor']==2)
    a = analysis.spectrum(analysis.OLD/'controls'/f"{default['id']:03d}"/'masked_clean_cl.npy')
    b = analysis.spectrum(analysis.OLD/'controls'/f"{refined['id']:03d}"/'masked_clean_cl.npy')
    results.append(dict(label=label,**analysis.compare(a,b,label,weights,noise)))
report = dict(results=results,ell_min=80,ell_max=7979,
    noise_scope='Each case has its own 64 held-out split-noise draws; fixed sky, no cosmic variance',
    compression_scope='Local linear unbinned MOPED at B12 and FL, relative singular cutoffs 1e-3 and 1e-6',
    note='Cache error only at fixed raw NSIDE8192. No posterior bias interpretation.')
(ROOT/'baseline_cache.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez(ROOT/'compression_context.npz',
    **{name:matrix for name,(matrix,_) in weights.items()},
    **{label+'__'+name:values[64:]@matrix for label,values in noise.items() for name,(matrix,_) in weights.items()})
print(json.dumps(report,indent=2))
