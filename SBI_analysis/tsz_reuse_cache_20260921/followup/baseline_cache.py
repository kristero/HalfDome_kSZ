"""Close the outstanding noise-weighted default-cache convergence question."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import analyze

weights,noise = analyze.build_weights()
plan = json.loads((analyze.OLD/'plan.json').read_text())
results=[]
for label in ['Battaglia12','FL_L1_m9','compact','extended_shallow']:
    default = next(t for t in plan['tasks'] if t['label']==label and t['nside']==8192 and t['grid_factor']==1)
    refined = next(t for t in plan['tasks'] if t['label']==label and t['grid_factor']==2)
    a = analyze.spectrum(analyze.OLD/'controls'/f"{default['id']:03d}"/'masked_clean_cl.npy')
    b = analyze.spectrum(analyze.OLD/'controls'/f"{refined['id']:03d}"/'masked_clean_cl.npy')
    results.append(dict(label=label,**analyze.compare(a,b,label,weights,noise)))
report = dict(results=results,ell_min=80,ell_max=7979,
    noise_scope='Each case has its own 64 held-out split-noise draws; fixed sky, no cosmic variance',
    compression_scope='Local linear unbinned MOPED at B12 and FL, relative singular cutoffs 1e-3 and 1e-6',
    note='Cache error only at fixed raw NSIDE8192. No posterior bias interpretation.')
(ROOT/'baseline_cache.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez(ROOT/'compression_context.npz',
    **{name:matrix for name,(matrix,_) in weights.items()},
    **{label+'__'+name:values[64:]@matrix for label,values in noise.items() for name,(matrix,_) in weights.items()})
print(json.dumps(report,indent=2))
