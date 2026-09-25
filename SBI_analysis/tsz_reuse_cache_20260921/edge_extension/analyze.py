"""Compare exterior-continuation grids using saved full-range noise context."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('progress',ROOT.parent/'report_progress.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
context=np.load(ROOT.parent/'followup/compression_context.npz')
plan=json.loads((ROOT/'plan.json').read_text())
cases=plan[0]['cases']
rows=[]
for task in plan:
    folder=ROOT/'controls'/f"{task['id']:03d}"
    status=json.loads((folder/'status.json').read_text())
    assert status['returncode']==0
    comparisons=[]
    for case in cases:
        label=case['label']
        a=metrics.dl(folder/label/'masked_clean_cl.npy')
        b=metrics.dl(ROOT/'controls/002'/label/'masked_clean_cl.npy')
        comparisons.append(dict(label=label,relative_Dell_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            moped={name:metrics.distance((a-b)@context[name],context[label+'__'+name])
                   for name in context.files if '__' not in name}))
    rows.append(dict(task=task,status=status,comparisons_to_doubled_smooth_cache=comparisons))
(ROOT/'report.json').write_text(json.dumps(dict(rows=rows,
    ell_max=7979,scope='Exterior cache continuation; physical pressure support unchanged'),indent=2)+'\n')
print('Completed three smooth-cache full-catalogue comparisons.')
