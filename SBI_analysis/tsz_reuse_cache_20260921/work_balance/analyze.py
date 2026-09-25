"""Full-catalogue parity of the dynamically scheduled painter."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import toml

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('progress',ROOT.parent/'report_progress.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
plan=json.loads((ROOT/'plan.json').read_text())[0]
folder=ROOT/'controls/000'
status=json.loads((folder/'status.json').read_text())
assert status['returncode']==0
timing=toml.load(folder/'benchmark.toml')
assert set(timing['selected_halos_per_case'])=={85224251}
rows=[]
for case in plan['cases']:
    a=metrics.dl(folder/case['label']/'masked_clean_cl.npy')
    b=metrics.dl(metrics.OLD/'controls'/f"{case['reference_id']:03d}"/'masked_clean_cl.npy')
    error=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    rows.append(dict(label=case['label'],relative_Dell_l2=error))
    assert error<1e-10,(case['label'],error)
(ROOT/'report.json').write_text(json.dumps(dict(parity_passed=True,rows=rows,
    status=status,timing=timing,ell_max=7979),indent=2)+'\n')
print('PASS full-catalogue balanced-painter parity')
