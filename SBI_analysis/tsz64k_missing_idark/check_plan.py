"""The collaborator's batch_plan.json against the frozen design: every row's theta and seeds, and the missing batches."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
bundle, plan_path, missing_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
manifest = json.loads((bundle / 'design' / 'design_manifest.json').read_text())
theta = np.load(bundle / 'design' / manifest['files']['theta_design']); seeds = np.load(bundle / 'design' / manifest['files']['noise_seeds'])
plan = json.loads(plan_path.read_text())
assert len(plan['batches']) == 16384 and plan['rows'] == 65536 and not plan['deferred_rows'], (len(plan['batches']), plan['rows'])
rows = 0
for b in plan['batches']:
    for c in b['cases']:
        r = c['row']; rows += 1
        assert np.array_equal(np.asarray(c['theta']), theta[r]) and np.array_equal(np.asarray(c['seeds']), seeds[r]), r
missing = [int(v) for v in missing_path.read_text().split(',')]
nodes = {}
for i in missing:
    assert plan['batches'][i]['id'] == i
    for c in plan['batches'][i]['cases']:
        nodes[str(c['nodes'])] = nodes.get(str(c['nodes']), 0) + 1
print(json.dumps(dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), rows_checked=rows, missing_batches=len(missing),
                      missing_rows=4 * len(missing), cache_grids_of_missing_rows=nodes), indent=1))
