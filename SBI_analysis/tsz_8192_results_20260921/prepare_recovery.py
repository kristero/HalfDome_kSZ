"""Prepare disjoint cluster shards without changing the frozen physics code."""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'tsz_8192_validation_20260920'
NEW=ROOT.parent/'tsz_8192_validation_recovery_20260921'
NEW.mkdir(exist_ok=True)
for path in OLD.iterdir():
    if path.suffix in ['.py','.jl','.pbs'] and not (NEW/path.name).exists():
        shutil.copy2(path,NEW/path.name)
shutil.copytree(OLD/'inputs',NEW/'inputs',dirs_exist_ok=True)
shutil.copy2(OLD/'external_dependencies.json',NEW/'external_dependencies.json')
for name in ['partition_worker.py','test_partition.py']:
    shutil.copy2(ROOT/name,NEW/name)
plan=json.loads((OLD/'plan.json').read_text())
for original in [0,1,2,4,6,8]:
    task=plan['tasks'][original].copy();task.pop('anchor',None);task.pop('paired_4096',None)
    task.update(id=len(plan['tasks']),label=task['label']+'_recheck',noise_draws=2,
                repeats_task=original)
    task.pop('identity_sha256',None)
    task['identity_sha256']=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
    plan['tasks'].append(task)
plan['recovery_note']='Lustre localflock does not provide cross-node exclusion. Static disjoint shards and atomic directory claims replace flock.'
(NEW/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(NEW)
