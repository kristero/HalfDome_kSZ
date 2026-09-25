"""Disjoint PBS shards with a filesystem-wide exclusive directory claim.

This deliberately never calls the old flock-based worker.main(). A shard
owns exactly task_id % 4; no other shard can touch those result directories.
The mkdir claim additionally rejects duplicate launches of the same shard.
"""
import argparse,hashlib,json,os,socket,time
from pathlib import Path
import worker

ROOT=Path(__file__).resolve().parent

def owned_tasks(tasks,shard,count=4):
    # Recheck preserved data and measure speed first, then finish derivatives.
    priority=lambda t:(0 if 'repeats_task' in t else 1 if t.get('benchmark') else 2,t['id'])
    return sorted((t for t in tasks if t['id']%count==shard),key=priority)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True)
    args=parser.parse_args();assert args.shard in range(4)
    worker.ROOT=ROOT
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen source mismatch: '+name)
    for name,expected in json.loads((ROOT/'external_dependencies.json').read_text())['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('External source mismatch: '+name)
    claims=ROOT/'shard_claims';claims.mkdir(exist_ok=True)
    claim=claims/str(args.shard)
    claim.mkdir()  # atomic across Lustre clients; FileExistsError is intentional
    owner=dict(job=os.environ.get('PBS_JOBID'),host=socket.gethostname(),pid=os.getpid(),utc=time.time())
    worker.atomic_json(claim/'owner.json',owner)
    try:(ROOT/'cross_node_claim_probe').mkdir();won=True
    except FileExistsError:won=False
    worker.atomic_json(ROOT/f'claim_probe_{args.shard}.json',dict(owner,won=won))
    tasks=owned_tasks(json.loads((ROOT/'plan.json').read_text())['tasks'],args.shard)
    try:
        for task in tasks:
            marker=ROOT/'controls'/f"{task['id']:03d}"/'status.json'
            if marker.exists():
                status=json.loads(marker.read_text())
                if status['returncode']==0:continue
                raise RuntimeError('Failed task requires explicit inspection: '+str(task['id']))
            worker.run_task(task)
        worker.atomic_json(ROOT/f'shard_{args.shard}_complete.json',owner)
    finally:
        # Only this owner removes its own empty claim. Interrupted jobs leave
        # a visible claim for explicit review; a second job never steals it.
        (claim/'owner.json').unlink();claim.rmdir()

if __name__=='__main__':main()
