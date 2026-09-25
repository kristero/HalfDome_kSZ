"""Check unique ownership and duplicate-launch exclusion across processes."""
import json,multiprocessing as mp,tempfile
from pathlib import Path
from partition_worker import owned_tasks

def race_claim(path,queue):
    try:Path(path).mkdir();queue.put(True)
    except FileExistsError:queue.put(False)

def main():
    plan=json.loads((Path(__file__).resolve().parent.parent/'tsz_8192_validation_recovery_20260921/plan.json').read_text())
    shards=[owned_tasks(plan['tasks'],i) for i in range(4)]
    ids=[t['id'] for shard in shards for t in shard]
    assert sorted(ids)==list(range(86)) and len(set(ids))==86
    with tempfile.TemporaryDirectory() as directory:
        queue=mp.Queue();jobs=[mp.Process(target=race_claim,args=(directory+'/claim',queue)) for _ in range(8)]
        for job in jobs:job.start()
        for job in jobs:job.join();assert job.exitcode==0
        assert sum(queue.get() for _ in jobs)==1
    print('PASS 86 unique assignments and single-owner mkdir claim')

if __name__=='__main__':main()
