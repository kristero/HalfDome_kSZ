"""Read-only shared-host capacity and raw-input file-stat provenance."""
import argparse
import json
from pathlib import Path
from remote_control import remote,REMOTE
from control import PYTHON

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['during','after']);args=parser.parse_args()
    source=f'''import json,os,psutil,shutil,socket,time
from pathlib import Path
root=Path({REMOTE!r});plan=json.loads((root/'metadata_plan.json').read_text())
snapshot=Path(plan['snapshot_path']);directory=snapshot.parent
chunks=list(directory.glob('flamingo_0077.*.hdf5'))
if not chunks:chunks=list((directory/'swift_snapshot_0077').glob('flamingo_0077.*.hdf5'))
membership=Path(plan['soap_path']).parent/'membership_0077'
files=sorted(set([snapshot,Path(plan['soap_path'])]+chunks+list(membership.glob('membership_0077.*.hdf5'))))
assert len(chunks)==64
stats={{str(p):{{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'inode':p.stat().st_ino}} for p in files}}
vm=psutil.virtual_memory()
disk={{str(p):dict(zip(('total_bytes','used_bytes','free_bytes'),shutil.disk_usage(p))) for p in (root,directory)}}
print(json.dumps({{'host':socket.getfqdn(),'unix_time':time.time(),'logical_CPUs':psutil.cpu_count(),
    'physical_CPU_cores':psutil.cpu_count(logical=False),'RAM_total_GiB':vm.total/1024**3,
    'RAM_available_GiB':vm.available/1024**3,'load_average':os.getloadavg(),'shared_filesystems':disk,
    'disk_note':'Shared filesystem capacity, not per-user quota or reserved allocation.',
    'raw_input_stats':stats,'raw_file_content_hashes_computed':False}}))
'''
    result=json.loads(remote(source,python=PYTHON))
    if args.phase=='after':
        earlier=json.loads((HERE/'resources_during.json').read_text())
        assert result['raw_input_stats']==earlier['raw_input_stats']
        result['raw_input_file_stats_unchanged_since_during_run']=True
    (HERE/f'resources_{args.phase}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_input_stats'},indent=2))


if __name__=='__main__':main()
