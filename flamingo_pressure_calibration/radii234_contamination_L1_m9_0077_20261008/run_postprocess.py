"""Wait for this already launched run, then compute compact local diagnostics.

No job is submitted and no extraction is relaunched. A failure stops the chain.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from remote_control import remote,REMOTE
from control import PYTHON

HERE=Path(__file__).resolve().parent


def main():
    previous=None
    while True:
        source=f'''import json,psutil,time
from pathlib import Path
root=Path({REMOTE!r});result={{}}
for name in ('progress.json','complete.json','failed.json'):
    if (root/name).exists():result[name]=json.loads((root/name).read_text())
pid=json.loads((root/'launch_receipt.json').read_text())['pid'];result['pid_exists']=psutil.pid_exists(pid)
if result['pid_exists']:
    p=psutil.Process(pid);result['elapsed_minutes']=(time.time()-p.create_time())/60
    result['RSS_GiB']=sum(x.memory_info().rss for x in [p]+p.children(recursive=True) if x.is_running())/1024**3
print(json.dumps(result))
'''
        status=json.loads(remote(source,python=PYTHON))
        (HERE/'latest_status.json').write_text(json.dumps(status,indent=2)+'\n')
        key=json.dumps(status.get('progress.json',{}))
        if key!=previous:print(json.dumps(status),flush=True);previous=key
        if 'failed.json' in status:raise RuntimeError(status['failed.json']['error'])
        if status.get('complete.json',{}).get('completed'):break
        if not status['pid_exists']:raise RuntimeError('Extraction exited without a completion marker')
        time.sleep(45)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    steps=[['control.py','fetch'],['verify_study.py'],['fit_study.py'],['interpret_results.py'],['selection_overview.py'],
           ['plot_study.py'],['build_viewer.py'],['build_report.py']]
    receipts=[]
    for args in steps:
        started=time.perf_counter();print('START '+args[0],flush=True)
        subprocess.run([sys.executable,'-B',*args],cwd=HERE,env=env,check=True)
        receipts.append({'script':args[0],'wall_seconds':time.perf_counter()-started})
        (HERE/'postprocess_steps.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print('POSTPROCESS_COMPLETE',flush=True)


if __name__=='__main__':main()
