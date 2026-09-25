"""Monitor only this campaign and fetch verified outputs when PBS finishes.

No job submission, cancellation, model mutation or message sending occurs here.
Network failures are recorded and retried. Scientific/PBS failures stop the
watcher with their status preserved in run_status.json.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from calibration import digest,write_json

REPO=Path(__file__).resolve().parents[2]
SSH=REPO/'SBI_analysis/outputs/unbinned_moped_524k_20260914/cluster_ssh.sh'


def remote(command):
    result=subprocess.run(['bash',str(SSH),command],cwd=REPO,text=True,capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def inspect(root):
    script='''import json,re,subprocess
from pathlib import Path
root=Path(ROOT)
receipt=json.loads((root/'submission.json').read_text())
jobs=[dict(job=receipt['pilot_job'],worker=4)]+receipt['jobs']
status={}
tasks=json.loads((root/'manifest.json').read_text())['tasks']
for record in jobs:
    job=record['job']
    result=subprocess.run(['qstat','-f',job],text=True,capture_output=True)
    values={}
    for name in ['job_state','Exit_status','comment','resources_used.walltime']:
        match=re.search(r'^\\s*'+re.escape(name)+r' = (.+)$',result.stdout,re.M)
        if match:values[name]=match.group(1).strip()
    if not values:
        if 'worker' in record:
            expected=[i for i in range(len(tasks)) if i%5==record['worker']]
            done=all((root/'tasks'/f'task{i:03d}'/'complete.json').exists() for i in expected)
        else:
            done=(root/'comparison/complete.json').exists()
        values={'job_state':'complete_verified' if done else 'missing_without_completion',
                'scheduler_message':result.stderr.strip()}
    status[job]=values
print(json.dumps(dict(jobs=status,completed_tasks=len(list((root/'tasks').glob('*/complete.json'))),
    total_tasks=len(tasks),
    analysis_complete=(root/'comparison/complete.json').exists())))
'''.replace('ROOT',repr(str(root)))
    return json.loads(remote('/home/anaconda3/bin/python3 -c '+shlex.quote(script)))


def fetch(root,output):
    archive=root/'comparison_export.tar.gz'
    command='tar -czf '+shlex.quote(str(archive))+' -C '+shlex.quote(str(root))
    command+=' comparison manifest.json submission.json analysis_sources.sha256 code analysis_code tasks logs'
    remote(command)
    expected=remote('sha256sum '+shlex.quote(str(archive))).split()[0]
    part=output/'comparison_export.tar.gz.part'
    result=subprocess.run(['bash',str(SSH),'--scp','-O','kristero10@idark.ipmu.jp:'+str(archive),str(part)],
                          cwd=REPO,capture_output=True,text=True,timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr)
    assert digest(part)==expected,'Archive checksum mismatch'
    local=part.with_suffix('')
    if local.exists():
        assert digest(local)==expected,'A different previous export already exists'
        part.unlink()
    else:
        part.rename(local)
    with tarfile.open(local) as package:
        for member in package.getmembers():
            assert member.isdir() or member.isfile(),'Refusing archive links or special files'
            assert (output/member.name).resolve().is_relative_to(output.resolve()),'Archive path escapes output'
        package.extractall(output,filter='data')
    comparison=output/'comparison'
    complete=json.loads((comparison/'complete.json').read_text())
    for name,expected_file in complete['artifacts'].items():
        assert digest(comparison/name)==expected_file,name
    return dict(archive_sha256=expected,verified_artifacts=len(complete['artifacts']),
                comparison_directory=str(comparison))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--hours',type=float,default=48)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # A resumed interactive turn can meet the still-running background watcher.
    # Hold one lock for the entire process so two SCP clients never share .part.
    lock_file=(args.output/'.watch_results.lock').open('a')
    try:
        fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1,'Another watcher already owns this output directory.\n')
    deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline:
        state=dict(updated_utc=datetime.now(timezone.utc).isoformat(),cluster_root=str(args.root))
        try:
            status=inspect(args.root)
            state.update(status)
            failed={k:v for k,v in status['jobs'].items()
                    if v.get('Exit_status','0')!='0' or v.get('job_state')=='missing_without_completion'}
            if status['analysis_complete']:
                state.update(fetch(args.root,args.output.resolve()),state='complete_and_downloaded')
            elif failed:
                state.update(state='failed',failed_jobs=failed)
            else:
                state['state']='running_or_queued'
        except Exception as error:
            state.update(state='connection_or_transfer_error',error=str(error))
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(args.output/'run_status.json',state)
        print(json.dumps(state),flush=True)
        if args.once or state['state'] in ('complete_and_downloaded','failed'):
            return
        time.sleep(60)
    write_json(args.output/'run_status.json',dict(state='watch_timeout',
        cluster_root=str(args.root),note='Jobs were left running; rerun this watcher to continue collection'))


if __name__=='__main__':
    main()
