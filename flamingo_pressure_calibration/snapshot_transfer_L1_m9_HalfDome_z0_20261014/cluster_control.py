"""Copy only this study's compact inputs/code and submit its bounded PBS jobs."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
SSH = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
IDARK_ROOT = '/lustre/work/kristero10/' + ROOT.name
STRW_ROOT = '/home/nagainis/HalfDome/flamingo_pressure_calibration/' + ROOT.name


def remote(host, program, data=None, timeout=120):
    encoded = base64.b64encode(program.encode()).decode()
    command = 'python3 -c "import base64;exec(base64.b64decode(\'%s\'))"' % encoded
    result = subprocess.run([SSH,'-o','BatchMode=yes','-o','ConnectTimeout=12',host,
        command],input=data,capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-5000:] + result.stdout.decode(errors='replace')[-3000:])
    return result.stdout


def deploy(host,include_inputs=True):
    target = IDARK_ROOT if host == 'idark' else STRW_ROOT
    files = {p.name:p for p in ROOT.glob('*.py')}
    files.update({p.name:p for p in ROOT.glob('*.pbs')})
    files['README.md'] = ROOT/'README.md'
    if host == 'idark' and include_inputs:
        source = ROOT.parent/'radii234_contamination_L1_m9_0077_20261008'
        for name in ['selected_halos.npz','measurements.npz','metadata_plan.json']:
            files['inputs/'+name] = source/name
    if host == 'idark':
        files['inputs/legacy_fit_comparison.csv']=ROOT.parent/'radii234_contamination_L1_m9_0077_20261008/products/fit_comparison.csv'
    hashes = {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for name,p in files.items():
            tar.add(p,arcname=name)
    archive = ROOT/('.transfer_'+host+'.tar')
    archive.write_bytes(buffer.getvalue())
    remote(host,f'from pathlib import Path; Path({target!r}).mkdir(parents=True,exist_ok=True)')
    windows_path = subprocess.check_output(['wslpath','-w',str(archive)],text=True).strip()
    copy = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/scp.exe','-o','BatchMode=yes',
        windows_path,host+':'+target+'/'+archive.name],capture_output=True,timeout=600)
    if copy.returncode:
        raise RuntimeError(copy.stderr.decode(errors='replace'))
    program = f'''import sys,tarfile,io,json,hashlib
from pathlib import Path
root=Path({target!r});root.mkdir(parents=True,exist_ok=True)
with tarfile.open(root/{archive.name!r}) as tar:
 for entry in tar.getmembers():
  assert (root/entry.name).resolve().is_relative_to(root.resolve())
 tar.extractall(root)
expected={hashes!r}
for name,value in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
(root/{archive.name!r}).unlink()
print(json.dumps({{'root':str(root),'verified_files':expected}}))
'''
    # Python 3.8 does not have Path.is_relative_to.
    program = program.replace('assert (root/entry.name).resolve().is_relative_to(root.resolve())',
        'assert str((root/entry.name).resolve()).startswith(str(root.resolve())+"/")')
    receipt = json.loads(remote(host,program,timeout=120))
    archive.unlink()
    (ROOT/('deployment_'+host+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'host':host,'bytes':len(buffer.getvalue()),'files':len(files)}))


def submit(script):
    assert script in ['run_prepare.pbs','run_compare.pbs','run_aggregate.pbs']
    receipt = ROOT/(script+'.job.json')
    assert not receipt.exists(), 'A submission receipt already exists.'
    program = f'''import subprocess,json
from pathlib import Path
root=Path({IDARK_ROOT!r})
r=subprocess.run(['qsub',{script!r}],cwd=root,capture_output=True,text=True,check=True)
job=r.stdout.strip()
receipt={{'job_id':job,'script':{script!r},'queue':'mini','ncpus':26,'mem_gb':32}}
(root/({script!r}+'.job.json')).write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    result = remote('idark',program)
    receipt.write_bytes(result)
    print(result.decode())


def launch_strw():
    program = f'''import os,json,subprocess,socket
from pathlib import Path
root=Path({STRW_ROOT!r})
assert not (root/'launch_receipt.json').exists(), 'Already launched'
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
with (root/'extraction.log').open('wb') as stream:
 p=subprocess.Popen(['nice','-n','10','python3','-u','extract_strw.py'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r={{'pid':p.pid,'host':socket.getfqdn(),'workers':4,'memory_guard_gib':24,'wall_guard_hours':4,'scheduler_job':False}}
(root/'launch_receipt.json').write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))
'''
    result=remote('strw-maas',program)
    (ROOT/'launch_strw.json').write_bytes(result)
    print(result.decode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['deploy-idark','deploy-idark-code','deploy-strw','submit-prepare','submit-compare','launch-strw'])
    args = parser.parse_args()
    if args.action.startswith('deploy'):
        deploy('idark' if 'idark' in args.action else 'strw-maas',include_inputs=args.action!='deploy-idark-code')
    elif args.action == 'launch-strw':
        launch_strw()
    else:
        submit('run_prepare.pbs' if args.action.endswith('prepare') else 'run_compare.pbs')
