"""Deploy and run this isolated pilot through the existing Windows STRW alias.

No production calibration files are edited. The inherited IO helpers are copied
verbatim into this pilot and pinned by hashes. Subprocess arguments are lists;
passwords, keys and Authenticator codes are never part of this script.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = "/home/nagainis/HalfDome/flamingo_pressure_calibration/" + HERE.name
SSH = "/mnt/c/Windows/System32/OpenSSH/ssh.exe"


def remote(source, python="python3", timeout=120):
    command = [SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "strw-maas", python, "-"]
    result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:] + result.stdout[-3000:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "probe", "run", "launch", "status", "fetch"))
    parser.add_argument("--script")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.action == "deploy":
        sources = {p.name: p.read_text() for p in HERE.glob("*.py")}
        original = HERE.parent / "fullsnap_calibration"
        for name in ("common.py", "flamingo_io.py", "catalogue_stage.py"):
            source = (original/name).read_text()
            sources["dependencies/"+name] = source
            target = HERE / "dependencies" / name
            target.parent.mkdir(exist_ok=True)
            target.write_text(source)
        payload = base64.b64encode(json.dumps(sources).encode()).decode()
        program = f'''import base64,json,hashlib
from pathlib import Path
root=Path({REMOTE!r}); root.mkdir(parents=True,exist_ok=True)
sources=json.loads(base64.b64decode({payload!r}))
hashes={{}}
for name,source in sources.items():
    target=root/name
    assert target.resolve().is_relative_to(root.resolve())
    target.parent.mkdir(exist_ok=True)
    target.write_text(source)
    hashes[name]=hashlib.sha256(source.encode()).hexdigest()
(root/'deployed_sources.json').write_text(json.dumps(hashes,indent=2)+'\\n')
print(json.dumps(hashes,indent=2))
'''
        result = remote(program, timeout=args.timeout)
        (HERE / "deployed_sources.json").write_text(result)
    elif args.action == "probe":
        result = remote('''import importlib.util,json,sys,os
names=['numpy','scipy','h5py','swiftsimio','unyt','numba']
print(json.dumps({'python':sys.version,'modules':{n:(importlib.util.find_spec(n).origin if importlib.util.find_spec(n) else None) for n in names},'home_conda':os.path.isdir(os.path.expanduser('~/miniconda3'))},indent=2))
''')
        (HERE / "environment_probe.json").write_text(result)
    elif args.action in ("run", "launch"):
        if args.script is None or Path(args.script).name != args.script:
            parser.error("Pass a script basename")
        if args.action == "run":
            program = f"import os,runpy;os.environ.update({{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'}});os.chdir({REMOTE!r});runpy.run_path({args.script!r},run_name='__main__')"
            result = remote(program, python=args.python, timeout=args.timeout)
            (HERE / (Path(args.script).stem + "_stdout.txt")).write_text(result)
        else:
            program = f'''import json,os,subprocess
from pathlib import Path
root=Path({REMOTE!r}); log=root/'analysis.log'
assert not (root/'launch_receipt.json').exists(), 'A pilot was already launched'
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    env[key]='1'
with log.open('wb') as output:
    process=subprocess.Popen([{args.python!r},'-u',{args.script!r}],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
receipt={{'pid':process.pid,'host':'maas.strw.leidenuniv.nl','script':{args.script!r},'scheduler_job':False,'scientific_process_launches':1+len(list((root/'attempts').glob('*/launch_receipt.json')))}}
(root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt,indent=2))
'''
            result = remote(program)
            (HERE / "launch_receipt.json").write_text(result)
    elif args.action == "status":
        result = remote(f'''import json
from pathlib import Path
root=Path({REMOTE!r})
for name in ('progress.json','complete.json','failed.json'):
    if (root/name).exists(): print(name+': '+(root/name).read_text())
log=root/'analysis.log'
if log.exists(): print(log.read_text()[-5000:])
''')
    else:
        result = remote(f'''import base64,json
from pathlib import Path
root=Path({REMOTE!r})
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.png','.pdf','.csv','.hdf5') and 'venv' not in p.parts]
assert sum(p.stat().st_size for p in files)<128*1024**2
print(json.dumps({{str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in files}}))
''', timeout=args.timeout)
        payload = json.loads(result)
        for name, encoded in payload.items():
            target = HERE / "cluster_results" / name
            assert target.resolve().is_relative_to((HERE / "cluster_results").resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(encoded))
        result = json.dumps({"fetched_files": sorted(payload)}, indent=2)
    print(result)


if __name__ == "__main__":
    main()
