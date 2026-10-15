"""Run the analysis tests (A, B, D, F) on STRW maas, politely, from a separate runtime tree.

Layout on maas (nothing outside it is written; existing study directories are never modified):
  RUNTIME = /home/nagainis/HalfDome_runtime/overlap_covariance_tests
  RUNTIME/HalfDome/<same relative paths as /home/cbllover/HalfDome>   code and small read-only inputs
  RUNTIME/HalfDome/<study>/<data product>  ->  symlink to the particle-pass product in the real remote study
  RUNTIME/jobs/                                                       cluster_job.py, receipts, logs

Actions:
  python -B cluster_tests.py mirror [--extra FILE ...]   upload the code/input set (from CLUSTER_RUN.json files)
  python -B cluster_tests.py run <A|B|D|F> [--workers N] [--rss-stop GiB]
  python -B cluster_tests.py status
  python -B cluster_tests.py fetch <A|B|D|F>             results/ -> local <test dir>/results/ (SHA256 checked)
Transport: remote_control.remote (Windows OpenSSH alias strw-maas, ProxyJump strw; Python on stdin).
"""
import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

from remote_control import remote, REMOTE

HERE = Path(__file__).resolve().parent
LOCAL_REPO = Path('/home/cbllover/HalfDome')
STUDY_REL = HERE.relative_to(LOCAL_REPO)
RUNTIME = '/home/nagainis/HalfDome_runtime/overlap_covariance_tests'
MIRROR = RUNTIME + '/HalfDome'
REMOTE_REPO = '/home/nagainis/HalfDome'
CFG = json.loads((HERE / 'study_config.json').read_text())
PYTHON = CFG['remote_python']
TESTS = {'A': ('A_overlap', 'run_test_a.py'), 'B': ('B_covariance', 'run_test_b.py'),
         'D': ('D_literature', 'run_test_d.py'), 'F': ('F_observables', 'run_test_f.py'),
         'G': ('G_physical_outer', 'run_test_g.py'), 'H': ('H_sensitivity', 'run_test_h.py'),
         'I': ('I_fitting_a2ii', 'run_test_i.py'), 'J': ('J_halo_dm_lowmass', 'run_test_j.py'),
         'K': ('K_soap_battaglia', 'run_test_k.py')}
# particle-pass products in the real remote study directory (symlinked, never copied or changed)
PRODUCTS = ['measurements_owner.npz', 'measurements_excision.npz', 'measurements_partition.npz',
            'measurements_legacy.npz', 'apertures.npz', 'selected_halos.npz', 'overlap_tables.npz',
            'census_pairs.npz', 'census_population.npz', 'metadata_plan.json', 'schema.json',
            'soap_extra_columns.npz', 'soap_extra_columns.json']
# always needed besides the test directories and the CLUSTER_RUN.json lists
CORE = ['study_config.json', 'verification.json', 'grids.py', 'geometry.py', 'pilot_physics.py', 'measure.py',
        'measure_overlap.py', 'models.py', 'projection_model.py', 'cluster_job.py']
SKIP_PARTS = {'__pycache__', 'results', 'figures', 'tmp', 'attempts', 'products', 'cache', 'smoke'}
KERNEL_DATA = ('jacobians.npz', 'importance.npz')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def collect(extra=()):
    """Local files to mirror: core study files, analysis/, kernels/ code+data, test dirs (code only) and the
    files listed under any key containing 'files' in each test's CLUSTER_RUN.json."""
    files = set()
    for name in CORE:
        if (HERE / name).is_file():
            files.add(HERE / name)
    for sub in ('analysis', 'dependencies'):
        for path in (HERE / sub).rglob('*'):
            if path.is_file() and not SKIP_PARTS & set(path.relative_to(HERE).parts) and path.suffix != '.pyc':
                files.add(path)
    # kernels: code plus the two small derived tables; anything else must be named in a CLUSTER_RUN.json
    files |= {path for path in (HERE / 'kernels').glob('*.py')}
    files |= {HERE / 'kernels' / name for name in KERNEL_DATA if (HERE / 'kernels' / name).is_file()}
    for directory, _ in TESTS.values():
        root = HERE / directory
        for path in root.rglob('*') if root.exists() else []:
            # code and small configs only; local products/caches are regenerated on the cluster
            if path.is_file() and not SKIP_PARTS & set(path.relative_to(HERE).parts) and path.suffix in ('.py', '.json'):
                files.add(path)
        spec = root / 'CLUSTER_RUN.json'
        if spec.exists():
            files |= {Path(p) for p in referenced_files(json.loads(spec.read_text()))}
    deps = HERE / 'kernels_cluster_deps.json'
    if deps.exists():
        files |= {Path(p) for p in json.loads(deps.read_text())['engine_dependency_files']}
    files |= {Path(p) for p in extra}
    out = []
    for path in sorted(files):
        path = path.resolve()
        if path.name in PRODUCTS and path.parent == HERE:
            continue                                   # products are symlinked to the real remote study
        assert path.is_relative_to(LOCAL_REPO), f'{path} is outside the HalfDome repository'
        assert path.is_file(), f'missing {path}'
        out.append(path)
    return out


def referenced_files(spec):
    """Absolute local paths under LOCAL_REPO mentioned anywhere in a CLUSTER_RUN.json."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for token in value.replace(',', ' ').split():
                path = Path(token)
                if token.startswith(str(LOCAL_REPO)) and path.is_file():
                    found.append(token)
                elif not path.is_absolute() and (HERE / token).is_file():
                    found.append(str(HERE / token))
    walk(spec)
    return found


def mirror(extra=()):
    files = collect(extra)
    # (only data files may be symlinked to the real tree; .py modules resolve their inputs from __file__)
    manifest = {str(path.relative_to(LOCAL_REPO)): sha256(path.read_bytes()) for path in files}
    # files already present, identical, in the real remote tree are symlinked instead of uploaded
    same = set(json.loads(remote(f'''import hashlib, json
from pathlib import Path
real = Path({REMOTE_REPO!r}); manifest = json.loads({json.dumps(manifest)!r})
print(json.dumps([rel for rel, digest in manifest.items() if not rel.endswith('.py') and (real / rel).is_file()
                  and hashlib.sha256((real / rel).read_bytes()).hexdigest() == digest]))''', python=PYTHON, timeout=900)))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for path in files:
            rel = str(path.relative_to(LOCAL_REPO))
            if rel not in same:
                tar.add(path, arcname=rel)
    payload = buffer.getvalue()
    staging = RUNTIME + '/staging.tar.gz'
    remote(f'''from pathlib import Path
Path({RUNTIME!r}).mkdir(parents=True, exist_ok=True); Path({staging!r}).write_bytes(b'')''', python=PYTHON)
    step = 12 * 1024 ** 2
    for offset in range(0, len(payload), step):
        chunk = base64.b64encode(payload[offset:offset + step]).decode()
        remote(f'''import base64
with open({staging!r}, 'ab') as f:
    f.write(base64.b64decode({chunk!r}))''', python=PYTHON, timeout=600)
    products = {name: f'{REMOTE}/{name}' for name in PRODUCTS}
    result = remote(f'''import hashlib, json, os, tarfile
from pathlib import Path
mirror = Path({MIRROR!r}); mirror.mkdir(parents=True, exist_ok=True)
staging = Path({staging!r})
assert hashlib.sha256(staging.read_bytes()).hexdigest() == {sha256(payload)!r}, 'upload corrupted'
manifest = json.loads({json.dumps(manifest)!r})
with tarfile.open(staging) as tar:
    for member in tar.getmembers():
        if (mirror / member.name).is_symlink():
            (mirror / member.name).unlink()            # code must be a real file: modules locate inputs from __file__
        target = (mirror / member.name).resolve()
        assert target.is_relative_to(mirror.resolve()), member.name
        if target.exists():
            same = hashlib.sha256(target.read_bytes()).hexdigest() == manifest[member.name]
            if same:
                continue
            os.remove(target)                          # inside the runtime tree only: replace stale copies
        tar.extract(member, mirror, filter='data')
for rel in json.loads({json.dumps(sorted(same))!r}):
    link = mirror / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path({REMOTE_REPO!r}) / rel)
for rel, digest in manifest.items():
    assert hashlib.sha256((mirror / rel).read_bytes()).hexdigest() == digest, rel
study = mirror / {str(STUDY_REL)!r}
links = {{}}
for name, source in json.loads({json.dumps(products)!r}).items():
    link = study / name
    if Path(source).exists():
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source)
        links[name] = source
jobs = Path({RUNTIME!r}) / 'jobs'; jobs.mkdir(exist_ok=True)
(jobs / 'cluster_job.py').write_bytes((study / 'cluster_job.py').read_bytes())
staging.unlink()
(Path({RUNTIME!r}) / 'mirror_manifest.json').write_text(json.dumps({{'files': manifest, 'symlinks': links}}, indent=1))
print(json.dumps({{'files': len(manifest), 'linked_to_real_tree': {len(same)}, 'product_symlinks': len(links), 'MiB': round(sum((mirror / r).stat().st_size for r in manifest) / 2**20, 1)}}))
''', python=PYTHON, timeout=900)
    (HERE / 'cluster_mirror_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(result.strip())


def run(test, workers, rss_stop, extra_args=()):
    directory, driver = TESTS[test]
    job = f'test_{test}'
    workdir = f'{MIRROR}/{STUDY_REL}'
    command = [PYTHON, '-B', f'{directory}/{driver}', '--full', '--workers', '{WORKERS}'] + list(extra_args)
    result = remote(f'''import json, os, subprocess, psutil
from pathlib import Path
jobs = Path({RUNTIME!r}) / 'jobs'
receipt = jobs / {job + '_receipt.json'!r}
if receipt.exists():
    pid = json.loads(receipt.read_text()).get('pid')
    if pid and psutil.pid_exists(pid):
        raise SystemExit({job!r} + ' is still running')
    archive = jobs / 'archive'; archive.mkdir(exist_ok=True)
    stamp = str(len(list(archive.glob({job + '_*'!r}))))
    for name in ({job + '_receipt.json'!r}, {job + '.log'!r}, {job + '_done.json'!r}, {job + '_failed.json'!r}):
        if (jobs / name).exists():
            (jobs / name).rename(archive / (name.replace('.', '_' + stamp + '.', 1)))
assert (Path({workdir!r}) / {directory + '/' + driver!r}).exists(), 'driver not mirrored'
args = [{PYTHON!r}, '-B', str(jobs / 'cluster_job.py'), {job!r}, {workdir!r}, {str(workers)!r}, {str(rss_stop)!r}, '--'] + {command!r}
p = subprocess.Popen(args, cwd=str(jobs), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(json.dumps({{'job': {job!r}, 'wrapper_pid': p.pid}}))
''', python=PYTHON)
    print(result.strip())


def status():
    print(remote(f'''import json, os, psutil
from pathlib import Path
jobs = Path({RUNTIME!r}) / 'jobs'
out = {{'load': os.getloadavg(), 'available_GiB': round(psutil.virtual_memory().available / 2**30, 1), 'jobs': {{}}}}
for receipt in sorted(jobs.glob('*_receipt.json')):
    job = receipt.name[:-len('_receipt.json')]
    r = json.loads(receipt.read_text())
    item = {{'workers': r.get('workers'), 'alive': bool(r.get('pid')) and psutil.pid_exists(r['pid'])}}
    if item['alive']:
        p = psutil.Process(r['pid']); tree = [p] + p.children(recursive=True)
        item['tree_RSS_GiB'] = round(sum(x.memory_info().rss for x in tree if x.is_running()) / 2**30, 2)
        item['elapsed_min'] = round((psutil.time.time() - r['started_unix']) / 60, 1)
    for end in ('done', 'failed'):
        path = jobs / (job + '_' + end + '.json')
        if path.exists():
            item[end] = {{k: v for k, v in json.loads(path.read_text()).items() if k in ('returncode', 'wall_seconds', 'peak_tree_rss_GiB', 'stop_reason', 'min_machine_available_GiB')}}
    log = jobs / (job + '.log')
    item['log_tail'] = log.read_text(errors='replace')[-800:] if log.exists() else ''
    out['jobs'][job] = item
print(json.dumps(out, indent=1))
''', python=PYTHON))


def fetch(test):
    directory, _ = TESTS[test]
    source = f'{MIRROR}/{STUDY_REL}/{directory}/results'
    listing = json.loads(remote(f'''import hashlib, json
from pathlib import Path
root = Path({source!r})
print(json.dumps({{str(p.relative_to(root)): [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()]
                   for p in root.rglob('*') if p.is_file() and 'work' not in p.relative_to(root).parts}} if root.exists() else {{}}))
''', python=PYTHON, timeout=900))
    target = HERE / directory / 'results'
    for rel, (size, digest) in sorted(listing.items()):
        path = target / rel
        if path.exists() and sha256(path.read_bytes()) == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray()
        for offset in range(0, size, 12 * 1024 ** 2):
            data += base64.b64decode(remote(f'''import base64
with open({source + '/' + rel!r}, 'rb') as f:
    f.seek({offset}); print(base64.b64encode(f.read(12 * 1024 ** 2)).decode())''', python=PYTHON, timeout=600))
        assert sha256(bytes(data)) == digest, rel
        path.write_bytes(bytes(data))
    for name in (f'test_{test}_receipt.json', f'test_{test}_done.json', f'test_{test}_failed.json', f'test_{test}.log'):
        text = remote(f'''from pathlib import Path
p = Path({RUNTIME!r}) / 'jobs' / {name!r}
print(p.read_text(errors='replace') if p.exists() else '')''', python=PYTHON)
        if text.strip():
            (HERE / directory / 'results' / name).write_text(text)
    print(json.dumps({'test': test, 'files': len(listing), 'MiB': round(sum(v[0] for v in listing.values()) / 2**20, 2)}))


def stop(test):
    """Stop a running test job (its whole process group); the wrapper then records *_failed.json."""
    job = f'test_{test}'
    print(remote(f'''import json, os, signal, psutil
from pathlib import Path
receipt = Path({RUNTIME!r}) / 'jobs' / {job + '_receipt.json'!r}
r = json.loads(receipt.read_text())
pid = r.get('pid')
if pid and psutil.pid_exists(pid):
    os.killpg(pid, signal.SIGTERM)
    print(json.dumps({{'stopped': {job!r}, 'pid': pid}}))
else:
    print(json.dumps({{'not_running': {job!r}}}))
''', python=PYTHON).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['mirror', 'run', 'status', 'fetch', 'stop'])
    parser.add_argument('test', nargs='?', choices=sorted(TESTS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rss-stop', type=float, default=24.0)
    parser.add_argument('--extra', nargs='*', default=[])
    parser.add_argument('--driver-args', default='', help='extra driver arguments, e.g. "--stages b2 b3"')
    args = parser.parse_args()
    if args.action == 'mirror':
        mirror(args.extra)
    elif args.action == 'run':
        run(args.test, args.workers, args.rss_stop, args.driver_args.split())
    elif args.action == 'status':
        status()
    elif args.action == 'stop':
        stop(args.test)
    else:
        fetch(args.test)


if __name__ == '__main__':
    main()
