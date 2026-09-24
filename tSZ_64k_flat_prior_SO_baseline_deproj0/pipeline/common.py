"""Paths, configuration, frozen-source checks and the Julia launcher.

The Julia sources, noise tables and package manifest under engine/,
halfdome_sources/ and julia_env/ are byte-identical copies of the inputs of the
validated 256-row test (the only exception, the XGPaint source line of
Manifest.toml, is documented in source_manifest.json). This module never edits
them; it only checks their SHA-256 values and starts engine.jl with the same
settings as the test, substituting local paths.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ENGINE = BUNDLE / 'engine'
SOURCES = BUNDLE / 'halfdome_sources'            # FLAMINGO_CAMPAIGN of the test
HALFDOME_SOURCE_DIR = SOURCES / 'code' / 'halfdome'
SO_DIR = HALFDOME_SOURCE_DIR / 'other_sims' / 'SO'
JULIA_ENV = BUNDLE / 'julia_env'
DESIGN = BUNDLE / 'design'
REFERENCE = BUNDLE / 'reference' / 'test256'
SOURCE_MANIFEST = HERE / 'source_manifest.json'

SELECTED_HALOS = 85224251
MASK_SHA256 = 'a6c3d64d5ab83e79b6d9b76cccbaf66a708c3e68adce85b9a0d3f610d4100689'
ROW_FILES = ('masked_clean_cl.npy', 'masked_noisy_cross_cl.npy', 'unmasked_clean_cl.npy', 'observation.toml')
# Variables the launcher sets for engine.jl; recorded in command.json.
LAUNCH_KEYS = ('FLAMINGO_CAMPAIGN', 'HALFDOME_SOURCE_DIR', 'PREFLIGHT_OUTPUT', 'BENCHMARK_TASK',
               'HALO_BLOCK_SIZE', 'JULIA_DEPOT_PATH', 'LD_LIBRARY_PATH', 'OPENBLAS_NUM_THREADS',
               'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'HDF5_USE_FILE_LOCKING')
# Every path in the test's argv, each replaced by build_command.
PATH_ARGUMENTS = ('--project=', 'halfdome_path=', 'output_dir=', 'cache_dir=',
                  'baseline_noise_path=', 'goal_noise_path=')
# Read by the HalfDome code only for log/tag names while sobol_row=0 is passed.
HARMLESS_ENV = {'SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID'}
ENV_PATTERN = re.compile(r'env\s*=\s*"([A-Z][A-Z0-9_]*)"|ENV\["([A-Z][A-Z0-9_]*)"\]'
                         r'|haskey\(ENV,\s*"([A-Z][A-Z0-9_]*)"\)|get\(ENV,\s*"([A-Z][A-Z0-9_]*)"')


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def job_id():
    return os.environ.get('SLURM_JOB_ID') or os.environ.get('PBS_JOBID')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2) + '\n')


def load_toml(path):
    try:
        import tomllib
        with open(path, 'rb') as stream:
            return tomllib.load(stream)
    except ModuleNotFoundError:
        pass
    for name in ('tomli', 'toml'):
        try:
            module = __import__(name)
        except ModuleNotFoundError:
            continue
        if name == 'tomli':
            with open(path, 'rb') as stream:
                return module.load(stream)
        return module.load(str(path))
    raise SystemExit('Reading TOML needs Python >= 3.11, or the "tomli" or "toml" package.')


def toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('non-finite value in a task')
        return repr(value)                       # shortest round-trip decimal
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(toml_value(item) for item in value) + ']'
    raise TypeError('unsupported task value: ' + repr(value))


def task_toml(task):
    lines = [key + ' = ' + toml_value(value) for key, value in task.items() if key != 'cases']
    for case in task['cases']:
        lines += ['', '[[cases]]'] + [key + ' = ' + toml_value(value) for key, value in case.items()]
    return '\n'.join(lines) + '\n'


def load_config(path):
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f'Missing {path}. Copy pipeline/config.example.toml to it and edit the paths.')
    raw = load_toml(path)
    paths, run = raw['paths'], raw.get('run', {})
    cfg = dict(config_path=path, julia=Path(paths['julia']), julia_depot=str(paths['julia_depot']),
               halfdome_catalogue=Path(paths['halfdome_catalogue']), run_root=Path(paths['run_root']),
               threads=int(run.get('threads', 26)), audit_threads=int(run.get('audit_threads', 6)),
               batch_size=int(run.get('batch_size', 4)),
               audit_rows_per_task=int(run.get('audit_rows_per_task', 256)))
    for key in ('julia', 'halfdome_catalogue', 'run_root'):
        if not cfg[key].is_absolute():
            raise SystemExit(f'{key} must be an absolute path in {path}')
    if not all(Path(part).is_absolute() for part in cfg['julia_depot'].split(':')):
        raise SystemExit(f'julia_depot must be absolute path(s) in {path}')
    return cfg


def check_runtime_paths(cfg):
    if not os.access(cfg['julia'], os.X_OK):
        raise SystemExit(f"Julia executable not found: {cfg['julia']}")
    if not cfg['halfdome_catalogue'].is_file():
        raise SystemExit(f"HalfDome catalogue not found: {cfg['halfdome_catalogue']}")


def source_manifest():
    return json.loads(SOURCE_MANIFEST.read_text())


def verify_sources():
    """Refuse to run if any frozen input differs from its recorded SHA-256."""
    manifest = source_manifest()
    changed = [name for name, expected in manifest['frozen_sha256'].items()
               if not (BUNDLE / name).is_file() or sha256_file(BUNDLE / name) != expected]
    design = json.loads((DESIGN / 'design_manifest.json').read_text())
    changed += ['design/' + name for name, expected in design['sha256'].items()
                if sha256_file(DESIGN / name) != expected]
    if changed:
        raise RuntimeError('Frozen input changed (restore it from git): ' + ', '.join(changed))
    return manifest


def julia_env_names():
    """Environment variables read by the frozen Julia sources."""
    names = set()
    for path in sorted(ENGINE.glob('*.jl')) + sorted(SOURCES.rglob('*.jl')):
        for match in ENV_PATTERN.finditer(path.read_text(errors='replace')):
            names.update(group for group in match.groups() if group)
    return names


def scrubbed_environment():
    """Copy os.environ without anything that could override the command line.

    The HalfDome configuration gives environment variables precedence over
    key=value arguments, so a leftover NSIDE, BATTAGLIA_P0_AMP or
    BATTAGLIA_SOBOL_ROW from an older pipeline would silently change a row.
    """
    names = julia_env_names() - HARMLESS_ENV
    removed = sorted(key for key in os.environ if key in names or key in ('JULIA_PROJECT', 'JULIA_LOAD_PATH')
                     or key.startswith(('TSZ_', 'BATTAGLIA_', 'BARYON_SLICE_')))
    return {key: value for key, value in os.environ.items() if key not in removed}, removed


def build_command(cfg, folder, threads):
    """The test's argv (reference/test256/base_command.json) with local paths."""
    base = json.loads((REFERENCE / 'base_command.json').read_text())['argv']
    command = [str(cfg['julia'])]
    for value in base[1:]:
        if '/lustre/' in value and not value.startswith(PATH_ARGUMENTS) and not value.endswith('/gate.jl'):
            raise RuntimeError('Unhandled test path in base_command.json: ' + value)
        if value.startswith('--threads='):
            value = f'--threads={threads}'
        elif value.startswith('--project='):
            value = f'--project={JULIA_ENV}'
        elif value.endswith('/gate.jl'):
            value = str(ENGINE / 'engine.jl')
        elif value.startswith('halfdome_path='):
            value = f"halfdome_path={cfg['halfdome_catalogue']}"
        elif value.startswith('output_dir='):
            value = f'output_dir={folder / "raw"}'
        elif value.startswith('cache_dir='):
            value = f'cache_dir={folder / "cache"}'
        elif value.startswith(('baseline_noise_path=', 'goal_noise_path=')):
            key, original = value.split('=', 1)
            value = f'{key}={SO_DIR / Path(original).name}'
        command.append(value)
    return command


def build_environment(cfg, folder, threads):
    env, removed = scrubbed_environment()
    julia_lib = cfg['julia'].parent.parent / 'lib' / 'julia'
    inherited = env.get('LD_LIBRARY_PATH', '')
    env.update(FLAMINGO_CAMPAIGN=str(SOURCES), HALFDOME_SOURCE_DIR=str(HALFDOME_SOURCE_DIR),
               PREFLIGHT_OUTPUT=str(folder), BENCHMARK_TASK=str(folder / 'task.toml'), HALO_BLOCK_SIZE='256',
               JULIA_DEPOT_PATH=cfg['julia_depot'],
               LD_LIBRARY_PATH=str(julia_lib) + (':' + inherited if inherited else ''),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS='1',
               HDF5_USE_FILE_LOCKING='FALSE')
    return env, removed


class TaskFailed(RuntimeError):
    pass


class PreviousFailure(RuntimeError):
    pass


class Terminated(RuntimeError):
    """The scheduler is stopping this worker; do not start further tasks."""


def task_identity(task, manifest):
    return hashlib.sha256(json.dumps(dict(task=task, sources=manifest['frozen_sha256']),
                                     sort_keys=True).encode()).hexdigest()


def launch(task, folder, threads, cfg, dry_run=False):
    """Run engine.jl for one task folder, keeping every identity and failure.

    Returns 'done' if an identical task already succeeded there, 'ran' after a
    successful run, or the planned command for dry_run. A previous failure is
    never retried silently (see manage.py reset). The atomic claim directory
    stops two workers from running the same folder.
    """
    manifest = verify_sources()
    identity = task_identity(task, manifest)
    marker = folder / 'status.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['identity_sha256'] != identity:
            raise RuntimeError(f'Refusing a changed request in {folder}')
        if previous['returncode'] == 0:
            return 'done'
        raise PreviousFailure(f"{folder}: previous attempt failed (returncode {previous['returncode']}); "
                              'inspect run.log, then `manage.py reset`')
    command = build_command(cfg, folder, threads)
    env, removed = build_environment(cfg, folder, threads)
    if dry_run:
        return dict(argv=command, environment={key: env[key] for key in LAUNCH_KEYS},
                    removed_environment=removed)
    check_runtime_paths(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / 'claim'
    claim.mkdir()                              # FileExistsError: another worker owns it
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminated(signum, frame):
        raise Terminated(f'Worker received signal {signum} while running {folder}')
    signal.signal(signal.SIGTERM, terminated)
    started, code = time.monotonic(), None
    try:
        atomic_json(claim / 'owner.json', dict(job=job_id(), host=socket.gethostname(),
                                              pid=os.getpid(), utc=now_utc()))
        atomic_json(folder / 'request.json', task)
        atomic_text(folder / 'task.toml', task_toml(task))
        atomic_json(folder / 'command.json', dict(argv=command, job=job_id(), host=socket.gethostname(),
                                                  environment={key: env[key] for key in LAUNCH_KEYS},
                                                  removed_environment=removed))
        timer = ['/usr/bin/time', '-v', '-o', str(folder / 'time.txt')] if os.access('/usr/bin/time', os.X_OK) else []
        with (folder / 'run.log').open('w') as log:
            process = subprocess.Popen(timer + command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait()
            except Terminated:
                process.terminate()
                try:
                    code = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                raise
        if code:
            raise TaskFailed(f'Task failed (returncode {code}); all files retained: {folder}')
        return 'ran'
    finally:
        if code is not None or (folder / 'run.log').exists():
            atomic_json(marker, dict(returncode=-1 if code is None else code,
                                     seconds=time.monotonic() - started, identity_sha256=identity,
                                     utc=now_utc(), job=job_id(), host=socket.gethostname()))
        signal.signal(signal.SIGTERM, previous_handler)
        shutil.rmtree(claim, ignore_errors=True)
