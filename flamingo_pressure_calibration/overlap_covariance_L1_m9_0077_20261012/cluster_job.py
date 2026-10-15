"""Polite job wrapper for the shared STRW host maas (no scheduler).

Runs one test driver as a child process and guards the machine:
  * the child runs at nice 19 and best-effort I/O priority 7 (set before exec);
  * its worker count is capped by the live load (never above --workers, never more threads than
    max_load_fraction of the machine minus the current 1-minute load, never below min_workers);
  * the whole process tree is stopped (resumable job, partial results kept) when its resident memory
    exceeds --rss-stop-gib or the machine's free memory falls below max(fraction * total, floor);
  * glibc returns freed buffers to the OS (fixed mmap threshold), threads of numeric libraries = 1.
Writes <job>_receipt.json, <job>.log and <job>_done.json (or <job>_failed.json) next to itself.

Usage (on maas): python -B cluster_job.py <job> <workdir> <workers> <rss_stop_gib> -- <command ...>
The literal string {WORKERS} in the command is replaced by the capped worker count.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
POLITE = {'nice': 19, 'ionice_level': 7, 'max_load_fraction': 0.75, 'min_workers': 1,
          'min_available_fraction': 0.12, 'min_available_floor_gib': 4}
ENV = {'MALLOC_MMAP_THRESHOLD_': '4194304', 'MALLOC_TRIM_THRESHOLD_': '33554432', 'PYTHONDONTWRITEBYTECODE': '1',
       'HDF5_USE_FILE_LOCKING': 'FALSE', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'NUMBA_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def polite():
    os.nice(POLITE['nice'])
    try:
        psutil.Process().ionice(psutil.IOPRIO_CLASS_BE, POLITE['ionice_level'])
    except Exception:
        pass


def main():
    job, workdir, workers, rss_stop = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    command = sys.argv[sys.argv.index('--') + 1:]
    load = os.getloadavg()[0]
    room = int(POLITE['max_load_fraction'] * os.cpu_count() - load)
    capped = max(POLITE['min_workers'], min(workers, room))
    command = [part.replace('{WORKERS}', str(capped)) for part in command]
    total = psutil.virtual_memory().total
    floor = max(POLITE['min_available_fraction'] * total, POLITE['min_available_floor_gib'] * 1024 ** 3)
    env = os.environ.copy()
    env.update(ENV)
    env['STUDY_ROOT'] = str(workdir)
    record = {'job': job, 'command': command, 'workdir': str(workdir), 'workers_requested': workers,
              'workers': capped, 'load_at_start': os.getloadavg(), 'available_GiB_at_start':
              psutil.virtual_memory().available / 1024 ** 3, 'rss_stop_GiB': rss_stop,
              'free_memory_floor_GiB': floor / 1024 ** 3, 'nice': POLITE['nice'],
              'ionice': f"best-effort {POLITE['ionice_level']}", 'started_unix': time.time(), 'host': os.uname()[1]}
    log = (HERE / f'{job}.log').open('wb')
    child = subprocess.Popen(command, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT,
                             preexec_fn=polite, start_new_session=True)
    record['pid'] = child.pid
    write_json(HERE / f'{job}_receipt.json', record)
    peak = {'rss': 0, 'min_available': psutil.virtual_memory().available}
    stop = threading.Event()

    def watch():
        root = psutil.Process(child.pid)
        while not stop.wait(2):
            try:
                tree = [root] + root.children(recursive=True)
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
            except psutil.NoSuchProcess:
                return
            available = psutil.virtual_memory().available
            peak['rss'] = max(peak['rss'], rss)
            peak['min_available'] = min(peak['min_available'], available)
            reason = None
            if rss > rss_stop * 1024 ** 3:
                reason = 'own RSS %.1f GiB above %.1f GiB' % (rss / 1024 ** 3, rss_stop)
            elif available < floor:
                reason = 'machine free memory %.1f GiB below %.1f GiB' % (available / 1024 ** 3, floor / 1024 ** 3)
            if reason:
                record['stop_reason'] = reason
                os.killpg(child.pid, signal.SIGTERM)
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    code = child.wait()
    stop.set()
    watcher.join(timeout=5)
    log.close()
    record.update(returncode=code, wall_seconds=time.time() - record['started_unix'],
                  peak_tree_rss_GiB=peak['rss'] / 1024 ** 3, min_machine_available_GiB=peak['min_available'] / 1024 ** 3)
    write_json(HERE / (f'{job}_done.json' if code == 0 and 'stop_reason' not in record else f'{job}_failed.json'), record)


if __name__ == '__main__':
    main()
