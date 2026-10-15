"""Local orchestration of the staged beam-fix run on idark (see beamfix.py for what is computed).

Every --poll seconds:
  1. idark: beamfix.py status and this user's PBS jobs.
  2. Regen workers (regen.pbs): while tasks remain, keep --mini2 workers on queue mini2 and --mini workers on queue
     mini queued or running. When no worker is alive, stale claims are released (beamfix.py reclaim).
  3. Relay: relay.py for the first stage that is not fully uploaded (one relay process at a time, in stage order,
     at most one stage ahead of the noise regeneration).
  4. Stage jobs (stage.pbs, job name bf_sKK): when stage k is regenerated and uploaded and the series of stage k-1 is
     done, stage k is built and its NPE series trained and evaluated. A stage whose job fails twice stops the driver.
  5. Results: after every stage job, RUN/results/ and the stage logs are fetched into beamfix/fetched/ (json, png, log).
The driver keeps no state that idark does not have, except the failure counts (driver_state.json), so it can be
stopped and restarted at any time.

    python driver.py [--poll 600 --mini2 2 --mini 3 ...]
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = 'tsz64k_beamfix_20260930'                     # run root on idark, relative to $HOME
PY_REMOTE = '/home/kristero10/envs/sbi27/bin/python'
PY_LOCAL = '/home/cbllover/miniconda3/envs/halfdome/bin/python'
LOCAL_ENV = {k: v for k, v in os.environ.items() if not k.startswith('BEAMFIX_')}      # no test overrides
LOCAL_ENV.update(LD_LIBRARY_PATH='/home/cbllover/miniconda3/envs/halfdome/lib',
                 PATH='/root/globus-venv/bin:' + os.environ.get('PATH', '/usr/bin:/bin'))
SSH = ['ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', '-o', 'ServerAliveInterval=30', 'idark']
GCP = Path('/home/cbllover/HalfDome/globusconnectpersonal-3.2.8')
STATE = HERE / 'driver_state.json'
FETCHED = HERE / 'fetched'


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def ssh(command, timeout=600):
    """Run a command on idark; None if idark is unreachable (exit 255) or the call times out."""
    try:
        r = subprocess.run(SSH + [command], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode == 255:
        return None
    return r


def jobs():
    """This user's PBS jobs: list of dicts id, name, queue, state (Q, R, H, ...)."""
    r = ssh('qstat -u kristero10')
    if r is None or r.returncode != 0:                 # a PBS error must not look like "no jobs"
        log(f'qstat failed: {r.stderr.strip()[:200] if r else "idark unreachable"}')
        return None
    out = []
    for line in r.stdout.splitlines():
        m = re.match(r'^(\d+)\.\S+\s+\S+\s+(\S+)\s+(\S+)\s+.*\s([A-Z])\s+\S+\s*$', line)
        if m:
            out.append(dict(id=m.group(1), queue=m.group(2), name=m.group(3), state=m.group(4)))
    return out


def status():
    r = ssh(f'cd {RUN} && {PY_REMOTE} code/beamfix.py --run $HOME/{RUN} status')
    if r is None or r.returncode != 0:
        return None
    return json.loads(r.stdout)


def gcp_running():
    r = subprocess.run(['runuser', '-u', 'cbllover', '--', 'env', 'HOME=/home/cbllover', './globusconnectpersonal',
                        '-status'], cwd=GCP, capture_output=True, text=True)
    return 'Globus Online:   connected' in r.stdout


def start_gcp():
    subprocess.Popen(['runuser', '-u', 'cbllover', '--', 'env', 'HOME=/home/cbllover', './globusconnectpersonal',
                      '-start'], cwd=GCP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(30)


def submit_worker(queue, args):
    spec = dict(mini2=(args.mini2_ncpus, args.mini2_mem, args.mini2_wall, args.mini2_procs),
                mini=(args.mini_ncpus, args.mini_mem, args.mini_wall, args.mini_procs),
                mini_B=(args.mini_ncpus, args.mini_mem, args.mini_wall, args.mini_procs))[queue]
    ncpus, mem, wall, procs = spec
    h, m = (int(v) for v in wall.split(':')[:2])
    seconds = 3600 * h + 60 * m
    r = ssh(f'cd {RUN} && qsub -q {queue} -N bf_regen -l select=1:ncpus={ncpus}:mem={mem}gb -l walltime={wall} '
            f'-v RUN=$HOME/{RUN},PROCS={procs},WALL_SECONDS={seconds} code/regen.pbs')
    log(f'submitted regen worker on {queue}: {r.stdout.strip() if r else "idark unreachable"} {r.stderr.strip() if r else ""}')


def submit_stage(k, args):
    r = ssh(f'cd {RUN} && qsub -q {args.stage_queue} -N bf_s{k:02d} -l select=1:ncpus={args.stage_ncpus}:mem=64gb '
            f'-l walltime={args.stage_wall} -v RUN=$HOME/{RUN},K={k} code/stage.pbs')
    log(f'submitted stage {k}: {r.stdout.strip() if r else "idark unreachable"} {r.stderr.strip() if r else ""}')


def fetch():
    """RUN/results and the stage logs -> fetched/ (small files only)."""
    FETCHED.mkdir(exist_ok=True)
    tar = subprocess.run(SSH + [f'cd {RUN} && tar -cf - results logs/stage*.log 2>/dev/null'],
                         stdin=subprocess.DEVNULL, capture_output=True, timeout=1800)
    if tar.returncode in (0, 2) and tar.stdout:
        subprocess.run(['tar', '-xf', '-', '-C', str(FETCHED)], input=tar.stdout, check=False)
        log(f'fetched results ({len(tar.stdout) / 1e6:.1f} MB) into {FETCHED}')


def main():
    relay_box = []
    try:
        return loop(relay_box)
    finally:                                           # never leave an orphaned relay behind
        for k, proc in relay_box:
            if proc.poll() is None:
                proc.terminate()                       # its SIGTERM handler cancels its Globus tasks
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log(f'terminated the relay of stage {k}')


def loop(relay_box):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--poll', type=int, default=600)
    p.add_argument('--mini2', type=int, default=2, help='regen workers on mini2 (queue limit 2 per user)')
    p.add_argument('--mini', type=int, default=0, help='regen workers on mini (6 jobs/user; keep one for stage jobs)')
    p.add_argument('--minib', type=int, default=6, help='regen workers on mini_B (16 jobs/user; same node sizes as mini)')
    p.add_argument('--mini2-ncpus', type=int, default=56)
    p.add_argument('--mini2-mem', type=int, default=200)
    p.add_argument('--mini2-wall', default='72:00:00')
    p.add_argument('--mini2-procs', type=int, default=2)
    p.add_argument('--mini-ncpus', type=int, default=26)
    p.add_argument('--mini-mem', type=int, default=64)
    p.add_argument('--mini-wall', default='48:00:00')
    p.add_argument('--mini-procs', type=int, default=1)
    p.add_argument('--stage-queue', default='mini')
    p.add_argument('--stage-ncpus', type=int, default=24)
    p.add_argument('--stage-wall', default='24:00:00')
    p.add_argument('--last-stage', type=int, default=99, help='stop after the series of this stage')
    args = p.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(stage_failures={}, fetched_after=[])
    relay = None                                   # (stage, Popen) of the running relay
    while True:
        js = jobs()                                    # qstat first: a job that ends in between is still listed
        st = status() if js is not None else None
        if st is None or js is None:
            log('idark unreachable or status failed; waiting')
            time.sleep(args.poll)
            continue
        stages = st['stages']
        total_tasks = sum(s['tasks'] for s in stages)
        workers = [j for j in js if j['name'] == 'bf_regen']
        stage_jobs = {int(j['name'][4:]): j for j in js if re.match(r'bf_s\d\d$', j['name'])}
        line = ' | '.join(f"s{s['stage']} regen {s['regen_done']}/{s['tasks']} up {s['uploaded']}/{s['rows']}"
                          f"{' built' if s['built'] else ''}{' SBI' if s['series_done'] else ''}"
                          for s in stages if s['stage'] <= args.last_stage)
        log(f'{line} | workers {len(workers)} ({",".join(j["queue"] + ":" + j["state"] for j in workers)})')

        # 2. regen workers
        needed = [s for s in stages if s['stage'] <= args.last_stage]
        failed = sorted({label for s in needed for label in s.get('failed_tasks', [])})
        if failed:
            log(f'regen tasks failed (nn/<label>.failed; a worker gives a task up after 3 failures): {failed[:20]}')
        if any(s['regen_done'] < s['tasks'] for s in needed):
            if st['pending_claims']:
                # claims without output of dead workers (only running jobs count as alive: a requeued job skips its
                # old claims when it restarts); see beamfix.py reclaim for the age limits
                alive = ' '.join(j['id'] for j in workers if j['state'] == 'R')
                r = ssh(f'cd {RUN} && {PY_REMOTE} code/beamfix.py --run $HOME/{RUN} reclaim --alive {alive}')
                removed = json.loads(r.stdout).get('removed', []) if r and r.returncode == 0 else '?'
                if removed:
                    log(f'released stale claims: {removed}')
            for queue, want in (('mini2', args.mini2), ('mini', args.mini), ('mini_B', args.minib)):
                have = sum(j['queue'] == queue for j in workers)
                for _ in range(want - have):
                    submit_worker(queue, args)

        # 3. relay, one stage at a time in order
        if relay is not None and relay[1].poll() is not None:
            code = relay[1].returncode
            log(f'relay of stage {relay[0]} ended with {code}' + (' (another relay holds the lock)' if code == 3 else ''))
            if code == 4:
                log('the relay and idark disagree on the stage rows; stopping')
                return 1
            if code == 5:
                log('the clean spectra are not on idark yet (inputs/clean_dl_unbinned.npy); the relay waits')
            relay = None
        if any(s['uploaded'] < s['rows'] for s in needed) and not gcp_running():
            log('Globus Connect Personal is not connected; starting it')
            start_gcp()
        if relay is None:
            # relay at most one stage ahead of the noise regeneration: uploads far ahead only occupy disk
            frontier = min([s['stage'] for s in needed if s['regen_done'] < s['tasks']] or [99])
            pending = [s for s in needed if s['uploaded'] < s['rows'] and s['stage'] <= frontier + 1]
            if pending:
                k = pending[0]['stage']
                logfile = open(HERE / f'relay_stage{k:02d}.log', 'a')
                relay = (k, subprocess.Popen([PY_LOCAL, str(HERE / 'relay.py'), '--stage', str(k)], stdout=logfile,
                                             stderr=subprocess.STDOUT, env=LOCAL_ENV, cwd=HERE,
                                             start_new_session=True))
                relay_box[:] = [relay]
                log(f'relay of stage {k} started (log relay_stage{k:02d}.log)')

        # 4. stage jobs, in order
        for s in needed:
            k = s['stage']
            if s['series_done']:
                if k not in state['fetched_after']:
                    fetch()
                    state['fetched_after'].append(k)
                    STATE.write_text(json.dumps(state, indent=1))
                continue
            if k in stage_jobs:
                break                                           # running or queued
            ready = s['regen_done'] == s['tasks'] and s['uploaded'] == s['rows']
            if not ready:
                break
            failures = state['stage_failures'].get(str(k), 0)
            log_exists = ssh(f'ls {RUN}/logs/stage{k:02d}_*.log 2>/dev/null | wc -l')
            attempts = int(log_exists.stdout.strip() or 0) if log_exists else 0
            if attempts > failures:                              # an earlier job of this stage ended without output
                failures = attempts
                state['stage_failures'][str(k)] = failures
                STATE.write_text(json.dumps(state, indent=1))
                fetch()
            if failures >= 2:
                log(f'stage {k} failed twice; stopping (see fetched/logs/stage{k:02d}_*.log)')
                return 1
            submit_stage(k, args)
            break
        if all(s['series_done'] for s in needed):
            fetch()
            log('all stages done')
            return 0
        time.sleep(args.poll)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))       # run the finally block on kill
    sys.exit(main())
