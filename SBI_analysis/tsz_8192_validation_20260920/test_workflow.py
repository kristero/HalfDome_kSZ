"""Operational tests with fake subprocess work; no simulation is executed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='tsz_worker_test_') as directory:
    run=Path(directory);(run/'plan.json').write_text(json.dumps(dict(tasks=[dict(id=0,label='fake')])) )
    harness=run/'harness.py'
    harness.write_text('''import json,pathlib,sys,time
sys.path.insert(0,%r)
import worker
worker.ROOT=pathlib.Path(%r)
worker.source_hashes=lambda: {}
def fake(task):
    out=worker.ROOT/'controls'/'000';out.mkdir(parents=True,exist_ok=True)
    worker.atomic_json(out/'request.json',dict(task=task,source_sha256={}))
    started=out/'started'
    if not started.exists():
        started.write_text('partial')
        time.sleep(60)
    worker.atomic_json(out/'status.json',dict(task_id=0,returncode=0))
worker.run_task=fake
worker.main()
'''%(str(ROOT),str(run)))
    child=subprocess.Popen([sys.executable,str(harness)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    deadline=time.monotonic()+15
    while not (run/'controls/000/started').exists():
        if time.monotonic()>deadline:child.kill();raise RuntimeError('Fake worker did not start')
        time.sleep(.05)
    child.terminate();child.wait(timeout=10)
    assert not (run/'controls/000/status.json').exists()
    subprocess.run([sys.executable,str(harness)],check=True,capture_output=True)
    assert json.loads((run/'controls/000/status.json').read_text())['returncode']==0
    # A successful row must also reject changed identity, not silently skip it.
    request=run/'controls/000/request.json';identity=json.loads(request.read_text())
    identity['source_sha256']={'changed':'yes'};request.write_text(json.dumps(identity))
    failed=subprocess.run([sys.executable,str(harness)],capture_output=True,text=True)
    assert failed.returncode!=0 and 'Refusing resume' in failed.stderr
(ROOT/'results/workflow.json').write_text(json.dumps(dict(
    interruption_resume_passed=True,changed_identity_refused=True,
    scope='Python lock/status software test; physical Julia row checks are queued'),indent=2)+'\n')
print('PASS interruption/resume and changed-source refusal')
