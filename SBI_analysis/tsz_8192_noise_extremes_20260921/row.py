import json,sys,pathlib,hashlib
import worker
root=pathlib.Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['source_sha256'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
task=next(t for t in json.loads((root/'plan.json').read_text())['tasks'] if t['id']==int(sys.argv[1]))
(root/('claim_'+str(task['id']))).mkdir()
worker.run_task(task)
