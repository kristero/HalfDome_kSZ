"""Upload the prepared 256-row design. This script cannot submit PBS jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parent
TARGET='/lustre/work/kristero10/tsz_spherical_diagnostic_256_prepared_20260920'
files=list((ROOT/'diagnostic_256').glob('*'))
files=[p for p in files if p.name in ['manifest.json','theta_design.npy','noise_seeds.npy','heldout_design.npy']]
expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert len(files)==4
expected.update({'observations/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT/'diagnostic_256/observations').glob('*') if p.is_file()})
manifest_text=(ROOT/'diagnostic_256/manifest.json').read_text()
# Refuse to overwrite a distinct prepared configuration or any generated rows.
probe='''import pathlib,json,sys
p=pathlib.Path("%s")
if (p/"rows").exists() and any((p/"rows").iterdir()):raise RuntimeError("Rows already exist")
expected=json.loads(%r)
if (p/"manifest.json").exists() and json.loads((p/"manifest.json").read_text())!=expected:
    (p/"manifest.previous.json").write_text((p/"manifest.json").read_text())
p.mkdir(parents=True,exist_ok=True)
'''%(TARGET,manifest_text)
subprocess.run(['ssh','idark','/home/anaconda3/bin/python3','-'],input=probe.encode(),check=True)
with tarfile.open(ROOT/'prepared_256.tar.gz','w:gz') as archive:
    for p in files:archive.add(p,arcname=p.name)
    for p in (ROOT/'diagnostic_256/observations').glob('*'):archive.add(p,arcname='observations/'+p.name)
    archive.add(ROOT/'PREPARED_256.md',arcname='README.md')
    archive.add(ROOT/'NUMERICAL_CHANGE_AUDIT.md',arcname='NUMERICAL_CHANGE_AUDIT.md')
subprocess.run(['scp',str(ROOT/'prepared_256.tar.gz'),'idark:'+TARGET+'/prepared.tar.gz'],check=True)
subprocess.run(['ssh','idark','tar','-xzf',TARGET+'/prepared.tar.gz','-C',TARGET],check=True)
probe='''import pathlib,hashlib,json
p=pathlib.Path("%s")
names=%r
print(json.dumps({name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in names}))
'''%(TARGET,list(expected))
actual=json.loads(subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=probe.encode()))
assert actual==expected
result=dict(cluster_root=TARGET,count=256,sha256=actual,submitted=False,generated_rows=0)
(ROOT/'results/prepared_256_cluster.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
