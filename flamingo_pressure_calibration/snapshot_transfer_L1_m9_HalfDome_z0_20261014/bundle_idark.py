"""Run on idark after aggregation; package compact deliverables and evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import tarfile

ROOT=Path('/lustre/work/kristero10/snapshot_transfer_L1_m9_HalfDome_z0_20261014')
complete=json.loads((ROOT/'cluster_analysis_complete.json').read_text())
assert complete['completed']
for name,value in complete['figures'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name

jobs=['602967.idark','602969.idark']+['602974[%d].idark'%i for i in range(5)]
jobs+=['602976.idark','603000.idark','603002.idark']
retry=ROOT/'run_aggregate_final.job.json'
if retry.exists():jobs.append(json.loads(retry.read_text())['job_id'])
refine=ROOT/'run_refine_figures.job.json'
if refine.exists():jobs.append(json.loads(refine.read_text())['job_id'])
report=ROOT/'run_report.job.json'
if report.exists():jobs.append(json.loads(report.read_text())['job_id'])
report_retry=ROOT/'run_report_retry.job.json'
if report_retry.exists():jobs.append(json.loads(report_retry.read_text())['job_id'])
report_final=ROOT/'run_report_final.job.json'
if report_final.exists():jobs.append(json.loads(report_final.read_text())['job_id'])
report_layout=ROOT/'run_report_layout.job.json'
if report_layout.exists():jobs.append(json.loads(report_layout.read_text())['job_id'])
accounting={}
for job in jobs:
    result=subprocess.run(['qstat','-x','-f',job],capture_output=True,text=True)
    accounting[job]=dict(returncode=result.returncode,text=result.stdout or result.stderr)
(ROOT/'pbs_accounting_raw.json').write_text(json.dumps(accounting,indent=2))

frozen=['inputs/measurements.npz','inputs/selected_halos.npz','inputs/metadata_plan.json',
        'fits.json','fits_sem_weighting_control.json','mass_mapping.json',
        'regions_halfdome.h5','regions_flamingo.h5','measured_flamingo_compact.h5',
        'measured_halo_union_compact.h5']
hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in frozen if (ROOT/n).exists()}
code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
verified=dict(completed=True,timestamp_utc=datetime.datetime.utcnow().isoformat()+'Z',
              cluster='idark',root=str(ROOT),frozen_input_hashes=hashes,code_hashes=code,
              computed_cases=57,derived_cases=2,verified_figure_count=len(complete['figures']),
              job_instances=len(jobs),raw_snapshot_transfer=False,
              scope='Live hashes of compact inputs/results; not a hash of all raw simulation files.')
(ROOT/'CLUSTER_VERIFIED.json').write_text(json.dumps(verified,indent=2))

files=list(ROOT.glob('*.json'))+list((ROOT/'figures').glob('*'))
files += [p for p in ROOT.glob('*.log') if p.stat().st_size<2_000_000]
files += list((ROOT/'cases').glob('*.json'))
files += list((ROOT/'output/pdf').glob('*.pdf'))
files += list((ROOT/'tmp/pdfs/summary_render').glob('*.png'))
files += list((ROOT/'tmp/pdfs/detail_render').glob('*.png'))
archive=ROOT/'delivery_bundle.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=4) as tar:
    for p in sorted(files):
        tar.add(p,arcname=str(p.relative_to(ROOT)))
receipt=dict(file=archive.name,bytes=archive.stat().st_size,
             sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))
(ROOT/'delivery_bundle_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
