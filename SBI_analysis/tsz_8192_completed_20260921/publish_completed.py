"""Stage the report and unsubmitted preparation on idark, then verify hashes."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
PREPARED=ROOT.parent/'tsz_diagnostic_256_8192_prepared_20260921'
PDF=ROOT.parents[1]/'output/pdf/tsz_completed_validation_20260921.pdf'
REMOTE='/lustre/work/kristero10'


def main():
    inventory=['# Files added or updated for completed validation','',
               'The earlier partial-results PDF and its scripts are retained.','',
               '## New report and audit sources','']
    inventory.extend('- `'+p.name+'`' for p in sorted(ROOT.glob('*.py')))
    inventory.extend(['','## New prepared package','',
        '`../tsz_diagnostic_256_8192_prepared_20260921/`:','',
        '- Revised `diagnostic.py`: raw8192, atomic row claims, preserved failures, frozen-input checks and one final collector.',
        '- Revised `diagnostic_row.jl`: raw8192 assertion, validated lean allocation and a load-only guard.',
        '- Revised `diagnostic.pbs` and `diagnostic_analysis.pbs`: four explicit workers, one dependent collector/analysis; not submitted.',
        '- `fullsky_test.jl` and `spherical_truncation_profiles.jl`: byte-identical copies from the completed validation.',
        '- `diagnostic_analysis.py`, `unbinned_moped.py`, `design_tests.py`, `run_fullsky.py`, `stage_observations.py`, `test_unbinned_moped.py`: copied support and analysis sources.',
        '- `diagnostic_256/manifest.json`: updated resolution, source hashes and preparation provenance.',
        '- Saved theta, noise seeds, held-out mask and three FLAMINGO observation artifacts: preserved byte-for-byte.',
        '- `README.md`: scope and remaining numerical limits.','',
        '## Retrieved/generated evidence','',
        '- Completed task/analysis products refreshed in the recovery and extreme-noise directories.',
        '- This directory: cluster reports, audits, five figure pairs, REPORT.md and artifact manifest.',
        '- Final standalone report: output/pdf/tsz_completed_validation_20260921.pdf.',
        '- No simulations or SBI training were launched.'])
    (ROOT/'FILES.md').write_text('\n'.join(inventory)+'\n',encoding='utf-8')
    entries=[]
    for root in [ROOT,PREPARED]:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if path.name in ['artifact_manifest.json','publication.json']:continue
            entries.append((path,root.name+'/'+path.relative_to(root).as_posix()))
    entries.append((PDF,ROOT.name+'/'+PDF.name))
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in entries}
    marker=ROOT/'artifact_manifest.json';marker.write_text(json.dumps(manifest,indent=2)+'\n')
    entries.append((marker,ROOT.name+'/'+marker.name))
    check=f"from pathlib import Path; p=Path('{REMOTE}/{PREPARED.name}/diagnostic_256/rows'); assert not p.exists() or not any(p.iterdir()), 'Prepared target already started'"
    subprocess.run(['ssh','-o','ConnectTimeout=15','idark','/home/anaconda3/bin/python3','-'],input=check.encode(),check=True)
    blob=io.BytesIO()
    with tarfile.open(fileobj=blob,mode='w:gz') as archive:
        for path,name in entries:archive.add(path,arcname=name)
    subprocess.run(['ssh','-o','ConnectTimeout=15','idark','tar','-xzf','-','-C',REMOTE],input=blob.getvalue(),check=True)
    check=f"import pathlib,json,hashlib; r=pathlib.Path('{REMOTE}'); m=json.loads((r/'{ROOT.name}/artifact_manifest.json').read_text()); assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v for k,v in m.items()); print(len(m))"
    verified=subprocess.check_output(['ssh','idark','/home/anaconda3/bin/python3','-'],input=check.encode()).decode().strip()
    result=dict(verified_files=int(verified),report_root=REMOTE+'/'+ROOT.name,
                prepared_root=REMOTE+'/'+PREPARED.name,submitted=False,archive_bytes=len(blob.getvalue()))
    (ROOT/'results/publication.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
