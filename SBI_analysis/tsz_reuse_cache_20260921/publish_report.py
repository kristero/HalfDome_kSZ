"""Publish supplementary reports, without replacing any frozen producer file."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/'+ROOT.name
sources=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.jl','.pbs')
               and '__pycache__' not in p.parts)
lines=['# Files created for the reuse/cache experiments','',
       'All implementation changes are additions under `SBI_analysis/tsz_reuse_cache_20260921/`. '
       'No existing simulator or prepared-dataset source was edited. Frozen numerical copies are '
       'identified in README.md.','',
       '## Source and scheduler files','']
lines+=['- `'+p.relative_to(ROOT).as_posix()+'`' for p in sources]
lines+=['','## Main reports and measured evidence','',
        '- `README.md`: scope, physics, memory, cache grids, commands.',
        '- `REPORT.md`, `results/progress.json`: completed numerical checks and full-catalogue controls.',
        '- `PIXEL_PROGRESS.md`, `results/pixel_progress.json`: finite pixel-quadrature comparison.',
        '- `followup/baseline_cache.json`: default-cache error in unbinned MOPED units.',
        '- `followup/probe/cache_probe.toml`: 67,584 scalar cache comparisons.',
        '- `edge_extension/README.md`: analytic explanation of the exterior derivative kink.',
        '- `plots/`: figures with concise labels and large fonts.',
        '- `manifest.json`, subdirectory manifests and `fetch_manifest.json`: source/product SHA256 evidence.',
        '- Submission journals and per-control request/status/timing files: cluster provenance.']
(ROOT/'FILES.md').write_text('\n'.join(lines)+'\n')
publish=[ROOT/p for p in ['README.md','REPORT.md','PIXEL_PROGRESS.md','FILES.md',
    'report_progress.py','pixel_progress.py','fetch.py','snapshot_cluster.py','publish_report.py',
    'results/progress.json','results/pixel_progress.json','results/cluster_snapshot.json',
    'followup/recovery.json','edge_extension/README.md']]+list((ROOT/'plots').glob('*.png'))
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in publish}
(ROOT/'report_manifest.json').write_text(json.dumps(dict(sha256=hashes),indent=2)+'\n')
with tarfile.open(ROOT/'reports.tar.gz','w:gz') as archive:
    for p in publish+[ROOT/'report_manifest.json']:archive.add(p,arcname=p.relative_to(ROOT).as_posix())
subprocess.run(['scp',str(ROOT/'reports.tar.gz'),'idark:'+REMOTE+'/reports.tar.gz'],check=True)
subprocess.run(['ssh','idark','tar','-xzf',REMOTE+'/reports.tar.gz','-C',REMOTE],check=True)
script='''from pathlib import Path
import hashlib,json
r=Path(%r)
m=json.loads((r/'report_manifest.json').read_text())['sha256']
for n,h in m.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
print(json.dumps(dict(verified=len(m))))
'''%REMOTE
result=subprocess.run(['ssh','idark','/home/anaconda3/bin/python3','-'],input=script,
                      text=True,capture_output=True,check=True)
print(result.stdout)
(ROOT/'results/publication.json').write_text(result.stdout)
