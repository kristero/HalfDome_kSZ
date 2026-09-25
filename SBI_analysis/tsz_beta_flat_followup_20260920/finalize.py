"""Inventory a completed, separate experiment and append validation evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
validation=json.loads((root/'results/independent_validation.json').read_text())
flat=json.loads((root/'results/continuous_flat_validation.json').read_text())
julia=(root/'results/julia_verification.toml').read_text()
assert 'passed = true' in julia
assert flat['sampled_violations']==0 and flat['marginal_max_relative_error']<1e-10
assert validation['production']['manifest_unchanged']
execution=dict(finalized_utc=datetime.now(timezone.utc).isoformat(),
    main_experiment=dict(host='idark',pbs_job='598374.idark',python_stage='completed',
                         first_julia_stage='failed test-driver import; fixed'),
    scalar_validation=dict(host='idark',mode='direct bounded single-thread validation',timeout_seconds=90,
        note='Queued follow-up 598434 was cancelled before direct validation; production jobs were not changed'),
    checks='passed',production_hashes_checked=validation['production']['checked_hashes'])
(root/'results/execution.json').write_text(json.dumps(execution,indent=2)+'\n')
report=root/'RESULTS.md'
text=report.read_text().split('\n## Final verification record\n')[0]
text+='\n## Final verification record\n\n'
text+='The main profile, beta-support and noise experiments ran in the Python stage of PBS job 598374 on the cluster. Two Julia test-driver issues (dependency import and local scope) and a plotting-keyword incompatibility with the older cluster Matplotlib were fixed; the diagnostic logs are retained. The follow-up queue was saturated; job 598434 was cancelled and the short scalar validation was completed directly on idark with one thread and a 90-second timeout. Plotting and the continuous-prior construction used separate 60-second limits. No production job was changed.\n\n'
text+='Final Julia record:\n\n```toml\n'+julia.strip()+'\n```\n\n'
text+='Independent 60-digit quadrature maximum LOS relative error: %.5g; maximum energy relative error: %.5g. All %d production hashes were unchanged. The continuous sampler was also rebuilt and checked on the cluster, with %d sampled support violations.\n' % (validation['max_los_relative_error'],validation['max_energy_relative_error'],validation['production']['checked_hashes'],flat['sampled_violations'])
report.write_text(text)
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and p.name not in ('artifact_manifest.json','changed_files.txt'))
(root/'changed_files.txt').write_text('All files are new under this separate experiment root.\nNo existing production or previous-study file was edited.\n\n'+'\n'.join(str(p.relative_to(root)) for p in files)+'\n')
files.append(root/'changed_files.txt')
manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(execution,indent=2))
