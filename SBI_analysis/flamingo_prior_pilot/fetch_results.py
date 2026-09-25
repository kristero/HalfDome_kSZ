"""Fetch verified pilot spectra, audits, tables and plots; leave maps on cluster."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from prior_model import save_json, sha256

REMOTE_SCRIPT = r'''
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve()
verification=json.loads((root/'verification.json').read_text())
assert verification['status']=='passed'
manifest=json.loads((root/'artifact_manifest.json').read_text())
files=[root/p for p in manifest['files']]
for name in ('pilot_results.json','resource_usage.json','verification.json','artifact_manifest.json',
             'parameters.csv','fit_spectra.csv','noise_comparison.json','job_history.json','RUN_REPORT.md',
             'changed_files.txt','split_jobs.json','queue_moves.json','disk_usage.json'):
    if (root/name).exists(): files.append(root/name)
files.extend(p for p in (root/'logs').glob('*') if p.is_file())
if sum(p.stat().st_size for p in files)>100*1024**2:
    raise SystemExit('Unexpectedly large artifact bundle')
archive=root/'pilot_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as handle:
    for path in files:
        handle.add(path,arcname=path.relative_to(root).as_posix())
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="idark")
    parser.add_argument("--remote-root", default="/lustre/work/kristero10/flamingo_prior_pilot_20260914")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "cluster_results")
    args = parser.parse_args()
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host,
        "/home/anaconda3/bin/python3 - "+shlex.quote(args.remote_root)],
        input=REMOTE_SCRIPT, text=True, capture_output=True, check=True, timeout=120)
    receipt = json.loads(response.stdout)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "pilot_artifacts.tar.gz"
    subprocess.run(["scp", "-o", "ConnectTimeout=15", args.host+":"+receipt["path"], str(archive)],
                   check=True, timeout=180)
    assert sha256(archive) == receipt["sha256"] and archive.stat().st_size == receipt["bytes"]
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            (output / member.name).resolve().relative_to(output)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Unexpected non-file archive member")
        handle.extractall(output, filter="data")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        path.relative_to(output)
        assert sha256(path) == expected, relative
    receipt.update(status="all_checksums_passed", verified_artifacts=len(manifest["files"]))
    save_json(output / "local_verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
