#!/usr/bin/env python3
"""Fetch only completed comparison artifacts, never the full FLAMINGO maps."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


REMOTE_SCRIPT = r'''
import hashlib, json, sys, tarfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
complete = root / 'campaign_complete.json'
if not complete.exists():
    raise SystemExit('Campaign is not complete; inspect its cluster logs')
files = [complete, root/'approval.json', root/'inputs/download_complete.json',
         root/'cache/complete.toml', root/'cache/masked_noise_cross_cl.npy',
         root/'preflight/operator_comparison.json', root/'preflight/metadata_manifest.json']
for directory in ('comparison', 'results', 'logs'):
    files.extend(p for p in (root/directory).rglob('*') if p.is_file())
for directory in ('report',):
    files.extend(p for p in (root/directory).rglob('*') if p.is_file())
for name in ('resource_usage.json', 'job_history.json', 'RUN_REPORT.md',
             'preflight/halfdome_control_precheck.json'):
    if (root/name).is_file():
        files.append(root/name)
if sum(p.stat().st_size for p in files) > 100*1024**2:
    raise SystemExit('Unexpectedly large result bundle; inspect before transfer')
archive = root / 'comparison_results.tar.gz'
with tarfile.open(archive, 'w:gz') as handle:
    for path in files:
        handle.add(path, arcname=path.relative_to(root).as_posix())
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
print(json.dumps(dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest)))
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="idark")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        args.host, "/home/anaconda3/bin/python3 - " + shlex.quote(args.remote_root)],
        input=REMOTE_SCRIPT, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    receipt = json.loads(result.stdout)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "comparison_results.tar.gz"
    subprocess.run(["scp", "-o", "ConnectTimeout=15", args.host+":"+receipt["path"], str(archive)],
                   check=True, timeout=180)
    if archive.stat().st_size != receipt["bytes"] or sha256(archive) != receipt["sha256"]:
        raise ValueError("Result archive checksum mismatch")
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            (output / member.name).resolve().relative_to(output)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Unexpected non-file archive entry")
        handle.extractall(output, filter="data")
    complete_path = output / "comparison/complete.json"
    complete = json.loads(complete_path.read_text())
    campaign = json.loads((output / "campaign_complete.json").read_text())
    if sha256(complete_path) != campaign["completion_sha256"]:
        raise ValueError("Campaign completion manifest changed")
    for relative, expected in complete["files"].items():
        path = (output / relative).resolve()
        path.relative_to(output)
        if sha256(path) != expected:
            raise ValueError("Changed result artifact: " + relative)
    receipt.update(verified_artifacts=len(complete["files"]), status="all_checksums_passed")
    (output / "local_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
