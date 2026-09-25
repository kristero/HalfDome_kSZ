"""Publish the report and remove only verified duplicate transfer archives.

Use Windows Python with the configured SSH alias. --cleanup checks every archive
member against its extracted file before unlinking the archive. It never deletes
spectra, catalogues, logs, producer source, previous reports or unique versions.
--publish copies only this report bundle, verifies its SHA256 manifest on idark,
then removes the now-redundant delivery tarball. Neither mode submits a job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "tsz_reuse_cache_20260921"
REMOTE = "/lustre/work/kristero10/" + ROOT.name
REMOTE_SOURCE = "/lustre/work/kristero10/" + SOURCE.name
PDF = ROOT.parents[1] / "output/pdf/tsz_completed_performance_20260922.pdf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ssh_python(script):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "idark",
                             "/home/anaconda3/bin/python3", "-"], input=script,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def remove_duplicate_archives(root):
    """Only unlink an archive if every regular member has an identical live copy."""
    root = root.resolve()
    removed, retained = [], []
    # Explicit allowlist confines cleanup to this experiment's transfer files.
    names = ["evidence.tar.gz", "reports.tar.gz", "source.tar.gz", "followup/source.tar.gz",
             "edge_extension/source.tar.gz", "work_balance/source.tar.gz"]
    for name in names:
        archive = (root / name).resolve()
        if not archive.exists():
            continue
        if root not in archive.parents or archive.is_symlink():
            raise RuntimeError("Unsafe archive path: " + str(archive))
        reason, count = None, 0
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                target = (archive.parent / member.name).resolve()
                if root not in target.parents:
                    reason = "member outside experiment"; break
                if member.isdir():
                    continue
                if not member.isfile() or not target.is_file() or target.is_symlink():
                    reason = "member has no safe extracted copy"; break
                expected = hashlib.sha256(source.extractfile(member).read()).hexdigest()
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    reason = "archive retains a distinct version: " + member.name; break
                count += 1
        record = dict(path=str(archive), bytes=archive.stat().st_size,
                      sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), compared_members=count)
        if reason is None and count:
            archive.unlink()
            record["reason"] = "Every archived file has an identical extracted copy"
            removed.append(record)
        else:
            record["reason"] = reason or "empty archive"
            retained.append(record)
    return dict(removed=removed, retained=retained, reclaimed_bytes=sum(p["bytes"] for p in removed))


def cleanup():
    if (ROOT / "cleanup.json").exists():
        print("Cleanup already recorded; retaining the original receipt.")
        return
    manifest = json.loads((SOURCE / "fetch_manifest.json").read_text())
    # Verify remote extracted products first, then preserve the transfer manifest
    # itself on the cluster so it too has a durable copy outside the archive.
    script = "from pathlib import Path\nimport hashlib,json,tarfile\n" + inspect.getsource(remove_duplicate_archives)
    script += "\nroot=Path(" + repr(REMOTE_SOURCE) + ")\nexpected=" + repr(manifest) + "\n"
    script += "for name,digest in expected.items():\n    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n"
    script += "(root/'fetch_manifest.json').write_text(json.dumps(expected,indent=2)+'\\n')\n"
    script += "print(json.dumps(remove_duplicate_archives(root)))\n"
    remote = ssh_python(script)
    local = remove_duplicate_archives(SOURCE)
    result = dict(utc=datetime.now(timezone.utc).isoformat(), cluster=remote, local=local,
                  science_products_deleted=False)
    (ROOT / "cleanup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({side: result[side]["reclaimed_bytes"] for side in ("local", "cluster")}))


def publish():
    assert (ROOT / "cleanup.json").exists() and PDF.exists()
    scripts = sorted(p.name for p in ROOT.glob("*.py"))
    lines = ["# Files created or updated on 22 September 2026", "",
             "All new analysis source is in this directory. Frozen simulator/producer code was not edited in this action.", "",
             "## New source files", ""] + ["- `" + p + "`" for p in scripts]
    lines += ["", "## New results and documents", "",
              "- `REPORT.md`: completed report, from the same content as the eight-page PDF.",
              "- `results.json`, `spectra.npz`: recomputed comparisons and plotting data.",
              "- `timings.csv`, `errors.csv`, `resolution.csv`, `pixel_quadrature.csv`: machine-readable tables.",
              "- `plots/`: five PNG and five editable SVG figures.",
              "- `cluster_audit.json`: empty queue and cluster-side checksum verification.",
              "- `cleanup.json`: per-archive hashes and reasons for removal or retention.",
              "- `qa.json`: final PDF and result validation.",
              "- `delivery_manifest.json`, `publication.json`: deliverable hashes and cluster verification.",
              "- `../../output/pdf/tsz_completed_performance_20260922.pdf`: final PDF (local); `report.pdf` inside the cluster bundle.",
              "", "## Existing experiment paths affected", "",
              "- `../tsz_reuse_cache_20260921/controls/`, `edge_extension/controls/`, `work_balance/controls/`, `results/`, `plots/`: refreshed local copies of completed cluster products via the existing fetch.py; 312 products verified.",
              "- `../tsz_reuse_cache_20260921/fetch_manifest.json`: new complete fetch manifest; also saved on the cluster.",
              "- Duplicate transfer archives listed in cleanup.json were deleted only after every member matched an extracted file. Distinct archived versions were retained.",
              "", "No production launch, new prior cut or noise-seed change is performed by these reporting scripts."]
    (ROOT / "FILES.md").write_text("\n".join(lines) + "\n")
    files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*") if p.is_file() and
             "__pycache__" not in p.parts and p.name not in ("delivery.tar.gz", "delivery_manifest.json", "publication.json")}
    files["report.pdf"] = PDF
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), sha256={name: sha(path) for name, path in files.items()})
    (ROOT / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files["delivery_manifest.json"] = ROOT / "delivery_manifest.json"
    archive = ROOT / "delivery.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, path in files.items():
            output.add(path, arcname=name)
    ssh_python("from pathlib import Path\nimport json\nr=Path(" + repr(REMOTE) + ")\nr.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({'created':str(r)}))\n")
    subprocess.run(["scp", str(archive), "idark:" + REMOTE + "/delivery.tar.gz"], check=True)
    script = r'''
from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime, timezone
r=Path(REMOTE_ROOT).resolve()
with tarfile.open(r/'delivery.tar.gz','r:gz') as source:
    for member in source.getmembers():
        target=(r/member.name).resolve()
        assert r in target.parents and member.isfile(),member.name
    source.extractall(r)
manifest=json.loads((r/'delivery_manifest.json').read_text())['sha256']
for name,digest in manifest.items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
(r/'delivery.tar.gz').unlink()
result=dict(utc=datetime.now(timezone.utc).isoformat(),verified_files=len(manifest),root=str(r),pdf_sha256=manifest['report.pdf'])
(r/'publication.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''.replace("REMOTE_ROOT", repr(REMOTE))
    result = ssh_python(script)
    (ROOT / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    archive.unlink()
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
    if args.publish:
        publish()
