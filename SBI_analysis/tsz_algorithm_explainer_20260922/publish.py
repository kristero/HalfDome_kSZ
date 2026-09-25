"""Publish only the illustrated note and verify all delivered files on idark."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
PDF = ROOT.parents[1] / "output/pdf/tsz_algorithms_illustrated_20260922.pdf"
REMOTE = "/lustre/work/kristero10/" + ROOT.name


def run_remote(script):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "idark",
                             "/home/anaconda3/bin/python3", "-"], input=script, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    (ROOT / "FILES.md").write_text("""# Files created for the illustrated explanation

- `make_figures.py`: analytic boundary curves, HEALPix child geometry, cache counts, illustrative scheduling and beam diagrams; checks and source hashes.
- `build_note.py`: generates the illustrated PDF and `EXPLANATION.md`.
- `publish.py`: copies this explanatory bundle to idark and verifies file hashes; submits no jobs.
- `illustration_data.json`: numerical curve checks, pixel coordinates, resource counts and measured comparison values.
- `boundary_curves.csv`: all plotted pressure-shape, derivative and physical-projection curves.
- `plots/`: seven PNG figures and seven editable SVG figures.
- `qa.json`: PDF validation and illustration checks.
- `manifest.json`, `publication.json`: delivered checksums and remote verification.
- `../../output/pdf/tsz_algorithms_illustrated_20260922.pdf`: the final local PDF; published as `report.pdf` on the cluster.

Existing simulator, prior, cache implementation and prepared dataset files were read, not edited. No full-sky job or dataset was launched. Boundary curves and scheduling illustrations are identified separately from measured cluster results. Temporary rendered PDF pages and the verified delivery tarball are disposable.

Reproduce locally: run `make_figures.py` in the HalfDome scientific Python environment, then `build_note.py` with system Python and reportlab. Publishing uses Windows Python and the existing idark SSH alias. The source paths and hashes refer to the preserved sibling experiment `tsz_reuse_cache_20260921`.
""")
    paths = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.name not in ("manifest.json", "publication.json", "delivery.tar.gz")}
    paths["report.pdf"] = PDF
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    (ROOT / "manifest.json").write_text(json.dumps(dict(sha256=manifest), indent=2) + "\n")
    paths["manifest.json"] = ROOT / "manifest.json"
    archive = ROOT / "delivery.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, path in paths.items(): output.add(path, arcname=name)
    run_remote("from pathlib import Path\nimport json\nr=Path(" + repr(REMOTE) + ")\nr.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({'root':str(r)}))\n")
    subprocess.run(["scp", "-o", "ConnectTimeout=15", str(archive), "idark:" + REMOTE + "/delivery.tar.gz"], check=True)
    result = run_remote(r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,tarfile
root=Path(REMOTE_ROOT).resolve()
with tarfile.open(root/'delivery.tar.gz','r:gz') as source:
    for member in source.getmembers():
        assert member.isfile() and root in (root/member.name).resolve().parents,member.name
    source.extractall(root)
manifest=json.loads((root/'manifest.json').read_text())['sha256']
for name,digest in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'delivery.tar.gz').unlink()
result=dict(utc=datetime.now(timezone.utc).isoformat(),verified_files=len(manifest),root=str(root),pdf_sha256=manifest['report.pdf'])
(root/'publication.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''.replace("REMOTE_ROOT", repr(REMOTE)))
    (ROOT / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    archive.unlink()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
