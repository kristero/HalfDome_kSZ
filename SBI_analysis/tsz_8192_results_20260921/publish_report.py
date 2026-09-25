"""Copy the dated report and its small reproducibility bundle to idark.

Run with the Python installation whose SSH configuration defines `idark`.
This only publishes artifacts into the dedicated audit directory; it does
not submit jobs or modify simulator source files.
"""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF = ROOT.parents[1] / "output/pdf/tsz_resolution_and_original_xgpaint_20260921.pdf"
REMOTE = "/lustre/work/kristero10/tsz_8192_results_20260921"
SUFFIXES = {".py", ".jl", ".pbs", ".md", ".json", ".toml", ".csv", ".npy", ".png", ".pdf", ".txt", ".log"}


def main():
    sources = sorted(p for p in ROOT.iterdir() if p.suffix in {".py", ".jl", ".pbs"})
    inventory = ["# Files created or updated for the 21 September audit", "",
        "All paths below are relative to this report directory.", "",
        "## New audit and orchestration source files", ""]
    inventory.extend(f"- `{p.name}`" for p in sources)
    inventory.extend(["", "## Generated artifacts and copied sources", "",
        "- `REPORT.md`, `plots/`, `results/`, and the dated PDF: report, evidence and figures.",
        "- `inputs/stock_profiles*.jl`: snapshots of original XGPaint commit 5dd0b57.",
        "- `../tsz_8192_validation_recovery_20260921/`: frozen physical producers, revised task plan, disjoint workers and PBS scripts.",
        "- `../tsz_8192_noise_extremes_20260921/`: frozen producers and two additional matched-noise controls.",
        "- `../tsz_spherical_preflight_20260920/results/fullsky.json` and its plots: regenerated from the completed independent resolution controls.", "",
        "The recovered jobs use byte-identical physical producer files. This audit changed job orchestration and added analysis/probes; it did not change prior bounds or the simulation physics.", ""])
    (ROOT / "FILES.md").write_text("\n".join(inventory), encoding="utf-8")
    paths = [p for p in sorted(ROOT.rglob("*")) if p.is_file()
             and p.suffix in SUFFIXES and "__pycache__" not in p.parts
             and p.name not in {"artifact_manifest.json", "publication.json"}]
    entries = [(p, p.relative_to(ROOT).as_posix()) for p in paths]
    entries.append((PDF, PDF.name))
    manifest = {name: dict(bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for path, name in entries}
    manifest_path = ROOT / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    entries.append((manifest_path, manifest_path.name))
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        for path, name in entries:
            archive.add(path, arcname=name)
    subprocess.run(["ssh", "-o", "ConnectTimeout=15", "idark", "mkdir", "-p", REMOTE], check=True)
    subprocess.run(["ssh", "-o", "ConnectTimeout=15", "idark", "tar", "-xzf", "-", "-C", REMOTE],
                   input=blob.getvalue(), check=True)
    # Verify bytes on the destination, including the final PDF.
    check = "import pathlib,json,hashlib; r=pathlib.Path('" + REMOTE + "'); "
    check += "m=json.loads((r/'artifact_manifest.json').read_text()); "
    check += "assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v['sha256'] for k,v in m.items()); print(len(m))"
    verified = subprocess.check_output(["ssh", "idark", "/home/anaconda3/bin/python3", "-"],
                                       input=check.encode()).decode().strip()
    publication = dict(remote=REMOTE, verified_files=int(verified), archive_bytes=len(blob.getvalue()))
    (ROOT / "results/publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(json.dumps(publication, indent=2))


if __name__ == "__main__":
    main()
