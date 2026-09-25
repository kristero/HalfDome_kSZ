#!/usr/bin/env python3
"""Package the collaborator handoff, including both exact 32k design prefixes."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "SBI_analysis/so_32k_independent_noise"
XGPAINT = Path("/home/kn18001/.julia/dev/XGPaint")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def refresh_inputs():
    config = json.loads((BUNDLE / "config.json").read_text())
    if (config["n_rows"], config["sequence_offset"], config["design_dir"]) != (32768, 0, "designs"):
        raise ValueError("Refreshing original inputs is only supported for the default 32k configuration")
    helpers = ("run_halfdome_fullsky_so_noise.jl", "run_tSZ_visuals.jl", "config.jl",
               "binning.jl", "cosmology_helpers.jl", "instrumentation.jl", "output.jl",
               "painting.jl", "model.jl", "catalog_halfdome.jl", "catalog_websky.jl")
    for name in helpers:
        copy(ROOT / "tSZ_visuals" / name, BUNDLE / "simulator/tSZ_visuals" / name)
    for name in ("Project.toml", "Manifest.toml", "Artifacts.toml", "LICENSE", "README.md"):
        copy(XGPAINT / name, BUNDLE / "vendor/XGPaint" / name)
    for path in (XGPAINT / "src").glob("*.jl"):
        copy(path, BUNDLE / "vendor/XGPaint/src" / path.name)
    for case in ("baseline", "goal"):
        name = f"SO_LAT_Nell_T_atmv1_{case}_fsky0p4_ILC_tSZ.txt"
        copy(ROOT / "other_sims/SO" / name, BUNDLE / "noise" / name)
    designs = {
        "two_param": ROOT / "Sobol_tSZ/two_param_P0_beta_524288/battaglia_sobol_P0_beta_524288.csv",
        "nine_param": ROOT / "Sobol_tSZ/battaglia_sobol_1048576.csv",
    }
    provenance = {}
    for mode, source in designs.items():
        destination = BUNDLE / "designs" / (mode + ".csv")
        destination.parent.mkdir(exist_ok=True)
        with source.open(newline="") as src, destination.open("w", newline="") as dst:
            reader, writer = csv.reader(src), csv.writer(dst)
            writer.writerow(next(reader))
            for _ in range(32768):
                writer.writerow(next(reader))
        provenance[mode] = dict(original_file=source.name, source_sha256=digest(source),
            n_rows=32768, sequence_offset=0,
            selection="first 32768 data rows, in original CSV order", csv_sha256=digest(destination))
    (BUNDLE / "designs/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--refresh-inputs", action="store_true", help="Replace vendored sources/designs; requires renewed validation")
args = parser.parse_args()
if args.refresh_inputs:
    refresh_inputs()
# Default: repackage the validated snapshot, not possibly changed upstream sources.
paths = sorted(p for p in BUNDLE.rglob("*") if p.is_file()
               and p.name != "SHA256SUMS" and "__pycache__" not in p.parts)
(BUNDLE / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(BUNDLE)}\n" for p in paths))
archive = BUNDLE.with_suffix(".tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for path in paths + [BUNDLE / "SHA256SUMS"]:
        tar.add(path, arcname=str(Path(BUNDLE.name) / path.relative_to(BUNDLE)))
print("Folder:", BUNDLE)
print("Archive:", archive, "bytes:", archive.stat().st_size)
print("SHA256:", digest(archive))
