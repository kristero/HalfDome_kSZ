#!/usr/bin/env python3
"""Package the reference Julia environment's required dependency closure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia-depot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster-runtime", required=True)
    args = parser.parse_args()
    text = (args.julia_depot / "environments/v1.12/Manifest.toml").read_text()
    manifest = tomllib.loads(text)
    if manifest["julia_version"] != "1.12.2":
        raise ValueError("Reference Julia version changed")
    dependencies = manifest["deps"]
    roots = ["XGPaint", "Healpix", "HDF5", "Interpolations"]
    selected, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        for entry in dependencies[name]:
            pending.extend(entry.get("deps", []))
            # A manifest weakdeps list resolves UUIDs through other manifest
            # entries. Keep those entries too; dictionary-form weakdeps already
            # supplies UUIDs and can legally refer to absent optional packages.
            weak = entry.get("weakdeps", {})
            if isinstance(weak, list):
                pending.extend(weak)
    args.output.mkdir(exist_ok=True)
    staging = args.output / "runtime_env"
    staging.mkdir(exist_ok=True)
    project = "[deps]\n" + "\n".join(
        name + " = " + json.dumps(dependencies[name][0]["uuid"]) for name in roots) + "\n"
    blocks = re.split(r"(?m)(?=^\[\[deps\.)", text)
    kept = [blocks[0]] + [block for block in blocks[1:] if block.split("]]", 1)[0][7:] in selected]
    pinned = "".join(kept).replace("/home/kn18001/.julia/dev/XGPaint", args.cluster_runtime + "/XGPaint")
    (staging / "Project.toml").write_text(project)
    (staging / "Manifest.toml").write_text(pinned)
    source = args.julia_depot / "dev/XGPaint"
    sources = list((source / "src").rglob("*.jl")) + [source / "Project.toml", source / "Artifacts.toml"]
    hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    with tarfile.open(args.output / "runtime_sources.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname="XGPaint/" + path.relative_to(source).as_posix())
        for name in ("Project.toml", "Manifest.toml"):
            archive.add(staging / name, arcname="julia_env/" + name)
    (args.output / "runtime_source_manifest.json").write_text(json.dumps(dict(
        julia_version="1.12.2", packages=sorted(selected), source_hashes=hashes,
        original_manifest_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2) + "\n")
    print("Pinned packages:", len(selected), "; XGPaint source files:", len(sources))


if __name__ == "__main__":
    main()
