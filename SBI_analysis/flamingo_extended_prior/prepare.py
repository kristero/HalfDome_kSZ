"""Freeze a deterministic conditional design and the inputs for one run."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from prior import JointPrior, digest, write_json
from noise_seeds import split_seeds


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--pilot-results", type=Path)
    args = parser.parse_args()
    if args.count <= 0 or args.count > 524288:
        parser.error("count must be in 1..524288")
    source = Path(__file__).resolve().parent
    root = args.root.resolve()
    if (root / "manifest.json").exists():
        raise SystemExit("Run already frozen; resume its worker, or choose a new root")
    for name in ("code", "design", "chunks", "logs", "locks", "scratch", "preflight", "observations"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix in (".py", ".jl", ".json", ".pbs", ".md") and path.is_file():
            target = root / "code" / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
    run = json.loads((source / "run_config.json").read_text())
    run["count"] = args.count
    write_json(root / "run_config.json", run)
    prior = JointPrior()
    theta, proposal_index = prior.sobol(args.count)
    np.save(root / "design/theta.npy", theta)
    np.save(root / "design/proposal_index.npy", proposal_index)
    seeds = np.array([split_seeds(i, run["noise_master_seed"]) for i in range(args.count)], dtype=np.int64)
    assert len(np.unique(seeds)) == 2*args.count
    np.save(root / "design/noise_split_seeds.npy", seeds)
    # Stable validation split: rows retain their role when 8k grows to 524k.
    # Hash before selecting: index % 10 directly correlates with Sobol bits
    # and can produce a biased validation subset after conditional rejection.
    hashed = proposal_index.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    hashed = hashed ^ (hashed >> np.uint64(31))
    split = (hashed % np.uint64(10) == 0).astype(np.uint8)
    np.save(root / "design/validation_split.npy", split)
    np.savetxt(root / "design/parameters.csv", theta, delimiter=",",
               header=",".join(prior.names), comments="", fmt="%.17g")
    good, metrics = prior.contains(theta, True)
    assert good.all()
    pilot = args.pilot_results or Path(run["pilot"])
    fits = list(csv.DictReader((pilot / "parameters.csv").open()))[:4]
    cases = []
    for row in fits:
        values = [float(row[key]) for key in prior.names]
        assert prior.contains(values), row["name"]
        cases.append(dict(name=row["name"], theta=values, role=row["kind"]))
    for label, index in (("compact_boundary", np.argmin(metrics["size_min"])),
                         ("steep_boundary", np.argmax(metrics["beta_max"]))):
        cases.append(dict(name=label, theta=theta[index].tolist(),
                          role="accepted_design_boundary", design_row=int(index)))
    write_json(root / "preflight/cases.json", cases)
    manifest = dict(schema_version=1, prior=prior.config,
        accepted_count=args.count, last_proposal_index=int(proposal_index[-1]),
        accepted_fraction_prefix=float(args.count/(proposal_index[-1]+1)),
        parameter_order=prior.names,
        metrics={k: [float(v.min()), float(v.max())] for k,v in metrics.items()},
        design_sha256={p.name: digest(p) for p in (root/"design").iterdir()},
        code_sha256={p.name: digest(p) for p in (root/"code").iterdir() if p.is_file()},
        run_config_sha256=digest(root/"run_config.json"),
        preflight_cases_sha256=digest(root/"preflight/cases.json"))
    # Frozen campaign source hashes include pressure implementation, not just
    # the XGPaint module entrypoint fingerprint in the old noise cache.
    campaign = Path(run["campaign"])
    if campaign.exists():
        dependencies = list((campaign/"code").rglob("*.jl"))
        dependencies += list((campaign/"runtime/XGPaint/src").glob("*.jl"))
        dependencies += [campaign/"cache/complete.toml", campaign/"preflight/metadata_manifest.json"]
        manifest["dependency_sha256"] = {str(p): digest(p) for p in dependencies}
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"root": str(root), "count": len(theta),
                      "acceptance": manifest["accepted_fraction_prefix"],
                      "chunks": (args.count+run["chunk_size"]-1)//run["chunk_size"]}, indent=2))


if __name__ == "__main__":
    main()
