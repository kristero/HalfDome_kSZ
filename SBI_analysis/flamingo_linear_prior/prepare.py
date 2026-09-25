"""Freeze broad linear-prior generation, preserving the existing map physics."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from design import build_design, THINNING_SALT, OLD_SLOT_STRIDE
from noise_seeds import split_seeds
from prior import JointPrior, digest, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    args = parser.parse_args()
    if not 1 <= args.count <= 524288:
        parser.error("count must be in 1..524288")
    source, root = Path(__file__).resolve().parent, args.root.resolve()
    if (root / "manifest.json").exists():
        raise SystemExit("Run already frozen; use a new root or resume its workers")
    for name in ("code", "inputs", "design", "chunks", "logs", "locks", "scratch",
                 "preflight", "observations", "plots", "audit"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".py", ".jl", ".json", ".pbs", ".md"):
            if path.resolve() != (root / "code" / path.name).resolve():
                shutil.copy2(path, root / "code" / path.name)
    run = json.loads((source / "run_config.json").read_text())
    run["count"] = args.count
    write_json(root / "run_config.json", run)
    old, campaign, refit = [Path(run[k]) for k in ("previous_run", "campaign", "cosmology_refit")]
    records = {"previous_manifest.json": old / "manifest.json",
               "previous_theta.npy": old / "design/theta.npy",
               "old_metadata.json": campaign / "preflight/metadata_manifest.json",
               "old_comparison.json": campaign / "comparison/comparison_summary.json",
               "flamingo_fits.json": refit / "results/comparison_summary.json"}
    for name, path in records.items():
        shutil.copy2(path, root / "inputs" / name)
    prior = JointPrior()
    design = build_design(root, args.count)
    theta = design["theta"]
    good, metrics = prior.contains(theta, True)
    assert good.all()
    for name in ("theta", "proposal_index", "source_old_row", "noise_seed_row_ids"):
        np.save(root / "design" / (name + ".npy"), design[name])
    seeds = np.array([split_seeds(int(i), run["noise_master_seed"]) for i in design["noise_seed_row_ids"]],
                     dtype=np.int64)
    assert len(np.unique(seeds)) == seeds.size
    np.save(root / "design/noise_split_seeds.npy", seeds)
    hashed = design["proposal_index"].astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    hashed ^= hashed >> np.uint64(31)
    np.save(root / "design/validation_split.npy", (hashed % np.uint64(10) == 0).astype(np.uint8))
    np.savetxt(root / "design/parameters.csv", theta, delimiter=",", header=",".join(prior.names),
               comments="", fmt="%.17g")
    # Identical support and projector: retain the same four regression controls
    # and two challenging validated profiles for a fresh operator preflight.
    cases = json.loads((old / "preflight/cases.json").read_text())
    for case in cases:
        if "design_row" in case:
            case["previous_design_row"] = case.pop("design_row")
            case["role"] = "previous_design_boundary_regression"
    assert all(prior.contains(case["theta"]) for case in cases)
    write_json(root / "preflight/cases.json", cases)
    dependencies = list((campaign / "code").rglob("*.jl"))
    dependencies += list((campaign / "runtime/XGPaint/src").glob("*.jl"))
    dependencies += [campaign / "cache/complete.toml", campaign / "preflight/metadata_manifest.json"]
    dependencies += list((root / "inputs").iterdir())
    manifest = dict(schema_version=2, prior=prior.config, accepted_count=args.count,
                    parameter_order=prior.names,
                    metrics={k: [float(v.min()), float(v.max())] for k,v in metrics.items()},
                    design_sha256={p.name: digest(p) for p in (root / "design").iterdir()},
                    code_sha256={p.name: digest(p) for p in (root / "code").iterdir() if p.is_file()},
                    run_config_sha256=digest(root / "run_config.json"),
                    preflight_cases_sha256=digest(root / "preflight/cases.json"),
                    dependency_sha256={str(p): digest(p) for p in dependencies},
                    reuse_design=dict(thinning_probability="P0*xc/(60*4)", salt=THINNING_SALT,
                                      old_slot_stride=OLD_SLOT_STRIDE,
                                      selected_from_complete_frozen_design=True,
                                      old_source_points=int(len(design["old_slots"])),
                                      fresh_points=int(len(design["fresh_slots"])),
                                      completed_state_not_used_for_sampling=True),
                    same_broad_support_as_previous=True, same_projector_as_previous=True)
    write_json(root / "manifest.json", manifest)
    write_json(root / "audit/thinning_design.json", dict(
        old_selected_rows=design["old_selected_rows"].tolist(), old_slots=design["old_slots"].tolist(),
        probability_formula="theta[:,0]*theta[:,1]/240", salt=THINNING_SALT,
        explanation="Old density proportional to 1/(P0*xc), so this independent thinning gives constant density on the unchanged support; the full frozen design is considered regardless of completed jobs."))
    print(json.dumps(dict(root=str(root), rows=len(theta), reuse_design=manifest["reuse_design"]), indent=2))


if __name__ == "__main__":
    main()
