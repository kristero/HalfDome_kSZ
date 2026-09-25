"""Freeze a uniform design, its validation cases and every worker input."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from noise_seeds import split_seeds
from prior import B12, UniformPrior, digest, write_json


def preflight_cases(prior, run, old):
    cases = []

    def add(name, theta, role, reference=None):
        theta = np.asarray(theta, dtype=float)
        assert prior.contains(theta), name
        for previous in cases:
            if np.array_equal(previous["theta"], theta):
                previous.setdefault("additional_roles", []).append(role)
                return
        cases.append(dict(name=name, theta=theta.tolist(), role=role, reference=reference))

    add("Battaglia12", B12, "historical_clean_regression",
        dict(kind="directory", path=str(Path(run["campaign"]) / "results/HalfDome")))
    refit = Path(run["cosmology_refit"])
    for variant in ("L1_m9", "fgas-8sigma", "Mstar-1sigma"):
        fit = json.loads((refit / "results" / (variant + "_fit.json")).read_text())["best"]
        add(variant, fit["theta"], "cosmology_fit_clean_regression",
            dict(kind="npz", path=str(refit / "results" / (fit["name"] + "_full_cl.npz"))))
    corners = prior.corners()
    metrics = prior.metrics(corners)
    for name, metric, maximize in (("shallow_boundary", "beta_min", False),
                                   ("steep_boundary", "beta_max", True),
                                   ("compact_boundary", "size_min", False),
                                   ("diffuse_boundary", "tail", True),
                                   ("bright_boundary", "Y200_max", True),
                                   ("faint_boundary", "Y200_min", False)):
        index = np.argmax(metrics[metric]) if maximize else np.argmin(metrics[metric])
        add(name, corners[index], "uniform_box_corner:" + metric)
    low, high = np.array(old["prior_low"]), np.array(old["prior_high"])
    unit = prior.to_unit(prior.corners())
    old_corners = low + unit * (high - low)
    old_metrics = prior.metrics(old_corners)
    for name, metric in (("old_compact_corner", "size_min"), ("old_shallow_corner", "beta_min")):
        add(name, old_corners[np.argmin(old_metrics[metric])], "old_box_corner:" + metric)
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    args = parser.parse_args()
    if args.count <= 0 or args.count > 524288:
        parser.error("count must be in 1..524288")
    source, root = Path(__file__).resolve().parent, args.root.resolve()
    if (root / "manifest.json").exists():
        raise SystemExit("Already frozen; choose a new root or resume its workers")
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
    campaign, refit = Path(run["campaign"]), Path(run["cosmology_refit"])
    source_records = {
        "old_metadata.json": campaign / "preflight/metadata_manifest.json",
        "old_comparison.json": campaign / "comparison/comparison_summary.json",
        "flamingo_fits.json": refit / "results/comparison_summary.json"}
    for name, path in source_records.items():
        shutil.copy2(path, root / "inputs" / name)
    prior = UniformPrior()
    old = json.loads((root / "inputs/old_metadata.json").read_text())["verified_bundle"]
    fits = json.loads((root / "inputs/flamingo_fits.json").read_text())
    assert old["param_names"] == fits["parameter_order"] == prior.names
    assert np.all(prior.low <= old["prior_low"]) and np.all(prior.high >= old["prior_high"])
    assert all(prior.contains(t) for t in fits["corrected_parameters"].values())
    theta, proposal_index = prior.sobol(args.count)
    np.save(root / "design/theta.npy", theta)
    np.save(root / "design/proposal_index.npy", proposal_index)
    np.save(root / "design/noise_split_seeds.npy",
            np.array([split_seeds(i, run["noise_master_seed"]) for i in range(args.count)], dtype=np.int64))
    # Stable hash split, independent of count; do not select on raw Sobol index modulo 10.
    hashed = proposal_index.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    hashed ^= hashed >> np.uint64(31)
    np.save(root / "design/validation_split.npy", (hashed % np.uint64(10) == 0).astype(np.uint8))
    np.savetxt(root / "design/parameters.csv", theta, delimiter=",",
               header=",".join(prior.names), comments="", fmt="%.17g")
    cases = preflight_cases(prior, run, old)
    write_json(root / "preflight/cases.json", cases)
    dependencies = list((campaign / "code").rglob("*.jl"))
    dependencies += list((campaign / "runtime/XGPaint/src").glob("*.jl"))
    dependencies += [campaign / "cache/complete.toml"]
    for case in cases:
        reference = case["reference"]
        if reference:
            path = Path(reference["path"])
            dependencies.extend([path] if path.is_file() else [path / (key + ".npy") for key in
                                ("masked_clean_cl", "unmasked_clean_cl")])
    dependencies += list(source_records.values())
    dependencies += list((root / "inputs").iterdir())
    manifest = dict(schema_version=2, prior=prior.config, accepted_count=args.count,
                    last_proposal_index=int(proposal_index[-1]), accepted_fraction_prefix=1.0,
                    parameter_order=prior.names, box_certificate=prior.certify_box(),
                    design_sha256={p.name: digest(p) for p in (root / "design").iterdir()},
                    code_sha256={p.name: digest(p) for p in (root / "code").iterdir() if p.is_file()},
                    run_config_sha256=digest(root / "run_config.json"),
                    preflight_cases_sha256=digest(root / "preflight/cases.json"),
                    dependency_sha256={str(p): digest(p) for p in dependencies},
                    sampling="Unconditioned scrambled Sobol affine design; independent linear uniform target",
                    original_map_definition_preserved=True)
    write_json(root / "manifest.json", manifest)
    print(json.dumps(dict(root=str(root), rows=len(theta), preflight_cases=len(cases),
                          box_certificate=manifest["box_certificate"]), indent=2))


if __name__ == "__main__":
    main()
