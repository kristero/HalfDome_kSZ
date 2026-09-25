#!/usr/bin/env python3
"""Export a small, checksummed MOPED inference bundle in the training environment."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from so_sbi_compression import PARAM_NAMES, project, save_npz, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.root / "experiment.json").read_text())
    training = json.loads((args.root / "moped/training_complete.json").read_text())
    if training["experiment_id"] != config["experiment_id"]:
        raise ValueError("Model and transform belong to different experiments")
    if training.get("weights_selection") != "best_validation_snapshot":
        raise ValueError("Use the corrected best-validation MOPED model")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh bundle directory: {args.output}")
    with np.load(args.root / "shared.npz", allow_pickle=False) as z:
        low, high, theta = z["low"], z["high"], z["theta"]
        names, fit, ids = z["param_names"], z["fit_indices"], z["sobol_global_row"]
    with np.load(args.root / "moped_transform.npz", allow_pickle=False) as z:
        transform = dict(z)
    with np.load(args.dataset or config["dataset"], allow_pickle=False) as z:
        if not np.array_equal(z["theta"], theta) or not np.array_equal(z["sobol_global_row"], ids):
            raise ValueError("Prepared dataset differs from the fitted compression")
        for key, expected in (("prior_low", low), ("prior_high", high), ("param_names", names)):
            if not np.array_equal(z[key], expected):
                raise ValueError(f"Prepared dataset differs: {key}")
        metadata = json.loads(str(z["metadata_json"].item()))
        if metadata["product"] != "masked_baseline_noise_cross_deproj0":
            raise ValueError("Wrong SO observable")
        x = z["x"]
        contract = {key: z[key] for key in ("ell_unbinned", "ell_binned", "bin_ell_min",
                     "bin_ell_max", "metadata_json", "product")}
    if tuple(str(n) for n in names) != PARAM_NAMES:
        raise ValueError("Unexpected parameter order")
    reference = np.random.default_rng(723).choice(fit, min(512, len(fit)), replace=False)
    context = project(x[reference], transform)
    saved_x = np.load(args.root / "moped_x.npy", mmap_mode="r")
    np.testing.assert_array_equal(context, saved_x[reference])
    import torch
    torch.set_num_threads(1)
    with (args.root / "moped/density_estimator.pkl").open("rb") as stream:
        model = pickle.load(stream)
    model = model.cpu().eval()
    # Fixed evaluations test local deserialization, internal theta scaling and context use.
    with torch.no_grad():
        log_prob = model.log_prob(torch.tensor(theta[reference[:32]], dtype=torch.float32),
                                  context=torch.tensor(context[:32])).cpu().numpy()
    if not np.isfinite(log_prob).all():
        raise ValueError("Nonfinite reference log probabilities")
    args.output.mkdir(parents=True)
    save_npz(args.output / "observation_contract.npz", **contract, low=low, high=high,
             param_names=names, reference_indices=reference, reference_theta=theta[reference],
             reference_x=x[reference], reference_context=context, reference_log_prob=log_prob,
             experiment_id=np.asarray(config["experiment_id"]))
    sources = {"density_estimator.pkl": args.root / "moped/density_estimator.pkl",
               "moped_transform.npz": args.root / "moped_transform.npz",
               "training_complete.json": args.root / "moped/training_complete.json",
               "experiment.json": args.root / "experiment.json"}
    for name, source in sources.items():
        shutil.copy2(source, args.output / name)
    hashes = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "bundle_manifest.json", dict(experiment_id=config["experiment_id"],
        source_root=str(args.root), files=hashes,
        note="Training-row references are sanity controls, not held-out accuracy tests."))
    print("Exported:", args.output, flush=True)


if __name__ == "__main__":
    main()
