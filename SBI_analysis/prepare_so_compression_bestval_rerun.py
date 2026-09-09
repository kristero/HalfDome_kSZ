#!/usr/bin/env python3
"""Copy fixed compression inputs and a completed bins40 control to a fresh run."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from so_sbi_compression import METHODS, write_json


def prepare_rerun(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and rerun must be separate, non-nested directories")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing rerun: {target}")
    config = json.loads((source / "experiment.json").read_text())
    experiment_id = config["experiment_id"]
    for name in ("training_complete.json", "evaluation/evaluation_complete.json"):
        status = json.loads((source / "bins40" / name).read_text())
        if status["experiment_id"] != experiment_id:
            raise ValueError(f"Stale bins40 completion record: {name}")
    training = json.loads((source / "bins40/training_complete.json").read_text())
    if not (training.get("converged_by_early_stopping")
            or training.get("weights_selection") == "best_validation_snapshot"):
        raise ValueError("The reused bins40 control must have selected best-validation weights")
    with np.load(source / "shared.npz", allow_pickle=False) as shared:
        indices = shared["test_indices"]
    missing = [int(i) for i in indices if not (source / f"bins40/evaluation/profiles/row{i}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing baseline posterior checkpoints: {missing[:10]}")
    names = ["experiment.json", "shared.npz", "pca_diagnostics.npz", "moped_diagnostics.npz"]
    names += [f"{method}_{suffix}" for method in METHODS for suffix in ("transform.npz", "x.npy")]
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    target.mkdir(parents=True)
    for name in names:
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / "bins40", target / "bins40")
    # Keep the original experiment ID: data, splits, compression and hyperparameters
    # are identical. Track the changed weight-selection protocol separately.
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()
    write_json(target / "rerun_provenance.json", dict(
        source_root=str(source), experiment_id=experiment_id, source_commit=commit,
        reused_methods=["bins40"], retrained_methods=["pca", "moped"],
        weights_selection="best_validation_snapshot", prepared_inputs_copied_unchanged=True,
    ))
    print(f"Prepared fresh PCA/MOPED rerun: {target}")
    print("Reused complete bins40 control; no old PCA/MOPED posteriors copied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    prepare_rerun(args.source_root, args.output_root)
