"""Train and select noiseless tSZ emulators without reading test targets.

Example: python train.py --output runs/main --device cuda
Only validation loss chooses the architecture/checkpoint/ensemble. Evaluation
of the untouched test set is a separate, subsequent evaluate.py command.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from model import PARAMETERS, SpectrumNetwork, features

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DATA = REPO / "SBI_analysis/tsz64k_sbi_local/data/rows_65536"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source(folder):
    """Verify full design and both independently checksummed source pieces."""
    meta = json.loads((folder / "meta.json").read_text())
    theta = np.load(folder / "theta.npy")
    spectra = np.load(folder / "clean_dl.npy")
    rows = np.load(folder / "row_id.npy")
    assert tuple(meta["parameter_order"]) == PARAMETERS
    assert theta.shape == (65536, 9) and spectra.shape == (65536, 40)
    assert np.array_equal(rows, np.arange(65536))
    assert len(np.unique(theta, axis=0)) == len(theta)
    assert np.isfinite(spectra).all() and np.all(spectra > 0)
    assert np.all(theta >= meta["lower"]) and np.all(theta <= meta["upper"])
    design = REPO / "tSZ_64k_flat_prior_SO_baseline_deproj0/design"
    for filename in ("theta_design_65536.npy", "noise_seeds_65536.npy"):
        assert sha256(design / filename) == meta["design_sha256"][filename]
    assert np.array_equal(theta, np.load(design / "theta_design_65536.npy"))
    assert np.array_equal(np.load(folder / "noise_seeds.npy"), np.load(design / "noise_seeds_65536.npy"))
    source_parts = []
    covered = np.zeros(len(theta), dtype=bool)
    for name in ("rows_64440", "rows_validation_1096"):
        part = folder.parent / name
        for line in (part / "SHA256SUMS").read_text().splitlines():
            expected, filename = line.split(maxsplit=1)
            # The required local arrays have independent archived checksums.
            if filename in ("theta.npy", "clean_dl.npy", "row_id.npy", "meta.json"):
                assert sha256(part / filename) == expected, filename
        indices = np.load(part / "row_id.npy")
        assert not covered[indices].any()
        covered[indices] = True
        assert np.array_equal(theta[indices], np.load(part / "theta.npy"))
        reference = np.load(part / "clean_dl.npy")
        # Independently collected bins can differ at floating-point summation precision.
        np.testing.assert_allclose(spectra[indices], reference, rtol=1e-11, atol=0)
        source_parts.append({"folder": str(part), "rows": len(indices),
            "max_relative_binning_difference": float(np.max(np.abs(spectra[indices]/reference-1)))})
    assert covered.all()
    hashes = {name: sha256(folder / name) for name in
              ("theta.npy", "clean_dl.npy", "row_id.npy", "meta.json", "noise_seeds.npy")}
    return theta, spectra, rows, meta, {"sha256": hashes, "parts": source_parts,
            "shape": list(spectra.shape), "min_dl": float(spectra.min()),
            "max_dl": float(spectra.max()), "all_checks_passed": True}


def predict_scaled(model, values, batch_size=4096):
    with torch.inference_mode():
        return torch.cat([model(values[i:i+batch_size])
                          for i in range(0, len(values), batch_size)])


def train_candidate(args, width, index, arrays, output, metadata):
    xtrain, ytrain, xval, yval = arrays
    seed = args.seed + 100 + index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SpectrumNetwork(width, args.blocks).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-6)
    target_scale = torch.as_tensor(metadata["target_std"], dtype=torch.float32,
                                   device=args.device)
    label = f"width{width}_seed{seed}"
    checkpoint = output / f"{label}.pt"
    best, best_epoch = float("inf"), 0
    start = time.perf_counter()
    history = []
    with open(output / f"{label}_history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_scaled_mse",
            "validation_scaled_mse", "validation_log10_rmse", "learning_rate", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            model.train()
            permutation = torch.randperm(len(xtrain), device=args.device)
            total = torch.zeros((), device=args.device)
            lr = optimizer.param_groups[0]["lr"]
            for batch in permutation.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                residual = model(xtrain[batch]) - ytrain[batch]
                loss = residual.square().mean()
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(batch)
            scheduler.step()
            model.eval()
            with torch.inference_mode():
                residual = predict_scaled(model, xval) - yval
                val_mse = residual.square().mean().item()
                val_log_mse = (residual * target_scale).square().mean().item()
            record = dict(epoch=epoch, train_scaled_mse=total.item()/len(xtrain),
                validation_scaled_mse=val_mse, validation_log10_rmse=np.sqrt(val_log_mse),
                learning_rate=lr, seconds=time.perf_counter()-start)
            writer.writerow(record)
            handle.flush()
            history.append(record)
            if val_log_mse < best:
                best, best_epoch = val_log_mse, epoch
                member = {"config": {"width": width, "blocks": args.blocks},
                          "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "label": label, "seed": seed, "best_epoch": epoch}
                torch.save(member, checkpoint)
            if epoch == 1 or epoch % 25 == 0:
                message = {"candidate": label, **record, "best_epoch": best_epoch,
                           "best_validation_log10_rmse": np.sqrt(best)}
                print(json.dumps(message), flush=True)
                write_json(output / "progress.json", message)
            if epoch >= 300 and epoch-best_epoch >= args.patience:
                break
    member = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(member["state_dict"])
    prediction = predict_scaled(model, xval).detach().cpu().numpy().astype(float)
    summary = {"label": label, "best_epoch": best_epoch, "epochs_run": epoch,
               "validation_log10_rmse": float(np.sqrt(best)),
               "seconds": time.perf_counter()-start}
    del model, optimizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return member, prediction, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=HERE / "runs/main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--widths", type=int, nargs="+", default=[256, 384])
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "split_indices.npz").exists():
        raise FileExistsError("Use a fresh output directory to preserve completed work")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    theta, spectra, rows, meta, provenance = audit_source(args.dataset)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(theta))
    ntest, nval = round(.15*len(theta)), round(.10*len(theta))
    test, val, train = perm[:ntest], perm[ntest:ntest+nval], perm[ntest+nval:]
    np.savez(args.output / "split_indices.npz", train=train, validation=val, test=test,
             train_row_id=rows[train], validation_row_id=rows[val], test_row_id=rows[test])
    # No test targets enter feature scaling, target scaling, optimization, or selection.
    shape_train = np.log10(spectra[train]) - 2*np.log10(theta[train, 0, None])
    xtrain = features(theta[train])
    metadata = {"version": 1, "parameter_order": list(PARAMETERS),
        "lower": np.array(meta["lower"]), "upper": np.array(meta["upper"]),
        "feature_mean": xtrain.mean(0), "feature_std": xtrain.std(0),
        "target_mean": shape_train.mean(0), "target_std": shape_train.std(0),
        "target": "masked clean 40-bin D_ell of Compton-y; exact P0 squared scaling",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": provenance["sha256"],
        "mask_sha256": meta["mask_sha256"], "split_seed": args.seed}
    def tensors(indices):
        x = (features(theta[indices])-metadata["feature_mean"])/metadata["feature_std"]
        y = (np.log10(spectra[indices])-2*np.log10(theta[indices, 0, None])-metadata["target_mean"])/metadata["target_std"]
        return (torch.tensor(x, dtype=torch.float32, device=args.device),
                torch.tensor(y, dtype=torch.float32, device=args.device))
    arrays = (*tensors(train), *tensors(val))
    write_json(args.output / "provenance.json", provenance)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(train_rows=len(train), validation_rows=len(val), test_rows=len(test),
                  python=platform.python_version(), torch=torch.__version__,
                  device_name=torch.cuda.get_device_name() if args.device.startswith("cuda") else platform.processor(),
                  test_usage="Only evaluate.py after selection; never used to tune models")
    write_json(args.output / "run_config.json", config)
    print(json.dumps(config), flush=True)
    started = time.perf_counter()
    members, predictions, summaries = [], [], []
    for i, width in enumerate(args.widths):
        member, pred, summary = train_candidate(args, width, i, arrays, args.output, metadata)
        members.append(member); predictions.append(pred); summaries.append(summary)
    truth_val = arrays[3].cpu().numpy().astype(float)
    choices = {s["label"]: [i] for i, s in enumerate(summaries)}
    if len(members) > 1:
        choices["log_space_ensemble"] = list(range(len(members)))
    scores = {}
    for name, indices in choices.items():
        residual = (np.mean([predictions[i] for i in indices], axis=0)-truth_val)*metadata["target_std"]
        scores[name] = float(np.sqrt(np.mean(residual**2)))
    selected = min(scores, key=scores.get)
    artifact = {**metadata, "members": [members[i] for i in choices[selected]], "selection": selected}
    torch.save(artifact, args.output / "emulator.pt")
    write_json(args.output / "selection.json", {"selected": selected, "validation_log10_rmse": scores,
        "candidates": summaries, "selection_rule": "Lowest validation log10 RMSE; no test information"})
    write_json(args.output / "TRAINING_COMPLETE.json", {"status": "complete", "test_evaluated": False,
        "seconds": time.perf_counter()-started, "model_sha256": sha256(args.output / "emulator.pt"),
        "peak_gpu_memory_GiB": torch.cuda.max_memory_allocated()/1024**3 if args.device.startswith("cuda") else None})
    print("TRAINING_COMPLETE " + selected, flush=True)


if __name__ == "__main__":
    main()
