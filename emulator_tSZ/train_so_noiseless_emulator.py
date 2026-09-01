#!/usr/bin/env python3
"""Train and test a deterministic emulator for noiseless binned SO profiles."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from so_noiseless_emulator import (
    PARAM_NAMES,
    SOProfileEmulator,
    load_emulator,
    predict_profiles,
)


DEFAULT_DATASET = Path(
    "/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/"
    "adrian_so_sbi_cases_ell80_7979_dataset_row_metadata_verified/"
    "so_masked_no_noise_ell80_7979_sbi_run.npz"
)
DEFAULT_OUTPUT = Path("/lustre/work/kristero10/so_noiseless_emulator_512k")
VERIFIED_THETA_SOURCES = {
    "metadata_npz_theta",
    "metadata_csv_theta_columns",
    "metadata_csv_sobol_global_row_plus_sobol_csv",
    "sobol_global_row_mapping_plus_sobol_csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual MLP mapping the 9 Battaglia pressure parameters to "
            "the 40 noiseless, binned SO D_ell values. The held-out test set is "
            "read only after model selection and final refitting."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("SO_EMULATOR_DATASET", DEFAULT_DATASET)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SO_EMULATOR_OUTPUT_DIR", DEFAULT_OUTPUT)),
    )
    parser.add_argument("--target-key", default="x")
    parser.add_argument("--expected-rows", type=int, default=524_288)
    parser.add_argument("--expected-bins", type=int, default=40)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--selection-val-fraction",
        type=float,
        default=0.05,
        help="Fraction of the outer 85%% training partition used only to choose the epoch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug-only row limit.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dataset-hash", action="store_true")
    parser.add_argument("--allow-unverified-noiseless", action="store_true")
    parser.add_argument("--require-verified-order", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-save-test-predictions", action="store_true")
    parser.add_argument("--fail-on-quality-gate", action="store_true")
    parser.add_argument("--gate-median-ape-pct", type=float, default=1.0)
    parser.add_argument("--gate-p95-ape-pct", type=float, default=5.0)
    parser.add_argument("--gate-max-bin-bias-pct", type=float, default=1.0)
    parser.add_argument("--gate-worst-bin-p95-ape-pct", type=float, default=7.5)
    parser.add_argument("--gate-min-bin-r2", type=float, default=0.99)
    parser.add_argument("--gate-worst-quartile-p95-ape-pct", type=float, default=7.5)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        return ""
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = {"theta", args.target_key, "prior_low", "prior_high"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays {missing}; keys={data.files}")

        theta = np.asarray(data["theta"], dtype=np.float32)
        target = np.asarray(data[args.target_key], dtype=np.float32)
        ell_key = "ell_binned" if "ell_binned" in data.files else "ell"
        if ell_key not in data.files:
            raise KeyError(f"{path} needs ell_binned or ell")
        ell = np.asarray(data[ell_key], dtype=np.float32).reshape(-1)
        prior_low = np.asarray(data["prior_low"], dtype=np.float32).reshape(-1)
        prior_high = np.asarray(data["prior_high"], dtype=np.float32).reshape(-1)
        names_key = "theta_columns" if "theta_columns" in data.files else "param_names"
        if names_key not in data.files:
            raise KeyError(f"{path} needs theta_columns or param_names")
        theta_columns = tuple(str(item) for item in np.asarray(data[names_key]).reshape(-1))

        bin_ell_min = (
            np.asarray(data["bin_ell_min"], dtype=np.float32).reshape(-1)
            if "bin_ell_min" in data.files
            else np.full(ell.shape, np.nan, dtype=np.float32)
        )
        bin_ell_max = (
            np.asarray(data["bin_ell_max"], dtype=np.float32).reshape(-1)
            if "bin_ell_max" in data.files
            else np.full(ell.shape, np.nan, dtype=np.float32)
        )
        labels = {
            key: scalar_text(data[key])
            for key in ("case_name", "source_case", "product")
            if key in data.files
        }
        theta_source = scalar_text(data["theta_source"]) if "theta_source" in data.files else ""
        source_cl_path = (
            scalar_text(data["source_cl_path"]) if "source_cl_path" in data.files else ""
        )
        sobol_global_row = (
            np.asarray(data["sobol_global_row"], dtype=np.int64).reshape(-1)
            if "sobol_global_row" in data.files
            else np.empty(0, dtype=np.int64)
        )
        source_metadata = {}
        if "metadata_json" in data.files:
            raw_metadata = scalar_text(data["metadata_json"])
            if raw_metadata:
                try:
                    source_metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    source_metadata = {"unparsed_metadata_json": raw_metadata}
        dataset_test_indices = (
            np.asarray(data["test_indices"], dtype=np.int64).reshape(-1)
            if "test_indices" in data.files
            else np.empty(0, dtype=np.int64)
        )

    if theta.ndim != 2 or theta.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"theta must have shape (N, 9), got {theta.shape}")
    if target.ndim != 2 or target.shape[0] != theta.shape[0]:
        raise ValueError(
            f"{args.target_key} must be 2D with {theta.shape[0]} rows, got {target.shape}"
        )
    if theta_columns != PARAM_NAMES:
        raise ValueError(
            f"Parameter order must be {list(PARAM_NAMES)}, got {list(theta_columns)}"
        )
    if target.shape[1] != ell.size:
        raise ValueError(f"Target columns {target.shape[1]} do not match ell size {ell.size}")
    if bin_ell_min.size != ell.size or bin_ell_max.size != ell.size:
        raise ValueError("bin_ell_min/bin_ell_max must match ell when present")
    if args.expected_bins > 0 and target.shape[1] != args.expected_bins:
        raise ValueError(
            f"Expected {args.expected_bins} output bins, found {target.shape[1]}"
        )

    full_rows = int(theta.shape[0])
    if args.expected_rows > 0 and full_rows != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {full_rows}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{args.target_key} contains non-finite values")
    if np.any(target <= 0.0):
        count = int(np.count_nonzero(target <= 0.0))
        raise ValueError(
            f"log10 target requires strictly positive D_ell; found {count} non-positive values"
        )
    if prior_low.shape != (len(PARAM_NAMES),) or prior_high.shape != (len(PARAM_NAMES),):
        raise ValueError("prior_low and prior_high must each have length 9")
    if np.any(~np.isfinite(prior_low)) or np.any(~np.isfinite(prior_high)):
        raise ValueError("Prior bounds contain non-finite values")
    if np.any(prior_high <= prior_low):
        raise ValueError("Every prior_high value must exceed prior_low")

    label_values = [str(path.name), source_cl_path, *labels.values()]
    verified_noiseless = any(
        "no_noise" in value.lower() and "noise_cross" not in value.lower()
        for value in label_values
    )
    if not verified_noiseless and not args.allow_unverified_noiseless:
        raise ValueError(
            "Could not verify a noiseless product from the dataset filename or metadata. "
            "Use the masked_no_noise case, or pass --allow-unverified-noiseless after checking it."
        )
    if args.require_verified_order and theta_source not in VERIFIED_THETA_SOURCES:
        raise ValueError(
            f"theta_source={theta_source!r} is not a verified ordering source. "
            f"Accepted values are {sorted(VERIFIED_THETA_SOURCES)}"
        )
    if sobol_global_row.size:
        if sobol_global_row.size != full_rows:
            raise ValueError(
                f"sobol_global_row has {sobol_global_row.size} rows, expected {full_rows}"
            )
        if np.unique(sobol_global_row).size != full_rows:
            raise ValueError("sobol_global_row contains duplicate mappings")

    tolerance = 1.0e-5 * np.maximum(1.0, np.abs(prior_high - prior_low))
    outside = (theta < prior_low - tolerance) | (theta > prior_high + tolerance)
    if np.any(outside):
        row, column = np.argwhere(outside)[0]
        raise ValueError(
            f"theta row {row} is outside the saved prior for {PARAM_NAMES[column]}"
        )

    used_rows = full_rows
    if args.max_rows > 0:
        if args.max_rows < 100:
            raise ValueError("--max-rows must be zero or at least 100")
        used_rows = min(int(args.max_rows), full_rows)
        theta = np.ascontiguousarray(theta[:used_rows])
        target = np.ascontiguousarray(target[:used_rows])
        sobol_global_row = np.ascontiguousarray(sobol_global_row[:used_rows])

    contract = {
        "dataset_path": str(path),
        "dataset_size_bytes": int(path.stat().st_size),
        "available_rows": full_rows,
        "used_rows": used_rows,
        "theta_shape": list(theta.shape),
        "target_key": args.target_key,
        "target_shape": list(target.shape),
        "target_representation": "linear_binned_D_ell",
        "target_transform_for_training": "log10_then_per_bin_standardization",
        "theta_columns": list(theta_columns),
        "ell": ell,
        "bin_ell_min": bin_ell_min,
        "bin_ell_max": bin_ell_max,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "labels": labels,
        "source_cl_path": source_cl_path,
        "theta_source": theta_source,
        "verified_noiseless": verified_noiseless,
        "sobol_global_row_present": bool(sobol_global_row.size),
        "source_metadata": source_metadata,
        "dataset_test_indices_count": int(dataset_test_indices.size),
        "dataset_test_indices_usage": "ignored; a new seeded random 85/15 split is used",
    }
    return {
        "theta": theta,
        "target": target,
        "ell": ell,
        "bin_ell_min": bin_ell_min,
        "bin_ell_max": bin_ell_max,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "sobol_global_row": sobol_global_row,
        "contract": contract,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5")
    if not 0.0 < args.selection_val_fraction < 0.5:
        raise ValueError("--selection-val-fraction must be between 0 and 0.5")
    for name in ("threads", "batch_size", "eval_batch_size", "hidden_width"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("residual_blocks", "max_epochs", "patience"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if args.min_delta < 0.0:
        raise ValueError("--min-delta must be non-negative")


def configure_runtime(seed: int, threads: int, device_name: str) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.manual_seed_all(seed)
    return device


def make_splits(
    n_rows: int,
    test_fraction: float,
    selection_val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_rows).astype(np.int64)
    n_test = int(math.ceil(n_rows * test_fraction))
    test_idx = np.sort(permutation[:n_test])
    train_idx = permutation[n_test:]
    n_selection_val = int(math.ceil(train_idx.size * selection_val_fraction))
    selection_val_idx = np.sort(train_idx[:n_selection_val])
    selection_fit_idx = np.sort(train_idx[n_selection_val:])
    train_idx = np.sort(train_idx)

    covered = np.concatenate([train_idx, test_idx])
    if np.unique(covered).size != n_rows or covered.size != n_rows:
        raise RuntimeError("Outer train/test split does not cover each row exactly once")
    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError("Outer training and test indices overlap")
    if np.intersect1d(selection_fit_idx, selection_val_idx).size:
        raise RuntimeError("Internal model-selection fit and validation indices overlap")
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "selection_fit_idx": selection_fit_idx,
        "selection_val_idx": selection_val_idx,
    }


def fit_standardizers(
    theta: np.ndarray,
    log10_target: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    theta_rows = np.asarray(theta[indices], dtype=np.float64)
    target_rows = np.asarray(log10_target[indices], dtype=np.float64)
    theta_mean = theta_rows.mean(axis=0)
    theta_std = theta_rows.std(axis=0)
    target_mean = target_rows.mean(axis=0)
    target_std = target_rows.std(axis=0)
    if np.any(theta_std <= 1.0e-12):
        raise ValueError("At least one theta column is constant in the fitting subset")
    if np.any(target_std <= 1.0e-12):
        raise ValueError("At least one log10(D_ell) bin is constant in the fitting subset")
    return {
        "theta_mean": theta_mean.astype(np.float32),
        "theta_std": theta_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }


def transform_arrays(
    theta: np.ndarray,
    log10_target: np.ndarray,
    standardizers: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_scaled = np.ascontiguousarray(
        (theta - standardizers["theta_mean"]) / standardizers["theta_std"],
        dtype=np.float32,
    )
    target_scaled = np.ascontiguousarray(
        (log10_target - standardizers["target_mean"])
        / standardizers["target_std"],
        dtype=np.float32,
    )
    return torch.from_numpy(theta_scaled), torch.from_numpy(target_scaled)


def make_model(args: argparse.Namespace, output_dim: int) -> SOProfileEmulator:
    return SOProfileEmulator(
        input_dim=len(PARAM_NAMES),
        output_dim=output_dim,
        hidden_width=int(args.hidden_width),
        residual_blocks=int(args.residual_blocks),
    )


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    theta_all: torch.Tensor,
    target_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    index_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    order = torch.randperm(index_tensor.numel(), generator=generator)
    total_loss = 0.0
    total_rows = 0
    for start in range(0, order.numel(), batch_size):
        batch_indices = index_tensor[order[start : start + batch_size]]
        theta_batch = theta_all.index_select(0, batch_indices).to(device)
        target_batch = target_all.index_select(0, batch_indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(theta_batch)
        loss = torch.mean((prediction - target_batch) ** 2)
        loss.backward()
        optimizer.step()
        rows = int(batch_indices.numel())
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    return total_loss / total_rows


def evaluate_scaled_mse(
    model: nn.Module,
    theta_all: torch.Tensor,
    target_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    index_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for start in range(0, index_tensor.numel(), batch_size):
            batch_indices = index_tensor[start : start + batch_size]
            theta_batch = theta_all.index_select(0, batch_indices).to(device)
            target_batch = target_all.index_select(0, batch_indices).to(device)
            loss = torch.mean((model(theta_batch) - target_batch) ** 2)
            rows = int(batch_indices.numel())
            total_loss += float(loss.cpu()) * rows
            total_rows += rows
    return total_loss / total_rows


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def select_training_epoch(
    args: argparse.Namespace,
    theta_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    fit_idx: np.ndarray,
    validation_idx: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    model = make_model(args, target_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.max_epochs),
        eta_min=float(args.learning_rate) * 0.02,
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(args.seed) + 101)

    best_epoch = 0
    best_validation = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "selection_best_checkpoint.pt"

    for epoch in range(1, int(args.max_epochs) + 1):
        started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_mse = train_epoch(
            model,
            optimizer,
            theta_scaled,
            target_scaled,
            fit_idx,
            int(args.batch_size),
            device,
            shuffle_generator,
        )
        validation_mse = evaluate_scaled_mse(
            model,
            theta_scaled,
            target_scaled,
            validation_idx,
            int(args.eval_batch_size),
            device,
        )
        scheduler.step()
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": float(epoch),
                "train_scaled_mse": float(train_mse),
                "validation_scaled_mse": float(validation_mse),
                "learning_rate": learning_rate,
                "seconds": float(elapsed),
            }
        )
        print(
            f"selection epoch {epoch:03d}: train_mse={train_mse:.7g} "
            f"val_mse={validation_mse:.7g} lr={learning_rate:.3g} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

        if validation_mse < best_validation - float(args.min_delta):
            best_validation = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "validation_scaled_mse": best_validation,
                    "model_state_dict": cpu_state_dict(model),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.patience):
                print(
                    f"Early stopping after {epoch} epochs; selected epoch {best_epoch}.",
                    flush=True,
                )
                break

    if best_epoch <= 0:
        raise RuntimeError("Model selection did not produce a finite validation loss")
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, history


def refit_on_outer_training_set(
    args: argparse.Namespace,
    theta_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    train_idx: np.ndarray,
    selected_epoch: int,
    device: torch.device,
) -> tuple[SOProfileEmulator, list[dict[str, float]]]:
    random.seed(int(args.seed) + 1)
    np.random.seed(int(args.seed) + 1)
    torch.manual_seed(int(args.seed) + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + 1)

    model = make_model(args, target_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.max_epochs),
        eta_min=float(args.learning_rate) * 0.02,
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(args.seed) + 202)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(selected_epoch) + 1):
        started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_mse = train_epoch(
            model,
            optimizer,
            theta_scaled,
            target_scaled,
            train_idx,
            int(args.batch_size),
            device,
            shuffle_generator,
        )
        scheduler.step()
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": float(epoch),
                "train_scaled_mse": float(train_mse),
                "learning_rate": learning_rate,
                "seconds": float(elapsed),
            }
        )
        print(
            f"final refit epoch {epoch:03d}/{selected_epoch:03d}: "
            f"train_mse={train_mse:.7g} lr={learning_rate:.3g} seconds={elapsed:.1f}",
            flush=True,
        )

    model.eval()
    return model, history


def predict_log10(
    model: nn.Module,
    theta_scaled: torch.Tensor,
    indices: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    index_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, index_tensor.numel(), batch_size):
            batch_indices = index_tensor[start : start + batch_size]
            theta_batch = theta_scaled.index_select(0, batch_indices).to(device)
            scaled = model(theta_batch).cpu().numpy()
            predictions.append(scaled * target_std + target_mean)
    result = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("The emulator produced non-finite log10(D_ell)")
    return result


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = math.sqrt(
        float(np.sum(x_centered**2)) * float(np.sum(y_centered**2))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def compute_test_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    ell: np.ndarray,
    bin_ell_min: np.ndarray,
    bin_ell_max: np.ndarray,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError(f"Metric inputs must be matching 2D arrays, got {truth.shape} and {prediction.shape}")
    if np.any(truth <= 0.0):
        raise ValueError("Percentage metrics require strictly positive truth values")
    if np.any(prediction <= 0.0) or not np.all(np.isfinite(prediction)):
        raise ValueError("Predictions must be finite and strictly positive")

    difference = prediction - truth
    signed_percentage = 100.0 * difference / truth
    absolute_percentage = np.abs(signed_percentage)
    smape = 200.0 * np.abs(difference) / (np.abs(truth) + np.abs(prediction))
    truth_log10 = np.log10(truth)
    prediction_log10 = np.log10(prediction)
    log_difference = prediction_log10 - truth_log10

    per_bin: list[dict[str, Any]] = []
    r2_values: list[float] = []
    correlation_values: list[float] = []
    for column in range(truth.shape[1]):
        truth_column = truth[:, column]
        prediction_column = prediction[:, column]
        residual_column = difference[:, column]
        denominator = float(np.sum((truth_column - np.mean(truth_column)) ** 2))
        r2 = (
            1.0 - float(np.sum(residual_column**2)) / denominator
            if denominator > 0.0
            else float("nan")
        )
        correlation = safe_pearson(truth_column, prediction_column)
        r2_values.append(r2)
        correlation_values.append(correlation)
        per_bin.append(
            {
                "bin_index": column,
                "ell": float(ell[column]),
                "ell_min": float(bin_ell_min[column]),
                "ell_max": float(bin_ell_max[column]),
                "truth_min": float(np.min(truth_column)),
                "truth_median": float(np.median(truth_column)),
                "truth_max": float(np.max(truth_column)),
                "rmse_linear_dl": float(np.sqrt(np.mean(residual_column**2))),
                "mae_linear_dl": float(np.mean(np.abs(residual_column))),
                "rmse_over_mean_truth_pct": float(
                    100.0 * np.sqrt(np.mean(residual_column**2)) / np.mean(truth_column)
                ),
                "mean_bias_pct": float(np.mean(signed_percentage[:, column])),
                "mape_pct": float(np.mean(absolute_percentage[:, column])),
                "median_ape_pct": float(np.median(absolute_percentage[:, column])),
                "p68_ape_pct": float(np.percentile(absolute_percentage[:, column], 68)),
                "p95_ape_pct": float(np.percentile(absolute_percentage[:, column], 95)),
                "p99_ape_pct": float(np.percentile(absolute_percentage[:, column], 99)),
                "max_ape_pct": float(np.max(absolute_percentage[:, column])),
                "smape_pct": float(np.mean(smape[:, column])),
                "rmse_log10_dl": float(np.sqrt(np.mean(log_difference[:, column] ** 2))),
                "r2": r2,
                "pearson_r": correlation,
            }
        )

    profile_metrics = {
        "mean_ape_pct": np.mean(absolute_percentage, axis=1),
        "median_ape_pct": np.median(absolute_percentage, axis=1),
        "p95_ape_pct": np.percentile(absolute_percentage, 95, axis=1),
        "max_ape_pct": np.max(absolute_percentage, axis=1),
        "mean_bias_pct": np.mean(signed_percentage, axis=1),
        "rmse_log10_dl": np.sqrt(np.mean(log_difference**2, axis=1)),
    }
    overall = {
        "n_test_profiles": int(truth.shape[0]),
        "n_bins": int(truth.shape[1]),
        "n_test_values": int(truth.size),
        "rmse_log10_dl": float(np.sqrt(np.mean(log_difference**2))),
        "mae_log10_dl": float(np.mean(np.abs(log_difference))),
        "max_abs_log10_dl": float(np.max(np.abs(log_difference))),
        "mean_bias_pct": float(np.mean(signed_percentage)),
        "mape_pct": float(np.mean(absolute_percentage)),
        "median_ape_pct": float(np.median(absolute_percentage)),
        "p68_ape_pct": float(np.percentile(absolute_percentage, 68)),
        "p95_ape_pct": float(np.percentile(absolute_percentage, 95)),
        "p99_ape_pct": float(np.percentile(absolute_percentage, 99)),
        "max_ape_pct": float(np.max(absolute_percentage)),
        "rms_percentage_error_pct": float(np.sqrt(np.mean(signed_percentage**2))),
        "smape_pct": float(np.mean(smape)),
        "fraction_abs_pct_below_1": float(np.mean(absolute_percentage < 1.0)),
        "fraction_abs_pct_below_2": float(np.mean(absolute_percentage < 2.0)),
        "fraction_abs_pct_below_5": float(np.mean(absolute_percentage < 5.0)),
        "fraction_abs_pct_below_10": float(np.mean(absolute_percentage < 10.0)),
        "median_bin_r2": float(np.nanmedian(r2_values)),
        "min_bin_r2": float(np.nanmin(r2_values)),
        "median_bin_pearson_r": float(np.nanmedian(correlation_values)),
        "min_bin_pearson_r": float(np.nanmin(correlation_values)),
        "max_abs_bin_bias_pct": float(
            np.max(np.abs([row["mean_bias_pct"] for row in per_bin]))
        ),
        "worst_bin_p95_ape_pct": float(
            np.max([row["p95_ape_pct"] for row in per_bin])
        ),
    }
    return overall, per_bin, profile_metrics, absolute_percentage, signed_percentage


def parameter_quartile_metrics(
    theta_test: np.ndarray,
    absolute_percentage: np.ndarray,
    log10_truth: np.ndarray,
    log10_prediction: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
) -> list[dict[str, Any]]:
    normalized = (theta_test - prior_low) / (prior_high - prior_low)
    rows: list[dict[str, Any]] = []
    edges = (0.0, 0.25, 0.5, 0.75, 1.0)
    for parameter_index, parameter_name in enumerate(PARAM_NAMES):
        for quartile in range(4):
            low = edges[quartile]
            high = edges[quartile + 1]
            if quartile == 3:
                mask = (normalized[:, parameter_index] >= low) & (
                    normalized[:, parameter_index] <= high + 1.0e-6
                )
            else:
                mask = (normalized[:, parameter_index] >= low) & (
                    normalized[:, parameter_index] < high
                )
            if not np.any(mask):
                raise RuntimeError(f"No test rows in prior quartile {quartile + 1} for {parameter_name}")
            subset_ape = absolute_percentage[mask]
            subset_log_residual = log10_prediction[mask] - log10_truth[mask]
            rows.append(
                {
                    "parameter": parameter_name,
                    "quartile": quartile + 1,
                    "prior_fraction_min": low,
                    "prior_fraction_max": high,
                    "n_profiles": int(np.count_nonzero(mask)),
                    "median_ape_pct": float(np.median(subset_ape)),
                    "mape_pct": float(np.mean(subset_ape)),
                    "p95_ape_pct": float(np.percentile(subset_ape, 95)),
                    "max_ape_pct": float(np.max(subset_ape)),
                    "rmse_log10_dl": float(np.sqrt(np.mean(subset_log_residual**2))),
                }
            )
    return rows


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def history_csv_rows(
    selection_history: list[dict[str, float]],
    final_history: list[dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, history in (
        ("selection", selection_history),
        ("final_refit", final_history),
    ):
        for row in history:
            rows.append({"phase": phase, **row})
    return rows


def profile_metric_rows(
    test_indices: np.ndarray,
    profile_metrics: dict[str, np.ndarray],
    sobol_global_row: np.ndarray,
) -> Iterable[dict[str, Any]]:
    for position, dataset_index in enumerate(test_indices):
        row: dict[str, Any] = {
            "test_position": position,
            "dataset_index": int(dataset_index),
        }
        if sobol_global_row.size:
            row["sobol_global_row"] = int(sobol_global_row[dataset_index])
        for name, values in profile_metrics.items():
            row[name] = float(values[position])
        yield row


def evaluate_quality_gate(
    overall: dict[str, Any],
    quartile_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    worst_quartile_p95 = float(
        max(row["p95_ape_pct"] for row in quartile_rows)
    )
    checks = {
        "median_ape_pct": {
            "value": float(overall["median_ape_pct"]),
            "operator": "<=",
            "threshold": float(args.gate_median_ape_pct),
            "passed": bool(overall["median_ape_pct"] <= args.gate_median_ape_pct),
        },
        "p95_ape_pct": {
            "value": float(overall["p95_ape_pct"]),
            "operator": "<=",
            "threshold": float(args.gate_p95_ape_pct),
            "passed": bool(overall["p95_ape_pct"] <= args.gate_p95_ape_pct),
        },
        "max_abs_bin_bias_pct": {
            "value": float(overall["max_abs_bin_bias_pct"]),
            "operator": "<=",
            "threshold": float(args.gate_max_bin_bias_pct),
            "passed": bool(
                overall["max_abs_bin_bias_pct"] <= args.gate_max_bin_bias_pct
            ),
        },
        "worst_bin_p95_ape_pct": {
            "value": float(overall["worst_bin_p95_ape_pct"]),
            "operator": "<=",
            "threshold": float(args.gate_worst_bin_p95_ape_pct),
            "passed": bool(
                overall["worst_bin_p95_ape_pct"]
                <= args.gate_worst_bin_p95_ape_pct
            ),
        },
        "min_bin_r2": {
            "value": float(overall["min_bin_r2"]),
            "operator": ">=",
            "threshold": float(args.gate_min_bin_r2),
            "passed": bool(overall["min_bin_r2"] >= args.gate_min_bin_r2),
        },
        "worst_parameter_quartile_p95_ape_pct": {
            "value": worst_quartile_p95,
            "operator": "<=",
            "threshold": float(args.gate_worst_quartile_p95_ape_pct),
            "passed": bool(
                worst_quartile_p95 <= args.gate_worst_quartile_p95_ape_pct
            ),
        },
    }
    return {
        "passed": bool(all(check["passed"] for check in checks.values())),
        "checks": checks,
        "interpretation": (
            "These are adjustable engineering acceptance thresholds, not a "
            "replacement for science-specific end-to-end validation after noise is added."
        ),
    }


def save_diagnostic_plots(
    output_dir: Path,
    ell: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    absolute_percentage: np.ndarray,
    signed_percentage: np.ndarray,
    per_bin: list[dict[str, Any]],
    profile_metrics: dict[str, np.ndarray],
    selection_history: list[dict[str, float]],
    final_history: list[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.semilogy(
        [row["epoch"] for row in selection_history],
        [row["train_scaled_mse"] for row in selection_history],
        label="selection fit",
    )
    ax.semilogy(
        [row["epoch"] for row in selection_history],
        [row["validation_scaled_mse"] for row in selection_history],
        label="selection validation",
    )
    ax.semilogy(
        [row["epoch"] for row in final_history],
        [row["train_scaled_mse"] for row in final_history],
        label="final 85% refit",
        linestyle="--",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE in standardized log10(D_ell)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "training_history.png", dpi=200)
    plt.close(fig)

    median_ape = np.median(absolute_percentage, axis=0)
    p68_ape = np.percentile(absolute_percentage, 68, axis=0)
    p95_ape = np.percentile(absolute_percentage, 95, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    axes[0].plot(ell, median_ape, label="median |% difference|")
    axes[0].plot(ell, p68_ape, label="68th percentile")
    axes[0].plot(ell, p95_ape, label="95th percentile")
    axes[0].axhline(1.0, color="0.4", linestyle=":", linewidth=1)
    axes[0].axhline(5.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Absolute percentage difference")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].plot(ell, [row["mean_bias_pct"] for row in per_bin], label="mean bias")
    axes[1].plot(ell, [row["r2"] for row in per_bin], label="R-squared")
    axes[1].axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Multipole ell")
    axes[1].set_ylabel("Bias (%) or R-squared")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "test_metrics_by_ell.png", dpi=200)
    plt.close(fig)

    score = profile_metrics["p95_ape_pct"]
    ordered = np.argsort(score)
    examples = [
        ("best", int(ordered[0])),
        ("median", int(ordered[len(ordered) // 2])),
        ("worst", int(ordered[-1])),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.0), sharex="col")
    for row_index, (label, index) in enumerate(examples):
        axes[row_index, 0].loglog(ell, truth[index], marker="o", ms=2.5, label="truth")
        axes[row_index, 0].loglog(
            ell, prediction[index], marker=".", ms=2.5, label="emulator"
        )
        axes[row_index, 0].set_ylabel(f"{label} D_ell")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].plot(ell, signed_percentage[index], marker=".", ms=2.5)
        axes[row_index, 1].axhline(0.0, color="0.4", linestyle=":", linewidth=1)
        axes[row_index, 1].set_ylabel("Difference (%)")
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Multipole ell")
    axes[-1, 1].set_xlabel("Multipole ell")
    fig.tight_layout()
    fig.savefig(figure_dir / "best_median_worst_test_profiles.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    upper = float(np.percentile(absolute_percentage, 99.9))
    ax.hist(
        np.clip(absolute_percentage.reshape(-1), 0.0, upper),
        bins=100,
        histtype="stepfilled",
        alpha=0.65,
    )
    ax.set_xlabel(f"Absolute percentage difference (clipped at 99.9th={upper:.3g}%)")
    ax.set_ylabel("Test-bin count")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "test_absolute_percentage_histogram.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    artifact_path = output_dir / "so_noiseless_emulator.pt"

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to update "
            "this run directory; unrelated files are not deleted."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(args.seed, args.threads, args.device)

    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Torch threads: {torch.get_num_threads()}")
    dataset = load_dataset(dataset_path, args)
    contract = dict(dataset["contract"])
    if not args.skip_dataset_hash:
        print("Computing dataset SHA256...", flush=True)
        contract["dataset_sha256"] = sha256_file(dataset_path)
    else:
        contract["dataset_sha256"] = None
        contract["dataset_sha256_skipped"] = True
    write_json(output_dir / "input_provenance.json", contract)

    theta = dataset["theta"]
    target = dataset["target"]
    log10_target = np.ascontiguousarray(np.log10(target), dtype=np.float32)
    splits = make_splits(
        theta.shape[0],
        args.test_fraction,
        args.selection_val_fraction,
        args.seed,
    )
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]
    selection_fit_idx = splits["selection_fit_idx"]
    selection_val_idx = splits["selection_val_idx"]
    split_summary = {
        "seed": int(args.seed),
        "requested_train_fraction": float(1.0 - args.test_fraction),
        "requested_test_fraction": float(args.test_fraction),
        "n_total": int(theta.shape[0]),
        "n_train_outer": int(train_idx.size),
        "n_test": int(test_idx.size),
        "actual_train_fraction": float(train_idx.size / theta.shape[0]),
        "actual_test_fraction": float(test_idx.size / theta.shape[0]),
        "selection_val_fraction_within_outer_train": float(
            args.selection_val_fraction
        ),
        "n_selection_fit": int(selection_fit_idx.size),
        "n_selection_validation": int(selection_val_idx.size),
        "final_refit_rows": int(train_idx.size),
        "test_usage": "untouched until after epoch selection and final 85% refit",
    }
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        selection_fit_idx=selection_fit_idx,
        selection_val_idx=selection_val_idx,
    )
    write_json(output_dir / "split_summary.json", split_summary)
    print(json.dumps(jsonable(split_summary), indent=2), flush=True)

    run_config = {
        "arguments": vars(args),
        "model_config": {
            "input_dim": len(PARAM_NAMES),
            "output_dim": int(target.shape[1]),
            "hidden_width": int(args.hidden_width),
            "residual_blocks": int(args.residual_blocks),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "threads": int(torch.get_num_threads()),
        },
        "split": split_summary,
        "dataset_contract": contract,
    }
    write_json(output_dir / "run_config.json", run_config)
    if args.validate_only:
        write_json(
            output_dir / "validation_only_complete.json",
            {
                "status": "validated",
                "dataset_contract": contract,
                "split": split_summary,
            },
        )
        print("Dataset contract and 85/15 split validated; no model was trained.")
        return 0

    print("Selecting the training epoch inside the outer 85% partition...", flush=True)
    selection_standardizers = fit_standardizers(
        theta, log10_target, selection_fit_idx
    )
    theta_selection_scaled, target_selection_scaled = transform_arrays(
        theta, log10_target, selection_standardizers
    )
    selected_epoch, selection_history = select_training_epoch(
        args,
        theta_selection_scaled,
        target_selection_scaled,
        selection_fit_idx,
        selection_val_idx,
        output_dir,
        device,
    )
    del theta_selection_scaled, target_selection_scaled
    gc.collect()

    print(
        f"Refitting a new model for {selected_epoch} epochs on all "
        f"{train_idx.size} outer-training rows...",
        flush=True,
    )
    final_standardizers = fit_standardizers(theta, log10_target, train_idx)
    theta_final_scaled, target_final_scaled = transform_arrays(
        theta, log10_target, final_standardizers
    )
    final_model, final_history = refit_on_outer_training_set(
        args,
        theta_final_scaled,
        target_final_scaled,
        train_idx,
        selected_epoch,
        device,
    )

    print("Evaluating the untouched 15% test partition...", flush=True)
    prediction_log10 = predict_log10(
        final_model,
        theta_final_scaled,
        test_idx,
        final_standardizers["target_mean"],
        final_standardizers["target_std"],
        int(args.eval_batch_size),
        device,
    )
    prediction = np.ascontiguousarray(
        np.power(10.0, prediction_log10).astype(np.float32)
    )
    truth = np.ascontiguousarray(target[test_idx], dtype=np.float32)
    theta_test = np.ascontiguousarray(theta[test_idx], dtype=np.float32)
    overall, per_bin, profile_metrics, absolute_percentage, signed_percentage = (
        compute_test_metrics(
            truth,
            prediction,
            dataset["ell"],
            dataset["bin_ell_min"],
            dataset["bin_ell_max"],
        )
    )
    quartile_rows = parameter_quartile_metrics(
        theta_test,
        absolute_percentage,
        np.log10(truth.astype(np.float64)),
        prediction_log10.astype(np.float64),
        dataset["prior_low"].astype(np.float64),
        dataset["prior_high"].astype(np.float64),
    )
    quality_gate = evaluate_quality_gate(overall, quartile_rows, args)

    training_summary = {
        "selected_epoch": int(selected_epoch),
        "best_selection_validation_scaled_mse": float(
            min(row["validation_scaled_mse"] for row in selection_history)
        ),
        "selection_epochs_run": len(selection_history),
        "final_refit_epochs": len(final_history),
        "outer_train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
        "selection_fit_rows": int(selection_fit_idx.size),
        "selection_validation_rows": int(selection_val_idx.size),
    }
    model_config = {
        "input_dim": len(PARAM_NAMES),
        "output_dim": int(target.shape[1]),
        "hidden_width": int(args.hidden_width),
        "residual_blocks": int(args.residual_blocks),
    }
    artifact = {
        "artifact_version": 1,
        "model_family": "residual_mlp",
        "model_config": model_config,
        "model_state_dict": cpu_state_dict(final_model),
        "theta_columns": list(PARAM_NAMES),
        "theta_mean": final_standardizers["theta_mean"],
        "theta_std": final_standardizers["theta_std"],
        "target_kind": "binned_linear_D_ell",
        "target_transform": "log10",
        "target_log10_mean": final_standardizers["target_mean"],
        "target_log10_std": final_standardizers["target_std"],
        "ell": dataset["ell"],
        "bin_ell_min": dataset["bin_ell_min"],
        "bin_ell_max": dataset["bin_ell_max"],
        "prior_low": dataset["prior_low"],
        "prior_high": dataset["prior_high"],
        "dataset_contract": contract,
        "split_summary": split_summary,
        "training_summary": training_summary,
        "test_metrics": overall,
        "quality_gate": quality_gate,
        "noise_contract": (
            "The emulator predicts the saved no-noise SO D_ell target only. "
            "Instrument/noise realizations must be added after prediction."
        ),
    }
    torch.save(artifact, artifact_path)

    print("Reloading the saved artifact for an inference round-trip check...", flush=True)
    reloaded_model, reloaded_artifact = load_emulator(artifact_path, device="cpu")
    roundtrip_count = min(16, theta_test.shape[0])
    roundtrip_prediction = predict_profiles(
        theta_test[:roundtrip_count],
        reloaded_model,
        reloaded_artifact,
        device="cpu",
        batch_size=roundtrip_count,
    )
    np.testing.assert_allclose(
        roundtrip_prediction,
        prediction[:roundtrip_count],
        rtol=2.0e-5,
        atol=0.0,
    )

    write_json(output_dir / "test_metrics_overall.json", overall)
    write_csv_rows(output_dir / "test_metrics_by_bin.csv", per_bin)
    write_csv_rows(
        output_dir / "test_metrics_by_parameter_quartile.csv", quartile_rows
    )
    write_csv_rows(
        output_dir / "test_profile_metrics.csv",
        profile_metric_rows(test_idx, profile_metrics, dataset["sobol_global_row"]),
    )
    write_csv_rows(
        output_dir / "training_history.csv",
        history_csv_rows(selection_history, final_history),
    )
    write_json(output_dir / "quality_gate.json", quality_gate)
    write_json(output_dir / "training_summary.json", training_summary)
    artifact_summary = {
        key: value
        for key, value in artifact.items()
        if key != "model_state_dict"
    }
    artifact_summary["artifact_path"] = str(artifact_path)
    artifact_summary["inference_roundtrip_profiles"] = roundtrip_count
    artifact_summary["inference_roundtrip_passed"] = True
    write_json(output_dir / "artifact_summary.json", artifact_summary)

    if not args.no_save_test_predictions:
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            test_idx=test_idx,
            theta=theta_test,
            truth_dl=truth,
            predicted_dl=prediction,
            signed_percentage_difference=signed_percentage.astype(np.float32),
            absolute_percentage_difference=absolute_percentage.astype(np.float32),
            ell=dataset["ell"],
            bin_ell_min=dataset["bin_ell_min"],
            bin_ell_max=dataset["bin_ell_max"],
            theta_columns=np.asarray(PARAM_NAMES),
        )

    if not args.no_plots:
        save_diagnostic_plots(
            output_dir,
            dataset["ell"],
            truth,
            prediction,
            absolute_percentage,
            signed_percentage,
            per_bin,
            profile_metrics,
            selection_history,
            final_history,
        )

    completion = {
        "status": "complete" if quality_gate["passed"] else "quality_gate_failed",
        "artifact": str(artifact_path),
        "dataset": str(dataset_path),
        "split": split_summary,
        "training": training_summary,
        "test_metrics": overall,
        "quality_gate": quality_gate,
        "inference_roundtrip_passed": True,
    }
    write_json(output_dir / "training_complete.json", completion)
    print("Held-out test metrics:")
    print(json.dumps(jsonable(overall), indent=2, sort_keys=True))
    print("Quality gate:")
    print(json.dumps(jsonable(quality_gate), indent=2, sort_keys=True))
    print(f"Saved emulator artifact: {artifact_path}")

    if args.fail_on_quality_gate and not quality_gate["passed"]:
        print(
            "The emulator artifact and diagnostics were saved, but one or more "
            "requested quality thresholds failed.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
