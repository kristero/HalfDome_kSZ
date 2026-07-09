#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CASES = [
    "no_noise",
    "goal_deproj0",
    "baseline_deproj0",
    "goal_deproj2",
    "baseline_deproj2",
]

PARAM_LABELS = {
    "P0": r"$P_0$",
    "xc": r"$x_{\rm c}$",
    "beta": r"$\beta$",
    "alpha_m_P0": r"$\alpha_{m,P_0}$",
    "alpha_m_xc": r"$\alpha_{m,x_{\rm c}}$",
    "alpha_m_beta": r"$\alpha_{m,\beta}$",
    "alpha_z_P0": r"$\alpha_{z,P_0}$",
    "alpha_z_xc": r"$\alpha_{z,x_{\rm c}}$",
    "alpha_z_beta": r"$\alpha_{z,\beta}$",
}

CASE_LABELS = {
    "no_noise": "no noise",
    "goal_deproj0": "goal, deproj. 0",
    "baseline_deproj0": "baseline, deproj. 0",
    "goal_deproj2": "goal, deproj. 2",
    "baseline_deproj2": "baseline, deproj. 2",
}

CASE_COLORS = {
    "no_noise": "#222222",
    "goal_deproj0": "#1f77b4",
    "baseline_deproj0": "#d62728",
    "goal_deproj2": "#2ca02c",
    "baseline_deproj2": "#9467bd",
}

BEST_VALIDATION_RE = re.compile(
    r"Best validation performance:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


class ForwardingCapture:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.buffer.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def parse_bool(value: Any) -> bool:
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y"}:
        return True
    if raw in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def normalize_cases(values: list[str]) -> list[str]:
    cases: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part:
                cases.append(part)
    return cases


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def parse_validation_losses(output: str) -> list[float]:
    return [float(match.group(1)) for match in BEST_VALIDATION_RE.finditer(output or "")]


def import_sbi_stack() -> tuple[Any, Any, Any, Any]:
    import torch

    try:
        from sbi.inference import NPE as sbi_npe
    except ImportError:
        from sbi.inference import SNPE as sbi_npe

    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        from sbi.utils.get_nn_models import posterior_nn

    from sbi.utils import BoxUniform

    return torch, sbi_npe, posterior_nn, BoxUniform


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def configure_torch_threads(torch: Any) -> dict[str, int]:
    num_threads = env_int("SO_NPE32_NUM_THREADS", env_int("TORCH_NUM_THREADS", 0))
    interop_threads = env_int("TORCH_NUM_INTEROP_THREADS", 0)

    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as exc:
            print(f"Could not set torch inter-op threads: {exc}")

    active = {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }
    print(f"torch num threads: {active['torch_num_threads']}")
    print(f"torch inter-op threads: {active['torch_num_interop_threads']}")
    return active


def make_prior(torch: Any, box_uniform: Any, low: np.ndarray, high: np.ndarray, device: str) -> Any:
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    try:
        return box_uniform(low=low_t, high=high_t, device=device)
    except TypeError:
        return box_uniform(low=low_t, high=high_t)


def make_embedding_net(torch: Any, x_dim: int, context: int) -> Any:
    if int(context) <= 0 or int(context) == int(x_dim):
        return torch.nn.Identity()
    width = max(int(x_dim), int(context), 64)
    return torch.nn.Sequential(
        torch.nn.Linear(int(x_dim), width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, int(context)),
    )


def make_density_builder(
    *,
    torch: Any,
    posterior_nn: Any,
    x_dim: int,
    context: int,
    hidden_features: int,
    num_transforms: int,
    randperm: bool,
    randperm_kw: str,
) -> tuple[Any, dict[str, Any]]:
    embedding_net = make_embedding_net(torch, x_dim, context)
    kwargs: dict[str, Any] = {
        "model": "maf",
        "hidden_features": int(hidden_features),
        "num_transforms": int(num_transforms),
        "z_score_x": "none",
        "z_score_theta": "independent",
        "embedding_net": embedding_net,
    }
    effective_randperm_kw = None
    if randperm_kw != "none":
        effective_randperm_kw = randperm_kw
        kwargs[randperm_kw] = bool(randperm)

    try:
        builder = posterior_nn(**kwargs)
    except TypeError as exc:
        if randperm_kw == "none":
            raise
        print(
            f"Warning: posterior_nn rejected {randperm_kw}={randperm!r}: {exc}. "
            "Retrying without explicit random-permutation control.",
            file=sys.stderr,
        )
        kwargs.pop(randperm_kw, None)
        effective_randperm_kw = None
        builder = posterior_nn(**kwargs)

    meta = {
        "density_estimator": "maf",
        "hidden_features": int(hidden_features),
        "num_transforms": int(num_transforms),
        "context": int(context),
        "x_dim": int(x_dim),
        "embedding_net": "identity" if int(context) == int(x_dim) else f"mlp_{int(context)}",
        "randperm_requested": bool(randperm),
        "randperm_kw": effective_randperm_kw,
    }
    return builder, meta


def make_npe(sbi_npe: Any, prior: Any, density_builder: Any, device: str) -> Any:
    try:
        return sbi_npe(prior=prior, density_estimator=density_builder, device=device)
    except TypeError:
        return sbi_npe(prior=prior, density_estimator=density_builder)


def sample_posterior(posterior: Any, x_obs_t: Any, num_samples: int) -> np.ndarray:
    try:
        samples = posterior.sample((int(num_samples),), x=x_obs_t, show_progress_bars=False)
    except TypeError:
        try:
            samples = posterior.sample((int(num_samples),), x=x_obs_t)
        except TypeError:
            posterior_x = posterior.set_default_x(x_obs_t)
            if posterior_x is None:
                posterior_x = posterior
            try:
                samples = posterior_x.sample((int(num_samples),), show_progress_bars=False)
            except TypeError:
                samples = posterior_x.sample((int(num_samples),))

    try:
        import torch

        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
    except Exception:
        pass

    samples_np = np.asarray(samples, dtype=np.float64)
    if samples_np.ndim == 1:
        samples_np = samples_np.reshape(1, -1)
    elif samples_np.ndim > 2:
        samples_np = samples_np.reshape(-1, samples_np.shape[-1])
    if samples_np.ndim != 2 or samples_np.shape[0] == 0:
        raise ValueError(f"Posterior sampling returned unsupported shape: {samples_np.shape}")
    return samples_np


def save_pickle(path: Path, obj: Any, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {label}: {path}")


def maybe_save_pickle(path: Path, obj: Any, label: str) -> None:
    try:
        save_pickle(path, obj, label)
    except Exception as exc:
        print(f"Warning: could not save {label} to {path}: {exc!r}", file=sys.stderr)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_existing_posterior(run_dir: Path) -> Any:
    posterior_path = run_dir / "posterior.pkl"
    if posterior_path.is_file():
        return load_pickle(posterior_path)

    inference_path = run_dir / "inference.pkl"
    density_path = run_dir / "density_estimator.pkl"
    if inference_path.is_file() and density_path.is_file():
        inference = load_pickle(inference_path)
        density_estimator = load_pickle(density_path)
        return inference.build_posterior(density_estimator)

    raise FileNotFoundError(f"No saved posterior found in {run_dir}")


def asinh_transform_train_eval(
    torch: Any,
    x_train_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    device: str,
    eps: float,
) -> tuple[Any, Any, np.ndarray]:
    x_train_t = torch.as_tensor(x_train_raw, dtype=torch.float32, device=device)
    x_eval_t = torch.as_tensor(x_eval_raw, dtype=torch.float32, device=device)

    scale = torch.median(torch.abs(x_train_t), dim=0).values
    scale = torch.clamp(scale, min=float(eps))

    x_train_final = torch.asinh(x_train_t / scale)
    x_eval_final = torch.asinh(x_eval_t / scale)
    return x_train_final, x_eval_final, scale.detach().cpu().numpy().astype(np.float32)


def asinh_transform_eval_from_scale(torch: Any, x_eval_raw: np.ndarray, scale: np.ndarray, device: str) -> Any:
    x_eval_t = torch.as_tensor(x_eval_raw, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    return torch.asinh(x_eval_t / scale_t)


def build_row_order(n_rows: int, seed: int, order: str) -> np.ndarray:
    if order == "sequential":
        return np.arange(n_rows, dtype=np.int64)
    if order != "shuffle":
        raise ValueError(f"Unsupported dataset order: {order!r}")
    rng = np.random.default_rng(int(seed))
    return rng.permutation(n_rows).astype(np.int64)


def select_sbc_indices(eval_pool: np.ndarray, max_sbc: int, stride: int, seed: int) -> np.ndarray:
    indices = np.asarray(eval_pool, dtype=np.int64)
    if int(stride) > 1:
        indices = indices[:: int(stride)]
    if int(max_sbc) > 0 and indices.size > int(max_sbc):
        rng = np.random.default_rng(int(seed) + 10_003)
        indices = np.sort(rng.choice(indices, size=int(max_sbc), replace=False).astype(np.int64))
    if indices.size == 0:
        raise ValueError("No SBC indices selected")
    return np.ascontiguousarray(indices, dtype=np.int64)


def find_case_dataset(case: str, dataset_dir: Path, index_json: Path | None, root: Path) -> Path:
    if index_json is not None and index_json.is_file():
        index = json.loads(index_json.read_text(encoding="utf-8"))
        case_entry = index.get("cases", {}).get(case)
        if case_entry and case_entry.get("path"):
            path = Path(case_entry["path"]).expanduser()
            candidates = [path]
            if not path.is_absolute():
                candidates.extend([index_json.parent / path, root / path])
            for candidate in candidates:
                if candidate.is_file():
                    return candidate

    matches = sorted(dataset_dir.glob(f"so_{case}_*_sbi_run.npz"))
    if not matches:
        raise FileNotFoundError(f"Could not find case dataset for {case} in {dataset_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple case datasets for {case} in {dataset_dir}: {matches}")
    return matches[0]


def load_dataset(path: Path, fallback_case: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        theta = np.asarray(data["theta"], dtype=np.float32)
        x = np.asarray(data["x"], dtype=np.float32)
        prior_low = np.asarray(data["prior_low"], dtype=np.float32)
        prior_high = np.asarray(data["prior_high"], dtype=np.float32)
        param_names = [str(v) for v in data["param_names"]] if "param_names" in data.files else [f"p{i}" for i in range(theta.shape[1])]
        case_name = scalar_string(data["case_name"], fallback_case) if "case_name" in data.files else fallback_case

    if theta.ndim != 2:
        raise ValueError(f"{path}: theta must be 2D, got {theta.shape}")
    if x.ndim != 2:
        raise ValueError(f"{path}: x must be 2D, got {x.shape}")
    if theta.shape[0] != x.shape[0]:
        raise ValueError(f"{path}: theta rows {theta.shape[0]} do not match x rows {x.shape[0]}")
    if prior_low.shape[0] != theta.shape[1] or prior_high.shape[0] != theta.shape[1]:
        raise ValueError(f"{path}: prior dimension does not match theta dimension")

    return {
        "dataset_path": str(path),
        "case_name": case_name,
        "theta": np.ascontiguousarray(theta, dtype=np.float32),
        "x": np.ascontiguousarray(x, dtype=np.float32),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "param_names": param_names,
        "n_rows": int(theta.shape[0]),
        "x_dim": int(x.shape[1]),
        "theta_dim": int(theta.shape[1]),
    }


def datasets_to_run(args: argparse.Namespace, root: Path) -> list[tuple[str, Path]]:
    if args.prepared_dataset_path:
        path = resolve_path(args.prepared_dataset_path, root)
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset not found: {path}")
        return [(args.single_case_name or path.stem, path)]

    dataset_dir = resolve_path(args.case_dataset_dir, root)
    index_json = resolve_path(args.case_index_json, root) if args.case_index_json else dataset_dir / "case_dataset_index.json"
    return [
        (case, find_case_dataset(case, dataset_dir, index_json, root))
        for case in normalize_cases(args.cases)
    ]


def train_or_load_case(
    *,
    case: str,
    arrays: dict[str, Any],
    train_indices: np.ndarray,
    sbc_indices: np.ndarray,
    case_dir: Path,
    args: argparse.Namespace,
    sbi_stack: tuple[Any, Any, Any, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    torch, sbi_npe, posterior_nn, box_uniform = sbi_stack
    transform_path = case_dir / "x_transform.npz"

    if args.resume and (case_dir / "posterior.pkl").is_file() and transform_path.is_file():
        print(f"Loading existing posterior for {case}: {case_dir}")
        posterior = load_existing_posterior(case_dir)
        with np.load(transform_path, allow_pickle=True) as transform_data:
            scale = np.asarray(transform_data["scale"], dtype=np.float32)
            validation_losses = [float(v) for v in transform_data["validation_losses"]] if "validation_losses" in transform_data.files else []
        x_eval_t = asinh_transform_eval_from_scale(
            torch,
            arrays["x"][sbc_indices],
            scale,
            args.device,
        )
        return posterior, x_eval_t, {
            "status": "loaded",
            "validation_losses": validation_losses,
            "best_validation_loss": validation_losses[-1] if validation_losses else None,
            "x_transform_path": str(transform_path),
        }

    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(int(args.seed))
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    x_train_raw = np.ascontiguousarray(arrays["x"][train_indices], dtype=np.float32)
    theta_train = np.ascontiguousarray(arrays["theta"][train_indices], dtype=np.float32)
    x_eval_raw = np.ascontiguousarray(arrays["x"][sbc_indices], dtype=np.float32)

    x_train_t, x_eval_t, scale_np = asinh_transform_train_eval(
        torch,
        x_train_raw,
        x_eval_raw,
        args.device,
        args.asinh_eps,
    )
    theta_train_t = torch.as_tensor(theta_train, dtype=torch.float32, device=args.device)

    prior = make_prior(
        torch,
        box_uniform,
        arrays["prior_low"],
        arrays["prior_high"],
        args.device,
    )
    density_builder, builder_meta = make_density_builder(
        torch=torch,
        posterior_nn=posterior_nn,
        x_dim=int(arrays["x_dim"]),
        context=int(args.context),
        hidden_features=int(args.hidden_features),
        num_transforms=int(args.num_transforms),
        randperm=bool(args.randperm),
        randperm_kw=str(args.randperm_kw),
    )

    print("")
    print(f"=== Training case={case} ===")
    print(f"dataset: {arrays['dataset_path']}")
    print(f"rows: {arrays['n_rows']}")
    print(f"train rows: {len(train_indices)}")
    print(f"SBC rows: {len(sbc_indices)}")
    print(f"x dim: {arrays['x_dim']}")
    print(f"theta dim: {arrays['theta_dim']}")
    print(f"MAF hidden_features: {args.hidden_features}")
    print(f"MAF num_transforms: {args.num_transforms}")
    print(f"MAF randperm: {args.randperm}")
    print(f"context: {args.context}")
    print(f"x rescale: asinh(x / median(abs(x_train)))")
    print(f"output: {case_dir}")

    inference = make_npe(sbi_npe, prior, density_builder, args.device)
    training_stdout = ForwardingCapture(sys.stdout)
    with contextlib.redirect_stdout(training_stdout):
        density_estimator = inference.append_simulations(theta_train_t, x_train_t).train(
            stop_after_epochs=int(args.stop_after_epochs),
            show_train_summary=True,
        )
    training_output = training_stdout.getvalue()
    validation_losses = parse_validation_losses(training_output)
    training_output_path = case_dir / "training_summary_stdout.txt"
    training_output_path.write_text(training_output, encoding="utf-8")
    write_json(case_dir / "validation_losses.json", validation_losses)

    posterior = inference.build_posterior(density_estimator)

    np.save(case_dir / "train_indices.npy", np.asarray(train_indices, dtype=np.int64))
    np.save(case_dir / "sbc_indices.npy", np.asarray(sbc_indices, dtype=np.int64))
    np.savez_compressed(
        transform_path,
        mode=np.asarray("asinh_median_abs"),
        scale=scale_np,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        sbc_indices=np.asarray(sbc_indices, dtype=np.int64),
        validation_losses=np.asarray(validation_losses, dtype=np.float64),
    )
    save_pickle(case_dir / "density_estimator.pkl", density_estimator, "density estimator")
    save_pickle(case_dir / "posterior.pkl", posterior, "posterior")
    maybe_save_pickle(case_dir / "inference.pkl", inference, "inference")
    maybe_save_pickle(case_dir / "prior.pkl", prior, "prior")

    metadata = {
        "status": "trained",
        "case": case,
        "dataset_path": arrays["dataset_path"],
        "n_rows": arrays["n_rows"],
        "n_train": int(args.n_train),
        "n_sbc": int(len(sbc_indices)),
        "train_index_min": int(np.min(train_indices)),
        "train_index_max": int(np.max(train_indices)),
        "sbc_index_min": int(np.min(sbc_indices)),
        "sbc_index_max": int(np.max(sbc_indices)),
        "dataset_order": args.dataset_order,
        "seed": int(args.seed),
        "x_dim": arrays["x_dim"],
        "theta_dim": arrays["theta_dim"],
        "x_rescale_mode": "asinh_median_abs",
        "asinh_eps": float(args.asinh_eps),
        "x_transform_path": str(transform_path),
        "training_summary_stdout_path": str(training_output_path),
        "validation_losses": validation_losses,
        "best_validation_loss": validation_losses[-1] if validation_losses else None,
        "elapsed_train_seconds": float(time.time() - started),
        **builder_meta,
    }
    write_json(case_dir / "run_metadata.json", metadata)

    del theta_train_t, x_train_t, inference, density_estimator, prior
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return posterior, x_eval_t, metadata


def sbc_for_case(
    *,
    case: str,
    arrays: dict[str, Any],
    posterior: Any,
    x_eval_t: Any,
    sbc_indices: np.ndarray,
    case_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rank_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    param_names = list(arrays["param_names"])
    n_params_all = min(len(param_names), int(arrays["theta_dim"]))

    samples_dir = case_dir / "posterior_samples"
    if args.save_posterior_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for local_i, test_index in enumerate(sbc_indices):
        if local_i == 0 or (local_i + 1) % int(args.progress_every) == 0 or local_i + 1 == len(sbc_indices):
            print(f"  SBC {case}: {local_i + 1}/{len(sbc_indices)}")

        samples = sample_posterior(posterior, x_eval_t[local_i], int(args.num_posterior_samples))
        sample_count = int(samples.shape[0])
        n_params = min(samples.shape[1], n_params_all)
        samples = samples[:, :n_params]
        theta_true = np.asarray(arrays["theta"][int(test_index), :n_params], dtype=np.float64)

        if args.save_posterior_samples:
            np.save(
                samples_dir / f"{case}_idx{int(test_index)}_posterior_samples.npy",
                np.asarray(samples, dtype=np.float32 if args.posterior_sample_dtype == "float32" else np.float64),
            )

        mean = np.nanmean(samples, axis=0)
        std = np.nanstd(samples, axis=0, ddof=1)
        std = np.where(std > 0.0, std, np.nan)
        pull = (mean - theta_true) / std
        profile_rows.append(
            {
                "case": case,
                "n_train": int(args.n_train),
                "test_index": int(test_index),
                "num_posterior_samples": sample_count,
                "mean_abs_pull": float(np.nanmean(np.abs(pull))),
                "rmse_over_std": float(np.sqrt(np.nanmean(pull**2))),
                "mean_posterior_std": float(np.nanmean(std)),
            }
        )

        for j in range(n_params):
            values = samples[:, j]
            rank = int(np.count_nonzero(values < theta_true[j]))
            rank_fraction = float(rank / float(sample_count))
            rank_rows.append(
                {
                    "case": case,
                    "n_train": int(args.n_train),
                    "test_index": int(test_index),
                    "sbc_local_index": int(local_i),
                    "param": param_names[j],
                    "param_index": int(j),
                    "theta_true": float(theta_true[j]),
                    "rank": rank,
                    "rank_fraction": rank_fraction,
                    "num_posterior_samples": sample_count,
                    "posterior_mean": float(mean[j]),
                    "posterior_std": float(std[j]),
                    "pull": float(pull[j]),
                    "x_rescale_mode": "asinh_median_abs",
                    "hidden_features": int(args.hidden_features),
                    "num_transforms": int(args.num_transforms),
                    "randperm": bool(args.randperm),
                }
            )

    cdf_rows, summary_rows = summarize_sbc_rows(rank_rows, param_names)
    write_csv(case_dir / "sbc_ranks.csv", rank_rows, SBC_RANK_FIELDNAMES)
    write_csv(case_dir / "sbc_profile_metrics.csv", profile_rows, SBC_PROFILE_FIELDNAMES)
    write_csv(case_dir / "sbc_summary.csv", summary_rows, SBC_SUMMARY_FIELDNAMES)
    write_csv(case_dir / "sbc_cdf.csv", cdf_rows, SBC_CDF_FIELDNAMES)
    write_json(
        case_dir / "sbc_runtime.json",
        {
            "case": case,
            "n_sbc": int(len(sbc_indices)),
            "num_posterior_samples": int(args.num_posterior_samples),
            "elapsed_sbc_seconds": float(time.time() - started),
        },
    )
    return rank_rows, summary_rows, cdf_rows


def summarize_sbc_rows(
    rank_rows: list[dict[str, Any]],
    param_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_param: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        by_param[str(row["param"])].append(row)

    cdf_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for j, param in enumerate(param_names):
        rows = by_param.get(param, [])
        if not rows:
            continue
        values = np.sort(np.asarray([float(row["rank_fraction"]) for row in rows], dtype=float))
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        empirical = np.arange(1, values.size + 1, dtype=float) / float(values.size)
        empirical_before = np.arange(0, values.size, dtype=float) / float(values.size)
        diff = empirical - values
        ks_statistic = max(
            float(np.nanmax(empirical - values)),
            float(np.nanmax(values - empirical_before)),
        )
        for k, (rank_fraction, empirical_cdf, cdf_minus_uniform) in enumerate(zip(values, empirical, diff)):
            cdf_rows.append(
                {
                    "case": rows[0]["case"],
                    "n_train": int(rows[0]["n_train"]),
                    "param": param,
                    "param_index": int(j),
                    "cdf_order": int(k),
                    "rank_fraction": float(rank_fraction),
                    "empirical_cdf": float(empirical_cdf),
                    "uniform_cdf": float(rank_fraction),
                    "cdf_minus_uniform": float(cdf_minus_uniform),
                }
            )
        summary_rows.append(
            {
                "case": rows[0]["case"],
                "n_train": int(rows[0]["n_train"]),
                "param": param,
                "param_index": int(j),
                "n_sbc": int(values.size),
                "rank_mean_fraction": float(np.nanmean(values)),
                "rank_std_fraction": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
                "rank_min_fraction": float(np.nanmin(values)),
                "rank_max_fraction": float(np.nanmax(values)),
                "ks_statistic": ks_statistic,
                "mean_abs_cdf_minus_uniform": float(np.nanmean(np.abs(diff))),
            }
        )
    return cdf_rows, summary_rows


def plot_sbc_rank_histograms(
    rank_rows: list[dict[str, Any]],
    *,
    case: str,
    param_names: list[str],
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    if not rank_rows:
        return
    n_cols = 3
    n_rows = int(math.ceil(len(param_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54))
    axes = np.asarray(axes).reshape(-1)
    color = CASE_COLORS.get(case, "#1f77b4")

    for ax, param in zip(axes, param_names):
        values = np.asarray([float(row["rank_fraction"]) for row in rank_rows if row["param"] == param], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.axis("off")
            continue
        ax.hist(values, bins=int(bins), range=(0.0, 1.0), color=color, alpha=0.75)
        ax.axhline(values.size / float(bins), color="black", lw=0.8, ls=":", alpha=0.75)
        ax.set_title(PARAM_LABELS.get(param, param), pad=2.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"SBC rank fraction")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    fig.suptitle(CASE_LABELS.get(case, case), y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_sbc_cdfs(
    cdf_rows: list[dict[str, Any]],
    *,
    case: str,
    param_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    if not cdf_rows:
        return
    n_cols = 3
    n_rows = int(math.ceil(len(param_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.0 / 2.54, max(6.0, 5.4 * n_rows) / 2.54))
    axes = np.asarray(axes).reshape(-1)
    color = CASE_COLORS.get(case, "#1f77b4")

    for ax, param in zip(axes, param_names):
        rows = sorted(
            [row for row in cdf_rows if row["param"] == param],
            key=lambda row: int(row["cdf_order"]),
        )
        if not rows:
            ax.axis("off")
            continue
        x = np.asarray([float(row["rank_fraction"]) for row in rows], dtype=float)
        y = np.asarray([float(row["empirical_cdf"]) for row in rows], dtype=float)
        n = max(int(x.size), 1)
        delta = 1.36 / math.sqrt(float(n))
        grid = np.linspace(0.0, 1.0, 200)
        ax.fill_between(
            grid,
            np.clip(grid - delta, 0.0, 1.0),
            np.clip(grid + delta, 0.0, 1.0),
            color="0.85",
            alpha=0.55,
            lw=0.0,
        )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", lw=0.8, ls=":")
        ax.step(np.r_[0.0, x, 1.0], np.r_[0.0, y, 1.0], where="post", color=color, lw=1.0)
        ax.set_title(PARAM_LABELS.get(param, param), pad=2.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"SBC rank fraction")
        ax.set_ylabel(r"Empirical CDF")
        ax.grid(True, alpha=0.25, lw=0.5)

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    fig.suptitle(CASE_LABELS.get(case, case), y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_sbc_summary_by_param(
    summary_rows: list[dict[str, Any]],
    *,
    case: str,
    output_path: Path,
    dpi: int,
) -> None:
    if not summary_rows:
        return
    rows = sorted(summary_rows, key=lambda row: int(row["param_index"]))
    labels = [PARAM_LABELS.get(str(row["param"]), str(row["param"])) for row in rows]
    x = np.arange(len(rows))
    y = np.asarray([float(row["rank_mean_fraction"]) for row in rows], dtype=float)
    yerr = np.asarray([float(row["rank_std_fraction"]) / math.sqrt(max(float(row["n_sbc"]), 1.0)) for row in rows], dtype=float)
    ks = np.asarray([float(row["ks_statistic"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(18.0 / 2.54, 10.0 / 2.54), sharex=True)
    color = CASE_COLORS.get(case, "#1f77b4")

    axes[0].errorbar(x, y, yerr=yerr, marker="o", ms=3.0, lw=1.0, capsize=2.5, color=color)
    axes[0].axhline(0.5, color="black", lw=0.8, ls=":")
    axes[0].set_ylabel(r"Mean rank fraction")
    axes[0].grid(True, axis="y", alpha=0.25, lw=0.5)

    axes[1].plot(x, ks, marker="o", ms=3.0, lw=1.0, color=color)
    axes[1].set_ylabel(r"KS statistic")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25, lw=0.5)

    fig.suptitle(CASE_LABELS.get(case, case), y=0.995, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


SBC_RANK_FIELDNAMES = [
    "case",
    "n_train",
    "test_index",
    "sbc_local_index",
    "param",
    "param_index",
    "theta_true",
    "rank",
    "rank_fraction",
    "num_posterior_samples",
    "posterior_mean",
    "posterior_std",
    "pull",
    "x_rescale_mode",
    "hidden_features",
    "num_transforms",
    "randperm",
]

SBC_PROFILE_FIELDNAMES = [
    "case",
    "n_train",
    "test_index",
    "num_posterior_samples",
    "mean_abs_pull",
    "rmse_over_std",
    "mean_posterior_std",
]

SBC_SUMMARY_FIELDNAMES = [
    "case",
    "n_train",
    "param",
    "param_index",
    "n_sbc",
    "rank_mean_fraction",
    "rank_std_fraction",
    "rank_min_fraction",
    "rank_max_fraction",
    "ks_statistic",
    "mean_abs_cdf_minus_uniform",
]

SBC_CDF_FIELDNAMES = [
    "case",
    "n_train",
    "param",
    "param_index",
    "cdf_order",
    "rank_fraction",
    "empirical_cdf",
    "uniform_cdf",
    "cdf_minus_uniform",
]


def run_case(
    *,
    case: str,
    dataset_path: Path,
    args: argparse.Namespace,
    root: Path,
    sbi_stack: tuple[Any, Any, Any, Any],
) -> dict[str, Any]:
    arrays = load_dataset(dataset_path, case)
    case = str(arrays["case_name"] or case)
    if int(args.n_train) >= int(arrays["n_rows"]):
        raise ValueError(f"n_train={args.n_train} must be smaller than n_rows={arrays['n_rows']} for case={case}")

    row_order = build_row_order(int(arrays["n_rows"]), int(args.seed), args.dataset_order)
    train_indices = np.ascontiguousarray(row_order[: int(args.n_train)], dtype=np.int64)
    eval_pool = np.ascontiguousarray(row_order[int(args.n_train) :], dtype=np.int64)
    sbc_indices = select_sbc_indices(eval_pool, int(args.max_sbc), int(args.sbc_stride), int(args.seed))

    output_root = resolve_path(args.output_dir, root)
    rp_tag = "rp1" if bool(args.randperm) else "rp0"
    case_dir = output_root / case / f"N{int(args.n_train)}_hf{int(args.hidden_features)}_nt{int(args.num_transforms)}_{rp_tag}"
    case_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "case": case,
        "dataset_path": str(dataset_path),
        "output_dir": str(case_dir),
        "n_rows": arrays["n_rows"],
        "n_train": int(args.n_train),
        "n_eval_pool": int(eval_pool.size),
        "n_sbc": int(sbc_indices.size),
        "dataset_order": args.dataset_order,
        "hidden_features": int(args.hidden_features),
        "num_transforms": int(args.num_transforms),
        "randperm": bool(args.randperm),
        "randperm_kw": args.randperm_kw,
        "context": int(args.context),
        "num_posterior_samples": int(args.num_posterior_samples),
        "stop_after_epochs": int(args.stop_after_epochs),
        "x_rescale_mode": "asinh_median_abs",
        "seed": int(args.seed),
        "device": args.device,
    }
    write_json(case_dir / "config.json", config)
    np.save(case_dir / "row_order.npy", row_order)
    np.save(case_dir / "train_indices.npy", train_indices)
    np.save(case_dir / "eval_pool_indices.npy", eval_pool)
    np.save(case_dir / "sbc_indices.npy", sbc_indices)

    posterior, x_eval_t, train_meta = train_or_load_case(
        case=case,
        arrays=arrays,
        train_indices=train_indices,
        sbc_indices=sbc_indices,
        case_dir=case_dir,
        args=args,
        sbi_stack=sbi_stack,
    )

    rank_rows, summary_rows, cdf_rows = sbc_for_case(
        case=case,
        arrays=arrays,
        posterior=posterior,
        x_eval_t=x_eval_t,
        sbc_indices=sbc_indices,
        case_dir=case_dir,
        args=args,
    )

    jpg_dir = case_dir / "jpg"
    param_names = list(arrays["param_names"])
    plot_sbc_rank_histograms(
        rank_rows,
        case=case,
        param_names=param_names,
        output_path=jpg_dir / f"{case}_N{int(args.n_train)}_sbc_rank_histograms.jpg",
        bins=int(args.rank_bins),
        dpi=int(args.dpi),
    )
    plot_sbc_cdfs(
        cdf_rows,
        case=case,
        param_names=param_names,
        output_path=jpg_dir / f"{case}_N{int(args.n_train)}_sbc_rank_cdfs.jpg",
        dpi=int(args.dpi),
    )
    plot_sbc_summary_by_param(
        summary_rows,
        case=case,
        output_path=jpg_dir / f"{case}_N{int(args.n_train)}_sbc_summary_by_param.jpg",
        dpi=int(args.dpi),
    )

    case_summary = {
        "case": case,
        "status": "ok",
        "dataset_path": str(dataset_path),
        "output_dir": str(case_dir),
        "n_rows": arrays["n_rows"],
        "n_train": int(args.n_train),
        "n_sbc": int(sbc_indices.size),
        "mean_ks_statistic": float(np.nanmean([row["ks_statistic"] for row in summary_rows])) if summary_rows else np.nan,
        "mean_abs_cdf_minus_uniform": float(np.nanmean([row["mean_abs_cdf_minus_uniform"] for row in summary_rows])) if summary_rows else np.nan,
        "best_validation_loss": train_meta.get("best_validation_loss"),
    }
    write_json(case_dir / "case_summary.json", case_summary)

    del posterior, x_eval_t
    gc.collect()
    return case_summary


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_dataset = os.environ.get(
        "SO_NPE32_PREPARED_DATASET_PATH",
        os.environ.get("PREPARED_DATASET_PATH", ""),
    )
    parser = argparse.ArgumentParser(
        description=(
            "Train a fixed MAF NPE with hidden_features=64, num_transforms=6, "
            "randperm=False on N=32000 rows, then run SBC on the remaining rows."
        )
    )
    parser.add_argument("--prepared-dataset-path", default=default_dataset)
    parser.add_argument("--single-case-name", default=os.environ.get("SO_NPE32_SINGLE_CASE_NAME", ""))
    parser.add_argument(
        "--case-dataset-dir",
        default=os.environ.get(
            "SO_NPE32_CASE_DATASET_DIR",
            str(root / "SBI_analysis" / "data_for_cluster" / "so_noise_sbi_cases_ell80_7979_battaglia12"),
        ),
    )
    parser.add_argument("--case-index-json", default=os.environ.get("SO_NPE32_CASE_INDEX_JSON", ""))
    parser.add_argument("--cases", nargs="+", default=os.environ.get("SO_NPE32_CASES", " ".join(DEFAULT_CASES)).split())
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "SO_NPE32_OUTPUT_DIR",
            str(root / "SBI_analysis" / "outputs" / "cluster_outputs" / "SBI_SO_npe32k_sbc_hf64_nt6_rp0"),
        ),
    )
    parser.add_argument("--n-train", type=int, default=env_int("SO_NPE32_N_TRAIN", 32_000))
    parser.add_argument("--dataset-order", choices=("sequential", "shuffle"), default=os.environ.get("SO_NPE32_DATASET_ORDER", "sequential"))
    parser.add_argument("--hidden-features", type=int, default=env_int("SO_NPE32_HIDDEN_FEATURES", 64))
    parser.add_argument("--num-transforms", type=int, default=env_int("SO_NPE32_NUM_TRANSFORMS", 6))
    parser.add_argument("--randperm", type=parse_bool, default=parse_bool(os.environ.get("SO_NPE32_RANDPERM", "false")))
    parser.add_argument(
        "--randperm-kw",
        default=os.environ.get("SO_NPE32_RANDPERM_KW", "use_random_permutations"),
        help="Keyword forwarded to posterior_nn. Use 'none' if this sbi version does not expose it.",
    )
    parser.add_argument("--context", type=int, default=env_int("SO_NPE32_CONTEXT", 40))
    parser.add_argument("--stop-after-epochs", type=int, default=env_int("SO_NPE32_STOP_AFTER_EPOCHS", 60))
    parser.add_argument("--num-posterior-samples", type=int, default=env_int("SO_NPE32_NUM_POSTERIOR_SAMPLES", 2000))
    parser.add_argument("--max-sbc", type=int, default=env_int("SO_NPE32_MAX_SBC", 0), help="Use <=0 for all rows after N_train.")
    parser.add_argument("--sbc-stride", type=int, default=env_int("SO_NPE32_SBC_STRIDE", 1))
    parser.add_argument("--rank-bins", type=int, default=env_int("SO_NPE32_RANK_BINS", 20))
    parser.add_argument("--seed", type=int, default=env_int("SO_NPE32_SEED", 42))
    parser.add_argument("--device", default=os.environ.get("SO_NPE32_DEVICE", os.environ.get("SBI_DEVICE", "cpu")))
    parser.add_argument("--asinh-eps", type=float, default=float(os.environ.get("SO_NPE32_ASINH_EPS", "1e-30")))
    parser.add_argument("--posterior-sample-dtype", choices=("float32", "float64"), default=os.environ.get("SO_NPE32_POSTERIOR_SAMPLE_DTYPE", "float32"))
    parser.add_argument("--save-posterior-samples", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=env_int("SO_NPE32_PROGRESS_EVERY", 50))
    parser.add_argument("--dpi", type=int, default=env_int("SO_NPE32_DPI", 300))
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if int(args.n_train) <= 0:
        raise ValueError("--n-train must be positive")
    if int(args.sbc_stride) <= 0:
        raise ValueError("--sbc-stride must be positive")
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive")

    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    sbi_stack = import_sbi_stack()
    thread_info = configure_torch_threads(sbi_stack[0])
    datasets = datasets_to_run(args, root)

    write_json(
        output_dir / "global_config.json",
        {
            **vars(args),
            "output_dir": str(output_dir),
            "datasets": [(case, str(path)) for case, path in datasets],
            "cases": normalize_cases(args.cases),
            **thread_info,
        },
    )

    summaries: list[dict[str, Any]] = []
    for case, dataset_path in datasets:
        try:
            summaries.append(
                run_case(
                    case=case,
                    dataset_path=dataset_path,
                    args=args,
                    root=root,
                    sbi_stack=sbi_stack,
                )
            )
        except Exception as exc:
            failure = {
                "case": case,
                "status": "failed",
                "dataset_path": str(dataset_path),
                "error_type": type(exc).__name__,
                "error": repr(exc),
            }
            summaries.append(failure)
            write_json(output_dir / f"{case}_failed.json", failure)
            print(f"Case {case} failed: {exc!r}", file=sys.stderr)
            if args.fail_fast:
                raise

    write_csv(
        output_dir / "npe32k_sbc_summary.csv",
        summaries,
        [
            "case",
            "status",
            "dataset_path",
            "output_dir",
            "n_rows",
            "n_train",
            "n_sbc",
            "mean_ks_statistic",
            "mean_abs_cdf_minus_uniform",
            "best_validation_loss",
            "error_type",
            "error",
        ],
    )
    write_json(output_dir / "npe32k_sbc_summary.json", summaries)
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
