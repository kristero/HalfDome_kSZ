#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import itertools
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


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


def parse_int_list(value: str) -> list[int]:
    parts = [part for part in str(value).replace(";", ",").replace(" ", ",").split(",") if part]
    out = []
    for part in parts:
        number = float(part.replace("_", ""))
        rounded = int(round(number))
        if rounded <= 0 or not math.isclose(number, rounded):
            raise ValueError(f"Expected positive integer, got {part!r}")
        out.append(rounded)
    return out


def parse_bool_list(value: str) -> list[bool]:
    out = []
    for part in str(value).replace(";", ",").replace(" ", ",").split(","):
        raw = part.strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif raw in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Expected boolean value, got {part!r}")
    return out


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


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


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
            "Retrying without explicit randperm control.",
            file=sys.stderr,
        )
        kwargs.pop(randperm_kw, None)
        effective_randperm_kw = None
        builder = posterior_nn(**kwargs)

    meta = {
        "model": "maf",
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
            return samples.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(samples)


def asinh_transform_train_and_obs(
    torch: Any,
    x_train_raw: np.ndarray,
    x_obs_raw: np.ndarray,
    device: str,
    eps: float,
) -> tuple[Any, Any, np.ndarray]:
    x_train_t = torch.as_tensor(x_train_raw, dtype=torch.float32, device=device)
    x_obs_t = torch.as_tensor(x_obs_raw, dtype=torch.float32, device=device)

    scale = torch.median(torch.abs(x_train_t), dim=0).values
    scale = torch.clamp(scale, min=float(eps))

    x_train_asinh = torch.asinh(x_train_t / scale)
    x_obs_asinh = torch.asinh(x_obs_t / scale)
    return x_train_asinh, x_obs_asinh, scale.detach().cpu().numpy().astype(np.float32)


def build_row_order(n_pool: int, seed: int, order: str) -> np.ndarray:
    if order == "sequential":
        return np.arange(n_pool, dtype=np.int64)
    if order != "shuffle":
        raise ValueError(f"Unsupported dataset order: {order!r}")
    rng = np.random.default_rng(int(seed))
    return rng.permutation(n_pool).astype(np.int64)


def metric_rows_for_obs(
    *,
    samples: np.ndarray,
    theta_true: np.ndarray,
    param_names: list[str],
    trial_number: int,
    obs_index: int,
    obs_label: str,
    trial_dir: Path,
    dtype: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = np.asarray(samples, dtype=np.float64)
    n_params = min(samples.shape[1], theta_true.size, len(param_names))
    samples = samples[:, :n_params]
    theta_true = np.asarray(theta_true, dtype=np.float64)[:n_params]

    mean = np.nanmean(samples, axis=0)
    std = np.nanstd(samples, axis=0, ddof=1)
    std = np.where(std > 0.0, std, np.nan)
    error = mean - theta_true
    pull = error / std

    samples_path = trial_dir / f"posterior_samples_{obs_label}.npy"
    np.save(samples_path, np.asarray(samples, dtype=np.float32 if dtype == "float32" else np.float64))

    mse = float(np.nanmean(error**2))
    rmse = float(np.sqrt(mse))
    rmse_over_std = float(np.sqrt(np.nanmean(pull**2)))
    mse_over_std = float(np.nanmean(pull**2))
    mean_std = float(np.nanmean(std))

    profile_row = {
        "trial_number": int(trial_number),
        "obs_index": int(obs_index),
        "obs_label": obs_label,
        "mse": mse,
        "rmse": rmse,
        "rmse_over_std": rmse_over_std,
        "mse_over_std": mse_over_std,
        "mean_posterior_std": mean_std,
        "posterior_samples_path": str(samples_path),
    }

    param_rows = []
    for j in range(n_params):
        param_rows.append(
            {
                "trial_number": int(trial_number),
                "obs_index": int(obs_index),
                "obs_label": obs_label,
                "param": param_names[j],
                "param_index": int(j),
                "theta_true": float(theta_true[j]),
                "posterior_mean": float(mean[j]),
                "posterior_std": float(std[j]),
                "error": float(error[j]),
                "pull": float(pull[j]),
                "abs_pull": float(abs(pull[j])),
            }
        )
    return profile_row, param_rows


def summarize_profile_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_mse": float(np.nanmean([row["mse"] for row in rows])),
        "mean_rmse": float(np.nanmean([row["rmse"] for row in rows])),
        "mean_rmse_over_std": float(np.nanmean([row["rmse_over_std"] for row in rows])),
        "mean_mse_over_std": float(np.nanmean([row["mse_over_std"] for row in rows])),
        "mean_posterior_std": float(np.nanmean([row["mean_posterior_std"] for row in rows])),
    }


def objective_value(summary: dict[str, Any], objective: str) -> float:
    if summary.get("status") != "ok":
        return float("inf")
    if objective == "neg_validation":
        best = summary.get("best_validation_loss")
        return float("inf") if best is None else -float(best)
    return float(summary[objective])


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(path: Path, obj: Any, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {label}: {path}")


def run_trial(
    *,
    trial_number: int,
    params: dict[str, Any],
    args: argparse.Namespace,
    arrays: dict[str, Any],
    sbi_stack: tuple[Any, Any, Any, Any],
) -> dict[str, Any]:
    trial_name = (
        f"trial{int(trial_number):03d}_"
        f"hf{int(params['hidden_features'])}_"
        f"nt{int(params['num_transforms'])}_"
        f"rp{int(bool(params['randperm']))}"
    )
    trial_dir = Path(args.output_dir) / trial_name
    summary_path = trial_dir / "trial_summary.json"
    if args.resume:
        cached = read_json_if_exists(summary_path)
        if cached is not None and cached.get("status") == "ok":
            print(f"Skipping completed {trial_name}")
            return cached

    trial_dir.mkdir(parents=True, exist_ok=True)
    torch, sbi_npe, posterior_nn, box_uniform = sbi_stack
    np.save(trial_dir / "train_indices.npy", arrays["train_indices"])
    np.save(trial_dir / "obs_indices.npy", arrays["obs_indices"])

    started = time.time()
    summary: dict[str, Any] = {
        "trial_number": int(trial_number),
        "trial_name": trial_name,
        "status": "running",
        "hidden_features": int(params["hidden_features"]),
        "num_transforms": int(params["num_transforms"]),
        "randperm": bool(params["randperm"]),
        "context": int(args.context),
        "n_train": int(args.n_train),
        "last_n_obs": int(args.last_n_obs),
        "num_posterior_samples": int(args.num_posterior_samples),
        "x_rescale_mode": "asinh_median_abs",
        "output_dir": str(trial_dir),
    }
    write_json(summary_path, summary)

    try:
        torch.manual_seed(int(args.seed) + int(trial_number))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + int(trial_number))

        x_train_t, x_obs_t, scale_np = asinh_transform_train_and_obs(
            torch,
            arrays["x_train_raw"],
            arrays["x_obs_raw"],
            args.device,
            args.asinh_eps,
        )
        theta_t = torch.as_tensor(arrays["theta_train"], dtype=torch.float32, device=args.device)
        np.savez_compressed(
            trial_dir / "x_transform.npz",
            mode=np.asarray("asinh_median_abs"),
            scale=scale_np,
            train_indices=arrays["train_indices"],
            obs_indices=arrays["obs_indices"],
        )

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
            hidden_features=int(params["hidden_features"]),
            num_transforms=int(params["num_transforms"]),
            randperm=bool(params["randperm"]),
            randperm_kw=str(args.randperm_kw),
        )
        summary.update(builder_meta)

        inference = make_npe(sbi_npe, prior, density_builder, args.device)
        training_stdout = ForwardingCapture(sys.stdout)
        with contextlib.redirect_stdout(training_stdout):
            density_estimator = inference.append_simulations(theta_t, x_train_t).train(
                stop_after_epochs=int(args.stop_after_epochs),
                show_train_summary=True,
            )
        training_output = training_stdout.getvalue()
        validation_losses = parse_validation_losses(training_output)
        training_output_path = trial_dir / "training_summary_stdout.txt"
        training_output_path.write_text(training_output, encoding="utf-8")
        write_json(trial_dir / "validation_losses.json", validation_losses)

        posterior = inference.build_posterior(density_estimator)
        profile_rows: list[dict[str, Any]] = []
        param_rows: list[dict[str, Any]] = []

        for local_i, obs_index in enumerate(arrays["obs_indices"]):
            obs_label = f"obs{int(obs_index)}"
            samples = sample_posterior(posterior, x_obs_t[local_i], int(args.num_posterior_samples))
            profile_row, obs_param_rows = metric_rows_for_obs(
                samples=samples,
                theta_true=arrays["theta_obs"][local_i],
                param_names=arrays["param_names"],
                trial_number=int(trial_number),
                obs_index=int(obs_index),
                obs_label=obs_label,
                trial_dir=trial_dir,
                dtype=args.posterior_sample_dtype,
            )
            profile_rows.append(profile_row)
            param_rows.extend(obs_param_rows)

        write_csv(
            trial_dir / "profile_metrics.csv",
            profile_rows,
            [
                "trial_number",
                "obs_index",
                "obs_label",
                "mse",
                "rmse",
                "rmse_over_std",
                "mse_over_std",
                "mean_posterior_std",
                "posterior_samples_path",
            ],
        )
        write_csv(
            trial_dir / "param_metrics.csv",
            param_rows,
            [
                "trial_number",
                "obs_index",
                "obs_label",
                "param",
                "param_index",
                "theta_true",
                "posterior_mean",
                "posterior_std",
                "error",
                "pull",
                "abs_pull",
            ],
        )

        metric_summary = summarize_profile_rows(profile_rows)
        summary.update(metric_summary)
        summary.update(
            {
                "status": "ok",
                "validation_losses": validation_losses,
                "best_validation_loss": validation_losses[-1] if validation_losses else None,
                "training_summary_stdout_path": str(training_output_path),
                "elapsed_seconds": float(time.time() - started),
            }
        )

        if args.save_density_estimator:
            save_pickle(trial_dir / "density_estimator.pkl", density_estimator, "density estimator")
            save_pickle(trial_dir / "posterior.pkl", posterior, "posterior")
            save_pickle(trial_dir / "inference.pkl", inference, "inference")

        del theta_t, x_train_t, x_obs_t, inference, density_estimator, posterior
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "elapsed_seconds": float(time.time() - started),
            }
        )
        write_json(summary_path, summary)
        print(f"Trial {trial_name} failed: {exc!r}", file=sys.stderr)
        if args.fail_fast:
            raise

    summary["objective_name"] = args.objective
    summary["objective_value"] = objective_value(summary, args.objective)
    write_json(summary_path, summary)
    print(
        f"{trial_name}: status={summary['status']} objective={summary['objective_value']:.6g} "
        f"mse={summary.get('mean_mse')} rmse/std={summary.get('mean_rmse_over_std')} "
        f"std={summary.get('mean_posterior_std')} val={summary.get('best_validation_loss')}"
    )
    return summary


def load_arrays(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    dataset_path = resolve_path(args.prepared_dataset_path, root)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")

    print(f"Reading prepared dataset: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as data:
        theta = np.asarray(data["theta"], dtype=np.float32)
        x = np.asarray(data["x"], dtype=np.float32)
        prior_low = np.asarray(data["prior_low"], dtype=np.float32)
        prior_high = np.asarray(data["prior_high"], dtype=np.float32)
        param_names = [str(v) for v in data["param_names"]] if "param_names" in data.files else [f"p{i}" for i in range(theta.shape[1])]
        case_name = scalar_string(data["case_name"], "") if "case_name" in data.files else ""

    if theta.ndim != 2:
        raise ValueError(f"theta must be 2D, got {theta.shape}")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")
    if theta.shape[0] != x.shape[0]:
        raise ValueError(f"theta rows {theta.shape[0]} do not match x rows {x.shape[0]}")
    if int(args.last_n_obs) <= 0:
        raise ValueError("--last-n-obs must be positive")
    if int(args.last_n_obs) >= theta.shape[0]:
        raise ValueError("--last-n-obs must be smaller than the number of rows")

    n_pool = theta.shape[0] - int(args.last_n_obs)
    if int(args.n_train) > n_pool:
        raise ValueError(f"--n-train={args.n_train} exceeds training pool rows={n_pool}")

    row_order = build_row_order(n_pool, int(args.seed), args.dataset_order)
    train_indices = row_order[: int(args.n_train)]
    obs_indices = np.arange(theta.shape[0] - int(args.last_n_obs), theta.shape[0], dtype=np.int64)

    arrays = {
        "dataset_path": str(dataset_path),
        "case_name": case_name,
        "theta": theta,
        "x": x,
        "theta_train": np.ascontiguousarray(theta[train_indices], dtype=np.float32),
        "x_train_raw": np.ascontiguousarray(x[train_indices], dtype=np.float32),
        "theta_obs": np.ascontiguousarray(theta[obs_indices], dtype=np.float32),
        "x_obs_raw": np.ascontiguousarray(x[obs_indices], dtype=np.float32),
        "train_indices": np.ascontiguousarray(train_indices, dtype=np.int64),
        "obs_indices": np.ascontiguousarray(obs_indices, dtype=np.int64),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "param_names": param_names,
        "x_dim": int(x.shape[1]),
        "theta_dim": int(theta.shape[1]),
        "n_rows": int(theta.shape[0]),
        "training_pool_rows": int(n_pool),
    }
    return arrays


def hyperparameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    hidden_values = parse_int_list(args.hidden_features)
    transform_values = parse_int_list(args.num_transforms)
    randperm_values = parse_bool_list(args.randperm_values)
    combos = [
        {
            "hidden_features": hidden_features,
            "num_transforms": num_transforms,
            "randperm": randperm,
        }
        for hidden_features, num_transforms, randperm in itertools.product(
            hidden_values,
            transform_values,
            randperm_values,
        )
    ]
    return combos[: int(args.max_trials)] if int(args.max_trials) > 0 else combos


def run_with_optuna(args: argparse.Namespace, arrays: dict[str, Any], sbi_stack: tuple[Any, Any, Any, Any]) -> list[dict[str, Any]]:
    import optuna

    search_space = {
        "hidden_features": parse_int_list(args.hidden_features),
        "num_transforms": parse_int_list(args.num_transforms),
        "randperm": parse_bool_list(args.randperm_values),
    }
    try:
        sampler = optuna.samplers.GridSampler(search_space, seed=int(args.seed))
    except TypeError:
        sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=args.optuna_storage or None,
        load_if_exists=bool(args.optuna_storage),
    )
    rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = {
            "hidden_features": trial.suggest_categorical("hidden_features", search_space["hidden_features"]),
            "num_transforms": trial.suggest_categorical("num_transforms", search_space["num_transforms"]),
            "randperm": trial.suggest_categorical("randperm", search_space["randperm"]),
        }
        summary = run_trial(
            trial_number=int(trial.number),
            params=params,
            args=args,
            arrays=arrays,
            sbi_stack=sbi_stack,
        )
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                trial.set_user_attr(key, value)
        rows.append(summary)
        return float(summary["objective_value"])

    n_grid = math.prod(len(v) for v in search_space.values())
    n_trials = min(int(args.max_trials), n_grid) if int(args.max_trials) > 0 else n_grid
    study.optimize(objective, n_trials=n_trials)
    write_json(Path(args.output_dir) / "optuna_best_trial.json", study.best_trial.user_attrs)
    return rows


def run_with_grid(args: argparse.Namespace, arrays: dict[str, Any], sbi_stack: tuple[Any, Any, Any, Any]) -> list[dict[str, Any]]:
    rows = []
    for trial_number, params in enumerate(hyperparameter_grid(args)):
        rows.append(
            run_trial(
                trial_number=trial_number,
                params=params,
                args=args,
                arrays=arrays,
                sbi_stack=sbi_stack,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_dataset = os.environ.get(
        "NPE_TUNE_PREPARED_DATASET_PATH",
        os.environ.get("PREPARED_DATASET_PATH", ""),
    )
    parser = argparse.ArgumentParser(
        description=(
            "Tune sbi NPE/MAF hyperparameters on a prepared SO SBI dataset. "
            "The last profiles are held out as observations and x is transformed as asinh(x/s)."
        )
    )
    parser.add_argument("--prepared-dataset-path", default=default_dataset)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "NPE_TUNE_OUTPUT_DIR",
            str(root / "SBI_analysis" / "outputs" / "cluster_outputs" / "SBI_SO_npe_maf_tuning"),
        ),
    )
    parser.add_argument("--n-train", type=int, default=int(os.environ.get("NPE_TUNE_N_TRAIN", 16384)))
    parser.add_argument("--last-n-obs", type=int, default=int(os.environ.get("NPE_TUNE_LAST_N_OBS", 10)))
    parser.add_argument("--context", type=int, default=int(os.environ.get("NPE_TUNE_CONTEXT", 40)))
    parser.add_argument("--hidden-features", default=os.environ.get("NPE_TUNE_HIDDEN_FEATURES", "64,128,256"))
    parser.add_argument("--num-transforms", default=os.environ.get("NPE_TUNE_NUM_TRANSFORMS", "4,6,8,10"))
    parser.add_argument("--randperm-values", default=os.environ.get("NPE_TUNE_RANDPERM_VALUES", "false,true"))
    parser.add_argument(
        "--randperm-kw",
        default=os.environ.get("NPE_TUNE_RANDPERM_KW", "use_random_permutations"),
        help="Keyword forwarded to posterior_nn for MAF permutation control. Use 'none' if unsupported.",
    )
    parser.add_argument("--max-trials", type=int, default=int(os.environ.get("NPE_TUNE_MAX_TRIALS", 24)))
    parser.add_argument(
        "--objective",
        choices=("mean_mse", "mean_rmse_over_std", "mean_mse_over_std", "mean_posterior_std", "neg_validation"),
        default=os.environ.get("NPE_TUNE_OBJECTIVE", "mean_mse"),
        help="Quantity minimized to choose the best trial. neg_validation minimizes -best_validation_loss.",
    )
    parser.add_argument(
        "--use-optuna",
        choices=("auto", "yes", "no"),
        default=os.environ.get("NPE_TUNE_USE_OPTUNA", "auto"),
    )
    parser.add_argument("--optuna-storage", default=os.environ.get("NPE_TUNE_OPTUNA_STORAGE", ""))
    parser.add_argument("--study-name", default=os.environ.get("NPE_TUNE_STUDY_NAME", "so_npe_maf_context40"))
    parser.add_argument("--dataset-order", choices=("shuffle", "sequential"), default=os.environ.get("NPE_TUNE_DATASET_ORDER", "shuffle"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("NPE_TUNE_SEED", 42)))
    parser.add_argument("--device", default=os.environ.get("NPE_TUNE_DEVICE", os.environ.get("SBI_DEVICE", "cpu")))
    parser.add_argument("--stop-after-epochs", type=int, default=int(os.environ.get("NPE_TUNE_STOP_AFTER_EPOCHS", 60)))
    parser.add_argument("--num-posterior-samples", type=int, default=int(os.environ.get("NPE_TUNE_NUM_POSTERIOR_SAMPLES", 50000)))
    parser.add_argument("--asinh-eps", type=float, default=float(os.environ.get("NPE_TUNE_ASINH_EPS", "1e-30")))
    parser.add_argument("--posterior-sample-dtype", choices=("float32", "float64"), default=os.environ.get("NPE_TUNE_POSTERIOR_SAMPLE_DTYPE", "float32"))
    parser.add_argument("--save-density-estimator", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if not args.prepared_dataset_path:
        raise ValueError("Provide --prepared-dataset-path or PREPARED_DATASET_PATH.")

    args.output_dir = str(resolve_path(args.output_dir, root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sbi_stack = import_sbi_stack()
    arrays = load_arrays(args, root)
    np.save(output_dir / "train_indices.npy", arrays["train_indices"])
    np.save(output_dir / "obs_indices.npy", arrays["obs_indices"])

    config = {
        "prepared_dataset_path": arrays["dataset_path"],
        "case_name": arrays["case_name"],
        "n_rows": arrays["n_rows"],
        "training_pool_rows": arrays["training_pool_rows"],
        "n_train": int(args.n_train),
        "last_n_obs": int(args.last_n_obs),
        "obs_indices": arrays["obs_indices"],
        "x_dim": arrays["x_dim"],
        "theta_dim": arrays["theta_dim"],
        "context": int(args.context),
        "hidden_features": parse_int_list(args.hidden_features),
        "num_transforms": parse_int_list(args.num_transforms),
        "randperm_values": parse_bool_list(args.randperm_values),
        "randperm_kw": args.randperm_kw,
        "x_rescale_mode": "asinh_median_abs",
        "objective": args.objective,
        "max_trials": int(args.max_trials),
        "num_posterior_samples": int(args.num_posterior_samples),
        "stop_after_epochs": int(args.stop_after_epochs),
        "seed": int(args.seed),
        "dataset_order": args.dataset_order,
        "device": args.device,
    }
    write_json(output_dir / "tuning_config.json", config)

    use_optuna = args.use_optuna
    if use_optuna in {"auto", "yes"}:
        try:
            rows = run_with_optuna(args, arrays, sbi_stack)
            search_backend = "optuna_grid"
        except ModuleNotFoundError:
            if use_optuna == "yes":
                raise
            print("Optuna is not installed; falling back to deterministic grid search.")
            rows = run_with_grid(args, arrays, sbi_stack)
            search_backend = "grid"
    else:
        rows = run_with_grid(args, arrays, sbi_stack)
        search_backend = "grid"

    rows = sorted(rows, key=lambda row: (float(row.get("objective_value", float("inf"))), int(row.get("trial_number", 999999))))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["search_backend"] = search_backend

    fieldnames = [
        "rank",
        "trial_number",
        "trial_name",
        "status",
        "hidden_features",
        "num_transforms",
        "randperm",
        "context",
        "objective_name",
        "objective_value",
        "best_validation_loss",
        "mean_mse",
        "mean_rmse",
        "mean_rmse_over_std",
        "mean_mse_over_std",
        "mean_posterior_std",
        "elapsed_seconds",
        "output_dir",
        "error_type",
        "error",
        "search_backend",
    ]
    write_csv(output_dir / "tuning_summary.csv", rows, fieldnames)
    write_json(output_dir / "tuning_summary.json", rows)
    if rows:
        write_json(output_dir / "best_trial.json", rows[0])
        print(f"Best trial: {rows[0]['trial_name']} objective={rows[0]['objective_value']}")
    print(f"Wrote tuning outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
