#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from sbi.inference import NPE as SBI_NPE
except ImportError:
    from sbi.inference import SNPE as SBI_NPE


DEFAULT_POSTERIOR_SAVE_PATH = (
    "outputs/posteriors/posterior_16e3emul_N4e4_binned16_s42_xo_Planck_synthetic_no_noise.npy"
)
DEFAULT_GAUSSIAN_BEAM_FWHM_ARCMIN = 0.0
DEFAULT_GAUSSIAN_BEAM_MODE = "off"
DEFAULT_X_RESCALE_MODE = "asinh"
DEFAULT_X_RESCALE_EPS = 1.0e-30
DEFAULT_X_STANDARDIZE_EPS = 1.0e-8
DEFAULT_DATASET_SIZES = "1024,2048,4096,8192,16384,32768,50e3,70e3,85e3,100e3"
BEST_VALIDATION_RE = re.compile(
    r"Best validation performance:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def parse_dataset_size(value: str) -> int:
    raw = str(value).strip().lower().replace("_", "")
    if not raw:
        raise ValueError("Empty dataset size")
    if raw.endswith("k"):
        size = float(raw[:-1]) * 1_000.0
    else:
        size = float(raw)
    rounded = int(round(size))
    if rounded <= 0 or not np.isclose(size, rounded):
        raise ValueError(f"Dataset size must be a positive integer count, got {value!r}")
    return rounded


def parse_dataset_sizes(value: str) -> list[int]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parts = [part for part in raw.replace(";", ",").replace(" ", ",").split(",") if part]
    sizes = [parse_dataset_size(part) for part in parts]
    seen: set[int] = set()
    unique_sizes = []
    for size in sizes:
        if size not in seen:
            seen.add(size)
            unique_sizes.append(size)
    return unique_sizes


def configure_torch_threads() -> None:
    num_threads = env_int("TORCH_NUM_THREADS", 0)
    interop_threads = env_int("TORCH_NUM_INTEROP_THREADS", 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as exc:
            print(f"Could not set torch inter-op threads: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an SBI NPE posterior from prepared theta/x/obs/prior arrays. "
            "All paths can also be supplied through PBS environment variables."
        )
    )
    parser.add_argument("--prepared-dataset-path", default=os.environ.get("PREPARED_DATASET_PATH", ""))
    parser.add_argument("--prepared-theta-path", default=os.environ.get("PREPARED_THETA_PATH", ""))
    parser.add_argument("--prepared-x-path", default=os.environ.get("PREPARED_X_PATH", ""))
    parser.add_argument("--prepared-obs-path", default=os.environ.get("PREPARED_OBS_PATH", ""))
    parser.add_argument("--prepared-prior-path", default=os.environ.get("PREPARED_PRIOR_PATH", ""))
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SBI_OUTPUT_DIR", ""),
        help="Base output directory for sweep mode. Single-run mode uses the posterior path parent by default.",
    )
    parser.add_argument(
        "--dataset-sizes",
        default=os.environ.get("SBI_DATASET_SIZES", DEFAULT_DATASET_SIZES),
        help=(
            "Comma-separated training-set sizes, e.g. "
            f"{DEFAULT_DATASET_SIZES!r}. Empty means one run with all rows."
        ),
    )
    parser.add_argument(
        "--dataset-order",
        choices=("shuffle", "sequential"),
        default=os.environ.get("SBI_DATASET_ORDER", "shuffle"),
        help="How rows are selected for each training-set size. Shuffle is deterministic from --seed.",
    )
    parser.add_argument(
        "--exclude-last-n-from-training",
        type=int,
        default=env_int("SBI_EXCLUDE_LAST_N_FROM_TRAINING", 0),
        help=(
            "Hold out the last N rows from the training pool. Use this for diagnostics "
            "that condition on the last profiles as test observations."
        ),
    )
    parser.add_argument(
        "--posterior-save-path",
        default=os.environ.get("POSTERIOR_SAVE_PATH", DEFAULT_POSTERIOR_SAVE_PATH),
    )
    parser.add_argument(
        "--gaussian-beam-fwhm-arcmin",
        type=float,
        default=env_float("SBI_GAUSSIAN_BEAM_FWHM_ARCMIN", DEFAULT_GAUSSIAN_BEAM_FWHM_ARCMIN),
        help=(
            "Gaussian beam FWHM in arcmin. In the default signal_only mode this is applied only "
            "to the prepared simulation signal x, not to the observed/noisy vector. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--gaussian-beam-mode",
        choices=("signal_only", "final_vector", "off"),
        default=os.environ.get("SBI_GAUSSIAN_BEAM_MODE", DEFAULT_GAUSSIAN_BEAM_MODE),
        help=(
            "signal_only beams the prepared signal x and leaves obs untouched; final_vector keeps "
            "the old behavior and beams both final log10(D_ell) vectors; off disables the beam."
        ),
    )
    parser.add_argument(
        "--x-rescale-mode",
        choices=("none", "standardize", "asinh", "asinh_standardize"),
        default=os.environ.get("SBI_X_RESCALE_MODE", DEFAULT_X_RESCALE_MODE),
        help=(
            "Transform x and obs before SBI training. Default asinh uses "
            "asinh(x / median(abs(x_train))) with the scale recomputed for each N_train."
        ),
    )
    parser.add_argument(
        "--x-rescale-eps",
        type=float,
        default=env_float("SBI_X_RESCALE_EPS", DEFAULT_X_RESCALE_EPS),
        help="Minimum median-absolute scale for asinh x rescaling.",
    )
    parser.add_argument(
        "--x-standardize-eps",
        type=float,
        default=env_float("SBI_X_STANDARDIZE_EPS", DEFAULT_X_STANDARDIZE_EPS),
        help="Minimum standard deviation for standardize/asinh_standardize x rescaling.",
    )
    parser.add_argument("--seed", type=int, default=env_int("SBI_SEED", 42))
    parser.add_argument(
        "--density-estimator",
        default=os.environ.get("SBI_DENSITY_ESTIMATOR", "maf"),
    )
    parser.add_argument("--stop-after-epochs", type=int, default=env_int("SBI_STOP_AFTER_EPOCHS", 60))
    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=env_int("SBI_NUM_POSTERIOR_SAMPLES", 100000),
    )
    parser.add_argument("--device", default=os.environ.get("SBI_DEVICE", "cpu"))
    return parser.parse_args()


def require_file(path_value: str, label: str) -> Path:
    if not path_value:
        raise ValueError(f"Missing required {label}")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def unpack_numpy_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return value.reshape(()).item()
    return value


def load_numpy_or_pickle(path: str | Path, label: str) -> Any:
    path = require_file(str(path), label)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    return unpack_numpy_object(np.load(path, allow_pickle=True))


def load_prepared_dataset(path_value: str) -> dict[str, Any]:
    path = require_file(path_value, "prepared dataset")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: unpack_numpy_object(data[key]) for key in data.files}
    loaded = unpack_numpy_object(np.load(path, allow_pickle=True))
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Prepared dataset {path} must be a dict-like .npy file or an .npz archive; got {type(loaded)!r}"
        )
    return loaded


def first_present(mapping: dict[str, Any], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Prepared dataset is missing {label}; tried keys {names}")


def as_numpy_float(value: Any, label: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
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


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)


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


def parse_validation_losses_from_training_output(output: str) -> list[float]:
    return [float(match.group(1)) for match in BEST_VALIDATION_RE.finditer(output or "")]


def save_training_output_and_validation_losses(
    output_dir: str | Path,
    training_output: str,
    validation_losses: list[float],
) -> Path:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_output_path = output_dir / "training_summary_stdout.txt"
    with training_output_path.open("w", encoding="utf-8") as handle:
        handle.write(training_output)

    write_json(output_dir / "validation_losses.json", validation_losses)
    if validation_losses:
        print(f"Captured validation losses from SBI summary: {validation_losses}")
    else:
        print("Warning: no 'Best validation performance' values were found in the SBI training summary.")
    print(f"Saved SBI training summary output to {training_output_path}")
    print(f"Saved validation losses to {output_dir / 'validation_losses.json'}")
    return training_output_path


def save_pickle(path: str | Path, obj: Any, label: str) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {label} to {path}")
    return path


def maybe_save_pickle(path: str | Path, obj: Any, label: str) -> Path | None:
    try:
        return save_pickle(path, obj, label)
    except Exception as exc:
        print(f"Warning: could not save {label} to {path}: {exc!r}")
        return None


def load_array_from_path(path_value: str | Path, aliases: tuple[str, ...], label: str) -> np.ndarray:
    path = require_file(str(path_value), label)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for alias in aliases:
                if alias in data:
                    return as_numpy_float(unpack_numpy_object(data[alias]), f"{label}:{alias}")
            if len(data.files) == 1:
                key = data.files[0]
                return as_numpy_float(unpack_numpy_object(data[key]), f"{label}:{key}")
            raise KeyError(f"{label} file {path} is missing one of {aliases}; keys={data.files}")
    return as_numpy_float(load_numpy_or_pickle(path, label), label)


def resolve_array(path_value: str, prepared: dict[str, Any], aliases: tuple[str, ...], label: str) -> np.ndarray:
    if path_value:
        return load_array_from_path(path_value, aliases, label)
    return as_numpy_float(first_present(prepared, aliases, label), label)


def resolve_optional_array(prepared: dict[str, Any], aliases: tuple[str, ...], label: str) -> np.ndarray | None:
    for alias in aliases:
        if alias in prepared:
            return as_numpy_float(prepared[alias], f"{label}:{alias}")
    return None


def build_prior_from_bounds(prepared: dict[str, Any], device: str) -> Any:
    from sbi.utils import BoxUniform

    low = as_numpy_float(first_present(prepared, ("prior_low", "low", "theta_low"), "prior lower bounds"), "prior_low")
    high = as_numpy_float(
        first_present(prepared, ("prior_high", "high", "theta_high"), "prior upper bounds"),
        "prior_high",
    )
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    try:
        return BoxUniform(low=low_t, high=high_t, device=device)
    except TypeError:
        return BoxUniform(low=low_t, high=high_t)


def resolve_prior(path_value: str, prepared: dict[str, Any], device: str) -> Any:
    if path_value:
        return load_numpy_or_pickle(path_value, "prepared prior")
    for key in ("prior", "prepared_prior", "sbi_prior"):
        if key in prepared:
            return unpack_numpy_object(prepared[key])
    if not prepared:
        raise ValueError(
            "Missing prepared prior. Provide --prepared-prior-path or "
            "--prepared-dataset-path containing prior or prior_low/prior_high."
        )
    return build_prior_from_bounds(prepared, device)


def make_npe(prior: Any, density_estimator: str, device: str) -> Any:
    try:
        return SBI_NPE(prior=prior, density_estimator=density_estimator, device=device)
    except TypeError:
        return SBI_NPE(prior=prior, density_estimator=density_estimator)


def sample_posterior(posterior: Any, obs_t: torch.Tensor, num_samples: int) -> np.ndarray:
    try:
        samples = posterior.sample((int(num_samples),), x=obs_t)
    except TypeError:
        if not hasattr(posterior, "set_default_x"):
            raise
        posterior_with_x = posterior.set_default_x(obs_t)
        if posterior_with_x is None:
            posterior_with_x = posterior
        samples = posterior_with_x.sample((int(num_samples),))
    if torch.is_tensor(samples):
        return samples.detach().cpu().numpy()
    return np.asarray(samples)


def build_row_order(n_rows: int, seed: int, dataset_order: str) -> np.ndarray:
    if dataset_order == "sequential":
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.permutation(n_rows).astype(np.int64)


def validate_training_size(size: int, n_rows: int) -> None:
    if size > n_rows:
        raise ValueError(f"Requested training size {size} exceeds available rows {n_rows}")


def gaussian_beam_window(ell: np.ndarray, fwhm_arcmin: float) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    fwhm_arcmin = float(fwhm_arcmin)
    if fwhm_arcmin < 0.0:
        raise ValueError("gaussian_beam_fwhm_arcmin must be non-negative")
    if fwhm_arcmin == 0.0:
        return np.ones_like(ell, dtype=np.float64)
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
    sigma_rad = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma_rad**2)


def apply_gaussian_beam_to_log10_dl(
    values: np.ndarray,
    ell: np.ndarray,
    fwhm_arcmin: float,
    label: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    fwhm_arcmin = float(fwhm_arcmin)
    if fwhm_arcmin == 0.0:
        return values

    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    if values.shape[-1] != ell.size:
        raise ValueError(
            f"Cannot apply Gaussian beam to {label}: last dimension {values.shape[-1]} "
            f"does not match ell length {ell.size}."
        )

    beam = gaussian_beam_window(ell, fwhm_arcmin)
    beam_log_factor = np.log10(np.maximum(beam**2, 1.0e-40)).astype(np.float32)
    return np.ascontiguousarray(values + beam_log_factor, dtype=np.float32)


def apply_requested_gaussian_beam(
    x: np.ndarray,
    obs: np.ndarray,
    ell: np.ndarray | None,
    fwhm_arcmin: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(mode or DEFAULT_GAUSSIAN_BEAM_MODE).strip().lower().replace("-", "_")
    if mode == "off" or float(fwhm_arcmin) == 0.0:
        print("Gaussian beam application disabled.")
        return x, obs
    if ell is None:
        raise ValueError(
            "Gaussian beam application requires an ell array in the prepared dataset. "
            "Add ell=... to the NPZ or set SBI_GAUSSIAN_BEAM_FWHM_ARCMIN=0."
        )

    if mode == "signal_only":
        print(
            "Applying Gaussian beam only to prepared simulation signal x: "
            f"FWHM={float(fwhm_arcmin)} arcmin. The observation is left unchanged; "
            "it must already be generated as cross_spectrum(beam(signal) + noise_1, "
            "beam(signal) + noise_2)."
        )
        x_beamed = apply_gaussian_beam_to_log10_dl(x, ell, fwhm_arcmin, "prepared signal x")
        return x_beamed, obs

    if mode == "final_vector":
        print(
            "Warning: applying Gaussian beam to final prepared x and obs vectors. "
            "This beams any noise already present and is not the correct map-level SO ordering."
        )
        x_beamed = apply_gaussian_beam_to_log10_dl(x, ell, fwhm_arcmin, "prepared x")
        obs_beamed = apply_gaussian_beam_to_log10_dl(obs, ell, fwhm_arcmin, "prepared observation")
        return x_beamed, obs_beamed

    raise ValueError(f"Unsupported gaussian beam mode: {mode!r}")


def transform_x_for_sbi(
    x_train: np.ndarray,
    obs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mode = str(args.x_rescale_mode or DEFAULT_X_RESCALE_MODE).strip().lower().replace("-", "_")
    x_t = torch.as_tensor(np.asarray(x_train, dtype=np.float32), dtype=torch.float32, device=args.device)
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(-1), dtype=torch.float32, device=args.device)

    transform: dict[str, Any] = {
        "mode": mode,
        "x_rescale_eps": float(args.x_rescale_eps),
        "x_standardize_eps": float(args.x_standardize_eps),
    }

    if mode == "none":
        return x_t, obs_t, transform

    if mode == "standardize":
        x_mean = x_t.mean(dim=0)
        x_std = torch.clamp(x_t.std(dim=0), min=float(args.x_standardize_eps))
        x_final = (x_t - x_mean) / x_std
        obs_final = (obs_t - x_mean) / x_std
        transform.update(
            {
                "mean": x_mean.detach().cpu().numpy().astype(np.float32),
                "std": x_std.detach().cpu().numpy().astype(np.float32),
            }
        )
        return x_final, obs_final, transform

    if mode in {"asinh", "asinh_standardize"}:
        scale = torch.median(torch.abs(x_t), dim=0).values
        scale = torch.clamp(scale, min=float(args.x_rescale_eps))
        x_asinh = torch.asinh(x_t / scale)
        obs_asinh = torch.asinh(obs_t / scale)
        transform["scale"] = scale.detach().cpu().numpy().astype(np.float32)

        if mode == "asinh":
            return x_asinh, obs_asinh, transform

        x_mean = x_asinh.mean(dim=0)
        x_std = torch.clamp(x_asinh.std(dim=0), min=float(args.x_standardize_eps))
        x_final = (x_asinh - x_mean) / x_std
        obs_final = (obs_asinh - x_mean) / x_std
        transform.update(
            {
                "mean": x_mean.detach().cpu().numpy().astype(np.float32),
                "std": x_std.detach().cpu().numpy().astype(np.float32),
            }
        )
        return x_final, obs_final, transform

    raise ValueError(f"Unsupported x_rescale_mode: {args.x_rescale_mode!r}")


def save_x_transform(path: Path, transform: dict[str, Any], train_indices: np.ndarray) -> None:
    payload: dict[str, Any] = {
        "mode": np.asarray(str(transform["mode"])),
        "x_rescale_eps": np.asarray(float(transform.get("x_rescale_eps", DEFAULT_X_RESCALE_EPS)), dtype=np.float32),
        "x_standardize_eps": np.asarray(
            float(transform.get("x_standardize_eps", DEFAULT_X_STANDARDIZE_EPS)),
            dtype=np.float32,
        ),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
    }
    for key in ("scale", "mean", "std"):
        if key in transform:
            payload[key] = np.asarray(transform[key], dtype=np.float32)
    np.savez_compressed(path, **payload)


def train_one_size(
    *,
    theta: np.ndarray,
    x: np.ndarray,
    obs: np.ndarray,
    prior: Any,
    row_order: np.ndarray,
    n_train: int,
    args: argparse.Namespace,
    output_dir: Path,
    posterior_save_path: Path,
) -> dict[str, Any]:
    validate_training_size(n_train, row_order.size)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_indices = row_order[:n_train]
    np.save(output_dir / "train_indices.npy", train_indices)
    if int(args.exclude_last_n_from_training) > 0:
        test_indices = np.arange(
            theta.shape[0] - int(args.exclude_last_n_from_training),
            theta.shape[0],
            dtype=np.int64,
        )
        np.save(output_dir / "heldout_test_indices.npy", test_indices)
    theta_train = np.ascontiguousarray(theta[train_indices], dtype=np.float32)
    x_train = np.ascontiguousarray(x[train_indices], dtype=np.float32)
    obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)

    theta_t = torch.as_tensor(theta_train, dtype=torch.float32, device=args.device)
    x_t, obs_t, x_transform = transform_x_for_sbi(x_train, obs_np, args)
    save_x_transform(output_dir / "x_transform.npz", x_transform, train_indices)
    np.save(output_dir / "obs_transformed.npy", obs_t.detach().cpu().numpy().astype(np.float32))

    print("")
    print(f"=== Training SBI with n_train={n_train} ===")
    print(f"theta shape: {tuple(theta_t.shape)}")
    print(f"x shape: {tuple(x_t.shape)}")
    print(f"obs shape: {tuple(obs_t.shape)}")
    print(f"x rescale mode: {x_transform['mode']}")
    print(f"density estimator: {args.density_estimator}")
    print(f"stop_after_epochs: {args.stop_after_epochs}")
    print(f"posterior samples: {args.num_posterior_samples}")
    print(f"output directory: {output_dir}")

    inference = make_npe(prior, args.density_estimator, args.device)
    training_stdout = ForwardingCapture(sys.stdout)
    try:
        with contextlib.redirect_stdout(training_stdout):
            density_estimator = inference.append_simulations(theta_t, x_t).train(
                stop_after_epochs=args.stop_after_epochs,
                show_train_summary=True,
            )
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"The installed sbi version could not build density_estimator={args.density_estimator!r}. "
            "Try SBI_DENSITY_ESTIMATOR=maf for older sbi versions, or use a newer sbi environment "
            "if you need zuko_maf."
        ) from exc
    training_output = training_stdout.getvalue()
    validation_losses = parse_validation_losses_from_training_output(training_output)

    posterior_for_sampling = inference.build_posterior(density_estimator)
    samples = sample_posterior(posterior_for_sampling, obs_t, args.num_posterior_samples)

    posterior_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(posterior_save_path, samples)
    print(f"Saved posterior samples to {posterior_save_path}")

    try:
        posterior_for_save = inference.build_posterior(density_estimator)
    except Exception as exc:
        print(f"Warning: could not rebuild clean posterior for saving: {exc!r}")
        posterior_for_save = posterior_for_sampling

    save_pickle(output_dir / "density_estimator.pkl", density_estimator, "density estimator")
    save_pickle(output_dir / "posterior.pkl", posterior_for_save, "posterior")
    maybe_save_pickle(output_dir / "inference.pkl", inference, "inference")
    maybe_save_pickle(output_dir / "prior.pkl", prior, "prior")
    if hasattr(density_estimator, "state_dict"):
        torch.save(density_estimator.state_dict(), output_dir / "density_estimator_state_dict.pt")
        print(f"Saved density estimator state_dict to {output_dir / 'density_estimator_state_dict.pt'}")

    training_summary_stdout_path = save_training_output_and_validation_losses(
        output_dir,
        training_output,
        validation_losses,
    )

    metadata = {
        "n_train": int(n_train),
        "available_rows": int(theta.shape[0]),
        "training_pool_rows": int(row_order.size),
        "exclude_last_n_from_training": int(args.exclude_last_n_from_training),
        "x_dim": int(x.shape[1]),
        "theta_dim": int(theta.shape[1]),
        "seed": int(args.seed),
        "dataset_order": args.dataset_order,
        "gaussian_beam_fwhm_arcmin": float(args.gaussian_beam_fwhm_arcmin),
        "gaussian_beam_mode": args.gaussian_beam_mode,
        "x_rescale_mode": str(x_transform["mode"]),
        "x_rescale_eps": float(args.x_rescale_eps),
        "x_standardize_eps": float(args.x_standardize_eps),
        "x_transform_path": str(output_dir / "x_transform.npz"),
        "density_estimator": args.density_estimator,
        "stop_after_epochs": int(args.stop_after_epochs),
        "num_posterior_samples": int(args.num_posterior_samples),
        "posterior_samples_path": str(posterior_save_path),
        "training_summary_stdout_path": str(training_summary_stdout_path),
        "validation_losses": validation_losses,
        "best_validation_loss": validation_losses[-1] if validation_losses else None,
    }
    write_json(output_dir / "run_metadata.json", metadata)

    del theta_t, x_t, obs_t, inference, density_estimator, posterior_for_sampling, posterior_for_save
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        **metadata,
        "output_dir": str(output_dir),
    }


def save_sweep_summary(base_output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    csv_path = base_output_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    write_json(base_output_dir / "sweep_summary.json", rows)
    print(f"Saved sweep summary to {csv_path}")


def save_validation_loss_outputs(base_output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    all_validation_losses: list[float] = []
    by_run: list[dict[str, Any]] = []
    for row in rows:
        validation_losses = [float(value) for value in row.get("validation_losses", [])]
        all_validation_losses.extend(validation_losses)
        by_run.append(
            {
                "n_train": int(row["n_train"]),
                "validation_losses": validation_losses,
                "best_validation_loss": row.get("best_validation_loss"),
                "output_dir": row.get("output_dir", ""),
            }
        )

    loss_path = base_output_dir / "validation_losses.json"
    write_json(loss_path, all_validation_losses)
    write_json(base_output_dir / "validation_loss_by_run.json", by_run)
    print(f"Saved aggregate validation losses to {loss_path}")


def main() -> int:
    args = parse_args()
    configure_torch_threads()
    prepared = load_prepared_dataset(args.prepared_dataset_path) if args.prepared_dataset_path else {}

    theta = resolve_array(
        args.prepared_theta_path,
        prepared,
        ("theta", "prepared_theta", "theta_train"),
        "prepared theta",
    )
    x = resolve_array(
        args.prepared_x_path,
        prepared,
        (
            "x",
            "prepared_x",
            "x_train",
            "x_combined",
            "x_log10_dl",
            "x_log10",
            "x_noisy_log10",
            "x_simulations",
        ),
        "prepared x",
    )
    obs = resolve_array(
        args.prepared_obs_path,
        prepared,
        (
            "obs",
            "prepared_obs",
            "observed",
            "x_obs",
            "x_o",
            "x_obs_log10_dl",
            "x_observed_log10",
            "x_observed_log10_dl",
            "x_observed_noisy_log10",
        ),
        "prepared observation",
    )
    prior = resolve_prior(args.prepared_prior_path, prepared, args.device)
    ell = resolve_optional_array(prepared, ("ell", "ells", "l", "multipole"), "ell")

    if theta.ndim != 2:
        raise ValueError(f"prepared theta must be 2D, got shape {theta.shape}")
    if x.ndim != 2:
        raise ValueError(f"prepared x must be 2D, got shape {x.shape}")
    if theta.shape[0] != x.shape[0]:
        raise ValueError(f"theta rows {theta.shape[0]} do not match x rows {x.shape[0]}")
    if obs.ndim > 1:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[-1] != x.shape[-1]:
        raise ValueError(f"observation length {obs.shape[-1]} does not match x dimension {x.shape[-1]}")
    x, obs = apply_requested_gaussian_beam(
        x,
        obs,
        ell,
        args.gaussian_beam_fwhm_arcmin,
        args.gaussian_beam_mode,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"torch num threads: {torch.get_num_threads()}")
    print(f"torch inter-op threads: {torch.get_num_interop_threads()}")

    dataset_sizes = parse_dataset_sizes(args.dataset_sizes)
    if not dataset_sizes:
        dataset_sizes = [int(theta.shape[0])]
        sweep_mode = False
    else:
        sweep_mode = True
    exclude_last_n = int(args.exclude_last_n_from_training)
    if exclude_last_n < 0:
        raise ValueError("--exclude-last-n-from-training must be non-negative")
    if exclude_last_n >= theta.shape[0]:
        raise ValueError(
            f"Cannot exclude the last {exclude_last_n} rows from a dataset with {theta.shape[0]} rows"
        )
    training_pool_rows = theta.shape[0] - exclude_last_n

    for n_train in dataset_sizes:
        validate_training_size(n_train, training_pool_rows)

    row_order = build_row_order(training_pool_rows, args.seed, args.dataset_order)
    posterior_path = Path(args.posterior_save_path).expanduser()
    base_output_dir = Path(args.output_dir).expanduser() if args.output_dir else posterior_path.parent
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"available dataset rows: {theta.shape[0]}")
    print(f"training pool rows: {training_pool_rows}")
    print(f"excluded last rows from training: {exclude_last_n}")
    print(f"dataset sizes: {dataset_sizes}")
    print(f"dataset order: {args.dataset_order}")
    print(f"x rescale mode: {args.x_rescale_mode}")
    print(f"base output directory: {base_output_dir}")

    run_summaries: list[dict[str, Any]] = []
    for n_train in dataset_sizes:
        if sweep_mode:
            run_dir = base_output_dir / f"N{n_train}"
            run_posterior_path = run_dir / "posterior_samples.npy"
        else:
            run_dir = base_output_dir
            run_posterior_path = posterior_path

        summary = train_one_size(
            theta=theta,
            x=x,
            obs=obs,
            prior=prior,
            row_order=row_order,
            n_train=n_train,
            args=args,
            output_dir=run_dir,
            posterior_save_path=run_posterior_path,
        )
        run_summaries.append(summary)

    if sweep_mode:
        save_sweep_summary(base_output_dir, run_summaries)
        save_validation_loss_outputs(base_output_dir, run_summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
