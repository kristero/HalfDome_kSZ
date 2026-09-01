#!/usr/bin/env python3
"""Load and evaluate the deterministic 9-parameter SO profile emulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PARAM_NAMES = (
    "P0",
    "xc",
    "beta",
    "alpha_m_P0",
    "alpha_m_xc",
    "alpha_m_beta",
    "alpha_z_P0",
    "alpha_z_xc",
    "alpha_z_beta",
)

# Authoritative bounds for the SO Sobol dataset.
SO_PARAMETER_PRIOR_LOW = np.asarray(
    [1.832524, 0.150011, 3.480627, 0.000312, -0.099718, -0.019935, -1.363457, 0.147393, 0.083808],
    dtype=np.float32,
)
SO_PARAMETER_PRIOR_HIGH = np.asarray(
    [34.341221, 0.844503, 5.216611, 0.292251, 0.099795, 0.099767, -0.228839, 1.314474, 0.745884],
    dtype=np.float32,
)


class ResidualBlock(nn.Module):
    """A small fully connected residual block for smooth profile emulation."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(self.norm(values)) / np.sqrt(2.0)


class SOProfileEmulator(nn.Module):
    """Residual MLP mapping nine pressure parameters to standardized profiles."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        residual_blocks: int,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_width) for _ in range(residual_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_width)
        self.output_layer = nn.Linear(hidden_width, output_dim)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(theta)
        hidden = self.blocks(hidden)
        return self.output_layer(self.output_norm(hidden))


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary artifact in {path}, got {type(payload)!r}")
    return payload


def load_emulator(
    artifact_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SOProfileEmulator, dict[str, Any]]:
    """Load a saved emulator artifact and return the model plus its metadata."""

    artifact_path = Path(artifact_path).expanduser().resolve()
    device = torch.device(device)
    artifact = _torch_load(artifact_path, map_location=device)
    required = {
        "artifact_version",
        "model_config",
        "model_state_dict",
        "theta_mean",
        "theta_std",
        "target_log10_mean",
        "target_log10_std",
        "theta_columns",
        "ell",
        "prior_low",
        "prior_high",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise KeyError(f"Artifact {artifact_path} is missing keys: {missing}")
    if int(artifact["artifact_version"]) != 1:
        raise ValueError(
            f"Unsupported artifact version {artifact['artifact_version']!r}; expected 1"
        )
    if str(artifact.get("target_transform", "")) != "log10":
        raise ValueError("This loader expects a log10(D_ell) target transform.")

    config = dict(artifact["model_config"])
    model = SOProfileEmulator(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        hidden_width=int(config["hidden_width"]),
        residual_blocks=int(config["residual_blocks"]),
    )
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, artifact


def predict_profiles(
    theta: np.ndarray,
    model: SOProfileEmulator,
    artifact: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 8192,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Predict positive, linear binned D_ell profiles for rows of theta."""

    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2 or theta.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"theta must have shape (N, {len(PARAM_NAMES)}), got {theta.shape}"
        )
    columns = tuple(str(value) for value in artifact["theta_columns"])
    if columns != PARAM_NAMES:
        raise ValueError(f"Unexpected artifact parameter order: {columns}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")

    prior_low = np.asarray(artifact["prior_low"], dtype=np.float64)
    prior_high = np.asarray(artifact["prior_high"], dtype=np.float64)
    tolerance = 1.0e-6 * np.maximum(1.0, np.abs(prior_high - prior_low))
    outside = (theta < prior_low - tolerance) | (theta > prior_high + tolerance)
    if np.any(outside) and not allow_extrapolation:
        row, column = np.argwhere(outside)[0]
        raise ValueError(
            f"theta row {row}, parameter {columns[column]}={theta[row, column]:.8g} "
            f"is outside [{prior_low[column]:.8g}, {prior_high[column]:.8g}]. "
            "Pass allow_extrapolation=True only if this is intentional."
        )

    theta_mean = np.asarray(artifact["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(artifact["theta_std"], dtype=np.float32)
    target_mean = np.asarray(artifact["target_log10_mean"], dtype=np.float32)
    target_std = np.asarray(artifact["target_log10_std"], dtype=np.float32)
    theta_scaled = np.ascontiguousarray((theta - theta_mean) / theta_std)

    device = torch.device(device)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, theta_scaled.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), theta_scaled.shape[0])
            batch = torch.from_numpy(theta_scaled[start:stop]).to(device)
            pred_scaled = model(batch).cpu().numpy()
            pred_log10 = pred_scaled * target_std + target_mean
            predictions.append(np.power(10.0, pred_log10).astype(np.float32))

    result = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise FloatingPointError("Emulator produced non-finite or non-positive D_ell values")
    return result


def load_theta(path: Path, key: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{path} does not contain key {key!r}; keys={data.files}")
            return np.asarray(data[key], dtype=np.float32)
    if suffix in {".csv", ".txt"}:
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        if table.dtype.names is None:
            return np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float32)
        missing = [name for name in PARAM_NAMES if name not in table.dtype.names]
        if missing:
            raise KeyError(f"CSV {path} is missing parameter columns: {missing}")
        return np.column_stack([table[name] for name in PARAM_NAMES]).astype(np.float32)
    raise ValueError(f"Unsupported theta file extension: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict noiseless binned SO D_ell profiles with a trained emulator."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--theta-key", default="theta")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--allow-extrapolation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    theta_path = args.theta.expanduser().resolve()
    artifact_path = args.artifact.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    theta = load_theta(theta_path, args.theta_key)
    model, artifact = load_emulator(artifact_path, args.device)
    profiles = predict_profiles(
        theta,
        model,
        artifact,
        device=args.device,
        batch_size=args.batch_size,
        allow_extrapolation=args.allow_extrapolation,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact": str(artifact_path),
        "theta_source": str(theta_path),
        "theta_columns": list(PARAM_NAMES),
        "target": "binned_linear_D_ell",
        "noise": "none; add noise only after emulation",
        "n_profiles": int(profiles.shape[0]),
        "n_bins": int(profiles.shape[1]),
    }
    np.savez_compressed(
        output_path,
        theta=np.asarray(theta, dtype=np.float32),
        dl=profiles,
        ell=np.asarray(artifact["ell"], dtype=np.float32),
        theta_columns=np.asarray(PARAM_NAMES),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved {profiles.shape[0]} noiseless profiles to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
