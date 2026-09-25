#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sbi_utils import (
    dataset_to_log10_dl,
    estimate_lightcone_sigma,
    load_combined_dataset,
    load_config,
    prior_bounds,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare transformed Battaglia SBI arrays from the emulator dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output or config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_combined_dataset(
        config["combined_dataset_path"],
        int(config.get("ell_min", 2)),
        config.get("ell_max", None),
    )
    transformed = dataset_to_log10_dl(dataset)
    low, high = prior_bounds(config, dataset["x_columns"])
    noise_cfg = config.get("noise", {})
    sigma = estimate_lightcone_sigma(
        transformed["y100"],
        transformed["y102"],
        target=noise_cfg.get("target", "single_lightcone"),
        sigma_floor=float(noise_cfg.get("sigma_floor", 1.0e-4)),
        scale=float(noise_cfg.get("scale", 1.0)),
    )

    output_npz = output_dir / "sbi_prepared_dataset.npz"
    np.savez_compressed(
        output_npz,
        theta=dataset["theta"].astype(np.float32),
        x_combined=transformed["combined"].astype(np.float32),
        x_y100=transformed["y100"].astype(np.float32),
        x_y102=transformed["y102"].astype(np.float32),
        ell=dataset["ell"].astype(np.float32),
        sigma_log10_dl=sigma.astype(np.float32),
        x_columns=np.asarray(dataset["x_columns"]),
        prior_low=low,
        prior_high=high,
    )
    write_json(
        output_dir / "sbi_prepared_dataset_summary.json",
        {
            "source_dataset": dataset["source_path"],
            "output_npz": str(output_npz),
            "n_parameter_points": int(dataset["theta"].shape[0]),
            "n_ell": int(dataset["ell"].size),
            "x_columns": dataset["x_columns"],
            "noise_target": noise_cfg.get("target", "single_lightcone"),
            "median_sigma_log10_dl": float(np.median(sigma)),
            "max_sigma_log10_dl": float(np.max(sigma)),
        },
    )
    print(f"Wrote {output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
