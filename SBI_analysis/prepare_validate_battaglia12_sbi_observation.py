#!/usr/bin/env python3
"""Build a Battaglia12 observation with exactly the saved SBI preprocessing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np


EXPECTED_PRODUCT = "masked_baseline_noise_cross_deproj0"
EXPECTED_FILENAME_PATTERNS = (
    r"masked_baseline_noise_cross_cl",
    r"gaussbeam_2(?:p0+)?arcmin",
    r"so_fsky0p4(?:0*)?_apo60(?:p0+)?arcmin_seed12345_deproj0",
    r"lmax7979",
)
BATTAGLIA12 = {
    "P0": 18.1,
    "xc": 0.497,
    "beta": 4.35,
    "alpha_m_P0": 0.154,
    "alpha_m_xc": -0.00865,
    "alpha_m_beta": 0.0393,
    "alpha_z_P0": -0.758,
    "alpha_z_xc": 0.731,
    "alpha_z_beta": 0.415,
}


def scalar_string(value: Any, default: str = "") -> str:
    array = np.asarray(value)
    return str(array.reshape(()).item()) if array.size == 1 else default


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bin_weights(ell: np.ndarray, weighting: str) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    mode = str(weighting).lower()
    if mode in {"uniform", "none", "flat"}:
        return np.ones_like(ell)
    if mode == "ell":
        return ell
    if mode in {"2ell_plus_1", "modes", "mode_count"}:
        return 2.0 * ell + 1.0
    raise ValueError(f"Unsupported bin weighting: {weighting!r}")


def make_cl_to_dl_matrix(
    ell: np.ndarray,
    bin_min: np.ndarray,
    bin_max: np.ndarray,
    weighting: str,
) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    bin_min = np.asarray(bin_min, dtype=np.float64).reshape(-1)
    bin_max = np.asarray(bin_max, dtype=np.float64).reshape(-1)
    matrix = np.zeros((ell.size, bin_min.size), dtype=np.float64)
    for column, (low, high) in enumerate(zip(bin_min, bin_max)):
        indices = np.flatnonzero((ell >= low) & (ell <= high))
        if indices.size == 0:
            raise ValueError(f"No multipoles in bin {low:g}..{high:g}.")
        weights = bin_weights(ell[indices], weighting)
        matrix[indices, column] = weights / weights.sum()
    dl_factor = ell * (ell + 1.0) / (2.0 * np.pi)
    # This float32 cast is also done before matmul in dataset preparation.
    return np.ascontiguousarray(dl_factor[:, None] * matrix, dtype=np.float32)


def read_profile(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path))
    if values.ndim == 1:
        return np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim == 2 and 1 in values.shape:
        return np.ascontiguousarray(values.reshape(-1), dtype=np.float32)
    if values.ndim == 2 and values.shape[1] >= 2:
        return np.ascontiguousarray(values[:, -1], dtype=np.float32)
    raise ValueError(f"Cannot interpret {path} with shape {values.shape}.")


def align_to_ell(values: np.ndarray, ell: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    if values.size == ell.size:
        return np.ascontiguousarray(values)
    ell_int = np.rint(ell).astype(np.int64)
    if np.allclose(ell, ell_int, rtol=0.0, atol=1e-7) and values.size > ell_int.max():
        return np.ascontiguousarray(values[ell_int], dtype=np.float32)
    raise ValueError(
        f"Profile length {values.size} cannot be aligned to "
        f"ell={ell[0]:g}..{ell[-1]:g} ({ell.size} entries)."
    )


def discover_profile(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*masked_baseline_noise_cross_cl*.npy"))
    else:
        raise FileNotFoundError(f"Raw Battaglia12 path does not exist: {path}")
    if len(candidates) != 1:
        all_npy = sorted(path.rglob("*.npy")) if path.is_dir() else candidates
        raise ValueError(
            f"Expected one baseline-noise cross profile under {path}; found {candidates}. "
            f"All NPY files there are: {all_npy}"
        )
    result = candidates[0].resolve()
    filename = result.name.lower()
    missing = [
        pattern
        for pattern in EXPECTED_FILENAME_PATTERNS
        if re.search(pattern, filename) is None
    ]
    if missing:
        raise ValueError(
            f"{result} does not encode the required beam/mask/seed/deprojection "
            f"configuration. Missing filename patterns: {missing}"
        )
    return result


def find_npe_run(root: Path, case: str, requested_n: int) -> Path:
    case_root = root.expanduser().resolve() / case
    matches: list[tuple[int, Path]] = []
    for path in case_root.rglob("N*"):
        match = re.fullmatch(r"N(\d+)", path.name)
        if (
            match
            and (path / "x_transform.npz").is_file()
            and (path / "density_estimator.pkl").is_file()
        ):
            matches.append((int(match.group(1)), path.resolve()))
    if not matches:
        raise FileNotFoundError(f"No completed NPE runs found under {case_root}.")
    selected_n = requested_n if requested_n > 0 else max(value for value, _ in matches)
    selected = [path for value, path in matches if value == selected_n]
    if len(selected) != 1:
        raise ValueError(f"Expected one NPE run for N={selected_n}; found {selected}.")
    return selected[0]


def load_transform(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "x_transform.npz"
    with np.load(path, allow_pickle=True) as data:
        transform = {key: np.asarray(data[key]).copy() for key in data.files}
    transform["mode"] = scalar_string(transform.get("mode", "none"), "none").lower().replace("-", "_")
    transform["path"] = str(path.resolve())
    return transform


def apply_transform(values: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    mode = transform["mode"]
    if mode in {"none", "raw", ""}:
        result = x
    elif mode == "asinh":
        result = np.arcsinh(x / np.asarray(transform["scale"], dtype=np.float32))
    elif mode == "asinh_standardize":
        scaled = np.arcsinh(x / np.asarray(transform["scale"], dtype=np.float32))
        result = (
            scaled - np.asarray(transform["mean"], dtype=np.float32)
        ) / np.asarray(transform["std"], dtype=np.float32)
    elif mode == "standardize":
        result = (
            x - np.asarray(transform["mean"], dtype=np.float32)
        ) / np.asarray(transform["std"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported saved transform mode: {mode!r}")
    return np.ascontiguousarray(result, dtype=np.float32)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.abs(np.asarray(expected, dtype=np.float64)), 1e-30)
    return np.abs(np.asarray(actual, dtype=np.float64) - expected) / denominator


def lower_median_abs(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=np.float32))
    kth = (absolute.shape[0] - 1) // 2
    return np.partition(absolute, kth, axis=0)[kth]


def self_test() -> int:
    ell = np.arange(2, 10, dtype=np.float32)
    matrix = make_cl_to_dl_matrix(
        ell,
        np.asarray([2, 6]),
        np.asarray([5, 9]),
        "2ell_plus_1",
    )
    binned = align_to_ell(np.arange(10, dtype=np.float32), ell) @ matrix
    transform = {"mode": "asinh", "scale": np.asarray([2.0, 4.0], dtype=np.float32)}
    np.testing.assert_allclose(
        apply_transform(binned, transform),
        np.arcsinh(binned / transform["scale"]),
        rtol=2e-7,
        atol=0.0,
    )
    julia_filename = (
        "halfdome_fullsky_masked_baseline_noise_cross_cl_m200c_nside4096_"
        "base_planck18_gaussbeam_2p0arcmin_"
        "so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    )
    for pattern in EXPECTED_FILENAME_PATTERNS:
        if re.search(pattern, julia_filename) is None:
            raise AssertionError(f"Filename pattern does not match Julia formatting: {pattern}")
    print("Self-test passed: ell alignment, C_ell -> D_ell binning, and asinh.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dataset", type=Path)
    parser.add_argument("--raw-profile", type=Path)
    parser.add_argument("--training-raw-cl", type=Path)
    parser.add_argument("--npe-run-dir", type=Path)
    parser.add_argument("--sbi-run-root", type=Path)
    parser.add_argument("--case", default=EXPECTED_PRODUCT)
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--roundtrip-rows", type=int, default=5)
    parser.add_argument("--expected-transform", default="asinh")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    required = {
        "--prepared-dataset": args.prepared_dataset,
        "--raw-profile": args.raw_profile,
        "--training-raw-cl": args.training_raw_cl,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    prepared_path = args.prepared_dataset.expanduser().resolve()
    training_raw_path = args.training_raw_cl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_profile_path = discover_profile(args.raw_profile)
    if args.npe_run_dir:
        run_dir = args.npe_run_dir.expanduser().resolve()
    elif args.sbi_run_root:
        run_dir = find_npe_run(args.sbi_run_root, args.case, args.n_train)
    else:
        raise ValueError("Pass --npe-run-dir or --sbi-run-root.")
    transform = load_transform(run_dir)

    with np.load(prepared_path, allow_pickle=True) as data:
        needed = {
            "x", "theta", "param_names", "ell_unbinned", "ell_binned",
            "bin_ell_min", "bin_ell_max", "product",
        }
        missing_keys = sorted(needed.difference(data.files))
        if missing_keys:
            raise KeyError(f"Prepared dataset lacks keys: {missing_keys}")
        product = scalar_string(data["product"])
        source_cl_path = (
            scalar_string(data["source_cl_path"])
            if "source_cl_path" in data.files
            else ""
        )
        metadata = {}
        if "metadata_json" in data.files:
            metadata_text = scalar_string(data["metadata_json"])
            metadata = json.loads(metadata_text) if metadata_text else {}
        weighting = str(metadata.get("bin_weighting", ""))
        x_all = np.asarray(data["x"], dtype=np.float32)
        theta = np.asarray(data["theta"], dtype=np.float32)
        param_names = [str(item) for item in data["param_names"]]
        ell_unbinned = np.asarray(data["ell_unbinned"], dtype=np.float32)
        ell_binned = np.asarray(data["ell_binned"], dtype=np.float32)
        bin_min = np.asarray(data["bin_ell_min"], dtype=np.float32)
        bin_max = np.asarray(data["bin_ell_max"], dtype=np.float32)

    checks: dict[str, bool] = {
        "prepared_product_is_masked_baseline_noise_cross_deproj0":
            product == EXPECTED_PRODUCT,
        "requested_case_matches_prepared_product": args.case == product,
        "training_raw_cl_is_prepared_source_cl":
            bool(source_cl_path)
            and Path(source_cl_path).expanduser().resolve() == training_raw_path,
        "bin_weighting_is_2ell_plus_1": weighting == "2ell_plus_1",
        "prepared_x_is_2d_finite":
            x_all.ndim == 2 and bool(np.all(np.isfinite(x_all))),
        "prepared_theta_rows_match_x":
            theta.ndim == 2 and theta.shape[0] == x_all.shape[0],
        "prepared_bin_dimension_matches_x":
            x_all.ndim == 2 and x_all.shape[1] == ell_binned.size,
    }

    matrix = make_cl_to_dl_matrix(ell_unbinned, bin_min, bin_max, weighting)
    training_raw = np.load(training_raw_path, mmap_mode="r")
    if training_raw.ndim != 2 or training_raw.shape[0] < x_all.shape[0]:
        raise ValueError(
            f"Training C_ell shape {training_raw.shape} is incompatible with x {x_all.shape}."
        )
    count = max(1, min(args.roundtrip_rows, x_all.shape[0]))
    indices = np.unique(np.linspace(0, x_all.shape[0] - 1, count, dtype=np.int64))
    raw_rows = np.asarray(training_raw[indices], dtype=np.float32)
    if raw_rows.shape[1] == ell_unbinned.size:
        aligned_rows = raw_rows
    else:
        ell_int = np.rint(ell_unbinned).astype(np.int64)
        if raw_rows.shape[1] <= ell_int.max():
            raise ValueError("Training C_ell columns cannot be aligned to ell_unbinned.")
        aligned_rows = raw_rows[:, ell_int]
    reproduced = np.asarray(aligned_rows, dtype=np.float32) @ matrix
    expected = x_all[indices]
    roundtrip_abs = np.abs(reproduced.astype(np.float64) - expected)
    roundtrip_rel = relative_error(reproduced, expected)
    checks["raw_training_rows_reproduce_prepared_x"] = bool(
        np.allclose(reproduced, expected, rtol=2e-5, atol=1e-6)
    )

    mode = str(transform["mode"])
    expected_mode = args.expected_transform.lower().replace("-", "_")
    checks["saved_transform_mode_matches_expected"] = mode == expected_mode
    if "train_indices" not in transform:
        raise KeyError(f"Saved transform lacks train_indices: {transform['path']}")
    train_indices = np.asarray(transform["train_indices"], dtype=np.int64).reshape(-1)
    if (
        train_indices.size == 0
        or train_indices.min() < 0
        or train_indices.max() >= x_all.shape[0]
    ):
        raise ValueError("Saved transform train_indices are outside the prepared dataset.")
    x_train = np.asarray(x_all[train_indices], dtype=np.float32)
    checks["saved_transform_feature_dimension_matches_x"] = all(
        np.asarray(transform[key]).size == x_all.shape[1]
        for key in ("scale", "mean", "std")
        if key in transform
    )
    scale_rel_max: Optional[float] = None
    if mode in {"asinh", "asinh_standardize"}:
        epsilon = float(np.asarray(transform.get("x_rescale_eps", 1e-30)).reshape(()))
        reconstructed_scale = np.maximum(lower_median_abs(x_train), epsilon).astype(np.float32)
        saved_scale = np.asarray(transform["scale"], dtype=np.float32)
        scale_rel_max = float(np.max(relative_error(saved_scale, reconstructed_scale)))
        checks["saved_asinh_scale_reproduces_training_median_abs"] = bool(
            np.array_equal(saved_scale, reconstructed_scale)
            or np.allclose(saved_scale, reconstructed_scale, rtol=2e-6, atol=0.0)
        )

    raw_profile = read_profile(raw_profile_path)
    aligned_profile = align_to_ell(raw_profile, ell_unbinned)
    binned_dell = np.ascontiguousarray(aligned_profile @ matrix, dtype=np.float32)
    transformed = apply_transform(binned_dell, transform)
    checks["battaglia_binned_dimension_matches_training"] = (
        binned_dell.shape == (x_all.shape[1],)
    )
    checks["battaglia_binned_values_are_finite"] = bool(np.all(np.isfinite(binned_dell)))
    checks["battaglia_transformed_values_are_finite"] = bool(np.all(np.isfinite(transformed)))

    transformed_train = apply_transform(x_train, transform)
    train_mean = transformed_train.mean(axis=0, dtype=np.float64)
    train_std = transformed_train.std(axis=0, dtype=np.float64)
    transformed_z = (
        transformed.astype(np.float64) - train_mean
    ) / np.maximum(train_std, 1e-30)
    percentile = np.mean(transformed_train <= transformed.reshape(1, -1), axis=0)

    truth = np.asarray([BATTAGLIA12[name] for name in param_names], dtype=np.float32)
    binned_path = output_dir / (
        "battaglia12_masked_baseline_noise_cross_deproj0_binned_dell.npy"
    )
    transformed_path = output_dir / (
        "battaglia12_masked_baseline_noise_cross_deproj0_transformed.npy"
    )
    bundle_path = output_dir / (
        "battaglia12_masked_baseline_noise_cross_deproj0_sbi_observation.npz"
    )
    np.save(binned_path, binned_dell)
    np.save(transformed_path, transformed)
    np.savez_compressed(
        bundle_path,
        theta=truth,
        param_names=np.asarray(param_names),
        ell_unbinned=ell_unbinned,
        ell_binned=ell_binned,
        cl_unbinned=aligned_profile,
        x_binned_dell=binned_dell,
        x_transformed=transformed,
        transformed_z=transformed_z.astype(np.float32),
        transformed_percentile=percentile.astype(np.float32),
        product=np.asarray(product),
        raw_profile_path=np.asarray(str(raw_profile_path)),
        prepared_dataset=np.asarray(str(prepared_path)),
        npe_run_dir=np.asarray(str(run_dir)),
    )

    report_path = output_dir / "battaglia12_baseline_deproj0_validation.json"
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "all_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "required_product": EXPECTED_PRODUCT,
        "simulator_contract": {
            "input_statistic": "C_ell",
            "product": EXPECTED_PRODUCT,
            "noise": "SO baseline split-noise cross-spectrum",
            "deprojection": 0,
            "beam_applied_to_signal_map": True,
            "gaussian_beam_fwhm_arcmin": 2.0,
            "mask_fsky": 0.4,
            "mask_apodization_arcmin": 60.0,
            "seed": 12345,
            "ell_min": int(round(float(ell_unbinned.min()))),
            "ell_max": int(round(float(ell_unbinned.max()))),
        },
        "preprocessing_contract": {
            "prepared_statistic": "weighted mean of linear D_ell",
            "cl_to_dl": "ell*(ell+1)/(2*pi)",
            "bin_weighting": weighting,
            "bin_ell_min": bin_min,
            "bin_ell_max": bin_max,
            "x_transform_mode": mode,
            "x_transform_path": transform["path"],
        },
        "prepared_dataset": str(prepared_path),
        "prepared_source_cl": source_cl_path,
        "training_raw_cl": str(training_raw_path),
        "raw_battaglia12_profile": str(raw_profile_path),
        "npe_run_dir": str(run_dir),
        "n_train": int(train_indices.size),
        "roundtrip_indices": indices,
        "roundtrip_max_abs_error": float(roundtrip_abs.max()),
        "roundtrip_max_relative_error": float(roundtrip_rel.max()),
        "saved_scale_max_relative_error": scale_rel_max,
        "battaglia_transformed_z": transformed_z,
        "battaglia_transformed_percentile": percentile,
        "battaglia_max_abs_transformed_z": float(np.max(np.abs(transformed_z))),
        "outputs": {
            "binned_dell": str(binned_path),
            "transformed": str(transformed_path),
            "bundle": str(bundle_path),
            "validation_report": str(report_path),
        },
    }
    write_json(report_path, report)

    print(f"Prepared product: {product}")
    print(f"Raw Battaglia12 profile: {raw_profile_path}")
    print(f"NPE run: {run_dir}")
    print(f"Saved transform mode: {mode}")
    print(f"Training-row roundtrip max relative error: {roundtrip_rel.max():.3e}")
    if scale_rel_max is not None:
        print(f"Saved asinh-scale max relative error: {scale_rel_max:.3e}")
    print(f"Battaglia12 max |transformed z|: {np.max(np.abs(transformed_z)):.3f}")
    print(f"Validation report: {report_path}")
    if not report["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Battaglia12 observation validation failed: {failed}")
    print(f"PASSED: matched SBI observation saved to {binned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
