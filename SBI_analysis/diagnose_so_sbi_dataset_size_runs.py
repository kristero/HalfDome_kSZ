#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CASES = [
    "masked_no_noise",
    "masked_baseline_noise_cross_deproj0",
]

DEFAULT_DATASET_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    49152,
    65536,
    81920,
    98304,
    114688,
    131072,
    163840,
    196608,
    229376,
    262144,
    327680,
    393216,
    458752,
    523788,
    524288,
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [part for part in str(value).replace(";", ",").replace(" ", ",").split(",") if part]
    return [int(float(part.replace("_", ""))) for part in parts]


def n_sort_key(path: Path) -> int:
    match = re.search(r"N(\d+)$", path.name)
    return int(match.group(1)) if match else 999999999


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def find_case_dataset(case: str, dataset_dir: Path, index_json: Path | None) -> Path:
    if index_json is not None and index_json.is_file():
        index = read_json(index_json)
        case_entry = index.get("cases", {}).get(case)
        if case_entry and case_entry.get("path"):
            path = Path(case_entry["path"]).expanduser()
            if path.is_file():
                return path
    matches = sorted(dataset_dir.glob(f"so_{case}_*_sbi_run.npz"))
    if not matches:
        raise FileNotFoundError(f"Could not find case dataset for {case} in {dataset_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple case datasets for {case}: {matches}")
    return matches[0]


def find_run_dirs(run_root: Path, case: str, dataset_sizes: list[int], allow_missing: bool) -> dict[int, Path]:
    case_root = run_root / case
    if not case_root.is_dir():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Case output directory not found: {case_root}")

    found: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = {}
    for path in sorted(case_root.glob("**/N*"), key=n_sort_key):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"N(\d+)", path.name)
        if match is None:
            continue
        n_train = int(match.group(1))
        if n_train not in dataset_sizes:
            continue
        if not ((path / "inference.pkl").is_file() and (path / "density_estimator.pkl").is_file()):
            continue
        if n_train in found:
            duplicates.setdefault(n_train, [found[n_train]]).append(path)
        else:
            found[n_train] = path

    if duplicates:
        msg = "\n".join(f"N={n}: {[str(p) for p in paths]}" for n, paths in sorted(duplicates.items()))
        raise ValueError(f"Duplicate completed run directories under {case_root}:\n{msg}")

    if not allow_missing:
        missing = [n for n in dataset_sizes if n not in found]
        if missing:
            raise FileNotFoundError(f"Missing completed runs for case={case}: {missing}")
    return dict(sorted(found.items()))


def load_x_transform(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def dataset_summary(case: str, dataset_path: Path) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        sobol_global_row = np.asarray(data["sobol_global_row"], dtype=np.int64) if "sobol_global_row" in data.files else np.empty(0, dtype=np.int64)
        metadata_json = scalar_string(data["metadata_json"], "{}") if "metadata_json" in data.files else "{}"
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        metadata = {}

    theta_first = theta[: min(32768, theta.shape[0])]
    theta_after = theta[32768:] if theta.shape[0] > 32768 else np.empty((0, theta.shape[1]))
    return {
        "case": case,
        "dataset_path": str(dataset_path),
        "dataset_rows": int(theta.shape[0]),
        "x_dim": int(x.shape[1]),
        "theta_dim": int(theta.shape[1]),
        "theta_std_mean_all": float(np.nanmean(np.nanstd(theta, axis=0))),
        "theta_std_mean_first32768": float(np.nanmean(np.nanstd(theta_first, axis=0))),
        "theta_std_mean_after32768": float(np.nanmean(np.nanstd(theta_after, axis=0))) if theta_after.size else "",
        "source_metadata_path": metadata.get("source_metadata_path", ""),
        "source_metadata_csv_path": metadata.get("source_metadata_csv_path", ""),
        "source_sobol_csv_path": metadata.get("source_sobol_csv_path", ""),
        "theta_source": metadata.get("theta_source", ""),
        "sobol_global_row_first": int(sobol_global_row[0]) if sobol_global_row.size else "",
        "sobol_global_row_last": int(sobol_global_row[-1]) if sobol_global_row.size else "",
        "sobol_global_row_monotonic": bool(np.all(np.diff(sobol_global_row) > 0)) if sobol_global_row.size > 1 else "",
    }


def run_summary(case: str, n_train: int, run_dir: Path, dataset_rows: int, last_n_test: int) -> dict[str, Any]:
    metadata = read_json(run_dir / "run_metadata.json")
    transform = load_x_transform(run_dir / "x_transform.npz")
    train_indices = np.asarray(transform.get("train_indices", np.empty(0)), dtype=np.int64).reshape(-1)
    eval_indices = np.arange(int(dataset_rows) - int(last_n_test), int(dataset_rows), dtype=np.int64)
    overlap = np.intersect1d(train_indices, eval_indices, assume_unique=False) if train_indices.size else np.empty(0, dtype=np.int64)

    warnings: list[str] = []
    if not metadata:
        warnings.append("missing_run_metadata")
    if train_indices.size and train_indices.size != int(n_train):
        warnings.append("train_indices_count_not_n_train")
    if train_indices.size and np.unique(train_indices).size != train_indices.size:
        warnings.append("duplicate_train_indices")
    if overlap.size:
        warnings.append("train_eval_overlap")
    if metadata.get("x_dim", "") and int(metadata["x_dim"]) <= 0:
        warnings.append("bad_x_dim")
    if metadata.get("dataset_order", "") == "sequential":
        warnings.append("sequential_training_order")
    if int(n_train) > 32768 and train_indices.size and np.max(train_indices) <= 32767:
        warnings.append("large_n_but_indices_only_first32768")

    return {
        "case": case,
        "n_train": int(n_train),
        "run_dir": str(run_dir),
        "dataset_order": metadata.get("dataset_order", ""),
        "exclude_last_n_from_training": metadata.get("exclude_last_n_from_training", ""),
        "training_pool_rows": metadata.get("training_pool_rows", ""),
        "available_rows": metadata.get("available_rows", ""),
        "x_rescale_mode": metadata.get("x_rescale_mode", ""),
        "density_estimator": metadata.get("density_estimator", ""),
        "stop_after_epochs": metadata.get("stop_after_epochs", ""),
        "best_validation_loss": metadata.get("best_validation_loss", ""),
        "train_indices_count": int(train_indices.size),
        "train_indices_unique_count": int(np.unique(train_indices).size) if train_indices.size else 0,
        "train_indices_min": int(np.min(train_indices)) if train_indices.size else "",
        "train_indices_max": int(np.max(train_indices)) if train_indices.size else "",
        "fraction_train_indices_ge_32768": float(np.mean(train_indices >= 32768)) if train_indices.size else "",
        "last_n_test": int(last_n_test),
        "train_eval_overlap_count": int(overlap.size),
        "train_eval_overlap_examples": " ".join(str(int(v)) for v in overlap[:10]),
        "warnings": ";".join(warnings),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Inspect prepared SO SBI datasets and completed dataset-size runs.")
    parser.add_argument(
        "--run-root",
        default=str(root / "SBI_analysis" / "outputs" / "cluster_outputs" / "SBI_Adrian_SO_dataset_size_ell80_7979_dataset_row_sobolrow_asinh"),
    )
    parser.add_argument(
        "--case-dataset-dir",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow"),
    )
    parser.add_argument("--case-index-json", default="")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--dataset-sizes", default=",".join(str(v) for v in DEFAULT_DATASET_SIZES))
    parser.add_argument("--last-n-test", type=int, default=500)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_root = resolve_path(args.run_root, root)
    dataset_dir = resolve_path(args.case_dataset_dir, root)
    index_json = resolve_path(args.case_index_json, root) if args.case_index_json else dataset_dir / "case_dataset_index.json"
    output_dir = resolve_path(args.output_dir, root) if args.output_dir else run_root / "diagnostics_run_consistency"
    dataset_sizes = parse_int_list(args.dataset_sizes)

    dataset_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for case in args.cases:
        try:
            dataset_path = find_case_dataset(case, dataset_dir, index_json)
        except FileNotFoundError as exc:
            if args.allow_missing:
                print(f"Skipping missing case dataset for {case}: {exc}")
                continue
            raise
        drow = dataset_summary(case, dataset_path)
        dataset_rows.append(drow)
        print("")
        print(f"Case {case}")
        print(f"  dataset: {dataset_path}")
        print(f"  rows={drow['dataset_rows']} x_dim={drow['x_dim']} theta_dim={drow['theta_dim']}")
        print(f"  metadata={drow['source_metadata_path'] or 'none'}")
        print(f"  metadata_csv={drow['source_metadata_csv_path'] or 'none'}")
        print(f"  sobol_csv={drow['source_sobol_csv_path'] or 'none'}")
        print(f"  theta_source={drow['theta_source'] or 'unknown'}")
        if drow["theta_source"] == "sobol_csv_identity_row_order":
            print("  WARNING: theta came from fallback Sobol CSV identity row order; verify C_ell row i is Sobol row i.")

        runs = find_run_dirs(run_root, case, dataset_sizes, args.allow_missing)
        print(f"  completed N values: {list(runs)}")
        for n_train, run_dir in runs.items():
            rrow = run_summary(case, n_train, run_dir, int(drow["dataset_rows"]), int(args.last_n_test))
            run_rows.append(rrow)
            warning_text = f" warnings={rrow['warnings']}" if rrow["warnings"] else ""
            print(
                f"    N={n_train}: order={rrow['dataset_order']} "
                f"exclude={rrow['exclude_last_n_from_training']} "
                f"idx=[{rrow['train_indices_min']},{rrow['train_indices_max']}] "
                f"overlap={rrow['train_eval_overlap_count']}{warning_text}"
            )

    write_csv(output_dir / "dataset_consistency.csv", dataset_rows)
    write_csv(output_dir / "run_consistency.csv", run_rows)
    print("")
    print(f"Wrote {output_dir / 'dataset_consistency.csv'}")
    print(f"Wrote {output_dir / 'run_consistency.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
