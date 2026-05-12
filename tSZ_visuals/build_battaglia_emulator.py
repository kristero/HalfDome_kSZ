#!/usr/bin/env python3
"""
Build a Battaglia-parameter tSZ profile emulator from y100/y102 realizations.

The script is intentionally separate from the Julia map painter. It consumes the
saved profile products, pairs y100 and y102 by the actual Sobol/Battaglia
parameters, averages the paired profiles, and trains a PCA + regression emulator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PARAM_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("P0", ("p0", "p0_amp", "battaglia_p0_amp")),
    ("xc", ("xc", "x_c", "x_c_amp", "battaglia_x_c_amp")),
    ("beta", ("beta", "beta_amp", "battaglia_beta_amp")),
    ("alpha", ("alpha", "alpha_amp", "battaglia_alpha_amp")),
    ("gamma", ("gamma", "gamma_amp", "battaglia_gamma_amp")),
    ("alpha_m_P0", ("alpha_m_p0", "p0_alpha_m", "battaglia_p0_alpha_m")),
    ("alpha_m_xc", ("alpha_m_xc", "alpha_m_x_c", "x_c_alpha_m", "battaglia_x_c_alpha_m")),
    ("alpha_m_beta", ("alpha_m_beta", "beta_alpha_m", "battaglia_beta_alpha_m")),
    ("alpha_m_alpha", ("alpha_m_alpha", "alpha_alpha_m", "battaglia_alpha_alpha_m")),
    ("alpha_m_gamma", ("alpha_m_gamma", "gamma_alpha_m", "battaglia_gamma_alpha_m")),
    ("alpha_z_P0", ("alpha_z_p0", "p0_alpha_z", "battaglia_p0_alpha_z")),
    ("alpha_z_xc", ("alpha_z_xc", "alpha_z_x_c", "x_c_alpha_z", "battaglia_x_c_alpha_z")),
    ("alpha_z_beta", ("alpha_z_beta", "beta_alpha_z", "battaglia_beta_alpha_z")),
    ("alpha_z_alpha", ("alpha_z_alpha", "alpha_alpha_z", "battaglia_alpha_alpha_z")),
    ("alpha_z_gamma", ("alpha_z_gamma", "gamma_alpha_z", "battaglia_gamma_alpha_z")),
)

ALLOWED_REGRESSORS = (
    "mlp",
    "krr",
    "gpr",
    "extra_trees",
    "random_forest",
    "gradient_boosting",
    "knn",
)


@dataclass(frozen=True)
class SobolTable:
    path: Path
    tag: str
    rows: Any
    column_map: dict[str, str]


@dataclass
class ProfileRecord:
    path: Path
    key: tuple[str, ...]
    params: dict[str, float]
    csv_path: Path
    csv_row: int
    values: Any

    @property
    def source_id(self) -> tuple[str, int]:
        return (str(self.csv_path), int(self.csv_row))


def normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def safe_filename_tag(value: str) -> str:
    tag = value.strip().lower()
    tag = re.sub(r"[^A-Za-z0-9_+\-.]+", "_", tag)
    tag = re.sub(r"_+", "_", tag)
    return tag or "simulation"


def split_cli_values(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        comma_chunks = value.split(",")
        for comma_chunk in comma_chunks:
            comma_chunk = comma_chunk.strip()
            if not comma_chunk:
                continue
            if os.pathsep == ":" and not re.match(r"^[A-Za-z]:[\\/]", comma_chunk):
                chunks = comma_chunk.split(":")
            elif os.pathsep != ":":
                chunks = comma_chunk.split(os.pathsep)
            else:
                chunks = [comma_chunk]
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    expanded.append(chunk)
    return expanded


def split_comma_values(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if chunk:
                expanded.append(chunk)
    return expanded


def param_key(params: dict[str, float], columns: list[str], precision: int) -> tuple[str, ...]:
    return tuple(format(float(params[col]), f".{precision}g") for col in columns)


def load_sobol_tables(csv_paths: list[str]):
    import numpy as np
    import pandas as pd

    if not csv_paths:
        raise ValueError("At least one Sobol CSV path is required")

    tables: list[SobolTable] = []
    for raw_path in csv_paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Sobol CSV not found: {path}")

        rows = pd.read_csv(path)
        if rows.empty:
            raise ValueError(f"Sobol CSV has no data rows: {path}")
        normalized_lookup = {normalize_name(str(col)): str(col) for col in rows.columns}
        column_map: dict[str, str] = {}
        for canonical, aliases in PARAM_SPECS:
            for alias in aliases:
                match = normalized_lookup.get(normalize_name(alias))
                if match is not None:
                    column_map[canonical] = match
                    break

        if not column_map:
            raise ValueError(f"No recognized Battaglia parameter columns in {path}")
        for canonical, source_col in column_map.items():
            numeric_col = pd.to_numeric(rows[source_col], errors="coerce")
            if not np.all(np.isfinite(numeric_col.to_numpy(dtype=float))):
                raise ValueError(f"{path} column {source_col} for {canonical} contains non-finite values")

        tables.append(
            SobolTable(
                path=path.resolve(),
                tag=safe_filename_tag(path.stem),
                rows=rows,
                column_map=column_map,
            )
        )

    x_columns = [canonical for canonical, _ in PARAM_SPECS if canonical in tables[0].column_map]
    for table in tables[1:]:
        missing = [col for col in x_columns if col not in table.column_map]
        if missing:
            raise ValueError(
                f"{table.path} is missing parameter columns used by {tables[0].path}: {missing}"
            )
    return tables, x_columns


def params_for_row(table: SobolTable, row_number: int, x_columns: list[str]) -> dict[str, float]:
    if row_number < 1 or row_number > len(table.rows):
        raise IndexError(
            f"Row {row_number} is outside {table.path}, which has {len(table.rows)} data rows"
        )
    row = table.rows.iloc[row_number - 1]
    return {col: float(row[table.column_map[col]]) for col in x_columns}


def compile_row_matchers(tables: list[SobolTable]):
    matchers = []
    for table in tables:
        pattern = re.compile(rf"sobol_{re.escape(table.tag)}_row0*([0-9]+)", re.IGNORECASE)
        matchers.append((table, pattern))
    generic = re.compile(r"row0*([0-9]+)", re.IGNORECASE)
    return matchers, generic


def locate_row_for_file(path: Path, tables: list[SobolTable]):
    path_text = str(path)
    matchers, generic = compile_row_matchers(tables)
    for table, pattern in matchers:
        match = pattern.search(path_text)
        if match:
            return table, int(match.group(1))

    if len(tables) == 1:
        match = generic.search(path.name)
        if match:
            return tables[0], int(match.group(1))
    return None


def read_profile_file(path: Path):
    import numpy as np

    suffix = path.suffix.lower()
    if suffix in (".npy",):
        return np.asarray(np.load(path), dtype=float).squeeze(), None
    if suffix in (".npz",):
        data = np.load(path)
        for key in ("cl", "cls", "profile", "profiles", "y", "data"):
            if key in data:
                arr = np.asarray(data[key], dtype=float).squeeze()
                ell = np.asarray(data["ell"], dtype=float).squeeze() if "ell" in data else None
                return arr, ell
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=float).squeeze(), None
        raise ValueError(f"Could not choose profile array inside {path}; keys={data.files}")
    if suffix in (".csv", ".txt", ".dat"):
        delimiter = "," if suffix == ".csv" else None
        raw = np.loadtxt(path, delimiter=delimiter)
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return raw[:, 1], raw[:, 0]
        return raw.squeeze(), None
    if suffix in (".fits", ".fit", ".fts"):
        return read_fits_profile(path)

    raise ValueError(f"Unsupported profile file extension for {path}")


def read_fits_profile(path: Path):
    try:
        return read_fits_profile_astropy(path)
    except ModuleNotFoundError as exc:
        if exc.name != "astropy":
            raise
        return read_fits_profile_fitsio(path)


def read_fits_profile_astropy(path: Path):
    import numpy as np
    from astropy.io import fits

    profile_candidates: list[tuple[int, str, Any]] = []
    ell_candidates: list[Any] = []
    preferred_tokens = ("cl", "tt", "power", "profile", "temperature", "y")
    ell_names = {"ell", "l", "multipole"}

    with fits.open(path, memmap=False) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue

            if getattr(data, "dtype", None) is not None and data.dtype.fields:
                for field_name in data.dtype.names or ():
                    arr = np.asarray(data[field_name]).squeeze()
                    if arr.size < 2:
                        continue
                    if not np.issubdtype(arr.dtype, np.number):
                        continue
                    name_norm = normalize_name(field_name)
                    arr = np.asarray(arr, dtype=float).reshape(-1)
                    if name_norm in ell_names:
                        ell_candidates.append(arr)
                        continue
                    priority = 0 if any(token in name_norm for token in preferred_tokens) else 1
                    profile_candidates.append((priority, f"hdu{hdu_index}:{field_name}", arr))
            else:
                arr = np.asarray(data).squeeze()
                if arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                    profile_candidates.append((2, f"hdu{hdu_index}", np.asarray(arr, dtype=float).reshape(-1)))

    if not profile_candidates:
        raise ValueError(f"No numeric profile-like data found in {path}")

    profile_candidates.sort(key=lambda item: (item[0], -item[2].size))
    profile = profile_candidates[0][2]
    ell = ell_candidates[0] if ell_candidates and ell_candidates[0].size == profile.size else None
    return profile, ell


def read_fits_profile_fitsio(path: Path):
    import numpy as np
    import fitsio

    profile_candidates: list[tuple[int, str, Any]] = []
    ell_candidates: list[Any] = []
    preferred_tokens = ("cl", "tt", "power", "profile", "temperature", "y")
    ell_names = {"ell", "l", "multipole"}

    with fitsio.FITS(str(path)) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            try:
                data = hdu.read()
            except Exception:
                continue
            if data is None:
                continue

            if getattr(data, "dtype", None) is not None and data.dtype.fields:
                for field_name in data.dtype.names or ():
                    arr = np.asarray(data[field_name]).squeeze()
                    if arr.size < 2:
                        continue
                    if not np.issubdtype(arr.dtype, np.number):
                        continue
                    name_norm = normalize_name(field_name)
                    arr = np.asarray(arr, dtype=float).reshape(-1)
                    if name_norm in ell_names:
                        ell_candidates.append(arr)
                        continue
                    priority = 0 if any(token in name_norm for token in preferred_tokens) else 1
                    profile_candidates.append((priority, f"hdu{hdu_index}:{field_name}", arr))
            else:
                arr = np.asarray(data).squeeze()
                if arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                    profile_candidates.append((2, f"hdu{hdu_index}", np.asarray(arr, dtype=float).reshape(-1)))

    if not profile_candidates:
        raise ValueError(f"No numeric profile-like data found in {path}")

    profile_candidates.sort(key=lambda item: (item[0], -item[2].size))
    profile = profile_candidates[0][2]
    ell = ell_candidates[0] if ell_candidates and ell_candidates[0].size == profile.size else None
    return profile, ell


def select_ell(profile, ell, ell_min: int, ell_max: int | None):
    import numpy as np

    profile = np.asarray(profile, dtype=float).reshape(-1)
    if ell is None:
        ell = np.arange(profile.size, dtype=float)
    else:
        ell = np.asarray(ell, dtype=float).reshape(-1)
    if ell.size != profile.size:
        raise ValueError(f"ell length {ell.size} does not match profile length {profile.size}")

    mask = ell >= ell_min
    if ell_max is not None:
        mask &= ell <= ell_max
    if not np.any(mask):
        raise ValueError(f"ell selection is empty for ell_min={ell_min}, ell_max={ell_max}")
    selected_ell = ell[mask]
    selected_profile = profile[mask]
    if not np.all(np.isfinite(selected_ell)):
        raise ValueError("selected ell values contain NaN or infinity")
    if not np.all(np.isfinite(selected_profile)):
        raise ValueError("selected C_l/profile values contain NaN or infinity")
    return selected_ell, selected_profile


def compile_optional_regexes(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def path_matches_any(path: Path, patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return False
    text = str(path)
    name = path.name
    return any(pattern.search(text) or pattern.search(name) for pattern in patterns)


def discover_profile_paths(
    dirs: list[str],
    globs: list[str],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> tuple[list[Path], list[tuple[str, str]]]:
    paths: list[Path] = []
    for raw_dir in dirs:
        root = Path(raw_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Profile directory not found: {root}")
        for pattern in globs:
            paths.extend(path for path in root.rglob(pattern) if path.is_file())
    candidates = sorted(set(path.resolve() for path in paths))

    include_regexes = compile_optional_regexes(include_patterns)
    exclude_regexes = compile_optional_regexes(exclude_patterns)
    kept: list[Path] = []
    filtered: list[tuple[str, str]] = []
    for path in candidates:
        if include_regexes and not path_matches_any(path, include_regexes):
            filtered.append((str(path), "did not match include-path regex"))
            continue
        if exclude_regexes and path_matches_any(path, exclude_regexes):
            filtered.append((str(path), "matched exclude-path regex"))
            continue
        kept.append(path)

    return kept, filtered


def load_profile_group(
    label: str,
    dirs: list[str],
    globs: list[str],
    tables: list[SobolTable],
    x_columns: list[str],
    ell_min: int,
    ell_max: int | None,
    key_precision: int,
    include_patterns: list[str],
    exclude_patterns: list[str],
):
    import numpy as np

    paths, filtered = discover_profile_paths(dirs, globs, include_patterns, exclude_patterns)
    print(
        f"{label}: selected {len(paths)} profile candidate files after path filters; "
        f"filtered {len(filtered)}",
        flush=True,
    )
    for path, reason in filtered[:10]:
        print(f"{label}: filtered {path}: {reason}", flush=True)
    if len(filtered) > 10:
        print(f"{label}: ... {len(filtered) - 10} more filtered files", flush=True)

    records_by_key: dict[tuple[str, ...], list[ProfileRecord]] = {}
    skipped: list[tuple[str, str]] = []
    base_ell = None

    for path in paths:
        located = locate_row_for_file(path, tables)
        if located is None:
            skipped.append((str(path), "could not infer Sobol row from filename"))
            continue
        table, row_number = located
        try:
            params = params_for_row(table, row_number, x_columns)
            profile, ell = read_profile_file(path)
            selected_ell, selected_profile = select_ell(profile, ell, ell_min, ell_max)
        except Exception as exc:  # noqa: BLE001 - keep scanning and report all bad files.
            skipped.append((str(path), str(exc)))
            continue

        if base_ell is None:
            base_ell = selected_ell
        elif base_ell.shape != selected_ell.shape or not np.allclose(base_ell, selected_ell):
            raise ValueError(f"{path} has a different ell grid than earlier files")

        key = param_key(params, x_columns, key_precision)
        records_by_key.setdefault(key, []).append(
            ProfileRecord(
                path=path,
                key=key,
                params=params,
                csv_path=table.path,
                csv_row=row_number,
                values=selected_profile,
            )
        )

    matched_count = sum(len(records) for records in records_by_key.values())
    print(
        f"{label}: matched {matched_count} files to {len(records_by_key)} unique parameter points; "
        f"skipped {len(skipped)}",
        flush=True,
    )
    for path, reason in skipped[:10]:
        print(f"{label}: skipped {path}: {reason}", flush=True)
    if len(skipped) > 10:
        print(f"{label}: ... {len(skipped) - 10} more skipped files", flush=True)
    if not records_by_key:
        raise ValueError(f"{label}: no matched profile files")
    return records_by_key, base_ell


def choose_profile_record(records: list[ProfileRecord], label: str, key: tuple[str, ...]) -> ProfileRecord:
    if len(records) == 1:
        return records[0]

    def sort_key(record: ProfileRecord):
        try:
            stat = record.path.stat()
            return (stat.st_mtime, stat.st_size, str(record.path))
        except OSError:
            return (0.0, 0, str(record.path))

    chosen = max(records, key=sort_key)
    print(
        f"{label}: found {len(records)} duplicate files for Sobol row "
        f"{chosen.csv_row} ({chosen.csv_path.name}); using newest {chosen.path.name}",
        flush=True,
    )
    return chosen


def deduplicate_profile_records(records_by_key: dict[tuple[str, ...], list[ProfileRecord]], label: str):
    deduped_by_key: dict[tuple[str, ...], list[ProfileRecord]] = {}
    duplicate_file_count = 0

    for key, records in records_by_key.items():
        by_source: dict[tuple[str, int], list[ProfileRecord]] = {}
        for record in records:
            by_source.setdefault(record.source_id, []).append(record)

        deduped_records: list[ProfileRecord] = []
        for source_records in by_source.values():
            duplicate_file_count += max(0, len(source_records) - 1)
            deduped_records.append(choose_profile_record(source_records, label, key))
        deduped_by_key[key] = deduped_records

    if duplicate_file_count:
        total_after = sum(len(records) for records in deduped_by_key.values())
        print(
            f"{label}: removed {duplicate_file_count} duplicate files from reruns; "
            f"kept {total_after} files for {len(deduped_by_key)} parameter points",
            flush=True,
        )

    return deduped_by_key


def mean_arrays_at_each_ell(cl_values, label: str):
    import numpy as np

    cl_by_realization = np.stack(
        [np.asarray(values, dtype=float).reshape(-1) for values in cl_values],
        axis=0,
    )
    if cl_by_realization.ndim != 2:
        raise ValueError(f"{label}: expected stacked C_l values to be 2D")

    # Rows are realizations/files and columns are ell bins. Averaging axis 0
    # gives the mean C_l at each fixed ell; it does not average over ell.
    return np.mean(cl_by_realization, axis=0)


def mean_cl_at_each_ell(records: list[ProfileRecord], label: str):
    if not records:
        raise ValueError(f"{label}: cannot average an empty realization list")
    return mean_arrays_at_each_ell([record.values for record in records], label)


def combine_realizations(y100_by_key, y102_by_key, x_columns: list[str], ell):
    import numpy as np
    import pandas as pd

    common_keys = sorted(set(y100_by_key).intersection(y102_by_key))
    y100_only_keys = sorted(set(y100_by_key).difference(y102_by_key))
    y102_only_keys = sorted(set(y102_by_key).difference(y100_by_key))
    if not common_keys:
        y100_examples = list(y100_by_key)[:5]
        y102_examples = list(y102_by_key)[:5]
        raise ValueError(
            "No y100/y102 parameter matches. "
            f"First y100 keys={y100_examples}; first y102 keys={y102_examples}"
        )
    if y100_only_keys or y102_only_keys:
        print(
            "Warning: using only matched y100/y102 parameter points. "
            f"y100-only={len(y100_only_keys)}, y102-only={len(y102_only_keys)}",
            flush=True,
        )
        if y100_only_keys:
            print(f"First y100-only keys: {y100_only_keys[:5]}", flush=True)
        if y102_only_keys:
            print(f"First y102-only keys: {y102_only_keys[:5]}", flush=True)

    rows = []
    x_values = []
    y_combined = []
    y100_values = []
    y102_values = []

    for key in common_keys:
        y100_records = y100_by_key[key]
        y102_records = y102_by_key[key]
        params = y100_records[0].params
        y100_mean = mean_cl_at_each_ell(y100_records, "y100")
        y102_mean = mean_cl_at_each_ell(y102_records, "y102")
        combined = mean_arrays_at_each_ell([y100_mean, y102_mean], "y100_y102")

        x_values.append([params[col] for col in x_columns])
        y100_values.append(y100_mean)
        y102_values.append(y102_mean)
        y_combined.append(combined)

        row = {
            "param_key": "|".join(key),
            "y100_n": len(y100_records),
            "y102_n": len(y102_records),
            "y100_files": ";".join(str(record.path) for record in y100_records),
            "y102_files": ";".join(str(record.path) for record in y102_records),
            "sobol_csv": str(y100_records[0].csv_path),
            "sobol_row": y100_records[0].csv_row,
        }
        row.update(params)
        rows.append(row)

    metadata = pd.DataFrame(rows)
    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_combined, dtype=float),
        np.asarray(y100_values, dtype=float),
        np.asarray(y102_values, dtype=float),
        metadata,
        np.asarray(ell, dtype=float),
    )


def transform_targets(y, transform: str, floor: float):
    import numpy as np

    if transform == "none":
        return np.asarray(y, dtype=float)
    if transform == "log10":
        y = np.asarray(y, dtype=float)
        clipped = np.maximum(y, floor)
        return np.log10(clipped)
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_transform_targets(y, transform: str):
    import numpy as np

    if transform == "none":
        return np.asarray(y, dtype=float)
    if transform == "log10":
        return np.power(10.0, np.asarray(y, dtype=float))
    raise ValueError(f"Unsupported target transform: {transform}")


def make_regressor(params: dict[str, Any], seed: int, n_jobs: int):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    name = params["regressor"]
    if name == "krr":
        return KernelRidge(
            alpha=params["krr_alpha"],
            kernel=params["krr_kernel"],
            gamma=params["krr_gamma"],
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=params["mlp_hidden_layer_sizes"],
            activation=params["mlp_activation"],
            alpha=params["mlp_alpha"],
            learning_rate_init=params["mlp_learning_rate_init"],
            batch_size=params["mlp_batch_size"],
            max_iter=params["mlp_max_iter"],
            early_stopping=params["mlp_early_stopping"],
            validation_fraction=params["mlp_validation_fraction"],
            n_iter_no_change=params["mlp_n_iter_no_change"],
            tol=params["mlp_tol"],
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=params["trees_n_estimators"],
            max_features=params["trees_max_features"],
            min_samples_leaf=params["trees_min_samples_leaf"],
            max_depth=params["trees_max_depth"],
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["trees_n_estimators"],
            max_features=params["trees_max_features"],
            min_samples_leaf=params["trees_min_samples_leaf"],
            max_depth=params["trees_max_depth"],
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "gradient_boosting":
        base = GradientBoostingRegressor(
            n_estimators=params["gbr_n_estimators"],
            learning_rate=params["gbr_learning_rate"],
            max_depth=params["gbr_max_depth"],
            min_samples_leaf=params["gbr_min_samples_leaf"],
            subsample=params["gbr_subsample"],
            random_state=seed,
        )
        return MultiOutputRegressor(base, n_jobs=n_jobs)
    if name == "knn":
        return KNeighborsRegressor(
            n_neighbors=params["knn_n_neighbors"],
            weights=params["knn_weights"],
            p=params["knn_p"],
            n_jobs=n_jobs,
        )
    if name == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

        length_scale = params["gpr_length_scale"]
        if params["gpr_kernel"] == "rbf":
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(
                noise_level=params["gpr_noise"]
            )
        else:
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=length_scale,
                nu=params["gpr_matern_nu"],
            ) + WhiteKernel(noise_level=params["gpr_noise"])
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=params["gpr_alpha"],
            normalize_y=True,
            n_restarts_optimizer=params["gpr_restarts"],
            random_state=seed,
        )
    raise ValueError(f"Unknown regressor: {name}")


def suggest_params(trial, args, max_components: int, n_samples: int):
    params: dict[str, Any] = {}
    params["n_pca_components"] = trial.suggest_int("n_pca_components", 2, max_components)
    params["pca_whiten"] = trial.suggest_categorical("pca_whiten", [False, True])

    params["regressor"] = trial.suggest_categorical("regressor", args.regressors)

    if params["regressor"] == "krr":
        params["krr_alpha"] = 10.0 ** trial.suggest_float("krr_log10_alpha", -9.0, 0.0)
        params["krr_gamma"] = 10.0 ** trial.suggest_float("krr_log10_gamma", -4.0, 2.0)
        params["krr_kernel"] = trial.suggest_categorical("krr_kernel", ["rbf", "laplacian"])
    elif params["regressor"] == "mlp":
        depth = trial.suggest_int("mlp_depth", 1, 4)
        width = trial.suggest_categorical("mlp_width", [32, 64, 128, 256])
        batch_size = trial.suggest_categorical("mlp_batch_size", ["auto", 16, 32, 64])
        params["mlp_hidden_layer_sizes"] = tuple(width for _ in range(depth))
        params["mlp_activation"] = trial.suggest_categorical("mlp_activation", ["relu", "tanh"])
        params["mlp_alpha"] = 10.0 ** trial.suggest_float("mlp_log10_alpha", -8.0, -1.0)
        params["mlp_learning_rate_init"] = 10.0 ** trial.suggest_float("mlp_log10_learning_rate_init", -5.0, -2.0)
        params["mlp_batch_size"] = batch_size
        params["mlp_max_iter"] = trial.suggest_int("mlp_max_iter", 1000, 5000, step=1000)
        if n_samples >= 50:
            params["mlp_early_stopping"] = trial.suggest_categorical("mlp_early_stopping", [True, False])
        else:
            params["mlp_early_stopping"] = False
        params["mlp_validation_fraction"] = 0.15
        params["mlp_n_iter_no_change"] = 40
        params["mlp_tol"] = 1.0e-6
    elif params["regressor"] in ("extra_trees", "random_forest"):
        params["trees_n_estimators"] = trial.suggest_int("trees_n_estimators", 200, 1200, step=100)
        params["trees_max_features"] = trial.suggest_float("trees_max_features", 0.35, 1.0)
        params["trees_min_samples_leaf"] = trial.suggest_int("trees_min_samples_leaf", 1, 8)
        max_depth_choice = trial.suggest_int("trees_max_depth", 0, 40)
        params["trees_max_depth"] = None if max_depth_choice == 0 else max_depth_choice
    elif params["regressor"] == "gradient_boosting":
        params["gbr_n_estimators"] = trial.suggest_int("gbr_n_estimators", 80, 800, step=40)
        params["gbr_learning_rate"] = 10.0 ** trial.suggest_float("gbr_log10_learning_rate", -3.0, -0.3)
        params["gbr_max_depth"] = trial.suggest_int("gbr_max_depth", 1, 5)
        params["gbr_min_samples_leaf"] = trial.suggest_int("gbr_min_samples_leaf", 1, 8)
        params["gbr_subsample"] = trial.suggest_float("gbr_subsample", 0.55, 1.0)
    elif params["regressor"] == "knn":
        params["knn_n_neighbors"] = trial.suggest_int("knn_n_neighbors", 2, min(30, n_samples - 1))
        params["knn_weights"] = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        params["knn_p"] = trial.suggest_int("knn_p", 1, 2)
    elif params["regressor"] == "gpr":
        params["gpr_kernel"] = trial.suggest_categorical("gpr_kernel", ["rbf", "matern"])
        params["gpr_length_scale"] = 10.0 ** trial.suggest_float("gpr_log10_length_scale", -2.0, 2.0)
        params["gpr_noise"] = 10.0 ** trial.suggest_float("gpr_log10_noise", -10.0, -3.0)
        params["gpr_alpha"] = 10.0 ** trial.suggest_float("gpr_log10_alpha", -12.0, -5.0)
        params["gpr_matern_nu"] = trial.suggest_categorical("gpr_matern_nu", [1.5, 2.5])
        params["gpr_restarts"] = trial.suggest_int("gpr_restarts", 0, 2)
    return params


def fit_components(x_train, y_train_t, params: dict[str, Any], seed: int, n_jobs: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)

    pca = PCA(
        n_components=params["n_pca_components"],
        whiten=params["pca_whiten"],
        random_state=seed,
    )
    z_train = pca.fit_transform(y_train_t)

    regressor = make_regressor(params, seed, n_jobs)
    regressor.fit(x_train_s, z_train)
    return x_scaler, pca, regressor


def predict_transformed(x, x_scaler, pca, regressor):
    z_pred = regressor.predict(x_scaler.transform(x))
    return pca.inverse_transform(z_pred)


def cv_score(x, y_t, params: dict[str, Any], args):
    import numpy as np
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
    fold_scores = []
    for train_idx, valid_idx in cv.split(x):
        x_scaler, pca, regressor = fit_components(
            x[train_idx],
            y_t[train_idx],
            params,
            args.random_seed,
            args.n_jobs,
        )
        pred_t = predict_transformed(x[valid_idx], x_scaler, pca, regressor)
        fold_scores.append(float(np.sqrt(np.mean((pred_t - y_t[valid_idx]) ** 2))))
    return float(np.mean(fold_scores))


def default_params(max_components: int, regressor: str):
    common = {
        "n_pca_components": min(max_components, 16),
        "pca_whiten": True,
        "regressor": regressor,
    }
    if regressor == "mlp":
        return {
            **common,
            "mlp_hidden_layer_sizes": (128, 128),
            "mlp_activation": "relu",
            "mlp_alpha": 1.0e-4,
            "mlp_learning_rate_init": 1.0e-3,
            "mlp_batch_size": "auto",
            "mlp_max_iter": 3000,
            "mlp_early_stopping": False,
            "mlp_validation_fraction": 0.15,
            "mlp_n_iter_no_change": 40,
            "mlp_tol": 1.0e-6,
        }
    if regressor == "krr":
        return {
            **common,
            "pca_whiten": False,
            "krr_alpha": 1.0e-6,
            "krr_gamma": 1.0,
            "krr_kernel": "rbf",
        }
    if regressor == "gpr":
        return {
            **common,
            "gpr_kernel": "rbf",
            "gpr_length_scale": 1.0,
            "gpr_noise": 1.0e-8,
            "gpr_alpha": 1.0e-10,
            "gpr_matern_nu": 2.5,
            "gpr_restarts": 0,
        }
    if regressor in ("extra_trees", "random_forest"):
        return {
            **common,
            "trees_n_estimators": 400,
            "trees_max_features": 1.0,
            "trees_min_samples_leaf": 1,
            "trees_max_depth": None,
        }
    if regressor == "gradient_boosting":
        return {
            **common,
            "gbr_n_estimators": 200,
            "gbr_learning_rate": 0.05,
            "gbr_max_depth": 2,
            "gbr_min_samples_leaf": 1,
            "gbr_subsample": 0.9,
        }
    if regressor == "knn":
        return {
            **common,
            "knn_n_neighbors": 5,
            "knn_weights": "distance",
            "knn_p": 2,
        }
    raise ValueError(f"No default parameters are defined for regressor={regressor}")


def optimize_hyperparameters(x, y_t, args, max_components: int):
    if args.n_trials <= 0:
        baseline_regressor = "mlp" if "mlp" in args.regressors else args.regressors[0]
        params = default_params(max_components, baseline_regressor)
        score = cv_score(x, y_t, params, args)
        return params, score, None

    import optuna

    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.cv_folds))
    storage = args.optuna_storage or None
    study_name = args.study_name or None
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=args.load_if_exists,
    )

    def objective(trial):
        params = suggest_params(trial, args, max_components, x.shape[0])
        try:
            score = cv_score(x, y_t, params, args)
        except Exception as exc:  # noqa: BLE001 - failed hyperparameters should not kill the cluster job.
            trial.set_user_attr("failure", repr(exc))
            return float("inf")
        if not math.isfinite(score):
            trial.set_user_attr("failure", "non-finite CV score")
            return float("inf")
        return score

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_seconds)
    if len(study.trials) == 0 or not math.isfinite(study.best_value):
        raise RuntimeError(
            "Optuna did not find a finite-scoring emulator. "
            "Inspect optuna_trials.csv or lower the search space with --regressors mlp,krr."
        )
    best_params = suggest_params_from_completed_trial(study.best_trial, args, max_components, x.shape[0])
    return best_params, float(study.best_value), study


def suggest_params_from_completed_trial(trial, args, max_components: int, n_samples: int):
    class FixedTrialView:
        def __init__(self, base_trial):
            self.base_trial = base_trial

        def suggest_int(self, name, low, high, step=1):
            return self.base_trial.params[name]

        def suggest_float(self, name, low, high, step=None, log=False):
            return self.base_trial.params[name]

        def suggest_categorical(self, name, choices):
            return self.base_trial.params[name]

    return suggest_params(FixedTrialView(trial), args, max_components, n_samples)


def holdout_report(x, y, y_t, params: dict[str, Any], args):
    import numpy as np
    from sklearn.model_selection import train_test_split

    if x.shape[0] < 5:
        return {}

    indices = np.arange(x.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    x_scaler, pca, regressor = fit_components(
        x[train_idx],
        y_t[train_idx],
        params,
        args.random_seed,
        args.n_jobs,
    )
    pred_t = predict_transformed(x[test_idx], x_scaler, pca, regressor)
    pred = inverse_transform_targets(pred_t, args.target_transform)
    truth = y[test_idx]
    denom = np.maximum(np.abs(truth), args.target_floor)
    rel = (pred - truth) / denom
    return {
        "holdout_size": int(test_idx.size),
        "holdout_transformed_rmse": float(np.sqrt(np.mean((pred_t - y_t[test_idx]) ** 2))),
        "holdout_relative_rmse": float(np.sqrt(np.mean(rel**2))),
        "holdout_median_abs_relative_error": float(np.median(np.abs(rel))),
    }


def write_outputs(args, x_columns, ell, x, y, y100, y102, metadata, params, cv_best, study, report, artifact):
    import joblib
    import numpy as np

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_npz = output_dir / "combined_battaglia_profiles.npz"
    if args.save_combined_dataset:
        np.savez_compressed(
            combined_npz,
            x=x,
            y_combined=y,
            y100=y100,
            y102=y102,
            ell=ell,
            x_columns=np.asarray(x_columns),
            param_key=metadata["param_key"].astype(str).to_numpy(),
        )
    else:
        combined_npz = None

    metadata_path = output_dir / "combined_battaglia_profiles_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    artifact_path = output_dir / args.artifact_name
    joblib.dump(artifact, artifact_path)

    best_params_path = output_dir / "best_emulator_hyperparameters.json"
    with best_params_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_cv_transformed_rmse": cv_best,
                "best_params": params,
                "holdout_report": report,
                "artifact_path": str(artifact_path),
                "combined_npz": None if combined_npz is None else str(combined_npz),
                "metadata_csv": str(metadata_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    if study is not None:
        trials_path = output_dir / "optuna_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)

    if combined_npz is None:
        print("Skipped writing combined dataset NPZ.", flush=True)
    else:
        print(f"Wrote combined dataset: {combined_npz}", flush=True)
    print(f"Wrote metadata: {metadata_path}", flush=True)
    print(f"Wrote emulator artifact: {artifact_path}", flush=True)
    print(f"Wrote best hyperparameters: {best_params_path}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine y100/y102 Battaglia tSZ profiles and train an Optuna-tuned emulator."
    )
    parser.add_argument("--y100-dirs", nargs="+", required=True, help="y100 profile directories; ':' or ',' separated is also accepted.")
    parser.add_argument("--y102-dirs", nargs="+", required=True, help="y102 profile directories; ':' or ',' separated is also accepted.")
    parser.add_argument("--sobol-csvs", nargs="+", required=True, help="Sobol CSVs used to generate the profile filenames.")
    parser.add_argument("--profile-globs", nargs="+", default=["*tSZ_cl*.fits"], help="Recursive file globs for profile products.")
    parser.add_argument(
        "--include-path-regex",
        nargs="*",
        default=[],
        help="Optional comma-separated regexes; profile paths must match at least one. Use this to select the current Sobol split products.",
    )
    parser.add_argument(
        "--exclude-path-regex",
        nargs="*",
        default=[],
        help="Optional comma-separated regexes for profile paths to reject before row matching.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for combined profiles and emulator artifact.")
    parser.add_argument("--artifact-name", default="battaglia_tsz_emulator.joblib")
    parser.add_argument("--ell-min", type=int, default=2)
    parser.add_argument("--ell-max", type=int, default=None)
    parser.add_argument("--key-precision", type=int, default=12)
    parser.add_argument("--target-transform", choices=("log10", "none"), default="log10")
    parser.add_argument("--target-floor", type=float, default=1.0e-40)
    parser.add_argument("--expected-points", type=int, default=0, help="Require this many matched y100/y102 parameter points; 0 disables the check.")
    parser.add_argument("--no-save-combined-dataset", dest="save_combined_dataset", action="store_false", help="Do not write combined_battaglia_profiles.npz.")
    parser.set_defaults(save_combined_dataset=True)
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-pca-components", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=["mlp,krr"],
        help=(
            "Comma/path-separator separated regressor candidates for Optuna. "
            f"Allowed: {', '.join(ALLOWED_REGRESSORS)}. Default: mlp,krr."
        ),
    )
    parser.add_argument("--include-gpr", action="store_true", help="Append Gaussian-process regression to the Optuna search; slower.")
    parser.add_argument("--optuna-storage", default="", help="Optional Optuna storage URL, e.g. sqlite:////path/study.db")
    parser.add_argument("--study-name", default="")
    parser.add_argument("--load-if-exists", action="store_true")
    args = parser.parse_args(argv)

    args.y100_dirs = split_cli_values(args.y100_dirs)
    args.y102_dirs = split_cli_values(args.y102_dirs)
    args.sobol_csvs = split_cli_values(args.sobol_csvs)
    args.profile_globs = split_cli_values(args.profile_globs)
    args.include_path_regex = split_comma_values(args.include_path_regex)
    args.exclude_path_regex = split_comma_values(args.exclude_path_regex)
    args.regressors = split_cli_values(args.regressors)
    if args.include_gpr and "gpr" not in args.regressors:
        args.regressors.append("gpr")
    unknown_regressors = [name for name in args.regressors if name not in ALLOWED_REGRESSORS]
    if unknown_regressors:
        raise ValueError(
            f"Unknown regressors: {unknown_regressors}. Allowed regressors are {list(ALLOWED_REGRESSORS)}"
        )
    args.regressors = list(dict.fromkeys(args.regressors))
    if not args.regressors:
        raise ValueError("--regressors must contain at least one regressor")

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")
    if args.max_pca_components < 2:
        raise ValueError("--max-pca-components must be at least 2")
    if args.key_precision < 6:
        raise ValueError("--key-precision should be at least 6 significant digits")
    if args.expected_points < 0:
        raise ValueError("--expected-points must be nonnegative")
    return args


def main(argv: list[str]) -> int:
    import numpy as np

    args = parse_args(argv)
    tables, x_columns = load_sobol_tables(args.sobol_csvs)
    print(f"Using Battaglia emulator inputs: {', '.join(x_columns)}", flush=True)
    print(f"Using Sobol CSV tags: {', '.join(table.tag for table in tables)}", flush=True)
    if args.include_path_regex:
        print(f"Using include path regexes: {args.include_path_regex}", flush=True)
    if args.exclude_path_regex:
        print(f"Using exclude path regexes: {args.exclude_path_regex}", flush=True)

    y100_by_key, ell = load_profile_group(
        "y100",
        args.y100_dirs,
        args.profile_globs,
        tables,
        x_columns,
        args.ell_min,
        args.ell_max,
        args.key_precision,
        args.include_path_regex,
        args.exclude_path_regex,
    )
    y100_by_key = deduplicate_profile_records(y100_by_key, "y100")
    y102_by_key, ell_102 = load_profile_group(
        "y102",
        args.y102_dirs,
        args.profile_globs,
        tables,
        x_columns,
        args.ell_min,
        args.ell_max,
        args.key_precision,
        args.include_path_regex,
        args.exclude_path_regex,
    )
    y102_by_key = deduplicate_profile_records(y102_by_key, "y102")
    if ell.shape != ell_102.shape or not np.allclose(ell, ell_102):
        raise ValueError("y100 and y102 ell grids differ")

    x, y, y100, y102, metadata, ell = combine_realizations(y100_by_key, y102_by_key, x_columns, ell)
    print(f"Combined {x.shape[0]} matched y100/y102 parameter points", flush=True)
    print(f"Profile length after ell cut: {y.shape[1]}", flush=True)
    if args.expected_points and x.shape[0] != args.expected_points:
        raise ValueError(
            f"Expected {args.expected_points} matched parameter points, got {x.shape[0]}. "
            "Check the Sobol CSVs and path filters before training."
        )

    if x.shape[0] <= args.cv_folds:
        raise ValueError(f"Need more samples than cv_folds; samples={x.shape[0]}, cv_folds={args.cv_folds}")

    min_train_size = x.shape[0] - math.ceil(x.shape[0] / args.cv_folds)
    max_components = min(args.max_pca_components, y.shape[1], min_train_size - 1)
    if max_components < 2:
        raise ValueError("Not enough training samples/profile dimensions for PCA emulator")

    y_t = transform_targets(y, args.target_transform, args.target_floor)
    params, cv_best, study = optimize_hyperparameters(x, y_t, args, max_components)
    print(f"Best CV transformed RMSE: {cv_best:.6g}", flush=True)
    print(f"Best parameters: {json.dumps(params, sort_keys=True)}", flush=True)

    holdout = holdout_report(x, y, y_t, params, args)
    if holdout:
        print(f"Holdout report: {json.dumps(holdout, sort_keys=True)}", flush=True)

    x_scaler, pca, regressor = fit_components(x, y_t, params, args.random_seed, args.n_jobs)
    artifact = {
        "x_columns": x_columns,
        "ell": ell,
        "x_scaler": x_scaler,
        "pca": pca,
        "regressor": regressor,
        "target_transform": args.target_transform,
        "target_floor": args.target_floor,
        "best_params": params,
        "best_cv_transformed_rmse": cv_best,
        "holdout_report": holdout,
        "training_param_min": dict(zip(x_columns, np.min(x, axis=0).tolist())),
        "training_param_max": dict(zip(x_columns, np.max(x, axis=0).tolist())),
    }
    write_outputs(args, x_columns, ell, x, y, y100, y102, metadata, params, cv_best, study, holdout, artifact)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - top-level cluster logs should show the failure plainly.
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
