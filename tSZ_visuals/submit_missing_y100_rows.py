#!/usr/bin/env python3
"""
Submit y100 repair jobs for missing Sobol rows, then optionally recombine.

The input is the missing-row CSV written by combine_sbi_dataset_y100.py. Each
row is recomputed with run_full_map_sobol_parallel.pbs using SOBOL_ROW_LIST, and
the combine PBS can be submitted with an afterok dependency on all repair jobs.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_MISSING_CSV = (
    "/lustre/work/kristero10/tSZ_data/sbi_battaglia_y100_16384_minus4096_fullcl/"
    "sbi_battaglia_y100_16384_minus4096_fullcl_missing_rows.csv"
)


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def split_chunks(values: list[int], chunk_size: int):
    if chunk_size < 1:
        raise ValueError("--rows-per-job must be >= 1")
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def read_missing_rows(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing-row CSV not found: {path}")

    grouped: dict[tuple[int, str], set[int]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sobol_split", "sobol_row", "sobol_csv"}
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                split = int(row["sobol_split"])
                sobol_row = int(row["sobol_row"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Bad split/row value in {path} at line {line_number}: {row}") from exc
            sobol_csv = os.path.expanduser(str(row["sobol_csv"]).strip())
            grouped[(split, sobol_csv)].add(sobol_row)

    return {
        key: sorted(rows)
        for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    }


def qsub_var_list(values: dict[str, str]) -> str:
    bad = {key: value for key, value in values.items() if "," in str(value)}
    if bad:
        formatted = ", ".join(f"{key}={value}" for key, value in bad.items())
        raise ValueError(f"qsub -v values cannot contain commas: {formatted}")
    return ",".join(f"{key}={value}" for key, value in values.items())


def run_command(cmd: list[str], dry_run: bool) -> str:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(printable, flush=True)
    if dry_run:
        return ""

    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr, flush=True)
    stdout = result.stdout.strip()
    if stdout:
        print(stdout, flush=True)
    return stdout.split()[0] if stdout else ""


def build_row_job_vars(args: argparse.Namespace, split: int, sobol_csv: str, rows: list[int]) -> dict[str, str]:
    return {
        "LIGHTCONE_ID": "100",
        "SOBOL_SPLIT": str(split),
        "SOBOL_CSV": sobol_csv,
        "SOBOL_ROW_LIST": ":".join(str(row) for row in rows),
        "HALFDOME_PATH": f"{args.halfdome_base_dir}/lightcone_100.hdf5",
        "OUTPUT_DIR": f"{args.output_base_dir}/y100",
        "CACHE_DIR": args.cache_dir,
        "LOG_DIR": args.log_dir,
        "JOB_SET_TAG": args.job_set_tag,
        "SIMULATION_NAME": "halfdome_lightcone_100",
        "NSIDE": str(args.nside),
        "THREADS_PER_TASK": str(args.threads_per_task),
        "MAX_PARALLEL": str(args.max_parallel),
        "MODEL_EXISTS": args.model_exists,
        "REUSE_EXISTING_CACHE": args.reuse_existing_cache,
        "SEPARATE_INTERPOLATOR_STEP": args.separate_interpolator_step,
        "CL_LMAX": str(args.cl_lmax),
        "SKIP_EXISTING_OUTPUTS": args.skip_existing_outputs,
        "SKIP_EXISTING_ANY_RUN_INSTANCE": args.skip_existing_any_run_instance,
        "SKIP_INVALID_BATTAGLIA_ROWS": args.skip_invalid_battaglia_rows,
        "CONTINUE_ON_ROW_ERROR": args.continue_on_row_error,
        "SHORT_LOGS": args.short_logs,
        "PRINT_RUNTIME_ENVIRONMENT": args.print_runtime_environment,
    }


def submit_repair_jobs(args: argparse.Namespace, grouped_rows: dict[tuple[int, str], list[int]]) -> list[str]:
    job_ids = []
    for (split, sobol_csv), rows in grouped_rows.items():
        for chunk in split_chunks(rows, args.rows_per_job):
            first = chunk[0]
            last = chunk[-1]
            suffix = f"s{split}_r{first}" if first == last else f"s{split}_r{first}to{last}"
            job_name = f"{args.job_name_prefix}_{suffix}"
            vars_text = qsub_var_list(build_row_job_vars(args, split, sobol_csv, chunk))
            cmd = [args.qsub, "-N", job_name, "-v", vars_text, args.row_pbs_script]
            job_id = run_command(cmd, args.dry_run)
            if job_id:
                job_ids.append(job_id)
            elif args.dry_run:
                job_ids.append(f"DRYRUN{len(job_ids) + 1}")
    return job_ids


def submit_combine_job(args: argparse.Namespace, dependencies: list[str]) -> str:
    cmd = [args.qsub, "-N", args.combine_job_name]
    if dependencies:
        cmd.extend(["-W", f"depend=afterok:{':'.join(dependencies)}"])
    cmd.append(args.combine_pbs_script)
    return run_command(cmd, args.dry_run)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit y100 jobs for missing Sobol rows and optionally submit a dependent combine job."
    )
    parser.add_argument("--missing-csv", default=DEFAULT_MISSING_CSV)
    parser.add_argument("--row-pbs-script", default="run_full_map_sobol_parallel.pbs")
    parser.add_argument("--combine-pbs-script", default="run_combine_sbi_dataset_y100_16384_minus4096_fullcl.pbs")
    parser.add_argument("--qsub", default="qsub")
    parser.add_argument("--rows-per-job", type=int, default=1)
    parser.add_argument("--submit-combine", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job-name-prefix", default="tSZ_y100_miss")
    parser.add_argument("--combine-job-name", default="tSZ_sbi_y100_s16384_extra")

    parser.add_argument("--log-dir", default="/home/kristero10/logs/tSZ_baryon_run")
    parser.add_argument("--halfdome-base-dir", default="/lustre/work/Globus-lt/halfdome/full_res/halos")
    parser.add_argument("--output-base-dir", default="/lustre/work/kristero10/tSZ_data")
    parser.add_argument("--cache-dir", default="/lustre/work/kristero10/tSZ_data/cache")
    parser.add_argument("--nside", type=int, default=4096)
    parser.add_argument("--threads-per-task", type=int, default=25)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--job-set-tag", default="s16384repair")
    parser.add_argument("--model-exists", default="false")
    parser.add_argument("--reuse-existing-cache", default="true")
    parser.add_argument("--separate-interpolator-step", default="false")
    parser.add_argument("--cl-lmax", default="-1")
    parser.add_argument("--skip-existing-outputs", default="false")
    parser.add_argument("--skip-existing-any-run-instance", default="true")
    parser.add_argument("--skip-invalid-battaglia-rows", default="false")
    parser.add_argument("--continue-on-row-error", default="false")
    parser.add_argument("--short-logs", default="true")
    parser.add_argument("--print-runtime-environment", default="false")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    grouped_rows = read_missing_rows(Path(args.missing_csv).expanduser())
    total_rows = sum(len(rows) for rows in grouped_rows.values())

    print(f"Missing rows to repair: {total_rows}", flush=True)
    print(f"Split groups: {len(grouped_rows)}", flush=True)
    if total_rows == 0:
        print("No missing rows found.", flush=True)
        if args.submit_combine:
            submit_combine_job(args, [])
        return 0

    job_ids = submit_repair_jobs(args, grouped_rows)
    print(f"Submitted repair jobs: {len(job_ids)}", flush=True)
    if args.submit_combine:
        combine_id = submit_combine_job(args, job_ids)
        if combine_id:
            print(f"Submitted dependent combine job: {combine_id}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
