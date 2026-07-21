#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BASE_PBS="$SCRIPT_DIR/run_sbi_for_cluster.pbs"

: "${PYTHON:=python3}"
: "${ELL_MIN:=80}"
: "${ELL_MAX:=7979}"
: "${OBS_SOURCE:=dataset_row_sobolrow}"
: "${LAST_N_TEST:=500}"
: "${TRAIN_EXCLUDE_LAST_N:=$LAST_N_TEST}"
: "${SBI_DATASET_SIZES_ALL:=256,512,1024,2048,4096,8192,16384,32768,49152,65536,81920,98304,114688,131072,163840,196608,229376,262144,327680,393216,458752,524288}"
: "${SBI_NUM_GROUPS:=4}"
: "${SUBMIT_MODE:=per_size}" # groups or per_size
: "${MAX_CONCURRENT_JOBS:=5}"
: "${SBI_DATASET_ORDER:=shuffle}"
: "${SBI_SEED:=42}"
: "${SBI_DENSITY_ESTIMATOR:=maf}"
: "${SBI_STOP_AFTER_EPOCHS:=60}"
: "${SBI_NUM_POSTERIOR_SAMPLES:=100000}"
: "${SBI_DEVICE:=cpu}"
: "${SBI_GAUSSIAN_BEAM_FWHM_ARCMIN:=0.0}"
: "${SBI_GAUSSIAN_BEAM_MODE:=off}"
: "${SBI_X_RESCALE_MODE:=asinh}"
: "${SBI_X_RESCALE_EPS:=1e-30}"
: "${SBI_X_STANDARDIZE_EPS:=1e-8}"
: "${SBI_CASES:=masked_no_noise,masked_baseline_noise_cross_deproj0}"
: "${PYTHON_ENV_SETUP:=}"

: "${PBS_QUEUE:=mini}"
: "${PBS_NCPUS:=26}"
: "${PBS_MEM:=128gb}"
: "${PBS_WALLTIME:=24:00:00}"
: "${PBS_LOG_DIR:=/home/kristero10/logs/SBI_runs}"
: "${TMP_SUBMIT_DIR:=$SCRIPT_DIR/generated_adrian_dataset_size_pbs}"
: "${DRY_RUN:=0}"

ELL_TAG=$("$PYTHON" - "$ELL_MIN" "$ELL_MAX" <<'PY'
import sys

def fmt(value):
    value = float(value)
    text = str(int(value)) if value.is_integer() else f"{value:g}"
    return text.replace("-", "m").replace(".", "p")

print(f"ell{fmt(sys.argv[1])}_{fmt(sys.argv[2])}")
PY
)

: "${CASE_DATASET_DIR:=$SCRIPT_DIR/data_for_cluster/adrian_so_sbi_cases_${ELL_TAG}_${OBS_SOURCE}}"
SCALE_TAG=$(printf "%s" "$SBI_X_RESCALE_MODE" | tr -c 'A-Za-z0-9_' '_')
: "${OUTPUT_ROOT:=$SCRIPT_DIR/outputs/cluster_outputs/SBI_Adrian_SO_dataset_size_${ELL_TAG}_${OBS_SOURCE}_${SCALE_TAG}}"

CASE_INDEX_JSON="$CASE_DATASET_DIR/case_dataset_index.json"

if [ ! -f "$BASE_PBS" ]; then
  echo "Base PBS script not found: $BASE_PBS" >&2
  exit 2
fi
if [ ! -f "$CASE_INDEX_JSON" ]; then
  echo "Prepared case dataset index not found: $CASE_INDEX_JSON" >&2
  echo "First run the preparation job, for example:" >&2
  echo "  qsub SBI_analysis/run_prepare_adrian_so_sbi_cases.pbs" >&2
  echo "or set CASE_DATASET_DIR to the directory containing case_dataset_index.json." >&2
  exit 2
fi
if [ "$SBI_NUM_GROUPS" != "4" ]; then
  echo "This helper currently creates 4 grouped jobs per case; set SBI_NUM_GROUPS=4." >&2
  exit 2
fi
if [ "$SUBMIT_MODE" != "groups" ] && [ "$SUBMIT_MODE" != "per_size" ]; then
  echo "SUBMIT_MODE must be groups or per_size; got $SUBMIT_MODE" >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_CONCURRENT_JOBS" -lt 1 ]; then
  echo "MAX_CONCURRENT_JOBS must be a positive integer; got $MAX_CONCURRENT_JOBS" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$TMP_SUBMIT_DIR" "$PBS_LOG_DIR"

CASE_LIST=$("$PYTHON" - "$CASE_INDEX_JSON" "${SBI_CASES:-}" <<'PY'
import json
import sys
from pathlib import Path

index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
available = list(index.get("cases", {}).keys())
requested_raw = sys.argv[2].strip() if len(sys.argv) > 2 else ""
if requested_raw:
    requested = [part for part in requested_raw.replace(",", " ").split() if part]
    missing = [case for case in requested if case not in available]
    if missing:
        raise SystemExit(f"Requested cases not prepared: {missing}. Available: {available}")
    print(" ".join(requested))
else:
    print(" ".join(available))
PY
)
read -r -a CASE_ARRAY <<< "$CASE_LIST"

if [ "${#CASE_ARRAY[@]}" -eq 0 ]; then
  echo "No prepared cases found in $CASE_INDEX_JSON" >&2
  exit 2
fi

SBI_DATASET_SIZES_ALL=$("$PYTHON" - "$CASE_INDEX_JSON" "$CASE_LIST" "$SBI_DATASET_SIZES_ALL" "$TRAIN_EXCLUDE_LAST_N" <<'PY'
import json
import math
import sys
from pathlib import Path


def parse_size(value):
    raw = str(value).strip().lower().replace("_", "")
    if not raw:
        raise ValueError("empty dataset size")
    if raw.endswith("k"):
        number = float(raw[:-1]) * 1000.0
    else:
        number = float(raw)
    rounded = int(round(number))
    if rounded <= 0 or not math.isclose(number, rounded):
        raise ValueError(f"dataset size must be a positive integer count: {value!r}")
    return rounded


index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cases = [part for part in sys.argv[2].split() if part]
requested = [part for part in sys.argv[3].replace(";", ",").replace(" ", ",").split(",") if part]
exclude_last_n = int(sys.argv[4])
if exclude_last_n < 0:
    raise SystemExit("TRAIN_EXCLUDE_LAST_N must be non-negative")
rows = [int(index["cases"][case]["n_rows"]) for case in cases]
max_train = min(rows) - exclude_last_n
if max_train <= 0:
    raise SystemExit(
        f"TRAIN_EXCLUDE_LAST_N={exclude_last_n} leaves no training rows for cases={cases} with row counts={rows}"
    )

sizes = []
adjusted = []
for part in requested:
    size = parse_size(part)
    if size > max_train:
        adjusted.append((size, max_train))
        size = max_train
    if size not in sizes:
        sizes.append(size)

if adjusted:
    unique_adjusted = sorted(set(adjusted))
    print(
        "Adjusted oversized dataset sizes for the held-out diagnostic rows: "
        + ", ".join(f"{old}->{new}" for old, new in unique_adjusted),
        file=sys.stderr,
    )
print(",".join(str(size) for size in sizes))
PY
)
read -r -a SIZE_ARRAY <<< "${SBI_DATASET_SIZES_ALL//,/ }"

BALANCED_GROUP_ASSIGNMENTS=$("$PYTHON" - "$SBI_DATASET_SIZES_ALL" <<'PY'
import math
import shlex
import sys

def parse_size(value):
    raw = str(value).strip().lower().replace("_", "")
    if not raw:
        raise ValueError("empty dataset size")
    if raw.endswith("k"):
        number = float(raw[:-1]) * 1000.0
    else:
        number = float(raw)
    rounded = int(round(number))
    if rounded <= 0 or not math.isclose(number, rounded):
        raise ValueError(f"dataset size must be a positive integer count: {value!r}")
    return rounded

parts = [part for part in sys.argv[1].replace(";", ",").replace(" ", ",").split(",") if part]
items = [(parse_size(part), str(parse_size(part))) for part in parts]
groups = [{"sizes": [], "load": 0} for _ in range(4)]
for size, label in sorted(items, key=lambda item: item[0], reverse=True):
    group = min(groups, key=lambda candidate: (candidate["load"], len(candidate["sizes"])))
    group["sizes"].append(label)
    group["load"] += size

for idx, group in enumerate(groups, start=1):
    print(f"SBI_DATASET_SIZE_GROUP_{idx}={shlex.quote(','.join(group['sizes']))}")
    print(f"SBI_DATASET_SIZE_GROUP_{idx}_LOAD={group['load']}")
PY
)
eval "$BALANCED_GROUP_ASSIGNMENTS"

echo "Prepared case dataset dir: $CASE_DATASET_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Cases: ${CASE_ARRAY[*]}"
echo "Submit mode: $SUBMIT_MODE"
echo "Max concurrent submitted jobs via dependency lanes: $MAX_CONCURRENT_JOBS"
echo "Exclude last N rows from training: $TRAIN_EXCLUDE_LAST_N"
echo "X rescale mode: $SBI_X_RESCALE_MODE"
echo "Dataset-size groups:"
echo "  group1: $SBI_DATASET_SIZE_GROUP_1  total_rows=$SBI_DATASET_SIZE_GROUP_1_LOAD"
echo "  group2: $SBI_DATASET_SIZE_GROUP_2  total_rows=$SBI_DATASET_SIZE_GROUP_2_LOAD"
echo "  group3: $SBI_DATASET_SIZE_GROUP_3  total_rows=$SBI_DATASET_SIZE_GROUP_3_LOAD"
echo "  group4: $SBI_DATASET_SIZE_GROUP_4  total_rows=$SBI_DATASET_SIZE_GROUP_4_LOAD"

get_case_dataset_path() {
  local case_name=$1
  "$PYTHON" - "$CASE_INDEX_JSON" "$case_name" <<'PY'
import json
import sys
from pathlib import Path

index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(index["cases"][sys.argv[2]]["path"])
PY
}

submit_pbs() {
  local pbs_path=$1
  local dependency=${2:-}
  if [ "$DRY_RUN" = "1" ]; then
    if [ -n "$dependency" ]; then
      echo "DRY_RUN qsub -W depend=afterany:$dependency $pbs_path"
    else
      echo "DRY_RUN qsub $pbs_path"
    fi
  else
    if [ -n "$dependency" ]; then
      qsub -W depend=afterany:"$dependency" "$pbs_path"
    else
      qsub "$pbs_path"
    fi
  fi
}

short_job_case() {
  "$PYTHON" - "$1" <<'PY'
import re
import sys

name = re.sub(r"[^A-Za-z0-9]+", "_", sys.argv[1]).strip("_")
parts = name.split("_")
if len(name) > 8:
    name = "".join(part[:2] for part in parts)[:8]
print(name or "case")
PY
}

make_training_pbs() {
  local case_name=$1
  local group_id=$2
  local group_sizes=$3
  local prepared_dataset_path=$4
  local output_dir=$5
  local pbs_path=$6
  local job_name=$7

  cat > "$pbs_path" <<EOF
#PBS -N $job_name
#PBS -q $PBS_QUEUE
#PBS -l select=1:ncpus=$PBS_NCPUS:mpiprocs=1:mem=$PBS_MEM
#PBS -l walltime=$PBS_WALLTIME
#PBS -o $PBS_LOG_DIR
#PBS -e $PBS_LOG_DIR
#PBS -S /bin/bash

set -euo pipefail

export PYTHON="$PYTHON"
export PYTHON_ENV_SETUP="$PYTHON_ENV_SETUP"
export PREPARED_DATASET_PATH="$prepared_dataset_path"
export SBI_OUTPUT_DIR="$output_dir"
export SBI_GROUP_ID="$group_id"
export SBI_DATASET_SIZE_GROUP_1="$group_sizes"
export SBI_DATASET_SIZE_GROUP_2="$group_sizes"
export SBI_DATASET_SIZE_GROUP_3="$group_sizes"
export SBI_DATASET_SIZE_GROUP_4="$group_sizes"
export SBI_DATASET_ORDER="$SBI_DATASET_ORDER"
export SBI_EXCLUDE_LAST_N_FROM_TRAINING="$TRAIN_EXCLUDE_LAST_N"
export SBI_SEED="$SBI_SEED"
export SBI_DENSITY_ESTIMATOR="$SBI_DENSITY_ESTIMATOR"
export SBI_STOP_AFTER_EPOCHS="$SBI_STOP_AFTER_EPOCHS"
export SBI_NUM_POSTERIOR_SAMPLES="$SBI_NUM_POSTERIOR_SAMPLES"
export SBI_DEVICE="$SBI_DEVICE"
export SBI_GAUSSIAN_BEAM_FWHM_ARCMIN="$SBI_GAUSSIAN_BEAM_FWHM_ARCMIN"
export SBI_GAUSSIAN_BEAM_MODE="$SBI_GAUSSIAN_BEAM_MODE"
export SBI_X_RESCALE_MODE="$SBI_X_RESCALE_MODE"
export SBI_X_RESCALE_EPS="$SBI_X_RESCALE_EPS"
export SBI_X_STANDARDIZE_EPS="$SBI_X_STANDARDIZE_EPS"

cd "\${PBS_O_WORKDIR:-$PROJECT_ROOT}"
bash "$BASE_PBS"
EOF
}

job_ids=()
lane_last_job_ids=()
for ((lane=0; lane<MAX_CONCURRENT_JOBS; lane++)); do
  lane_last_job_ids[$lane]=""
done
submit_count=0

submit_scheduled_pbs() {
  local pbs_path=$1
  local lane=$((submit_count % MAX_CONCURRENT_JOBS))
  local dependency=${lane_last_job_ids[$lane]}
  local job_id

  if [ -n "$dependency" ]; then
    echo "  dependency lane $((lane + 1)) afterany:$dependency"
  else
    echo "  dependency lane $((lane + 1)) starts immediately"
  fi

  job_id=$(submit_pbs "$pbs_path" "$dependency")
  if [ "$DRY_RUN" = "1" ]; then
    echo "$job_id"
    lane_last_job_ids[$lane]="DRYRUN_${submit_count}"
  else
    job_ids+=("$job_id")
    lane_last_job_ids[$lane]="$job_id"
  fi
  submit_count=$((submit_count + 1))
}

for case_name in "${CASE_ARRAY[@]}"; do
  prepared_dataset_path=$(get_case_dataset_path "$case_name")
  if [ ! -f "$prepared_dataset_path" ]; then
    echo "Prepared case dataset not found: $prepared_dataset_path" >&2
    exit 2
  fi

  case_short=$(short_job_case "$case_name")

  if [ "$SUBMIT_MODE" = "groups" ]; then
    for group_id in 1 2 3 4; do
      group_var="SBI_DATASET_SIZE_GROUP_${group_id}"
      group_sizes=${!group_var}
      pbs_path="$TMP_SUBMIT_DIR/adrian_sbi_${case_name}_group${group_id}.pbs"
      output_dir="$OUTPUT_ROOT/$case_name/group${group_id}"
      job_name=$(printf "ad%s_g%s" "$case_short" "$group_id" | cut -c1-14)
      make_training_pbs "$case_name" "$group_id" "$group_sizes" "$prepared_dataset_path" "$output_dir" "$pbs_path" "$job_name"
      echo "Submitting $case_name group $group_id: sizes=$group_sizes"
      submit_scheduled_pbs "$pbs_path"
    done
  else
    for size in "${SIZE_ARRAY[@]}"; do
      pbs_path="$TMP_SUBMIT_DIR/adrian_sbi_${case_name}_N${size}.pbs"
      output_dir="$OUTPUT_ROOT/$case_name/job_N${size}"
      job_name=$(printf "ad%s_%s" "$case_short" "$size" | cut -c1-14)
      make_training_pbs "$case_name" "1" "$size" "$prepared_dataset_path" "$output_dir" "$pbs_path" "$job_name"
      echo "Submitting $case_name N=$size"
      submit_scheduled_pbs "$pbs_path"
    done
  fi
done

if [ "$DRY_RUN" != "1" ]; then
  echo "Submitted jobs:"
  printf '  %s\n' "${job_ids[@]}"
fi
