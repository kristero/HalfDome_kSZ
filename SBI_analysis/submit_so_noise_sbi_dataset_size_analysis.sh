#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BASE_PBS="$SCRIPT_DIR/run_sbi_for_cluster.pbs"

: "${PYTHON:=python3}"
: "${COMBINED_DATASET_PATH:=$SCRIPT_DIR/data_for_cluster/so_multi_case_delta200_sbi_dataset.npz}"
: "${ELL_MIN:=80}"
: "${ELL_MAX:=7979}"
: "${ELL_SELECTION:=center}"
: "${OBS_SOURCE:=battaglia12}" # battaglia12 or dataset-row
: "${OBS_INDEX:=-1}"
: "${LAST_N_TEST:=100}"
: "${TRAIN_EXCLUDE_LAST_N:=0}"
: "${BATTAGLIA12_DIR:=$PROJECT_ROOT/tSZ_visuals/outputs/so_noise_battaglia12_fiducial_local}"
: "${SBI_CASES:=no_noise,goal_deproj0,baseline_deproj0,goal_deproj2,baseline_deproj2}"
: "${SBI_DATASET_SIZES_ALL:=256,512,1024,2048,4096,8192,16384,32600}"
: "${SBI_NUM_GROUPS:=4}"
: "${SUBMIT_MODE:=groups}" # groups or per_size
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
: "${PYTHON_ENV_SETUP:=}"

: "${PBS_QUEUE:=mini}"
: "${PBS_NCPUS:=26}"
: "${PBS_MEM:=128gb}"
: "${PBS_WALLTIME:=04:00:00}"
: "${PBS_LOG_DIR:=/home/kristero10/logs/SBI_runs}"
: "${TMP_SUBMIT_DIR:=$SCRIPT_DIR/generated_so_noise_dataset_size_pbs}"
: "${DRY_RUN:=0}"

: "${SUBMIT_DIAGNOSTICS:=0}"
: "${DIAG_NUM_POSTERIOR_SAMPLES:=50000}"
: "${DIAG_WALLTIME:=08:00:00}"

ELL_TAG=$("$PYTHON" - "$ELL_MIN" "$ELL_MAX" <<'PY'
import sys

def fmt(value):
    value = float(value)
    if value.is_integer():
        text = str(int(value))
    else:
        text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")

print(f"ell{fmt(sys.argv[1])}_{fmt(sys.argv[2])}")
PY
)

: "${CASE_DATASET_DIR:=$SCRIPT_DIR/data_for_cluster/so_noise_sbi_cases_${ELL_TAG}_${OBS_SOURCE}}"
SCALE_TAG=$(printf "%s" "$SBI_X_RESCALE_MODE" | tr -c 'A-Za-z0-9_' '_')
: "${OUTPUT_ROOT:=$SCRIPT_DIR/outputs/cluster_outputs/SBI_SO_noise_dataset_size_${ELL_TAG}_${OBS_SOURCE}_${SCALE_TAG}}"

if [ ! -f "$BASE_PBS" ]; then
  echo "Base PBS script not found: $BASE_PBS" >&2
  exit 2
fi
if [ ! -f "$COMBINED_DATASET_PATH" ]; then
  echo "Combined dataset not found: $COMBINED_DATASET_PATH" >&2
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

mkdir -p "$CASE_DATASET_DIR" "$OUTPUT_ROOT" "$TMP_SUBMIT_DIR" "$PBS_LOG_DIR"

echo "Preparing case-specific SBI datasets..."
"$PYTHON" "$SCRIPT_DIR/prepare_so_sbi_case_datasets.py" \
  --combined-dataset "$COMBINED_DATASET_PATH" \
  --output-dir "$CASE_DATASET_DIR" \
  --cases ${SBI_CASES//,/ } \
  --ell-min "$ELL_MIN" \
  --ell-max "$ELL_MAX" \
  --ell-selection "$ELL_SELECTION" \
  --obs-source "$OBS_SOURCE" \
  --battaglia12-dir "$BATTAGLIA12_DIR" \
  --obs-index "$OBS_INDEX" \
  --test-last-n "$LAST_N_TEST"

CASE_INDEX_JSON="$CASE_DATASET_DIR/case_dataset_index.json"
if [ ! -f "$CASE_INDEX_JSON" ]; then
  echo "Case dataset index was not written: $CASE_INDEX_JSON" >&2
  exit 2
fi

read -r -a CASE_ARRAY <<< "${SBI_CASES//,/ }"
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

echo "Output root: $OUTPUT_ROOT"
echo "Case dataset dir: $CASE_DATASET_DIR"
echo "Observation source: $OBS_SOURCE"
echo "Battaglia12 profile dir: $BATTAGLIA12_DIR"
echo "Exclude last N rows from training: $TRAIN_EXCLUDE_LAST_N"
echo "X rescale mode: $SBI_X_RESCALE_MODE"
echo "Dataset-size groups:"
echo "  group1: $SBI_DATASET_SIZE_GROUP_1  total_rows=$SBI_DATASET_SIZE_GROUP_1_LOAD"
echo "  group2: $SBI_DATASET_SIZE_GROUP_2  total_rows=$SBI_DATASET_SIZE_GROUP_2_LOAD"
echo "  group3: $SBI_DATASET_SIZE_GROUP_3  total_rows=$SBI_DATASET_SIZE_GROUP_3_LOAD"
echo "  group4: $SBI_DATASET_SIZE_GROUP_4  total_rows=$SBI_DATASET_SIZE_GROUP_4_LOAD"
echo "Note: N=32600 is a single large training run, so exact wall-time balancing is not possible."

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
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN qsub $pbs_path"
  else
    qsub "$pbs_path"
  fi
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

for case_name in "${CASE_ARRAY[@]}"; do
  prepared_dataset_path=$(get_case_dataset_path "$case_name")
  if [ ! -f "$prepared_dataset_path" ]; then
    echo "Prepared case dataset not found: $prepared_dataset_path" >&2
    exit 2
  fi

  if [ "$SUBMIT_MODE" = "groups" ]; then
    for group_id in 1 2 3 4; do
      group_var="SBI_DATASET_SIZE_GROUP_${group_id}"
      group_sizes=${!group_var}
      pbs_path="$TMP_SUBMIT_DIR/so_sbi_${case_name}_group${group_id}.pbs"
      output_dir="$OUTPUT_ROOT/$case_name/group${group_id}"
      job_name=$(printf "sbi_%s_g%s" "$case_name" "$group_id" | cut -c1-14)
      make_training_pbs "$case_name" "$group_id" "$group_sizes" "$prepared_dataset_path" "$output_dir" "$pbs_path" "$job_name"
      echo "Submitting $case_name group $group_id: sizes=$group_sizes"
      job_id=$(submit_pbs "$pbs_path")
      if [ "$DRY_RUN" = "1" ]; then
        echo "$job_id"
      else
        job_ids+=("$job_id")
      fi
    done
  else
    for size in "${SIZE_ARRAY[@]}"; do
      pbs_path="$TMP_SUBMIT_DIR/so_sbi_${case_name}_N${size}.pbs"
      output_dir="$OUTPUT_ROOT/$case_name/job_N${size}"
      job_name=$(printf "sbi_%s_%s" "$case_name" "$size" | cut -c1-14)
      make_training_pbs "$case_name" "1" "$size" "$prepared_dataset_path" "$output_dir" "$pbs_path" "$job_name"
      echo "Submitting $case_name N=$size"
      job_id=$(submit_pbs "$pbs_path")
      if [ "$DRY_RUN" = "1" ]; then
        echo "$job_id"
      else
        job_ids+=("$job_id")
      fi
    done
  fi
done

if [ "$SUBMIT_DIAGNOSTICS" = "1" ]; then
  diag_pbs="$TMP_SUBMIT_DIR/so_sbi_dataset_size_diagnostics.pbs"
  cat > "$diag_pbs" <<EOF
#PBS -N so_sbi_diag
#PBS -q $PBS_QUEUE
#PBS -l select=1:ncpus=$PBS_NCPUS:mpiprocs=1:mem=$PBS_MEM
#PBS -l walltime=$DIAG_WALLTIME
#PBS -o $PBS_LOG_DIR
#PBS -e $PBS_LOG_DIR
#PBS -S /bin/bash

set -euo pipefail

export PYTHON="$PYTHON"
export PYTHON_ENV_SETUP="$PYTHON_ENV_SETUP"

cd "\${PBS_O_WORKDIR:-$PROJECT_ROOT}"
if [ -n "\$PYTHON_ENV_SETUP" ]; then
  eval "\$PYTHON_ENV_SETUP"
fi

"\$PYTHON" "$SCRIPT_DIR/analyze_so_sbi_dataset_size.py" \
  --run-root "$OUTPUT_ROOT" \
  --case-dataset-dir "$CASE_DATASET_DIR" \
  --case-index-json "$CASE_INDEX_JSON" \
  --output-dir "$OUTPUT_ROOT/diagnostics_battaglia12" \
  --cases ${SBI_CASES//,/ } \
  --dataset-sizes "$SBI_DATASET_SIZES_ALL" \
  --analysis-target obs \
  --analysis-tag battaglia12 \
  --last-n-test "$LAST_N_TEST" \
  --num-posterior-samples "$DIAG_NUM_POSTERIOR_SAMPLES"
EOF

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN qsub diagnostics: $diag_pbs"
  elif [ "${#job_ids[@]}" -gt 0 ]; then
    dep=$(IFS=:; echo "${job_ids[*]}")
    echo "Submitting diagnostics after training jobs."
    qsub -W "depend=afterok:$dep" "$diag_pbs"
  else
    echo "Submitting diagnostics without dependency."
    qsub "$diag_pbs"
  fi
else
  echo ""
  echo "Training jobs submitted. After they finish, run diagnostics with:"
  echo "$PYTHON $SCRIPT_DIR/analyze_so_sbi_dataset_size.py \\"
  echo "  --run-root \"$OUTPUT_ROOT\" \\"
  echo "  --case-dataset-dir \"$CASE_DATASET_DIR\" \\"
  echo "  --case-index-json \"$CASE_INDEX_JSON\" \\"
  echo "  --output-dir \"$OUTPUT_ROOT/diagnostics_battaglia12\" \\"
  echo "  --cases ${SBI_CASES//,/ } \\"
  echo "  --dataset-sizes \"$SBI_DATASET_SIZES_ALL\" \\"
  echo "  --analysis-target obs \\"
  echo "  --analysis-tag battaglia12 \\"
  echo "  --last-n-test \"$LAST_N_TEST\" \\"
  echo "  --num-posterior-samples \"$DIAG_NUM_POSTERIOR_SAMPLES\""
fi
