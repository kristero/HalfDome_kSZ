#!/bin/bash
# WSL/Linux: same experiment as PBS, with three sequential model jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${PYTHON:=python3}"
: "${COMPRESSION_ROOT:=${PROJECT_ROOT}/SBI_analysis/outputs/local_so_9param_compression}"
: "${LOCAL_THREADS:=4}"
STAGE="${1:-all}"
METHOD="${2:-all}"
case "${STAGE}" in all|prepare|run|summarize) ;; *) echo "Stage must be all, prepare, run, or summarize" >&2; exit 2 ;; esac
case "${METHOD}" in all|bins40|pca|moped) ;; *) echo "Method must be all, bins40, pca, or moped" >&2; exit 2 ;; esac
[[ "${LOCAL_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "LOCAL_THREADS must be positive" >&2; exit 2; }
cd "${PROJECT_ROOT}"
mkdir -p "${COMPRESSION_ROOT}/logs"
exec > >(tee -a "${COMPRESSION_ROOT}/logs/local_${STAGE}_$(date +%Y%m%dT%H%M%S).log") 2>&1

# Use this interpreter's C++ runtime, avoiding WSL's older system libstdc++.
ENV_PREFIX="$("${PYTHON}" -c 'import sys; print(sys.prefix)')"
if [[ -f "${ENV_PREFIX}/lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export OMP_NUM_THREADS="${LOCAL_THREADS}"
export OPENBLAS_NUM_THREADS="${LOCAL_THREADS}"
export MKL_NUM_THREADS="${LOCAL_THREADS}"
export NUMEXPR_NUM_THREADS="${LOCAL_THREADS}"
export TORCH_NUM_THREADS="${LOCAL_THREADS}"
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
DATA_DIR="${PROJECT_ROOT}/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow"
: "${PREPARED_DATASET:=${DATA_DIR}/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${CLEAN_DATASET:=${DATA_DIR}/so_masked_no_noise_ell80_7979_sbi_run.npz}"
SCRIPT="${SCRIPT_DIR}/run_so_sbi_compression_comparison.py"
echo "Python: ${PYTHON}; threads: ${LOCAL_THREADS}; output: ${COMPRESSION_ROOT}"

if [[ "${STAGE}" == all || "${STAGE}" == prepare ]]; then
  "${PYTHON}" "${SCRIPT}" prepare --output-root "${COMPRESSION_ROOT}" \
    --dataset "${PREPARED_DATASET}" --clean-dataset "${CLEAN_DATASET}" \
    --holdout-last-n "${HOLDOUT_LAST_N:-500}" --pca-components "${PCA_COMPONENTS:-9}" \
    --moped-local-n "${MOPED_LOCAL_N:-20000}" \
    --covariance-shrinkage "${COVARIANCE_SHRINKAGE:-0.05}" --moped-rcond "${MOPED_RCOND:-1e-6}" \
    --hidden-features "${HIDDEN_FEATURES:-64}" --num-transforms "${NUM_TRANSFORMS:-6}" \
    --training-batch-size "${TRAINING_BATCH_SIZE:-1024}" \
    --stop-after-epochs "${STOP_AFTER_EPOCHS:-20}" --max-num-epochs "${MAX_NUM_EPOCHS:-200}" \
    --validation-fraction "${VALIDATION_FRACTION:-0.1}" \
    --posterior-samples "${POSTERIOR_SAMPLES:-2000}" \
    --max-proposals "${MAX_PROPOSALS:-200000}" --sampling-seconds "${SAMPLING_SECONDS:-120}" \
    --seed "${SBI_SEED:-42}"
fi
if [[ "${STAGE}" == all || "${STAGE}" == run ]]; then
  "${PYTHON}" "${SCRIPT}" check-runtime
  for selected in bins40 pca moped; do
    if [[ "${METHOD}" == all || "${METHOD}" == "${selected}" ]]; then
      "${PYTHON}" "${SCRIPT}" run --output-root "${COMPRESSION_ROOT}" --method "${selected}"
    fi
  done
fi
if [[ "${STAGE}" == all || "${STAGE}" == summarize ]]; then
  "${PYTHON}" "${SCRIPT}" summarize --output-root "${COMPRESSION_ROOT}"
fi
