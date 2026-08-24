#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
: "${PROJECT_DIR:=${DEFAULT_PROJECT_DIR}}"
: "${FISHER_ROOT:=/lustre/work/kristero10/adrian_fisher_baseline_deproj0}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${FISHER_QUEUE:=mini2}"
: "${FISHER_WORKER_COUNT:=${MAX_CONCURRENT_JOBS:-5}}"
: "${REBUILD_GRID:=0}"

cd "${PROJECT_DIR}"

GENERATOR="${PROJECT_DIR}/SBI_analysis/generate_so_fisher_variations.py"
PBS_SCRIPT="${PROJECT_DIR}/SBI_analysis/run_so_fisher_derivative.pbs"
MANIFEST="${FISHER_ROOT}/fisher_variations_manifest.csv"

for path in "${GENERATOR}" "${PBS_SCRIPT}" "${PREPARED_DATASET}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

generator_args=(
  --prepared-dataset "${PREPARED_DATASET}"
  --output-root "${FISHER_ROOT}"
  --step-fractions 0.01 0.02
)
if [[ "${REBUILD_GRID}" == "1" ]]; then
  generator_args+=(--overwrite)
fi

if [[ ! -f "${MANIFEST}" || "${REBUILD_GRID}" == "1" ]]; then
  python3 "${GENERATOR}" "${generator_args[@]}"
else
  echo "Using existing Fisher grid: ${MANIFEST}"
fi

N_ROWS=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( N_ROWS < 1 )); then
  echo "No rows found in ${MANIFEST}." >&2
  exit 2
fi
if [[ ! "${FISHER_WORKER_COUNT}" =~ ^[0-9]+$ ]] || (( FISHER_WORKER_COUNT < 1 )); then
  echo "FISHER_WORKER_COUNT must be a positive integer; got ${FISHER_WORKER_COUNT}." >&2
  exit 2
fi
if (( FISHER_WORKER_COUNT > N_ROWS )); then
  FISHER_WORKER_COUNT="${N_ROWS}"
fi

echo "Submitting ${N_ROWS} derivative simulations across ${FISHER_WORKER_COUNT} PBS workers in queue ${FISHER_QUEUE}."
echo "Each worker processes every ${FISHER_WORKER_COUNT}-th variation row sequentially."
qsub \
  -q "${FISHER_QUEUE}" \
  -J "1-${FISHER_WORKER_COUNT}" \
  -v "PROJECT_DIR=${PROJECT_DIR},FISHER_ROOT=${FISHER_ROOT},FISHER_WORKER_COUNT=${FISHER_WORKER_COUNT}" \
  "${PBS_SCRIPT}"
