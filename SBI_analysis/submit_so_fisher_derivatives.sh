#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)`nDEFAULT_PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)`n: "${PROJECT_DIR:=${DEFAULT_PROJECT_DIR}}"
: "${FISHER_ROOT:=/lustre/work/kristero10/adrian_fisher_baseline_deproj0}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${MAX_CONCURRENT_JOBS:=5}"
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

echo "Submitting ${N_ROWS} derivative simulations, at most ${MAX_CONCURRENT_JOBS} at once."
qsub \
  -J "1-${N_ROWS}%${MAX_CONCURRENT_JOBS}" \
  -v "PROJECT_DIR=${PROJECT_DIR},FISHER_ROOT=${FISHER_ROOT}" \
  "${PBS_SCRIPT}"
