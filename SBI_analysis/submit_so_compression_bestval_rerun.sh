#!/bin/bash
# Two fresh training/evaluation jobs plus a summary with visible failure logs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${PYTHON:=python3}"
: "${SOURCE_ROOT:=/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0}"
: "${COMPRESSION_ROOT:=${SOURCE_ROOT}_bestval_$(date +%Y%m%dT%H%M%S)}"
[[ "${COMPRESSION_ROOT}" != *,* ]] || { echo "Output root must not contain commas" >&2; exit 2; }
cd "${PROJECT_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/prepare_so_compression_bestval_rerun.py" \
  --source-root "${SOURCE_ROOT}" --output-root "${COMPRESSION_ROOT}"
mkdir -p "${COMPRESSION_ROOT}/logs" /home/kristero10/logs/SBI_runs
CONFIG="${COMPRESSION_ROOT}/submission_config.sh"
for variable in PROJECT_ROOT PYTHON COMPRESSION_ROOT; do
  printf '%s=%q\n' "${variable}" "${!variable}" >> "${CONFIG}"
done
PBS="${SCRIPT_DIR}/run_so_sbi_compression_comparison.pbs"
LEDGER="${COMPRESSION_ROOT}/submission_jobs.tsv"
dependency=""
for method in pca moped; do
  job=$(qsub -N "SOcb_${method}" \
    -v "COMPRESSION_CONFIG=${CONFIG},COMPRESSION_STAGE=run,COMPRESSION_METHOD=${method}" "${PBS}")
  printf '%s\t%s\n' "${method}" "${job}" | tee -a "${LEDGER}"
  dependency="${dependency}:${job}"
done
summary=$(qsub -N SOcb_summary -W "depend=afterany${dependency}" \
  -v "COMPRESSION_CONFIG=${CONFIG},COMPRESSION_STAGE=summarize" "${PBS}")
printf 'summary\t%s\n' "${summary}" | tee -a "${LEDGER}"
echo "Only 3 jobs submitted. Comparison outputs: ${COMPRESSION_ROOT}/summary"
