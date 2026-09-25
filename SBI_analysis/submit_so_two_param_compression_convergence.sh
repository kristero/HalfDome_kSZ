#!/bin/bash
set -euo pipefail
: "${COMPARISON_CONFIG:?Set COMPARISON_CONFIG to the reviewed experiment configuration}"
source "${COMPARISON_CONFIG}"
: "${MAX_ACTIVE_JOBS:=4}"
: "${QSUB:=qsub}"
: "${PYTHON:=python3}"
: "${DRY_RUN:=0}"
[[ "${CSV_MAPPING_CONFIRMED:-0}" == 1 ]] || {
  echo "STOP: confirm the generating CSV and prefix before setting CSV_MAPPING_CONFIRMED=1." >&2
  exit 2
}
[[ "${MAX_ACTIVE_JOBS}" =~ ^[1-5]$ ]] || { echo "Use 1..5 active jobs" >&2; exit 2; }
for path in "${SOBOL_CSV}" "${BATTAGLIA_RAW}" "${RAW_DATA}/combined_cl_layout.txt"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "${ROOT}/logs" /home/kristero10/logs/SBI_runs
JOB_RECORD="${ROOT}/submission_jobs.txt"
if [[ "${DRY_RUN}" == 1 ]]; then
  JOB_RECORD="${ROOT}/submission_dry_run_jobs.txt"
elif [[ -s "${JOB_RECORD}" ]]; then
  echo "Existing submission record: ${JOB_RECORD}. Resume individual jobs; do not duplicate this workflow." >&2
  exit 2
fi
PBS="${CODE_DIR}/run_so_two_param_compression_convergence.pbs"
set -- --data "${RAW_DATA}" --sobol-csv "${SOBOL_CSV}" \
  --expected-csv-sha256 "${CSV_SHA256}" --prior-low "${P0_LOW}" "${BETA_LOW}" \
  --prior-high "${P0_HIGH}" "${BETA_HIGH}" --battaglia-raw "${BATTAGLIA_RAW}" \
  --output "${DATASET}" --validate-only --holdout-last-n "${HOLDOUT}" \
  --validation-report "${ROOT}/input_validation.json"
if [[ "${ALLOW_CSV_PREFIX:-0}" == 1 ]]; then set -- "$@" --allow-csv-prefix; fi
"${PYTHON}" "${CODE_DIR}/prepare_so_two_param_combined_convergence.py" "$@"
# Resolve max using the actual in-prior training pool, before submitting any job.
size_list=$(PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -c \
  'import json, sys; from run_so_two_param_compression_convergence import parse_sizes; d=json.load(open(sys.argv[1])); assert d["n_rows"] == int(sys.argv[3]); print(" ".join(map(str, parse_sizes(sys.argv[2], d["training_pool"]))))' \
  "${ROOT}/input_validation.json" "${SIZES}" "${N_EXPECTED}")
"${PYTHON}" -c 'import getdist, matplotlib, pandas; print("Plotting dependencies available")'
submit() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'DRY RUN qsub: ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    printf 'dry.%s\n' "$2"
  else
    "${QSUB}" "$@"
  fi
}
counter=0
prepare_id=$(submit -N S2cPrepare -l select=1:ncpus=26:mpiprocs=1:mem=4gb \
  -v "COMPARISON_CONFIG=${COMPARISON_CONFIG},STAGE=prepare" "${PBS}")
printf 'prepare %s\n' "${prepare_id}" | tee -a "${JOB_RECORD}"
declare -a lane_last
for ((lane=0; lane<MAX_ACTIVE_JOBS; lane++)); do lane_last[lane]=""; done
counter=0
for n in ${size_list}; do
  [[ "${n}" =~ ^[0-9]+$ ]] || { echo "Invalid N: ${n}" >&2; exit 2; }
  for method in bins40 moped; do
    lane=$((counter % MAX_ACTIVE_JOBS))
    dependency="afterok:${prepare_id}"
    if [[ -n "${lane_last[lane]}" ]]; then
      dependency+=",afterany:${lane_last[lane]}"
    fi
    job=$(submit -N "S2c${counter}" -W "depend=${dependency}" \
      -v "COMPARISON_CONFIG=${COMPARISON_CONFIG},STAGE=run,METHOD=${method},N_TRAIN=${n}" "${PBS}")
    lane_last[lane]="${job}"
    printf '%s N%s %s\n' "${method}" "${n}" "${job}" | tee -a "${JOB_RECORD}"
    counter=$((counter+1))
  done
done
dependencies=""
for ((lane=0; lane<MAX_ACTIVE_JOBS; lane++)); do
  if [[ -n "${lane_last[lane]}" ]]; then dependencies+=":${lane_last[lane]}"; fi
done
job=$(submit -N S2cSummary -W "depend=afterany${dependencies}" \
  -v "COMPARISON_CONFIG=${COMPARISON_CONFIG},STAGE=summarize" "${PBS}")
printf 'summary %s\n' "${job}" | tee -a "${JOB_RECORD}"
echo "Separate jobs; at most ${MAX_ACTIVE_JOBS} training jobs runnable concurrently."
