#!/bin/bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${BUNDLE}/cluster.env"
: "${DRY_RUN:=0}"
[[ "${HALFDOME_PATH}" != /EDIT/* && "${OUTPUT_ROOT}" != /EDIT/* ]] || {
  echo "Edit HALFDOME_PATH and OUTPUT_ROOT in cluster.env first." >&2; exit 2;
}
[[ "${N_WORKERS}" =~ ^[1-9][0-9]*$ && "${MAX_ACTIVE_JOBS}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${OUTPUT_ROOT}" != *','* && "${BUNDLE}" != *','* ]] || {
  echo "PBS -v paths must not contain commas." >&2; exit 2;
}
mode="${1:-wave}"
case "${mode}" in
  wave) workers="${N_WORKERS}"; max_rows=0 ;;
  smoke) workers=1; max_rows=2 ;;
  *) echo "Usage: bash submit.sh [smoke|wave]" >&2; exit 2 ;;
esac
mkdir -p "${OUTPUT_ROOT}/logs"
exec 9>"${OUTPUT_ROOT}/.submission.lock"
flock -n 9 || { echo "Another submitter is active for this output root." >&2; exit 2; }
"${PYTHON}" "${BUNDLE}/generate.py" check
# Do not overlap waves or collide with a smoke test still running.
if [[ -f "${OUTPUT_ROOT}/latest_jobs.txt" && "${DRY_RUN}" != 1 ]]; then
  while read -r job; do
    if status=$(qstat -f "${job}" 2>/dev/null); then
      state=$(printf '%s\n' "${status}" | awk '/job_state =/{print $3; exit}')
      case "${state}" in
        F|C) ;;
        *) echo "Previous job ${job} is still ${state}. Wait for the wave to finish." >&2; exit 2 ;;
      esac
    fi
  done < "${OUTPUT_ROOT}/latest_jobs.txt"
fi
record="${OUTPUT_ROOT}/jobs_${mode}_$(date +%Y%m%dT%H%M%S).txt"
if [[ "${DRY_RUN}" != 1 ]]; then
  touch "${record}"
  ln -sfn "$(basename -- "${record}")" "${OUTPUT_ROOT}/latest_jobs.txt"
fi
declare -a last
for ((lane=0; lane<MAX_ACTIVE_JOBS; lane++)); do last[lane]=""; done
for ((worker=0; worker<workers; worker++)); do
  lane=$((worker % MAX_ACTIVE_JOBS))
  set -- -q "${QUEUE}" -N "SOgen${worker}" \
    -l "select=1:ncpus=${CPUS}:mpiprocs=1:mem=${MEMORY}" -l "walltime=${WALLTIME}" \
    -o "${OUTPUT_ROOT}/logs/" -j oe \
    -v "BUNDLE=${BUNDLE},WORKER=${worker},WORKERS=${workers},MAX_ROWS=${max_rows},HALFDOME_PATH=${HALFDOME_PATH},OUTPUT_ROOT=${OUTPUT_ROOT},CPUS=${CPUS},JULIA=${JULIA},PYTHON=${PYTHON},WORK_SECONDS=${WORK_SECONDS},ROW_TIMEOUT_SECONDS=${ROW_TIMEOUT_SECONDS}"
  if [[ -n "${last[lane]}" ]]; then set -- "$@" -W "depend=afterany:${last[lane]}"; fi
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'qsub '; printf '%q ' "$@" "${BUNDLE}/worker.pbs"; printf '\n'
    job="dry${worker}"
  else
    job=$(qsub "$@" "${BUNDLE}/worker.pbs")
    printf '%s\n' "${job}" | tee -a "${record}"
  fi
  last[lane]="${job}"
done
echo "${workers} separate jobs, at most ${MAX_ACTIVE_JOBS} active in this wave."
echo "Run status after the wave; resubmit another wave if rows remain."
