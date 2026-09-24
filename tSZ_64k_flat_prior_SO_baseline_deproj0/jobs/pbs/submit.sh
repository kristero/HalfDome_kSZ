#!/usr/bin/env bash
# Submit one stage with PBS Pro (written for idark; edit QUEUE for other sites).
#
#   jobs/pbs/submit.sh precompile|repro|audit|production|pool|collect [qsub options]
#
# Environment: CONFIG, PYTHON (default /home/anaconda3/bin/python3 on idark, else python3),
# QUEUE (default mini), MAX_ARRAY_SIZE (default 10000), BATCHES_PER_TASK (default 4),
# FIRST_TASK/LAST_TASK (array-task range), POOL_WORKERS (default 8), POOL_BATCHES
# (default 4), STOP_AFTER_HOURS (default 3), COLLECT_ARGS, DRY_RUN=1.
# PBS has no array concurrency limit here: submit production in parts with
# FIRST_TASK/LAST_TASK if the queue should not be flooded.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/../.." && pwd)"
CONFIG="${CONFIG:-${BUNDLE}/pipeline/config.toml}"
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON=python3
  [[ -x /home/anaconda3/bin/python3 ]] && PYTHON=/home/anaconda3/bin/python3
fi
BATCHES_PER_TASK="${BATCHES_PER_TASK:-4}"
STOP_AFTER_HOURS="${STOP_AFTER_HOURS:-3}"
QUEUE="${QUEUE:-mini}"
stage="${1:?usage: submit.sh precompile|repro|audit|production|pool|collect [qsub options]}"
shift
manage=("${PYTHON}" "${BUNDLE}/pipeline/manage.py" --config "${CONFIG}")
LOGS="$("${manage[@]}" show run_root)/logs"
mkdir -p "${LOGS}"
VARS="BUNDLE=${BUNDLE},CONFIG=${CONFIG},PYTHON=${PYTHON},BATCHES_PER_TASK=${BATCHES_PER_TASK}"
VARS="${VARS},POOL_BATCHES=${POOL_BATCHES:-4},STOP_AFTER_HOURS=${STOP_AFTER_HOURS},TSZ64K_STAGE=${stage}"
[[ -n "${COLLECT_ARGS:-}" ]] && VARS="${VARS},COLLECT_ARGS=${COLLECT_ARGS}"
run() { if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; echo; else "$@"; fi; }

single() {
  local script="$1"; shift
  run qsub -q "${QUEUE}" -j oe -o "${LOGS}/" -v "${VARS},ARRAY_OFFSET=0" "$@" "${script}"
}

array() {  # array <first task> <last task> <script> [qsub options]
  local first="$1" last="$2" script="$3" offset count
  shift 3
  (( last >= first )) || { echo "nothing to submit (tasks ${first}-${last})" >&2; exit 1; }
  offset="${first}"
  while (( offset <= last )); do
    count=$(( last - offset + 1 < ${MAX_ARRAY_SIZE:-10000} ? last - offset + 1 : ${MAX_ARRAY_SIZE:-10000} ))
    if (( count == 1 )); then     # PBS arrays need at least two subjobs
      run qsub -q "${QUEUE}" -j oe -o "${LOGS}/" -v "${VARS},ARRAY_OFFSET=${offset}" "$@" "${script}"
    else
      run qsub -q "${QUEUE}" -j oe -o "${LOGS}/" -J "0-$(( count - 1 ))" -v "${VARS},ARRAY_OFFSET=${offset}" \
        "$@" "${script}"
    fi
    offset=$(( offset + count ))
  done
}

case "${stage}" in
  precompile) single "${HERE}/00_precompile.pbs" "$@" ;;
  repro)      single "${HERE}/01_repro.pbs" "$@" ;;
  audit)
    total="$("${manage[@]}" show audit_tasks)"
    array "${FIRST_TASK:-0}" "${LAST_TASK:-$(( total - 1 ))}" "${HERE}/02_audit.pbs" "$@" ;;
  production)
    total="$("${manage[@]}" show production_tasks --batches-per-task "${BATCHES_PER_TASK}")"
    array "${FIRST_TASK:-0}" "${LAST_TASK:-$(( total - 1 ))}" "${HERE}/03_production.pbs" "$@" ;;
  pool)       array 0 "$(( ${POOL_WORKERS:-8} - 1 ))" "${HERE}/03_production.pbs" "$@" ;;
  collect)    single "${HERE}/04_collect.pbs" "$@" ;;
  *) echo "unknown stage ${stage}" >&2; exit 2 ;;
esac
