#!/usr/bin/env bash
# Submit one stage with SLURM; extra arguments go to sbatch.
#
#   jobs/slurm/submit.sh precompile|repro|audit|production|pool|collect [sbatch options]
#
#   jobs/slurm/submit.sh repro --partition=cpu --account=myproject
#   ARRAY_LIMIT=64 jobs/slurm/submit.sh audit --partition=cpu
#   ARRAY_LIMIT=200 jobs/slurm/submit.sh production --partition=cpu
#   FIRST_TASK=0 LAST_TASK=511 jobs/slurm/submit.sh production   # rows 0-8191 only
#   POOL_WORKERS=16 jobs/slurm/submit.sh pool                     # mop up pending batches
#   COLLECT_ARGS="--rows 8192 --npz" jobs/slurm/submit.sh collect
#
# Environment: CONFIG (default pipeline/config.toml), PYTHON (default python3),
# ARRAY_LIMIT (concurrent array tasks, default 32), MAX_ARRAY_SIZE (the cluster's
# limit; larger arrays are split, default 1000), BATCHES_PER_TASK (default 4),
# FIRST_TASK/LAST_TASK (array-task range), POOL_WORKERS (default 8), POOL_BATCHES
# (batches per pool worker, default 4), STOP_AFTER_HOURS (default 3),
# DRY_RUN=1 prints the sbatch commands only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/../.." && pwd)"
CONFIG="${CONFIG:-${BUNDLE}/pipeline/config.toml}"
PYTHON="${PYTHON:-python3}"
BATCHES_PER_TASK="${BATCHES_PER_TASK:-4}"
STOP_AFTER_HOURS="${STOP_AFTER_HOURS:-3}"
stage="${1:?usage: submit.sh precompile|repro|audit|production|pool|collect [sbatch options]}"
shift
manage=("${PYTHON}" "${BUNDLE}/pipeline/manage.py" --config "${CONFIG}")
LOGS="$("${manage[@]}" show run_root)/logs"
mkdir -p "${LOGS}"
EXPORTS="ALL,BUNDLE=${BUNDLE},CONFIG=${CONFIG},PYTHON=${PYTHON},BATCHES_PER_TASK=${BATCHES_PER_TASK}"
EXPORTS="${EXPORTS},POOL_BATCHES=${POOL_BATCHES:-4},STOP_AFTER_HOURS=${STOP_AFTER_HOURS}"
EXPORTS="${EXPORTS},COLLECT_ARGS=${COLLECT_ARGS:-},TSZ64K_STAGE=${stage}"
run() { if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; echo; else "$@"; fi; }

single() {
  local script="$1"; shift
  run sbatch --export="${EXPORTS},ARRAY_OFFSET=0" --output="${LOGS}/%x_%j.out" "$@" "${script}"
}

array() {  # array <first task> <last task> <script> [sbatch options]
  local first="$1" last="$2" script="$3" offset count
  shift 3
  (( last >= first )) || { echo "nothing to submit (tasks ${first}-${last})" >&2; exit 1; }
  offset="${first}"
  while (( offset <= last )); do
    count=$(( last - offset + 1 < ${MAX_ARRAY_SIZE:-1000} ? last - offset + 1 : ${MAX_ARRAY_SIZE:-1000} ))
    run sbatch --export="${EXPORTS},ARRAY_OFFSET=${offset}" --array="0-$(( count - 1 ))%${ARRAY_LIMIT:-32}" \
      --output="${LOGS}/%x_%A_%a.out" "$@" "${script}"
    offset=$(( offset + count ))
  done
}

case "${stage}" in
  precompile) single "${HERE}/00_precompile.sbatch" "$@" ;;
  repro)      single "${HERE}/01_repro.sbatch" "$@" ;;
  audit)
    total="$("${manage[@]}" show audit_tasks)"
    array "${FIRST_TASK:-0}" "${LAST_TASK:-$(( total - 1 ))}" "${HERE}/02_audit.sbatch" "$@" ;;
  production)
    total="$("${manage[@]}" show production_tasks --batches-per-task "${BATCHES_PER_TASK}")"
    array "${FIRST_TASK:-0}" "${LAST_TASK:-$(( total - 1 ))}" "${HERE}/03_production.sbatch" "$@" ;;
  pool)       array 0 "$(( ${POOL_WORKERS:-8} - 1 ))" "${HERE}/03_production.sbatch" "$@" ;;
  collect)    single "${HERE}/04_collect.sbatch" "$@" ;;
  *) echo "unknown stage ${stage}" >&2; exit 2 ;;
esac
