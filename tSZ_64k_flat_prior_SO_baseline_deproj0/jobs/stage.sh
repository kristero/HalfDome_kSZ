#!/bin/bash
# Body shared by the SLURM (jobs/slurm/*.sbatch) and PBS (jobs/pbs/*.pbs) stage scripts.
# Usage: stage.sh precompile|repro|audit|production|pool|collect
# Needs BUNDLE; optional CONFIG, PYTHON, ARRAY_OFFSET, BATCHES_PER_TASK,
# POOL_BATCHES, STOP_AFTER_HOURS, COLLECT_ARGS. The submit.sh helpers set them.
set -euo pipefail
: "${BUNDLE:?BUNDLE must point to the tSZ_64k_flat_prior_SO_baseline_deproj0 folder}"
CONFIG="${CONFIG:-${BUNDLE}/pipeline/config.toml}"
PYTHON="${PYTHON:-python3}"
THREADS="${SLURM_CPUS_PER_TASK:-${NCPUS:-${OMP_NUM_THREADS:-1}}}"
INDEX="${SLURM_ARRAY_TASK_ID:-${PBS_ARRAY_INDEX:-0}}"
INDEX=$(( ${ARRAY_OFFSET:-0} + INDEX ))
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS="${THREADS}"
manage=("${PYTHON}" "${BUNDLE}/pipeline/manage.py" --config "${CONFIG}")
echo "stage=$1 host=$(hostname) threads=${THREADS} index=${INDEX} start=$(date -Is)"
case "$1" in
  precompile)
    JULIA="$("${manage[@]}" show julia)"
    DEPOT="$("${manage[@]}" show julia_depot)"
    env -u JULIA_PROJECT -u JULIA_LOAD_PATH JULIA_DEPOT_PATH="${DEPOT}" \
      LD_LIBRARY_PATH="$(dirname "$(dirname "${JULIA}")")/lib/julia:${LD_LIBRARY_PATH:-}" \
      "${JULIA}" --startup-file=no --threads="${THREADS}" --project="${BUNDLE}/julia_env" \
      -e 'using Pkg; Pkg.precompile()'
    env -u JULIA_PROJECT -u JULIA_LOAD_PATH JULIA_DEPOT_PATH="${DEPOT}" \
      "${JULIA}" --startup-file=no --project="${BUNDLE}/julia_env" "${BUNDLE}/setup/verify_env.jl" ;;
  repro)      "${manage[@]}" repro-check --run --threads "${THREADS}" ;;
  audit)      "${manage[@]}" audit --task "${INDEX}" --threads "${THREADS}" ;;
  production) "${manage[@]}" run --array-index "${INDEX}" --batches-per-task "${BATCHES_PER_TASK:-4}" \
                --threads "${THREADS}" --stop-after-hours "${STOP_AFTER_HOURS:-3}" ;;
  pool)       "${manage[@]}" run --pool --max-batches "${POOL_BATCHES:-4}" \
                --threads "${THREADS}" --stop-after-hours "${STOP_AFTER_HOURS:-3}" ;;
  collect)    "${manage[@]}" collect ${COLLECT_ARGS:-} ;;
  *) echo "unknown stage $1" >&2; exit 2 ;;
esac
echo "stage=$1 finished=$(date -Is)"
