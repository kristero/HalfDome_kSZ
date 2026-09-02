#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:=${PWD}}"
: "${PBS_SCRIPT:=${PROJECT_ROOT}/SBI_analysis/run_battaglia12_baseline_deproj0_observation.pbs}"
: "${OUTPUT_BASE:=/lustre/work/kristero10/adrian_fisher_baseline_deproj0/battaglia12_baseline_deproj0_ensemble}"
: "${N_PROFILES:=64}"
: "${MASK_SEED:=12345}"
: "${NOISE_SEED_START:=20001}"
: "${MAX_ACTIVE_JOBS:=5}"
: "${POLL_SECONDS:=60}"

if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "PBS script does not exist: ${PBS_SCRIPT}" >&2
  exit 2
fi
if (( N_PROFILES < 1 || MAX_ACTIVE_JOBS < 1 )); then
  echo "N_PROFILES and MAX_ACTIVE_JOBS must be positive." >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
run_id="$(date +%Y%m%dT%H%M%S)"
active_file="${OUTPUT_BASE}/active_jobs_${run_id}.txt"
record_file="${OUTPUT_BASE}/submitted_jobs_${run_id}.tsv"
: > "${active_file}"
printf "noise_seed\tjob_id\toutput_root\n" > "${record_file}"

refresh_active_jobs() {
  local refreshed="${active_file}.new"
  : > "${refreshed}"
  while IFS= read -r job_id; do
    [[ -n "${job_id}" ]] || continue
    if qstat "${job_id}" >/dev/null 2>&1; then
      printf "%s\n" "${job_id}" >> "${refreshed}"
    fi
  done < "${active_file}"
  mv "${refreshed}" "${active_file}"
}

active_count() {
  refresh_active_jobs
  wc -l < "${active_file}"
}

echo "Submitting ${N_PROFILES} separate Battaglia12 jobs."
echo "Mask seed remains fixed at ${MASK_SEED}; only the noise seed changes."
echo "At most ${MAX_ACTIVE_JOBS} submitted jobs are kept active at once."
echo "Submission record: ${record_file}"

for ((offset=0; offset<N_PROFILES; offset++)); do
  noise_seed=$((NOISE_SEED_START + offset))
  while (( $(active_count) >= MAX_ACTIVE_JOBS )); do
    echo "Five-job limit reached; checking again in ${POLL_SECONDS} s."
    sleep "${POLL_SECONDS}"
  done

  output_root="${OUTPUT_BASE}/noise_seed${noise_seed}"
  job_id="$(
    qsub       -N "B12n${noise_seed}"       -v "PROJECT_ROOT=${PROJECT_ROOT},MASK_SEED=${MASK_SEED},NOISE_SEED=${noise_seed},OUTPUT_ROOT=${output_root}"       "${PBS_SCRIPT}"
  )"
  printf "%s\n" "${job_id}" >> "${active_file}"
  printf "%s\t%s\t%s\n" "${noise_seed}" "${job_id}" "${output_root}" >> "${record_file}"
  echo "Submitted noise seed ${noise_seed}: ${job_id}"
done

echo "All ${N_PROFILES} jobs have been submitted as separate PBS jobs."
echo "The last jobs may still be queued or running; inspect IDs in ${record_file}."
