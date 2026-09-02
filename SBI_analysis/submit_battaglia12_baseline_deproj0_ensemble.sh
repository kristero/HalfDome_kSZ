#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:=${PWD}}"
: "${SINGLE_PROFILE_PBS:=${PROJECT_ROOT}/SBI_analysis/run_battaglia12_baseline_deproj0_observation.pbs}"
: "${CHUNK_PBS_SCRIPT:=${PROJECT_ROOT}/SBI_analysis/run_battaglia12_baseline_deproj0_ensemble_chunk.pbs}"
: "${OUTPUT_BASE:=/lustre/work/kristero10/adrian_fisher_baseline_deproj0/battaglia12_baseline_deproj0_ensemble}"
: "${N_PROFILES:=64}"
: "${N_JOBS:=6}"
: "${MASK_SEED:=12345}"
: "${NOISE_SEED_START:=20001}"

if [[ ! -f "${SINGLE_PROFILE_PBS}" || ! -f "${CHUNK_PBS_SCRIPT}" ]]; then
  echo "Required PBS script is missing." >&2
  echo "Single-profile worker: ${SINGLE_PROFILE_PBS}" >&2
  echo "Chunk worker: ${CHUNK_PBS_SCRIPT}" >&2
  exit 2
fi
if (( N_PROFILES < 1 || N_JOBS < 1 || N_JOBS > N_PROFILES )); then
  echo "Require N_PROFILES >= N_JOBS >= 1." >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"
run_id="$(date +%Y%m%dT%H%M%S)"
record_file="${OUTPUT_BASE}/submitted_chunks_${run_id}.tsv"
printf "chunk_index\tprofile_start\tprofile_count\tfirst_noise_seed\tlast_noise_seed\tjob_id\n" > "${record_file}"

echo "Submitting ${N_JOBS} Battaglia12 ensemble jobs for ${N_PROFILES} profiles."
echo "Mask seed remains fixed at ${MASK_SEED}; only the noise seed changes."
echo "Each job processes its assigned profiles sequentially."
echo "Submission record: ${record_file}"

base_count=$((N_PROFILES / N_JOBS))
remainder=$((N_PROFILES % N_JOBS))
profile_start=0

for ((chunk_index=0; chunk_index<N_JOBS; chunk_index++)); do
  profile_count=${base_count}
  if (( chunk_index < remainder )); then
    profile_count=$((profile_count + 1))
  fi
  first_noise_seed=$((NOISE_SEED_START + profile_start))
  last_noise_seed=$((first_noise_seed + profile_count - 1))

  job_id="$(
    qsub \
      -N "B12ens$((chunk_index + 1))" \
      -v "PROJECT_ROOT=${PROJECT_ROOT},SINGLE_PROFILE_PBS=${SINGLE_PROFILE_PBS},OUTPUT_BASE=${OUTPUT_BASE},MASK_SEED=${MASK_SEED},NOISE_SEED_START=${NOISE_SEED_START},PROFILE_START=${profile_start},PROFILE_COUNT=${profile_count},CHUNK_INDEX=${chunk_index}" \
      "${CHUNK_PBS_SCRIPT}"
  )"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${chunk_index}" "${profile_start}" "${profile_count}" \
    "${first_noise_seed}" "${last_noise_seed}" "${job_id}" \
    >> "${record_file}"
  echo "Submitted chunk $((chunk_index + 1))/${N_JOBS}: ${profile_count} profiles, seeds ${first_noise_seed}-${last_noise_seed}, job ${job_id}"
  profile_start=$((profile_start + profile_count))
done

if (( profile_start != N_PROFILES )); then
  echo "Internal error: assigned ${profile_start}/${N_PROFILES} profiles." >&2
  exit 3
fi

echo "Submitted exactly ${N_JOBS} PBS jobs covering all ${N_PROFILES} profiles."
echo "Inspect job IDs and seed ranges in ${record_file}."
