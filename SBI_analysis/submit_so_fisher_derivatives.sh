#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
: "${PROJECT_DIR:=${DEFAULT_PROJECT_DIR}}"
: "${FISHER_ROOT:=/lustre/work/kristero10/adrian_fisher_baseline_deproj0}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${FISHER_QUEUE:=mini}"
: "${MAX_CONCURRENT_JOBS:=6}"
: "${REBUILD_GRID:=0}"
: "${SKIP_COMPLETED:=1}"
: "${CANCEL_EXISTING:=0}"

cd "${PROJECT_DIR}"

GENERATOR="${PROJECT_DIR}/SBI_analysis/generate_so_fisher_variations.py"
PBS_SCRIPT="${PROJECT_DIR}/SBI_analysis/run_so_fisher_derivative.pbs"
MANIFEST="${FISHER_ROOT}/fisher_variations_manifest.csv"
SUBMISSION_LOG="${FISHER_ROOT}/fisher_derivative_submitted_jobs.csv"

for path in "${GENERATOR}" "${PBS_SCRIPT}" "${PREPARED_DATASET}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

if [[ "${CANCEL_EXISTING}" == "1" ]]; then
  if ! command -v qselect >/dev/null 2>&1; then
    echo "CANCEL_EXISTING=1 requires qselect, but qselect is unavailable." >&2
    echo "Cancel the old SO_fisher_deriv job manually with qdel, then rerun without CANCEL_EXISTING." >&2
    exit 2
  fi

  mapfile -t old_job_ids < <(
    qselect -u "${USER}" -N SO_fisher_deriv 2>/dev/null || true
  )
  if (( ${#old_job_ids[@]} > 0 )); then
    echo "Cancelling previous SO_fisher_deriv jobs: ${old_job_ids[*]}"
    for job_id in "${old_job_ids[@]}"; do
      qdel "${job_id}"
    done
  else
    echo "No existing SO_fisher_deriv jobs found."
  fi
fi

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
if [[ ! "${MAX_CONCURRENT_JOBS}" =~ ^[0-9]+$ ]] || (( MAX_CONCURRENT_JOBS < 1 )); then
  echo "MAX_CONCURRENT_JOBS must be a positive integer; got ${MAX_CONCURRENT_JOBS}." >&2
  exit 2
fi

pending_rows=()
pending_labels=()
for ((row=1; row<=N_ROWS; row++)); do
  label="$(awk -F, -v target="${row}" 'NR == target + 1 {gsub(/\r/, "", $2); print $2}' "${MANIFEST}")"
  if [[ -z "${label}" ]]; then
    echo "Could not read label for row ${row} from ${MANIFEST}." >&2
    exit 2
  fi

  row_tag="$(printf 'row%03d_%s' "${row}" "${label}")"
  output_dir="${FISHER_ROOT}/variations/${row_tag}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && [[ -d "${output_dir}" ]] && find "${output_dir}" -maxdepth 1 -type f -name '*masked_no_noise_cl*.npy' -print -quit | grep -q .; then
    echo "Already complete, not submitting: row ${row} (${label})"
    continue
  fi

  pending_rows+=("${row}")
  pending_labels+=("${label}")
done

if (( ${#pending_rows[@]} == 0 )); then
  echo "All ${N_ROWS} Fisher derivative rows are already complete."
  exit 0
fi

lane_count="${MAX_CONCURRENT_JOBS}"
if (( lane_count > ${#pending_rows[@]} )); then
  lane_count="${#pending_rows[@]}"
fi
lane_last_job=()

printf 'row_1based,label,lane,depends_on,job_id\n' > "${SUBMISSION_LOG}"

echo "Submitting ${#pending_rows[@]} independent derivative jobs to queue ${FISHER_QUEUE}."
echo "Five dependency lanes enforce at most ${lane_count} simultaneously runnable jobs."

for ((index=0; index<${#pending_rows[@]}; index++)); do
  row="${pending_rows[index]}"
  label="${pending_labels[index]}"
  lane=$(( index % lane_count ))
  dependency="${lane_last_job[lane]:-}"
  job_name="$(printf 'SOF_%03d' "${row}")"

  qsub_args=(
    -q "${FISHER_QUEUE}"
    -N "${job_name}"
    -v "PROJECT_DIR=${PROJECT_DIR},FISHER_ROOT=${FISHER_ROOT},FISHER_ROW=${row}"
  )
  if [[ -n "${dependency}" ]]; then
    qsub_args+=(
      -W "depend=afterany:${dependency}"
    )
  fi

  submission_output="$(qsub "${qsub_args[@]}" "${PBS_SCRIPT}")"
  job_id="$(printf '%s\n' "${submission_output}" | tail -n 1 | tr -d '[:space:]')"
  if [[ -z "${job_id}" ]]; then
    echo "qsub returned no job ID for row ${row}: ${submission_output}" >&2
    exit 2
  fi

  lane_last_job[lane]="${job_id}"
  printf '%s,%s,%s,%s,%s\n' \
    "${row}" "${label}" "$((lane + 1))" "${dependency}" "${job_id}" \
    >> "${SUBMISSION_LOG}"
  echo "Submitted row ${row} (${label}) as ${job_id}, lane $((lane + 1)), depends on ${dependency:-none}."
done

echo "Submission map: ${SUBMISSION_LOG}"
echo "Every variation is a separate PBS job; at most ${lane_count} are dependency-eligible at once."
