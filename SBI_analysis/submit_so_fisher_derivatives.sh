#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
: "${PROJECT_DIR:=${DEFAULT_PROJECT_DIR}}"
: "${FISHER_ROOT:=/lustre/work/kristero10/adrian_fisher_baseline_deproj0}"
: "${PREPARED_DATASET:=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz}"
: "${FISHER_QUEUE:=mini2}"
: "${SUBMIT_BATCH_SIZE:=4}"
: "${REBUILD_GRID:=0}"
: "${SKIP_COMPLETED:=1}"
: "${CANCEL_EXISTING:=0}"

cd "${PROJECT_DIR}"

GENERATOR="${PROJECT_DIR}/SBI_analysis/generate_so_fisher_variations.py"
PBS_SCRIPT="${PROJECT_DIR}/SBI_analysis/run_so_fisher_derivative.pbs"
MANIFEST="${FISHER_ROOT}/fisher_variations_manifest.csv"
CURRENT_JOBS_LOG="${FISHER_ROOT}/fisher_derivative_submitted_jobs.csv"
SUBMISSION_HISTORY_DIR="${FISHER_ROOT}/submission_history"

for path in "${GENERATOR}" "${PBS_SCRIPT}" "${PREPARED_DATASET}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ ! "${SUBMIT_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( SUBMIT_BATCH_SIZE < 1 )); then
  echo "SUBMIT_BATCH_SIZE must be a positive integer; got ${SUBMIT_BATCH_SIZE}." >&2
  exit 2
fi
if (( SUBMIT_BATCH_SIZE > 4 )); then
  echo "Clamping SUBMIT_BATCH_SIZE=${SUBMIT_BATCH_SIZE} to the four-job limit."
  SUBMIT_BATCH_SIZE=4
fi

job_is_active() {
  local job_id="$1"
  qstat "${job_id}" >/dev/null 2>&1
}

cancel_job_ids() {
  local ids_text="$1"
  local reason="$2"
  local job_id
  local cancelled=0

  while IFS= read -r job_id; do
    [[ -n "${job_id}" ]] || continue
    if job_is_active "${job_id}"; then
      echo "Cancelling ${reason}: ${job_id}"
      qdel "${job_id}"
      cancelled=$((cancelled + 1))
    fi
  done <<< "${ids_text}"

  return 0
}

logged_job_ids_text=""
if [[ -f "${CURRENT_JOBS_LOG}" ]]; then
  logged_job_ids_text="$(
    awk -F, 'NR > 1 && $NF != "" {gsub(/\r/, "", $NF); print $NF}' "${CURRENT_JOBS_LOG}"
  )"
fi

old_array_ids_text=""
if command -v qselect >/dev/null 2>&1; then
  old_array_ids_text="$(
    qselect -u "${USER}" -N SO_fisher_deriv 2>/dev/null || true
  )"
fi

# Array jobs belong to the obsolete implementation and are always removed.
if [[ -n "${old_array_ids_text}" ]]; then
  cancel_job_ids "${old_array_ids_text}" "obsolete Fisher array job"
fi

obsolete_submission_log=0
if [[ -f "${CURRENT_JOBS_LOG}" ]] && head -n 1 "${CURRENT_JOBS_LOG}" | grep -q 'lane,depends_on'; then
  obsolete_submission_log=1
fi

# The old helper submitted 37 dependent jobs at once. Remove those automatically.
if [[ "${obsolete_submission_log}" == "1" ]] && [[ -n "${logged_job_ids_text}" ]]; then
  cancel_job_ids "${logged_job_ids_text}" "obsolete Fisher dependency job"
  logged_job_ids_text=""
fi

if [[ "${CANCEL_EXISTING}" == "1" ]]; then
  if [[ -n "${logged_job_ids_text}" ]]; then
    cancel_job_ids "${logged_job_ids_text}" "current Fisher batch job"
  else
    echo "No current separate-job batch was found."
  fi
  logged_job_ids_text=""
else
  active_job_ids_text=""
  while IFS= read -r job_id; do
    [[ -n "${job_id}" ]] || continue
    if job_is_active "${job_id}"; then
      active_job_ids_text+="${job_id}"$'\n'
    fi
  done <<< "${logged_job_ids_text}"

  if [[ -n "${active_job_ids_text}" ]]; then
    echo "A four-job Fisher batch is still active:" >&2
    printf '%s' "${active_job_ids_text}" >&2
    echo "Wait for it to finish, or rerun with CANCEL_EXISTING=1." >&2
    exit 3
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

pending_rows=()
pending_labels=()
completed_count=0
for ((row=1; row<=N_ROWS; row++)); do
  label="$(awk -F, -v target="${row}" 'NR == target + 1 {gsub(/\r/, "", $2); print $2}' "${MANIFEST}")"
  if [[ -z "${label}" ]]; then
    echo "Could not read label for row ${row} from ${MANIFEST}." >&2
    exit 2
  fi

  row_tag="$(printf 'row%03d_%s' "${row}" "${label}")"
  output_dir="${FISHER_ROOT}/variations/${row_tag}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && [[ -d "${output_dir}" ]] && find "${output_dir}" -maxdepth 1 -type f -name '*masked_no_noise_cl*.npy' -print -quit | grep -q .; then
    completed_count=$((completed_count + 1))
    continue
  fi

  pending_rows+=("${row}")
  pending_labels+=("${label}")
done

if (( ${#pending_rows[@]} == 0 )); then
  echo "All ${N_ROWS} Fisher derivative rows are complete."
  exit 0
fi

batch_count="${SUBMIT_BATCH_SIZE}"
if (( batch_count > ${#pending_rows[@]} )); then
  batch_count="${#pending_rows[@]}"
fi

mkdir -p "${SUBMISSION_HISTORY_DIR}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -f "${CURRENT_JOBS_LOG}" ]]; then
  cp "${CURRENT_JOBS_LOG}" "${SUBMISSION_HISTORY_DIR}/fisher_jobs_before_${timestamp}.csv"
fi
printf 'submitted_utc,row_1based,label,job_id\n' > "${CURRENT_JOBS_LOG}"

echo "Completed derivative rows: ${completed_count}/${N_ROWS}"
echo "Submitting only ${batch_count} separate, non-array jobs to queue ${FISHER_QUEUE}."

for ((index=0; index<batch_count; index++)); do
  row="${pending_rows[index]}"
  label="${pending_labels[index]}"
  job_name="$(printf 'SOF_%03d' "${row}")"

  submission_output="$(
    qsub \
      -q "${FISHER_QUEUE}" \
      -N "${job_name}" \
      -v "PROJECT_DIR=${PROJECT_DIR},FISHER_ROOT=${FISHER_ROOT},FISHER_ROW=${row}" \
      "${PBS_SCRIPT}"
  )"
  job_id="$(printf '%s\n' "${submission_output}" | tail -n 1 | tr -d '[:space:]')"
  if [[ -z "${job_id}" ]]; then
    echo "qsub returned no job ID for row ${row}: ${submission_output}" >&2
    exit 2
  fi
  if [[ "${job_id}" == *"["* ]] || [[ "${job_id}" == *"]"* ]]; then
    echo "Refusing array job ID returned for row ${row}: ${job_id}" >&2
    qdel "${job_id}" >/dev/null 2>&1 || true
    exit 2
  fi

  printf '%s,%s,%s,%s\n' \
    "${timestamp}" "${row}" "${label}" "${job_id}" \
    >> "${CURRENT_JOBS_LOG}"
  echo "Submitted separate row ${row} (${label}) as ${job_id}."
done

remaining_after_batch=$(( ${#pending_rows[@]} - batch_count ))
echo "Current batch map: ${CURRENT_JOBS_LOG}"
echo "Remaining unfinished rows after this batch: ${remaining_after_batch}"
if (( remaining_after_batch > 0 )); then
  echo "After these jobs finish, run this same submission command again for the next batch."
fi
