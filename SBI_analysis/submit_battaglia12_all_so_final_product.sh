#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p /home/kristero10/logs/tSZ_two_param

qsub -v "PROJECT_DIR=${PROJECT_DIR}" \
  "${SCRIPT_DIR}/run_battaglia12_all_so_final_product.pbs"
