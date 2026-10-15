#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/kn18001/.conda/envs/class_sz/bin/python}"
export PATH_TO_CLASS_SZ_DATA="${PATH_TO_CLASS_SZ_DATA:-/home/kn18001/class_sz_data_directory}"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
exec "$PYTHON" -u "$SCRIPT_DIR/reproduce_tau_ksz.py" "$@"
