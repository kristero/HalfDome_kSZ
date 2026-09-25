#!/usr/bin/env bash
# Register the "Julia 1.12.2 tSZ64k (20 threads)" Jupyter kernel for the current user.
# The kernel runs the local Julia 1.12.2 with SBI_analysis/tsz_local_profile_tests/julia_env
# (the validated tSZ_64k manifest plus IJulia and Plots) and 20 threads.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/julia-1.12.2-tsz64k"
mkdir -p "$TARGET"
cp "$HERE/kernelspec/julia-1.12.2-tsz64k/kernel.json" "$TARGET/kernel.json"
echo "installed $TARGET/kernel.json"
