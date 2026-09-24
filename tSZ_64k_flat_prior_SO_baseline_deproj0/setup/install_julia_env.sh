#!/usr/bin/env bash
# Install Julia 1.12.2 and the pinned package environment of this bundle.
#
#   setup/install_julia_env.sh /path/to/julia_tsz64k            # install + precompile
#   setup/install_julia_env.sh /path/to/julia_tsz64k --no-precompile
#
# Needs internet access (julialang.org, pkg.julialang.org, github.com). The
# precompile step takes 10-20 minutes of CPU; on clusters that forbid heavy
# login-node work, use --no-precompile there and run slurm/00_precompile.sbatch.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:?usage: install_julia_env.sh <install_dir> [--no-precompile]}"
PRECOMPILE=1
[[ "${2:-}" == "--no-precompile" ]] && PRECOMPILE=0
ARCHIVE=julia-1.12.2-linux-x86_64.tar.gz
# SHA-256 of the archive used by the validated 256-row test.
EXPECTED="a6d0c39ea57303ebcffa7a8d453429b86eb271e150c7cb0f5958fe65909b493a  ${ARCHIVE}"

mkdir -p "${PREFIX}/depot"
cd "${PREFIX}"
if [[ ! -x julia-1.12.2/bin/julia ]]; then
  curl --fail --retry 3 --connect-timeout 30 -L -o julia-1.12.2.sha256 \
    https://julialang-s3.julialang.org/bin/checksums/julia-1.12.2.sha256
  curl --fail --retry 3 --connect-timeout 30 -L -C - -o "${ARCHIVE}" \
    "https://julialang-s3.julialang.org/bin/linux/x64/1.12/${ARCHIVE}"
  awk -v archive="${ARCHIVE}" '$2 == archive {print}' julia-1.12.2.sha256 > selected_checksum.txt
  [[ "$(cat selected_checksum.txt)" == "${EXPECTED}" ]] || { echo "Unexpected Julia checksum list" >&2; exit 1; }
  sha256sum -c selected_checksum.txt
  tar -xzf "${ARCHIVE}"
fi

JULIA="${PREFIX}/julia-1.12.2/bin/julia"
export JULIA_DEPOT_PATH="${PREFIX}/depot"
export LD_LIBRARY_PATH="${PREFIX}/julia-1.12.2/lib/julia:${LD_LIBRARY_PATH:-}"
export JULIA_PKG_PRECOMPILE_AUTO=0
unset JULIA_PROJECT JULIA_LOAD_PATH

"${JULIA}" --startup-file=no --project="${BUNDLE}/julia_env" -e 'using Pkg; Pkg.instantiate()'
"${JULIA}" --startup-file=no --project="${BUNDLE}/julia_env" "${BUNDLE}/setup/verify_env.jl"
if [[ "${PRECOMPILE}" == 1 ]]; then
  "${JULIA}" --startup-file=no --threads=auto --project="${BUNDLE}/julia_env" -e 'using Pkg; Pkg.precompile()'
fi

cat <<EOF

Julia environment ready. In pipeline/config.toml set:
  julia       = "${JULIA}"
  julia_depot = "${PREFIX}/depot"
EOF
