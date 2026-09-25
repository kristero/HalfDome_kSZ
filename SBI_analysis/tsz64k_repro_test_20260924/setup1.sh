set -euo pipefail
ROOT=/lustre/work/kristero10/tsz64k_repro_test_20260924
cd "$ROOT"
export GIT_TERMINAL_PROMPT=0
echo "== fresh clone from GitHub, as the README instructs  $(date -Is)"
git -c advice.detachedHead=false -c credential.helper= clone -q --filter=blob:none --sparse \
  --branch tsz64k-flat-prior-v1 https://github.com/kristero/HalfDome_kSZ.git
cd HalfDome_kSZ
git sparse-checkout set tSZ_64k_flat_prior_SO_baseline_deproj0
git log -1 --format='commit %H  %s'
echo "files: $(git ls-files tSZ_64k_flat_prior_SO_baseline_deproj0 | wc -l)"
cd tSZ_64k_flat_prior_SO_baseline_deproj0
echo "== Julia 1.12.2 + packages from the internet into a new depot  $(date -Is)"
setup/install_julia_env.sh "$ROOT/julia_tsz64k" --no-precompile
echo "== done  $(date -Is)"
