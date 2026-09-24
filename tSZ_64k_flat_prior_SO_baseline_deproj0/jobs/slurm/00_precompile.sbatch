#!/bin/bash
#SBATCH --job-name=tsz64k_precompile
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Precompile julia_env on a compute node (after setup/install_julia_env.sh --no-precompile).
# Submit with jobs/slurm/submit.sh, which adds --export and --output.
exec bash "${BUNDLE:?submit with jobs/slurm/submit.sh}/jobs/stage.sh" precompile
