#!/bin/bash
#SBATCH --job-name=tsz64k_audit
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# One array element = one audit task of 256 rows (typically 30-90 min on 6 cores).
# Submit with jobs/slurm/submit.sh audit, which sets --array and ARRAY_OFFSET.
exec bash "${BUNDLE:?submit with jobs/slurm/submit.sh}/jobs/stage.sh" audit
