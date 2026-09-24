#!/bin/bash
#SBATCH --job-name=tsz64k_prod
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# One array element = BATCHES_PER_TASK batches of 4 rows (default 4 batches, about
# 20 min each on 26 cores in the test), or with TSZ64K_STAGE=pool up to POOL_BATCHES
# batches claimed from whatever is still pending. With more batches per task, raise
# --time and STOP_AFTER_HOURS (no new batch is started after that many hours).
# Submit with jobs/slurm/submit.sh production|pool, which sets --array and ARRAY_OFFSET.
exec bash "${BUNDLE:?submit with jobs/slurm/submit.sh}/jobs/stage.sh" "${TSZ64K_STAGE:-production}"
