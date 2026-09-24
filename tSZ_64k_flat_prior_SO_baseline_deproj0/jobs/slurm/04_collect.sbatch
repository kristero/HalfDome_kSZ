#!/bin/bash
#SBATCH --job-name=tsz64k_collect
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Verify every row and write the dataset arrays; COLLECT_ARGS passes options
# such as "--rows 8192 --npz".
# Submit with jobs/slurm/submit.sh collect.
exec bash "${BUNDLE:?submit with jobs/slurm/submit.sh}/jobs/stage.sh" collect
