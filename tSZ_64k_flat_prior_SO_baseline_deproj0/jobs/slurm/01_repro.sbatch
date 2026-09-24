#!/bin/bash
#SBATCH --job-name=tsz64k_repro
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Rerun rows 0-3 of the validated 256-row test and compare (about 20 min on 26 cores).
# Submit with jobs/slurm/submit.sh, which adds --export and --output.
exec bash "${BUNDLE:?submit with jobs/slurm/submit.sh}/jobs/stage.sh" repro
