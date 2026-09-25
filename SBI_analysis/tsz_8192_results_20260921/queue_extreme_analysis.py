"""Queue the small analysis once both new matched-noise controls finish."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/tsz_8192_noise_extremes_20260921'
pbs='''#!/bin/bash
#PBS -N tsz_extreme_report
#PBS -q mini2
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:20:00
#PBS -j oe
set -euo pipefail
cd /lustre/work/kristero10/tsz_8192_noise_extremes_20260921
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/anaconda3/bin/python3 analyze_extreme_noise.py
'''
path=ROOT/'extreme_analysis.pbs';path.write_bytes(pbs.encode())
subprocess.run(['scp',str(path),str(ROOT/'analyze_extreme_noise.py'),'idark:'+REMOTE+'/'],check=True)
jobs=json.loads((ROOT/'results/extreme_noise_submission.json').read_text())
dependency='afterok:'+':'.join(r['job'] for r in jobs)
job=subprocess.check_output(['ssh','idark','qsub','-W','depend='+dependency,'-o',REMOTE+'/analysis.log',REMOTE+'/extreme_analysis.pbs'],text=True).strip()
(ROOT/'results/extreme_analysis_submission.json').write_text(json.dumps(dict(job=job,depends_on=jobs),indent=2))
print(job)
