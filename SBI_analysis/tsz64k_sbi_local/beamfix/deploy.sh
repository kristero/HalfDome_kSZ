#!/bin/bash
# One-time setup of the beam-fix run root on idark ($HOME/tsz64k_beamfix_20260930): code, small inputs (checksums verified
# there) and the stage plan. Re-running only replaces code and inputs; outputs (nn/, upload/, stages/, series/) are kept.
set -euo pipefail
cd "$(dirname "$0")"
L=..                                                     # tsz64k_sbi_local/
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
RUN=tsz64k_beamfix_20260930
SSH="ssh.exe -o BatchMode=yes -o ConnectTimeout=60 -o ServerAliveInterval=30 idark"
B=deploy_bundle
rm -rf $B; mkdir -p $B/code $B/inputs/b12_mocks
cp beamfix.py regen_noise.jl regen.pbs stage.pbs $L/sbi_final.py $L/train_ensemble.py $L/analyze_64k.py $B/code/
cp $L/results_final/split.npz $B/inputs/
for f in theta clean_dl noisy_dl moped_raw_noisy row_id; do cp $L/data/rows_64440/$f.npy $B/inputs/; done
cp $L/data/rows_64440/meta.json $B/inputs/
cp /home/cbllover/HalfDome/tSZ_64k_flat_prior_SO_baseline_deproj0/design/noise_seeds_65536.npy $B/inputs/
cp $L/mocks/pass1_B12/B12/masked_clean_cl.npy $L/mocks/pass1_B12/B12/noisy_cross_*.npy $B/inputs/b12_mocks/
( cd $L && $PY - "$(pwd)/beamfix/$B" <<'PYEOF'
import sys, json, glob, shutil, numpy as np
import analyze_64k as A
out = sys.argv[1]
np.save(out + '/inputs/moped_matrix.npy', np.load(A.TEST256 / 'reference' / 'compression_context.npz')['Battaglia12_rcond1e-06'])
shutil.copy(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz', out + '/inputs/Battaglia12.npz')
ref = {}
for f in glob.glob('results_final/*/N*/metrics.json'):
    variant, n = f.split('/')[1], f.split('/')[2][1:]
    if not n.isdigit():
        continue
    m = json.load(open(f))
    ref.setdefault(variant, {})[n] = {k: m[k] for k in ('mean_log_prob_theta', 'mean_log_prob_theta_sem', 'pearson_r',
                                                        'tarp68', 'tarp95', 'median_width68_over_prior')}
open(out + '/inputs/results_final_metrics.json', 'w').write(json.dumps(ref, indent=1))
PYEOF
) 2>&1 | grep -v -i warn || true
( cd $B && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
du -sh $B
tar -C $B -cf - . | $SSH "mkdir -p $RUN/logs && tar -C $RUN -xf - && cd $RUN && sha256sum -c --quiet SHA256SUMS && echo checksums ok"
$SSH "cd $RUN && chmod +x code/*.pbs && /home/kristero10/envs/sbi27/bin/python code/beamfix.py --run \$HOME/$RUN plan | tail -5"
rm -rf $B
echo deployed
