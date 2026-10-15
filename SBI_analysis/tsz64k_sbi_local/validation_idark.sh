#!/bin/bash
# Independent validation rows: the 1,096 rows (274 batches) the collaborator run lost, regenerated on idark with the
# frozen engine. On idark: the bundle's own collector (manage.py collect --allow-missing: status, design theta/seeds,
# SHA-256 of every file, unique noise seeds) writes the dataset to /lustre next to rows_64440_small, then the raw
# fixed-B12 MOPED projections are added there. Locally: the small arrays go to data/rows_validation_1096 (checksums
# verified). Then: sbi_final.py validate.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
RUN=/home/kristero10/tsz64k_missing_20260928
BUNDLE=/lustre/work/kristero10/tsz64k_repro_test_20260924/HalfDome_kSZ/tSZ_64k_flat_prior_SO_baseline_deproj0
OUT=/lustre/work/kristero10/tsz64k_dataset_20260928/rows_validation_1096
LOCAL=data/rows_validation_1096
SSH="ssh.exe -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 idark"

n=$($SSH "ls $RUN/run_root/rows | wc -l")
echo "rows with a folder on idark: $n"
[ "$n" -eq 1096 ] || { echo "not finished"; exit 1; }
$SSH "test -e $OUT" && { echo "$OUT exists; remove it first"; exit 1; }
$SSH "cd $RUN && /home/anaconda3/bin/python3 $BUNDLE/pipeline/manage.py --config $RUN/config_groupA.toml collect --allow-missing --out $OUT" \
    | tail -n 3

# the fixed-B12 MOPED matrix of the analysis (7900 x 9), then the raw projections next to the collected arrays
MATRIX=$(mktemp --suffix=.npy)
$PY -c "
import numpy as np, sys
m = np.load('../tsz_diagnostic_256_accelerated_20260922/reference/compression_context.npz')['Battaglia12_rcond1e-06']
np.save(sys.argv[1], np.ascontiguousarray(m))" "$MATRIX"
$SSH "cat > $OUT/moped_matrix_Battaglia12_rcond1e-06.npy" < "$MATRIX"
[ "$($SSH "sha256sum < $OUT/moped_matrix_Battaglia12_rcond1e-06.npy" | cut -d' ' -f1)" = "$(sha256sum < "$MATRIX" | cut -d' ' -f1)" ] \
    || { echo "MOPED matrix upload corrupted"; exit 1; }
rm -f "$MATRIX"
$SSH "cd $OUT && /home/anaconda3/bin/python3 -c \"
import json, numpy as np
m = np.load('moped_matrix_Battaglia12_rcond1e-06.npy')
assert m.shape == (7900, 9)
for kind in ('clean', 'noisy'):
    np.save('moped_raw_%s.npy' % kind, np.load('%s_dl_unbinned.npy' % kind, mmap_mode='r') @ m)
meta = json.load(open('meta.json'))
meta['missing_rows'] = 'all rows except row_id.npy (the 1,096 rows of the 274 batches the collaborator run lost)'
meta['moped'] = 'moped_raw_* = D_ell_unbinned @ moped_matrix_Battaglia12_rcond1e-06.npy (compression_context.npz of the 256-row test)'
meta['generated'] = 'idark (IPMU), job arrays miss_A/miss_B, 2026-09-28/29, run root $RUN'
json.dump(meta, open('meta.json', 'w'), indent=1)
\" && sha256sum *.npy meta.json > SHA256SUMS && ls -la"

SMALL="theta.npy row_id.npy noise_seeds.npy ell_unbinned.npy clean_dl.npy noisy_dl.npy moped_raw_clean.npy moped_raw_noisy.npy meta.json SHA256SUMS"
mkdir -p "$LOCAL"
$SSH "cd $OUT && tar -cf - $SMALL" | tar -C "$LOCAL" -xf -
(cd "$LOCAL" && grep -E " ($(echo $SMALL | sed 's/ SHA256SUMS//; s/ /|/g'))$" SHA256SUMS | sha256sum -c -)
echo "validation rows in $LOCAL; full dataset (with the unbinned spectra) on idark: $OUT"
