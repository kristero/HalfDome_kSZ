"""Rows needed vs HDF5 chunks touched for halo samples of different sizes (4 R200c spheres, L1_m9 z=0), using the
4R preflight's (halo, cell) pairs and the maas cell index. Shows how little halo sampling saves in decompression."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, 'maas_cell_index_and_field_storage.json')))
counts = np.array(meta['counts'], np.int64); offs = np.array(meta['offsets_in_file'], np.int64); files = np.array(meta['files'], np.int64)
CH = 1048576
file_rows = np.bincount(files, weights=counts).astype(np.int64)
total_chunks = int((-(-file_rows // CH)).sum())
N = counts.sum()
P = np.load('/home/cbllover/HalfDome/flamingo_pressure_calibration/mass12p5_15p5_r4_preflight_L1_m9_0077_20261008/candidate_metadata.npz')
pair_halo, pair_cell = P['pair_halo'], P['pair_cell']
logm = np.log10(P['m200c_msun'])
def chunks_of(cells):
    s = set()
    for c in cells:
        if counts[c] == 0:
            continue
        lo, hi = offs[c], offs[c] + counts[c]
        for ch in range(lo // CH, (hi - 1) // CH + 1):
            s.add((int(files[c]), int(ch)))
    return s
def summary(halo_rows, label):
    m = np.isin(pair_halo, halo_rows)
    cells = np.unique(pair_cell[m])
    ch = chunks_of(cells)
    return {'label': label, 'haloes': int(len(halo_rows)), 'cells': int(len(cells)), 'rows_fraction': float(counts[cells].sum() / N),
            'chunks_touched_fraction': len(ch) / total_chunks}
out = []
above145 = np.flatnonzero(logm >= 14.5)
for k in (32, 64, 128, 256):
    rows = P[f'sample_{k}_rows']
    out.append(summary(np.union1d(rows, above145), f'random {k} per 0.2-dex bin 12.5-14.5 + all >= 14.5'))
for thr in (14.5, 14.0, 13.5, 13.0, 12.5):
    out.append(summary(np.flatnonzero(logm >= thr), f'all centrals >= 10^{thr}'))
json.dump({'total_chunks_per_dataset': total_chunks, 'samples': out}, open(os.path.join(HERE, 'sample_vs_chunks.json'), 'w'), indent=1)
for r in out:
    print(f"{r['label']:55s} haloes {r['haloes']:7d} cells {r['cells']:6d} rows {r['rows_fraction']:.3f} chunks {r['chunks_touched_fraction']:.3f}")
