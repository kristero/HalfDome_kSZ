"""Quantify HDF5 chunk re-decompression ('chunk amplification') of cell-sliced reads vs whole-chunk streaming, from the
L1_m9 0077 cell index (maas_cell_index_and_field_storage.json) and the selected cells of the particle passes."""
import json, os
from collections import OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012'
meta = json.load(open(os.path.join(HERE, 'maas_cell_index_and_field_storage.json')))
counts = np.array(meta['counts'], np.int64)
offs = np.array(meta['offsets_in_file'], np.int64)
files = np.array(meta['files'], np.int64)
CH = 1048576  # rows per HDF5 chunk (probe)
N = counts.sum()
# rows per chunk file and chunks per file
file_rows = np.bincount(files, weights=counts).astype(np.int64)
chunks_per_file = -(-file_rows // CH)
total_chunks = int(chunks_per_file.sum())

# per-field stored bytes per particle (file 0 + file 56 average) and logical bytes (chunk dtype)
f0, f56 = meta['chunk0'], meta['chunk56']
field = {}
for k in f0:
    n0, n56 = f0[k]['shape'][0], f56[k]['shape'][0]
    stored = (f0[k]['stored_bytes'] + f56[k]['stored_bytes']) / (n0 + n56)
    ncomp = int(np.prod(f0[k]['shape'][1:])) if len(f0[k]['shape']) > 1 else 1
    logical = np.dtype(f0[k]['dtype']).itemsize * ncomp
    field[k] = {'stored_B_per_particle': round(stored, 3), 'logical_B_per_particle': int(logical),
                'chunk_logical_MiB': CH * logical / 2**20, 'filters': f0[k]['filters']}
memb = meta['membership0']['GroupNr_bound']
field['membership:GroupNr_bound'] = {'stored_B_per_particle': round(memb['stored_bytes'] / memb['shape'][0], 3),
                                     'logical_B_per_particle': 8, 'chunk_logical_MiB': 10000 * 8 / 2**20,
                                     'filters': memb['filters'], 'chunk_rows': 10000}

def read_set(names):
    return (sum(field[n]['stored_B_per_particle'] for n in names), sum(field[n]['logical_B_per_particle'] for n in names))

SETS = {
  'pass_20261012 (13 fields + membership)': ['Coordinates', 'SmoothingLengths', 'Masses', 'Densities', 'ComptonYParameters',
      'ElectronNumberDensities', 'Temperatures', 'LastAGNFeedbackScaleFactors', 'ParticleIDs', 'SmoothedElementMassFractions',
      'Pressures', 'StarFormationRates', 'XrayLuminosities', 'membership:GroupNr_bound'],
  'radii234 (11 fields + membership)': ['Coordinates', 'SmoothingLengths', 'Masses', 'Densities', 'ComptonYParameters',
      'ElectronNumberDensities', 'Temperatures', 'LastAGNFeedbackScaleFactors', 'ParticleIDs', 'SmoothedElementMassFractions',
      'Pressures', 'membership:GroupNr_bound'],
  'minimal y + n_e, AGN flag, resolution, ownership': ['Coordinates', 'Masses', 'Densities', 'ComptonYParameters',
      'ElectronNumberDensities', 'Temperatures', 'LastAGNFeedbackScaleFactors', 'SmoothingLengths', 'membership:GroupNr_bound'],
  'minimal, no membership (spatial only)': ['Coordinates', 'Masses', 'Densities', 'ComptonYParameters',
      'ElectronNumberDensities', 'Temperatures', 'LastAGNFeedbackScaleFactors', 'SmoothingLengths'],
  'minimal + H/He (SmoothedElementMassFractions)': ['Coordinates', 'Masses', 'Densities', 'ComptonYParameters',
      'ElectronNumberDensities', 'Temperatures', 'LastAGNFeedbackScaleFactors', 'SmoothingLengths', 'SmoothedElementMassFractions', 'membership:GroupNr_bound'],
  'y only (Coordinates, Masses, ComptonY, LastAGN, Temperatures)': ['Coordinates', 'Masses', 'ComptonYParameters', 'Temperatures', 'LastAGNFeedbackScaleFactors'],
  'kSZ extras (+Velocities)': ['Coordinates', 'Masses', 'Densities', 'ComptonYParameters', 'ElectronNumberDensities', 'Temperatures',
      'LastAGNFeedbackScaleFactors', 'SmoothingLengths', 'Velocities', 'membership:GroupNr_bound'],
  'all 21 gas fields + membership': list(f0.keys()) + ['membership:GroupNr_bound'],
}
sets = {k: dict(zip(('stored_B_per_particle', 'logical_B_per_particle'), map(lambda x: round(x, 2), read_set(v))), fields=v) for k, v in SETS.items()}

def simulate(cells, cache_MiB, reopen_every, chunk_MiB):
    """Count chunk decompressions for cells processed per file in the given order with an LRU chunk cache of
    floor(cache/chunk) chunks per dataset (HDF5 bypasses the cache for chunks larger than it); cache cleared every
    `reopen_every` cells (file reopen)."""
    cap = int(cache_MiB // chunk_MiB)
    decomp = 0
    for fid in np.unique(files[cells]):
        fc = [c for c in cells if files[c] == fid]
        cache = OrderedDict()
        for i, c in enumerate(fc):
            if reopen_every and i % reopen_every == 0:
                cache.clear()
            lo, hi = offs[c], offs[c] + counts[c]
            if counts[c] == 0:
                continue
            for ch in range(lo // CH, (hi - 1) // CH + 1):
                if cap > 0 and ch in cache:
                    cache.move_to_end(ch)
                    continue
                decomp += 1
                if cap > 0:
                    cache[ch] = True
                    if len(cache) > cap:
                        cache.popitem(last=False)
    return decomp

res = {'gas_rows_total': int(N), 'cells': len(counts), 'chunk_rows': CH, 'total_hdf5_chunks_per_dataset': total_chunks,
       'fields': field, 'read_sets': sets}
for label, path in [('pass_20261012', os.path.join(STUDY, 'selected_halos.npz')),
                    ('radii234', '/home/cbllover/HalfDome/flamingo_pressure_calibration/radii234_contamination_L1_m9_0077_20261008/selected_halos.npz')]:
    cells = np.load(path)['cells'].astype(np.int64)
    rows = int(counts[cells].sum())
    touched = set()
    for c in cells:
        if counts[c] == 0:
            continue
        lo, hi = offs[c], offs[c] + counts[c]
        for ch in range(lo // CH, (hi - 1) // CH + 1):
            touched.add((int(files[c]), int(ch)))
    touched_rows = 0
    for fid, ch in touched:
        touched_rows += min(CH, file_rows[fid] - ch * CH)
    order_sorted = bool(np.all(np.diff(cells) > 0))
    sims = {}
    for cache, reopen in [(8, 0), (40, 16), (80, 16), (80, 0), (1024, 0)]:
        for cm in (36.0, 24.0, 8.0, 4.0):
            sims[f'cache{cache}MiB_reopen{reopen}_chunk{cm:g}MiB'] = simulate(list(cells), cache, reopen, cm)
    res[label] = {'cells': len(cells), 'rows_needed': rows, 'fraction_of_gas': rows / N,
                  'distinct_chunks_touched': len(touched), 'rows_in_touched_chunks': touched_rows,
                  'touched_chunk_rows_over_needed_rows': touched_rows / rows,
                  'touched_chunks_fraction_of_all': len(touched) / total_chunks,
                  'cells_sorted_ascending': order_sorted, 'decompressions_by_cache_model': sims,
                  'rows_per_cell_mean': rows / len(cells)}
json.dump(res, open(os.path.join(HERE, 'chunk_amplification.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('gas rows', N, 'chunks/dataset', total_chunks)
for k, v in sets.items():
    print(f"{k:60s} stored {v['stored_B_per_particle']:6.2f}  logical {v['logical_B_per_particle']:6.1f}")
for label in ('pass_20261012', 'radii234'):
    r = res[label]
    print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'decompressions_by_cache_model'})
    for k, v in r['decompressions_by_cache_model'].items():
        print('   ', k, v, round(v / r['distinct_chunks_touched'], 3))
