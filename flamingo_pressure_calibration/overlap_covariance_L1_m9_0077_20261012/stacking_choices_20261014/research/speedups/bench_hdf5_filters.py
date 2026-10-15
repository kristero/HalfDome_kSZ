"""Local micro-benchmark of the SWIFT/FLAMINGO HDF5 filter pipelines (nbit + custom reduced-mantissa floats, scale-offset,
shuffle, deflate, fletcher32) on SYNTHETIC data with FLAMINGO chunking (1,048,576 rows). Measures relative read costs:
per field type, whole-chunk vs cell-sliced reads with small/large chunk caches. Absolute numbers are for this machine
and synthetic data; use the ratios. Temporary HDF5 file under bench_tmp/ is deleted at the end."""
import json, os, time
import numpy as np
import h5py
from h5py import h5t, h5p, h5s, h5d

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, 'bench_tmp', 'bench.hdf5')
CH = 1048576
NCH = 4
N = CH * NCH
rng = np.random.default_rng(1)

def ftype(kind):
    if kind == 'FMantissa9':
        t = h5t.IEEE_F32LE.copy(); t.set_fields(31, 23, 8, 14, 9); t.set_offset(14); t.set_precision(18); t.set_ebias(127); return t
    if kind == 'DMantissa9':
        t = h5t.IEEE_F64LE.copy(); t.set_fields(63, 52, 11, 43, 9); t.set_offset(43); t.set_precision(21); t.set_ebias(1023); return t
    if kind == 'f4':
        return h5t.IEEE_F32LE.copy()
    if kind == 'f8':
        return h5t.IEEE_F64LE.copy()
    if kind == 'u8':
        return h5t.STD_U64LE.copy()
    raise ValueError(kind)

def make(f, name, data, kind, nbit=False, scaleoffset=None, ncomp=1):
    shape = data.shape
    space = h5s.create_simple(shape, shape)
    dcpl = h5p.create(h5p.DATASET_CREATE)
    dcpl.set_chunk((CH,) + shape[1:])
    if nbit:
        dcpl.set_filter(5, 0)  # H5Z_FILTER_NBIT; parameters filled by its set_local callback
    if scaleoffset is not None:
        dcpl.set_scaleoffset(h5py.h5z.SO_FLOAT_DSCALE, scaleoffset)
    dcpl.set_shuffle(); dcpl.set_deflate(4); dcpl.set_fletcher32()
    ds = h5d.create(f.id, name.encode(), ftype(kind), space, dcpl=dcpl)
    ds.write(h5s.ALL, h5s.ALL, np.ascontiguousarray(data))

# synthetic, spatially sorted-ish data
cellpos = np.sort(rng.uniform(0, 31.25, N))
coords = np.stack([cellpos, rng.uniform(0, 31.25, N), rng.uniform(0, 31.25, N)], axis=1)
ln = lambda mu, s: np.exp(rng.normal(mu, s, N))
fields = {
    'Coordinates_DScale5': (coords, 'f8', dict(scaleoffset=5)),
    'Velocities_DScale1_f4': (rng.normal(0, 300, (N, 3)).astype(np.float32), 'f4', dict(scaleoffset=1)),
    'ComptonY_FMantissa9': (ln(-20, 2).astype(np.float32), 'FMantissa9', dict(nbit=True)),
    'Densities_FMantissa9': (ln(3, 2).astype(np.float32), 'FMantissa9', dict(nbit=True)),
    'ElectronNumberDensities_DMantissa9': (ln(60, 2), 'DMantissa9', dict(nbit=True)),
    'SEMF_FMantissa9_x9': (rng.uniform(0, 1, (N, 9)).astype(np.float32), 'FMantissa9', dict(nbit=True)),
    'Masses_f4_lossless': (np.full(N, 1.07e9, np.float32) * (1 + 0.01 * rng.random(N, dtype=np.float32)), 'f4', {}),
    'SmoothingLengths_f4_lossless': (ln(-3, 0.7).astype(np.float32), 'f4', {}),
    'ParticleIDs_u8': (rng.integers(0, 2**40, N).astype(np.uint64), 'u8', {}),
}
os.makedirs(os.path.dirname(TMP), exist_ok=True)
with h5py.File(TMP, 'w') as f:
    for name, (data, kind, kw) in fields.items():
        make(f, name, data, kind, **kw)
res = {'rows': N, 'chunk_rows': CH, 'note': 'synthetic data; deflate level 4 assumed; reads into the numpy dtype h5py 3.11 reports on maas (custom FMantissa9 -> float32, DMantissa9 -> float64)', 'fields': {}}
MEMTYPE = {'FMantissa9': np.float32, 'DMantissa9': np.float64, 'f4': np.float32, 'f8': np.float64, 'u8': np.uint64}
KIND = {name: kind for name, (data, kind, kw) in fields.items()}
KIND.update({'ComptonY_plain_f4': 'f4', 'ElectronNumberDensities_plain_f8': 'f8', 'Coordinates_plain_f8': 'f8'})
def rd(d, name, sel=slice(None)):
    return d.astype(MEMTYPE[KIND[name]])[sel]
# 1) whole-dataset read cost (ns per row) and stored bytes per row
for name in fields:
    times = []
    for rep in range(2):
        with h5py.File(TMP, 'r', rdcc_nbytes=200 * 2**20) as f:
            d = f[name]
            t0 = time.perf_counter(); a = rd(d, name); times.append(time.perf_counter() - t0)
            stored = d.id.get_storage_size()
            logical = a.nbytes
    res['fields'][name] = {'ns_per_row_whole_read': 1e9 * min(times) / N, 'stored_B_per_row': stored / N,
                           'logical_B_per_row': logical / N, 'MB_per_s_logical': logical / min(times) / 1e6}
# 2) native-type control: same values written as plain f4/f8 with shuffle+deflate+fletcher32 (no nbit / scaleoffset)
with h5py.File(TMP, 'a') as f:
    make(f, 'ComptonY_plain_f4', fields['ComptonY_FMantissa9'][0], 'f4')
    make(f, 'ElectronNumberDensities_plain_f8', fields['ElectronNumberDensities_DMantissa9'][0], 'f8')
    make(f, 'Coordinates_plain_f8', coords, 'f8')
for name in ('ComptonY_plain_f4', 'ElectronNumberDensities_plain_f8', 'Coordinates_plain_f8'):
    times = []
    for rep in range(2):
        with h5py.File(TMP, 'r', rdcc_nbytes=200 * 2**20) as f:
            d = f[name]; t0 = time.perf_counter(); a = rd(d, name); times.append(time.perf_counter() - t0)
            stored = d.id.get_storage_size()
    res['fields'][name] = {'ns_per_row_whole_read': 1e9 * min(times) / N, 'stored_B_per_row': stored / N,
                           'logical_B_per_row': a.nbytes / N, 'MB_per_s_logical': a.nbytes / min(times) / 1e6}
# 3) cell-sliced reads (cells of 214,531 rows = mean selected cell) vs whole chunks, chunk cache 8 vs 80 MiB
cell = 214531
starts = np.arange(0, N - cell, cell)
for name in ('Coordinates_DScale5', 'SEMF_FMantissa9_x9', 'ComptonY_FMantissa9'):
    for cache in (8, 40, 48, 80):
        times = []
        for rep in range(2):
            with h5py.File(TMP, 'r', rdcc_nbytes=cache * 2**20, rdcc_nslots=10007) as f:
                d = f[name]; t0 = time.perf_counter()
                for s in starts:
                    a = rd(d, name, slice(s, s + cell))
                times.append(time.perf_counter() - t0)
        rows = len(starts) * cell
        res['fields'][name][f'ns_per_row_cell_slices_cache{cache}MiB'] = 1e9 * min(times) / rows
    # whole chunks one at a time
    times = []
    for rep in range(2):
        with h5py.File(TMP, 'r', rdcc_nbytes=1 * 2**20) as f:
            d = f[name]; t0 = time.perf_counter()
            for c in range(NCH):
                a = rd(d, name, slice(c * CH, (c + 1) * CH))
            times.append(time.perf_counter() - t0)
    res['fields'][name]['ns_per_row_whole_chunks_cache1MiB'] = 1e9 * min(times) / N
os.remove(TMP)
json.dump(res, open(os.path.join(HERE, 'bench_hdf5_filters.json'), 'w'), indent=1)
for k, v in res['fields'].items():
    print(f"{k:38s} " + ' '.join(f"{kk}={vv:.2f}" for kk, vv in v.items()))
