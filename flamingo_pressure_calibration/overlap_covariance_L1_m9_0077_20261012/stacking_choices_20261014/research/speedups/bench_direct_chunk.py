"""Benchmark: decode SWIFT nbit+shuffle+deflate+fletcher32 chunks ourselves (h5py read_direct_chunk + zlib + numpy
unshuffle + numba nbit unpack) vs the HDF5 filter pipeline, on synthetic FMantissa9 / DMantissa9 data with FLAMINGO
chunking. Verifies bit-exact equality with h5py. Temporary file under bench_tmp/ is deleted at the end."""
import json, os, time, zlib
import numpy as np
import h5py
from h5py import h5t, h5p, h5s, h5d
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, 'bench_tmp', 'direct.hdf5')
CH, NCH = 1048576, 4
N = CH * NCH
rng = np.random.default_rng(2)

def ftype(kind):
    if kind == 'FMantissa9':
        # FLAMINGO files use precision 18 at offset 0 (maas_filter_parameters.json); HDF5 1.14.6 refuses to create that
        # 'unusual datatype' here, so the same 18 significant bits are placed at offset 14 (identical nbit stream length).
        t = h5t.IEEE_F32LE.copy(); t.set_fields(31, 23, 8, 14, 9); t.set_offset(14); t.set_precision(18); t.set_ebias(127); return t, 18, 14, 4
    t = h5t.IEEE_F64LE.copy(); t.set_fields(63, 52, 11, 43, 9); t.set_offset(43); t.set_precision(21); t.set_ebias(1023); return t, 21, 43, 8

def make(f, name, data, kind):
    t = ftype(kind)[0]
    space = h5s.create_simple(data.shape, data.shape)
    dcpl = h5p.create(h5p.DATASET_CREATE); dcpl.set_chunk((CH,) + data.shape[1:])
    dcpl.set_filter(5, 0); dcpl.set_shuffle(); dcpl.set_deflate(4); dcpl.set_fletcher32()
    h5d.create(f.id, name.encode(), t, space, dcpl=dcpl).write(h5s.ALL, h5s.ALL, np.ascontiguousarray(data))

@njit(cache=False)
def nbit_unpack(buf, n, prec, off, out):
    mask = (np.uint64(1) << np.uint64(prec)) - np.uint64(1)
    nb = buf.shape[0]
    for i in range(n):
        bit = i * prec
        byte = bit >> 3
        sh = bit & 7
        w = np.uint64(0)
        for k in range(8):
            w = w << np.uint64(8)
            if byte + k < nb:
                w = w | np.uint64(buf[byte + k])
        v = (w >> np.uint64(64 - prec - sh)) & mask
        out[i] = v << np.uint64(off)

def decode_chunk(raw, n, prec, off, size):
    # size = shuffle element size read from the dataset's filter parameters (cd_values[0]); not always the type size
    body = raw[:-4]                                   # strip fletcher32 checksum (not verified here)
    t0 = time.perf_counter(); inflated = zlib.decompress(body); t1 = time.perf_counter()
    b = np.frombuffer(inflated, np.uint8)
    ne = b.size // size
    un = np.empty_like(b)
    un[:ne * size] = b[:ne * size].reshape(size, ne).T.ravel()
    un[ne * size:] = b[ne * size:]
    t2 = time.perf_counter()
    out = np.empty(n, np.uint64)
    nbit_unpack(un, n, prec, off, out)
    t3 = time.perf_counter()
    return out, (t1 - t0, t2 - t1, t3 - t2)

data = {'ComptonY_FMantissa9': (np.exp(rng.normal(-20, 2, N)).astype(np.float32), 'FMantissa9'),
        'SEMF_FMantissa9_x9': (rng.uniform(0, 1, (N, 9)).astype(np.float32), 'FMantissa9'),
        'ElectronNumberDensities_DMantissa9': (np.exp(rng.normal(60, 2, N)), 'DMantissa9')}
os.makedirs(os.path.dirname(TMP), exist_ok=True)
with h5py.File(TMP, 'w') as f:
    for k, (a, kind) in data.items():
        make(f, k, a, kind)
# warm up numba
nbit_unpack(np.zeros(16, np.uint8), 4, 18, 14, np.empty(4, np.uint64))
res = {}
for k, (a, kind) in data.items():
    _, prec, off, size = ftype(kind)
    memtype = np.float32 if size == 4 else np.float64
    with h5py.File(TMP, 'r', rdcc_nbytes=200 * 2**20) as f:
        d = f[k]
        plist = d.id.get_create_plist()
        shuffle_size = [plist.get_filter(i)[2][0] for i in range(plist.get_nfilters()) if plist.get_filter(i)[0] == 2][0]
        t0 = time.perf_counter(); ref = d.astype(memtype)[...]; t_ref = time.perf_counter() - t0
        per_chunk_elems = CH * (int(np.prod(a.shape[1:])) if a.ndim > 1 else 1)
        steps = np.zeros(3); outs = []
        t0 = time.perf_counter()
        for c in range(NCH):
            offset = (c * CH,) + (0,) * (a.ndim - 1)
            mask, raw = d.id.read_direct_chunk(offset)
            u, st = decode_chunk(raw, per_chunk_elems, prec, off, shuffle_size)
            steps += st
            outs.append(u)
        t_direct = time.perf_counter() - t0
    bits = np.concatenate(outs)
    got = (bits.astype(np.uint32).view(np.float32) if size == 4 else bits.view(np.float64)).reshape(ref.shape)
    exact = bool(np.array_equal(got.view(np.uint32 if size == 4 else np.uint64), ref.view(np.uint32 if size == 4 else np.uint64)))
    res[k] = {'hdf5_pipeline_ns_per_row': 1e9 * t_ref / N, 'direct_decode_ns_per_row': 1e9 * t_direct / N,
              'speedup': t_ref / t_direct, 'bit_exact': exact, 'shuffle_element_size': int(shuffle_size),
              'direct_steps_ns_per_row': {'inflate': 1e9 * steps[0] / N, 'unshuffle': 1e9 * steps[1] / N, 'nbit_unpack_numba': 1e9 * steps[2] / N}}
os.remove(TMP)
json.dump(res, open(os.path.join(HERE, 'bench_direct_chunk.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
