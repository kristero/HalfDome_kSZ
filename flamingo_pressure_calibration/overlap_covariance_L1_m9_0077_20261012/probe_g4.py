"""G4 dataset probe (read-only, tiny): X-ray luminosity source/column order, SFR sign, IDs, SOAP columns.

The remote program is sent on stdin to the existing maas venv (tcsh login shell: no shell
redirections). It loads the frozen dependencies/common.py and flamingo_io.py from an embedded
payload as in-memory modules, so nothing is written on maas. It reads:
  * attributes/shapes/virtual-source file names of PartType0 datasets (top-level and chunk files);
  * one smallest non-empty cell through the chunk file and through the top-level (virtual) file;
  * the first <= 200,000 rows of one cell around a ~1e13 Msun radii234 halo (SFR sign, y = 0 checks);
  * SOAP dataset names/shapes/dtypes/descriptions (metadata only, no columns).
Output: probe_g4.json (local). Usage: python -B probe_g4.py  (or python -B control.py probe).
"""
import base64
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PYTHON = '/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python'

REMOTE_BODY = r'''
import time, socket, hashlib
import numpy as np, h5py, scipy
import common as C, flamingo_io as F
started = time.time()
ROOT = '/net/hypernova/data2/FLAMINGO'
RUN = 'L1000N1800/HYDRO_FIDUCIAL'
SNAPSHOT = RUN + '/snapshots/flamingo_0077/flamingo_0077.hdf5'
SOAP = RUN + '/SOAP-HBT/halo_properties_0077.hdf5'
MEMBERSHIP = ROOT + '/' + RUN + '/SOAP-HBT/membership_0077/membership_0077.{i}.hdf5'
PROBE_FIELDS = ['ParticleIDs', 'XrayLuminosities', 'XrayPhotonLuminosities', 'StarFormationRates',
                'ComptonYParameters', 'ElectronNumberDensities', 'Temperatures', 'Masses',
                'LastAGNFeedbackScaleFactors', 'Densities']
out = {'probe': 'G4', 'host': socket.getfqdn(), 'load': list(__import__('os').getloadavg()),
       'versions': {'numpy': np.__version__, 'h5py': h5py.__version__, 'scipy': scipy.__version__,
                    'hdf5': h5py.version.hdf5_version, 'python': __import__('sys').version}}

def plain(v):
    if isinstance(v, bytes): return v.decode(errors='replace')
    if isinstance(v, np.ndarray):
        if v.dtype.kind == 'S': return [x.decode(errors='replace') for x in v.ravel().tolist()]
        return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    return v

def describe(ds):
    info = {'dtype': str(ds.dtype), 'shape': list(ds.shape), 'chunks': list(ds.chunks) if ds.chunks else None,
            'compression': ds.compression, 'is_virtual': bool(ds.is_virtual),
            'attrs': {k: plain(v) for k, v in ds.attrs.items()}}
    if ds.is_virtual:
        sources = [s.file_name for s in ds.virtual_sources()]
        names = sorted(set(sources))
        info['virtual_source_count'] = len(sources)
        info['virtual_source_files_first'] = names[:3]
        info['virtual_source_files_distinct'] = len(names)
        info['virtual_source_dataset_first'] = ds.virtual_sources()[0].dset_name
    return info

def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    r = {'dtype_chunk': str(a.dtype), 'dtype_top': str(b.dtype), 'shape': list(a.shape),
         'bytes_identical': bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())}
    if a.shape != b.shape:
        r['values_identical'] = False; return r
    wa, wb = a.astype(np.float64), b.astype(np.float64)
    same = (wa == wb) | (np.isnan(wa) & np.isnan(wb))
    r['values_identical'] = bool(same.all())
    r['n_values'] = int(a.size); r['n_differ'] = int((~same).sum())
    if (~same).any():
        rel = np.abs(wa - wb) / np.maximum(np.maximum(np.abs(wa), np.abs(wb)), 1e-300)
        r['max_relative_difference'] = float(rel[~same].max())
        r['median_relative_difference_of_differing'] = float(np.median(rel[~same]))
        if a.ndim == 2:
            r['per_column_n_differ'] = (~same).sum(axis=0).tolist()
            r['per_column_median_ratio_top_over_chunk'] = [
                float(np.median(wb[:, j][wa[:, j] > 0] / wa[:, j][wa[:, j] > 0])) if (wa[:, j] > 0).any() else None
                for j in range(a.shape[1])]
    return r

# ---- top-level and chunk 0 dataset metadata
with h5py.File(ROOT + '/' + SNAPSHOT, 'r') as top:
    gas = top['PartType0']
    out['top_gas_datasets'] = sorted(gas.keys())
    out['top_fields'] = {n: describe(gas[n]) for n in PROBE_FIELDS if n in gas}
    named = {}
    if 'SubgridScheme/NamedColumns' in top:
        for n, ds in top['SubgridScheme/NamedColumns'].items():
            named[n] = plain(ds[...])
    out['top_named_columns'] = named
    out['top_header_NumFilesPerSnapshot'] = plain(top['Header'].attrs['NumFilesPerSnapshot'])
snap = F.Snapshot(F.Store(ROOT), SNAPSHOT, 77, read_from='chunks', chunk_cache_bytes=8 * 1024**2)
out['snapshot_describe'] = snap.describe()
out['auxiliary_pipeline_fields'] = snap.auxiliary_fields()
out['auxiliary_probe_fields'] = snap.auxiliary_fields(tuple(PROBE_FIELDS))
chunk0 = snap._chunk(0)
chunk0_rows = int(chunk0['PartType0/Masses'].shape[0])
out['chunk0_gas_datasets'] = sorted(chunk0['PartType0'].keys())
out['chunk0_fields'] = {n: describe(chunk0['PartType0'][n]) for n in PROBE_FIELDS if n in chunk0['PartType0']}
named = {}
if 'SubgridScheme/NamedColumns' in chunk0:
    for n, ds in chunk0['SubgridScheme/NamedColumns'].items():
        named[n] = plain(ds[...])
out['chunk0_named_columns'] = named

# ---- smallest non-empty cell through both paths
positive = np.flatnonzero(snap.counts > 0)
cell = int(positive[np.argmin(snap.counts[positive])])
fields = [n for n in PROBE_FIELDS if n in out['top_gas_datasets'] and n in out['chunk0_gas_datasets']]
via_chunk = snap.read_cell_raw(cell, fields, source='chunks')
via_top = snap.read_cell_raw(cell, fields, source='top')
out['small_cell'] = {'cell': cell, 'count': int(snap.counts[cell]), 'file': int(snap.cell_file[cell]),
                     'offset_in_file': int(snap.cell_offset[cell]), 'global_offset': int(snap.top_global_offset[cell]),
                     'fields': {n: compare(via_chunk[n], via_top[n]) for n in fields}}

# ---- first rows of one cell around a ~1e13 Msun radii234 halo (SFR sign convention, y/ne zero for SF gas)
centre = np.array(CENTRE_CMPC)
cell2 = int(snap.cell_of(centre[None])[0])
start, count = int(snap.cell_offset[cell2]), int(snap.counts[cell2])
rows = min(count, 200000)
handle = snap._handle(cell2)
part = {n: handle['PartType0/' + n][start:start + rows] for n in
        ('StarFormationRates', 'ComptonYParameters', 'ElectronNumberDensities', 'Temperatures', 'ParticleIDs', 'XrayLuminosities')}
gstart = int(snap.top_global_offset[cell2])
with h5py.File(ROOT + '/' + SNAPSHOT, 'r') as top:
    top_xray = top['PartType0/XrayLuminosities'][gstart:gstart + rows]
    top_ids = top['PartType0/ParticleIDs'][gstart:gstart + rows]
    top_sfr = top['PartType0/StarFormationRates'][gstart:gstart + rows]
sfr = part['StarFormationRates'].astype(np.float64)
y = part['ComptonYParameters'].astype(np.float64)
ne = part['ElectronNumberDensities'].astype(np.float64)
T = part['Temperatures'].astype(np.float64)
out['halo_cell_slice'] = {
    'cell': cell2, 'cell_count': count, 'rows_read': rows, 'file': int(snap.cell_file[cell2]),
    'sfr_positive': int((sfr > 0).sum()), 'sfr_negative': int((sfr < 0).sum()), 'sfr_zero': int((sfr == 0).sum()),
    'sfr_negative_range': [float(sfr[sfr < 0].min()), float(sfr[sfr < 0].max())] if (sfr < 0).any() else None,
    'sfr_positive_range': [float(sfr[sfr > 0].min()), float(sfr[sfr > 0].max())] if (sfr > 0).any() else None,
    'y_zero_where_sfr_positive': bool(np.all(y[sfr > 0] == 0)),
    'y_zero_count_where_sfr_nonpositive': int((y[sfr <= 0] == 0).sum()),
    'ne_zero_where_sfr_positive': bool(np.all(ne[sfr > 0] == 0)),
    'temperature_sfr_positive_16_50_84': np.percentile(T[sfr > 0], [16, 50, 84]).tolist() if (sfr > 0).any() else None,
    'temperature_below_1e5_and_sfr_nonpositive': int(((T < 1e5) & (sfr <= 0)).sum()),
    'xray_chunk_vs_top': compare(part['XrayLuminosities'], top_xray),
    'ids_chunk_vs_top_identical': bool(np.array_equal(part['ParticleIDs'], top_ids)),
    'sfr_chunk_vs_top': compare(part['StarFormationRates'], top_sfr),
    'xray_column_sums_chunk': part['XrayLuminosities'].astype(np.float64).sum(axis=0).tolist(),
    'xray_column_sums_top': top_xray.astype(np.float64).sum(axis=0).tolist(),
    'xray_zero_where_sfr_positive_chunk': bool(np.all(part['XrayLuminosities'][sfr > 0] == 0)),
    'xray_zero_where_sfr_positive_top': bool(np.all(top_xray[sfr > 0] == 0)),
}

# ---- membership file 0: metadata only
with h5py.File(MEMBERSHIP.format(i=0), 'r') as m:
    names = []
    m.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    out['membership0'] = {n: {'dtype': str(m[n].dtype), 'shape': list(m[n].shape)} for n in names if n.startswith('PartType0')}
out['membership0_rows_equal_chunk0'] = int(out['membership0']['PartType0/GroupNr_bound']['shape'][0]) == chunk0_rows
snap.close()

# ---- SOAP: metadata only
with h5py.File(ROOT + '/' + SOAP, 'r') as s:
    listing = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            desc = obj.attrs.get('Description', b'')
            listing[name] = {'dtype': str(obj.dtype), 'shape': list(obj.shape),
                             'description': plain(desc)[:240] if desc is not None else ''}
    s.visititems(visit)
    out['soap_dataset_count'] = len(listing)
    keep = ('InputHalos/', 'SOAP/', 'BoundSubhalo/', 'SO/200_crit/', 'SO/500_crit/', 'SO/5xR_500_crit/')
    out['soap_datasets'] = {k: v for k, v in listing.items() if k.startswith(keep) and k.count('/') <= 2}
    required = ['InputHalos/HaloCatalogueIndex', 'InputHalos/IsCentral', 'SOAP/HostHaloIndex', 'InputHalos/HaloCentre',
                'BoundSubhalo/TotalMass', 'BoundSubhalo/GasMass', 'BoundSubhalo/NumberOfGasParticles',
                'SO/200_crit/TotalMass', 'SO/200_crit/SORadius', 'SO/500_crit/TotalMass', 'SO/500_crit/SORadius',
                'SO/5xR_500_crit/SORadius'] + [f'SO/{a}/{q}' for a in ('200_crit', '500_crit', '5xR_500_crit')
                for q in ('ComptonY', 'ComptonYWithoutRecentAGNHeating', 'GasMass', 'NumberOfGasParticles')] + \
               [f'SO/500_crit/{q}' for q in ('XRayLuminosity', 'XRayLuminosityWithoutRecentAGNHeating',
                'XRayLuminosityInRestframe', 'XRayLuminosityInRestframeWithoutRecentAGNHeating')]
    out['soap_required'] = {k: (k in listing) for k in required}
    out['soap_concentration_like'] = sorted(k for k in listing if 'oncentration' in k and k.startswith(('SO/200_crit/', 'SO/500_crit/', 'BoundSubhalo/', 'InputHalos/')))
    patterns = ('Birth', 'Formation', 'LastMaxMass', 'LastMaxVmax', 'Snapshot', 'Age', 'Depth', 'Descendant', 'Progenitor', 'Merger', 'Infall')
    out['soap_formation_like'] = sorted(k for k in listing if any(p in k for p in patterns))
    out['soap_xray_like'] = sorted(k for k in listing if 'XRay' in k and k.startswith('SO/500_crit/'))
    out['soap_heating_filter'] = {k: plain(v) for k, v in s['Parameters/RecentlyHeatedGasFilter'].attrs.items()}
    out['soap_cosmology'] = {k: plain(v) for k, v in s['Cosmology'].attrs.items()}
    named = {}
    if 'SubgridScheme/NamedColumns' in s:
        for n, ds in s['SubgridScheme/NamedColumns'].items():
            named[n] = plain(ds[...])
    out['soap_named_columns'] = named
    out['soap_rows'] = int(s['InputHalos/HaloCatalogueIndex'].shape[0])
out['wall_seconds'] = time.time() - started
print('PROBE_JSON_BEGIN')
print(json.dumps(out, default=str))
print('PROBE_JSON_END')
'''


def program():
    """The remote program: in-memory frozen dependencies + REMOTE_BODY. Nothing is written remotely."""
    deps = {name: (HERE / 'dependencies' / f'{name}.py').read_text() for name in ('common', 'flamingo_io')}
    payload = base64.b64encode(json.dumps(deps).encode()).decode()
    import numpy as np
    halos = np.load(HERE / 'inputs' / 'radii234_selected_halos.npz')
    # A ~1e13 Msun radii234 halo: its cell has star-forming gas for the SFR sign check.
    index = int(np.argmin(np.abs(np.log10(halos['m200c_msun']) - 13.0)))
    centre = [float(v) for v in halos['centre_cmpc'][index]]
    head = (
        "import os, sys, json, types, base64\n"
        "os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'\n"
        "sys.dont_write_bytecode = True\n"
        f"_SOURCES = json.loads(base64.b64decode({payload!r}))\n"
        "for _name in ('common', 'flamingo_io'):\n"
        "    _module = types.ModuleType(_name); _module.__file__ = '<embedded ' + _name + '.py>'\n"
        "    sys.modules[_name] = _module\n"
        "    exec(compile(_SOURCES[_name], _name + '.py', 'exec'), _module.__dict__)\n"
        f"CENTRE_CMPC = {centre!r}\n"
        f"PROBE_HALO_ROW = {index}\n")
    return head + REMOTE_BODY


def parse(stdout):
    text = stdout.split('PROBE_JSON_BEGIN', 1)[1].split('PROBE_JSON_END', 1)[0]
    return json.loads(text)


def main():
    sys.path.insert(0, str(HERE))
    from remote_control import remote
    result = parse(remote(program(), python=PYTHON, timeout=600))
    (HERE / 'probe_g4.json').write_text(json.dumps(result, indent=1) + '\n')
    summary = {
        'auxiliary_probe_fields': result['auxiliary_probe_fields'],
        'named_columns': result['top_named_columns'].get('XrayLuminosities'),
        'small_cell_xray': result['small_cell']['fields'].get('XrayLuminosities'),
        'halo_cell_xray': result['halo_cell_slice']['xray_chunk_vs_top'],
        'sfr': {k: result['halo_cell_slice'][k] for k in ('sfr_positive', 'sfr_negative', 'sfr_zero', 'y_zero_where_sfr_positive')},
        'sfr_description': result['top_fields']['StarFormationRates']['attrs'].get('Description'),
        'wall_seconds': result['wall_seconds']}
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main()
