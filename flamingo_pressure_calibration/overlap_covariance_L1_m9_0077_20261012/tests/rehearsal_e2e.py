"""End-to-end local rehearsal of the maas pipeline on a synthetic mini-FLAMINGO (no remote access).

Builds, under a work directory (default: the session scratchpad passed as argv[1]):
  data/  a distributed SWIFT-like snapshot (virtual top-level file + 2 chunk files, 4^3 cells), SOAP-HBT
         halo_properties (centrals + satellites; SO apertures computed exactly from the particles) and HBT
         membership files in chunk particle order;
  study/ a study directory: study_config.json pointing at data/, a fake radii234 old sample (inputs/), a probe
         JSON, and radii234_reference_measurements.npz = the radii234 per-file/per-cell sums loop emulated with
         the frozen measure.accumulate (the bit-identity reference for G7).
Then runs, with OVERLAP_STUDY_DIR=study/: census.py, prepare_study.py, extract_study.py --gate-cell,
--gate-file 0, the full extract_study.py (2 workers) and verify_study.py; writes tests/rehearsal_results.json.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY_CODE = HERE.parent
sys.path.insert(0, str(STUDY_CODE / 'dependencies'))
sys.path.insert(0, str(STUDY_CODE))
import common as C  # noqa: E402
import flamingo_io as F  # noqa: E402
import pilot_physics as P  # noqa: E402
import measure as M  # noqa: E402
from catalogue_stage import sphere_cells, csr_map  # noqa: E402

PY = sys.executable
BOX, DIM, A = 100.0, 4, 1.0
RHO_C = 1.2775e11                      # Msun / Mpc^3
HEAT = {'a_limit': 0.9989555948537917, 'Tmin_in_K': 8801441.971770002, 'Tmax_in_K': 175611854.8365807}
RUN = 'L1000N1800/HYDRO_FIDUCIAL'
SNAP_REL = RUN + '/snapshots/flamingo_0077/flamingo_0077.hdf5'
SOAP_REL = RUN + '/SOAP-HBT/halo_properties_0077.hdf5'
MEMB_REL = RUN + '/SOAP-HBT/membership_0077'
UNITS = {'Coordinates': (C.MPC_CM, 1), 'SmoothingLengths': (C.MPC_CM, 1), 'Masses': (1e10 * C.MSUN_G, 0),
         'Densities': (1e10 * C.MSUN_G / C.MPC_CM ** 3, -3), 'ComptonYParameters': (C.MPC_CM ** 2, 0),
         'ElectronNumberDensities': (C.MPC_CM ** -3, -3), 'Temperatures': (1.0, 0), 'LastAGNFeedbackScaleFactors': (1.0, 0),
         'ParticleIDs': (1.0, 0), 'SmoothedElementMassFractions': (1.0, 0), 'Pressures': (1e-2, -5),
         'StarFormationRates': (1.989e33 / 3.156e7, 0), 'XrayLuminosities': (6.443997950038577e33, 0)}
DTYPES = {'Coordinates': ('f8', 3), 'SmoothingLengths': ('f4', 1), 'Masses': ('f4', 1), 'Densities': ('f4', 1),
          'ComptonYParameters': ('f4', 1), 'ElectronNumberDensities': ('f8', 1), 'Temperatures': ('f4', 1),
          'LastAGNFeedbackScaleFactors': ('f4', 1), 'ParticleIDs': ('u8', 1), 'SmoothedElementMassFractions': ('f4', 9),
          'Pressures': ('f4', 1), 'StarFormationRates': ('f4', 1), 'XrayLuminosities': ('f8', 3)}
DESCRIPTIONS = {'StarFormationRates': 'If positive, star formation rates of the particles. If negative, stores the last '
                'time/scale-factor at which the gas particle was star-forming. If zero, the particle was never star-forming.'}


def attrs_for(name):
    factor, exponent = UNITS[name]
    return {F.COMOVING_ATTR: [factor], F.PHYS_ATTR: [factor * A ** exponent], 'a-scale exponent': [float(exponent)],
            'Description': DESCRIPTIONS.get(name, name)}


def build_haloes(rng):
    n_cen = 260
    logm = rng.uniform(12.5, 15.0, n_cen)
    logm[:6] = [14.9, 14.95, 14.6, 14.55, 13.0, 13.1]
    m200 = 10 ** logm
    r200 = np.cbrt(3 * m200 / (4 * np.pi * 200 * RHO_C))
    pos = rng.uniform(0, BOX, (n_cen, 3))
    pos[1] = (pos[0] + [1.3 * r200[0], 0.4, 0]) % BOX          # massive close pair: neighbour-central overlap
    pos[3] = (pos[2] + [0, 2.2 * r200[2], 0]) % BOX            # annulus-only neighbour
    pos[4] = (pos[0] + [0, 0, 2.6 * r200[0]]) % BOX             # low-mass halo near a cluster (painted neighbour)
    pos[5] = (np.array([BOX - .3, 50., .2]))                    # sphere crosses the periodic boundary
    # satellites: 0-6 per central, bound mass a fraction of the host, inside 1.5 R200c
    sat_host, sat_pos, sat_mass = [], [], []
    for c in range(n_cen):
        for _ in range(rng.integers(0, 7) if logm[c] > 13 else rng.integers(0, 3)):
            ratio = 10 ** rng.uniform(-2.3, -0.4)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            sat_host.append(c)
            sat_pos.append((pos[c] + direction * r200[c] * rng.uniform(.2, 1.5)) % BOX)
            sat_mass.append(ratio * m200[c])
    n_sat = len(sat_host)
    return {'n_cen': n_cen, 'm200': m200, 'r200': r200, 'pos': pos, 'sat_host': np.array(sat_host), 'sat_pos': np.array(sat_pos),
            'sat_mass': np.array(sat_mass), 'n_sat': n_sat}


def build_gas(rng, halo):
    """Background + per-halo NFW-ish gas to 4.5 R200c + satellite clumps, with membership."""
    pos, member = [rng.uniform(0, BOX, (120000, 3))], [np.full(120000, -1, np.int64)]
    hid_central = 10 * np.arange(halo['n_cen']) + 7                # HaloCatalogueIndex values (not SOAP rows)
    hid_sat = 10 * halo['n_cen'] + 13 + 3 * np.arange(halo['n_sat'])
    for c in range(halo['n_cen']):
        n = int(40 + 4e-11 * halo['m200'][c])
        u = rng.uniform(size=n)
        r = halo['r200'][c] * 4.5 * u ** 1.6
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        pos.append((halo['pos'][c] + d * r[:, None]) % BOX)
        bound = (r < 1.1 * halo['r200'][c]) & (rng.uniform(size=n) < .85)
        member.append(np.where(bound, hid_central[c], -1))
    for s in range(halo['n_sat']):
        n = int(20 + 6e-11 * halo['sat_mass'][s])
        rs = np.cbrt(3 * halo['sat_mass'][s] / (4 * np.pi * 200 * RHO_C))
        pos.append((halo['sat_pos'][s] + rng.normal(scale=rs / 2, size=(n, 3))) % BOX)
        member.append(np.where(rng.uniform(size=n) < .8, hid_sat[s], -1))
    pos, member = np.concatenate(pos), np.concatenate(member)
    n = len(pos)
    sfr = np.where(rng.uniform(size=n) < .04, rng.uniform(.1, 5, n), np.where(rng.uniform(size=n) < .1, -rng.uniform(.1, 1, n), 0))
    sf = sfr > 0
    T = (10 ** rng.uniform(3.8, 8.4, n)).astype(np.float32)
    gas = {'Coordinates': pos, 'SmoothingLengths': rng.uniform(.005, 1.0, n).astype(np.float32),
           'Masses': rng.uniform(.08, .12, n).astype(np.float32), 'Densities': (10 ** rng.uniform(1, 5, n)).astype(np.float32),
           'ComptonYParameters': np.where(sf, 0, 10 ** rng.uniform(-14, -10, n)).astype(np.float32),
           'ElectronNumberDensities': np.where(sf, 0, 10 ** rng.uniform(60, 66, n)), 'Temperatures': T,
           'LastAGNFeedbackScaleFactors': np.where(rng.uniform(size=n) < .3, rng.uniform(.995, 1, n), -1).astype(np.float32),
           'ParticleIDs': rng.choice(2 ** 50, n, replace=False).astype(np.uint64),
           'SmoothedElementMassFractions': np.column_stack([rng.uniform(.70, .76, n), rng.uniform(.24, .28, n)] +
                                                           [np.full(n, 1e-3)] * 7).astype(np.float32),
           'Pressures': (10 ** rng.uniform(-6, -1, n)).astype(np.float32), 'StarFormationRates': sfr.astype(np.float32),
           'XrayLuminosities': np.where(sf[:, None], 0, 10 ** rng.uniform(-6, 2, (n, 3)))}
    return gas, member, hid_central, hid_sat


def write_snapshot(root, gas, member):
    cell_size = BOX / DIM
    ijk = np.floor(gas['Coordinates'] / cell_size).astype(int) % DIM
    cell = (ijk[:, 0] * DIM + ijk[:, 1]) * DIM + ijk[:, 2]
    n_cells = DIM ** 3
    cell_file = (np.arange(n_cells) >= n_cells // 2).astype(int)
    order = np.lexsort((np.arange(len(cell)), cell))              # by cell, stable
    counts = np.bincount(cell, minlength=n_cells)
    starts = np.r_[0, np.cumsum(counts)[:-1]]                     # global offsets (file 0 cells precede file 1 cells)
    file_counts = np.bincount(cell_file, weights=counts, minlength=2).astype(int)
    file_starts = np.r_[0, np.cumsum(file_counts)[:-1]]
    per_file_offset = starts - file_starts[cell_file]
    centres = (np.stack(np.unravel_index(np.arange(n_cells), (DIM,) * 3), axis=1) + .5) * cell_size
    snapdir = root / Path(SNAP_REL).parent
    snapdir.mkdir(parents=True, exist_ok=True)
    cosmo = {'Redshift': [0.], 'Scale-factor': [A], 'h': [.681], 'Omega_b': [.0486], 'Omega_m': [.304], 'Omega_cdm': [.2553],
             'Omega_nu_0': [.00139], 'Omega_lambda': [.694], 'H [internal units]': [68.1], 'Critical density [internal units]': [RHO_C / 1e10],
             'Cosmological run': [1]}
    header = {'BoxSize': [BOX] * 3, 'NumFilesPerSnapshot': [1], 'NumPart_Total': [len(cell), 0, 0, 0, 0, 0, 0],
              'NumPart_Total_HighWord': [0] * 7, 'Scale-factor': [A], 'Redshift': [0.], 'Code': 'SWIFT'}
    with h5py.File(root / SNAP_REL, 'w') as f:
        for k, v in header.items():
            f.create_group('Header') if 'Header' not in f else None
            f['Header'].attrs[k] = v
        f.create_group('Cosmology').attrs.update(cosmo)
        f.create_group('Units').attrs.update({'Unit length in cgs (U_L)': [C.MPC_CM], 'Unit mass in cgs (U_M)': [1e10 * C.MSUN_G],
                                              'Unit time in cgs (U_t)': [3.0857e19]})
        f.create_group('Parameters').attrs.update({'Gravity:comoving_baryon_softening': b'0.0223',
                                                   'Gravity:max_physical_baryon_softening': b'0.0057'})
        meta = f.create_group('Cells/Meta-data')
        meta.attrs['size'] = [cell_size] * 3
        meta.attrs['dimension'] = [DIM] * 3
        f['Cells/Centres'] = centres
        f['Cells/Counts/PartType0'] = counts
        f['Cells/Files/PartType0'] = np.zeros(n_cells, int)
        f['Cells/OffsetsInFile/PartType0'] = starts
        f['SubgridScheme/NamedColumns/XrayLuminosities'] = np.array([b'erosita_low', b'erosita_high', b'ROSAT'])
    membdir = root / MEMB_REL
    membdir.mkdir(parents=True, exist_ok=True)
    sorted_index = order
    for i in range(2):
        rows = sorted_index[file_starts[i]:file_starts[i] + file_counts[i]]
        with h5py.File(snapdir / f'flamingo_0077.{i}.hdf5', 'w') as f:
            for k, v in header.items():
                f.create_group('Header') if 'Header' not in f else None
                f['Header'].attrs[k] = v
            f['Header'].attrs['NumFilesPerSnapshot'] = [2]
            f['Header'].attrs['NumPart_ThisFile'] = [len(rows), 0, 0, 0, 0, 0, 0]
            f['Cells/Counts/PartType0'] = counts
            f['Cells/Files/PartType0'] = cell_file
            f['Cells/OffsetsInFile/PartType0'] = per_file_offset
            g = f.create_group('PartType0')
            for name, (dtype, width) in DTYPES.items():
                values = gas[name][rows].astype(dtype)
                ds = g.create_dataset(name, data=values, chunks=(4096,) + values.shape[1:], compression='gzip', compression_opts=1)
                ds.attrs.update(attrs_for(name))
        with h5py.File(membdir / f'membership_0077.{i}.hdf5', 'w') as f:
            f.create_dataset('PartType0/GroupNr_bound', data=member[rows], chunks=(4096,))
    return order


def soap_values(halo, gas, member, hid_central, hid_sat):
    """SOAP rows: centrals then satellites (rows sorted by nothing in particular), SO sums exact from particles."""
    n_cen, n_sat = halo['n_cen'], halo['n_sat']
    n = n_cen + n_sat
    centre = np.concatenate([halo['pos'], halo['sat_pos'].reshape(-1, 3)])
    pos = gas['Coordinates']
    m200 = np.r_[halo['m200'], np.zeros(n_sat)]
    # SO sums use the radii exactly as stored (float32), as SOAP's own apertures do; 6 R500c = 4.2 R200c > 4 R200c
    r200 = np.r_[halo['r200'], np.zeros(n_sat)].astype(np.float32).astype(np.float64)
    r500 = np.r_[.7 * halo['r200'], np.zeros(n_sat)].astype(np.float32).astype(np.float64)
    y = gas['ComptonYParameters'].astype(np.float64)
    lx = gas['XrayLuminosities']
    T = gas['Temperatures'].astype(np.float64)
    last = gas['LastAGNFeedbackScaleFactors'].astype(np.float64)
    agn = (last >= HEAT['a_limit']) & (T >= HEAT['Tmin_in_K']) & (T <= HEAT['Tmax_in_K'])
    so = {k: np.zeros(n) for k in ('y200', 'yn200', 'y500', 'yn500', 'y5', 'yn5', 'g200', 'g500', 'g5', 'n200', 'n500', 'n5')}
    so['lx500'], so['lxn500'], so['lx5'], so['lxn5'] = (np.zeros((n, 3)) for _ in range(4))
    for i in range(n_cen):
        d = pos - centre[i]
        d -= BOX * np.rint(d / BOX)
        r = np.sqrt(np.einsum('ij,ij->i', d, d))
        for tag, radius in (('200', r200[i]), ('500', r500[i]), ('5', 5 * r500[i])):
            inside = r <= radius
            so['y' + tag][i] = y[inside].sum()
            so['yn' + tag][i] = y[inside & ~agn].sum()
            so['g' + tag][i] = gas['Masses'][inside].astype(np.float64).sum()
            so['n' + tag][i] = inside.sum()
            if tag in ('500', '5'):
                so['lx' + tag][i] = lx[inside].sum(axis=0)
                so['lxn' + tag][i] = lx[inside & ~agn].sum(axis=0)
    hid = np.r_[hid_central, hid_sat]
    bound_count = np.array([np.sum(member == h) for h in hid])
    bound_gas = np.array([gas['Masses'][member == h].astype(np.float64).sum() for h in hid])
    bound_mass = np.r_[1.1 * halo['m200'], halo['sat_mass']]
    host = np.r_[np.full(n_cen, -1), halo['sat_host']]
    return dict(hid=hid, centre=centre, m200=m200, r200=r200, r500=r500, bound_count=bound_count, bound_gas=bound_gas,
                bound_mass=bound_mass, host=host, central=np.r_[np.ones(n_cen, int), np.zeros(n_sat, int)], **so)


def write_soap(root, s):
    n = len(s['hid'])
    mass_unit, length = 1e10 * C.MSUN_G, C.MPC_CM

    def put(f, name, values, factor=None):
        ds = f.create_dataset(name, data=values)
        if factor is not None:
            ds.attrs[F.PHYS_ATTR] = [factor]
            ds.attrs[F.COMOVING_ATTR] = [factor]
    path = root / SOAP_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        put(f, 'InputHalos/HaloCatalogueIndex', s['hid'].astype(np.int64))
        put(f, 'InputHalos/IsCentral', s['central'].astype(np.int64))
        put(f, 'InputHalos/HaloCentre', s['centre'], length)
        put(f, 'SOAP/HostHaloIndex', s['host'].astype(np.int64))
        put(f, 'BoundSubhalo/TotalMass', (s['bound_mass'] / 1e10).astype(np.float32), mass_unit)
        put(f, 'BoundSubhalo/GasMass', (s['bound_gas'] * 1.0).astype(np.float32), mass_unit)
        put(f, 'BoundSubhalo/NumberOfGasParticles', s['bound_count'].astype(np.uint32))
        put(f, 'BoundSubhalo/CentreOfMass', s['centre'] + 0.01, length)
        put(f, 'SO/200_crit/TotalMass', (s['m200'] / 1e10).astype(np.float32), mass_unit)
        put(f, 'SO/200_crit/SORadius', s['r200'].astype(np.float32), length)
        put(f, 'SO/500_crit/TotalMass', (.7 * s['m200'] / 1e10).astype(np.float32), mass_unit)
        put(f, 'SO/500_crit/SORadius', s['r500'].astype(np.float32), length)
        put(f, 'SO/5xR_500_crit/SORadius', (5 * s['r500']).astype(np.float32), length)   # (5 R500c sums use 5 x float32 R500c)
        for tag, ap in (('200', '200_crit'), ('500', '500_crit'), ('5', '5xR_500_crit')):
            put(f, f'SO/{ap}/ComptonY', s['y' + tag], length ** 2)
            put(f, f'SO/{ap}/ComptonYWithoutRecentAGNHeating', s['yn' + tag], length ** 2)
            put(f, f'SO/{ap}/GasMass', (s['g' + tag]).astype(np.float32), mass_unit)
            put(f, f'SO/{ap}/NumberOfGasParticles', s['n' + tag].astype(np.uint32))
        for tag, ap in (('500', '500_crit'), ('5', '5xR_500_crit')):
            put(f, f'SO/{ap}/XRayLuminosity', s['lx' + tag] * UNITS['XrayLuminosities'][0], 1.0)
            put(f, f'SO/{ap}/XRayLuminosityWithoutRecentAGNHeating', s['lxn' + tag] * UNITS['XrayLuminosities'][0], 1.0)
        f['Cells/Counts/Subhalos'] = np.r_[n, np.zeros(DIM ** 3 - 1, int)]
        f['Cells/OffsetsInFile/Subhalos'] = np.zeros(DIM ** 3, int)
        f['Cells/Centres'] = np.zeros((DIM ** 3, 3))
        f.create_group('Cosmology').attrs.update({'Redshift': [0.], 'Scale-factor': [A], 'h': [.681], 'Omega_b': [.0486],
                                                  'Omega_m': [.304], 'Omega_nu_0': [.00139]})
        f.create_group('Parameters/RecentlyHeatedGasFilter').attrs.update(
            {'a_limit': HEAT['a_limit'], 'Tmin_in_K': [HEAT['Tmin_in_K']], 'Tmax_in_K': [HEAT['Tmax_in_K']], 'delta_time_in_Myr': 15.0})


def emulate_radii234(data_root, study, old):
    """The radii234 read_file/merge sums loop (frozen measure.accumulate) on the fixture: the G7 reference."""
    snap = F.Snapshot(F.Store(str(data_root)), SNAP_REL, 77, read_from='chunks', chunk_cache_bytes=1024 ** 2)
    fields = tuple(F.GAS_FIELDS) + ('ParticleIDs', 'SmoothedElementMassFractions', 'Pressures')
    cells, ptr, index = csr_map(old['pair_halo'], old['pair_cell'])
    read = snap.read_order(cells)
    files = sorted(int(x) for x in np.unique(snap.cell_file[read]))
    n = len(old['halo_index'])
    sums = np.zeros((n, 4, len(P.SUM_NAMES), M.NB))
    swift = set(json.loads((study / 'inputs' / 'radii234_metadata_plan.json').read_text())['swift_check_indices'])
    ids = {}
    heating = {'a_limit': HEAT['a_limit'], 'tmin_k': HEAT['Tmin_in_K'], 'tmax_k': HEAT['Tmax_in_K']}
    for file_id in files:
        parts = {}
        with h5py.File(data_root / MEMB_REL / f'membership_0077.{file_id}.hdf5', 'r') as mf:
            membership = mf['PartType0/GroupNr_bound']
            for cell in [int(c) for c in read if snap.cell_file[c] == file_id]:
                raw = snap.read_cell_raw(cell, fields)
                data = {name: raw[name].astype(np.float64) * snap.factor(name) for name in F.GAS_FIELDS}
                data.update(ParticleIDs=raw['ParticleIDs'], Hydrogen=raw['SmoothedElementMassFractions'][:, 0].astype(float),
                            Helium=raw['SmoothedElementMassFractions'][:, 1].astype(float),
                            Pressures=raw['Pressures'].astype(float) * C.physical_factor(snap.dataset('Pressures')) / 10)
                lo, size = int(snap.cell_offset[cell]), int(snap.counts[cell])
                owner = np.asarray(membership[lo:lo + size], dtype=np.int64)
                k = int(np.searchsorted(cells, cell))
                for h in index[ptr[k]:ptr[k + 1]]:
                    h = int(h)
                    radius = old['r_200_crit_mpc'][h]
                    delta = data['Coordinates'] - old['centre_cmpc'][h]
                    delta -= snap.box_size_cmpc * np.rint(delta / snap.box_size_cmpc)
                    inside = np.einsum('ij,ij->i', delta, delta) <= (4 * radius / A) ** 2
                    selected = {name: v[inside] for name, v in data.items()}
                    sat = old['satellite_ids'][old['satellite_ptr'][h]:old['satellite_ptr'][h + 1]]
                    one = {'centre_cmpc': old['centre_cmpc'][h], 'radius': radius, 'mass': old['m200c_msun'][h],
                           'id': old['halo_index'][h], 'satellites': sat, 'a': A, 'box': snap.box_size_cmpc}
                    item = M.accumulate(selected, owner[inside], one, .158, heating, np.eye(3)[None].repeat(2, axis=0))
                    if h not in parts:
                        parts[h] = np.zeros(item['sums'].shape)
                    parts[h] += item['sums']
                    if h in swift:
                        ids.setdefault(h, []).append(item['ids'])
        rows = np.array(sorted(parts), dtype=np.int64)
        sums[rows] += np.array([parts[int(h)] for h in rows])
    snap.close()
    np.savez_compressed(study / 'radii234_reference_measurements.npz', sums=sums)
    hashes = {str(h): P.hash_ids(np.concatenate(v)) for h, v in ids.items()}
    (study / 'inputs' / 'radii234_particle_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return sums


def make_old_sample(data_root, study, rng, n_old=70):
    soap = F.Soap(F.Store(str(data_root)), SOAP_REL)
    cols = soap.columns(names=('halo_index', 'centre_mpc', 'm200c_msun', 'r_200_crit_mpc', 'is_central'))
    central = cols['is_central'].astype(bool) & (cols['m200c_msun'] >= 10 ** 12.5)
    pool = np.flatnonzero(central)
    rows = np.sort(np.r_[np.arange(6), rng.choice(pool[6:], n_old - 6, replace=False)])
    host = soap.read('SOAP/HostHaloIndex')
    hid = cols['halo_index']
    old = {k: cols[k][rows] for k in ('halo_index', 'centre_mpc', 'm200c_msun', 'r_200_crit_mpc')}
    old['soap_row'] = rows
    old['centre_cmpc'] = old['centre_mpc'] / A
    edges = np.round(np.arange(12.5, 15.5001, .2), 10)
    old['mass_bin'] = np.searchsorted(edges, np.log10(old['m200c_msun']), side='right') - 1
    sat = np.flatnonzero(np.isin(host, rows))
    sat_halo = np.searchsorted(rows, host[sat])
    order = np.lexsort((hid[sat], sat_halo))
    old['satellite_ptr'] = np.r_[0, np.cumsum(np.bincount(sat_halo, minlength=len(rows)))]
    old['satellite_ids'] = hid[sat][order]
    snap = F.Snapshot(F.Store(str(data_root)), SNAP_REL, 77, read_from='chunks', chunk_cache_bytes=1024 ** 2)
    ph, pc = sphere_cells(old['centre_cmpc'], 4 * old['r_200_crit_mpc'] / A, snap.cell_size_cmpc, snap.dimension, snap.lookup,
                          snap.box_size_cmpc, .1 * snap.cell_size_cmpc)
    old['pair_halo'], old['pair_cell'] = ph, pc
    snap.close()
    soap.close()
    (study / 'inputs').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(study / 'inputs' / 'radii234_selected_halos.npz', **old)
    swift = sorted(rng.choice(n_old, 5, replace=False).tolist())
    (study / 'inputs' / 'radii234_metadata_plan.json').write_text(json.dumps(
        {'swift_check_indices': swift, 'membership_order_contract': 'synthetic rehearsal'}, indent=1))
    return old


def run(command, study, log):
    env = os.environ.copy()
    env['OVERLAP_STUDY_DIR'] = str(study)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OVERLAP_BATCH'] = os.environ.get('OVERLAP_BATCH', '5000')     # small batches: exercises the batched merge
    started = time.perf_counter()
    result = subprocess.run([PY, '-B'] + command, cwd=STUDY_CODE, env=env, capture_output=True, text=True, timeout=3600)
    log.append({'command': command, 'returncode': result.returncode, 'seconds': round(time.perf_counter() - started, 2),
                'stdout_tail': result.stdout[-2500:], 'stderr_tail': result.stderr[-2500:]})
    print(f"{'OK ' if result.returncode == 0 else 'ERR'} {' '.join(command)} ({log[-1]['seconds']} s)", flush=True)
    if result.returncode:
        print(result.stdout[-3000:], result.stderr[-6000:])
    return result.returncode == 0


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'rehearsal_work'
    if work.exists():
        shutil.rmtree(work)
    data_root, study = work / 'data', work / 'study'
    study.mkdir(parents=True)
    rng = np.random.default_rng(20261012)
    halo = build_haloes(rng)
    gas, member, hid_central, hid_sat = build_gas(rng, halo)
    write_snapshot(data_root, gas, member)
    write_soap(data_root, soap_values(halo, gas, member, hid_central, hid_sat))
    old = make_old_sample(data_root, study, rng)
    cfg = json.loads((STUDY_CODE / 'study_config.json').read_text())
    cfg.update(data_root=str(data_root), workers=2, RSS_stop_GiB=16, wall_limit_minutes=30, chunk_cache_MiB=8, gate_file=0)
    cfg['old_sample']['rows'] = [0, len(old['halo_index'])]
    cfg['enlargement'].update(apply=True, target_per_class=6)
    cfg['census'].update(kdtree_workers=1)
    cfg['reopen_every_cells'] = int(os.environ.get('REHEARSAL_REOPEN', cfg.get('reopen_every_cells', 16)))
    (study / 'study_config.json').write_text(json.dumps(cfg, indent=1))
    probe = json.loads((STUDY_CODE / 'probe_g4.json').read_text())
    (study / 'probe_g4.json').write_text(json.dumps(probe))
    reference = emulate_radii234(data_root, study, old)
    log = []
    ok = True
    for command in (['census.py'], ['prepare_study.py'], ['extract_study.py', '--gate-cell'], ['extract_study.py', '--gate-file', '0'],
                    ['extract_study.py'], ['verify_study.py', '--radii234-measurements', str(study / 'radii234_reference_measurements.npz')]):
        ok = run(command, study, log) and ok
        if not ok:
            break
    verification = json.loads((study / 'verification.json').read_text()) if (study / 'verification.json').exists() else {}
    gate = json.loads((study / 'one_cell_gate.json').read_text()) if (study / 'one_cell_gate.json').exists() else {}
    census = json.loads((study / 'census_summary.json').read_text()) if (study / 'census_summary.json').exists() else {}
    summary = {'all_steps_ok': ok, 'gas_particles': int(len(member)), 'centrals': halo['n_cen'], 'satellites': halo['n_sat'],
               'old_sample': int(len(old['halo_index'])), 'legacy_reference_nonzero_haloes': int(np.sum(reference[:, :, 0].sum(axis=(1, 2)) > 0)),
               'census': {k: census.get(k) for k in ('old_sample', 'proposed_additions', 'overlapper_pairs', 'partition_pairs', 'g3')},
               'one_cell_gate': {k: gate.get(k) for k in ('passed', 'empty_then_populated')} | {'haloes': len(gate.get('gate', []))},
               'verification': verification, 'steps': log}
    out = HERE / 'rehearsal_results.json'
    out.write_text(json.dumps(summary, indent=1, default=str) + '\n')
    statuses = {k: v.get('status') for k, v in verification.items() if isinstance(v, dict)}
    print(json.dumps({'all_steps_ok': ok, 'gates': statuses}, indent=1))


if __name__ == '__main__':
    main()
