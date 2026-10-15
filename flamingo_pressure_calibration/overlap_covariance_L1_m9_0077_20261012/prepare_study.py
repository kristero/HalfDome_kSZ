"""Metadata-only preparation of the particle pass (plan Step 1.1 -> 1.2); run on maas after census.py.

Reads: census outputs (census_pairs.npz, census_population.npz, census_summary.json), SOAP columns of the
final sample, snapshot cell metadata, dataset attributes, membership-file shapes. No particle values.
Writes (refuses to overwrite): selected_halos.npz (per-halo arrays, rows 0-4374 = radii234 in its order),
overlap_tables.npz (CSR tables for the workers), metadata_plan.json (plan, schema, budget, G4 re-check).

Module constants ROOT, SNAPSHOT, MEMBERSHIP, FIELDS are imported by extract_study.py and by the frozen
swift_reader.py; FIELDS is the radii234 tuple (EXTRA_FIELDS are read in addition).
The enlargement is applied only if study_config.json enlargement.apply is true (decided at Checkpoint 1).
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
# Data/config directory: this study root unless OVERLAP_STUDY_DIR is set (local end-to-end rehearsal only).
STUDY = Path(os.environ.get('OVERLAP_STUDY_DIR', str(HERE))).resolve()
sys.path.insert(0, str(HERE / 'dependencies'))
import common as C  # noqa: E402
import flamingo_io as F  # noqa: E402

CFG = json.loads((STUDY / 'study_config.json').read_text())
ROOT = CFG['data_root']
SNAPSHOT = CFG['snapshot_rel']
SOAP = CFG['soap_rel']
MEMBERSHIP = Path(ROOT) / CFG['membership_rel']
FIELDS = tuple(F.GAS_FIELDS) + ('ParticleIDs', 'SmoothedElementMassFractions', 'Pressures')
EXTRA_FIELDS = ('StarFormationRates', 'XrayLuminosities')
N_OLD = CFG['old_sample']['rows'][1]

# SOAP columns stored per halo: name -> (dataset, target CGS unit or None for raw values)
SOAP_COLUMNS = {
    'm500c_msun': ('SO/500_crit/TotalMass', C.MSUN_G), 'r_500_crit_mpc': ('SO/500_crit/SORadius', C.MPC_CM),
    'r_5xr500_mpc': ('SO/5xR_500_crit/SORadius', C.MPC_CM),
    'bound_mass_msun': ('BoundSubhalo/TotalMass', C.MSUN_G), 'bound_gas_mass_msun': ('BoundSubhalo/GasMass', C.MSUN_G),
    'bound_gas_count': ('BoundSubhalo/NumberOfGasParticles', None),
}
for _tag, _ap in (('200', '200_crit'), ('500', '500_crit'), ('5x500', '5xR_500_crit')):
    SOAP_COLUMNS[f'y_{_tag}_mpc2'] = (f'SO/{_ap}/ComptonY', C.MPC_CM ** 2)
    SOAP_COLUMNS[f'y_noagn_{_tag}_mpc2'] = (f'SO/{_ap}/ComptonYWithoutRecentAGNHeating', C.MPC_CM ** 2)
    SOAP_COLUMNS[f'gasmass_{_tag}_msun'] = (f'SO/{_ap}/GasMass', C.MSUN_G)
    SOAP_COLUMNS[f'ngas_{_tag}'] = (f'SO/{_ap}/NumberOfGasParticles', None)
for _tag, _ap in (('500', '500_crit'), ('5x500', '5xR_500_crit')):
    SOAP_COLUMNS[f'xray_{_tag}_erg_s'] = (f'SO/{_ap}/XRayLuminosity', 1.0)
    SOAP_COLUMNS[f'xray_noagn_{_tag}_erg_s'] = (f'SO/{_ap}/XRayLuminosityWithoutRecentAGNHeating', 1.0)
OPTIONAL_SOAP = {
    'concentration_200c': ('SO/200_crit/Concentration', 1.0),
    'concentration_unsoftened_200c': ('SO/200_crit/ConcentrationUnsoftened', 1.0),
    'dm_concentration_200c': ('SO/200_crit/DarkMatterConcentration', 1.0),
    'snapshot_of_birth': ('InputHalos/HBTplus/SnapshotIndexOfBirth', None),
    'snapshot_of_last_max_mass': ('InputHalos/HBTplus/SnapshotIndexOfLastMaxMass', None),
    'last_max_mass_msun': ('InputHalos/HBTplus/LastMaxMass', C.MSUN_G),
    'mass_weighted_stellar_age': ('BoundSubhalo/MassWeightedMeanStellarAge', None),
    'mass_fraction_satellites_200c': ('SO/200_crit/MassFractionSatellites', None),
    'mass_fraction_external_200c': ('SO/200_crit/MassFractionExternal', None),
    'xray_restframe_500_erg_s': ('SO/500_crit/XRayLuminosityInRestframe', 1.0),
    'xray_restframe_noagn_500_erg_s': ('SO/500_crit/XRayLuminosityInRestframeWithoutRecentAGNHeating', 1.0),
}


def plain(value):
    if isinstance(value, bytes):
        return value.decode(errors='replace')
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    return value


def read_rows(soap, path, rows, target):
    """SOAP column at arbitrary (unsorted) rows, in the given order."""
    order = np.argsort(rows)
    values = soap.read(path, rows[order], target)
    out = np.empty_like(values)
    out[order] = values
    return out


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def align_rotation(u, v):
    """Proper rotation taking unit vector u onto unit vector v."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(u @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(u, [1., 0, 0] if abs(u[0]) < .9 else [0, 1., 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.cross(u, v)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def axis_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def build_tables(halos, census, take, groups):
    """Per-halo CSR tables: overlappers, owner codes, excision spheres (4 variants), partition list."""
    import measure_overlap as MO
    import geometry as GEO
    import grids as G
    n = len(take)
    ptr = census['overlap_ptr']
    sat_ptr, sat_hid = census['satellite_ptr'], census['satellite_hid']
    pn_ptr = census['partition_ptr']
    group_index = {int(r): k for k, r in enumerate(groups['row'])}
    out = {k: [] for k in ('ov_index', 'owner_hid', 'owner_code', 'owner_group', 'owner_own',
                           'exc_offset', 'exc_radius', 'pn_index', 'satellite_hid')}
    lengths = {k: np.zeros(n, np.int64) for k in ('ov', 'owner', 'pn', 'satellite')}
    exc_lengths = np.zeros((n, 4), np.int64)
    spheres = []
    for h, c in enumerate(take):
        k0, k1 = ptr[c], ptr[c + 1]
        index = np.arange(k0, k1)
        out['ov_index'].append(index)
        lengths['ov'][h] = len(index)
        own = census['ov_own_satellite'][index]
        code = MO.owner_code(census['ov_mu_class'][index], (~census['ov_reach1'][index]).astype(np.int64),
                             (~own).astype(np.int64))
        entries = {int(halos['halo_index'][h]): (0, -1, True)}
        sats = sat_hid[sat_ptr[c]:sat_ptr[c + 1]]
        out['satellite_hid'].append(sats)
        lengths['satellite'][h] = len(sats)
        for s in sats:
            entries[int(s)] = (MO.OTHER_OWN, -1, True)
        for k_local, (hid, is_own) in enumerate(zip(census['ov_hid'][index], own)):
            entries[int(hid)] = (int(code[k_local]), k_local, bool(is_own))
        for k_local in np.flatnonzero(census['ov_is_central'][index] & ~own):
            g = group_index[int(census['ov_soap_row'][index][k_local])]
            for member in groups['member_hid'][groups['ptr'][g]:groups['ptr'][g + 1]]:
                entries[int(member)] = (int(code[k_local]), int(k_local), False)   # the group counts as its central
        keys = np.array(sorted(entries), dtype=np.int64)
        values = [entries[int(k)] for k in keys]
        out['owner_hid'].append(keys)
        out['owner_code'].append(np.array([v[0] for v in values], np.int8))
        out['owner_group'].append(np.array([v[1] for v in values], np.int32))
        out['owner_own'].append(np.array([v[2] for v in values], bool))
        lengths['owner'][h] = len(keys)
        exc = census['ov_mu'][index] >= CFG['excision']['mu_min']
        all_spheres = (census['ov_offset_mpc'][index][exc], census['ov_radius_mpc'][index][exc], census['ov_mu'][index][exc])
        nbr = exc & ~own
        spheres.append({'all': all_spheres, 'neighbours_only': (census['ov_offset_mpc'][index][nbr], census['ov_radius_mpc'][index][nbr])})
        pidx = np.arange(pn_ptr[c], pn_ptr[c + 1])
        out['pn_index'].append(pidx)
        lengths['pn'][h] = len(pidx)
    # placebo spheres for CLEAN haloes: a random CONT partner of the same mass bin, scaled and rotated
    rng = np.random.default_rng(CFG['excision']['placebo_seed'])
    cont = np.array([len(s['all'][1]) > 0 for s in spheres])
    partner = np.full(n, -1, np.int64)
    rot_iso, rot_nbr = np.tile(np.eye(3), (n, 1, 1)), np.tile(np.eye(3), (n, 1, 1))
    nbr_fallback = np.zeros(n, bool)
    partner_from_adjacent_bin = np.zeros(n, bool)
    nbr_dir = census['nearest_massive_offset_mpc'][take]
    for h in np.flatnonzero(~cont):
        b = halos['mass_bin'][h]
        pool = np.flatnonzero(cont & (halos['mass_bin'] == b))
        width = 1
        while not len(pool):
            pool = np.flatnonzero(cont & (np.abs(halos['mass_bin'] - b) <= width))
            partner_from_adjacent_bin[h] = True
            width += 1
        p = int(rng.choice(pool))
        partner[h] = p
        rot_iso[h] = random_rotation(rng)
        offsets, radii, mus = spheres[p]['all']
        dominant = offsets[int(np.argmax(mus))]
        roll = rng.uniform(0, 2 * np.pi)
        if np.all(np.isfinite(nbr_dir[h])) and np.linalg.norm(nbr_dir[h]) > 0 and np.linalg.norm(dominant) > 0:
            target = nbr_dir[h] / np.linalg.norm(nbr_dir[h])
            rot_nbr[h] = axis_rotation(target, roll) @ align_rotation(dominant, target)
        else:
            rot_nbr[h] = rot_iso[h]
            nbr_fallback[h] = True
    scale = np.where(partner >= 0, halos['r_200_crit_mpc'] / halos['r_200_crit_mpc'][np.maximum(partner, 0)], 0.)
    kappas = np.asarray(CFG['kappa'], float)
    volume_fraction = np.zeros((n, 4, len(kappas), G.N200), np.float32)
    for h in range(n):
        variants = [spheres[h]['all'][:2], spheres[h]['neighbours_only']]
        if partner[h] >= 0:
            offsets, radii, _ = spheres[partner[h]]['all']
            for rot in (rot_iso[h], rot_nbr[h]):
                variants.append(((offsets @ rot.T) * scale[h], radii * scale[h]))
        else:
            variants += [(np.empty((0, 3)), np.empty(0))] * 2
        edges = G.X200 * halos['r_200_crit_mpc'][h]
        for v, (offsets, radii) in enumerate(variants):
            out['exc_offset'].append(np.asarray(offsets, float).reshape(-1, 3))
            out['exc_radius'].append(np.asarray(radii, float))
            exc_lengths[h, v] = len(radii)
            for k, kappa in enumerate(kappas):
                if len(radii):
                    volume_fraction[h, v, k] = GEO.union_shell_fraction(edges, offsets, kappa * np.asarray(radii))
    tables = {
        'ov_ptr': np.r_[0, np.cumsum(lengths['ov'])].astype(np.int64),
        'owner_ptr': np.r_[0, np.cumsum(lengths['owner'])].astype(np.int64),
        'pn_ptr': np.r_[0, np.cumsum(lengths['pn'])].astype(np.int64),
        'satellite_ptr': np.r_[0, np.cumsum(lengths['satellite'])].astype(np.int64),
        'exc_ptr': np.r_[0, np.cumsum(exc_lengths.ravel())].astype(np.int64),      # index h * 4 + variant
        'owner_hid': np.concatenate(out['owner_hid']), 'owner_code': np.concatenate(out['owner_code']),
        'owner_group': np.concatenate(out['owner_group']), 'owner_own': np.concatenate(out['owner_own']),
        'satellite_hid': np.concatenate(out['satellite_hid']) if n else np.empty(0, np.int64),
        'exc_offset_mpc': np.concatenate(out['exc_offset']) if out['exc_offset'] else np.empty((0, 3)),
        'exc_radius_mpc': np.concatenate(out['exc_radius']) if out['exc_radius'] else np.empty(0),
        'placebo_partner': partner, 'placebo_rotation_iso': rot_iso, 'placebo_rotation_nbr': rot_nbr,
        'placebo_scale': scale, 'placebo_nbr_fallback_isotropic': nbr_fallback,
        'placebo_partner_from_adjacent_bin': partner_from_adjacent_bin,
        'excision_volume_fraction': volume_fraction, 'kappa': kappas}
    ov = np.concatenate(out['ov_index']) if out['ov_index'] else np.empty(0, np.int64)
    for key in ('hid', 'soap_row', 'is_central', 'host_row', 'host_hid', 'own_satellite', 'mass_msun', 'mu',
                'radius_mpc', 'offset_mpc', 'distance_mpc', 'reach1', 'mu_class', 'bound_gas_count', 'bound_gas_mass_msun'):
        tables['ov_' + key] = census['ov_' + key][ov]
    pn = np.concatenate(out['pn_index']) if out['pn_index'] else np.empty(0, np.int64)
    for key in ('soap_row', 'hid', 'm200c_msun', 'r200c_mpc', 'offset_mpc', 'distance_mpc'):
        tables['pn_' + key] = census['pn_' + key][pn]
    return tables


def main():
    import h5py
    import grids as G
    import census as K
    from catalogue_stage import sphere_cells, csr_map
    started = time.perf_counter()
    for name in ('selected_halos.npz', 'overlap_tables.npz', 'metadata_plan.json'):
        assert not (STUDY / name).exists(), f'{name} exists: prepare runs once per attempt'
    apply = CFG['enlargement']['apply']
    if not isinstance(apply, bool):
        raise SystemExit('study_config.json enlargement.apply must be set to true or false at Checkpoint 1')
    census = dict(np.load(STUDY / 'census_pairs.npz'))
    population = dict(np.load(STUDY / 'census_population.npz'))
    probe = json.loads((STUDY / 'probe_g4.json').read_text())
    old = dict(np.load(STUDY / 'inputs' / 'radii234_selected_halos.npz'))
    old_plan = json.loads((STUDY / 'inputs' / 'radii234_metadata_plan.json').read_text())
    role = census['role']
    take = np.flatnonzero((role == 0) | ((role == 1) & apply))
    assert np.array_equal(take[:N_OLD], np.arange(N_OLD)) and np.all(role[:N_OLD] == 0)
    rows = census['halo_soap_row'][take]
    assert np.array_equal(rows[:N_OLD], old['soap_row'])
    snapshot = F.Snapshot(F.Store(ROOT), SNAPSHOT, CFG['snapshot'], read_from='chunks', chunk_cache_bytes=1024 ** 2)
    soap = F.Soap(F.Store(ROOT), SOAP)
    a, box_c = snapshot.a, snapshot.box_size_cmpc
    halos = soap.columns(names=('halo_index', 'centre_mpc', 'm200c_msun', 'r_200_crit_mpc'), rows=np.sort(rows))
    order = np.argsort(np.argsort(rows))                    # sorted read -> sample order
    halos = {k: v[order] for k, v in halos.items()}
    for key in ('halo_index', 'centre_mpc', 'm200c_msun', 'r_200_crit_mpc', 'soap_row'):
        assert np.array_equal(halos[key][:N_OLD], old[key]), key
    for name, (path, target) in SOAP_COLUMNS.items():
        halos[name] = read_rows(soap, path, rows, target)
    optional_missing = []
    for name, (path, target) in OPTIONAL_SOAP.items():
        if soap.has(path):
            halos[name] = read_rows(soap, path, rows, target)
        else:
            optional_missing.append(path)
    halos['centre_cmpc'] = halos['centre_mpc'] / a
    halos['centre_cmpc'][:N_OLD] = old['centre_cmpc']           # bitwise radii234 values for the legacy path
    edges = np.asarray(CFG['mass_edges_log10_Msun'])
    halos['mass_bin'] = np.searchsorted(edges, np.log10(halos['m200c_msun']), side='right') - 1
    assert np.array_equal(halos['mass_bin'][:N_OLD], old['mass_bin'])
    halos['role'] = role[take].astype(np.int8)
    halos['r_ext_mpc'] = np.maximum(CFG['extraction']['r200c_multiple'] * halos['r_200_crit_mpc'],
                                    CFG['extraction']['r500c_multiple'] * halos['r_500_crit_mpc'])
    assert np.allclose(halos['r_ext_mpc'], census['r_ext_mpc'][take], rtol=1e-12, atol=0)
    # inclusion probabilities and weights
    pool_rows = population['pool_soap_row']
    where = np.searchsorted(pool_rows, rows)
    assert np.array_equal(pool_rows[where], rows)
    if apply:
        probability = population['pool_inclusion_probability_enlarged'][where]
    else:
        n_pool = np.bincount(population['pool_bin'], minlength=len(edges) - 1)
        n_sel = np.bincount(halos['mass_bin'], minlength=len(edges) - 1)
        probability = (n_sel / np.maximum(n_pool, 1))[halos['mass_bin']]
    halos['inclusion_probability'] = probability
    halos['weight'] = 1 / probability
    halos['census_class_main'] = population['pool_class'][where]
    halos['census_max_mu'] = population['pool_max_mu'][where]
    halos['nearest_massive_offset_mpc'] = census['nearest_massive_offset_mpc'][take]
    halos['nearest_massive_distance_mpc'] = census['nearest_massive_distance_mpc'][take]
    com = soap.read('BoundSubhalo/CentreOfMass', np.sort(rows), target=C.MPC_CM)[order]
    delta = K.periodic_delta(com, halos['centre_mpc'], box_c * a)
    halos['centre_offset_R200c'] = np.linalg.norm(delta, axis=1) / halos['r_200_crit_mpc']
    # per-halo tables
    groups = {'row': census['group_central_row'], 'ptr': census['group_ptr'], 'member_hid': census['group_member_hid']}
    tables = build_tables(halos, census, take, groups)
    sat_check = (np.array_equal(tables['satellite_ptr'][:N_OLD + 1], old['satellite_ptr'])
                 and np.array_equal(tables['satellite_hid'][:tables['satellite_ptr'][N_OLD]], old['satellite_ids']))
    assert sat_check, 'own-satellite lists differ from radii234 (G3)'
    # cells: extraction spheres, and the legacy (radii234) 4 R200c pairs
    pad = CFG['extraction']['cell_padding_fraction'] * snapshot.cell_size_cmpc
    ph, pc = sphere_cells(halos['centre_cmpc'], halos['r_ext_mpc'] / a, snapshot.cell_size_cmpc, snapshot.dimension,
                          snapshot.lookup, box_c, pad)
    lh, lc = sphere_cells(halos['centre_cmpc'][N_OLD:], 4 * halos['r_200_crit_mpc'][N_OLD:] / a,
                          snapshot.cell_size_cmpc, snapshot.dimension, snapshot.lookup, box_c, pad)
    n_cells_total = int(snapshot.n_cells)
    legacy_keys = np.unique(np.r_[old['pair_halo'] * n_cells_total + old['pair_cell'], (lh + N_OLD) * n_cells_total + lc])
    keys = ph * n_cells_total + pc
    assert np.all(np.isin(legacy_keys, keys)), 'legacy pairs must be a subset of the extraction pairs'
    cells, csr_ptr, csr_halo = csr_map(ph, pc)
    csr_cell = np.repeat(cells, np.diff(csr_ptr))
    csr_legacy = np.isin(csr_halo * n_cells_total + csr_cell, legacy_keys)
    halos.update(pair_halo=ph, pair_cell=pc, pair_legacy=np.isin(keys, legacy_keys), cells=snapshot.read_order(cells),
                 csr_cells=cells, csr_ptr=csr_ptr, csr_halo=csr_halo, csr_legacy=csr_legacy, mass_edges=edges)
    old_cells = np.unique(old['pair_cell'])
    # dataset schema and the G4 re-check
    schema, bpp = {}, 0
    for name in FIELDS + EXTRA_FIELDS:
        ds = snapshot.dataset(name)
        nb = ds.dtype.itemsize * int(np.prod(ds.shape[1:]))
        bpp += nb
        schema[name] = {'dtype': str(ds.dtype), 'shape': list(ds.shape), 'bytes_per_particle': nb,
                        'attributes': {k: plain(v) for k, v in ds.attrs.items()}}
    auxiliary = snapshot.auxiliary_fields(FIELDS + EXTRA_FIELDS)
    with h5py.File(str(Path(ROOT) / SNAPSHOT), 'r') as top:
        named = [plain(x) for x in top['SubgridScheme/NamedColumns/XrayLuminosities'][...]]
    g4 = {'auxiliary_fields': auxiliary, 'xray_named_columns': named,
          'xray_column': CFG['extraction']['xray_column'],
          'rosat_at_column': named.index('ROSAT') if 'ROSAT' in named else None,
          'probe_small_cell_xray_identical': probe['small_cell']['fields']['XrayLuminosities']['values_identical'],
          'probe_halo_cell_xray_identical': probe['halo_cell_slice']['xray_chunk_vs_top']['values_identical'],
          'probe_ids_identical': probe['halo_cell_slice']['ids_chunk_vs_top_identical'],
          'probe_y_zero_where_sfr_positive': probe['halo_cell_slice']['y_zero_where_sfr_positive'],
          'sfr_description': schema['StarFormationRates']['attributes'].get('Description')}
    g4['passed'] = (not auxiliary and g4['rosat_at_column'] == CFG['extraction']['xray_column']
                    and g4['probe_small_cell_xray_identical'] and g4['probe_halo_cell_xray_identical']
                    and g4['probe_ids_identical'] and g4['probe_y_zero_where_sfr_positive']
                    and 'If positive, star formation rates' in str(g4['sfr_description']))
    assert g4['passed'], g4
    membership_sources = []
    for file_id in np.unique(snapshot.cell_file[cells]):
        file_id = int(file_id)
        path = MEMBERSHIP / f'membership_{CFG["snapshot"]:04d}.{file_id}.hdf5'
        with h5py.File(path, 'r') as f:
            ds = f['PartType0/GroupNr_bound']
            assert len(ds) == len(snapshot._chunk(file_id)['PartType0/Masses'])
            membership_sources.append({'file': file_id, 'path': str(path), 'shape': list(ds.shape), 'dtype': str(ds.dtype)})
    bpp_total = bpp + 8
    unique_rows = int(snapshot.counts[cells].sum())
    old_rows_old_cells = int(snapshot.counts[old_cells].sum())
    budget = unique_rows * bpp_total
    assert budget <= CFG['extraction']['read_budget_bytes'], f'read budget {budget / 1e9:.1f} GB exceeds the cap'
    np.savez_compressed(STUDY / 'selected_halos.npz', **halos)
    np.savez_compressed(STUDY / 'overlap_tables.npz', **tables)
    swift = old_plan['swift_check_indices']
    plan = {'simulation': CFG['simulation'], 'snapshot': CFG['snapshot'], 'redshift': snapshot.redshift,
            'scale_factor': snapshot.a, 'box_size_cmpc': box_c,
            'snapshot_path': str(Path(ROOT) / SNAPSHOT), 'soap_path': str(Path(ROOT) / SOAP),
            'halo_count': int(len(rows)), 'old_rows': N_OLD, 'enlargement_applied': apply,
            'sample_per_bin': np.bincount(halos['mass_bin'], minlength=len(edges) - 1).tolist(),
            'population_per_bin': np.bincount(population['pool_bin'], minlength=len(edges) - 1).tolist(),
            'mass_edges_log10_msun': edges.tolist(), 'selected_cells': int(len(cells)),
            'selected_candidate_rows': unique_rows,
            'candidate_halo_particle_comparisons': int(snapshot.counts[pc].sum()),
            'bytes_per_particle_with_membership': bpp_total, 'logical_bytes_one_direct_pass': budget,
            'radii234_cells_rows_for_reference': old_rows_old_cells,
            'radii234_logical_bytes_at_new_bpp': old_rows_old_cells * bpp_total,
            'read_budget_bytes': CFG['extraction']['read_budget_bytes'],
            'swift_check_indices': swift, 'header': {k: plain(v) for k, v in snapshot.header.items()},
            'cosmology': {k: plain(v) for k, v in snapshot.file['Cosmology'].attrs.items()},
            'field_schema': schema, 'baryon_fraction': snapshot.cosmology['baryon_fraction'],
            'softening_physical_mpc': min(snapshot.a * snapshot.softening_comoving_mpc, snapshot.softening_max_physical_mpc),
            'recent_agn_filter': soap.heating_filter(),
            'xray': {'source': CFG['extraction']['xray_source'], 'column': CFG['extraction']['xray_column'],
                     'names': named, 'physical_cgs_factor': C.physical_factor(snapshot.dataset('XrayLuminosities'))},
            'g4_recheck': g4, 'membership_sources': membership_sources,
            'membership_order_contract': old_plan['membership_order_contract'],
            'grids': G.g2_report(), 'optional_soap_missing': optional_missing,
            'tables': {'overlapper_rows': int(len(tables['ov_hid'])), 'owner_table_rows': int(len(tables['owner_hid'])),
                       'partition_rows': int(len(tables['pn_hid'])), 'clean_with_placebo': int(np.sum(tables['placebo_partner'] >= 0)),
                       'placebo_nbr_fallback_isotropic': int(tables['placebo_nbr_fallback_isotropic'].sum())},
            'own_satellite_lists_equal_radii234': bool(sat_check),
            'particle_values_read': False, 'wall_seconds': time.perf_counter() - started}
    try:
        import psutil
        plan['resources'] = {'available_memory_GiB': psutil.virtual_memory().available / 1024 ** 3,
                             'load': list(os.getloadavg()), 'host': __import__('socket').getfqdn()}
    except ImportError:
        pass
    (STUDY / 'metadata_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k: plan[k] for k in ('halo_count', 'sample_per_bin', 'selected_cells', 'selected_candidate_rows',
                                           'logical_bytes_one_direct_pass', 'radii234_logical_bytes_at_new_bpp',
                                           'tables', 'wall_seconds')}, indent=2))
    snapshot.close()
    soap.close()


if __name__ == '__main__':
    main()
