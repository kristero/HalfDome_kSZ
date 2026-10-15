"""One resumable full-snapshot pass of the overlap/covariance study (plan Step 1.2) on maas.

Adapted from radii234 extract_study.read_file/merge_products. Per chunk file a worker streams the
selected top-level cells (80 MiB HDF5 chunk cache per dataset), pairs HBT membership by file number
and particle order, and for every (halo, cell) pair:
  * selects the extraction sphere r <= max(4 R200c, 6 R500c): per-axis periodic pre-filter, then the
    exact periodic test with radii234's arithmetic;
  * converts the remaining fields only for the union of selected particles of that cell;
  * accumulates the radii234 legacy sums on radii234's own (halo, cell) pairs and exact 4 R200c rule
    (bit-identical by construction) and every new block of measure_overlap.accumulate.
Workers save compact file products (file_products/file_NNN.npz + .json receipt) that survive an
interrupted run; the parent merges them into measurements_*.npz and apertures.npz. No particle data
are saved except the legacy-subset IDs of radii234's 45 SWIFT-check haloes (hashed, then discarded).

Usage on maas (via control.py): python -B extract_study.py [--gate-cell | --gate-file N]
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import flamingo_io as F  # noqa: E402
import pilot_physics as P  # noqa: E402
import measure as M  # noqa: E402
import grids as G  # noqa: E402
import measure_overlap as MO  # noqa: E402
from prepare_study import ROOT, SNAPSHOT, MEMBERSHIP, FIELDS, EXTRA_FIELDS, CFG, STUDY  # noqa: E402

CACHE = int(CFG['chunk_cache_MiB']) * 1024 ** 2
POLITE = CFG['politeness']
REOPEN = max(1, int(CFG.get('reopen_every_cells', 16)))
BATCH = int(os.environ.get('OVERLAP_BATCH', 1_000_000))   # particles per measure_overlap.accumulate call (env: tests)
PLAN = H = T = PAINTED = FB = CTX = None
HALO_CACHE = {}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=C.json_default) + '\n')
    temporary.replace(path)


def load_inputs():
    plan = json.loads((STUDY / 'metadata_plan.json').read_text())
    with np.load(STUDY / 'selected_halos.npz') as f:
        halos = {k: f[k] for k in f.files}
    with np.load(STUDY / 'overlap_tables.npz') as f:
        tables = {k: f[k] for k in f.files}
    with np.load(STUDY / 'census_population.npz') as f:
        painted = f['painted_by_hid']
    return plan, halos, tables, painted


def initialize(limit=True):
    global PLAN, H, T, PAINTED, FB, CTX
    sys.dont_write_bytecode = True
    if limit:
        size = int(CFG['RLIMIT_AS_GiB'] * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (size, size))
    PLAN, H, T, PAINTED = load_inputs()
    FB = PLAN['baryon_fraction']
    CTX = {'heating': PLAN['recent_agn_filter'], 'painted': PAINTED,
           't_xray_min': float(CFG['extraction']['xray_selected_temperature_min_k'])}


def halo_dict(h, box):
    """Per-halo inputs of legacy_sums and measure_overlap.accumulate (cached per worker)."""
    if h in HALO_CACHE:
        return HALO_CACHE[h]
    o0, o1 = T['owner_ptr'][h], T['owner_ptr'][h + 1]
    k0, k1 = T['ov_ptr'][h], T['ov_ptr'][h + 1]
    p0, p1 = T['pn_ptr'][h], T['pn_ptr'][h + 1]
    s0, s1 = T['satellite_ptr'][h], T['satellite_ptr'][h + 1]
    excision = []
    for v in range(len(MO.VARIANTS)):
        e0, e1 = T['exc_ptr'][4 * h + v], T['exc_ptr'][4 * h + v + 1]
        excision.append((T['exc_offset_mpc'][e0:e1], T['exc_radius_mpc'][e0:e1]))
    one = {'centre_cmpc': H['centre_cmpc'][h], 'radius': H['r_200_crit_mpc'][h], 'r500': H['r_500_crit_mpc'][h],
           'r_ext': H['r_ext_mpc'][h], 'a': PLAN['scale_factor'], 'box': box, 'id': H['halo_index'][h],
           'satellites': T['satellite_hid'][s0:s1],
           'owner_hid': T['owner_hid'][o0:o1], 'owner_code': T['owner_code'][o0:o1].astype(np.int64),
           'owner_group': T['owner_group'][o0:o1].astype(np.int64), 'owner_own': T['owner_own'][o0:o1],
           'ov_hid': T['ov_hid'][k0:k1], 'excision': excision,
           'pn_offset': T['pn_offset_mpc'][p0:p1], 'pn_r200': T['pn_r200c_mpc'][p0:p1]}
    HALO_CACHE[h] = one
    return one


def select_sphere(pos, centre, radius_cmpc, box):
    """Indices (ascending) of particles with |periodic delta| <= radius: per-axis pre-filter, exact test."""
    limit = radius_cmpc * (1 + 1e-9)
    dx = pos[:, 0] - centre[0]
    dx -= box * np.rint(dx / box)
    cand = np.flatnonzero(np.abs(dx) <= limit)
    for axis in (1, 2):
        d = pos[cand, axis] - centre[axis]
        d -= box * np.rint(d / box)
        cand = cand[np.abs(d) <= limit]
    delta = pos[cand] - centre
    delta -= box * np.rint(delta / box)
    return cand[np.einsum('ij,ij->i', delta, delta) <= radius_cmpc ** 2]


def convert(raw, index, snap):
    """radii234 unit conversion (same expressions) for the selected rows only, plus SFR and ROSAT L_X."""
    data = {name: raw[name][index].astype(np.float64) * snap.factor(name) for name in F.GAS_FIELDS}
    data.update(ParticleIDs=raw['ParticleIDs'][index],
                Hydrogen=raw['SmoothedElementMassFractions'][index, 0].astype(float),
                Helium=raw['SmoothedElementMassFractions'][index, 1].astype(float),
                Pressures=raw['Pressures'][index].astype(float) * C.physical_factor(snap.dataset('Pressures')) / 10,
                StarFormationRates=raw['StarFormationRates'][index].astype(np.float64),
                XrayRosat=raw['XrayLuminosities'][index, CFG['extraction']['xray_column']].astype(np.float64)
                * C.physical_factor(snap.dataset('XrayLuminosities')))
    return data


def check_values(data):
    assert all(np.isfinite(v).all() for k, v in data.items() if k != 'ParticleIDs')
    assert np.all(data['Masses'] > 0) & np.all(data['Densities'] > 0)
    assert np.all(data['ComptonYParameters'] >= 0) & np.all(data['ElectronNumberDensities'] >= 0)
    assert np.all(data['XrayRosat'] >= 0)


def new_part(n_ov):
    part = MO.empty_blocks(n_ov)
    part['legacy'] = np.zeros((4, len(P.SUM_NAMES), M.NB), dtype=np.float64)
    return part


def read_file(file_id, gate_cell=None, out_dir='file_products'):
    started, cpu = time.perf_counter(), time.process_time()
    directory = STUDY / out_dir
    directory.mkdir(exist_ok=True)
    path = directory / f'file_{file_id:03d}.npz'
    receipt = path.with_suffix('.json')
    if gate_cell is None and path.exists() and receipt.exists():
        return json.loads(receipt.read_text())
    snap = F.Snapshot(F.Store(ROOT), SNAPSHOT, CFG['snapshot'], read_from='chunks', chunk_cache_bytes=CACHE)
    box = snap.box_size_cmpc
    a = PLAN['scale_factor']
    cells = [int(c) for c in H['cells'] if snap.cell_file[c] == file_id]
    if gate_cell is not None:
        cells = [int(gate_cell)]
    parts, ids, gate = {}, {}, []
    candidate_rows = selected_entries = 0
    swift = set(PLAN['swift_check_indices'])
    # HDF5 holds decompression memory while a file stays open and releases it on close (memory_diagnostic.json):
    # reopen the snapshot chunk and membership files every REOPEN cells to bound each worker's footprint.
    member_file = None
    try:
        for index, cell in enumerate(cells):
            if index % REOPEN == 0:
                if member_file is not None:
                    member_file.close()
                    snap.close()
                    snap = F.Snapshot(F.Store(ROOT), SNAPSHOT, CFG['snapshot'], read_from='chunks', chunk_cache_bytes=CACHE)
                member_file = h5py.File(MEMBERSHIP / f'membership_{CFG["snapshot"]:04d}.{file_id}.hdf5', 'r', rdcc_nbytes=CACHE)
                membership = member_file['PartType0/GroupNr_bound']
                assert len(membership) == len(snap._chunk(file_id)['PartType0/Masses'])
            raw = snap.read_cell_raw(cell, FIELDS + EXTRA_FIELDS)
            pos = raw['Coordinates'].astype(np.float64) * snap.factor('Coordinates')
            lo, size = int(snap.cell_offset[cell]), int(snap.counts[cell])
            owner = np.asarray(membership[lo:lo + size], dtype=np.int64)
            candidate_rows += size
            k = int(np.searchsorted(H['csr_cells'], cell))
            assert H['csr_cells'][k] == cell
            pairs = [(int(h), bool(leg)) for h, leg in zip(H['csr_halo'][H['csr_ptr'][k]:H['csr_ptr'][k + 1]],
                                                           H['csr_legacy'][H['csr_ptr'][k]:H['csr_ptr'][k + 1]])]
            selections = {h: select_sphere(pos, H['centre_cmpc'][h], H['r_ext_mpc'][h] / a, box) for h, _ in pairs}
            union = np.unique(np.concatenate([selections[h] for h, _ in pairs])) if pairs else np.empty(0, np.int64)
            data_u = convert(raw, union, snap)
            check_values(data_u)
            for h, legacy in pairs:
                one = halo_dict(h, box)
                if len(selections[h]) == len(union):      # the only halo of the cell (or all select the same rows)
                    data = data_u
                else:
                    where = np.searchsorted(union, selections[h])
                    data = {name: value[where] for name, value in data_u.items()}
                member = owner[selections[h]]
                if h not in parts:
                    parts[h] = new_part(len(one['ov_hid']))
                # new blocks in batches (linear sums: exact merge; bounds the temporaries of cluster cells)
                in4 = np.zeros(len(member), bool)
                for b0 in range(0, max(len(member), 1), BATCH):
                    batch = slice(b0, b0 + BATCH)
                    item = MO.accumulate({name: value[batch] for name, value in data.items()}, member[batch], one, CTX)
                    in4[batch] = item.pop('in4')
                    MO.add_into(parts[h], item)
                selected_entries += len(member)
                if legacy:
                    subset = {name: value[in4] for name, value in data.items()}
                    legacy_sums = MO.legacy_sums(subset, member[in4], one, PLAN['recent_agn_filter'])
                    parts[h]['legacy'] += legacy_sums
                    if h in swift:
                        ids.setdefault(h, []).append(subset['ParticleIDs'])
                    if gate_cell is not None:
                        whole = MO.accumulate(data, member, one, CTX)
                        assert np.array_equal(whole.pop('in4'), in4)
                        gate.append(gate_checks(raw, snap, owner, h, one, selections[h], in4, legacy_sums, whole, data, member))
            del raw, pos, owner, data_u
    finally:
        if member_file is not None:
            member_file.close()
        snap.close()
    rows = np.array(sorted(parts), dtype=np.int64)
    result = {'file': int(file_id), 'cells': len(cells), 'candidate_rows': candidate_rows, 'halo_parts': len(rows),
              'selected_particle_halo_entries': selected_entries, 'cpu_seconds': time.process_time() - cpu,
              'wall_seconds': time.perf_counter() - started,
              'peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if gate_cell is not None:
        result['gate'] = gate
        first = int(rows[0]) if len(rows) else None
        result['empty_then_populated'] = empty_merge_check(parts[first] if first is not None else None,
                                                           halo_dict(first, box) if first is not None else None)
        return result
    arrays = {'halo_rows': rows}
    for key in ('legacy',) + tuple(k for k in MO.MERGE_KEYS if k != 'overlapper'):
        arrays[key] = np.array([parts[int(h)][key] for h in rows])
    ov_len = np.array([len(parts[int(h)]['overlapper']) for h in rows], np.int64)
    arrays['overlapper_ptr'] = np.r_[0, np.cumsum(ov_len)]
    arrays['overlapper'] = (np.concatenate([parts[int(h)]['overlapper'] for h in rows])
                            if ov_len.sum() else np.zeros((0, 2, 2, len(MO.OVERLAPPER_QUANTITIES))))
    id_rows = sorted(ids)
    arrays['id_halo_rows'] = np.array(id_rows, dtype=np.int64)
    arrays['id_ptr'] = np.r_[0, np.cumsum([sum(len(x) for x in ids[h]) for h in id_rows])].astype(np.int64)
    arrays['ids'] = np.concatenate([np.concatenate(ids[h]) for h in id_rows]) if id_rows else np.empty(0, np.uint64)
    temporary = path.with_suffix('.temporary.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    result['product_bytes'] = path.stat().st_size
    result['wall_seconds'] = time.perf_counter() - started
    write_json(receipt, result)
    return result


def gate_checks(raw, snap, owner, h, one, selection, in4, legacy_sums, item, data, member):
    """G5 on one (halo, cell): the radii234 code path on the full cell versus the new path."""
    old = {name: raw[name].astype(np.float64) * snap.factor(name) for name in F.GAS_FIELDS}
    old.update(ParticleIDs=raw['ParticleIDs'], Hydrogen=raw['SmoothedElementMassFractions'][:, 0].astype(float),
               Helium=raw['SmoothedElementMassFractions'][:, 1].astype(float),
               Pressures=raw['Pressures'].astype(float) * C.physical_factor(snap.dataset('Pressures')) / 10)
    radius = one['radius']
    delta = old['Coordinates'] - one['centre_cmpc']
    delta -= snap.box_size_cmpc * np.rint(delta / snap.box_size_cmpc)
    inside = np.einsum('ij,ij->i', delta, delta) <= (4 * radius / PLAN['scale_factor']) ** 2
    selected = {name: v[inside] for name, v in old.items()}
    reference = M.accumulate(selected, owner[inside], {'centre_cmpc': one['centre_cmpc'], 'radius': radius,
                             'mass': H['m200c_msun'][h], 'id': one['id'], 'satellites': one['satellites'],
                             'a': PLAN['scale_factor'], 'box': snap.box_size_cmpc}, FB, PLAN['recent_agn_filter'],
                             np.eye(3)[None].repeat(2, axis=0))['sums']
    tot = {q: G.to_old(item['owner200'][:, :, MO.QUANTITIES.index(q)].sum(axis=(0, 1))) for q in MO.QUANTITIES}
    legacy_total = legacy_sums.sum(axis=0)

    def close(a_, b_, tol=1e-12):
        scale = float(np.max(np.abs(b_))) if np.size(b_) else 0.0
        return bool(np.allclose(a_, b_, rtol=tol, atol=tol * scale + 1e-300))
    shared = ('count', 'mass', 'y', 'electrons', 'electron_kT', 'volume', 'gas_pressure_volume')
    components = {0: [0], 1: [c for c in range(1, 13) if (c - 1) % 2 == 0] + [MO.OTHER_OWN],
                  2: [c for c in range(1, 13) if (c - 1) % 2 == 1] + [MO.OTHER_PAINTED, MO.OTHER_NOT_PAINTED], 3: [MO.UNBOUND]}
    per_component = all(close(G.to_old(item['owner200'][codes][:, :, MO.QUANTITIES.index(q)].sum(axis=(0, 1))),
                              legacy_sums[comp, P.SUM_NAMES.index(q)])
                        for comp, codes in components.items() for q in shared)
    reduced_total = np.stack([item['owner200'][:, :, MO.QUANTITIES.index(q)].sum(axis=(0, 1)) for q in MO.REDUCED])
    excision_ok = all(close(item['excision'][v].sum(axis=(0, 1, 2)), reduced_total) for v in range(len(MO.VARIANTS)))
    member_ok = close(item['excision'][0][:, 0].sum(axis=(0, 1)),
                      np.stack([item['owner200'][0, :, MO.QUANTITIES.index(q)].sum(axis=0) for q in MO.REDUCED]))
    checks = {
        'halo': int(h), 'n_cell': int(len(owner)), 'n_sphere': int(len(selection)), 'n_4R': int(in4.sum()),
        'legacy_mask_equal': bool(np.array_equal(np.flatnonzero(inside), selection[in4])),
        'legacy_sums_bit_identical': bool(np.array_equal(legacy_sums, reference)),
        'legacy_dtype_float64': legacy_sums.dtype == np.float64,
        'owner_totals_match_legacy': all(close(tot[q], legacy_total[P.SUM_NAMES.index(q)]) for q in shared),
        'owner_count_exact': bool(np.array_equal(tot['count'], legacy_total[0])),
        'owner_components_match_legacy': per_component,
        'phase_split_matches_y_noagn': close(G.to_old(item['owner200'][:, 1:, MO.QUANTITIES.index('y')].sum(axis=(0, 1))),
                                             legacy_total[P.SUM_NAMES.index('y_noagn')]),
        'excision_variants_sum_to_totals': excision_ok, 'excision_member_central_matches_owner0': member_ok,
        'partition_sums_to_totals': close(item['partition'].sum(axis=(0, 1, 2)), reduced_total),
        'aux_matches_legacy': all(close(G.to_old(item['aux200'][:, i].sum(axis=0)), legacy_total[P.SUM_NAMES.index(name)])
                                  for i, name in enumerate(MO.AUX)),
        'owner500_count_equals_in6': int(item['owner500'][:, :, 0].sum()) == int(item['diagnostics'][6]),
        'y_zero_where_star_forming': int(item['diagnostics'][8]) == 0, 'star_forming_particles': int(item['diagnostics'][9]),
        'apertures_sphere_count': int(item['apertures'][3, 0, 0]) == len(selection),
        'membership_not_in_soap': int(item['diagnostics'][0])}
    checks['passed'] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def empty_merge_check(populated, one):
    """Merging an empty cell then a populated one gives the populated blocks exactly (and keeps dtypes)."""
    if populated is None:
        return False
    empty_data = {name: np.zeros((0, 3) if name == 'Coordinates' else 0) for name in
                  list(F.GAS_FIELDS) + ['Hydrogen', 'Helium', 'Pressures', 'StarFormationRates', 'XrayRosat']}
    empty_data['ParticleIDs'] = np.zeros(0, np.uint64)
    item = MO.accumulate(empty_data, np.zeros(0, np.int64), one, CTX)
    item.pop('in4')
    legacy = MO.legacy_sums({k: v for k, v in empty_data.items()}, np.zeros(0, np.int64), one, PLAN['recent_agn_filter'])
    total = new_part(len(one['ov_hid']))
    MO.add_into(total, item)
    total['legacy'] += legacy
    MO.add_into(total, {k: populated[k] for k in MO.MERGE_KEYS})
    total['legacy'] += populated['legacy']
    same = all(np.array_equal(total[k], populated[k]) and total[k].dtype == populated[k].dtype
               for k in MO.MERGE_KEYS + ('legacy',))
    return bool(same and legacy.dtype == np.float64 and all(item[k].dtype == MO.empty_blocks(0)[k].dtype for k in MO.MERGE_KEYS))


def polite_workers():
    """Worker count for the shared host: at most the configured number, and no more than the hardware threads
    left below max_load_fraction of the machine by the current 1-minute load (never fewer than min_workers)."""
    room = int(POLITE['max_load_fraction'] * os.cpu_count() - os.getloadavg()[0])
    return max(POLITE['min_workers'], min(CFG['workers'], room))


def monitor(stop, record, limit):
    import psutil
    process = psutil.Process()
    usage = {}
    # free-memory floor of the shared host: a fraction of its RAM (maas 188 GiB -> 22.6 GiB), never below the floor
    floor = max(POLITE['min_available_fraction_running'] * psutil.virtual_memory().total,
                POLITE['min_available_GiB_running_floor'] * 1024 ** 3)
    record['system_free_memory_floor_bytes'] = floor
    while not stop.wait(1):
        tree = [process] + process.children(recursive=True)
        rss = 0
        for p in tree:
            try:
                rss += p.memory_info().rss
                t = p.cpu_times()
                usage[p.pid] = max(usage.get(p.pid, 0), t.user + t.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        record['peak_tree_rss_bytes'] = max(record.get('peak_tree_rss_bytes', 0), rss)
        record['sampled_cpu_seconds'] = sum(usage.values())
        # shared host: stop (resumably) on our own RSS limit or when the machine's free memory runs low
        available = psutil.virtual_memory().available
        record['min_system_available_bytes'] = min(record.get('min_system_available_bytes', available), available)
        if rss > limit:
            record['stop_reason'] = 'own RSS %.1f GiB above %.0f GiB' % (rss / 1024 ** 3, limit / 1024 ** 3)
        elif available < floor:
            record['stop_reason'] = 'system free memory %.1f GiB below %.1f GiB' % (available / 1024 ** 3, floor / 1024 ** 3)
        if 'stop_reason' in record:
            for p in tree[1:]:
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
            os.kill(process.pid, signal.SIGTERM)


def merge_products(files):
    n = len(H['halo_index'])
    n_ov = int(T['ov_ptr'][-1])
    empty = MO.empty_blocks(0)
    total = {key: np.zeros((n,) + empty[key].shape, dtype=empty[key].dtype) for key in MO.MERGE_KEYS if key != 'overlapper'}
    total['legacy'] = np.zeros((n, 4, len(P.SUM_NAMES), M.NB))
    overlapper = np.zeros((n_ov, 2, 2, len(MO.OVERLAPPER_QUANTITIES)))
    ids = {}
    for file_id in sorted(files):
        with np.load(STUDY / 'file_products' / f'file_{file_id:03d}.npz') as p:
            rows = p['halo_rows']
            for key in total:
                if key == 'id_hash':
                    total[key][rows, :, 0] += p[key][:, :, 0]
                    total[key][rows, :, 1] ^= p[key][:, :, 1]
                else:
                    total[key][rows] += p[key]
            ptr = p['overlapper_ptr']
            block = p['overlapper']
            for i, h in enumerate(rows):
                k0 = T['ov_ptr'][h]
                overlapper[k0:k0 + ptr[i + 1] - ptr[i]] += block[ptr[i]:ptr[i + 1]]
            for i, h in enumerate(p['id_halo_rows']):
                ids.setdefault(int(h), []).append(p['ids'][p['id_ptr'][i]:p['id_ptr'][i + 1]])
    hashes = {str(h): P.hash_ids(np.concatenate(parts)) for h, parts in ids.items()}
    # full legacy-subset ID lists of the SWIFT-check haloes (kept remotely: swift_ids/ is not fetched by default)
    (STUDY / 'swift_ids').mkdir(exist_ok=True)
    id_rows = sorted(ids)
    lists = [np.sort(np.concatenate(ids[h]).astype(np.uint64)) for h in id_rows]
    np.savez_compressed(STUDY / 'swift_ids' / 'swift_check_ids.npz', halo_rows=np.array(id_rows, np.int64),
                        ptr=np.r_[0, np.cumsum([len(x) for x in lists])].astype(np.int64),
                        ids=np.concatenate(lists) if lists else np.empty(0, np.uint64))
    names = {'owners': MO.OWNERS, 'phases': MO.PHASES, 'quantities': MO.QUANTITIES, 'units': MO.UNITS,
             'reduced': MO.REDUCED, 'variants': MO.VARIANTS, 'zones': MO.ZONES, 'members': MO.MEMBERS, 'agn': MO.AGN,
             'partition_owners': MO.PARTITION_OWNERS, 'painted_bound': MO.PAINTED_BOUND, 'aux': MO.AUX,
             'overlapper_quantities': MO.OVERLAPPER_QUANTITIES, 'inertia': MO.INERTIA, 'apertures': MO.APERTURES,
             'aperture_quantities': MO.APERTURE_QUANTITIES, 'diagnostics': MO.DIAGNOSTICS}
    np.savez_compressed(STUDY / 'measurements_legacy.npz', sums=total['legacy'], edges=P.EDGES, sum_names=np.array(P.SUM_NAMES))
    np.savez_compressed(STUDY / 'measurements_owner.npz', owner200=total['owner200'], owner500=total['owner500'],
                        aux200=total['aux200'], aux500=total['aux500'], hs200=total['hs200'], hs500=total['hs500'],
                        x200=G.X200, x500=G.X500, hs_edges=MO.HS_EDGES)
    # float64 throughout: float32 would overflow the electron counts (n_e V ~ 1e68-1e72 > 3.4e38)
    np.savez_compressed(STUDY / 'measurements_excision.npz', excision=total['excision'], x200=G.X200,
                        kappa_zone_edges=MO.ZONE_EDGES)
    np.savez_compressed(STUDY / 'measurements_partition.npz', partition=total['partition'], x200=G.X200)
    np.savez_compressed(STUDY / 'apertures.npz', apertures=total['apertures'], inertia=total['inertia'],
                        id_hash=total['id_hash'], id_count=total['id_count'], diagnostics=total['diagnostics'],
                        overlapper=overlapper, overlapper_ptr=T['ov_ptr'])
    schema = {'halo_rows': n, 'names': names,
              'arrays': {'measurements_legacy.npz:sums': [n, 4, len(P.SUM_NAMES), M.NB],
                         'measurements_owner.npz:owner200': list(total['owner200'].shape),
                         'measurements_owner.npz:owner500': list(total['owner500'].shape),
                         'measurements_owner.npz:aux200': list(total['aux200'].shape),
                         'measurements_owner.npz:hs200': list(total['hs200'].shape),
                         'measurements_excision.npz:excision': list(total['excision'].shape),
                         'measurements_partition.npz:partition': list(total['partition'].shape),
                         'apertures.npz:apertures': list(total['apertures'].shape),
                         'apertures.npz:overlapper (rows follow overlap_tables ov_ptr)': list(overlapper.shape)},
              'contract': MO.__doc__}
    write_json(STUDY / 'schema.json', schema)
    write_json(STUDY / 'particle_hashes.json', hashes)
    return hashes


def choose_gate_cell():
    """Home cell of the old (radii234) halo whose home cell has the fewest particles (>= 20,000)."""
    snap = F.Snapshot(F.Store(ROOT), SNAPSHOT, CFG['snapshot'], read_from='chunks', chunk_cache_bytes=1024 ** 2)
    home = snap.cell_of(H['centre_cmpc'][:CFG['old_sample']['rows'][1]])
    counts = snap.counts[home]
    ok = counts >= min(20000, int(counts.max()))
    cell = int(home[ok][np.argmin(counts[ok])])
    file_id = int(snap.cell_file[cell])
    snap.close()
    return cell, file_id


def main():
    global PLAN, H, T, PAINTED, FB, CTX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-cell', action='store_true', help='G5 one-cell gate (writes one_cell_gate.json)')
    parser.add_argument('--gate-file', type=int, help='G6 one-file gate (writes gate_file.json)')
    args = parser.parse_args()
    if args.gate_cell:
        initialize()
        cell, file_id = choose_gate_cell()
        result = read_file(file_id, gate_cell=cell, out_dir='gate_cell')
        result['passed'] = bool(result['gate']) and all(g['passed'] for g in result['gate']) and result['empty_then_populated']
        write_json(STUDY / 'one_cell_gate.json', result)
        print(json.dumps(result, indent=2, default=C.json_default))
        return
    if args.gate_file is not None:
        initialize()
        import shutil
        shutil.rmtree(STUDY / 'gate_file', ignore_errors=True)   # always measure afresh (read_file resumes otherwise)
        peaks = {'rss': 0, 'vms': 0}
        done = threading.Event()

        def sample():
            import psutil
            me = psutil.Process()
            while not done.wait(0.5):
                info = me.memory_info()
                peaks['rss'], peaks['vms'] = max(peaks['rss'], info.rss), max(peaks['vms'], info.vms)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        result = read_file(args.gate_file, out_dir='gate_file')
        done.set()
        sampler.join(timeout=2)
        result['RLIMIT_AS_GiB'] = CFG['RLIMIT_AS_GiB']
        result['sampled_peak_rss_GiB'] = peaks['rss'] / 1024 ** 3
        result['sampled_peak_vms_GiB'] = peaks['vms'] / 1024 ** 3
        snap = F.Snapshot(F.Store(ROOT), SNAPSHOT, CFG['snapshot'], read_from='chunks', chunk_cache_bytes=1024 ** 2)
        files = np.unique(snap.cell_file[H['cells']])
        rows_per_file = {int(f): int(snap.counts[H['cells'][snap.cell_file[H['cells']] == f]].sum()) for f in files}
        snap.close()
        result['files'] = len(files)
        result['projected_worker_seconds_total'] = result['wall_seconds'] / max(result['candidate_rows'], 1) * sum(rows_per_file.values())
        result['projected_run_minutes'] = result['projected_worker_seconds_total'] / CFG['workers'] / 60
        result['passed'] = result['peak_rss_MiB'] < 0.9 * CFG['RLIMIT_AS_GiB'] * 1024
        write_json(STUDY / 'gate_file.json', result)
        print(json.dumps(result, indent=2))
        return
    initialize(limit=False)
    assert not (STUDY / 'complete.json').exists()
    assert json.loads((STUDY / 'one_cell_gate.json').read_text())['passed']
    assert json.loads((STUDY / 'gate_file.json').read_text())['passed']
    assert PLAN['logical_bytes_one_direct_pass'] <= CFG['extraction']['read_budget_bytes']
    if os.nice(0) < POLITE['nice']:
        os.nice(POLITE['nice'] - os.nice(0))
    workers = polite_workers()
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(f'{CFG["wall_limit_minutes"]}-minute stop')))
    signal.alarm(int(CFG['wall_limit_minutes'] * 60))
    record = {'started_UTC': datetime.now(timezone.utc).isoformat(), 'workers': workers,
              'workers_configured': CFG['workers'], 'nice': os.nice(0), 'load_at_start': list(os.getloadavg()),
              'RSS_stop_GiB': CFG['RSS_stop_GiB'], 'RLIMIT_AS_GiB': CFG['RLIMIT_AS_GiB'],
              'wall_limit_minutes': CFG['wall_limit_minutes'], 'chunk_cache_MiB': CFG['chunk_cache_MiB'],
              'logical_read_bytes': PLAN['logical_bytes_one_direct_pass'], 'physical_NFS_bytes_measured': False,
              'scheduler_jobs_submitted': 0, 'completed': False}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(MemoryError(record.get('stop_reason', 'stopped by SIGTERM'))))
    stop = threading.Event()
    watch = threading.Thread(target=monitor, args=(stop, record, CFG['RSS_stop_GiB'] * 1024 ** 3), daemon=True)
    watch.start()
    (STUDY / 'file_products').mkdir(exist_ok=True)
    try:
        snap = F.Snapshot(F.Store(ROOT), SNAPSHOT, CFG['snapshot'], read_from='chunks', chunk_cache_bytes=1024 ** 2)
        files = sorted(int(x) for x in np.unique(snap.cell_file[H['cells']]))
        snap.close()
        receipts = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            todo = iter(files)
            pending = {pool.submit(read_file, next(todo)) for _ in range(min(workers, len(files)))}
            while pending:
                future = next(as_completed(pending))
                pending.remove(future)
                receipt = future.result()
                receipts.append(receipt)
                write_json(STUDY / 'progress.json', {'phase': 'full-snapshot extraction', 'done': len(receipts),
                                                    'total': len(files), 'elapsed_seconds': time.perf_counter() - started})
                print(json.dumps({'file_done': receipt['file'], 'completed_files': len(receipts),
                                  'wall_seconds': time.perf_counter() - started}), flush=True)
                next_file = next(todo, None)
                if next_file is not None:
                    pending.add(pool.submit(read_file, next_file))
        write_json(STUDY / 'file_receipts.json', receipts)
        write_json(STUDY / 'progress.json', {'phase': 'merge', 'elapsed_seconds': time.perf_counter() - started})
        hashes = merge_products(files)
        swift = PLAN['swift_check_indices']
        stop.set()
        watch.join(timeout=2)
        record.update(completed=True, wall_seconds=time.perf_counter() - started, files=len(files),
                      main_cpu_seconds=time.process_time(),
                      children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                      + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
                      swift_hashes_written=len(hashes), swift_check_haloes=len(swift))
        write_json(STUDY / 'complete.json', record)
        print(json.dumps(record, indent=2), flush=True)
    except BaseException as error:
        stop.set()
        write_json(STUDY / 'failed.json', {'error': str(error), 'traceback': traceback.format_exc(), 'resources': record})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
