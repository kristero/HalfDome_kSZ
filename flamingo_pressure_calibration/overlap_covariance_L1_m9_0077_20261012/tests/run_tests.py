"""Local tests of the Step-1 code (no remote access). Writes tests/test_results.json.

  python -B tests/run_tests.py

Synthetic particle sets with known membership, phases and overlappers are run through
measure_overlap.accumulate / legacy_sums and extract_study.select_sphere and compared with brute-force
references; geometry is compared with Monte Carlo; the census search with brute force; grids with G2.
"""
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(STUDY / 'dependencies'))
sys.path.insert(0, str(STUDY))
import census as K  # noqa: E402
import extract_study as E  # noqa: E402
import geometry as GEO  # noqa: E402
import grids as G  # noqa: E402
import measure as M  # noqa: E402
import measure_overlap as MO  # noqa: E402
import pilot_physics as P  # noqa: E402
import prepare_study as PS  # noqa: E402

HEATING = {'a_limit': 0.9989555948537917, 'tmin_k': 8801441.971770002, 'tmax_k': 175611854.8365807}
RESULTS = {}


def record(name):
    def wrap(function):
        def run():
            started = time.perf_counter()
            try:
                details = function()
                passed = bool(details.pop('passed'))
                RESULTS[name] = {'passed': passed, 'seconds': round(time.perf_counter() - started, 3), **details}
            except Exception as error:  # noqa: BLE001
                RESULTS[name] = {'passed': False, 'error': repr(error), 'traceback': traceback.format_exc()[-3000:]}
            print(f"{'PASS' if RESULTS[name]['passed'] else 'FAIL'} {name}", flush=True)
        run.__name__ = function.__name__
        return run
    return wrap


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# --------------------------------------------------------------------------- synthetic halo
def synthetic(seed=1, a=1.0, n=60000, box=100.0):
    """Gas around a central near the periodic boundary, with a known overlapper/membership layout."""
    rng = np.random.default_rng(seed)
    r200, r500 = 1.0, 0.7                       # physical Mpc; r_ext = max(4, 4.2) = 4.2
    centre = np.array([99.4, 50.0, 0.3])        # comoving; sphere crosses the x and z boundaries
    # layout (physical offsets from the central)
    A = {'hid': 2001, 'offset': np.array([0.9, 0., 0.]), 'R': 0.5, 'mu': 0.15, 'own': True, 'central': False}
    B = {'hid': 2002, 'offset': np.array([0., 1.5, 0.]), 'R': 0.2, 'mu': 0.01, 'own': True, 'central': False}
    N = {'hid': 3001, 'offset': np.array([0., 0., 2.3]), 'R': 0.74, 'mu': 0.4, 'own': False, 'central': True}
    N1 = {'hid': 3002, 'offset': np.array([0., 0.4, 2.6]), 'R': 0.2, 'mu': 0.02, 'own': False, 'central': False}
    S = {'hid': 4001, 'offset': np.array([-2.5, 0., 0.]), 'R': 0.32, 'mu': 0.05, 'own': False, 'central': False}
    Pn = {'hid': 5001, 'offset': np.array([0., -6.0, 0.]), 'R': 1.3}
    rel = rng.uniform(-4.6, 4.6, size=(n, 3))
    clumps = []
    for obj, k in ((A, 3000), (B, 800), (N, 4000), (N1, 600), (S, 900)):
        clumps.append(obj['offset'] + rng.normal(scale=obj['R'] / 2, size=(k, 3)))
    rel = np.concatenate([rel, rng.normal(scale=.3, size=(6000, 3))] + clumps)
    n = len(rel)
    pos = (centre + rel / a) % box
    member = np.full(n, -1, np.int64)
    rng_m = rng.uniform(size=n)
    dist = lambda o: np.linalg.norm(rel - o, axis=1)  # noqa: E731
    member[(dist(np.zeros(3)) < 0.8) & (rng_m < .7)] = 1000
    for obj in (A, B, N, N1, S):
        member[(dist(obj['offset']) < obj['R']) & (rng_m < .8)] = obj['hid']
    member[(np.abs(rel[:, 0] - 3.0) < .3) & (rng_m < .5)] = 5001
    member[(np.abs(rel[:, 1] + 3.0) < .3) & (rng_m < .5)] = 6001
    member[(np.abs(rel[:, 2] + 3.5) < .2) & (rng_m < .5)] = 7001      # not in the painted lookup -> unknown
    sfr = np.where(rng.uniform(size=n) < .05, rng.uniform(1, 10, n), np.where(rng.uniform(size=n) < .1, -rng.uniform(.1, 1, n), 0.))
    sf = sfr > 0
    temperature = 10 ** rng.uniform(3.5, 8.5, n)
    last = np.where(rng.uniform(size=n) < .3, rng.uniform(.995, 1.0, n), -1.)
    data = {'Coordinates': pos, 'SmoothingLengths': rng.uniform(.005, .5, n), 'Masses': rng.uniform(.5, 1.5, n) * 1e9,
            'Densities': 10 ** rng.uniform(11, 15, n), 'ComptonYParameters': np.where(sf, 0., 10 ** rng.uniform(-14, -10, n)),
            'ElectronNumberDensities': np.where(sf, 0., 10 ** rng.uniform(60, 66, n)), 'Temperatures': temperature,
            'LastAGNFeedbackScaleFactors': last, 'ParticleIDs': rng.choice(2 ** 62, n, replace=False).astype(np.uint64),
            'Hydrogen': rng.uniform(.7, .76, n), 'Helium': rng.uniform(.24, .28, n), 'Pressures': 10 ** rng.uniform(-16, -11, n),
            'StarFormationRates': sfr, 'XrayRosat': np.where(sf, 0., 10 ** rng.uniform(30, 38, n))}
    painted = np.zeros(7000, np.uint8)
    painted[5001] = 1
    painted[3001] = painted[3002] = 1          # the neighbour group is painted
    # owner table per the rules (central 0; own satellites: class or 13; neighbour overlappers; group members)
    def code(obj):
        reach1 = np.linalg.norm(obj['offset']) < r200 + obj['R']
        return int(MO.owner_code(K.mu_class(obj['mu']), 0 if reach1 else 1, 0 if obj['own'] else 1))
    ov = [A, N, S]
    table = {1000: (0, -1, True), 2001: (code(A), 0, True), 2002: (MO.OTHER_OWN, -1, True),
             3001: (code(N), 1, False), 3002: (code(N), 1, False), 4001: (code(S), 2, False)}
    keys = np.array(sorted(table))
    halo = {'centre_cmpc': centre, 'radius': r200, 'r500': r500, 'r_ext': max(4 * r200, 6 * r500), 'a': a, 'box': box,
            'id': 1000, 'satellites': np.array([2001, 2002]), 'owner_hid': keys,
            'owner_code': np.array([table[k][0] for k in keys]), 'owner_group': np.array([table[k][1] for k in keys]),
            'owner_own': np.array([table[k][2] for k in keys]), 'ov_hid': np.array([o['hid'] for o in ov]),
            'excision': [(np.array([A['offset'], N['offset']]), np.array([A['R'], N['R']])),
                         (np.array([N['offset']]), np.array([N['R']])),
                         (np.array([[0., 1.2, 1.2]]), np.array([.6])), (np.empty((0, 3)), np.empty(0))],
            'pn_offset': np.array([N['offset'], Pn['offset']]), 'pn_r200': np.array([N['R'], Pn['R']])}
    ctx = {'heating': HEATING, 'painted': painted, 't_xray_min': 1e5}
    return data, member, halo, ctx, {'A': A, 'B': B, 'N': N, 'N1': N1, 'S': S}


def select(data, halo):
    sel = E.select_sphere(data['Coordinates'], halo['centre_cmpc'], halo['r_ext'] / halo['a'], halo['box'])
    return sel, {k: v[sel] for k, v in data.items()}


def brute(data, member, halo, ctx):
    """Per-particle reference classification by explicit rules (loops, no shared code paths)."""
    n = len(member)
    a, box = halo['a'], halo['box']
    delta = data['Coordinates'] - halo['centre_cmpc']
    delta -= box * np.rint(delta / box)
    in4 = np.array([d @ d <= (4 * halo['radius'] / a) ** 2 for d in delta])
    rp = delta * a
    x200 = np.sqrt(np.sum((delta * a / halo['radius']) ** 2, axis=1))
    x500 = np.sqrt(np.sum((rp / halo['r500']) ** 2, axis=1))
    table = dict(zip(halo['owner_hid'].tolist(), zip(halo['owner_code'].tolist(), halo['owner_group'].tolist(),
                                                     halo['owner_own'].tolist())))
    owner, group, own, painted_other = (np.zeros(n, int), np.full(n, -1), np.zeros(n, bool), np.zeros(n, bool))
    for i, m in enumerate(member):
        if m < 0:
            owner[i] = MO.UNBOUND
        elif m in table:
            owner[i], group[i], own[i] = table[m]
        else:
            flag = ctx['painted'][m] if m < len(ctx['painted']) else 255
            owner[i] = MO.OTHER_PAINTED if flag == 1 else MO.OTHER_NOT_PAINTED
            painted_other[i] = flag == 1
    T = data['Temperatures']
    agn = (data['LastAGNFeedbackScaleFactors'] >= HEATING['a_limit']) & (T >= HEATING['tmin_k']) & (T <= HEATING['tmax_k'])
    phase = np.array([0 if g else (1 if (s <= 0 and t >= 1e5) else 2) for g, s, t in zip(agn, data['StarFormationRates'], T)])
    zones = []
    for centres, radii in halo['excision']:
        z = np.full(n, 3)
        for i in range(n):
            s = min([np.linalg.norm(rp[i] - c) / r for c, r in zip(centres, radii)], default=np.inf)
            z[i] = 0 if s < .5 else 1 if s < 1 else 2 if s < 2 else 3
        zones.append(z)
    nbr = np.array([min([np.linalg.norm(rp[i] - c) / r for c, r in zip(halo['pn_offset'], halo['pn_r200'])], default=np.inf) < x200[i]
                    for i in range(n)])
    return {'in4': in4, 'x200': x200, 'x500': x500, 'owner': owner, 'group': group, 'own': own, 'agn': agn, 'phase': phase,
            'zones': zones, 'nbr': nbr, 'painted_other': painted_other, 'rp': rp, 'relative': delta * a / halo['radius']}


def weights_of(data):
    m, rho = data['Masses'], data['Densities']
    V = m / rho
    y, T = data['ComptonYParameters'], data['Temperatures']
    ne = data['ElectronNumberDensities'] * V
    pe = y / V * MO.PE_FACTOR
    lx = data['XrayRosat']
    return {'count': np.ones(len(m)), 'mass': m, 'volume': V, 'y': y, 'electrons': ne, 'electron_kT': ne * P.Boltzmann * T,
            'gas_pressure_volume': data['Pressures'] * V, 'mass_pe': m * pe, 'volume_pe2': V * pe * pe, 'mass_t': m * T,
            'volume_t': V * T, 'mass_rho': m * rho, 'lx_rosat': lx, 'lx_pe': lx * pe, 'lx_t': lx * T, 'y2': y * y}


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.maximum(np.abs(b).max() if b.size else 0, 1e-300)
    return float(np.max(np.abs(a - b)) / scale) if a.size else 0.0


# --------------------------------------------------------------------------- tests
@record('G2_grids_exact_subset_and_rebinning')
def test_grids():
    report = G.g2_report()
    values = np.random.default_rng(3).uniform(size=(5, G.N200))
    old = G.to_old(values)
    lee = G.to_lee22(values)
    b24 = G.to_b24(np.random.default_rng(4).uniform(size=(2, G.N500)))
    shell_vol = np.diff(G.X200 ** 3)
    ok_vol = np.allclose(G.to_old(shell_vol), np.diff(P.EDGES ** 3), rtol=1e-13)
    try:
        G.subgrid_index(G.X200, [0, 0.0333, 4])
        refused = False
    except ValueError:
        refused = True
    passed = (np.array_equal(G.X200[G.OLD_INDEX], P.EDGES) and old.shape == (5, 36) and lee.shape == (5, 27)
              and b24.shape == (2, 30) and np.allclose(old.sum(axis=1), values.sum(axis=1), rtol=1e-13) and ok_vol and refused
              and G.N200 == 67 and G.N500 == 40)
    return {'passed': passed, **{k: report[k] for k in ('X200_edges', 'X500_edges', 'thinnest_X200_log10_width')},
            'non_subset_refused': refused}


@record('legacy_sums_bit_identical_to_measure_accumulate')
def test_legacy():
    results = []
    for seed, a in ((1, 1.0), (2, 0.9)):
        data, member, halo, ctx, _ = synthetic(seed, a)
        sel, d = select(data, halo)
        item = MO.accumulate(d, member[sel], halo, ctx)
        in4 = item['in4']
        sub = {k: v[in4] for k, v in d.items() if k not in ('StarFormationRates', 'XrayRosat')}
        new = MO.legacy_sums(sub, member[sel][in4], halo, HEATING)
        ref = M.accumulate(sub, member[sel][in4], {'centre_cmpc': halo['centre_cmpc'], 'radius': halo['radius'], 'mass': 1e14,
                           'id': halo['id'], 'satellites': halo['satellites'], 'a': a, 'box': halo['box']}, .158, HEATING,
                           np.eye(3)[None].repeat(2, axis=0))['sums']
        # the old full-cell 4R mask equals the new pre-filtered selection's legacy subset
        delta = data['Coordinates'] - halo['centre_cmpc']
        delta -= halo['box'] * np.rint(delta / halo['box'])
        full = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= (4 * halo['radius'] / a) ** 2)
        empty = MO.legacy_sums({k: v[:0] for k, v in sub.items()}, member[:0], halo, HEATING)
        results.append({'a': a, 'bit_identical': bool(np.array_equal(new, ref)), 'mask_equal': bool(np.array_equal(full, sel[in4])),
                        'n_4R': int(in4.sum()), 'empty_dtype': str(empty.dtype)})
    passed = all(r['bit_identical'] and r['mask_equal'] and r['empty_dtype'] == 'float64' for r in results)
    return {'passed': passed, 'cases': results}


@record('blocks_sum_to_legacy_totals_1e-12')
def test_block_totals():
    data, member, halo, ctx, _ = synthetic(5)
    sel, d = select(data, halo)
    item = MO.accumulate(d, member[sel], halo, ctx)
    in4 = item['in4']
    sub = {k: v[in4] for k, v in d.items()}
    legacy = MO.legacy_sums(sub, member[sel][in4], halo, HEATING)
    total = legacy.sum(axis=0)
    errors = {}
    for q in ('count', 'mass', 'y', 'electrons', 'electron_kT', 'volume', 'gas_pressure_volume'):
        errors[q] = rel_err(G.to_old(item['owner200'][:, :, MO.QUANTITIES.index(q)].sum(axis=(0, 1))), total[P.SUM_NAMES.index(q)])
    errors['y_noagn_from_phases'] = rel_err(G.to_old(item['owner200'][:, 1:, MO.QUANTITIES.index('y')].sum(axis=(0, 1))),
                                            total[P.SUM_NAMES.index('y_noagn')])
    comps = {0: [0], 1: [c for c in range(1, 13) if (c - 1) % 2 == 0] + [13], 2: [c for c in range(1, 13) if (c - 1) % 2 == 1] + [14, 15], 3: [16]}
    for comp, codes in comps.items():
        for q in ('count', 'mass', 'y'):
            errors[f'component{comp}_{q}'] = rel_err(G.to_old(item['owner200'][codes][:, :, MO.QUANTITIES.index(q)].sum(axis=(0, 1))),
                                                     legacy[comp, P.SUM_NAMES.index(q)])
    reduced = np.stack([item['owner200'][:, :, MO.QUANTITIES.index(q)].sum(axis=(0, 1)) for q in MO.REDUCED])
    for v in range(4):
        errors[f'excision_{MO.VARIANTS[v]}'] = rel_err(item['excision'][v].sum(axis=(0, 1, 2)), reduced)
    errors['partition'] = rel_err(item['partition'].sum(axis=(0, 1, 2)), reduced)
    for i, name in enumerate(MO.AUX):
        errors[f'aux_{name}'] = rel_err(G.to_old(item['aux200'][:, i].sum(axis=0)), total[P.SUM_NAMES.index(name)])
    count_exact = bool(np.array_equal(G.to_old(item['owner200'][:, :, 0].sum(axis=(0, 1))), total[0]))
    passed = max(errors.values()) <= 1e-12 and count_exact
    return {'passed': passed, 'max_relative_error': max(errors.values()), 'errors': errors, 'count_exact': count_exact,
            'components_populated': [int(legacy[c, 0].sum()) for c in range(4)]}


@record('owner_phase_shell_blocks_vs_brute_force')
def test_owner_brute():
    data, member, halo, ctx, layout = synthetic(7)
    sel, d = select(data, halo)
    mem = member[sel]
    item = MO.accumulate(d, mem, halo, ctx)
    b = brute(d, mem, halo, ctx)
    w = weights_of(d)
    exp200 = np.zeros((17, 3, len(MO.QUANTITIES), G.N200))
    exp500 = np.zeros((17, 3, len(MO.QUANTITIES), G.N500))
    s200 = np.array([np.searchsorted(G.X200, x, 'right') - 1 for x in b['x200']]).clip(0, G.N200 - 1)
    s500 = np.array([np.searchsorted(G.X500, x, 'right') - 1 for x in b['x500']]).clip(0, G.N500 - 1)
    for qi, q in enumerate(MO.QUANTITIES):
        np.add.at(exp200[:, :, qi], (b['owner'][b['in4']], b['phase'][b['in4']], s200[b['in4']]), w[q][b['in4']])
        in6 = b['x500'] <= 6
        np.add.at(exp500[:, :, qi], (b['owner'][in6], b['phase'][in6], s500[in6]), w[q][in6])
    owners_present = sorted(set(b['owner'][b['in4']].tolist()))
    passed = (rel_err(item['owner200'], exp200) < 1e-12 and rel_err(item['owner500'], exp500) < 1e-12
              and np.array_equal(item['owner200'][:, :, 0], exp200[:, :, 0]))
    return {'passed': passed, 'owner200_rel_err': rel_err(item['owner200'], exp200), 'owner500_rel_err': rel_err(item['owner500'], exp500),
            'owner_codes_present': [MO.OWNERS[c] for c in owners_present],
            'unknown_membership_counted': int(item['diagnostics'][0]),
            'unknown_expected': int(np.sum((mem >= 7000))), 'phases_present': np.bincount(b['phase'], minlength=3).tolist()}


@record('excision_partition_overlapper_inertia_apertures_vs_brute_force')
def test_other_blocks():
    data, member, halo, ctx, _ = synthetic(11, n=20000)
    sel, d = select(data, halo)
    mem = member[sel]
    item = MO.accumulate(d, mem, halo, ctx)
    b = brute(d, mem, halo, ctx)
    w = weights_of(d)
    in4 = b['in4']
    s200 = np.array([np.searchsorted(G.X200, x, 'right') - 1 for x in b['x200']]).clip(0, G.N200 - 1)
    member_class = np.where(b['owner'] == 0, 0, np.where(b['owner'] <= 12, 1, 2))
    exc = np.zeros((4, 4, 3, 2, 5, G.N200))
    part = np.zeros((2, 2, 2, 5, G.N200))
    for qi, q in enumerate(MO.REDUCED):
        for v in range(4):
            np.add.at(exc[v, :, :, :, qi], (b['zones'][v][in4], member_class[in4], b['agn'][in4].astype(int), s200[in4]), w[q][in4])
        np.add.at(part[:, :, :, qi], (b['nbr'][in4].astype(int), b['painted_other'][in4].astype(int), b['agn'][in4].astype(int), s200[in4]), w[q][in4])
    # per-overlapper (own bound / group bound), x200 <= 2 and whole sphere
    ov = np.zeros((len(halo['ov_hid']), 2, 2, 6))
    yna = np.where(b['agn'], 0, w['y'])
    qs = (w['count'], w['mass'], w['y'], yna, w['electrons'], w['electron_kT'])
    for k, hid in enumerate(halo['ov_hid']):
        for t, mask in enumerate((mem == hid, b['group'] == k)):
            for j, q in enumerate(qs):
                ov[k, t, 0, j] = q[mask & in4 & (b['x200'] <= 2)].sum()
                ov[k, t, 1, j] = q[mask].sum()
    inner = in4 & (b['x200'] <= 1)
    inertia = []
    for mask in (inner, inner & (b['owner'] == 0)):
        r, m = b['relative'][mask], d['Masses'][mask]
        inertia.append([m.sum()] + [np.sum(m * r[:, i]) for i in range(3)] + [np.sum(m * r[:, i] * r[:, j]) for i, j in
                                                                             ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
    lxna = np.where(b['agn'], 0, w['lx_rosat'])
    ap = []
    for mask in (in4 & (b['x200'] <= 1), b['x500'] <= 1, b['x500'] <= 5, np.ones(len(mem), bool)):
        ap.append([[q[mm].sum() for q in (w['count'], w['mass'], w['y'], yna, w['lx_rosat'], lxna, w['electrons'])]
                   for mm in (mask, mask & (b['owner'] == 0))])
    errors = {'excision': rel_err(item['excision'], exc), 'partition': rel_err(item['partition'], part),
              'overlapper': rel_err(item['overlapper'], ov), 'inertia': rel_err(item['inertia'], np.array(inertia)),
              'apertures': rel_err(item['apertures'], np.array(ap))}
    zone_counts = [np.bincount(z[in4], minlength=4).tolist() for z in b['zones']]
    passed = max(errors.values()) < 1e-11 and all(np.count_nonzero(np.array(z)[:3]) > 0 for z in zone_counts[:3])
    return {'passed': passed, 'relative_errors': errors, 'zone_counts_per_variant': zone_counts,
            'partition_neighbour_owned': int(b['nbr'][in4].sum()), 'painted_other_bound': int(b['painted_other'][in4].sum())}


@record('cell_split_merge_and_empty_then_populated')
def test_merge():
    data, member, halo, ctx, _ = synthetic(13)
    sel, d = select(data, halo)
    mem = member[sel]
    whole = MO.accumulate(d, mem, halo, ctx)
    whole.pop('in4')
    rng = np.random.default_rng(1)
    first = np.sort(rng.choice(len(mem), len(mem) // 3, replace=False))
    second = np.setdiff1d(np.arange(len(mem)), first)
    total = MO.empty_blocks(len(halo['ov_hid']))
    empty = MO.accumulate({k: v[:0] for k, v in d.items()}, mem[:0], halo, ctx)
    empty.pop('in4')
    dtypes_ok = all(empty[k].dtype == total[k].dtype for k in MO.MERGE_KEYS)
    for part in (first, second):
        item = MO.accumulate({k: v[part] for k, v in d.items()}, mem[part], halo, ctx)
        item.pop('in4')
        MO.add_into(total, item)
    exact_int = all(np.array_equal(total[k], whole[k]) for k in ('hs200', 'hs500', 'id_hash', 'id_count'))
    floats = {k: rel_err(total[k], whole[k]) for k in ('owner200', 'owner500', 'excision', 'partition', 'aux200', 'overlapper', 'apertures')}
    base = MO.empty_blocks(len(halo['ov_hid']))
    MO.add_into(base, empty)
    MO.add_into(base, whole)
    empty_then_populated = all(np.array_equal(base[k], whole[k]) for k in MO.MERGE_KEYS)
    hashed = MO.splitmix64(d['ParticleIDs'])
    passed = exact_int and max(floats.values()) < 1e-12 and empty_then_populated and dtypes_ok and len(np.unique(hashed)) == len(hashed)
    return {'passed': passed, 'integer_blocks_exact': exact_int, 'float_relative_errors': floats,
            'empty_then_populated_exact': empty_then_populated, 'empty_dtypes_ok': dtypes_ok}


@record('select_sphere_periodic_vs_brute_force')
def test_select():
    rng = np.random.default_rng(17)
    box = 1000.0000004828763
    out = []
    for centre in (np.array([999.2, 0.4, 500.]), np.array([0.1, 999.9, 0.05]), np.array([500., 500., 500.])):
        pos = (centre + rng.uniform(-6, 6, size=(300000, 3))) % box
        for radius in (0.37, 2.0, 5.5):
            fast = E.select_sphere(pos, centre, radius, box)
            delta = pos - centre
            delta -= box * np.rint(delta / box)
            ref = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= radius ** 2)
            out.append(bool(np.array_equal(fast, ref)) and len(ref) > 0)
    return {'passed': all(out), 'cases': len(out)}


@record('owner_tables_from_build_tables_vs_rules')
def test_build_tables():
    # a synthetic census for two haloes: halo 0 has an own-satellite overlapper, a neighbour central
    # overlapper with two satellites (one of them also an overlapper itself) and a neighbour-satellite
    # overlapper of a non-overlapping host; halo 1 is clean (gets a placebo partner from halo 0's bin).
    halos = {'halo_index': np.array([10, 20]), 'mass_bin': np.array([3, 3]), 'r_200_crit_mpc': np.array([1.0, 0.5])}
    ov = {'hid': [101, 300, 302, 400], 'soap_row': [5, 30, 32, 40], 'is_central': [False, True, False, False],
          'host_row': [1, -1, 30, 39], 'host_hid': [10, -1, 300, 399], 'own_satellite': [True, False, False, False],
          'mass_msun': [2e13, 5e13, 4e12, 1e13], 'mu': [0.2, 0.5, 0.04, 0.1], 'radius_mpc': [0.6, 0.8, 0.3, 0.4],
          'offset_mpc': [[0.9, 0, 0], [0, 2.4, 0], [0, 2.2, 0.3], [-2.2, 0, 0]], 'distance_mpc': [0.9, 2.4, 2.22, 2.2],
          'reach1': [True, False, False, False], 'bound_gas_count': [10, 20, 3, 5], 'bound_gas_mass_msun': [1., 2., 3., 4.]}
    census = {'ov_' + k: np.asarray(v) for k, v in ov.items()}
    census['ov_mu_class'] = K.mu_class(census['ov_mu'])
    census.update(overlap_ptr=np.array([0, 4, 4]), satellite_ptr=np.array([0, 2, 2]), satellite_hid=np.array([101, 102]),
                  partition_ptr=np.array([0, 1, 1]), pn_soap_row=np.array([30]), pn_hid=np.array([300]),
                  pn_m200c_msun=np.array([5e13]), pn_r200c_mpc=np.array([.8]), pn_offset_mpc=np.array([[0, 2.4, 0]]),
                  pn_distance_mpc=np.array([2.4]), nearest_massive_offset_mpc=np.array([[0, 2.4, 0], [3., 0, 0]]))
    groups = {'row': np.array([30]), 'ptr': np.array([0, 2]), 'member_hid': np.array([301, 302])}
    tables = PS.build_tables(halos, census, np.array([0, 1]), groups)
    t = {int(k): (int(c), int(g), bool(o)) for k, c, g, o in zip(tables['owner_hid'][:tables['owner_ptr'][1]],
                                                                 tables['owner_code'][:tables['owner_ptr'][1]],
                                                                 tables['owner_group'][:tables['owner_ptr'][1]],
                                                                 tables['owner_own'][:tables['owner_ptr'][1]])}
    c = lambda mu, reach1, own: int(MO.owner_code(K.mu_class(mu), 0 if reach1 else 1, 0 if own else 1))  # noqa: E731
    expected = {10: (0, -1, True), 101: (c(.2, True, True), 0, True), 102: (13, -1, True),
                300: (c(.5, False, False), 1, False), 301: (c(.5, False, False), 1, False),
                302: (c(.5, False, False), 1, False),          # satellite of an overlapping central: counts as that central
                400: (c(.1, False, False), 3, False)}
    exc = tables['exc_ptr']
    n_all, n_nbr = exc[1] - exc[0], exc[2] - exc[1]
    placebo_ok = tables['placebo_partner'].tolist() == [-1, 0] and exc[4 + 3] - exc[4 + 2] == n_all
    scale_ok = np.isclose(tables['placebo_scale'][1], 0.5)
    vf = tables['excision_volume_fraction']
    vf_ok = bool(vf[0, 0, 2].max() > 0 and vf[1, 2, 1].max() > 0 and np.all(vf >= 0) and np.all(vf <= 1 + 1e-9) and vf[1, 0].max() == 0)
    rotated = tables['exc_offset_mpc'][exc[4 + 3]:exc[4 + 4]]
    # placebo_nbr: the partner's dominant overlapper (mu 0.5 at +y) is rotated onto halo 1's neighbour direction (+x)
    dominant = rotated[np.argmax(census['ov_mu'][census['ov_mu'] >= .1])] if len(rotated) else np.zeros(3)
    nbr_ok = bool(len(rotated) and np.allclose(dominant / np.linalg.norm(dominant), [1, 0, 0], atol=1e-9))
    passed = t == expected and n_all == 3 and n_nbr == 2 and placebo_ok and scale_ok and vf_ok and nbr_ok
    return {'passed': passed, 'table': {str(k): v for k, v in t.items()}, 'expected': {str(k): v for k, v in expected.items()},
            'excision_spheres_all': int(n_all), 'excision_spheres_neighbours_only': int(n_nbr), 'placebo_ok': bool(placebo_ok),
            'volume_fraction_ok': vf_ok, 'placebo_nbr_direction_ok': nbr_ok}


@record('census_tree_search_vs_brute_force')
def test_census():
    rng = np.random.default_rng(23)
    box = 150.
    n = 30000
    mass = 10 ** rng.uniform(11, 15.2, n)
    rho_c = 1.27e11
    radius = K.pseudo_r200c(mass, rho_c)
    pos = K.wrap(rng.uniform(0, box, (n, 3)), box)
    central = rng.uniform(size=n) < .6
    row = np.arange(n)
    src = np.flatnonzero(central & (mass >= 10 ** 12.5))
    src = rng.choice(src, 400, replace=False)
    pop = {'pos': pos, 'radius': radius, 'mass': mass, 'row': row}
    i, j, delta, d = K.overlapper_pairs(pos[src], radius[src], mass[src], src, pop, box, f_max=2)
    tree = {(int(a), int(b)) for a, b in zip(i, j)}
    bf = K.brute_force_pairs(pos[src], 2 * radius[src], pos, radius, box, src_mass=mass[src], tgt_mass=mass, mu_min=.03,
                             src_id=src, tgt_id=row)
    delta_ok = np.allclose(delta, K.periodic_delta(pos[j], pos[src][i], box))
    # symmetric counts vs brute force on the 'central' subset
    cen = np.flatnonzero(central)[:2500]
    counts, _ = K.symmetric_pair_counts(pos[cen], radius[cen], box)
    dd = np.linalg.norm(K.periodic_delta(pos[cen][:, None], pos[cen][None], box), axis=-1)
    q = dd / (radius[cen][:, None] + radius[cen][None])
    iu = np.triu_indices(len(cen), 1)
    bf_counts = {f: int(np.sum(q[iu] < f)) for f in (1, 2, 3, 4)}
    # max-mu summary vs brute force
    own = rng.uniform(size=len(i)) < .3
    reach1 = d < radius[src][i] + radius[j]
    mu = mass[j] / mass[src][i]
    summary = K.max_mu_summary(len(src), i, mu, reach1, own)
    ref = np.zeros((len(src), 2, 2))
    for a_, mu_, r1, o in zip(i, mu, reach1, own):
        ref[a_, 1, 0 if o else 1] = max(ref[a_, 1, 0 if o else 1], mu_)
        if r1:
            ref[a_, 0, 0 if o else 1] = max(ref[a_, 0, 0 if o else 1], mu_)
    # enlargement plan
    pool_bin = rng.integers(0, 3, 5000)
    pool_class = rng.choice(3, 5000, p=[.7, .2, .1])
    selected = np.zeros(5000, bool)
    selected[rng.choice(5000, 600, replace=False)] = True
    added, prob = K.enlargement_plan(pool_bin, pool_class, selected, 3, [0, 1, 2], 50, 7)
    chosen = selected.copy()
    chosen[added] = True
    enough = all(chosen[(pool_bin == b) & (pool_class == k)].sum() >= min(50, ((pool_bin == b) & (pool_class == k)).sum())
                 for b in range(3) for k in range(3))
    prob_ok = all(np.allclose(prob[(pool_bin == b) & (pool_class == k)], chosen[(pool_bin == b) & (pool_class == k)].mean())
                  for b in range(3) for k in range(3))
    passed = (tree == bf and delta_ok and counts == bf_counts and np.allclose(summary, ref.astype(np.float32))
              and enough and prob_ok and not np.any(selected[added]))
    return {'passed': passed, 'pairs_tree': len(tree), 'pairs_brute': len(bf), 'symmetric_counts': counts,
            'symmetric_brute': bf_counts, 'enlargement_added': int(len(added)), 'enlargement_targets_met': enough}


@record('geometry_vs_monte_carlo')
def test_geometry():
    out = {}
    lens = GEO.lens_volume(np.array([1., 1., 1., 1.]), np.array([1., 1., .5, .3]), np.array([0., 2., .2, 1.]))
    out['lens_analytic'] = bool(np.allclose(lens, [4 * np.pi / 3, 0, 4 * np.pi / 3 * .125, lens[3]]) and 0 < lens[3] < 4 * np.pi / 3 * .027)
    edges = np.r_[0, np.geomspace(.1, 4, 12)]
    single = GEO.union_shell_fraction(edges, [[1.5, 0, 0]], [.6])
    mc_single = GEO.mc_union_shell_fraction(edges, [[1.5, 0, 0]], [.6], n=300000)
    out['single_ball_max_abs'] = float(np.max(np.abs(single - mc_single)))
    centres = np.array([[1.0, 0, 0], [1.4, .4, 0], [0, 0, 2.5], [-.2, 0, 0]])
    radii = np.array([.6, .5, .9, .35])
    union = GEO.union_shell_fraction(edges, centres, radii)
    mc_union = GEO.mc_union_shell_fraction(edges, centres, radii, n=300000)
    out['union_four_balls_max_abs'] = float(np.max(np.abs(union - mc_union)))
    # offset template: B12-like profile centred at d = 0.8, truncated at cut = 1.5
    xc, beta = .5, 4.5
    F = lambda s: (np.maximum(s, 1e-12) / xc) ** -.3 * (1 + s / xc) ** -beta  # noqa: E731
    Gf = GEO.make_G(F, 1.5)
    s = np.array([1e-3, .1, .5, 1.4])
    Gref = []
    from scipy.integrate import quad
    for v in s:
        Gref.append(quad(lambda u: u * F(u), 0, v, limit=200, points=[1e-6, 1e-4, 1e-2])[0])
    out['G_max_rel'] = float(np.max(np.abs(Gf(s) / np.array(Gref) - 1)))
    template = GEO.offset_shell_average(Gf, edges, .8, 1.5)
    out['offset_template_quadrature_converged'] = float(np.max(np.abs(template - GEO.offset_shell_average(Gf, edges, .8, 1.5, n_gl=64))
                                                               / np.maximum(template, 1e-300)))
    runs = np.array([GEO.mc_offset_shell_average(F, edges, .8, 1.5, n=500000, seed=s) for s in range(4)])
    mc_template, mc_error = runs.mean(axis=0), runs.std(axis=0, ddof=1) / 2
    use = mc_template > 0
    out['offset_template_max_rel'] = float(np.max(np.abs(template[use] / mc_template[use] - 1)))
    out['offset_template_max_sigma'] = float(np.max(np.abs(template[use] - mc_template[use]) / np.maximum(mc_error[use], 1e-4 * mc_template[use])))
    out['offset_template_zero_beyond'] = bool(np.all(template[edges[:-1] >= 0.8 + 1.5] == 0))
    passed = (out['lens_analytic'] and out['single_ball_max_abs'] < 3e-3 and out['union_four_balls_max_abs'] < 4e-3
              and out['G_max_rel'] < 1e-4 and out['offset_template_quadrature_converged'] < 1e-5
              and out['offset_template_max_sigma'] < 5 and out['offset_template_zero_beyond'])
    return {'passed': passed, **out}


@record('frozen_sources_G1')
def test_frozen():
    import hashlib
    radii234 = STUDY.parent / 'radii234_contamination_L1_m9_0077_20261008'
    executed = json.loads((radii234 / 'executed_sources.json').read_text())
    final = json.loads((radii234 / 'final_source_manifest.json').read_text())
    names = ['pilot_physics.py', 'measure.py', 'models.py', 'projection_model.py', 'swift_reader.py', 'check_fresh_mask.py',
             'remote_control.py', 'dependencies/common.py', 'dependencies/flamingo_io.py', 'dependencies/catalogue_stage.py']
    hashes = {n: hashlib.sha256((STUDY / n).read_bytes()).hexdigest() for n in names}
    ok = {n: hashes[n] == executed[n] == final[n] for n in names}
    return {'passed': all(ok.values()), 'equal': ok}


def main():
    started = time.perf_counter()
    for test in (test_grids, test_frozen, test_legacy, test_block_totals, test_owner_brute, test_other_blocks, test_merge,
                 test_select, test_build_tables, test_census, test_geometry):
        test()
    summary = {'all_passed': all(r['passed'] for r in RESULTS.values()), 'n_tests': len(RESULTS),
               'n_passed': sum(r['passed'] for r in RESULTS.values()), 'wall_seconds': time.perf_counter() - started,
               'numpy': np.__version__, 'tests': RESULTS}
    (HERE / 'test_results.json').write_text(json.dumps(jsonable(summary), indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('all_passed', 'n_tests', 'n_passed', 'wall_seconds')}))


if __name__ == '__main__':
    main()
