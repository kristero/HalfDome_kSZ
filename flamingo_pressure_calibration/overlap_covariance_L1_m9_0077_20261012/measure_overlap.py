"""Per-halo accumulators of the overlap/covariance particle pass (plan Step 1.2).

accumulate() takes the gas of ONE (halo, cell) pair, already clipped to the halo's extraction sphere
r <= r_ext = max(4 R200c, 6 R500c) by the exact periodic test, in the radii234 units (Coordinates and
SmoothingLengths comoving Mpc; Masses Msun; Densities Msun/Mpc^3; ComptonYParameters Mpc^2;
ElectronNumberDensities Mpc^-3; Temperatures K; Pressures Pa) plus StarFormationRates (only the sign
is used) and XrayRosat [erg/s, XrayLuminosities column 2]. Every block is a linear sum, so blocks merge
by addition across cells and files.

legacy_sums() is the sums path of the frozen measure.accumulate, verbatim: the caller applies it to the
radii234 legacy subset (the same (halo, cell) pairs and the same exact 4 R200c test) and the result is
bit-identical to radii234 (gates G5, G7).

Blocks returned by accumulate() (shapes per halo; N200 = 67, N500 = 40, NH = 384):
  owner200 [17 OWNERS, 3 PHASES, 16 QUANTITIES, N200] float64  (x200 grid; same particle set as legacy)
  owner500 [17, 3, 16, N500] float64                           (x500 = r/R500c <= 6)
  excision [4 VARIANTS, 4 ZONES, 3 MEMBERS, 2 AGN, 5 REDUCED, N200] float64
  partition [2 PARTITION_OWNERS, 2 PAINTED_BOUND, 2 AGN, 5 REDUCED, N200] float64
  aux200 [2 AGN, 3 AUX, N200], aux500 [2, 3, N500] float64
  hs200 [N200, NH], hs500 [N500, NH] uint32       (h a / R200c and h a / R500c histograms, HS_EDGES)
  overlapper [n_ov, 2 (own bound, group bound), 2 (x200 <= 2, whole sphere), 6 OVERLAPPER_QUANTITIES]
  inertia [2 (all gas, central bound), 10 INERTIA] float64      (x200 <= 1, positions in R200c units)
  apertures [4 APERTURES, 2 (all gas, central bound), 7 APERTURE_QUANTITIES] float64
  id_hash [2 (4 R200c legacy rule, whole sphere), 2 (sum, xor)] uint64 of splitmix64(ParticleIDs)
  id_count [2] int64, diagnostics [len(DIAGNOSTICS)] int64
plus 'in4' (bool mask of the legacy 4 R200c rule; not stored).
"""
import numpy as np

import grids as G
import measure as M
import pilot_physics as P

NB = M.NB
QUANTITIES = ('count', 'mass', 'volume', 'y', 'electrons', 'electron_kT', 'gas_pressure_volume',
              'mass_pe', 'volume_pe2', 'mass_t', 'volume_t', 'mass_rho', 'lx_rosat', 'lx_pe', 'lx_t', 'y2')
UNITS = {'count': '1', 'mass': 'Msun', 'volume': 'Mpc^3 (physical, m/rho)', 'y': 'Mpc^2',
         'electrons': '1 (n_e V)', 'electron_kT': 'J (n_e V k T)', 'gas_pressure_volume': 'Pa Mpc^3',
         'mass_pe': 'Msun Pa', 'volume_pe2': 'Mpc^3 Pa^2', 'mass_t': 'Msun K', 'volume_t': 'Mpc^3 K',
         'mass_rho': 'Msun^2 Mpc^-3', 'lx_rosat': 'erg/s', 'lx_pe': 'erg/s Pa', 'lx_t': 'erg/s K', 'y2': 'Mpc^4'}
REDUCED = ('count', 'mass', 'volume', 'y', 'electrons')
REDUCED_INDEX = [QUANTITIES.index(q) for q in REDUCED]
MU_LABELS = ('mu0p03', 'mu0p1', 'mu0p3')


def owner_code(mu_class, reach_annulus_only, neighbour):
    """1..12: overlapper-bound gas of mu class (0, 1, 2), reach (0: inside 1R, 1: 1-2R only), (own sat, nbr)."""
    return 1 + (np.asarray(mu_class) * 2 + np.asarray(reach_annulus_only)) * 2 + np.asarray(neighbour)


OWNERS = ('central_bound',) + tuple(
    f'overlapper_{m}_{"reach1R" if r == 0 else "annulus"}_{"own_satellite" if t == 0 else "neighbour"}'
    for m in MU_LABELS for r in (0, 1) for t in (0, 1)) + (
    'other_own_satellites', 'other_bound_painted', 'other_bound_not_painted', 'unbound')
assert len(OWNERS) == 17 and all(OWNERS[owner_code(m, r, t)].startswith('overlapper_' + MU_LABELS[m])
                                 for m in range(3) for r in range(2) for t in range(2))
OTHER_OWN, OTHER_PAINTED, OTHER_NOT_PAINTED, UNBOUND = 13, 14, 15, 16
PHASES = ('recent_agn', 'xray_selected', 'other')   # recent AGN (SOAP filter); SFR <= 0 & T >= 1e5 K & not AGN; rest
VARIANTS = ('all', 'neighbours_only', 'placebo_iso', 'placebo_nbr')
ZONE_EDGES = np.array([.5, 1., 2.])
ZONES = ('s_lt_0p5', 's_0p5_1', 's_1_2', 's_ge_2')    # s = min_k |r - d_k| / R_k over the variant's spheres
MEMBERS = ('central_bound', 'overlapper_bound', 'rest')
AGN = ('not_recent_agn', 'recent_agn')
PARTITION_OWNERS = ('central', 'painted_neighbour')  # smallest r / R200c among the central and painted neighbours
PAINTED_BOUND = ('not_bound_to_other_painted', 'bound_to_other_painted')
AUX = ('hydrogen_mass', 'helium_mass', 'fully_ionized_electrons')
OVERLAPPER_QUANTITIES = ('count', 'mass', 'y', 'y_noagn', 'electrons', 'electron_kT')
INERTIA = ('m', 'mx', 'my', 'mz', 'mxx', 'myy', 'mzz', 'mxy', 'mxz', 'myz')
APERTURES = ('R200c', 'R500c', '5R500c', 'sphere')
APERTURE_QUANTITIES = ('count', 'mass', 'y', 'y_noagn', 'lx_rosat', 'lx_rosat_noagn', 'electrons')
DIAGNOSTICS = ('membership_not_in_soap', 'hs200_underflow', 'hs200_overflow', 'hs500_underflow', 'hs500_overflow',
               'n_in4', 'n_in6', 'n_sphere', 'star_forming_with_nonzero_y', 'star_forming')
HS_EDGES = np.geomspace(1e-4, 1e2, 385)
NH = len(HS_EDGES) - 1
PE_FACTOR = P.electron_mass * P.c ** 2 / P.SIGMA_T / P.MPC_M   # Pa per (Mpc^2 / Mpc^3) of y / V
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1, _MIX2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def splitmix64(ids):
    """64-bit finaliser of each ParticleID (uint64 arithmetic wraps); per-halo sum/xor are order independent."""
    z = np.asarray(ids, dtype=np.uint64) + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def legacy_sums(data, membership, halo, heating):
    """The sums path of measure.accumulate, verbatim (same expressions in the same order)."""
    delta=data['Coordinates']-halo['centre_cmpc']
    delta-=halo['box']*np.rint(delta/halo['box'])
    relative=delta*halo['a']/halo['radius']
    x=np.linalg.norm(relative,axis=1)
    assert np.all(x<=4.+1e-12)
    category=M.components(membership,halo['id'],halo['satellites'])
    shell=np.searchsorted(P.EDGES,x,side='right')-1
    shell=np.minimum(shell,NB-1)
    code=category*NB+shell
    volume=data['Masses']/data['Densities']
    ne=data['ElectronNumberDensities']*volume
    recently=(data['LastAGNFeedbackScaleFactors']>=heating['a_limit']) & (data['Temperatures']>=heating['tmin_k']) & (data['Temperatures']<=heating['tmax_k'])
    weights={
        'count':np.ones(len(x)),'mass':data['Masses'],'y':data['ComptonYParameters'],
        'y_noagn':np.where(recently,0.,data['ComptonYParameters']),
        'electrons':ne,'electron_kT':ne*P.Boltzmann*data['Temperatures'],
        'volume':volume,'gas_pressure_volume':data['Pressures']*volume,
        'hydrogen_mass':data['Masses']*data['Hydrogen'],
        'helium_mass':data['Masses']*data['Helium'],
        'fully_ionized_electrons':data['Masses']*P.MSUN_KG/P.proton_mass*(data['Hydrogen']+data['Helium']/2)}
    sums=np.stack([np.bincount(code,weights=weights[name],minlength=4*NB).reshape(4,NB) for name in P.SUM_NAMES],axis=1).astype(np.float64,copy=False)
    return sums


def _sums(code, n_codes, weights, select):
    """[len(weights), n_codes] float64 bincounts over the selected particles (float64 also when empty)."""
    c = code[select]
    return np.stack([np.bincount(c, weights=w[select], minlength=n_codes) for w in weights]).astype(np.float64, copy=False)


def empty_blocks(n_ov):
    return {'owner200': np.zeros((17, 3, len(QUANTITIES), G.N200)), 'owner500': np.zeros((17, 3, len(QUANTITIES), G.N500)),
            'excision': np.zeros((len(VARIANTS), len(ZONES), len(MEMBERS), 2, len(REDUCED), G.N200)),
            'partition': np.zeros((2, 2, 2, len(REDUCED), G.N200)),
            'aux200': np.zeros((2, 3, G.N200)), 'aux500': np.zeros((2, 3, G.N500)),
            'hs200': np.zeros((G.N200, NH), np.uint32), 'hs500': np.zeros((G.N500, NH), np.uint32),
            'overlapper': np.zeros((n_ov, 2, 2, len(OVERLAPPER_QUANTITIES))), 'inertia': np.zeros((2, len(INERTIA))),
            'apertures': np.zeros((len(APERTURES), 2, len(APERTURE_QUANTITIES))),
            'id_hash': np.zeros((2, 2), np.uint64), 'id_count': np.zeros(2, np.int64),
            'diagnostics': np.zeros(len(DIAGNOSTICS), np.int64)}


def owner_codes(membership, halo, painted):
    """Owner code, overlapper group index and own (central or own-satellite) flag of every particle."""
    n = len(membership)
    owner = np.full(n, UNBOUND, np.int64)
    group = np.full(n, -1, np.int64)
    own = np.zeros(n, bool)
    bound = membership >= 0
    table = halo['owner_hid']
    hit = np.zeros(n, bool)
    if len(table):
        pos = np.minimum(np.searchsorted(table, membership), len(table) - 1)
        hit = bound & (table[pos] == membership)
        owner[hit] = halo['owner_code'][pos[hit]]
        group[hit] = halo['owner_group'][pos[hit]]
        own[hit] = halo['owner_own'][pos[hit]]
    rest = bound & ~hit
    ids = membership[rest]
    known = ids < len(painted)
    flag = np.full(len(ids), 255, np.uint8)
    flag[known] = painted[ids[known]]
    owner[rest] = np.where(flag == 1, OTHER_PAINTED, OTHER_NOT_PAINTED)
    painted_other = np.zeros(n, bool)
    painted_other[rest] = flag == 1
    return owner, group, own, painted_other, int(np.sum(flag == 255))


def zones(rp, centres, radii):
    """Zone index of every particle for one excision variant (3 = no sphere within 2 R_k)."""
    if not len(radii):
        return np.full(len(rp), 3, np.int64)
    s = np.full(len(rp), np.inf)
    for c, r in zip(np.asarray(centres).reshape(-1, 3), radii):
        d = rp - c
        s = np.minimum(s, np.sqrt(np.einsum('ij,ij->i', d, d)) / r)
    return np.searchsorted(ZONE_EDGES, s, side='right').astype(np.int64)


def accumulate(data, membership, halo, ctx):
    """All new blocks for one (halo, cell) pair (see module docstring)."""
    n = len(membership)
    n_ov = len(halo['ov_hid'])
    out = empty_blocks(n_ov)
    a, box, radius, r500 = halo['a'], halo['box'], halo['radius'], halo['r500']
    delta = data['Coordinates'] - halo['centre_cmpc']
    delta -= box * np.rint(delta / box)
    in4 = np.einsum('ij,ij->i', delta, delta) <= (4 * radius / a) ** 2      # radii234's exact 4R rule
    relative = delta * a / radius
    x200 = np.linalg.norm(relative, axis=1)
    rp = delta * a                                                         # physical offsets [Mpc]
    x500 = np.linalg.norm(rp / r500, axis=1)
    in6 = x500 <= G.X500[-1]
    s200 = np.where(in4, G.shell_index(x200, G.X200), 0)
    s500 = np.where(in6, G.shell_index(x500, G.X500), 0)
    out['in4'] = in4
    if n == 0:
        return out
    m, rho = data['Masses'], data['Densities']
    V = m / rho
    y, T = data['ComptonYParameters'], data['Temperatures']
    ne = data['ElectronNumberDensities'] * V
    ekT = ne * P.Boltzmann * T
    pe = y / V * PE_FACTOR
    lx = data['XrayRosat']
    heating = ctx['heating']
    agn = (data['LastAGNFeedbackScaleFactors'] >= heating['a_limit']) & (T >= heating['tmin_k']) & (T <= heating['tmax_k'])
    star_forming = data['StarFormationRates'] > 0
    phase = np.where(agn, 0, np.where(~star_forming & (T >= ctx['t_xray_min']), 1, 2)).astype(np.int64)
    agn_i = agn.astype(np.int64)
    owner, group, own, painted_other, unknown = owner_codes(membership, halo, ctx['painted'])
    ones = np.ones(n)
    weights = (ones, m, V, y, ne, ekT, data['Pressures'] * V, m * pe, V * pe * pe, m * T, V * T, m * rho,
               lx, lx * pe, lx * T, y * y)
    reduced = [weights[k] for k in REDUCED_INDEX]
    # main owner x phase x quantity blocks on both grids
    code = (owner * 3 + phase) * G.N200 + s200
    out['owner200'] = _sums(code, 17 * 3 * G.N200, weights, in4).reshape(len(QUANTITIES), 17, 3, G.N200).transpose(1, 2, 0, 3).copy()
    code = (owner * 3 + phase) * G.N500 + s500
    out['owner500'] = _sums(code, 17 * 3 * G.N500, weights, in6).reshape(len(QUANTITIES), 17, 3, G.N500).transpose(1, 2, 0, 3).copy()
    # excision zones around overlappers (or placebo spheres)
    member = np.where(owner == 0, 0, np.where(owner <= 12, 1, 2)).astype(np.int64)
    for v, (centres, radii) in enumerate(halo['excision']):
        zone = zones(rp, centres, radii)
        code = ((zone * 3 + member) * 2 + agn_i) * G.N200 + s200
        block = _sums(code, len(ZONES) * 3 * 2 * G.N200, reduced, in4)
        out['excision'][v] = block.reshape(len(REDUCED), len(ZONES), 3, 2, G.N200).transpose(1, 2, 3, 0, 4)
    # scaled-distance partition against painted distinct neighbours
    nbr_owns = np.zeros(n, np.int64)
    if len(halo['pn_r200']):
        best = np.full(n, np.inf)
        for c, r in zip(halo['pn_offset'], halo['pn_r200']):
            d = rp - c
            best = np.minimum(best, np.sqrt(np.einsum('ij,ij->i', d, d)) / r)
        nbr_owns = (best < x200).astype(np.int64)
    code = ((nbr_owns * 2 + painted_other.astype(np.int64)) * 2 + agn_i) * G.N200 + s200
    out['partition'] = _sums(code, 2 * 2 * 2 * G.N200, reduced, in4).reshape(len(REDUCED), 2, 2, 2, G.N200).transpose(1, 2, 3, 0, 4).copy()
    # auxiliary composition sums (same expressions as the legacy estimators)
    aux = (m * data['Hydrogen'], m * data['Helium'],
           data['Masses'] * P.MSUN_KG / P.proton_mass * (data['Hydrogen'] + data['Helium'] / 2))
    out['aux200'] = _sums(agn_i * G.N200 + s200, 2 * G.N200, aux, in4).reshape(3, 2, G.N200).transpose(1, 0, 2).copy()
    out['aux500'] = _sums(agn_i * G.N500 + s500, 2 * G.N500, aux, in6).reshape(3, 2, G.N500).transpose(1, 0, 2).copy()
    # smoothing-length histograms
    diag = out['diagnostics']
    for key, scaled, shells, select, nshell, k in (('hs200', data['SmoothingLengths'] * a / radius, s200, in4, G.N200, 1),
                                                    ('hs500', data['SmoothingLengths'] * a / r500, s500, in6, G.N500, 3)):
        hb = np.searchsorted(HS_EDGES, scaled, side='right') - 1
        diag[k] = int(np.sum(select & (hb < 0)))
        diag[k + 1] = int(np.sum(select & (hb >= NH)))
        hb = np.clip(hb, 0, NH - 1)
        out[key] = np.bincount(shells[select] * NH + hb[select], minlength=nshell * NH).reshape(nshell, NH).astype(np.uint32)
    # bound gas of each overlapper (own bound) and of its group (satellites of a neighbouring central overlapper)
    y_noagn = np.where(agn, 0., y)
    if n_ov:
        order = np.argsort(halo['ov_hid'], kind='stable')
        sorted_ov = halo['ov_hid'][order]
        p = np.minimum(np.searchsorted(sorted_ov, membership), n_ov - 1)
        own_k = np.where((membership >= 0) & (sorted_ov[p] == membership), order[p], -1)
        inside2 = (in4 & (x200 <= 2)).astype(np.int64)
        ov_weights = (ones, m, y, y_noagn, ne, ekT)
        for t, k_index in enumerate((own_k, group)):
            select = k_index >= 0
            sums = _sums(k_index * 2 + inside2, 2 * n_ov, ov_weights, select).reshape(len(ov_weights), n_ov, 2)
            out['overlapper'][:, t, 0, :] = sums[:, :, 1].T
            out['overlapper'][:, t, 1, :] = (sums[:, :, 0] + sums[:, :, 1]).T
    # gas inertia tensor inside 1 R200c (R200c units)
    inner = in4 & (x200 <= 1)
    for k, select in enumerate((inner, inner & (owner == 0))):
        r, w = relative[select], m[select]
        out['inertia'][k] = [w.sum(), w @ r[:, 0], w @ r[:, 1], w @ r[:, 2], w @ (r[:, 0] * r[:, 0]), w @ (r[:, 1] * r[:, 1]),
                             w @ (r[:, 2] * r[:, 2]), w @ (r[:, 0] * r[:, 1]), w @ (r[:, 0] * r[:, 2]), w @ (r[:, 1] * r[:, 2])]
    # spherical apertures for the SOAP comparison (gate G7)
    lx_noagn = np.where(agn, 0., lx)
    central = owner == 0
    masks = []
    for select in (in4 & (x200 <= 1), x500 <= 1, x500 <= 5, np.ones(n, bool)):
        masks.extend((select, select & central))
    # one matrix product: [7 quantities, n] @ [n, 8 aperture/population masks]
    totals = np.stack((ones, m, y, y_noagn, lx, lx_noagn, ne)) @ np.stack(masks, axis=1).astype(np.float64)
    out['apertures'] = totals.T.reshape(len(APERTURES), 2, len(APERTURE_QUANTITIES)).copy()
    # order-independent particle-ID hashes
    hashed = splitmix64(data['ParticleIDs'])
    for k, select in enumerate((in4, np.ones(n, bool))):
        out['id_hash'][k, 0] = np.add.reduce(hashed[select], dtype=np.uint64)
        out['id_hash'][k, 1] = np.bitwise_xor.reduce(hashed[select]) if select.any() else np.uint64(0)
        out['id_count'][k] = int(select.sum())
    diag[0] = unknown
    diag[5], diag[6], diag[7] = int(in4.sum()), int(in6.sum()), n
    diag[8] = int(np.sum(star_forming & (y != 0)))
    diag[9] = int(star_forming.sum())
    return out


MERGE_KEYS = ('owner200', 'owner500', 'excision', 'partition', 'aux200', 'aux500', 'hs200', 'hs500', 'overlapper',
              'inertia', 'apertures', 'id_hash', 'id_count', 'diagnostics')


def add_into(total, item):
    """total += item for every block (uint64 hashes: sum wraps mod 2^64, xor combines)."""
    for key in MERGE_KEYS:
        if key == 'id_hash':
            total[key][:, 0] += item[key][:, 0]
            total[key][:, 1] ^= item[key][:, 1]
        else:
            total[key] += item[key]
    return total
