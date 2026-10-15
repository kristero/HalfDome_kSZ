"""Overlap census (plan Step 1.1): which subhalo spheres reach inside f R200c of a selected central.

Definitions (plan, user decision 2026-10-12):
  * overlapper of central c = any other subhalo j with mu = M_j / M200c_c >= 0.03 whose sphere reaches
    inside f R200c_c:  d_cj < f R200c_c + R_j  (f = 2 main definition, f = 1 the nested 'reaches 1R');
  * M_j = M200c for centrals, HBT bound mass (BoundSubhalo/TotalMass) for satellites;
  * R_j = R200c for centrals, pseudo-R200c (3 M_j / (4 pi 200 rho_c))^(1/3) for satellites;
  * own satellite: SOAP/HostHaloIndex of j is the SOAP row of c (radii234 convention);
  * painted distinct neighbour (Test A2 partition): another central with M200c >= 6.6e12 Msun whose
    scaled sphere can own gas inside c's extraction sphere: d < r_ext,c + (r_ext,c / R200c_c) R200c_n
    (= 4 R200c_c + 4 R200c_n when r_ext = 4 R200c_c).

The pure functions work on any population table (the local 421,868-central catalogue for the dry run,
or full SOAP on maas). Distances use the periodic minimum image; searches use per-bucket periodic
cKDTrees queried with (source reach + the bucket's largest radius), followed by the exact cut, so no
pair can be missed (verified against brute force in census_dryrun/ and gate G3).

Remote use (maas, after Checkpoint 1):  python -B census.py   (writes census_pairs.npz,
census_population.npz, census_summary.json, census_g3.json in this directory).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get('OVERLAP_STUDY_DIR', str(HERE))).resolve()   # data/config directory (rehearsal override)
MU_CLASSES = (0.03, 0.1, 0.3)
REACH_F = (1, 2)
PAINT_THRESHOLD_MSUN = 6.6e12
# 0.03 * 10^12.5 = 9.487e10: the floor must not exceed it, so every mu >= 0.03 overlapper of the lightest
# selectable central (10^12.5 Msun) is in the tree. (The plan's 9.5e10 rounded this value up.)
MASS_FLOOR_MSUN = 9.4e10
CLASS_NAMES = ('clean', 'cont_reach1R', 'cont_annulus_only')


# --------------------------------------------------------------------------- geometry helpers
def wrap(position, box):
    """Positions in [0, box) as cKDTree(boxsize=box) requires (x % box can round up to box)."""
    p = np.mod(np.asarray(position, dtype=np.float64), box)
    p[p >= box] -= box
    p[p < 0] = 0.0
    return p


def periodic_delta(a, b, box):
    """Minimum-image a - b."""
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return delta - box * np.rint(delta / box)


def pseudo_r200c(mass_msun, rho_crit_msun_mpc3):
    """Radius of a sphere of mean density 200 rho_c holding `mass` (physical Mpc)."""
    return np.cbrt(3 * np.asarray(mass_msun, dtype=np.float64) / (4 * np.pi * 200 * rho_crit_msun_mpc3))


def mu_class(mu):
    """0: 0.03 <= mu < 0.1, 1: 0.1 <= mu < 0.3, 2: mu >= 0.3, -1: below 0.03."""
    return np.searchsorted(np.asarray(MU_CLASSES), np.asarray(mu), side='right').astype(np.int64) - 1


# --------------------------------------------------------------------------- bucketed periodic search
def _buckets(target_radius, target_mass, n_buckets):
    key = np.log(np.maximum(target_mass if target_mass is not None else target_radius, 1e-300))
    order = np.argsort(key, kind='stable')
    return [order[s] for s in np.array_split(np.arange(len(order)), min(n_buckets, max(len(order), 1))) if len(s)]


def iter_ball_pairs(src_pos, src_reach, tgt_pos, tgt_radius, box, src_mass=None, tgt_mass=None, mu_min=None,
                    src_id=None, tgt_id=None, n_buckets=16, chunk=4096, workers=1):
    """Yield chunks (i, j, delta, d) of every pair with |x_j - x_i| < src_reach[i] + tgt_radius[j].

    delta = x_j - x_i (minimum image), d = |delta|. Optional mass pre-cut tgt_mass[j] >= mu_min * src_mass[i]
    (exact, applied after the search) and self-exclusion src_id[i] != tgt_id[j]. Targets are split into
    buckets by mass (or radius); a source queries a bucket with src_reach + the bucket's largest radius, and
    skips buckets whose largest mass is below mu_min * src_mass. Positions must already be wrapped.
    """
    src_pos, tgt_pos = np.asarray(src_pos, float), np.asarray(tgt_pos, float)
    src_reach, tgt_radius = np.asarray(src_reach, float), np.asarray(tgt_radius, float)
    for idx in _buckets(tgt_radius, tgt_mass, n_buckets):
        tree = cKDTree(tgt_pos[idx], boxsize=box)
        rmax = float(tgt_radius[idx].max())
        need = np.arange(len(src_pos))
        if mu_min is not None and tgt_mass is not None and src_mass is not None:
            need = need[float(np.max(tgt_mass[idx])) >= mu_min * np.asarray(src_mass)[need]]
        for start in range(0, len(need), chunk):
            sel = need[start:start + chunk]
            lists = tree.query_ball_point(src_pos[sel], src_reach[sel] + rmax, workers=workers)
            lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            if not lengths.sum():
                continue
            i = np.repeat(sel, lengths)
            j = idx[np.concatenate([np.asarray(x, dtype=np.int64) for x in lists if len(x)])]
            keep = np.ones(len(i), bool)
            if src_id is not None and tgt_id is not None:
                keep &= np.asarray(src_id)[i] != np.asarray(tgt_id)[j]
            if mu_min is not None and tgt_mass is not None and src_mass is not None:
                keep &= np.asarray(tgt_mass)[j] >= mu_min * np.asarray(src_mass)[i]
            i, j = i[keep], j[keep]
            delta = periodic_delta(tgt_pos[j], src_pos[i], box)
            d = np.sqrt(np.einsum('ij,ij->i', delta, delta))
            keep = d < src_reach[i] + tgt_radius[j]
            if keep.any():
                yield i[keep], j[keep], delta[keep], d[keep]


def ball_pairs(*args, **kwargs):
    """All pairs of iter_ball_pairs, sorted by (i, j)."""
    parts = list(iter_ball_pairs(*args, **kwargs))
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 3)), np.empty(0)
    i, j, delta, d = (np.concatenate(x) for x in zip(*parts))
    order = np.lexsort((j, i))
    return i[order], j[order], delta[order], d[order]


def symmetric_pair_counts(position, radius, box, multipliers=(1, 2, 3, 4)):
    """Unique central pairs whose equal-multiplier spheres intersect: d < f (R_a + R_b).

    Reproduces literature_overlap/overlap_geometry/analyze_overlap.find_pairs exactly: positions % box,
    d = |periodic_delta| via np.linalg.norm, the cut d < fmax (R_a + R_b), then q = d / (R_a + R_b) < f.
    """
    position = wrap(np.asarray(position) % box, box)
    radius = np.asarray(radius, float)
    fmax = max(multipliers)
    ids = np.arange(len(radius))
    # 1e-9 headroom: the exact legacy cuts below decide membership.
    i, j, _, _ = ball_pairs(position, fmax * radius * (1 + 1e-9), position, fmax * radius * (1 + 1e-9), box,
                            src_id=ids, tgt_id=ids)
    keep = i < j
    a, b = i[keep], j[keep]
    distance = np.linalg.norm(periodic_delta(position[a], position[b], box), axis=1)
    keep = distance < fmax * (radius[a] + radius[b])
    a, b, distance = a[keep], b[keep], distance[keep]
    q = distance / (radius[a] + radius[b])
    return {int(f): int(np.sum(q < f)) for f in multipliers}, (a, b, distance)


# --------------------------------------------------------------------------- the census proper
def overlapper_pairs(cen_pos, cen_r200, cen_m200, cen_row, pop, box, f_max=2, mu_min=MU_CLASSES[0],
                     n_buckets=16, workers=1):
    """Overlappers of each selected central: (i, j, delta, d) with mu >= mu_min and d < f_max R_c + R_j.

    pop: dict with pos (wrapped), radius (R_j), mass (M_j), row (SOAP row / catalogue index).
    Self-pairs (pop row == central row) are removed.
    """
    return ball_pairs(cen_pos, f_max * np.asarray(cen_r200), pop['pos'], pop['radius'], box,
                      src_mass=np.asarray(cen_m200), tgt_mass=pop['mass'], mu_min=mu_min,
                      src_id=np.asarray(cen_row), tgt_id=pop['row'], n_buckets=n_buckets, workers=workers)


def classify(any_reach1, any_reach2):
    """0 clean, 1 contaminated with an overlapper reaching inside 1R, 2 contaminated in the 1-2R annulus only."""
    any_reach1, any_reach2 = np.asarray(any_reach1, bool), np.asarray(any_reach2, bool)
    return np.where(any_reach1, 1, np.where(any_reach2, 2, 0)).astype(np.int8)


def max_mu_summary(n_src, i, mu, reach1, own_satellite):
    """Per source: max mu over overlappers (reaching 2R) split by reach-1R and own-satellite/neighbour.

    Returns float32 array [n_src, 2 (reach1R, reach2R), 2 (own satellite, neighbour)]; 0 if none.
    Classification at any threshold t: contaminated(f, type) = summary[:, f-1, type] >= t.
    """
    out = np.zeros((n_src, 2, 2), dtype=np.float64)
    kind = np.where(own_satellite, 0, 1)
    for f, sel in ((0, np.asarray(reach1, bool)), (1, np.ones(len(i), bool))):
        np.maximum.at(out[:, f, :], (i[sel], kind[sel]), mu[sel])
    return out.astype(np.float32)


def nearest_massive(cen_pos, cen_m200, cen_row, cand_pos, cand_m200, cand_row, box, ratio=0.1, radii=(10., 25., 60.)):
    """Offset vector to the nearest other central with M200c >= ratio * M200c_c (NaN if none within 60 Mpc).

    Used for the placebo 'along the nearest massive neighbour' direction (filament proxy)."""
    tree = cKDTree(cand_pos, boxsize=box)
    out = np.full((len(cen_pos), 3), np.nan)
    distance = np.full(len(cen_pos), np.nan)
    todo = np.arange(len(cen_pos))
    for r in radii:
        if not len(todo):
            break
        lists = tree.query_ball_point(cen_pos[todo], r)
        found = []
        for k, (c, nbrs) in enumerate(zip(todo, lists)):
            nbrs = np.asarray(nbrs, dtype=np.int64)
            nbrs = nbrs[(cand_m200[nbrs] >= ratio * cen_m200[c]) & (cand_row[nbrs] != cen_row[c])]
            if len(nbrs):
                delta = periodic_delta(cand_pos[nbrs], cen_pos[c], box)
                dd = np.linalg.norm(delta, axis=1)
                best = int(np.argmin(dd))
                out[c] = delta[best]
                distance[c] = dd[best]
                found.append(k)
        todo = np.delete(todo, found)
    return out, distance


def csr(index, n, *arrays):
    """Group rows by `index` (0..n-1) keeping the input order within a group: (ptr, order, arrays...)."""
    order = np.argsort(index, kind='stable')
    ptr = np.r_[0, np.cumsum(np.bincount(index, minlength=n))].astype(np.int64)
    return (ptr, order) + tuple(np.asarray(a)[order] for a in arrays)


def brute_force_pairs(src_pos, src_reach, tgt_pos, tgt_radius, box, src_mass=None, tgt_mass=None, mu_min=None,
                      src_id=None, tgt_id=None):
    """Reference implementation (no tree): the set {(i, j)} of iter_ball_pairs."""
    found = set()
    for i in range(len(src_pos)):
        delta = periodic_delta(tgt_pos, src_pos[i], box)
        d = np.sqrt(np.einsum('ij,ij->i', delta, delta))
        keep = d < src_reach[i] + tgt_radius
        if mu_min is not None:
            keep &= tgt_mass >= mu_min * src_mass[i]
        if src_id is not None:
            keep &= tgt_id != src_id[i]
        found.update((i, int(j)) for j in np.flatnonzero(keep))
    return found


def enlargement_plan(pool_bin, pool_class, selected_mask, n_bins, lower_bins, target, seed):
    """Seeded stratified additions: for each lower bin and class with fewer than `target` selected haloes,
    draw (without replacement) the missing number from the unselected members of that (bin, class)
    stratum (all of them if fewer remain). Returns the added pool indices (sorted) and, for every pool
    member, the stratum inclusion probability n_selected/N of its (bin, class) stratum after the additions
    (bins >= 14.5 are complete: probability 1)."""
    rng = np.random.default_rng(seed)
    added = []
    for b in range(n_bins):
        if b not in lower_bins:
            continue
        for k in range(len(CLASS_NAMES)):
            stratum = np.flatnonzero((pool_bin == b) & (pool_class == k))
            have = int(selected_mask[stratum].sum())
            if have >= target:
                continue
            free = stratum[~selected_mask[stratum]]
            take = min(target - have, len(free))
            if take > 0:
                added.append(rng.choice(free, take, replace=False))
    added = np.sort(np.concatenate(added)) if added else np.empty(0, np.int64)
    chosen = selected_mask.copy()
    chosen[added] = True
    probability = np.zeros(len(pool_bin))
    for b in range(n_bins):
        for k in range(len(CLASS_NAMES)):
            stratum = (pool_bin == b) & (pool_class == k)
            n = int(stratum.sum())
            if n:
                probability[stratum] = chosen[stratum].sum() / n
    return added, probability


# --------------------------------------------------------------------------- maas entry point
def read_population(soap, rho_crit, floor):
    """Full-SOAP population (read on maas only): every subhalo with M_j >= floor, and the global lookups."""
    import flamingo_io as F
    from common import MPC_CM, MSUN_G
    hid = soap.read('InputHalos/HaloCatalogueIndex')
    is_central_flag = soap.read('InputHalos/IsCentral').astype(bool)
    host = soap.read('SOAP/HostHaloIndex').astype(np.int64)
    central = host < 0                     # radii234 convention: satellites are the rows with a host
    centre = soap.read('InputHalos/HaloCentre', target=MPC_CM)
    bound = soap.read('BoundSubhalo/TotalMass', target=MSUN_G)
    m200 = soap.read('SO/200_crit/TotalMass', target=MSUN_G)
    r200 = soap.read('SO/200_crit/SORadius', target=MPC_CM)
    r500 = soap.read('SO/500_crit/SORadius', target=MPC_CM)
    consistency = {'iscentral_disagrees_with_host': int(np.sum(is_central_flag != central)),
                   'host_not_central': int(np.sum(~central[host[~central]])),
                   'hid_unique': bool(len(np.unique(hid)) == len(hid))}
    assert consistency['hid_unique'], 'HaloCatalogueIndex must be unique'
    mass = np.where(central, m200, bound)
    radius = np.where(central, r200, pseudo_r200c(bound, rho_crit))
    host_row = np.where(central, np.arange(len(hid)), host)
    host_m200 = m200[host_row]
    take = (mass >= floor) & (radius > 0)
    lookup_m200 = np.full(int(hid.max()) + 1, -1.0, dtype=np.float32)
    lookup_m200[hid] = host_m200.astype(np.float32)
    painted = np.full(int(hid.max()) + 1, 255, dtype=np.uint8)
    painted[hid] = (host_m200 >= PAINT_THRESHOLD_MSUN).astype(np.uint8)
    full = {'hid': hid, 'central': central, 'host': host, 'centre': centre, 'bound': bound, 'm200': m200,
            'r200': r200, 'r500': r500, 'consistency': consistency}
    pop = {'row': np.flatnonzero(take), 'hid': hid[take], 'central': central[take], 'host': host[take],
           'mass': mass[take], 'radius': radius[take], 'm200': m200[take], 'r200': r200[take]}
    return full, pop, lookup_m200, painted


def main():
    import h5py
    sys.path.insert(0, str(HERE / 'dependencies'))
    sys.path.insert(0, str(HERE))
    import common as C
    import flamingo_io as F
    from catalogue_stage import sphere_cells, csr_map
    started = time.perf_counter()
    cfg = json.loads((STUDY / 'study_config.json').read_text())
    root, snapshot_rel, soap_rel = cfg['data_root'], cfg['snapshot_rel'], cfg['soap_rel']
    out_pairs, out_population = STUDY / 'census_pairs.npz', STUDY / 'census_population.npz'
    assert not out_pairs.exists() and not out_population.exists(), 'census outputs exist; use a new attempt'
    snap = F.Snapshot(F.Store(root), snapshot_rel, cfg['snapshot'], read_from='chunks', chunk_cache_bytes=1024**2)
    soap = F.Soap(F.Store(root), soap_rel)
    a, box_c = snap.a, snap.box_size_cmpc
    box = box_c * a                                            # physical Mpc (a = 1 at z = 0)
    rho_cosmo = F.rho_crit_msun_mpc3(snap.cosmology, snap.unit_mass_cgs, snap.unit_length_cgs, snap.unit_time_cgs)
    full, pop, lookup_m200, painted = read_population(soap, rho_cosmo, cfg['census']['mass_floor_msun'])
    nrows = len(full['hid'])
    pos_all = wrap(full['centre'] * 1.0, box)                  # SOAP centres are physical at a = 1
    pop['pos'] = pos_all[pop['row']]
    valid = full['central'] & (full['m200'] > 0) & (full['r200'] > 0)
    rho_halo = float(np.median(C.rho_crit_from_halo(full['m200'][valid], full['r200'][valid])))
    # ---- pool: every central with 12.5 <= log M200c < 15.5
    edges = np.asarray(cfg['mass_edges_log10_Msun'])
    logm = np.log10(np.maximum(full['m200'], 1e-30))
    pool_rows = np.flatnonzero(valid & (logm >= edges[0]) & (logm < edges[-1]))
    pool_bin = np.searchsorted(edges, logm[pool_rows], side='right') - 1
    summary64 = np.zeros((len(pool_rows), 2, 2))      # [pool, reach 1R / 2R, own satellite / neighbour]
    own_host = full['host']
    pool_m200, pool_r200 = full['m200'][pool_rows], full['r200'][pool_rows]
    for i, j, delta, d in iter_ball_pairs(pos_all[pool_rows], 2 * pool_r200, pop['pos'], pop['radius'], box,
                                          src_mass=pool_m200, tgt_mass=pop['mass'], mu_min=MU_CLASSES[0],
                                          src_id=pool_rows, tgt_id=pop['row'], workers=cfg['census']['kdtree_workers']):
        mu = pop['mass'][j] / pool_m200[i]
        reach1 = d < pool_r200[i] + pop['radius'][j]
        kind = np.where(own_host[pop['row'][j]] == pool_rows[i], 0, 1)
        np.maximum.at(summary64[:, 1, :], (i, kind), mu)
        np.maximum.at(summary64[:, 0, :], (i[reach1], kind[reach1]), mu[reach1])
    summary = summary64.astype(np.float32)
    main_mu = cfg['main_definition']['mu']
    pool_class = classify(summary[:, 0, :].max(axis=1) >= main_mu, summary[:, 1, :].max(axis=1) >= main_mu)
    # ---- the old sample (radii234 rows, same order) and the seeded enlargement proposal
    old = np.load(STUDY / 'inputs' / 'radii234_selected_halos.npz')
    old_rows = old['soap_row']
    pos_in_pool = np.searchsorted(pool_rows, old_rows)
    assert np.array_equal(pool_rows[pos_in_pool], old_rows), 'every radii234 halo must be in the pool'
    assert np.array_equal(full['hid'][old_rows], old['halo_index'])
    selected = np.zeros(len(pool_rows), bool)
    selected[pos_in_pool] = True
    enl = cfg['enlargement']
    lower = [b for b in range(len(edges) - 1) if edges[b + 1] <= enl['below_log10_msun'] + 1e-9]
    added, probability = enlargement_plan(pool_bin, pool_class, selected, len(edges) - 1, lower,
                                          enl['target_per_class'], enl['seed'])
    sample_pool = np.r_[pos_in_pool, added]                     # old rows first, in radii234 order
    rows = pool_rows[sample_pool]
    role = np.r_[np.zeros(len(pos_in_pool), np.int8), np.ones(len(added), np.int8)]
    # ---- detailed census for old + proposed additions
    cen_pos, r200c, m200c = pos_all[rows], full['r200'][rows], full['m200'][rows]
    r500c = full['r500'][rows]
    r_ext = np.maximum(cfg['extraction']['r200c_multiple'] * r200c, cfg['extraction']['r500c_multiple'] * r500c)
    i, j, delta, d = overlapper_pairs(cen_pos, r200c, m200c, rows, pop, box, f_max=2,
                                      workers=cfg['census']['kdtree_workers'])
    prow = pop['row'][j]
    mu = pop['mass'][j] / m200c[i]
    ptr, order = csr(i, len(rows))[:2]
    ov = {'halo': i[order], 'soap_row': prow[order], 'hid': full['hid'][prow][order],
          'is_central': full['central'][prow][order], 'host_row': full['host'][prow][order],
          'host_hid': np.where(full['host'][prow] >= 0, full['hid'][np.maximum(full['host'][prow], 0)], -1)[order],
          'own_satellite': (full['host'][prow] == rows[i])[order], 'mass_msun': pop['mass'][j][order],
          'mu': mu[order], 'radius_mpc': pop['radius'][j][order], 'offset_mpc': delta[order], 'distance_mpc': d[order],
          'reach1': (d < r200c[i] + pop['radius'][j])[order], 'mu_class': mu_class(mu)[order]}
    gas_n = soap.read('BoundSubhalo/NumberOfGasParticles', np.unique(ov['soap_row'])) if len(ov['soap_row']) else np.empty(0)
    gas_m = soap.read('BoundSubhalo/GasMass', np.unique(ov['soap_row']), target=C.MSUN_G) if len(ov['soap_row']) else np.empty(0)
    k = np.searchsorted(np.unique(ov['soap_row']), ov['soap_row'])
    ov['bound_gas_count'], ov['bound_gas_mass_msun'] = gas_n[k], gas_m[k]
    # own satellites (all masses), sorted by HaloCatalogueIndex within each halo, as radii234 did
    sat = np.flatnonzero(full['host'] >= 0)
    sorted_rows = np.argsort(rows)
    where = np.searchsorted(rows[sorted_rows], full['host'][sat])
    where = np.minimum(where, len(rows) - 1)
    mine = rows[sorted_rows][where] == full['host'][sat]
    sat, sat_halo = sat[mine], sorted_rows[where[mine]]
    order_s = np.lexsort((full['hid'][sat], sat_halo))
    sat_ptr = np.r_[0, np.cumsum(np.bincount(sat_halo, minlength=len(rows)))].astype(np.int64)
    sat_hid = full['hid'][sat][order_s]
    # satellites of neighbouring-central overlappers (they count as that overlapper)
    group_rows = np.unique(ov['soap_row'][ov['is_central'] & ~ov['own_satellite']])
    gsat = np.flatnonzero(full['host'] >= 0)
    if len(group_rows):
        gpos = np.minimum(np.searchsorted(group_rows, full['host'][gsat]), len(group_rows) - 1)
        gmine = group_rows[gpos] == full['host'][gsat]
    else:
        gpos, gmine = np.zeros(len(gsat), np.int64), np.zeros(len(gsat), bool)
    gsat, gidx = gsat[gmine], gpos[gmine]
    order_g = np.lexsort((full['hid'][gsat], gidx))
    group_ptr = np.r_[0, np.cumsum(np.bincount(gidx, minlength=len(group_rows)))].astype(np.int64)
    group_member_hid = full['hid'][gsat][order_g]
    # painted distinct neighbours that can own gas inside the extraction sphere (partition)
    painted_c = full['central'] & (full['m200'] >= PAINT_THRESHOLD_MSUN) & (full['r200'] > 0)
    prow_all = np.flatnonzero(painted_c)
    scale = r_ext / r200c
    pi, pj, pdelta, pd = ball_pairs(cen_pos, r_ext, pos_all[prow_all], full['r200'][prow_all] * scale.max(), box,
                                    src_id=rows, tgt_id=prow_all, workers=cfg['census']['kdtree_workers'])
    keep = pd < r_ext[pi] + scale[pi] * full['r200'][prow_all][pj]
    pi, pj, pdelta, pd = pi[keep], pj[keep], pdelta[keep], pd[keep]
    pptr, porder = csr(pi, len(rows))[:2]
    part = {'halo': pi[porder], 'soap_row': prow_all[pj][porder], 'hid': full['hid'][prow_all][pj][porder],
            'm200c_msun': full['m200'][prow_all][pj][porder], 'r200c_mpc': full['r200'][prow_all][pj][porder],
            'offset_mpc': pdelta[porder], 'distance_mpc': pd[porder]}
    # nearest massive neighbour (mu >= 0.1) direction for placebo_nbr
    cand = np.flatnonzero(full['central'] & (full['m200'] >= 0.1 * 10**edges[0]))
    nbr_offset, nbr_distance = nearest_massive(cen_pos, m200c, rows, pos_all[cand], full['m200'][cand], cand, box)
    # ---- G3: brute force on a seeded subset; own-satellite lists vs radii234
    rng = np.random.default_rng(cfg['census']['g3_seed'])
    subset = np.unique(np.r_[rng.choice(len(rows), min(40, len(rows)), replace=False), np.argsort(m200c)[-5:]])
    brute = brute_force_pairs(cen_pos[subset], 2 * r200c[subset], pop['pos'], pop['radius'], box,
                              src_mass=m200c[subset], tgt_mass=pop['mass'], mu_min=MU_CLASSES[0],
                              src_id=rows[subset], tgt_id=pop['row'])
    brute_rows = {(int(subset[a_]), int(pop['row'][b_])) for a_, b_ in brute}
    subset_set = set(subset.tolist())
    tree_rows = {(int(h), int(r)) for h, r in zip(ov['halo'], ov['soap_row']) if int(h) in subset_set}
    n_old = len(pos_in_pool)
    same_sats = (np.array_equal(sat_ptr[:n_old + 1], old['satellite_ptr'])
                 and np.array_equal(sat_hid[:sat_ptr[n_old]], old['satellite_ids']))
    g3 = {'brute_force_subset_haloes': int(len(subset)), 'brute_force_pairs': len(brute_rows),
          'tree_pairs_on_subset': len(tree_rows), 'overlapper_sets_equal': brute_rows == tree_rows,
          'own_satellite_lists_equal_radii234': bool(same_sats),
          'rho_crit_cosmology_msun_mpc3': rho_cosmo, 'rho_crit_from_soap_m200_r200_median': rho_halo,
          'rho_crit_relative_difference': rho_halo / rho_cosmo - 1}
    (STUDY / 'census_g3.json').write_text(json.dumps(g3, indent=1) + '\n')
    assert g3['overlapper_sets_equal'] and g3['own_satellite_lists_equal_radii234'], g3
    # ---- read-budget estimate (cell metadata only)
    def budget(take):
        ph, pc = sphere_cells(cen_pos[take] / a, r_ext[take] / a, snap.cell_size_cmpc, snap.dimension, snap.lookup,
                              box_c, .1 * snap.cell_size_cmpc)
        return np.unique(pc)
    bpp = cfg['extraction']['bytes_per_particle']
    cells_old, cells_all = budget(role == 0), budget(np.ones(len(rows), bool))
    rows_old, rows_all = int(snap.counts[cells_old].sum()), int(snap.counts[cells_all].sum())
    np.savez_compressed(out_pairs, halo_soap_row=rows, halo_hid=full['hid'][rows], role=role,
                        overlap_ptr=ptr, **{'ov_' + k: v for k, v in ov.items()},
                        satellite_ptr=sat_ptr, satellite_hid=sat_hid,
                        group_central_row=group_rows, group_central_hid=full['hid'][group_rows],
                        group_ptr=group_ptr, group_member_hid=group_member_hid,
                        partition_ptr=pptr, **{'pn_' + k: v for k, v in part.items()},
                        nearest_massive_offset_mpc=nbr_offset, nearest_massive_distance_mpc=nbr_distance,
                        r_ext_mpc=r_ext, r500c_mpc=r500c)
    np.savez_compressed(out_population, pool_soap_row=pool_rows, pool_hid=full['hid'][pool_rows],
                        pool_m200c_msun=full['m200'][pool_rows], pool_bin=pool_bin, pool_class=pool_class,
                        pool_max_mu=summary, pool_inclusion_probability_enlarged=probability,
                        pool_selected_old=selected, pool_added=added,
                        painted_by_hid=painted, host_m200c_by_hid=lookup_m200,
                        mass_edges=edges, rho_crit_msun_mpc3=rho_cosmo)
    counts = {}
    for b in range(len(edges) - 1):
        inb = pool_bin == b
        counts[f'{edges[b]:.1f}-{edges[b+1]:.1f}'] = {
            'population': int(inb.sum()),
            'population_by_class': np.bincount(pool_class[inb], minlength=3).tolist(),
            'old_sample_by_class': np.bincount(pool_class[inb & selected], minlength=3).tolist(),
            'proposed_additions_by_class': np.bincount(pool_class[np.intersect1d(np.flatnonzero(inb), added)], minlength=3).tolist()}
    result = {'rows_soap': nrows, 'population_above_floor': int(len(pop['row'])), 'pool_centrals': int(len(pool_rows)),
              'old_sample': n_old, 'proposed_additions': int(len(added)), 'per_bin': counts,
              'overlapper_pairs': int(len(ov['halo'])), 'partition_pairs': int(len(part['halo'])),
              'neighbour_group_centrals': int(len(group_rows)), 'neighbour_group_members': int(len(group_member_hid)),
              'cells_old': int(len(cells_old)), 'rows_old': rows_old, 'cells_with_additions': int(len(cells_all)),
              'rows_with_additions': rows_all, 'bytes_per_particle': bpp,
              'logical_bytes_old': rows_old * bpp, 'logical_bytes_with_additions': rows_all * bpp,
              'read_budget_bytes': cfg['extraction']['read_budget_bytes'], 'g3': g3,
              'soap_consistency': full['consistency'],
              'wall_seconds': time.perf_counter() - started}
    (STUDY / 'census_summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_bin'}, indent=1))
    snap.close()
    soap.close()


if __name__ == '__main__':
    main()
