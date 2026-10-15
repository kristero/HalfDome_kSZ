"""Stage 1: the halo catalogue of one (simulation, snapshot) and the cells its spheres need.

Centrals (IsCentral == 1) with M200c >= 10^log_m_min and R200c > 0, selected without reference
to their environment:
  random_block every such central whose padded sphere (r_sphere + cell padding) lies entirely inside
               its block, for a random subset (random_block_fraction) of the blocks of k^3 top-level
               cells (random_block_k; block lattice aligned to the cell grid with a random offset,
               periodic; seeded by seed, sim, snap). The cells read are exactly those blocks' cells:
               the I/O-efficient default (random_cell reads most of the 26 neighbours of every cell);
  random_cell  every such central whose centre is in a random subset of top-level cells;
  complete     centrals above complete_log_m_min anywhere, at most max_complete_per_fine_bin per
               fine bin (random subset if more);
  reference    exactly the haloes of an earlier products file (InputHalos/HaloCatalogueIndex =
               cat_halo_index), with that file's shell edges and sphere radius.
Whether a halo fits in its block depends on its mass and on where the (random) block lattice falls,
never on its neighbours, so random_block is environment-unbiased; its mass-dependent efficiency
((1 - 2 (r_sphere + pad) / (k cell))^3) only changes how many haloes each mass bin gets.
SOAP rows are sorted by top-level cell, so random blocks/cells and reference haloes are read through
the SOAP cell index (a few rows); only the complete selection scans whole columns (local files only).
A halo needs every cell whose box, padded by cell_padding_fraction x cell size for particle drift,
intersects its sphere of r_sphere = sphere_r200c x R200c (periodic, minimum image).
Own subhaloes (for the 'exclude gas bound to other haloes' accumulators of the particle stage): every catalogued
central's own InputHalos/HaloCatalogueIndex plus those of the subhaloes whose SOAP/HostHaloIndex is its SOAP row,
searched among the SOAP rows of the needed cells (CSR own_member_ptr / own_member_index).

Usage:
    python catalogue_stage.py --config config.json --sim L2p8_m9 --snap 78 [--reference PRODUCTS.npz] [--max-haloes K]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import flamingo_io  # noqa: E402

RANDOM_CELL, COMPLETE, REFERENCE, RANDOM_BLOCK = 1, 2, 4, 8
REASON_BITS = {"random_cell": RANDOM_CELL, "complete": COMPLETE, "reference": REFERENCE, "random_block": RANDOM_BLOCK}
GRID_KEYS = ("edges", "hsml_log_edges", "fine_bin_edges")
READ_FIELDS = tuple(flamingo_io.GAS_FIELDS)


# --------------------------------------------------------------------------- geometry
def sphere_cells(centres_cmpc, radii_cmpc, size, dimension, lookup, box, pad, chunk=20_000):
    """(halo, cell) pairs of every cell whose padded box intersects a halo's sphere (comoving, periodic).

    The distance from the sphere centre to each candidate cell box, enlarged by `pad` on every side,
    is taken with the minimum-image convention; pairs are unique and sorted by halo then cell.
    """
    centres = np.asarray(centres_cmpc, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii_cmpc, dtype=float)
    size, pad = np.asarray(size, dtype=float), np.asarray(pad, dtype=float) * np.ones(3)
    dimension = np.asarray(dimension, dtype=np.int64)
    half = size / 2 + pad
    reach = radii[:, None] + pad[None, :]
    lo = np.floor((centres - reach) / size).astype(np.int64)
    hi = np.floor((centres + reach) / size).astype(np.int64)
    haloes, cells = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
    for start in range(0, len(centres), chunk):
        sl = slice(start, start + chunk)
        span = (hi[sl] - lo[sl] + 1).max(axis=0)
        offsets = np.stack(np.meshgrid(*[np.arange(s) for s in span], indexing="ij"), axis=-1).reshape(-1, 3)
        ijk = lo[sl, None, :] + offsets[None, :, :]                                  # (haloes, candidates, 3)
        valid = np.all(ijk <= hi[sl, None, :], axis=2)
        delta = np.abs((ijk % dimension + 0.5) * size - centres[sl, None, :])
        delta = np.minimum(delta, box - delta)                                        # minimum image
        gap = np.maximum(delta - half, 0.0)
        touch = valid & (np.einsum("hck,hck->hc", gap, gap) <= radii[sl, None] ** 2)
        h, c = np.nonzero(touch)
        wrapped = ijk[h, c] % dimension
        haloes.append(h + start)
        cells.append(lookup[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]])
    n_cells = int(np.prod(dimension))
    key = np.unique(np.concatenate(haloes) * n_cells + np.concatenate(cells))
    return key // n_cells, key % n_cells


def csr_map(pair_halo, pair_cell):
    """Cell -> haloes map: (cells, ptr, idx) with idx[ptr[i]:ptr[i+1]] the haloes of cells[i]."""
    order = np.lexsort((pair_halo, pair_cell))
    cells_sorted, haloes_sorted = pair_cell[order], pair_halo[order]
    cells, first = np.unique(cells_sorted, return_index=True)
    ptr = np.r_[first, len(cells_sorted)].astype(np.int64)
    return cells.astype(np.int64), ptr, haloes_sorted.astype(np.int64)


def greedy_subset(pair_halo, pair_cell, cell_ngas, n_haloes, k):
    """At most k haloes that need few gas particles: repeatedly take every halo whose cells are already
    chosen, else the one adding the fewest particles (rehearsals; deterministic, lowest index first)."""
    cells, inverse = np.unique(pair_cell, return_inverse=True)
    gas = np.asarray(cell_ngas, dtype=float)[cells][inverse]
    chosen = np.zeros(len(cells), bool)
    free = np.ones(n_haloes, bool)
    picked = []
    while len(picked) < min(k, n_haloes):
        cost = np.bincount(pair_halo, weights=gas * ~chosen[inverse], minlength=n_haloes)
        cost[~free] = np.inf
        zero = np.flatnonzero(cost == 0)
        take = zero[:k - len(picked)] if len(zero) else [int(np.argmin(cost))]
        for h in take:
            picked.append(int(h))
            free[h] = False
            chosen[inverse[pair_halo == h]] = True
    return np.sort(np.asarray(picked, dtype=np.int64))


def neighbour_cells(cells, cell_ijk, dimension, lookup):
    """The 27 cells around (and including) each cell."""
    offsets = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    ijk = (cell_ijk[np.asarray(cells)][:, None, :] + offsets[None]) % np.asarray(dimension)
    return np.unique(lookup[ijk[..., 0], ijk[..., 1], ijk[..., 2]])


# --------------------------------------------------------------------------- blocks of k^3 cells
def block_layout(dimension, k, offset=(0, 0, 0)):
    """(blocks per axis, offset in cells) of the lattice of k^3-cell blocks; k must divide the cell grid.

    `offset` is one integer per axis or a single integer for all three (e.g. random_block_offset = 1)."""
    dimension = np.asarray(dimension, dtype=np.int64).reshape(3)
    k = int(k)
    if k < 1 or np.any(dimension % k):
        divisors = [d for d in range(1, int(dimension.min()) + 1) if not np.any(dimension % d)]
        raise ValueError(f"selection.random_block_k = {k} must divide the cell grid {dimension.tolist()} "
                         f"(valid: {divisors})")
    try:
        offset = np.broadcast_to(np.asarray(offset, dtype=np.int64), (3,))
    except (TypeError, ValueError):
        raise ValueError(f"selection.random_block_offset = {offset!r} must be 'random', one integer or three") from None
    return dimension // k, offset % k


def block_of_cells(ijk, dimension, k, offset=(0, 0, 0)):
    """Block index (C order on the block lattice) of cells with grid indices ijk (..., 3); periodic."""
    nb, offset = block_layout(dimension, k, offset)
    dimension = np.asarray(dimension, dtype=np.int64).reshape(3)
    b = ((np.asarray(ijk, dtype=np.int64) - offset) % dimension) // k
    return (b[..., 0] * nb[1] + b[..., 1]) * nb[2] + b[..., 2]


def block_cells(blocks, dimension, k, lookup, offset=(0, 0, 0)):
    """Sorted cell indices of the k^3 cells of each block (cells offset + k b + [0, k) per axis, periodic)."""
    nb, offset = block_layout(dimension, k, offset)
    dimension = np.asarray(dimension, dtype=np.int64).reshape(3)
    blocks = np.asarray(blocks, dtype=np.int64).reshape(-1)
    if not len(blocks):
        return np.empty(0, np.int64)
    bijk = np.stack(np.unravel_index(blocks, tuple(int(n) for n in nb)), axis=1)
    sub = np.stack(np.meshgrid(*[np.arange(int(k))] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    ijk = (offset + int(k) * bijk[:, None, :] + sub[None, :, :]) % dimension
    return np.unique(lookup[ijk[..., 0], ijk[..., 1], ijk[..., 2]]).astype(np.int64)


def inside_block(centres_cmpc, reach_cmpc, size, dimension, k, offset=(0, 0, 0)):
    """(block of each centre, inside): inside = the centre's cube of half-side `reach` (the padded sphere's bounding
    box, reach = r_sphere + cell padding, comoving) lies strictly inside the centre's block (minimum image).

    Then no cell outside the block has a padded box within r_sphere of the centre, i.e. sphere_cells lists only
    cells of the block (checked again after sphere_cells in run()).
    """
    nb, offset = block_layout(dimension, k, offset)
    dimension = np.asarray(dimension, dtype=np.int64).reshape(3)
    size = np.asarray(size, dtype=float) * np.ones(3)
    box = size * dimension
    centres = np.asarray(centres_cmpc, dtype=float).reshape(-1, 3)
    b = ((np.floor(centres / size).astype(np.int64) % dimension - offset) % dimension) // int(k)
    d = centres - (offset + int(k) * b + int(k) / 2.0) * size
    d -= box * np.rint(d / box)
    half = int(k) * size / 2
    inside = np.all(np.abs(d) + np.asarray(reach_cmpc, dtype=float).reshape(-1, 1) < half - 1e-9 * size, axis=1)
    return (b[:, 0] * nb[1] + b[:, 1]) * nb[2] + b[:, 2], inside


def fine_bins(logm, dex, log_lo=None):
    """Fine mass bins anchored at multiples of `dex` (so 0.2-dex stacks are unions of whole fine bins)."""
    logm = np.asarray(logm, dtype=float)
    lo = logm.min() if len(logm) else (log_lo if log_lo is not None else 0.0)
    if log_lo is not None:
        lo = min(lo, log_lo)
    hi = logm.max() if len(logm) else lo
    i0 = int(np.floor(lo / dex + 1e-9))
    i1 = int(np.floor(hi / dex + 1e-9)) + 1
    edges = np.round(dex * np.arange(i0, i1 + 1), 6)
    index = np.clip(np.floor(logm / dex + 1e-9).astype(np.int64) - i0, 0, len(edges) - 2)
    return index, edges


# --------------------------------------------------------------------------- selection
def _log10(values):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.asarray(values, dtype=float))


def select_normal(cfg, sim, snap, soap, grid, log):
    """Rows (sorted) and reason bits of the random_block, random_cell and complete selections.

    grid: n_cells, size, dimension, lookup (cell grid of the snapshot), pad (cell padding, comoving) and a.
    """
    sel = common.selection_cfg(cfg, sim)
    modes = list(sel.get("modes", ["random_block", "complete"]))
    unknown = sorted(set(modes) - {"random_block", "random_cell", "complete"})
    if unknown:
        raise ValueError(f"selection.modes: unknown {unknown} (random_block, random_cell, complete)")
    n_cells = int(grid["n_cells"])
    parts = []                                            # (rows, reason bit)
    info = {"modes": modes, "random_cells": np.empty(0, np.int64)}

    def central_cut(values):
        return (values["is_central"] == 1) & (values["r_200_crit_mpc"] > 0) & (_log10(values["m200c_msun"]) >= sel["log_m_min"])

    if "random_block" in modes:
        k = int(sel.get("random_block_k", 2))
        rng = np.random.default_rng(common.seed_for(cfg, sim, snap, 3))
        drawn = rng.integers(0, k, 3)                     # drawn always, so the block choice never depends on it
        chosen_offset = sel.get("random_block_offset", "random")
        n_per_axis, offset = block_layout(grid["dimension"], k, drawn if chosen_offset in (None, "random") else chosen_offset)
        n_blocks = int(np.prod(n_per_axis))
        fraction = float(sel["random_block_fraction"])
        n = min(n_blocks, max(int(round(fraction * n_blocks)), 1 if fraction > 0 else 0))
        blocks = np.sort(rng.choice(n_blocks, n, replace=False)) if n else np.empty(0, np.int64)
        cells = block_cells(blocks, grid["dimension"], k, grid["lookup"], offset)
        rows, _ = soap.cell_rows(cells)
        values = soap.read_columns(rows, {**flamingo_io.SELECTION_COLUMNS, "centre_mpc": flamingo_io.CATALOGUE_COLUMNS["centre_mpc"]})
        ok = central_cut(values)
        reach = float(cfg["radial"]["sphere_r200c"]) * values["r_200_crit_mpc"] / grid["a"] + float(np.max(grid["pad"]))
        block, inside = inside_block(values["centre_mpc"].reshape(-1, 3) / grid["a"], reach, grid["size"], grid["dimension"], k, offset)
        keep = ok & inside & np.isin(block, blocks)
        parts.append((rows[keep], RANDOM_BLOCK))
        info.update(random_blocks=blocks, random_block_cells=cells, random_block_offset=offset.tolist(), random_block_k=k,
                    n_random_blocks=int(n), n_blocks_total=n_blocks, random_block_cells_count=int(len(cells)),
                    soap_rows_in_random_blocks=int(len(rows)), random_block_candidates=int(ok.sum()),
                    random_block_rejected_sphere_outside_block=int(np.sum(ok & ~inside)),
                    random_block_rejected_centre_in_other_block=int(np.sum(ok & inside & ~np.isin(block, blocks))),
                    n_random_block=int(keep.sum()))
        log("random blocks chosen", k=k, offset=offset.tolist(), blocks=int(n), of_blocks=n_blocks, cells=int(len(cells)),
            soap_rows=int(len(rows)), centrals=int(ok.sum()), selected=int(keep.sum()))
    if "random_cell" in modes:
        rng = np.random.default_rng(common.seed_for(cfg, sim, snap, 1))
        fraction = float(sel["random_cell_fraction"])
        n = min(n_cells, max(int(round(fraction * n_cells)), 1 if fraction > 0 else 0))
        cells = np.sort(rng.choice(n_cells, n, replace=False)) if n else np.empty(0, np.int64)
        rows, _ = soap.cell_rows(cells)
        values = soap.read_columns(rows, flamingo_io.SELECTION_COLUMNS)
        keep = central_cut(values)
        parts.append((rows[keep], RANDOM_CELL))
        info.update(random_cells=cells, n_random_cells=int(n), soap_rows_in_random_cells=int(len(rows)),
                    n_random=int(keep.sum()))
        log("random cells chosen", cells=int(n), soap_rows=int(len(rows)), centrals_selected=int(keep.sum()))
    if "complete" in modes:
        if soap.remote:
            raise RuntimeError("the complete selection scans whole SOAP columns and is not available over hdfstream; "
                               "use selection.modes = ['random_block'] (or ['random_cell']) or --reference")
        threshold = float(sel["complete_log_m_min"])
        rows = soap.scan(flamingo_io.SELECTION_COLUMNS,
                         lambda v: (v["is_central"] == 1) & (v["r_200_crit_mpc"] > 0) & (_log10(v["m200c_msun"]) >= threshold))
        logm = _log10(soap.read(flamingo_io.CATALOGUE_COLUMNS["m200c_msun"][0], rows, common.MSUN_G))
        index, _ = fine_bins(logm, float(sel["fine_bin_dex"]))
        rng = np.random.default_rng(common.seed_for(cfg, sim, snap, 2))
        cap = int(sel["max_complete_per_fine_bin"])
        chosen = [np.sort(rng.choice(rows[index == b], cap, replace=False)) if np.sum(index == b) > cap else rows[index == b]
                  for b in np.unique(index)]
        chosen = np.concatenate(chosen) if chosen else np.empty(0, np.int64)
        parts.append((chosen, COMPLETE))
        info.update(n_complete_available=int(len(rows)), n_complete=int(len(chosen)))
        log("complete selection", available=int(len(rows)), selected=int(len(chosen)))
    rows = np.unique(np.concatenate([p for p, _ in parts])) if parts else np.empty(0, np.int64)
    reasons = np.zeros(len(rows), dtype=np.int64)
    for part, bit in parts:
        reasons[np.searchsorted(rows, part)] |= bit
    return rows.astype(np.int64), reasons, info


def match_reference(soap, ids, cells, cell_ijk, dimension, lookup, log):
    """SOAP rows of the haloes `ids` (HaloCatalogueIndex), searched in the cell of each centre, then
    in the 26 neighbours, then (local files only) in the whole column."""
    found = np.full(len(ids), -1, dtype=np.int64)
    searched = np.empty(0, np.int64)
    for ring in (0, 1):
        todo = np.flatnonzero(found < 0)
        if not len(todo):
            break
        candidates = np.unique(cells[todo]) if ring == 0 else neighbour_cells(cells[todo], cell_ijk, dimension, lookup)
        candidates = np.setdiff1d(candidates, searched)
        searched = np.union1d(searched, candidates)
        rows, _ = soap.cell_rows(candidates)
        index = soap.read("InputHalos/HaloCatalogueIndex", rows)
        hit = np.zeros(len(todo), bool)
        if len(rows):
            order = np.argsort(index, kind="stable")
            pos = np.minimum(np.searchsorted(index[order], ids[todo]), len(order) - 1)
            hit = index[order][pos] == ids[todo]
            found[todo[hit]] = rows[order][pos[hit]]
        log("reference lookup", ring=ring, cells=int(len(candidates)), soap_rows=int(len(rows)), matched=int(hit.sum()),
            missing=int((found < 0).sum()))
    missing = np.flatnonzero(found < 0)
    if len(missing):
        if soap.remote:
            raise KeyError(f"{len(missing)} reference haloes not found in their cells or neighbours, e.g. id {ids[missing[0]]}")
        wanted = ids[missing]
        rows = soap.scan({"id": ("InputHalos/HaloCatalogueIndex", None)}, lambda v: np.isin(v["id"], wanted))
        index = soap.read("InputHalos/HaloCatalogueIndex", rows)
        lookup_rows = dict(zip(index.tolist(), rows.tolist()))
        for i in missing:
            found[i] = lookup_rows.get(int(ids[i]), -1)
        log("reference full-column lookup", matched=int((found[missing] >= 0).sum()))
        if (found < 0).any():
            raise KeyError(f"{int((found < 0).sum())} reference haloes are not in this SOAP catalogue")
    return found


def own_members(soap, halo_rows, halo_index, cells):
    """CSR (ptr, index) of each catalogued central's own subhaloes (HaloCatalogueIndex values, sorted per halo):
    itself and every subhalo of the SOAP rows of `cells` whose SOAP/HostHaloIndex (SOAP row of its top-level
    host, -1 for centrals) is the central's row. Returns (ptr, index, info)."""
    halo_rows = np.asarray(halo_rows, dtype=np.int64)
    halo_index = np.asarray(halo_index, dtype=np.int64)
    n = len(halo_rows)
    info = {"host_index": soap.has("SOAP/HostHaloIndex")}
    pair_halo, pair_index = [np.arange(n, dtype=np.int64)], [halo_index]
    if info["host_index"] and n:
        rows, _ = soap.cell_rows(cells)
        host = np.asarray(soap.read("SOAP/HostHaloIndex", rows), dtype=np.int64) if len(rows) else np.empty(0, np.int64)
        order = np.argsort(halo_rows, kind="stable")
        pos = np.clip(np.searchsorted(halo_rows[order], host), 0, max(n - 1, 0))
        mine = (host >= 0) & (halo_rows[order][pos] == host)
        if mine.any():
            sat_rows = rows[mine]
            pair_halo.append(order[pos[mine]])
            pair_index.append(np.asarray(soap.read("InputHalos/HaloCatalogueIndex", sat_rows), dtype=np.int64))
        info.update(soap_rows_searched=int(len(rows)), satellites_found=int(mine.sum()))
    halo = np.concatenate(pair_halo)
    index = np.concatenate(pair_index)
    key = np.unique(np.stack([halo, index], axis=1), axis=0) if len(halo) else np.empty((0, 2), np.int64)
    ptr = np.r_[0, np.cumsum(np.bincount(key[:, 0], minlength=n))].astype(np.int64)
    return ptr, key[:, 1].astype(np.int64), info


# --------------------------------------------------------------------------- driver
def _settings(cfg, sim, snap, reference, max_haloes):
    ref = None
    if reference is not None:
        ref = {"path": str(Path(reference).resolve()), "sha1": common.fingerprint(np.fromfile(reference, dtype=np.uint8))}
    return {"sim": sim, "snap": int(snap), "data_root": cfg["data_root"], "sim_cfg": common.sim_cfg(cfg, sim),
            "selection": common.selection_cfg(cfg, sim), "radial": cfg["radial"],
            "cell_padding_fraction": cfg["particles"]["cell_padding_fraction"], "reference": ref,
            "max_haloes": max_haloes, "softening": cfg.get("softening")}


def run(cfg, sim, snap, reference=None, max_haloes=None, force=False):
    """Build <output_root>/catalogue/<sim>_<snap>_haloes.npz (+ .json); returns the JSON summary."""
    started = time.monotonic()
    snap = int(snap)
    P = common.paths(cfg, sim, snap)
    logfile = P["logs_dir"] / f"catalogue_{P['tag']}.jsonl"
    log = lambda message, **kw: common.log_event(logfile, stage="catalogue", sim=sim, snap=snap, message=message, **kw)  # noqa: E731
    settings = _settings(cfg, sim, snap, reference, max_haloes)
    settings_hash = common.fingerprint(np.frombuffer(json.dumps(settings, sort_keys=True, default=common.json_default).encode(),
                                                     dtype=np.uint8))
    if not force and P["catalogue_npz"].exists() and P["catalogue_json"].exists():
        previous = json.loads(P["catalogue_json"].read_text())
        if previous.get("settings_hash") == settings_hash:
            log("catalogue exists with the same settings; skipped", haloes=previous.get("n_haloes"))
            return {**previous, "status": "exists"}

    snapshot = flamingo_io.open_snapshot(cfg, sim, snap)
    soap = flamingo_io.open_soap(cfg, sim, snap)
    try:
        cosmo = soap.cosmology
        a = cosmo["a"]
        if abs(cosmo["redshift"] - snapshot.redshift) > 1e-6:
            raise ValueError(f"SOAP z = {cosmo['redshift']} but snapshot z = {snapshot.redshift}")
        if not np.allclose(soap.cell_centres_cmpc * (snapshot.unit_length_cgs / common.MPC_CM), snapshot.cell_centres_cmpc,
                           rtol=1e-9, atol=1e-6):
            raise ValueError("SOAP and snapshot cell grids differ")
        heating = soap.heating_filter()
        size, dimension, lookup, box = snapshot.cell_size_cmpc, snapshot.dimension, snapshot.lookup, snapshot.box_size_cmpc
        pad = float(cfg["particles"]["cell_padding_fraction"]) * size
        cell_ijk = np.floor(snapshot.cell_centres_cmpc / size + 1e-6).astype(np.int64) % dimension
        log("opened", snapshot=snapshot.describe(), soap_rows=soap.n_rows, redshift=cosmo["redshift"])

        info = {}
        reference_row = None
        if reference is not None:
            ref = common.load_npz(reference)
            if abs(float(ref["redshift"]) - cosmo["redshift"]) > 1e-5:
                raise ValueError(f"reference file is at z = {float(ref['redshift'])}, this snapshot at z = {cosmo['redshift']}")
            ref_a = float(ref["scale_factor"])
            ids = ref["cat_halo_index"].astype(np.int64)
            ref_sphere = ref["cat_r_sphere_mpc"] if "cat_r_sphere_mpc" in ref else ref["cat_r_100_crit_mpc"]
            ref_centre_c = ref["cat_centre_mpc"] / ref_a
            keep = np.arange(len(ids))
            if max_haloes is not None and max_haloes < len(ids):
                ph, pc = sphere_cells(ref_centre_c, ref_sphere / ref_a, size, dimension, lookup, box, pad)
                keep = greedy_subset(ph, pc, snapshot.counts, len(ids), int(max_haloes))
            found = match_reference(soap, ids[keep], snapshot.cell_of(ref_centre_c[keep]), cell_ijk, dimension, lookup, log)
            order = np.argsort(found)
            rows, reference_row = found[order], keep[order]
            if len(np.unique(rows)) != len(rows):
                raise ValueError("reference file lists a halo twice")
            reasons = np.full(len(rows), REFERENCE, dtype=np.int64)
            edges, hsml_edges = np.asarray(ref["edges"], float), np.asarray(ref["hsml_log_edges"], float)
            r_sphere = np.asarray(ref_sphere, dtype=float)[reference_row]
            info = {"modes": ["reference"], "reference": str(reference), "reference_haloes": int(len(ids)),
                    "sphere": "cat_r_sphere_mpc" if "cat_r_sphere_mpc" in ref else "cat_r_100_crit_mpc"}
        else:
            grid = {"n_cells": snapshot.n_cells, "size": size, "dimension": dimension, "lookup": lookup, "pad": pad, "a": a}
            rows, reasons, info = select_normal(cfg, sim, snap, soap, grid, log)
            edges, hsml_edges = common.radial_edges(cfg), common.hsml_log_edges(cfg)

        cat = soap.read_columns(rows)
        cat["soap_row"] = rows
        cat["sel_reason"] = reasons
        cat["reference_row"] = reference_row if reference_row is not None else np.full(len(rows), -1, np.int64)
        if reference is None:
            cat["r_sphere_mpc"] = float(cfg["radial"]["sphere_r200c"]) * cat["r_200_crit_mpc"]
        else:
            cat["r_sphere_mpc"] = r_sphere
            checks = {"max_abs_centre_diff_mpc": float(np.max(np.abs(cat["centre_mpc"] - ref["cat_centre_mpc"][reference_row]), initial=0.0)),
                      "max_rel_m200c_diff": float(np.max(np.abs(cat["m200c_msun"] / ref["cat_m200c_msun"][reference_row] - 1), initial=0.0)),
                      "max_rel_r200c_diff": float(np.max(np.abs(cat["r_200_crit_mpc"] / ref["cat_r_200_crit_mpc"][reference_row] - 1), initial=0.0)),
                      "not_central": int(np.sum(cat["is_central"] != 1))}
            if "cat_r_100_crit_mpc" in ref:
                checks["max_rel_r100c_diff"] = float(np.max(np.abs(cat["r_100_crit_mpc"] / ref["cat_r_100_crit_mpc"][reference_row] - 1), initial=0.0))
            info["reference_checks"] = checks
            log("reference haloes matched", haloes=int(len(rows)), **checks)
            if checks["max_rel_m200c_diff"] > 1e-5 or checks["max_abs_centre_diff_mpc"] > 1e-4:
                raise ValueError(f"reference haloes differ from this SOAP catalogue: {checks}")

        # spheres -> cells (padded boxes, periodic); pairs only with cells that hold gas
        centre_cmpc = cat["centre_mpc"] / a
        pair_halo, pair_cell = sphere_cells(centre_cmpc, cat["r_sphere_mpc"] / a, size, dimension, lookup, box, pad)
        if reference is None and max_haloes is not None and max_haloes < len(rows):
            subset = greedy_subset(pair_halo, pair_cell, snapshot.counts, len(rows), int(max_haloes))
            cat = {k: v[subset] for k, v in cat.items()}
            centre_cmpc = centre_cmpc[subset]
            remap = np.full(len(rows), -1, np.int64)
            remap[subset] = np.arange(len(subset))
            use = remap[pair_halo] >= 0
            pair_halo, pair_cell = remap[pair_halo[use]], pair_cell[use]
            rows = cat["soap_row"]
        block_cell_list = np.asarray(info.get("random_block_cells", np.empty(0, np.int64)), dtype=np.int64)
        if len(block_cell_list):          # random_block haloes read their block's cells only (by construction)
            stray = ((cat["sel_reason"][pair_halo] & RANDOM_BLOCK) > 0) & ~np.isin(pair_cell, block_cell_list)
            if stray.any():
                raise RuntimeError(f"{int(np.unique(pair_halo[stray]).size)} random_block haloes need cells outside their "
                                   f"block, e.g. cell {int(pair_cell[stray][0])} (inside_block/sphere_cells disagree)")
        with_gas = snapshot.counts[pair_cell] > 0
        cells_needed, ptr, idx = csr_map(pair_halo[with_gas], pair_cell[with_gas])
        own_ptr, own_index, own_info = own_members(soap, rows, cat["halo_index"], np.unique(pair_cell))
        info["own_subhaloes"] = own_info
        if not own_info["host_index"]:
            log("WARNING: SOAP lacks SOAP/HostHaloIndex: satellites of the catalogued centrals cannot be identified, so "
                "gas bound to them counts as 'bound to other haloes' in the *_own accumulators")

        logm = _log10(cat["m200c_msun"])
        dex = float(common.selection_cfg(cfg, sim)["fine_bin_dex"])
        fine_bin, fine_edges = fine_bins(logm, dex, None if reference is not None else float(common.selection_cfg(cfg, sim)["log_m_min"]))
        soap_cell = soap.cell_of_rows(rows)
        centre_cell = snapshot.cell_of(centre_cmpc) if len(rows) else np.empty(0, np.int64)
        n_soap_geometry_mismatch = int(np.sum(soap_cell != centre_cell))
        cell_ngas = snapshot.counts[cells_needed]
        bytes_per_particle = snapshot.field_bytes(READ_FIELDS)

        out = {k: np.asarray(v) for k, v in cat.items()}
        out.update({
            "halo_index": cat["halo_index"].astype(np.int64), "centre_cmpc": centre_cmpc,
            "fine_bin": fine_bin, "fine_bin_edges": fine_edges, "edges": edges, "hsml_log_edges": hsml_edges,
            "p200_pa": common.p200_pa(cat["m200c_msun"], cat["r_200_crit_mpc"], cosmo["baryon_fraction"]),
            "rho_crit_halo_msun_mpc3": common.rho_crit_from_halo(cat["m200c_msun"], cat["r_200_crit_mpc"]),
            "soap_cell": soap_cell.astype(np.int64), "centre_cell": centre_cell.astype(np.int64),
            "cells_needed": cells_needed, "cell_ngas": cell_ngas, "cell_halo_ptr": ptr, "cell_halo_idx": idx,
            "cell_file": snapshot.top_files[cells_needed], "cell_offset": snapshot.top_offsets[cells_needed],
            "cell_centre_cmpc": snapshot.cell_centres_cmpc[cells_needed].reshape(-1, 3),
            "random_cells": np.asarray(info.get("random_cells", np.empty(0, np.int64)), dtype=np.int64),
            "random_blocks": np.asarray(info.get("random_blocks", np.empty(0, np.int64)), dtype=np.int64),
            "random_block_cells": block_cell_list,
            "random_block_offset": np.asarray(info.get("random_block_offset", [0, 0, 0]), dtype=np.int64),
            "random_block_k": np.asarray(int(info.get("random_block_k", 0))),
            "own_member_ptr": own_ptr, "own_member_index": own_index,
        })
        rho_crit = flamingo_io.rho_crit_msun_mpc3(cosmo, snapshot.unit_mass_cgs, snapshot.unit_length_cgs, snapshot.unit_time_cgs)
        scalars = {"snapshot": snap, "redshift": cosmo["redshift"], "scale_factor": a, "h": cosmo["h"],
                   "box_size_cmpc": box, "omega_b": cosmo["omega_b"], "omega_m": cosmo["omega_m"],
                   "omega_cdm": cosmo["omega_cdm"], "omega_nu_0": cosmo["omega_nu_0"],
                   "baryon_fraction": cosmo["baryon_fraction"], "rho_crit_msun_mpc3": rho_crit,
                   "a_limit": heating["a_limit"], "tmin_k": heating["tmin_k"], "tmax_k": heating["tmax_k"],
                   "softening_comoving_mpc": snapshot.softening_comoving_mpc,
                   "softening_max_physical_mpc": snapshot.softening_max_physical_mpc,
                   "cell_size_cmpc": float(size[0]), "cell_padding_cmpc": float(pad[0]),
                   "sphere_r200c": np.nan if reference is not None else float(cfg["radial"]["sphere_r200c"]),
                   "fine_bin_dex": dex, "n_haloes": int(len(rows)), "bytes_per_particle": int(bytes_per_particle)}
        out.update({k: np.asarray(v) for k, v in scalars.items()})
        fp = common.fingerprint(out["soap_row"], out["halo_index"], out["r_sphere_mpc"], out["centre_cmpc"], edges,
                                hsml_edges, cells_needed, ptr, idx, own_ptr, own_index)
        out.update({"fingerprint": np.asarray(fp), "sim": np.asarray(sim),
                    "selection_mode": np.asarray("reference" if reference is not None else "+".join(info["modes"]))})
        common.atomic_savez(P["catalogue_npz"], compress=True, **out)

        gas = int(cell_ngas.sum())
        bits = REASON_BITS
        per_bin = []
        for b in np.unique(fine_bin):
            members = fine_bin == b
            per_bin.append({"bin": int(b), "log10_m200c_lo": float(fine_edges[b]), "log10_m200c_hi": float(fine_edges[b + 1]),
                            "haloes": int(members.sum()),
                            **{name: int(np.sum(members & ((out["sel_reason"] & bit) > 0))) for name, bit in bits.items()}})
        halo_cells = np.bincount(idx, minlength=len(rows)) if len(rows) else np.empty(0, np.int64)
        summary = {
            "status": "complete", "utc": common.utc_now(), "sim": sim, "snapshot": snap, "redshift": cosmo["redshift"],
            "fingerprint": fp, "settings_hash": settings_hash, "settings": settings,
            "selection": {k: v for k, v in info.items() if k not in ("random_cells", "random_blocks", "random_block_cells")},
            "n_haloes": int(len(rows)), "reasons": {name: int(np.sum((out["sel_reason"] & bit) > 0)) for name, bit in bits.items()},
            "per_fine_bin": per_bin, "cells_needed": int(len(cells_needed)), "gas_particles": gas,
            "bytes_per_particle": int(bytes_per_particle), "estimated_bytes": int(gas * bytes_per_particle),
            "estimated_tb": gas * bytes_per_particle / 1e12,
            "fraction_of_snapshot": gas / max(snapshot.n_gas_total, 1), "n_cells": int(snapshot.n_cells),
            "cells_in_random_blocks": int(np.isin(cells_needed, block_cell_list).sum()),
            "gas_in_random_blocks": int(cell_ngas[np.isin(cells_needed, block_cell_list)].sum()),
            "cells_per_halo": {"mean": float(halo_cells.mean()) if len(rows) else 0.0, "max": int(halo_cells.max(initial=0))},
            "haloes_without_gas_cells": int(np.sum(halo_cells == 0)),
            "own_subhaloes": {**own_info, "members": int(len(own_index))},
            "soap_cell_vs_centre_cell_mismatch": n_soap_geometry_mismatch,
            "snapshot_layout": snapshot.describe(), "heating_filter": heating,
            "rho_crit_msun_mpc3": rho_crit,
            "rho_crit_halo_median_ratio": float(np.median(out["rho_crit_halo_msun_mpc3"]) / rho_crit) if len(rows) else None,
            "seconds": round(time.monotonic() - started, 2),
        }
        common.atomic_write_json(P["catalogue_json"], summary)
        log("catalogue written", haloes=int(len(rows)), cells=int(len(cells_needed)), gas_particles=gas,
            estimated_tb=round(summary["estimated_tb"], 4), seconds=summary["seconds"])
        return summary
    finally:
        snapshot.close()
        soap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--snap", type=int, required=True)
    parser.add_argument("--reference", default=None, help="products .npz whose haloes (cat_halo_index) are selected")
    parser.add_argument("--max-haloes", type=int, default=None, help="keep at most K haloes needing few cells (rehearsals)")
    parser.add_argument("--force", action="store_true", help="rebuild even if a catalogue with the same settings exists")
    args = parser.parse_args(argv)
    summary = run(common.load_config(args.config), args.sim, args.snap, reference=args.reference,
                  max_haloes=args.max_haloes, force=args.force)
    print("CATALOGUE", {k: summary.get(k) for k in ("status", "n_haloes", "cells_needed", "gas_particles", "estimated_tb",
                                                   "fraction_of_snapshot")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
