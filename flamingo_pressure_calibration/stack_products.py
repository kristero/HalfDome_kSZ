"""Compact per-halo tSZ products from the L2p8_m9 reduced snapshots, for stacking.

For each snapshot: pick central haloes in 0.2-dex bins of M200c (random with a fixed seed,
adding haloes until a gas-particle budget per bin is reached, between N_MIN and N_MAX
haloes), stream the gas of the reduced snapshot cell by cell (every particle read once:
Coordinates, ComptonYParameters, Temperatures, LastAGNFeedbackScaleFactors,
SmoothingLengths, 44 bytes each) and store for every selected halo:

  - 3D shells in x = r/R200c: particle count, summed Compton y (raw, and without the gas
    SOAP flags as recently AGN-heated), and a histogram of log10(h/R200c) per shell for the
    resolution cut;
  - projected annuli in R/R200c, summed over the three box axes: count and summed y;
  - Compton-y maps in R200c units, summed over the three axes (raw and filtered);
  - particle sums inside R200c and R100c, checked against SOAP's own ComptonY.

The line of sight is truncated at each halo's R100c, where the reduced snapshot ends.

Usage (flamingo venv):
    python stack_products.py --snapshots 78 68 58 48 38 18 --workers 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flamingo_reduced_io as fio  # noqa: E402

SIM = "L2p8_m9"
BASE = fio.SIMULATIONS[SIM]
CACHE = HERE / "notebook_cache"
PRODUCTS = CACHE / "stacks"
EDGES = np.r_[0.0, np.geomspace(0.02, 1.6, 31)]   # 3D shells and projected annuli, units of R200c
HSML_LOG_EDGES = np.linspace(-2.5, 0.5, 61)        # log10(h / R200c), 0.05 dex
MAP_HALF_EXTENT, MAP_NPIX = 1.5, 64                  # maps span +-1.5 R200c
KEEP = list(dict.fromkeys(["halo_index", "m200c_msun", "m500c_msun", "gas_temperature_200c_k", "dm_concentration_200c"]
                          + [f"r_{ap}_mpc" for ap in fio.APERTURES] + [f"y_{ap}_mpc2" for ap in fio.APERTURES]
                          + [f"y_noagn_{ap}_mpc2" for ap in fio.APERTURES] + [f"ngas_{ap}" for ap in fio.APERTURES]))


def products_path(snapshot):
    return PRODUCTS / f"{SIM}_{snapshot:04d}_stack_products.npz"


def set_output_dir(path):
    global PRODUCTS
    PRODUCTS = Path(path)


class Log:
    def __init__(self, snapshot):
        PRODUCTS.mkdir(parents=True, exist_ok=True)
        self.path = PRODUCTS / f"log_{snapshot:04d}.jsonl"
        self.snapshot = snapshot

    def __call__(self, message, **fields):
        record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "snapshot": self.snapshot,
                  "message": message, **fields}
        line = json.dumps(record, default=lambda v: v.item() if hasattr(v, "item") else str(v))
        with self.path.open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


# --------------------------------------------------------------------------- selection
def select_haloes(table, bin_width, target_particles, n_min, n_max, seed, log_min=13.0):
    """Random centrals per 0.2-dex bin until sum(ngas in R100c) >= target, clipped to [n_min, n_max]."""
    logm = table["log10_m200c"].to_numpy()
    n_bins = int(np.ceil((logm.max() - log_min) / bin_width + 1e-9))
    edges = np.round(log_min + bin_width * np.arange(n_bins + 1), 6)
    ngas = table["ngas_100_crit"].to_numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    rows, bins, summary = [], [], []
    for b in range(n_bins):
        members = np.flatnonzero((logm >= edges[b]) & (logm < edges[b + 1]))
        if not len(members):
            continue
        order = rng.permutation(members)
        cumulative = np.cumsum(ngas[order])
        n = int(np.clip(np.searchsorted(cumulative, target_particles) + 1, n_min, n_max))
        n = min(n, len(members))
        rows.extend(order[:n].tolist())
        bins.extend([b] * n)
        summary.append({"bin": b, "log10_m200c_lo": float(edges[b]), "log10_m200c_hi": float(edges[b + 1]),
                        "available": int(len(members)), "selected": n, "particles_r100": int(cumulative[n - 1])})
    return np.array(rows, dtype=np.int64), np.array(bins, dtype=np.int64), edges, summary


# --------------------------------------------------------------------------- cell geometry
def cell_lookup(centres, size, n_cells):
    ijk = np.floor(centres / size + 1e-6).astype(int) % n_cells
    lookup = np.full(tuple(n_cells), -1, dtype=np.int64)
    lookup[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(len(centres))
    assert (lookup >= 0).all(), "cell grid is not a complete lattice"
    return lookup


def cells_touching(centre, radius, size, n_cells, lookup, box):
    """Cells whose cube intersects the (periodic) sphere; all lengths comoving Mpc."""
    lo = np.floor((centre - radius) / size).astype(int)
    hi = np.floor((centre + radius) / size).astype(int)
    found = []
    for ijk in itertools.product(*[range(lo[d], hi[d] + 1) for d in range(3)]):
        delta = np.abs((np.array(ijk) + 0.5) * size - centre)
        delta = np.minimum(delta, box - delta)
        if np.sum(np.maximum(delta - size / 2, 0.0) ** 2) <= radius**2:
            found.append(int(lookup[tuple(np.array(ijk) % n_cells)]))
    return found


# --------------------------------------------------------------------------- streaming
def stream_haloes(snap, centre_phys, r200, r100, a, heating, log, particles_per_request=2_000_000,
                  edges=None, hsml_log_edges=None, map_half_extent=None, map_npix=None):
    """One pass over the snapshot gas in the cells touched by the haloes' spheres.

    `r100` is the sphere radius (physical Mpc) inside which particles are used: R100c for the reduced
    snapshots (nothing exists beyond it), any radius for a full snapshot. Grids default to the module's.
    """
    EDGES_ = EDGES if edges is None else np.asarray(edges)
    HSML_ = HSML_LOG_EDGES if hsml_log_edges is None else np.asarray(hsml_log_edges)
    HALF_ = MAP_HALF_EXTENT if map_half_extent is None else float(map_half_extent)
    NPIX_ = MAP_NPIX if map_npix is None else int(map_npix)
    box = fio.scalar(snap["Header"].attrs["BoxSize"])                     # comoving Mpc
    meta = dict(snap["Cells/Meta-data"].attrs)
    size = np.asarray(meta["size"], dtype=float)
    n_cells = np.asarray(meta["dimension"], dtype=int)
    centres = np.asarray(fio.retry(lambda: snap["Cells/Centres"][...], what="Cells/Centres"))
    counts = np.asarray(fio.retry(lambda: snap["Cells/Counts/PartType0"][...], what="Cells/Counts")).astype(np.int64)
    offsets = np.asarray(fio.retry(lambda: snap["Cells/OffsetsInFile/PartType0"][...], what="Cells/Offsets")).astype(np.int64)
    lookup = cell_lookup(centres, size, n_cells)
    pad = 0.1 * size.max()           # same drift padding swiftsimio applies without Cells/Min/MaxPositions
    centre_com = centre_phys / a
    n_haloes = len(r200)
    cell_haloes = {}
    for h in range(n_haloes):
        for cell in cells_touching(centre_com[h], r100[h] / a + pad, size, n_cells, lookup, box):
            if counts[cell] > 0:
                cell_haloes.setdefault(cell, []).append(h)
    needed = sorted(cell_haloes, key=lambda c: offsets[c])
    ranges = []
    for cell in needed:
        start, stop = int(offsets[cell]), int(offsets[cell] + counts[cell])
        for s in range(start, stop, particles_per_request):
            ranges.append((cell, s, min(s + particles_per_request, stop)))
    batches, current, current_n = [], [], 0
    for item in ranges:
        n = item[2] - item[1]
        if current and current_n + n > particles_per_request:
            batches.append(current)
            current, current_n = [], 0
        current.append(item)
        current_n += n
    if current:
        batches.append(current)
    total = int(sum(e - s for _, s, e in ranges))
    log("particle pass planned", haloes=n_haloes, cells=len(needed), particles=total, batches=len(batches),
        estimated_gb=round(total * 44 / 1e9, 3))

    fields = {"xyz": snap["PartType0/Coordinates"], "y": snap["PartType0/ComptonYParameters"],
              "T": snap["PartType0/Temperatures"], "last": snap["PartType0/LastAGNFeedbackScaleFactors"],
              "h": snap["PartType0/SmoothingLengths"]}
    y_factor = fio.physical_factor(fields["y"]) / fio.MPC_CM**2          # -> physical Mpc^2
    nb, nh, npix = len(EDGES_) - 1, len(HSML_) - 1, NPIX_
    pixel = 2 * HALF_ / npix
    acc = {"n3": np.zeros((n_haloes, nb), np.int64), "y3_raw": np.zeros((n_haloes, nb)), "y3_noagn": np.zeros((n_haloes, nb)),
           "hsml_hist": np.zeros((n_haloes, nb, nh), np.int32),
           "np": np.zeros((n_haloes, nb), np.int64), "yp_raw": np.zeros((n_haloes, nb)), "yp_noagn": np.zeros((n_haloes, nb)),
           "map_raw": np.zeros((n_haloes, npix, npix)), "map_noagn": np.zeros((n_haloes, npix, npix)),
           "n_r200": np.zeros(n_haloes, np.int64), "y_r200_raw": np.zeros(n_haloes), "y_r200_noagn": np.zeros(n_haloes),
           "n_r100": np.zeros(n_haloes, np.int64), "y_r100_raw": np.zeros(n_haloes), "y_r100_noagn": np.zeros(n_haloes)}
    x100 = r100 / r200
    scale = a / r200                                                      # comoving Mpc -> units of R200c
    started, seen = time.monotonic(), 0
    for b, batch in enumerate(batches):
        arrays = {}
        for name, dataset in fields.items():
            slices = [np.s_[s:e, :] for _, s, e in batch] if name == "xyz" else [np.s_[s:e] for _, s, e in batch]
            arrays[name] = np.asarray(fio.retry(dataset.request_slices, slices, what=name))
        seen += len(arrays["y"])
        offset = 0
        for cell, s, e in batch:
            n = e - s
            xyz = arrays["xyz"][offset:offset + n]
            y_raw_all = arrays["y"][offset:offset + n] * y_factor
            temperature = arrays["T"][offset:offset + n]
            last = arrays["last"][offset:offset + n]
            hsml = arrays["h"][offset:offset + n]
            offset += n
            heated_all = (last >= heating["a_limit"]) & (temperature >= heating["Tmin_K"]) & (temperature <= heating["Tmax_K"])
            for h in cell_haloes[cell]:
                delta = xyz - centre_com[h]
                delta -= box * np.rint(delta / box)
                d = delta * scale[h]
                x2 = np.einsum("ij,ij->i", d, d)
                inside = x2 <= x100[h] ** 2
                if not inside.any():
                    continue
                d, x = d[inside], np.sqrt(x2[inside])
                yr = y_raw_all[inside]
                yn = np.where(heated_all[inside], 0.0, yr)
                k = np.minimum(np.searchsorted(EDGES_, x, side="right") - 1, nb - 1)
                acc["n3"][h] += np.bincount(k, minlength=nb)
                acc["y3_raw"][h] += np.bincount(k, weights=yr, minlength=nb)
                acc["y3_noagn"][h] += np.bincount(k, weights=yn, minlength=nb)
                lh = np.log10(np.maximum(hsml[inside] * scale[h], 1e-12))
                hb = np.clip(np.searchsorted(HSML_, lh, side="right") - 1, 0, nh - 1)
                acc["hsml_hist"][h] += np.bincount(k * nh + hb, minlength=nb * nh).reshape(nb, nh).astype(np.int32)
                for axis in range(3):
                    u, v = d[:, (axis + 1) % 3], d[:, (axis + 2) % 3]
                    kp = np.minimum(np.searchsorted(EDGES_, np.hypot(u, v), side="right") - 1, nb - 1)
                    acc["np"][h] += np.bincount(kp, minlength=nb)
                    acc["yp_raw"][h] += np.bincount(kp, weights=yr, minlength=nb)
                    acc["yp_noagn"][h] += np.bincount(kp, weights=yn, minlength=nb)
                    iu = np.floor((u + HALF_) / pixel).astype(np.int64)
                    iv = np.floor((v + HALF_) / pixel).astype(np.int64)
                    ok = (iu >= 0) & (iu < npix) & (iv >= 0) & (iv < npix)
                    index = iv[ok] * npix + iu[ok]
                    acc["map_raw"][h] += np.bincount(index, weights=yr[ok], minlength=npix * npix).reshape(npix, npix)
                    acc["map_noagn"][h] += np.bincount(index, weights=yn[ok], minlength=npix * npix).reshape(npix, npix)
                in200 = x <= 1.0
                acc["n_r200"][h] += int(in200.sum())
                acc["y_r200_raw"][h] += yr[in200].sum()
                acc["y_r200_noagn"][h] += yn[in200].sum()
                acc["n_r100"][h] += len(x)
                acc["y_r100_raw"][h] += yr.sum()
                acc["y_r100_noagn"][h] += yn.sum()
        del arrays
        if (b + 1) % 5 == 0 or b + 1 == len(batches):
            log("batches", done=b + 1, total=len(batches), particles=seen, seconds=round(time.monotonic() - started, 1),
                downloaded_gb=round(fio._DOWNLOADED["bytes"] / 1e9, 3))
    # sums of y per pixel -> mean Compton y per pixel (dimensionless), still summed over the 3 axes
    area = (pixel * r200) ** 2
    acc["map_raw"] = (acc["map_raw"] / area[:, None, None]).astype(np.float32)
    acc["map_noagn"] = (acc["map_noagn"] / area[:, None, None]).astype(np.float32)
    return acc, {"cells": len(needed), "batches": len(batches), "particles_read": int(seen)}


# --------------------------------------------------------------------------- driver
STORE_KEYS_PER_HALO = ["n3", "y3_raw", "y3_noagn", "hsml_hist", "np", "yp_raw", "yp_noagn", "map_raw", "map_noagn",
                       "n_r200", "y_r200_raw", "y_r200_noagn", "n_r100", "y_r100_raw", "y_r100_noagn"]


def _load_store(path):
    """Existing per-halo products for this snapshot (any earlier selection), keyed by halo_index."""
    if not path.exists():
        return None
    data = dict(np.load(path))
    same_grid = (np.array_equal(data["edges"], EDGES) and np.array_equal(data["hsml_log_edges"], HSML_LOG_EDGES)
                 and float(data["map_half_extent"]) == MAP_HALF_EXTENT and data["map_raw"].shape[-1] == MAP_NPIX)
    return data if same_grid else None


def run_snapshot(snapshot, args):
    """Catalogue (cached), selection, then stream only the selected haloes not already in the store."""
    warnings.filterwarnings("ignore")
    set_output_dir(args.out_dir)
    log = Log(snapshot)
    out = products_path(snapshot)
    receipt_path = out.with_suffix(".json")
    started = time.monotonic()
    counter = fio.install_byte_counter()
    bytes_at_start = counter["bytes"]              # pool workers are reused: count this snapshot only
    counter["limit"] = bytes_at_start + args.cap_gb * 1e9
    remote = fio.connect()
    cat = fio.load_reduced_haloes(remote, BASE, snapshot, CACHE, progress=False, log=log)
    meta = cat["meta"]
    catalogue_gb = (counter["bytes"] - bytes_at_start) / 1e9
    cosmo = meta["cosmology"]
    table = fio.central_table(cat)
    table = table[table["log10_m200c"] < args.max_log_m].reset_index(drop=True)
    rows, bins, bin_edges, summary = select_haloes(table, args.bin_width, args.target_particles, args.n_min, args.n_max,
                                                   args.seed + snapshot)
    store = _load_store(out)
    have = set() if store is None else set(store["cat_halo_index"].tolist())
    selected_ids = table["halo_index"].to_numpy()[rows]
    missing = np.array([i for i, hid in enumerate(selected_ids) if hid not in have], dtype=np.int64)
    log("selection", haloes=len(rows), already_stored=len(rows) - len(missing), to_stream=len(missing), bins=summary)
    new = None
    stats = {"cells": 0, "batches": 0, "particles_read": 0}
    if len(missing):
        chosen = table.iloc[rows[missing]]
        snap = remote[fio.reduced_snapshot_path(BASE, snapshot)]
        header_z = fio.scalar(snap["Header"].attrs["Redshift"])
        assert abs(header_z - cosmo["redshift"]) < 1e-6, (header_z, cosmo["redshift"])
        centre = chosen[["centre_x_mpc", "centre_y_mpc", "centre_z_mpc"]].to_numpy()
        acc, stats = stream_haloes(snap, centre, chosen["r_200_crit_mpc"].to_numpy(), chosen["r_100_crit_mpc"].to_numpy(),
                                   cosmo["a"], meta["recently_heated_filter"], log)
        catalogue = {f"cat_{name}": chosen[name].to_numpy() for name in KEEP if name in chosen}
        catalogue["cat_centre_mpc"] = centre
        catalogue["cat_p200_pa"] = fio.p200_pa(catalogue["cat_m200c_msun"], catalogue["cat_r_200_crit_mpc"], cosmo["baryon_fraction"])
        new = {**catalogue, **acc}
    # merge: every halo ever processed for this snapshot is kept
    per_halo_keys = STORE_KEYS_PER_HALO + [f"cat_{name}" for name in KEEP] + ["cat_centre_mpc", "cat_p200_pa"]
    parts = [x for x in (store, new) if x is not None]
    merged = {k: np.concatenate([x[k] for x in parts if k in x]) for k in per_halo_keys if any(k in x for x in parts)}
    logm = np.log10(merged["cat_m200c_msun"])
    merged_bins = np.floor((logm - 13.0) / args.bin_width + 1e-9).astype(np.int64)
    all_edges = np.round(13.0 + args.bin_width * np.arange(merged_bins.max() + 2), 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        checks = {"y200_raw": merged["y_r200_raw"] / merged["cat_y_200_crit_mpc2"] - 1,
                  "y200_noagn": merged["y_r200_noagn"] / merged["cat_y_noagn_200_crit_mpc2"] - 1,
                  "y100_raw": merged["y_r100_raw"] / merged["cat_y_100_crit_mpc2"] - 1,
                  "n200": merged["n_r200"] / merged["cat_ngas_200_crit"] - 1}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(temporary, snapshot=snapshot, redshift=cosmo["redshift"], scale_factor=cosmo["a"],
                        baryon_fraction=cosmo["baryon_fraction"], edges=EDGES, hsml_log_edges=HSML_LOG_EDGES,
                        map_half_extent=MAP_HALF_EXTENT, bin=merged_bins, bin_edges=all_edges, **merged,
                        **{f"check_{k}": v for k, v in checks.items()})
    temporary.replace(out)
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"runs": []}
    run = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "selection_settings": {
               "bin_width": args.bin_width, "target_particles": args.target_particles, "n_min": args.n_min,
               "n_max": args.n_max, "max_log_m": args.max_log_m, "seed": args.seed + snapshot},
           "selected": int(len(rows)), "streamed": int(len(missing)), **stats,
           "downloaded_gb_catalogue": round(catalogue_gb, 3), "downloaded_gb": round((counter["bytes"] - bytes_at_start) / 1e9, 3),
           "seconds": round(time.monotonic() - started, 1)}
    record = {"snapshot": snapshot, "redshift": cosmo["redshift"], "cosmology": cosmo,
              "catalogue_meta": {k: v for k, v in meta.items() if k != "cosmology"},
              "haloes_stored": int(len(merged_bins)), "selection": summary, "runs": previous.get("runs", []) + [run],
              "downloaded_gb_all_runs": round(sum(r["downloaded_gb"] for r in previous.get("runs", []) + [run]), 3),
              "soap_check_median_abs": {k: float(np.nanmedian(np.abs(v))) for k, v in checks.items()},
              "soap_check_max_abs": {k: float(np.nanmax(np.abs(v))) for k, v in checks.items()}}
    receipt_path.write_text(json.dumps(record, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    log("done", haloes_stored=len(merged_bins), streamed=len(missing), seconds=run["seconds"], downloaded_gb=run["downloaded_gb"],
        soap_check_max_abs=record["soap_check_max_abs"])
    return {"snapshot": snapshot, "status": "complete", "haloes": int(len(merged_bins)), "gb": run["downloaded_gb"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[78, 68, 58, 48, 38, 18])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--bin-width", type=float, default=0.2)
    parser.add_argument("--target-particles", type=float, default=5e5, help="gas particles (in R100c) per mass bin")
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=200)
    parser.add_argument("--max-log-m", type=float, default=15.2, help="exclude haloes above this log10 M200c")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--cap-gb", type=float, default=8.0, help="per-snapshot download cap")
    parser.add_argument("--out-dir", default=str(PRODUCTS))
    args = parser.parse_args()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_snapshot, s, args): s for s in args.snapshots}
        for future in as_completed(futures):
            snapshot = futures[future]
            try:
                results.append(future.result())
                print("FINISHED", results[-1], flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(snapshot)
                print("FAILED", snapshot, repr(exc), traceback.format_exc(), flush=True)
    print("ALL DONE", {"completed": results, "failed": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
