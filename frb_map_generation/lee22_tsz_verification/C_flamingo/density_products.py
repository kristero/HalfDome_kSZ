"""Gas-density and electron-density shells for the haloes of the FLAMINGO L2p8_m9 preliminary pressure stacks.

Runs on idark (cluster venv of 2026-09-21, hdfstream from COSMA). For every snapshot it reads the
per-halo selection written by `make_selections.py` (the exact haloes of
flamingo_pressure_calibration/notebook_cache/stacks/L2p8_m9_NNNN_stack_products.npz, so pressure and
density come from the same haloes and the same shells), streams the reduced-snapshot gas of the cells
touched by each halo's R100c sphere (every particle read once, in cell order) and stores per halo and
3D shell of x = r/R200c (same edges as the stack products: [0, geomspace(0.02, 1.6, 31)]):

  n3         particle count
  m3         gas mass                       [Msun]           (all gas)
  ne3        electron number sum N_e        [1]              N_e,i = n_e,i m_i / rho_i (SPH volume)
  ne3_nosf   electrons of non-star-forming gas (StarFormationRates <= 0)
  m3_sf      star-forming gas mass          [Msun]
  m3_cold    gas mass with T < 1e5 K        [Msun]
  pe3        sum n_e k_B T V                [Pa m^3]         (electron pressure x volume)
  y3         sum ComptonYParameters         [physical Mpc^2] (cross-check with the stack products)
and inside R200c: m_r200, ne_r200, m_r200_sf, n_r200.

All quantities physical (conversion factors from the dataset attributes). FLAMINGO electron densities
come from SWIFT's cooling tables (ionisation state and metals included); X_H = 0.752 (primordial).
The reduced snapshots hold every particle within R100c of the flagged haloes, so shells are only
complete inside R100c (the straddling shell is truncated at R100c and flagged by the reader).

Usage (idark):
    python density_products.py --root <dir> --snapshots 78 68 58 48 38 18 14 --workers 4
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flamingo_reduced_io as fio  # noqa: E402  (unchanged copy of flamingo_pressure_calibration/flamingo_reduced_io.py)

BASE = fio.SIMULATIONS["L2p8_m9"]
K_B = 1.380649e-23
MSUN_KG = 1.98841e30


def cell_lookup(centres, size, n_cells):
    ijk = np.floor(centres / size + 1e-6).astype(int) % n_cells
    lookup = np.full(tuple(n_cells), -1, dtype=np.int64)
    lookup[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(len(centres))
    assert (lookup >= 0).all(), "cell grid is not a complete lattice"
    return lookup


def cells_touching(centre, radius, size, n_cells, lookup, box):
    lo = np.floor((centre - radius) / size).astype(int)
    hi = np.floor((centre + radius) / size).astype(int)
    found = []
    for ijk in itertools.product(*[range(lo[d], hi[d] + 1) for d in range(3)]):
        delta = np.abs((np.array(ijk) + 0.5) * size - centre)
        delta = np.minimum(delta, box - delta)
        if np.sum(np.maximum(delta - size / 2, 0.0) ** 2) <= radius**2:
            found.append(int(lookup[tuple(np.array(ijk) % n_cells)]))
    return found


def log_line(path, **fields):
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    line = json.dumps(record, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    with open(path, "a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def run_snapshot(snapshot, root, cap_gb, particles_per_request=2_000_000):
    root = Path(root)
    out = root / "products" / f"L2p8_m9_{snapshot:04d}_density_products.npz"
    if out.exists():
        return {"snapshot": snapshot, "status": "exists"}
    logp = root / "logs" / f"density_{snapshot:04d}.jsonl"
    sel = dict(np.load(root / "selections" / f"selection_{snapshot:04d}.npz"))
    a = float(sel["scale_factor"])
    counter = fio.install_byte_counter()
    start_bytes = counter["bytes"]
    counter["limit"] = start_bytes + cap_gb * 1e9
    started = time.monotonic()
    remote = fio.connect()
    snap = remote[fio.reduced_snapshot_path(BASE, snapshot)]
    header_z = fio.scalar(snap["Header"].attrs["Redshift"])
    assert abs(header_z - float(sel["redshift"])) < 1e-6, (header_z, float(sel["redshift"]))
    box = fio.scalar(snap["Header"].attrs["BoxSize"])
    meta = dict(snap["Cells/Meta-data"].attrs)
    size = np.asarray(meta["size"], dtype=float)
    n_cells = np.asarray(meta["dimension"], dtype=int)
    centres = np.asarray(fio.retry(lambda: snap["Cells/Centres"][...], what="Cells/Centres"))
    counts = np.asarray(fio.retry(lambda: snap["Cells/Counts/PartType0"][...], what="Cells/Counts")).astype(np.int64)
    offsets = np.asarray(fio.retry(lambda: snap["Cells/OffsetsInFile/PartType0"][...], what="Cells/Offsets")).astype(np.int64)
    lookup = cell_lookup(centres, size, n_cells)
    pad = 0.1 * size.max()
    r200, r100 = sel["r200_mpc"], sel["r100_mpc"]
    centre_com = sel["centre_mpc"] / a
    n_haloes = len(r200)
    cell_haloes = {}
    for h in range(n_haloes):
        for cell in cells_touching(centre_com[h], r100[h] / a + pad, size, n_cells, lookup, box):
            if counts[cell] > 0:
                cell_haloes.setdefault(cell, []).append(h)
    needed = sorted(cell_haloes, key=lambda c: offsets[c])
    ranges = []
    for cell in needed:
        s0, s1 = int(offsets[cell]), int(offsets[cell] + counts[cell])
        for s in range(s0, s1, particles_per_request):
            ranges.append((cell, s, min(s + particles_per_request, s1)))
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
    log_line(logp, snapshot=snapshot, message="planned", haloes=n_haloes, cells=len(needed), particles=total,
             batches=len(batches), estimated_gb=round(total * 56 / 1e9, 3))

    g = snap["PartType0"]
    fields = {"xyz": g["Coordinates"], "m": g["Masses"], "rho": g["Densities"], "ne": g["ElectronNumberDensities"],
              "T": g["Temperatures"], "sfr": g["StarFormationRates"], "y": g["ComptonYParameters"]}
    f_m = fio.physical_factor(fields["m"]) / (MSUN_KG * 1e3)          # -> Msun
    f_rho = fio.physical_factor(fields["rho"]) * 1e3                  # g cm^-3 -> kg m^-3 (physical)
    f_ne = fio.physical_factor(fields["ne"]) * 1e6                    # cm^-3 -> m^-3 (physical)
    f_y = fio.physical_factor(fields["y"]) / fio.MPC_CM**2            # -> physical Mpc^2
    edges = np.asarray(sel["edges"], float)
    nb = len(edges) - 1
    keys3 = ["m3", "ne3", "ne3_nosf", "m3_sf", "m3_cold", "pe3", "y3"]
    acc = {k: np.zeros((n_haloes, nb)) for k in keys3}
    acc["n3"] = np.zeros((n_haloes, nb), np.int64)
    for k in ["m_r200", "ne_r200", "m_r200_sf", "y_r200", "m_r100", "ne_r100"]:
        acc[k] = np.zeros(n_haloes)
    acc["n_r200"] = np.zeros(n_haloes, np.int64)
    x100 = r100 / r200
    scale = a / r200
    seen = 0
    for b, batch in enumerate(batches):
        arrays = {}
        for name, dataset in fields.items():
            slices = [np.s_[s:e, :] for _, s, e in batch] if name == "xyz" else [np.s_[s:e] for _, s, e in batch]
            arrays[name] = np.asarray(fio.retry(dataset.request_slices, slices, what=name))
        seen += len(arrays["m"])
        offset = 0
        for cell, s, e in batch:
            n = e - s
            sl = slice(offset, offset + n)
            offset += n
            xyz = arrays["xyz"][sl]
            m = arrays["m"][sl].astype(np.float64) * f_m
            rho = arrays["rho"][sl].astype(np.float64) * f_rho
            ne = arrays["ne"][sl].astype(np.float64) * f_ne
            T = arrays["T"][sl].astype(np.float64)
            sf = arrays["sfr"][sl] > 0
            yv = arrays["y"][sl].astype(np.float64) * f_y
            vol = m * MSUN_KG / rho                                        # physical m^3
            Ne = ne * vol
            pev = Ne * K_B * T                                             # Pa m^3
            for h in cell_haloes[cell]:
                delta = xyz - centre_com[h]
                delta -= box * np.rint(delta / box)
                d = delta * scale[h]
                x2 = np.einsum("ij,ij->i", d, d)
                inside = x2 <= x100[h] ** 2
                if not inside.any():
                    continue
                x = np.sqrt(x2[inside])
                k = np.minimum(np.searchsorted(edges, x, side="right") - 1, nb - 1)
                mi, Nei, sfi, Ti = m[inside], Ne[inside], sf[inside], T[inside]
                acc["n3"][h] += np.bincount(k, minlength=nb)
                acc["m3"][h] += np.bincount(k, weights=mi, minlength=nb)
                acc["ne3"][h] += np.bincount(k, weights=Nei, minlength=nb)
                acc["ne3_nosf"][h] += np.bincount(k, weights=np.where(sfi, 0.0, Nei), minlength=nb)
                acc["m3_sf"][h] += np.bincount(k, weights=np.where(sfi, mi, 0.0), minlength=nb)
                acc["m3_cold"][h] += np.bincount(k, weights=np.where(Ti < 1e5, mi, 0.0), minlength=nb)
                acc["pe3"][h] += np.bincount(k, weights=pev[inside], minlength=nb)
                acc["y3"][h] += np.bincount(k, weights=yv[inside], minlength=nb)
                in200 = x <= 1.0
                acc["n_r200"][h] += int(in200.sum())
                acc["m_r200"][h] += mi[in200].sum()
                acc["ne_r200"][h] += Nei[in200].sum()
                acc["m_r200_sf"][h] += mi[in200 & sfi].sum()
                acc["y_r200"][h] += yv[inside][in200].sum()
                acc["m_r100"][h] += mi.sum()
                acc["ne_r100"][h] += Nei.sum()
        del arrays
        if (b + 1) % 5 == 0 or b + 1 == len(batches):
            log_line(logp, snapshot=snapshot, message="batches", done=b + 1, total=len(batches), particles=seen,
                     seconds=round(time.monotonic() - started, 1), downloaded_gb=round((counter["bytes"] - start_bytes) / 1e9, 3))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, snapshot=snapshot, redshift=float(sel["redshift"]), scale_factor=a, edges=edges,
                        halo_index=sel["halo_index"], m200c_msun=sel["m200c_msun"], r200_mpc=r200, r100_mpc=r100,
                        **acc)
    tmp.replace(out)
    receipt = {"snapshot": snapshot, "redshift": float(sel["redshift"]), "haloes": n_haloes, "cells": len(needed),
               "particles_read": int(seen), "downloaded_gb": round((counter["bytes"] - start_bytes) / 1e9, 3),
               "seconds": round(time.monotonic() - started, 1),
               "fields": list(fields), "conversion": {"mass_to_msun": f_m, "density_to_kg_m3": f_rho,
                                                       "ne_to_m3": f_ne, "y_to_mpc2": f_y}}
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=1) + "\n")
    log_line(logp, message="done", **{k: v for k, v in receipt.items() if k != "conversion"})
    return {"snapshot": snapshot, "status": "complete", **{k: receipt[k] for k in ("haloes", "downloaded_gb", "seconds")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[78, 68, 58, 48, 38, 18, 14])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cap-gb", type=float, default=6.0)
    args = parser.parse_args()
    (Path(args.root) / "logs").mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_snapshot, s, args.root, args.cap_gb): s for s in args.snapshots}
        for future in as_completed(futures):
            s = futures[future]
            try:
                results.append(future.result())
                print("FINISHED", results[-1], flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(s)
                print("FAILED", s, repr(exc), traceback.format_exc(), flush=True)
    print("ALL DONE", {"completed": results, "failed": failures}, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
