#!/usr/bin/env python3
"""Build a continuous-direction, periodically tiled TNG halo lightcone to z=1.

Use all 50 snapshots at z<=1, with boundaries halfway in comoving distance.
The observer and box orientation stay fixed across shells, preserving spatial
continuity. Snapshot halo properties are not interpolated in time. This is a
halo-centre lightcone, not a reconstruction of Ralf's Voronoi gas-cell rays.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from preflight import H_TNG, MASS_MAX, MASS_MIN, OUTPUT, sha256


def distance_interpolators():
    redshift = np.linspace(0, 1.05, 50_001)
    expansion = np.sqrt(0.3089 * (1 + redshift)**3 + 0.6911)
    distance = cumulative_trapezoid(299792.458 / (100 * H_TNG * expansion), redshift, initial=0)
    return PchipInterpolator(redshift, distance), PchipInterpolator(distance, redshift)


def intersecting_tiles(box, observer, inner, outer):
    """Yield boxes whose axis-aligned bounding volume intersects the shell."""
    low = np.floor((observer - outer) / box).astype(int)
    high = np.floor((observer + outer) / box).astype(int)
    for tile in itertools.product(*(range(a, b + 1) for a, b in zip(low, high))):
        corner = np.asarray(tile) * box - observer
        end = corner + box
        nearest = np.maximum(np.maximum(corner, -end), 0)
        farthest = np.maximum(np.abs(corner), np.abs(end))
        if nearest @ nearest >= outer**2 or farthest @ farthest < inner**2:
            continue
        yield np.asarray(tile, dtype=np.int16)


def main():
    destination = OUTPUT / "tng_lightcone.h5"
    if destination.exists():
        raise FileExistsError(f"Preserving existing lightcone: {destination}")
    directory = OUTPUT / "tng_snapshots"
    snapshots = []
    for number in range(50, 100):
        path = directory / f"snapshot_{number:03d}_selected.h5"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != receipt["selected_sha256"]:
            raise ValueError(f"Snapshot {number} checksum mismatch")
        snapshots.append(receipt)
    snapshots.sort(key=lambda record: record["redshift"])
    chi_of_z, z_of_chi = distance_interpolators()
    chi = chi_of_z([s["redshift"] for s in snapshots])
    chi_source = float(chi_of_z(1.0))
    edges = np.r_[0.0, (chi[:-1] + chi[1:]) / 2, chi_source]
    assert np.all(np.diff(edges) > 0)
    box = 205.0 / H_TNG
    observer = np.random.default_rng(42).uniform(0, box, size=3)
    partial = destination.with_suffix(".partial.h5")
    if partial.exists():
        raise FileExistsError("Incomplete lightcone exists; inspect before restarting.")
    shell_records, total = [], 0
    with h5py.File(partial, "w") as output:
        specifications = {
            "position_comoving_mpc": ("f8", (3,)), "mass_msun": ("f8", ()),
            "redshift": ("f8", ()), "snapshot": ("u1", ()),
            "halo_id": ("i8", ()), "periodic_tile": ("i2", (3,)),
        }
        arrays = {name: output.create_dataset(name, shape=(0, *tail), maxshape=(None, *tail),
                                              dtype=dtype, chunks=(65_536, *tail),
                                              compression="gzip", compression_opts=1, shuffle=True)
                  for name, (dtype, tail) in specifications.items()}
        for index, record in enumerate(snapshots):
            with h5py.File(directory / record["selected_file"], "r") as source:
                position = source["position_comoving_mpc"][:]
                mass = source["mass_msun"][:]
                ids = source["halo_id"][:]
            if not np.all((mass >= MASS_MIN) & (mass < MASS_MAX)):
                raise ValueError("Downloaded selection extends outside the configured mass interval")
            lower, upper = edges[index:index + 2]
            count = 0
            buffers = {name: [] for name in arrays}
            buffered = 0

            def flush():
                nonlocal total, buffered
                if not buffered:
                    return
                for name, array in arrays.items():
                    array.resize(total + buffered, axis=0)
                    array[total:total + buffered] = np.concatenate(buffers[name], axis=0)
                    buffers[name].clear()
                total += buffered
                buffered = 0

            for tile in intersecting_tiles(box, observer, lower, upper):
                translated = position + tile.astype(float) * box - observer
                radius2 = np.einsum("ij,ij->i", translated, translated)
                keep = (radius2 >= lower**2) & (radius2 < upper**2) & (radius2 > 0)
                size = int(keep.sum())
                if not size:
                    continue
                buffers["position_comoving_mpc"].append(translated[keep])
                buffers["mass_msun"].append(mass[keep])
                buffers["redshift"].append(z_of_chi(np.sqrt(radius2[keep])))
                buffers["snapshot"].append(np.full(size, record["snapshot"], dtype=np.uint8))
                buffers["halo_id"].append(ids[keep])
                buffers["periodic_tile"].append(np.broadcast_to(tile, (size, 3)))
                buffered += size
                count += size
                if buffered >= 100_000:
                    flush()
            flush()
            expected = len(mass) / box**3 * 4 * np.pi / 3 * (upper**3 - lower**3)
            shell_records.append(dict(snapshot=record["snapshot"], snapshot_redshift=record["redshift"],
                                      chi_min_mpc=float(lower), chi_max_mpc=float(upper),
                                      halos=count, uniform_volume_expectation=float(expected)))
            print(f"Snapshot {record['snapshot']}: {lower:.1f}-{upper:.1f} Mpc, {count:,} halos", flush=True)
        output.attrs.update(simulation="TNG300-1-Dark", little_h=H_TNG, omega_m=0.3089,
                            omega_b=0.0486, box_comoving_mpc=box, source_redshift=1.0,
                            observer_comoving_mpc=observer, observer_seed=42, halo_finder="FoF/SUBFIND",
                            mass_min_msun=MASS_MIN, mass_max_msun=MASS_MAX,
                            shell_policy="nearest snapshot in comoving distance; fixed periodic orientation and observer",
                            radial_selection="halo centres; full sphere assigned to each selected halo",
                            profile_redshift="continuous lightcone redshift inferred from comoving distance",
                            complete=True, halo_count=total)
    partial.replace(destination)
    report = dict(lightcone_sha256=sha256(destination), halo_count=total,
                  disk_bytes=destination.stat().st_size, observer_comoving_mpc=observer.tolist(),
                  shells=shell_records, snapshot_receipts=snapshots)
    (OUTPUT / "lightcone_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Complete lightcone: {total:,} halos; {destination}", flush=True)


if __name__ == "__main__":
    main()
