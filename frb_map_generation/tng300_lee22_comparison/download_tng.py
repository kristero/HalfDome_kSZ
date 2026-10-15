#!/usr/bin/env python3
"""Download three TNG halo fields and retain the configured physical mass range.

Run on Windows if WSL has no DNS. Authentication comes exclusively from the
TNG_API_KEY environment variable. No credential is written to any artifact.
The API supports one group-catalogue field per request, not a comma-separated list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import requests

from preflight import API, H_TNG, MASS_MAX, MASS_MIN, OUTPUT, sha256

FIELDS = ("Group_M_Crit200", "GroupPos", "Group_R_Crit200")


def get_response(session, url, key, **kwargs):
    """Follow official download redirects without forwarding credentials off-host."""
    for _ in range(6):
        headers = {"api-key": key} if urlparse(url).hostname == "www.tng-project.org" else {}
        response = session.get(url, headers=headers, allow_redirects=False, timeout=(30, 180), **kwargs)
        if response.status_code not in (301, 302, 303, 307, 308):
            response.raise_for_status()
            return response
        next_url = urljoin(url, response.headers["Location"])
        response.close()
        if urlparse(next_url).scheme != "https":
            next_url = next_url.replace("http://", "https://", 1)
        url = next_url
        # The server's redirect already contains the complete query.
        kwargs.pop("params", None)
    raise RuntimeError("Too many download redirects")


def download_field(session, key, snapshot, field, target):
    """Stream to disk; cap one field at 512 MB to catch accidental full downloads."""
    url = f"{API}files/groupcat-{snapshot}/"
    response = get_response(session, url, key, params={"Group": field}, stream=True)
    partial = target.with_suffix(".part")
    maximum_bytes = 512_000_000
    if int(response.headers.get("Content-Length", 0)) > maximum_bytes:
        response.close()
        raise RuntimeError("Unexpectedly large field response; no download was started.")
    digest = __import__("hashlib").sha256()
    written = 0
    try:
        with partial.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                written += len(block)
                if written > maximum_bytes:
                    raise RuntimeError("Field response exceeds the audited 512 MB limit.")
                stream.write(block)
                digest.update(block)
        if not h5py.is_hdf5(partial):
            raise ValueError("The API response is not an HDF5 catalogue field.")
        partial.replace(target)
    finally:
        response.close()
        if partial.exists():
            partial.unlink()  # only this script's incomplete download
    return {"field": field, "bytes": written, "sha256": digest.hexdigest(),
            "api_endpoint": url, "query": {"Group": field}}


def dataset(handle, field):
    # API field dumps have used both layouts; require an unambiguous field.
    for name in (f"Group/{field}", field):
        if name in handle:
            return handle[name]
    names = []
    handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    if len(names) == 1 and names[0].split("/")[-1] == field:
        return handle[names[0]]
    raise ValueError(f"Missing {field}; datasets={names}")


def selected_values(path, field, selected, count):
    with h5py.File(path, "r") as handle:
        data = dataset(handle, field)
        if data.shape[0] != count:
            raise ValueError(f"{field} row count differs from snapshot metadata.")
        result = np.empty((len(selected), *data.shape[1:]), dtype=np.float64)
        for start in range(0, count, 250_000):
            stop = min(start + 250_000, count)
            lo, hi = np.searchsorted(selected, [start, stop])
            if hi > lo:
                result[lo:hi] = data[start:stop][selected[lo:hi] - start]
        return result


def selected_field_prefix(session, key, number, field, selected, count, directory):
    """Read exactly the prefix containing the requested rows of a contiguous HDF5 field.

    The mass field is always downloaded in full, so selection completeness does
    not assume mass ordering. Selected massive halos happen to have small IDs.
    TNG's field dumps are contiguous, uncompressed HDF5. Their header gives the
    exact data offset; only the byte prefix through the largest selected ID is
    needed for positions/radii. Unreceived rows in the temporary sparse file are
    NEVER used. Fail if the layout changes. This is not a complete-file checksum.
    """
    url = f"{API}files/groupcat-{number}/"
    response = get_response(session, url, key, params={"Group": field}, stream=True)
    remote_bytes = int(response.headers.get("Content-Length", 0))
    if not 8192 <= remote_bytes <= 512_000_000:
        response.close()
        raise ValueError("Field size is missing or outside the audited range.")
    temporary = directory / f"snapshot_{number:03d}_{field}.prefix.h5"
    digest = __import__("hashlib").sha256()
    received = 0
    blocks = response.iter_content(65536)
    try:
        with temporary.open("w+b") as stream:
            stream.truncate(remote_bytes)  # sparse holes; no full array allocation
            initial = next(blocks)
            stream.write(initial)
            digest.update(initial)
            received += len(initial)
            stream.flush()
            with h5py.File(temporary, "r") as handle:
                data = dataset(handle, field)
                offset = data.id.get_offset()
                if offset is None or data.chunks is not None or data.compression is not None:
                    raise ValueError("API field layout changed; contiguous-prefix extraction is not safe.")
                expected_shape = (count, 3) if field == "GroupPos" else (count,)
                if data.shape != expected_shape or data.dtype != np.dtype("float32"):
                    raise ValueError("API field shape or dtype differs from official metadata.")
                row_bytes = data.dtype.itemsize * int(np.prod(data.shape[1:]))
                needed = offset + (int(selected[-1]) + 1) * row_bytes if len(selected) else offset
            while received < needed:
                block = next(blocks)
                stream.write(block)
                digest.update(block)
                received += len(block)
            stream.flush()
            with h5py.File(temporary, "r") as handle:
                values = dataset(handle, field)[selected].astype(np.float64)
        record = dict(field=field, bytes=received, remote_content_length=remote_bytes,
                      sha256_received_prefix=digest.hexdigest(), data_offset_bytes=offset,
                      last_selected_row=int(selected[-1]) if len(selected) else -1,
                      mode="contiguous_hdf5_verified_prefix_only", api_endpoint=url, query={"Group": field})
        return values, record
    finally:
        response.close()
        if temporary.exists():
            temporary.unlink()


def process_snapshot(session, key, number, destination):
    output = destination / f"snapshot_{number:03d}_selected.h5"
    receipt = output.with_suffix(".json")
    if output.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if sha256(output) != record["selected_sha256"]:
            raise ValueError(f"Checksum mismatch in existing snapshot {number}")
        with h5py.File(output, "r") as handle:
            assert handle.attrs["mass_min_msun"] == MASS_MIN
            assert handle.attrs["mass_max_msun"] == MASS_MAX
        print(f"Snapshot {number}: verified existing {record['selected_halos']} halos", flush=True)
        return record
    if output.exists() or receipt.exists():
        raise FileExistsError(f"Incomplete pre-existing output for snapshot {number}; inspect before retrying.")
    with get_response(session, f"{API}snapshots/{number}/", key) as response:
        meta = response.json()
    count = int(meta["num_groups_fof"])
    redshift = max(float(meta["redshift"]), 0.0)
    records, temporary = [], []
    selected, values = None, {}
    for field in FIELDS:
        if field != "Group_M_Crit200" and number != 99:
            values[field], record = selected_field_prefix(session, key, number, field, selected, count, destination)
            records.append(record)
            continue
        target = destination / f"snapshot_{number:03d}_{field}_download.h5"
        if target.exists() and h5py.is_hdf5(target):
            records.append(dict(field=field, bytes=target.stat().st_size, sha256=sha256(target),
                                api_endpoint=f"{API}files/groupcat-{number}/", query={"Group": field}))
        else:
            records.append(download_field(session, key, number, field, target))
        temporary.append(target)
        if field == "Group_M_Crit200":
            with h5py.File(target, "r") as handle:
                data = dataset(handle, field)
                if data.shape != (count,):
                    raise ValueError("Mass field shape differs from snapshot metadata.")
                mass = data[:].astype(np.float64) * 1e10 / H_TNG
            selected = np.flatnonzero((mass >= MASS_MIN) & (mass < MASS_MAX))
            values["mass_msun"] = mass[selected]
            del mass
        else:
            # Snapshot 99 is deliberately a complete-download control, so the
            # prefix reader can be checked against an independent extraction.
            values[field] = selected_values(target, field, selected, count)
    positions = values["GroupPos"] / (1000 * H_TNG)
    radius = values["Group_R_Crit200"] / (1000 * H_TNG * (1 + redshift))
    # Validate TNG's mass/radius units independently through the SO definition.
    rho_critical = 2.77536627e11 * H_TNG**2 * (0.3089 * (1 + redshift)**3 + 0.6911)
    radius_from_mass = (3 * values["mass_msun"] / (4 * np.pi * 200 * rho_critical))**(1 / 3)
    fractional_radius_error = np.max(np.abs(radius / radius_from_mass - 1))
    if not np.all(np.isfinite(positions)) or not np.all(radius > 0) or fractional_radius_error > 0.01:
        raise ValueError("TNG SO mass/radius validation failed; inspect unit conventions.")
    partial = output.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as handle:
        handle["halo_id"] = selected.astype(np.int64)
        handle["position_comoving_mpc"] = positions
        handle["mass_msun"] = values["mass_msun"]
        handle["r200c_physical_mpc"] = radius
        handle.attrs.update(simulation="TNG300-1-Dark", snapshot=number, redshift=redshift,
                            little_h=H_TNG, omega_m=0.3089, omega_b=0.0486,
                            box_comoving_mpc=205 / H_TNG, total_fof_groups=count,
                            mass_definition="M200c", mass_min_msun=MASS_MIN, mass_max_msun=MASS_MAX,
                            halo_finder="FoF/SUBFIND", so_radius_max_fractional_error=fractional_radius_error)
    partial.replace(output)
    record = dict(snapshot=number, redshift=redshift, total_fof_groups=count,
                  selected_halos=len(selected), selected_file=output.name,
                  selected_sha256=sha256(output), selected_bytes=output.stat().st_size,
                  so_radius_max_fractional_error=float(fractional_radius_error), downloads=records)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    # Reopen and validate the compact product before deleting redundant raw fields.
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["halo_id"][:], selected)
        np.testing.assert_allclose(handle["mass_msun"][:], values["mass_msun"], rtol=0, atol=0)
    for path in temporary:
        path.unlink()
    # Remove raw field files left by an earlier interrupted full-field download.
    # Names are restricted to this snapshot and this script's own staging files.
    for field in FIELDS:
        for suffix in ("_download.h5", "_download.part"):
            staged = destination / f"snapshot_{number:03d}_{field}{suffix}"
            if staged.exists():
                staged.unlink()
    print(f"Snapshot {number}: z={redshift:.6f}, {len(selected)} selected halos, "
          f"SO radius error={fractional_radius_error:.2g}; raw fields removed", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", default="50:99", help="Inclusive range, or comma-separated numbers.")
    parser.add_argument("--workers", type=int, default=3, help="At most three independent snapshot downloads.")
    parser.add_argument("--api-key-stdin", action="store_true",
                        help="Read one API-key line from stdin instead of the environment.")
    parser.add_argument("--verify-prefix", action="store_true", help="Compare compact reads with the fully downloaded snapshot 99 control.")
    args = parser.parse_args()
    if ":" in args.snapshots:
        low, high = map(int, args.snapshots.split(":"))
        numbers = list(range(low, high + 1))
    else:
        numbers = list(map(int, args.snapshots.split(",")))
    if not numbers or any(n < 50 or n > 99 for n in numbers):
        parser.error("This z_source=1 workflow uses snapshots 50 through 99 only.")
    if not 1 <= args.workers <= 3:
        parser.error("Use 1-3 download workers to bound memory and server load.")
    key = sys.stdin.readline().strip() if args.api_key_stdin else os.environ.get("TNG_API_KEY", "").strip()
    if not key:
        parser.error("Set TNG_API_KEY in the process environment; do not put it in source code.")
    destination = OUTPUT / "tng_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    if args.verify_prefix:
        checks = []
        receipt = json.loads((destination / "snapshot_099_selected.json").read_text())
        if any("sha256" not in row for row in receipt["downloads"]):
            raise ValueError("Snapshot 99 must be a fully downloaded control for this check.")
        with h5py.File(destination / "snapshot_099_selected.h5", "r") as reference, requests.Session() as session:
            selected = reference["halo_id"][:]
            radius_factor = 1000 * H_TNG * (1 + float(reference.attrs["redshift"]))
            for field, name, factor in (("GroupPos", "position_comoving_mpc", 1000 * H_TNG),
                                         ("Group_R_Crit200", "r200c_physical_mpc", radius_factor)):
                values, record = selected_field_prefix(session, key, 99, field, selected, 13008511, destination)
                np.testing.assert_allclose(values / factor, reference[name][:], rtol=0, atol=0)
                checks.append(dict(field=field, exact_match=True, transferred_bytes=record["bytes"],
                                   full_field_bytes=record["remote_content_length"]))
        (OUTPUT / "download_prefix_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(checks, indent=2), flush=True)
        return
    def work(number):
        with requests.Session() as session:
            for attempt in range(3):
                try:
                    process_snapshot(session, key, number, destination)
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 2:
                        raise
                    print(f"Snapshot {number}: network retry {attempt + 1}", flush=True)
                    time.sleep(3)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(work, numbers))
    print("Requested snapshot downloads and compact-catalogue checks complete.", flush=True)


if __name__ == "__main__":
    main()
