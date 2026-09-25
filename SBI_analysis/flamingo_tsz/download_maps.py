#!/usr/bin/env python3
"""Download and validate exactly the three user-approved FLAMINGO y maps."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit

import h5py
import numpy as np
import requests

from preflight import sha256

APPROVED = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")
TOTAL_BYTES = 4556709696


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate(path, record):
    if path.stat().st_size != record["file_bytes"]:
        raise ValueError("Wrong file size: " + str(path))
    total, square_sum, minimum, maximum = 0.0, 0.0, np.inf, -np.inf
    with h5py.File(path, "r") as handle:
        dataset = handle["data"]
        # hdfstream advertises native byte order, while the actual HDF5 maps
        # store big-endian Float64. Byte order changes representation, not y.
        if list(dataset.shape) != record["shape"] or dataset.dtype.kind != "f" or dataset.dtype.itemsize != 8:
            raise ValueError("Map dtype or shape changed")
        for start in range(0, len(dataset), 1048576):
            values = np.asarray(dataset[start:start + 1048576], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite Compton-y pixels")
            total += values.sum(dtype=np.float64)
            square_sum += np.dot(values, values)
            minimum, maximum = min(minimum, values.min()), max(maximum, values.max())
        count, stored_dtype = len(dataset), dataset.dtype.str
    return dict(sha256=sha256(path), bytes=path.stat().st_size, npix=count,
                minimum=float(minimum), maximum=float(maximum), mean=total / count,
                rms=float(np.sqrt(square_sum / count)), all_pixels_finite=True,
                stored_dtype=stored_dtype, metadata_dtype=record["dtype"])


def download(record, destination):
    url = record["download_url"]
    parsed = urlsplit(url)
    if parsed.netloc != "dataweb.cosma.dur.ac.uk:8443" or not parsed.path.startswith("/hdfstream/download/FLAMINGO/L1_m9/"):
        raise ValueError("Download URL outside approved source")
    part = destination.with_suffix(".hdf5.part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > record["file_bytes"]:
        raise ValueError("Oversized partial file; inspect before retrying")
    if offset == record["file_bytes"]:
        validate(part, record)
        part.replace(destination)
        return
    headers = {"Range": "bytes={}-".format(offset)} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code not in (200, 206):
            raise ValueError("Unexpected download response")
        if response.status_code == 206:
            expected_range = "bytes {}-".format(offset)
            if not response.headers.get("Content-Range", "").startswith(expected_range):
                raise ValueError("Incorrect resume range")
        elif offset:
            offset = 0  # Server ignored Range; restart this incomplete file.
        written, last_print = offset, time.monotonic()
        with part.open("ab" if offset else "wb") as stream:
            for block in response.iter_content(chunk_size=1048576):
                if written + len(block) > record["file_bytes"]:
                    raise ValueError("Transfer exceeds the approved file size")
                stream.write(block)
                written += len(block)
                if time.monotonic() - last_print > 30:
                    print("{}: {:.1%}".format(record["variant"], written / record["file_bytes"]), flush=True)
                    last_print = time.monotonic()
    if written != record["file_bytes"]:
        raise ValueError("Incomplete download; resume on the next invocation")
    part.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    approval = json.loads((root / "approval.json").read_text())
    manifest_path = root / "preflight/metadata_manifest.json"
    if approval["download_approved"] is not True or approval["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Missing approval or changed approved manifest")
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest["variants"] if r["status"] == "available"]
    if tuple(r["variant"] for r in records) != APPROVED or sum(r["file_bytes"] for r in records) != TOTAL_BYTES:
        raise ValueError("Manifest does not match the three-map approval")
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 20 * 1024 ** 3:
        raise ValueError("Less than the approved workspace allowance is free")
    results = {}
    for record in records:
        destination = inputs / (record["variant"] + ".hdf5")
        if not destination.exists():
            print("Downloading " + record["variant"], flush=True)
            download(record, destination)
        results[record["variant"]] = dict(path=str(destination), url=record["download_url"],
                                         **validate(destination, record))
        save_json(inputs / "download_progress.json", results)
        print("Validated " + record["variant"], flush=True)
    save_json(inputs / "download_complete.json", dict(
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=sha256(manifest_path), files=results, total_file_bytes=TOTAL_BYTES))


if __name__ == "__main__":
    main()
