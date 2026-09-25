#!/usr/bin/env python3
"""Inspect FLAMINGO file metadata and budget a HalfDome tSZ comparison.

This program has no simulation-download or job-submission operation. Remote
HDF5 requests explicitly suppress dataset bodies, and every HTTP response is
capped at 1 MiB. Run it on the cluster before requesting download approval.
Requires requests, msgpack, and numpy (already in the cluster SBI environment).
"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import time
from urllib.parse import quote

import msgpack
import numpy as np
import requests


SERVER = "https://dataweb.cosma.dur.ac.uk:8443"
GIB = 1024 ** 3
MAX_METADATA_BYTES = 1024 ** 2
MAP_NAME = "lensed_tSZ_rot_same_rot.hdf5"
MAP_DIRECTORY = "integrated_maps/yang26/lightcone0_shells"
VARIANTS = (
    ("L1_m9", "Fiducial thermal AGN feedback"),
    ("fgas+2sigma", "Cluster gas fraction calibration shifted +2 sigma"),
    ("fgas-2sigma", "Cluster gas fraction calibration shifted -2 sigma"),
    ("fgas-4sigma", "Cluster gas fraction calibration shifted -4 sigma"),
    ("fgas-8sigma", "Cluster gas fraction calibration shifted -8 sigma"),
    ("Mstar-1sigma", "Stellar mass calibration shifted -1 sigma"),
    ("Mstar-1sigma_fgas-4sigma", "Stellar mass -1 sigma and gas fraction -4 sigma"),
    ("Jet", "Jet AGN feedback"),
    ("Jet_fgas-4sigma", "Jet AGN feedback with gas fraction -4 sigma"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_dataset_bodies(value):
    """Attributes are metadata; nonempty HDF5 dataset bodies are forbidden."""
    if isinstance(value, dict):
        if value.get("hdf5_object") == "dataset" and value.get("data") is not None:
            raise ValueError("Server returned a dataset body in a metadata-only request")
        for child in value.values():
            reject_dataset_bodies(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_dataset_bodies(child)


class MetadataClient:
    def __init__(self):
        self.session = requests.Session()
        self.receipts = []

    def get(self, path, hdf5_root=False):
        # The live viewer uses data_size_limit. The prose API page incorrectly
        # calls this max_data_size; that spelling DOES NOT suppress map data.
        params = {"data_size_limit": 0, "max_depth": 1}
        if hdf5_root:
            params["object"] = "/"
        url = SERVER + "/hdfstream/msgpack/" + quote(path.strip("/"), safe="/")
        receipt = {"url": url, "params": params, "received_bytes": 0}
        self.receipts.append(receipt)
        started = time.monotonic()
        with self.session.get(url, params=params, timeout=(15, 30), stream=True,
                              allow_redirects=False) as response:
            receipt["http_status"] = response.status_code
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Expected HTTP 200 without redirects")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_METADATA_BYTES:
                raise ValueError("Refused oversized metadata Content-Length")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                receipt["received_bytes"] += len(chunk)
                if receipt["received_bytes"] > MAX_METADATA_BYTES:
                    raise ValueError("Stopped oversized metadata response")
                if time.monotonic() - started > 60:
                    raise TimeoutError("Metadata request exceeded 60 seconds")
                body.extend(chunk)
        result = msgpack.unpackb(body, raw=False)
        reject_dataset_bodies(result)
        receipt["sha256"] = hashlib.sha256(body).hexdigest()
        return result


def inspect_variant(client, name, description):
    directory = "FLAMINGO/L1_m9/{}/{}".format(name, MAP_DIRECTORY)
    record = {"variant": name, "description": description,
              "remote_path": directory + "/" + MAP_NAME}
    try:
        listing = client.get(directory)
        file_info = listing["files"][MAP_NAME]
        root = client.get(record["remote_path"], hdf5_root=True)
        dataset = root["members"]["data"]
        shape = dataset["shape"]
        dtype = np.dtype(dataset["type"])
        nside = math.isqrt(shape[0] // 12) if len(shape) == 1 else 0
        if shape != [12 * nside ** 2] or nside != 4096 or dtype != np.dtype("<f8"):
            raise ValueError("Unexpected map shape, NSIDE, or dtype: {} {}".format(shape, dtype))
        record.update(status="available", file_bytes=int(file_info["size"]),
                      last_modified_ms=file_info.get("last_modified"),
                      dataset="data", shape=shape, dtype=dtype.str, nside=nside,
                      uncompressed_array_bytes=shape[0] * dtype.itemsize,
                      dataset_attributes=dataset.get("attributes", {}),
                      map_convention_source="FLAMINGO integrated-lightcones documentation",
                      download_url=SERVER + "/hdfstream/download/" + quote(record["remote_path"], safe="/"))
    except (requests.RequestException, ValueError, KeyError, TimeoutError) as error:
        record.update(status="unavailable_or_unverified", error=str(error))
    return record


def inspect_bundle(path):
    """Check existing files and small arrays; never deserialize model pickles."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("method") != "moped":
        raise ValueError("Expected the exported MOPED bundle")
    for filename, expected in manifest["files"].items():
        if Path(filename).name != filename or sha256(path / filename) != expected:
            raise ValueError("Missing or changed model bundle member: " + filename)
    training = json.loads((path / "training_complete.json").read_text())
    if training["weights_selection"] != "best_validation_snapshot":
        raise ValueError("Not the selected best-validation checkpoint")
    with np.load(path / "observation_contract.npz", allow_pickle=False) as contract:
        experiment_id = str(contract["experiment_id"].item())
        if experiment_id != manifest["experiment_id"] or experiment_id != training["experiment_id"]:
            raise ValueError("Inconsistent model experiment IDs")
        metadata = json.loads(str(contract["metadata_json"].item()))
        if metadata["product"] != "masked_baseline_noise_cross_deproj0":
            raise ValueError("Unexpected trained noise product")
        if len(contract["bin_ell_min"]) != 40:
            raise ValueError("Not the selected 40-bin model")
        result = dict(path=str(path), experiment_id=experiment_id,
                      files=manifest["files"], checksum_verification="passed",
                      param_names=contract["param_names"].tolist(),
                      prior_low=contract["low"].tolist(), prior_high=contract["high"].tolist(),
                      bin_ell_min=contract["bin_ell_min"].tolist(),
                      bin_ell_max=contract["bin_ell_max"].tolist(),
                      metadata=metadata, training=training)
    return result


def resource_budget(records):
    available = [record for record in records if record["status"] == "available"]
    total = sum(record["file_bytes"] for record in available)
    map_bytes = 12 * 4096 ** 2 * 8
    alm_bytes = (7979 + 1) * (7979 + 2) // 2 * 16
    # A planning envelope, not a measured peak. Ten maps covers signal, mask,
    # two noise maps, two noisy splits, clean masked signal and temporary copies.
    array_budget = 10 * map_bytes + 4 * alm_bytes
    return dict(
        available_maps=len(available), requested_maps=len(records),
        download_bytes=total, download_GB=total / 1e9, download_GiB=total / GIB,
        float64_map_bytes=map_bytes, float64_map_GiB=map_bytes / GIB,
        complex128_alm_bytes_at_lmax7979=alm_bytes,
        planned_array_allowance_GiB=array_budget / GIB,
        expected_process_peak_GiB_range=[24, 32], ram_estimate_measured=False,
        proposed_PBS_memory_gb=64, proposed_PBS_ncpus=26,
        proposed_PBS_queue="mini", proposed_PBS_walltime="23:59:00",
        simultaneous_map_jobs=1, planned_noise_and_mask_cache_GiB=3 * map_bytes / GIB,
        proposed_workspace_reservation_GiB=20 if len(available) <= 3 else 30,
        disk_budget_includes="Available raw inputs, 4.5 GiB reusable mask/noise cache, one optional working map, compact spectra/posteriors/plots, and margin",
        notes="Process feedback variants sequentially. Retain raw inputs; do not persist every beam/noise/split map. Measure peak RSS in the first approved map job.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--halfdome-observation", type=Path)
    args = parser.parse_args()
    client = MetadataClient()
    records = []
    for name, description in VARIANTS:
        record = inspect_variant(client, name, description)
        records.append(record)
        print("{}: {} ({} bytes)".format(name, record["status"], record.get("file_bytes", "?")), flush=True)
    report = dict(schema_version=1, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=socket.gethostname(), approval_status="awaiting_user_download_approval",
                  simulation_downloads_performed_by_this_script=0, script_sha256=sha256(__file__),
                  variants=records, budget=resource_budget(records), metadata_requests=client.receipts,
                  observation_contract=dict(nside=4096, signal_beam_fwhm_arcmin=2.0,
                      mask_support_fsky=0.4, mask_apodization_arcmin=60.0,
                      mask_seed=12345, noise_root_seed=12345,
                      baseline_deproj0_split_seeds=[22446, 22447],
                      noise_and_spectrum_lmax=7979, cl_niter=0,
                      spectrum="raw masked split cross pseudo-Cl, no fsky or mode-coupling correction",
                      noise="SO baseline deproj0, native Compton-y N_ell; no extra beam on noise",
                      inference="saved signed 40-bin D_ell, asinh, standardization, MOPED and nine-parameter SBI; original prior"))
    if args.bundle:
        report["verified_bundle"] = inspect_bundle(args.bundle)
    if args.halfdome_observation:
        root = args.halfdome_observation
        files = ("simulation_complete.json", "observation.npz", "posterior_samples.npy")
        report["halfdome_reference"] = dict(path=str(root), files={name: sha256(root / name) for name in files})
        simulation = json.loads((root / "simulation_complete.json").read_text())
        report["halfdome_reference"]["simulation_request"] = simulation["request"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.output.parent)
    report["filesystem"] = dict(path=str(args.output.parent.resolve()), free_bytes=usage.free,
                                note="Filesystem free space is not a user-quota check")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["budget"], indent=2), flush=True)
    print("Wrote {}. No simulation download or qsub was performed.".format(args.output), flush=True)
    if not any(record["status"] == "available" for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
