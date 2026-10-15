#!/usr/bin/env python3
"""Audit inputs and estimate resources without downloading catalogues or running PBS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "frb_map_generation/outputs/tng300_lee22_1r200c_20260921"
PROFILE = ROOT / "frb_map_generation/lee2022_frb_dm_profile.jl"
GENERATOR = ROOT / "frb_map_generation/generate_halfdome_z1_dm_mass_windows.jl"
OLD_RUN = ROOT / "frb_map_generation/outputs/zsrc1p0_nside4096_nrays120000_allhalos_sphere1p0_m200c_lee22_noconc_xgpnative_seed42"
RALF = ROOT / "frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM"
PAGE = "https://www.tng-project.org/data/downloads/TNG300-1-Dark/"
API = "https://www.tng-project.org/api/TNG300-1-Dark/"
H_TNG, H_HD = 0.6774, 0.68
# Use the intersection of the support in physical solar masses. Both simulations
# must receive identical physical mass cuts despite their slightly different h.
MASS_MIN = max(1e13 / H_TNG, 1e13 / H_HD)
MASS_MAX = min(10**14.8 / H_TNG, 10**14.8 / H_HD)
WINDOWS = [
    ("supported_low", MASS_MIN, 1e14),
    ("supported_high", 1e14, MASS_MAX),
    ("supported_all", MASS_MIN, MASS_MAX),
]

# Explicit repeatable case selection; the default preserves the calibrated run.
# The requested physical 10^13--10^14 bin needs a small low-mass extrapolation.
CASE = os.environ.get("TNG_LEE22_CASE", "supported")
CALIBRATED_MASS_MIN, CALIBRATED_MASS_MAX = MASS_MIN, MASS_MAX
if CASE == "mass13to14":
    OUTPUT = ROOT / "frb_map_generation/outputs/tng300_lee22_mass13to14_20260921"
    MASS_MIN, MASS_MAX = 1e13, 1e14
    WINDOWS = [("mass13to14", MASS_MIN, MASS_MAX)]
elif CASE != "supported":
    raise ValueError(f"Unknown TNG_LEE22_CASE: {CASE}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_provenance() -> dict[str, str]:
    path = OLD_RUN / "halfdome_uniform_fixedz_foreground_mass_histograms_provenance.txt"
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


class TableRows(HTMLParser):
    """Extract table cells without adding a browser or HTML dependency."""

    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def public_download_plan(page_file: Path | None = None) -> dict:
    if page_file is None:
        response = requests.get(PAGE, timeout=30)
        response.raise_for_status()
        html = response.text
    else:
        html = page_file.read_text(encoding="utf-8-sig")
    table = TableRows()
    table.feed(html)
    snapshots = []
    for row in table.rows:
        if len(row) < 9 or not row[0].isdigit():
            continue
        number, redshift = int(row[0]), float(row[1])
        if number >= 50 and redshift <= 1.0:
            groups = int(row[7].replace(",", ""))
            snapshots.append(dict(snapshot=number, rounded_redshift=redshift,
                                  fof_groups=groups, field_bytes=20 * groups))
    if {s["snapshot"] for s in snapshots} != set(range(50, 100)):
        raise ValueError("Official snapshot table changed: inspect before downloading.")
    return {
        "source": PAGE,
        "fields": ["GroupPos", "Group_M_Crit200", "Group_R_Crit200"],
        "bytes_per_halo": 20,
        "snapshots": snapshots,
        "total_field_GB": sum(s["field_bytes"] for s in snapshots) / 1e9,
        "largest_snapshot_field_GB": max(s["field_bytes"] for s in snapshots) / 1e9,
        "redshift_note": "Table redshifts are rounded; use authenticated snapshot metadata before lightcone construction.",
        "halo_finder": "Official FoF/SUBFIND spherical-overdensity catalogue, not Lee22's Rockstar catalogue.",
    }


def input_audit() -> dict:
    provenance = read_provenance()
    cache = Path(provenance["dm_cache_file"])
    profile_hash = sha256(PROFILE)
    cache_hash = sha256(cache)
    if profile_hash != provenance["dm_profile_implementation_sha256"]:
        raise ValueError("Lee22 source differs from the successful reference run.")
    if cache_hash != provenance["dm_cache_sha256"]:
        raise ValueError("Lee22 interpolation cache differs from the reference run.")
    with h5py.File(ROOT / "lightcone_100.hdf5", "r") as handle:
        schema = {name: {"shape": list(handle[name].shape), "dtype": str(handle[name].dtype)}
                  for name in ("Position", "halo_mass_m200c", "halo_mass_m200m", "redshift")}
    reference = []
    for path in sorted(RALF.glob("DMhalo_r200_mass*.npy")):
        token = path.stem.split("mass", 1)[1]
        lo, hi = (10**float(v) for v in token.split("to"))
        array = np.load(path, mmap_mode="r")
        reference.append(dict(file=str(path), shape=list(array.shape),
                              mass_min_msun=lo, mass_max_msun=hi,
                              wholly_inside_common_support=lo >= MASS_MIN and hi <= MASS_MAX))
    return {
        "profile_sha256": profile_hash, "cache_sha256": cache_hash,
        "cache_bytes": cache.stat().st_size, "cache_path": str(cache),
        "generator_sha256": sha256(GENERATOR), "halfdome_schema": schema,
        "ralf_mass_windows": reference,
        "ralf_strict_comparison_available": any(r["wholly_inside_common_support"] for r in reference),
        "ralf_rebinning_note": "Per-ray sums cannot be split at new halo-mass thresholds; neither histogram subtraction nor DM-value cuts can recover the missing information.",
    }


def resource_estimate(cache_bytes: int, threads=26, chunk_size=250_000) -> dict:
    # These are array-accounting estimates, NOT measurements of process RSS.
    arrays = {
        "thread_dm": 8 * 120_000 * len(WINDOWS) * (threads + 1),
        "merged_dm": 8 * 120_000 * len(WINDOWS),
        "ray_coordinates_and_indices_allowance": 8 * 120_000 * 20,
        "catalogue_chunk_and_temporaries_allowance": 8 * chunk_size * 24,
        "cache_and_interpolation_copies_allowance": cache_bytes * 4,
        "healpix_ring_workspace": (4 * 4096 - 1) * 4 * 8,
    }
    return {
        "threads": threads, "chunk_size": chunk_size,
        "array_bytes": arrays, "array_total_GiB": sum(arrays.values()) / 2**30,
        "planning_peak_GiB": 8,
        "planning_peak_note": "8 GiB provisional allowance including Julia compilation and allocator overhead; verify with /usr/bin/time -v. TNG lightcone peak is not measured.",
        "notify_user_before_cluster_at_GB": 40,
        "full_sky_map_allocated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--public-page", type=Path, help="Previously downloaded official HTML page.")
    args = parser.parse_args()
    audit = input_audit()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_only_no_cluster_submission",
        "simulation": "TNG300-1-Dark", "source_redshift": 1.0,
        "n_rays": 120_000, "seed": 42, "nside": 4096,
        "boundary": "sphere_1R200c", "dm_profile": "lee2022_xgpaint",
        "normalization": "xgpaint_ne2d", "n0_pivot": "mcut", "concentration": "none",
        "windows": [dict(label=k, mass_min_msun=lo, mass_max_msun=hi) for k, lo, hi in WINDOWS],
        "case": CASE,
        "requires_mass_extrapolation": MASS_MIN < CALIBRATED_MASS_MIN or MASS_MAX > CALIBRATED_MASS_MAX,
        "inputs": audit, "memory": resource_estimate(audit["cache_bytes"]),
    }
    if args.public_page or not args.offline:
        report["web_download_plan"] = public_download_plan(args.public_page)
    if not args.offline:
        # Probe without credentials; no data file is requested or saved here.
        response = requests.get(API, timeout=30)
        report["unauthenticated_api_status"] = response.status_code
        report["api_key_environment_present"] = bool(os.environ.get("TNG_API_KEY"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Preflight: {path}")
    print(f"Selected M200c interval: {MASS_MIN:.9g} <= M < {MASS_MAX:.9g} physical Msun")
    print(f"Mass extrapolation required: {report['requires_mass_extrapolation']}")
    print("Profile and cache hashes match the successful 2026-09-17 run.")
    print(f"Accounted arrays: {report['memory']['array_total_GiB']:.3f} GiB; provisional peak allowance: 8 GiB")
    print(f"Ralf strict mass-window reference available: {audit['ralf_strict_comparison_available']}")
    if "web_download_plan" in report:
        plan = report["web_download_plan"]
        print(f"Estimated field download: {plan['total_field_GB']:.3f} GB total; {plan['largest_snapshot_field_GB']:.3f} GB largest snapshot")
    if "unauthenticated_api_status" in report:
        print(f"Unauthenticated API HTTP status: {report['unauthenticated_api_status']}")


if __name__ == "__main__":
    main()
