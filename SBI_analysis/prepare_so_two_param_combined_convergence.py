#!/usr/bin/env python3
"""Prepare the fixed-noise combined 32k files using their EXACT generating CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_so_two_param_combined_spectra import binned_dell, discover, inspect_layout
from so_sbi_compression import FIDUCIAL, PARAM_NAMES, save_npz, write_json


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path,
                   default=Path("/lustre/work/kristero10/two_param_P0_beta_32k/y100"))
    p.add_argument("--sobol-csv", type=Path, required=True)
    p.add_argument("--prior-low", nargs=2, type=float, required=True, metavar=("P0", "BETA"))
    p.add_argument("--prior-high", nargs=2, type=float, required=True, metavar=("P0", "BETA"))
    p.add_argument("--allow-csv-prefix", action="store_true",
                   help="Only after confirming that the simulations used this CSV prefix")
    p.add_argument("--expected-csv-sha256", help="Reject incomplete transfers or a different design")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--validation-report", type=Path)
    p.add_argument("--holdout-last-n", type=int, default=1000)
    p.add_argument("--battaglia-raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and not args.validate_only:
        raise FileExistsError(f"Refusing to overwrite prepared data: {args.output}")
    csv_hash = digest(args.sobol_csv)
    if args.expected_csv_sha256 and csv_hash != args.expected_csv_sha256.lower():
        raise ValueError(f"Generating CSV checksum mismatch: {csv_hash}")
    inspect_layout(args.data / "combined_cl_layout.txt")
    records = discover(args.data)
    if [r["row"] for r in records] != list(range(1, 32769)):
        raise ValueError("Require all 32768 unique rows in the split/local identity mapping")
    if {r["seed_tag"] for r in records} != {12345}:
        raise ValueError("This preparation explicitly expects the fixed-seed diagnostic dataset")
    with args.sobol_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if set(PARAM_NAMES) - set(reader.fieldnames or []):
            raise ValueError("Generating CSV must contain all nine named pressure parameters")
        full = np.array([[float(row[k]) for k in PARAM_NAMES] for row in reader])
    if len(full) != len(records):
        if not args.allow_csv_prefix or len(full) < len(records):
            raise ValueError("CSV row count differs; no prefix assumption is made automatically")
        full = full[:len(records)]
    fixed = [1, 3, 4, 5, 6, 7, 8]
    np.testing.assert_allclose(full[:, fixed], np.broadcast_to(FIDUCIAL[fixed], (len(full), 7)),
                               rtol=2e-6, atol=1e-8)
    low, high = np.array(args.prior_low), np.array(args.prior_high)
    if (not np.isfinite(full).all() or not np.isfinite([low, high]).all()
            or not np.all(high > low)):
        raise ValueError("Invalid parameters or prior")
    # The seed tag, native multipoles and product must match the training statistic.
    name = args.battaglia_raw.name
    for tag in ("masked_baseline_noise_cross_cl", "_seed12345_", "_deproj0_", "_lmax7979"):
        if tag not in name:
            raise ValueError(f"Battaglia12 raw profile lacks expected tag {tag}")
    observed_cl = np.load(args.battaglia_raw, allow_pickle=False)
    if observed_cl.shape != (7980,) or not np.isfinite(observed_cl).all():
        raise ValueError("Expected raw Battaglia12 C_ell vector, ell=0..7979")
    ell, obs = binned_dell(observed_cl)
    if not 3 <= args.holdout_last_n < len(full) - 128:
        raise ValueError("Invalid held-out count")
    supported = np.all((full[:, [0, 2]].astype(np.float32) >= low)
                       & (full[:, [0, 2]].astype(np.float32) <= high), axis=1)
    validation = dict(n_rows=len(full), csv_sha256=csv_hash,
        csv_prefix_explicitly_allowed=args.allow_csv_prefix,
        training_pool=int(supported[:-args.holdout_last_n].sum()),
        n_test=int(supported[-args.holdout_last_n:].sum()),
        n_excluded=int((~supported).sum()), prior_low=low.tolist(), prior_high=high.tolist(),
        observation_sha256=digest(args.battaglia_raw),
        limitation="Row labels checked; CSV identity still depends on generation provenance")
    if args.validation_report:
        write_json(args.validation_report, validation)
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    x = np.empty((len(records), 40), dtype=np.float32)
    clean = np.empty_like(x)
    for i, record in enumerate(records):
        spectra = np.load(record["path"], mmap_mode="r", allow_pickle=False)
        if spectra.shape != (6, 7980) or not np.isfinite(spectra[[1, 2]]).all():
            raise ValueError(f"Invalid spectrum: {record['path']}")
        _, binned = binned_dell(spectra[[1, 2]])
        clean[i], x[i] = binned
        if (i+1) % 1024 == 0:
            print(f"Binned {i+1}/{len(records)}", flush=True)
    native = np.arange(80, 7980)
    groups = native//200
    bmin = np.array([native[groups == g].min() for g in np.unique(groups)])
    bmax = np.array([native[groups == g].max() for g in np.unique(groups)])
    meta = dict(complete=True, statistic="weighted mean of linear D_ell; x equals binned D_ell",
        bin_weighting="2ell_plus_1", independent_noise_all_rows=False,
        same_mask_all_rows=True, beam_applied_to_signal=True, beam_fwhm_arcmin=2.0,
        noise_policy="fixed_noise_diagnostic", source_csv=str(args.sobol_csv),
        source_csv_sha256=csv_hash, csv_prefix_explicitly_allowed=args.allow_csv_prefix,
        source_data=str(args.data), split_rows=128, product="masked_baseline_noise_cross_deproj0",
        theta_mapping="CSV row (split-1)*128+local; checked against rowNNNNN folder",
        prior_source="explicit user-supplied inference bounds; never inferred from extrema",
        observation_source=str(args.battaglia_raw), observation_sha256=digest(args.battaglia_raw))
    save_npz(args.output, theta=full[:, [0, 2]].astype(np.float32),
        theta_full=full.astype(np.float32), x=x, x_no_noise=clean,
        param_names=np.array(["P0", "beta"]), full_param_names=np.array(PARAM_NAMES),
        prior_low=low, prior_high=high, ell_binned=ell, ell_unbinned=native,
        bin_ell_min=bmin, bin_ell_max=bmax, sobol_global_row=np.arange(1, len(full)+1),
        mask_seed=np.full(len(full), 12345), noise_seed=np.full(len(full), 12345),
        obs=obs.astype(np.float32), obs_theta=FIDUCIAL[[0, 2]],
        obs_theta_full=FIDUCIAL, obs_source=np.asarray("Battaglia12 fixed-noise diagnostic"),
        obs_noise_seed=np.asarray(12345), obs_mask_seed=np.asarray(12345),
        product=np.asarray("masked_baseline_noise_cross_deproj0"),
        metadata_json=np.asarray(json.dumps(meta)))
    write_json(args.output.with_suffix(".json"), meta)
    print("Prepared:", args.output)


if __name__ == "__main__":
    main()
