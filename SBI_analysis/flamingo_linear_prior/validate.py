"""Scientific preflight gate; no dataset job can bypass missing full maps."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import toml

from prior import digest, write_json
from worker import BINNED_NAMES, SPECTRA, bin_cl, verify_frozen
from noise_seeds import ALGORITHM, PREFLIGHT_OFFSET, split_seeds


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify_frozen(root)
    source_hashes = {digest(root/"manifest.json")}
    run = json.loads((root/"run_config.json").read_text())
    cases = json.loads((root/"preflight/cases.json").read_text())
    cache = toml.load(str(Path(run["campaign"])/"cache/complete.toml"))
    seed_check = json.loads((root/"noise_seed_check.json").read_text())
    assert seed_check["passed"]
    assert seed_check["seed_code_sha256"] == digest(root/"code/noise_seeds.py")
    assert seed_check["design_sha256"] == digest(root/"design/noise_split_seeds.npy")
    noise_hashes = set()
    references = {"Battaglia12": Path(run["campaign"])/"results/HalfDome",
        "L1_m9": Path(run["pilot"])/"results/fit_L1_m9_iter2",
        "fgas-8sigma": Path(run["pilot"])/"results/fit_fgas-8sigma_iter3",
        "Mstar-1sigma": Path(run["pilot"])/"results/fit_Mstar-1sigma_iter4"}
    report = dict(passed=False,manifest_sha256=digest(root/"manifest.json"),cases=[])
    for i,case in enumerate(cases):
        with h5py.File(root/"preflight"/(str(i//2)+".h5"),"r") as handle:
            row = handle[str(i)]
            np.testing.assert_array_equal(row["theta"][:],case["theta"])
            assert row.attrs["manifest_sha256"] in source_hashes
            status = json.loads(row.attrs["status_json"])
            assert status["exit_code"] == 0
            assert status["peak_rss_GiB"] < 60
            assert status["numerics"]["maximum_column_relative_error"] < 1e-7
            noise = status["noise"]
            assert noise["mode"] == run["training_noise_mode"] == "independent_per_row"
            assert noise["seed_algorithm"] == ALGORITHM
            assert noise["split_seeds"] == split_seeds(PREFLIGHT_OFFSET+i, run["noise_master_seed"])
            assert noise["seed_row_id"] == PREFLIGHT_OFFSET+i
            assert noise["mask_pixel_sha256"] == cache["mask_pixel_sha256"]
            assert noise["noise_table_sha256"] == cache["noise_table_sha256"]
            assert noise["noise_lmax"] == 7979 and not noise["noise_beam_applied"]
            assert noise["split_Nell_multiplier"] == 1.0
            for key in ("noise1_pixel_sha256", "noise2_pixel_sha256"):
                sha = noise[key]
                assert sha not in noise_hashes
                assert sha not in (cache["noise1_pixel_sha256"],cache["noise2_pixel_sha256"])
                assert sha == status["operator"][key]
                noise_hashes.add(sha)
            if i == 0:
                assert noise["checks"]["full_resolution_replay_passed"]
                assert max(map(abs,noise["checks"]["harmonic_ensemble"]["standardized_errors"])) < 8
            for name in SPECTRA:
                assert np.isfinite(row[name][:]).all()
                np.testing.assert_array_equal(row[BINNED_NAMES[name]][:],bin_cl(row[name][:]))
            entry = dict(name=case["name"],elapsed_seconds=status["elapsed_seconds"],
                         peak_rss_GiB=status["peak_rss_GiB"],numerics=status["numerics"],
                         generation_manifest_sha256=row.attrs["manifest_sha256"], noise=noise)
            if case["name"] in references:
                comparison = {}
                for name in SPECTRA:
                    if "noisy" in name:
                        continue  # Different random draws are not regression targets.
                    # Rebin retained full Cl independently of internal HDF5
                    # field names used by an earlier bookkeeping version.
                    actual = bin_cl(row[name][:])
                    original = bin_cl(np.load(references[case["name"]]/(name+".npy")))
                    # Mixed-sign noisy cross spectra use an absolute scale.
                    error = float(np.max(abs(actual-original))/np.max(abs(original)))
                    comparison[name] = error
                    assert error < 1e-6, (case["name"],name,error)
                entry["full_map_regression_max_error_over_peak"] = comparison
            report["cases"].append(entry)
    report["passed"] = True
    report["scope"] = "Six NSIDE4096 maps with distinct SO splits, four historical clean regressions, full-resolution noise replay and harmonic ensemble; not all 8192 rows"
    report["distinct_noise_maps"] = len(noise_hashes)
    report["seed_check"] = seed_check
    write_json(root/"preflight/quality_gate.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
