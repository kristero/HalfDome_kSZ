#!/usr/bin/env python3
"""Quantify the historical angular interpolation's effect on the same HalfDome rays."""

import json

import h5py
import numpy as np

from preflight import OUTPUT, WINDOWS, sha256


def main():
    old_path = OUTPUT / "halfdome/halo_dm.h5"
    exact_path = OUTPUT / "halfdome_direct/halo_dm.h5"
    with h5py.File(old_path, "r") as old, h5py.File(exact_path, "r") as exact:
        np.testing.assert_array_equal(old["frb_pixel_ring_1based"][:], exact["frb_pixel_ring_1based"][:])
        np.testing.assert_array_equal(old["window_halo_count"][:], exact["window_halo_count"][:])
        cached, direct = old["dm_pc_cm3"][:], exact["dm_pc_cm3"][:]
    np.testing.assert_array_equal(cached > 0, direct > 0)
    rows = []
    for index, (label, _, _) in enumerate(WINDOWS):
        hit = direct[index] > 0
        relative = cached[index, hit] / direct[index, hit] - 1
        rows.append(dict(window=label, cached_mean=float(cached[index].mean()), direct_mean=float(direct[index].mean()),
                         cached_mean_bias_percent=100 * (cached[index].mean() / direct[index].mean() - 1),
                         median_ray_relative_error=float(np.median(relative)),
                         percentile99_ray_relative_error=float(np.quantile(relative, .99)),
                         maximum_ray_relative_error=float(relative.max())))
    report = dict(same_sightlines=True, same_selected_halos=True, same_hit_mask=True,
                  profile_source_changed=False, final_evaluation="unchanged profile; direct spherical chord integration",
                  historical_angular_cache_comparison=rows,
                  native_tng_cache_diagnostic=dict(mass_msun=8e14, redshift=.03, impact_r200c=.99,
                                                   cached_dm=80.63924534075983, direct_dm=52.26094263623433,
                                                   relative_error=0.5430116885195604, used_for_final_results=False),
                  cached_catalogue_sha256=sha256(old_path), direct_catalogue_sha256=sha256(exact_path))
    (OUTPUT / "numerical_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
