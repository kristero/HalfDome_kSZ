#!/usr/bin/env python3
"""Nested 71/31, 10k, and 100k real sightline comparisons.

No halo painting is repeated. Each selected row already has an individual DM
at its own position and source redshift. The unused 90k rows calibrate mean
halo DM(z); they are disjoint from both smaller catalogues. Smaller-catalogue
errors use an explicit delete-one-FRB jackknife, not a rescaled 100k error.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_halfdome_takahashi import MODELS, SURVEYS, read_rows, write_rows, sha256
from compare_takahashi_sightlines import decode_text


def weighted_delete_one(values, weights):
    """Estimate weighted means and all leave-one-out replicates.

    After deleting source i, renormalize the remaining weights to sum to one.
    The jackknife covariance is (N-1)/N sum_i (w_-i - mean(w_-i))^2.
    Columns can contain every model, beam choice and angular bin together.
    """
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    n = len(weights)
    if n < 3 or values.shape[0] != n or np.any(weights <= 0):
        raise ValueError("Invalid catalogue or weights")
    weights = weights/weights.sum()
    estimate = weights @ values
    leave_one = (estimate[None, :] - weights[:, None]*values)/(1-weights[:, None])
    centered = leave_one-leave_one.mean(axis=0)
    covariance = (n-1)/n*(centered.T @ centered)
    return estimate, covariance, leave_one


def balanced_counts(total, groups, rng):
    counts = np.full(groups, total//groups, dtype=int)
    counts[rng.permutation(groups)[:total % groups]] += 1
    return counts


def self_test():
    rng = np.random.RandomState(1)
    values = rng.normal(size=(31, 4))
    weights = rng.uniform(.5, 1.5, 31)
    estimate, covariance, leave = weighted_delete_one(values, weights)
    explicit = []
    for i in range(31):
        keep = np.arange(31) != i
        explicit.append(np.average(values[keep], weights=weights[keep], axis=0))
    np.testing.assert_allclose(leave, explicit, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(covariance, np.cov(explicit, rowvar=False, ddof=0)*30)
    _, equal_cov, _ = weighted_delete_one(values, np.ones(31))
    np.testing.assert_allclose(equal_cov, np.cov(values, rowvar=False)/31)
    # One source per redshift must NOT lead to zero marks or zero jackknife.
    dm = np.array([10., 30., 80.])
    independent_mean = np.array([5., 20., 50.])
    marks = ((dm-independent_mean)*np.array([1., 2., 4.]))[:, None]
    value, cov, _ = weighted_delete_one(marks, np.ones(3))
    assert value[0] != 0 and cov[0, 0] > 0
    print("PASS: explicit weighted delete-one covariance, equal-weight limit, nonzero one-source-per-z errors")


def analyze(parent, output, seed=20260915, realizations=5000):
    parent, output = Path(parent), Path(output)
    for name in ("analysis", "catalogues", "plots"):
        (output/name).mkdir(parents=True, exist_ok=True)
    target = output/"catalogues/nested_sightlines.h5"
    if target.exists():
        raise FileExistsError("Use a new output directory: " + str(target))
    self_test()
    rng = np.random.RandomState(seed)
    models = [m[0] for m in MODELS]
    previous = read_rows(parent/"analysis/sightline_comparison.csv")
    rows, metadata = [], {}
    positions_path = parent/"rays/source_positions.h5"
    with h5py.File(str(positions_path), "r") as positions, \
            h5py.File(str(parent/"rays/individual_dm.h5"), "r") as dmfile, \
            np.load(str(parent/"rays/annular_y_samples.npz")) as yfile, \
            h5py.File(str(target), "w") as saved:
        expected = sha256(positions_path)
        if decode_text(dmfile.attrs["source_positions_sha256"]) != expected:
            raise ValueError("DM catalogue checksum mismatch")
        if json.loads(str(yfile["metadata_json"].item()))["positions_sha256"] != expected:
            raise ValueError("Annular y catalogue checksum mismatch")
        if int(dmfile.attrs["catalog_rows_scanned"]) != int(dmfile.attrs["catalog_total_rows"]):
            raise ValueError("Incomplete parent halo calculation")
        saved.attrs["seed"] = seed
        saved.attrs["parent_positions_sha256"] = expected
        saved.attrs["parent_dm_sha256"] = sha256(parent/"rays/individual_dm.h5")
        saved.attrs["parent_annular_y_sha256"] = sha256(parent/"rays/annular_y_samples.npz")
        for survey, (nobserved, beam) in SURVEYS.items():
            groups = positions[survey+"/observed_source_index"][:]
            zs = positions[survey+"/redshift"][:]
            pool_groups = [np.flatnonzero(groups == g) for g in range(nobserved)]
            if len(groups) != 100000 or any(len(g) < 100 for g in pool_groups):
                raise ValueError("Expected the completed 100k stratified parent")
            counts10k = balanced_counts(10000, nobserved, rng)
            ten_groups, calibration, observed = [], [], []
            for pool, n in zip(pool_groups, counts10k):
                ordered = rng.permutation(pool)
                ten_groups.append(ordered[:n])
                calibration.extend(ordered[n:])
                observed.append(ordered[0])
            ten = np.concatenate(ten_groups)
            observed, calibration = np.asarray(observed), np.asarray(calibration)
            assert len(ten) == 10000 and len(calibration) == 90000
            assert not np.intersect1d(ten, calibration).size
            assert np.all(np.isin(observed, ten))
            dm = np.column_stack([dmfile[survey+"/"+m][:] for m in models])
            mean_dm = np.array([dm[calibration[groups[calibration] == g]].mean(axis=0)
                                for g in range(nobserved)])
            all_y = np.stack([yfile[survey], yfile["unbeamed"]], axis=1)
            # (source, filter, density model, angular bin).
            marks = ((dm[ten]-mean_dm[groups[ten]])[:, None, :, None]
                     *all_y[ten, :, None, :]).reshape(10000, -1)
            group_saved = saved.create_group(survey)
            group_saved["calibration_parent_indices"] = calibration
            group_saved["calibration_mean_dm_by_observed_source"] = mean_dm
            group_saved["observed_redshifts"] = positions[survey+"/observed_redshifts"][:]
            for size, indices in ((nobserved, observed), (10000, ten)):
                # ten is grouped in observed-source order; map parent to its row.
                ten_lookup = {int(v): i for i, v in enumerate(ten)}
                mark_rows = np.array([ten_lookup[int(v)] for v in indices])
                counts = np.bincount(groups[indices], minlength=nobserved)
                weights = 1.0/(nobserved*counts[groups[indices]])
                value, covariance, leave = weighted_delete_one(marks[mark_rows], weights)
                value = value.reshape(2, 3, 12)
                sigma = np.sqrt(np.diag(covariance)).reshape(value.shape)
                product = group_saved.create_group("n"+str(size))
                product["parent_indices"] = indices
                product["redshift"] = zs[indices]
                product["observed_source_index"] = groups[indices]
                product["analysis_weight"] = weights
                product["longitude_deg"] = positions["longitude_deg"][:][indices]
                product["latitude_deg"] = positions["latitude_deg"][:][indices]
                product["pixel_ring_zero_based"] = positions["pixel_ring_zero_based"][:][indices]
                product["foreground_halo_hits"] = dmfile[survey+"/foreground_halo_hits"][:][indices]
                for p, model in enumerate(models):
                    product[model+"_dm"] = dm[indices, p]
                product["cross_estimate"] = value
                product["jackknife_covariance"] = covariance
                product["leave_one_out_cross"] = leave
                product.attrs["array_order"] = "filter (survey,unbeamed), model (B16,Lee22,Lee22+c), angular bin"
                product.attrs["error_method"] = "delete one source; renormalized weights; independent 90k DM(z) calibration held fixed"
                for f, filter_name in enumerate((survey, "unbeamed")):
                    for p, model in enumerate(models):
                        ref = [r for r in previous if r["survey"] == survey and
                               r["filter"] == filter_name and r["model"] == model]
                        if len(ref) != 12:
                            raise ValueError("Missing parent comparison rows")
                        for b, old in enumerate(ref):
                            rows.append(dict(survey=survey, nrays=size, model=model, filter=filter_name,
                                theta_arcmin=float(old["theta_arcmin"]),
                                theta_lower_arcmin=float(old["theta_lower_arcmin"]),
                                theta_upper_arcmin=float(old["theta_upper_arcmin"]),
                                cross=value[f,p,b], sigma=sigma[f,p,b],
                                previous_fullmap=float(old["previous_fullmap"]),
                                error_method="delete_one_FRB_jackknife"))
                print(survey, size, "saved", len(leave), "jackknife replicates", flush=True)
            # Diagnostic ensemble: one ray per observed redshift drawn from the
            # held-out 10k pool. This is NOT substituted for the requested JK.
            offset = np.r_[0, np.cumsum(counts10k)]
            draws = np.zeros((realizations, marks.shape[1]))
            for g in range(nobserved):
                choice = offset[g]+rng.randint(counts10k[g], size=realizations)
                draws += marks[choice]/nobserved
            group_saved["observed_count_resampling_cross"] = draws
            group_saved.attrs["resampling_scope"] = "conditional on held-out 10k pool; independent of mean-DM calibration; not JK errors"
            metadata[survey] = dict(nobserved=nobserved, n10k=10000, calibration_size=90000,
                mean_redshift_observed_count=float(zs[observed].mean()),
                zero_hit_fraction_observed=float(np.mean(dmfile[survey+"/foreground_halo_hits"][:][observed] == 0)),
                nested_subsets=True, duplicate_parent_indices=False)
        for old in previous:
            rows.append(dict(survey=old["survey"], nrays=100000, model=old["model"], filter=old["filter"],
                theta_arcmin=float(old["theta_arcmin"]), theta_lower_arcmin=float(old["theta_lower_arcmin"]),
                theta_upper_arcmin=float(old["theta_upper_arcmin"]), cross=float(old["cross_100k"]),
                sigma=float(old["sampling_sigma"]), previous_fullmap=float(old["previous_fullmap"]),
                error_method="previous_100k_stratified_delete_one_jackknife"))
    write_rows(output/"analysis/source_count_comparison.csv", rows)
    metadata.update(dict(parent=str(parent), seed=seed, resampling_realizations=realizations,
        observed_errors="delete-one-FRB jackknife; NOT instrumental/host/IGM or cosmic variance",
        small_catalogue_mean_dm="independent unused 90k calibration at each observed redshift",
        parent_100k="unchanged previous estimator and errors; its fitted stratum means use all 100k",
        source_scope="uniform random sightlines, not host-halo placement; exact empirical redshift matching",
        calibration_uncertainty="held fixed in jackknife; not propagated",
        error_caution="source jackknife may miss rare foregrounds absent from a 71/31-source catalogue; it is not a spatial survey jackknife"))
    (output/"analysis/source_count_provenance.json").write_text(json.dumps(metadata, indent=2))
    print("Saved " + str(output), flush=True)


def check_output(parent, output):
    """Independent checks on the saved real subsets and jackknife replicates."""
    parent, output = Path(parent), Path(output)
    checks = []
    with h5py.File(output/"catalogues/nested_sightlines.h5", "r") as saved, \
            h5py.File(parent/"rays/source_positions.h5", "r") as positions, \
            h5py.File(parent/"rays/individual_dm.h5", "r") as dmfile, \
            np.load(parent/"rays/annular_y_samples.npz") as yfile:
        for survey, (nobs, _) in SURVEYS.items():
            calibration = saved[survey+"/calibration_parent_indices"][:]
            ten = saved[survey+"/n10000/parent_indices"][:]
            small = saved[survey+"/n"+str(nobs)+"/parent_indices"][:]
            assert len(small) == nobs and len(ten) == 10000 and len(calibration) == 90000
            assert np.all(np.isin(small, ten)) and not np.intersect1d(ten, calibration).size
            np.testing.assert_array_equal(np.sort(positions[survey+"/redshift"][:][small]),
                                          np.sort(saved[survey+"/observed_redshifts"][:]))
            mean_dm = saved[survey+"/calibration_mean_dm_by_observed_source"][:]
            for n in (nobs, 10000):
                group = saved[survey+"/n"+str(n)]
                idx, weights, zgroup = [group[k][:] for k in
                                        ("parent_indices", "analysis_weight", "observed_source_index")]
                np.testing.assert_allclose(np.bincount(zgroup, weights=weights, minlength=nobs), 1/nobs)
                dm = np.column_stack([dmfile[survey+"/"+m[0]][:][idx] for m in MODELS])
                y = np.stack([yfile[survey][idx], yfile["unbeamed"][idx]], axis=1)
                marks = ((dm-mean_dm[zgroup])[:, None, :, None]*y[:, :, None, :]).reshape(n, -1)
                np.testing.assert_allclose(np.average(marks, axis=0, weights=weights),
                                           group["cross_estimate"][:].ravel(), rtol=1e-11, atol=1e-18)
                # Explicit recomputation, rather than the algebraic shortcut.
                replicate_indices = range(n) if n == nobs else (0, n//2, n-1)
                for i in replicate_indices:
                    keep = np.arange(n) != i
                    explicit = np.average(marks[keep], axis=0, weights=weights[keep])
                    np.testing.assert_allclose(group["leave_one_out_cross"][i], explicit,
                                               rtol=1e-10, atol=1e-17)
                leave = group["leave_one_out_cross"][:]
                covariance = (n-1)*np.cov(leave, rowvar=False, ddof=0)
                np.testing.assert_allclose(covariance, group["jackknife_covariance"][:],
                                           rtol=1e-10, atol=1e-24)
                assert np.all(np.diag(covariance) > 0)
                checks.append(dict(survey=survey, nrays=n, saved_jackknife_replicates=len(leave),
                                   explicit_replicates_checked=len(replicate_indices),
                                   redshift_weights_match=True, calibration_disjoint=True))
    (output/"analysis/validation_checks.json").write_text(json.dumps(checks, indent=2))
    print("PASS: saved sample sizes, nesting, redshift weights, disjoint calibration, actual delete-one replicates and covariance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="frb_map_generation/outputs/takahashi_100k_20260914")
    parser.add_argument("--output", default="frb_map_generation/outputs/publication_comparison_20260915")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--realizations", type=int, default=5000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-output", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.check_output:
        check_output(args.parent, args.output)
    else:
        analyze(args.parent, args.output, args.seed, args.realizations)
