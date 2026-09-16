#!/usr/bin/env python3
"""Explicit finite-source test against the previous HalfDome full-map mean.

Compatible with cluster Python 3.6. Source redshifts are stratified replicas of
the observed catalogue, not samples of an averaged DM map. Within each stratum
the cross estimator is the unbiased sample covariance of DM and annular y.
Jackknife errors describe random-direction sampling of this fixed simulation;
they are NOT the observational noise or cosmological covariance.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import eval_legendre

from compare_halfdome_takahashi import (
    MODELS, SURVEYS, read_rows, write_rows, sha256, read_provenance,
    gaussian_beam, annular_correlation)

EDGES = np.logspace(0, 3, 13)
FILTERS = (("planck", 10.0), ("act", 1.6), ("unbeamed", 0.0))


def decode_text(value):
    """Julia HDF5 strings can be numpy.bytes_ in the cluster's older h5py."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def prepare(args):
    import healpy as hp
    root, previous = Path(args.output), Path(args.previous)
    for name in ("rays", "analysis", "logs", "plots"):
        (root/name).mkdir(parents=True, exist_ok=True)
    target = root/"rays/source_positions.h5"
    if target.exists():
        raise FileExistsError(str(target))
    rng = np.random.RandomState(args.seed)
    # Equal-area pixel centres retain exactly the previous map's resolution.
    # Draw with replacement: iid directions, including legitimate duplicates.
    pixels = rng.randint(0, hp.nside2npix(args.nside), args.nrays)
    theta, phi = hp.pix2ang(args.nside, pixels)
    with h5py.File(str(target), "w") as h5:
        h5["pixel_ring_zero_based"] = pixels
        h5["longitude_deg"] = np.rad2deg(phi)
        h5["latitude_deg"] = 90-np.rad2deg(theta)
        h5.attrs["nside"] = args.nside
        h5.attrs["seed"] = args.seed
        h5.attrs["nrays_per_survey"] = args.nrays
        h5.attrs["selection"] = "iid equal-area pixel centres; independent of halo positions"
        for survey in SURVEYS:
            rows = read_rows(previous/"kernels"/(survey+"_sources.csv"))
            z = np.array([float(row["redshift"]) for row in rows])
            weights = np.array([float(row["weight"]) for row in rows])
            if not np.allclose(weights, weights[0]):
                raise ValueError("This matched baseline requires equal observed-source weights")
            if args.nrays < 3*len(z):
                raise ValueError("Need at least three rays per redshift stratum")
            counts = np.full(len(z), args.nrays//len(z), dtype=int)
            counts[rng.permutation(len(z))[:args.nrays % len(z)]] += 1
            group_id = np.repeat(np.arange(len(z)), counts)
            rng.shuffle(group_id)
            group = h5.create_group(survey)
            group["redshift"] = z[group_id]
            group["observed_source_index"] = group_id
            group["observed_redshifts"] = z
            group["stratum_count"] = counts
            # Balancing introduces <=1 ray rounding per stratum. These tiny
            # weights make the target source kernel EXACTLY the old one.
            group["analysis_weight"] = 1.0/(len(z)*counts[group_id])
            group.attrs["kernel_sha256"] = sha256(previous/"kernels"/(survey+"_sources.csv"))
            np.testing.assert_allclose(np.sum(group["analysis_weight"][:]), 1)
    print("Saved {} individual positions for EACH survey to {}".format(args.nrays, target), flush=True)


def annulus_window(lmax, lower, upper):
    """Average P_l over an annulus in solid angle; no small-angle approximation."""
    a, b = np.deg2rad(np.array([lower, upper])/60.0)
    mu_hi, mu_lo = np.cos(a), np.cos(b)
    width = 2*np.sin((a+b)/2)*np.sin((b-a)/2)
    ell = np.arange(1, lmax+1)
    window = np.ones(lmax+1)
    window[1:] = ((eval_legendre(ell+1, mu_hi)-eval_legendre(ell-1, mu_hi)) -
                  (eval_legendre(ell+1, mu_lo)-eval_legendre(ell-1, mu_lo)))/((2*ell+1)*width)
    return window


def sample_y(args):
    import healpy as hp
    root, previous = Path(args.output), Path(args.previous)
    target = root/"rays/annular_y_samples.npz"
    if target.exists():
        raise FileExistsError(str(target))
    tsz = previous/"inputs/battaglia12_full_lightcone_repaint.fits"
    with np.load(str(previous/"spectra/battaglia16.npz")) as ref:
        reference_meta = json.loads(str(ref["metadata_json"].item()))
        reference_yy = ref["cl_yy"].copy()
    digest = sha256(tsz)
    if digest != reference_meta["tsz_map_sha256"]:
        raise ValueError("tSZ input differs from previous comparison")
    with h5py.File(str(root/"rays/source_positions.h5"), "r") as h5:
        pixels = h5["pixel_ring_zero_based"][:]
        nside = int(h5.attrs["nside"])
    if nside != 4096 or args.lmax != 8192:
        raise ValueError("Production comparison must match previous NSIDE=4096, lmax=8192")
    y = hp.read_map(str(tsz), dtype=np.float64, verbose=False)
    if hp.get_nside(y) != nside or not np.all(np.isfinite(y)):
        raise ValueError("Invalid tSZ map")
    y -= np.mean(y)
    alm = hp.map2alm(y, lmax=args.lmax, iter=3, pol=False)
    del y
    np.testing.assert_allclose(hp.alm2cl(alm), reference_yy, rtol=1e-8, atol=1e-26)
    arrays = {"lower_arcmin": EDGES[:-1], "upper_arcmin": EDGES[1:]}
    ell = np.arange(args.lmax+1)
    for name, beam in FILTERS:
        values = np.empty((len(pixels), 12))
        for b, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            window = annulus_window(args.lmax, lo, hi)*gaussian_beam(ell, beam)
            window[0] = 0.0
            filtered = hp.alm2map(hp.almxfl(alm, window), nside, pol=False, verbose=False)
            values[:, b] = filtered[pixels]
            del filtered
            print("{} annulus {}/12 sampled".format(name, b+1), flush=True)
        arrays[name] = values
    arrays["metadata_json"] = np.asarray(json.dumps({
        "tsz_map_sha256": digest, "lmax": args.lmax, "nside": nside,
        "positions_sha256": sha256(root/"rays/source_positions.h5"),
        "operation": "annular full-sky y average at each individual source position",
        "noise": "none", "mask": "none"}))
    np.savez_compressed(str(target), **arrays)
    print("Saved " + str(target), flush=True)


def stratified_covariance(dm, y, groups):
    """Unbiased cross estimate and delete-one-source jackknife covariance.

    Each observed-redshift stratum has equal total weight. Its estimator is
    sum[(D-Dbar)(Y-Ybar)]/(n-1), removing the finite-n bias of fitted means.
    Source directions are iid draws from the fixed full-sky pixel population.
    """
    dm = np.asarray(dm)
    if dm.ndim == 1:
        dm = dm[:, None]
    nprofile, nbin = dm.shape[1], y.shape[1]
    strata = np.unique(groups)
    value = np.zeros(nprofile*nbin)
    covariance = np.zeros((len(value), len(value)))
    for group in strata:
        keep = groups == group
        n = int(np.sum(keep))
        if n < 3:
            raise ValueError("Insufficient stratum count for jackknife")
        d, t = dm[keep], y[keep]
        d = d-d.mean(axis=0)
        t = t-t.mean(axis=0)
        products = (d[:, :, None]*t[:, None, :]).reshape(n, -1)
        estimate = products.sum(axis=0)/(n-1)
        # Exact delete-one pseudovalues for a sample covariance. Their mean
        # equals estimate; independent strata contribute weight^2 * var/n.
        pseudo = (n*products-(n-1)*estimate)/(n-2)
        value += estimate/len(strata)
        covariance += np.cov(pseudo, rowvar=False, ddof=1)/n/len(strata)**2
    return value.reshape(nprofile, nbin), covariance


def analyze(args):
    root, previous = Path(args.output), Path(args.previous)
    models = [item[0] for item in MODELS]
    with h5py.File(str(root/"rays/source_positions.h5"), "r") as positions, \
            h5py.File(str(root/"rays/individual_dm.h5"), "r") as dmfile, \
            np.load(str(root/"rays/annular_y_samples.npz")) as yfile:
        digest = sha256(root/"rays/source_positions.h5")
        if decode_text(dmfile.attrs["source_positions_sha256"]) != digest:
            raise ValueError("DM positions mismatch")
        if json.loads(str(yfile["metadata_json"].item()))["positions_sha256"] != digest:
            raise ValueError("tSZ positions mismatch")
        if dmfile.attrs["catalog_rows_scanned"] != dmfile.attrs["catalog_total_rows"]:
            raise ValueError("DM run did not scan the complete catalogue")
        arrays, rows = {}, []
        for survey in SURVEYS:
            groups = positions[survey+"/observed_source_index"][:]
            dm = np.column_stack([dmfile[survey+"/"+model][:] for model in models])
            for filter_name, beam in ((survey, SURVEYS[survey][1]), ("unbeamed", 0.0)):
                values, covariance = stratified_covariance(dm, yfile[filter_name], groups)
                errors = np.sqrt(np.maximum(0, np.diag(covariance))).reshape(values.shape)
                tag = survey if filter_name != "unbeamed" else survey+"_unbeamed"
                arrays[tag+"_covariance"] = covariance
                for p, model in enumerate(models):
                    with np.load(str(previous/"spectra"/(model+".npz"))) as baseline:
                        reference = annular_correlation(baseline["cl_y_dm_"+survey],
                                                        EDGES[:-1], EDGES[1:], beam)
                    for b in range(12):
                        rows.append(dict(survey=survey, filter=filter_name, model=model,
                            theta_lower_arcmin=EDGES[b], theta_upper_arcmin=EDGES[b+1],
                            theta_arcmin=np.sqrt(EDGES[b]*EDGES[b+1]),
                            cross_100k=values[p,b], sampling_sigma=errors[p,b],
                            previous_fullmap=reference[b],
                            percent_difference=100*(values[p,b]/reference[b]-1)
                                if abs(reference[b]) > 1e-15 else np.nan))
        write_rows(root/"analysis/sightline_comparison.csv", rows)
        np.savez_compressed(str(root/"analysis/sampling_covariances.npz"), **arrays)
        summary = dict(nrays_per_survey=int(positions.attrs["nrays_per_survey"]),
            nside=int(positions.attrs["nside"]),
            catalogue_rows=int(dmfile.attrs["catalog_rows_scanned"]),
            foreground_halos=int(dmfile.attrs["foreground_halos_considered"]),
            estimator="equal-stratum unbiased sample covariance; own-redshift halo-only DM residuals",
            errors="delete-one-source jackknife, conditional on this fixed full-sky simulation; no map noise or cosmic variance",
            preferred="diagnostic only; previous extrapolation gas-budget failure remains",
            all_source_positions_saved=True, no_redshift_averaged_dm_map_used=True)
        (root/"analysis/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def self_test(args):
    from scipy.special import roots_legendre
    assert decode_text(np.bytes_("abc")) == decode_text("abc") == "abc"
    lo, hi, lmax = 2., 12., 128
    window = annulus_window(lmax, lo, hi)
    nodes, weights = roots_legendre(160)
    a, b = np.cos(np.deg2rad(np.array([hi, lo])/60.))
    for ell in (0, 1, 5, 50, 128):
        expected = np.dot(weights, eval_legendre(ell, (a+b)/2+(b-a)*nodes/2))/2
        np.testing.assert_allclose(window[ell], expected, rtol=2e-8)
    rng = np.random.RandomState(28)
    d = rng.normal(size=(30, 2))
    y = rng.normal(size=(30, 3)) + d[:, :1]
    value, covariance = stratified_covariance(d, y, np.zeros(30))
    leave = []
    for i in range(30):
        keep = np.arange(30) != i
        dx, yy = d[keep], y[keep]
        leave.append(((dx-dx.mean(0)).T @ (yy-yy.mean(0))/(29-1)).ravel())
    np.testing.assert_allclose(covariance, np.cov(leave, rowvar=False, ddof=0)*(30-1), rtol=1e-12)
    np.testing.assert_allclose(value, np.cov(d.T, y.T)[:2, 2:], rtol=1e-12)
    # Individual redshifts must alter DM. A z-averaged field fails this test.
    halo_columns = np.array([[1., 2.], [3., 5.]])
    zs = np.array([0.2, 0.8])
    source_z = np.array([0.1, 0.5, 1.0])
    individual = np.array([halo_columns[zs <= z].sum(0) for z in source_z])
    np.testing.assert_array_equal(individual, [[0, 0], [1, 2], [4, 7]])
    print("PASS: annular filter quadrature, unbiased covariance, exact delete-one jackknife, individual source screens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "sample-y", "analyze", "self-test"))
    parser.add_argument("--output", default="frb_map_generation/outputs/takahashi_100k_20260914")
    parser.add_argument("--previous", default="frb_map_generation/outputs/takahashi_cross_comparison_20260913")
    parser.add_argument("--nrays", type=int, default=100000)
    parser.add_argument("--nside", type=int, default=4096)
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    {"prepare": prepare, "sample-y": sample_y, "analyze": analyze, "self-test": self_test}[args.stage](args)


if __name__ == "__main__":
    main()
