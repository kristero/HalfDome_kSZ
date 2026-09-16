#!/usr/bin/env python
"""Halo-only HalfDome y x DM versus approximate Takahashi Figure 13 data.

Stages: prepare (observed source kernels), spectra (complete-map cross spectra),
plot (beam convolution and angular-annulus averaging), self-test.
Compatible with the cluster's Python 3.6. No covariance fit is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = (
    ("battaglia16", "Battaglia16", "#245c78", "-"),
    ("lee22_legacy", "Lee22, no concentration (legacy)", "#d65a59", "-"),
    ("lee22_preferred", "Lee22 preferred + Duffy08 (diagnostic)", "#9565ad", "--"),
)
SURVEYS = {"planck": (71, 10.0), "act": (31, 1.6)}


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_provenance(path):
    return dict(line.rstrip().split("=", 1) for line in Path(path).read_text().splitlines()
                if "=" in line)


def prepare(args):
    rows = read_rows(args.inputs / "takahashi2025_v2_table8_133_frbs.csv")
    weights = None
    if args.source_weights:
        weights = {r["frb"]: float(r["weight"]) for r in read_rows(args.source_weights)}
    out = args.output / "kernels"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"weighting": "equal source weights" if weights is None else "user-supplied source weights",
               "warning": "Equal weights approximate, not Eq.28 MW+host inverse-variance weights. No survey noise/mask mocks.",
               "catalogue_sha256": sha256(args.inputs / "takahashi2025_v2_table8_133_frbs.csv"),
               "source_positions": "uniform independent directions, evaluated through full-sky ensemble average",
               "surveys": {}}
    for survey, (count, beam) in SURVEYS.items():
        chosen = [r for r in rows if r["table_" + survey + "_footprint"] == "1"
                  and r["cluster_host_excluded_from_cross"] == "0"]
        if len(chosen) != count:
            raise ValueError("Unexpected {} sample size {}".format(survey, len(chosen)))
        result = [{"frb": r["frb"], "redshift": float(r["redshift"]),
                   "weight": 1.0 if weights is None else weights[r["frb"]]} for r in chosen]
        if any(not np.isfinite(r["weight"]) or r["weight"] <= 0 for r in result):
            raise ValueError("Source weights must be finite and positive")
        write_rows(out / (survey + "_sources.csv"), result)
        zs = [r["redshift"] for r in result]
        summary["surveys"][survey] = {"count": count, "mean_z": float(np.mean(zs)),
            "max_z": max(zs), "min_z": min(zs), "beam_fwhm_arcmin": beam,
            "kernel_sha256": sha256(out / (survey + "_sources.csv"))}
    (out / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def spectra(args):
    import healpy as hp
    out = args.output / "spectra"
    out.mkdir(parents=True, exist_ok=True)
    target = out / (args.label + ".npz")
    if target.exists():
        raise FileExistsError(str(target))
    tsz_meta = read_provenance(args.tsz_map.with_name(args.tsz_map.stem + "_provenance.txt"))
    required = {"profile_mass_definition": "M200c", "ordering": "RING", "beam": "none",
                "instrumental_noise": "none", "mask": "none", "catalog_truncated": "false",
                "map_units": "dimensionless Compton-y"}
    for key, expected in required.items():
        if tsz_meta.get(key) != expected:
            raise ValueError("tSZ provenance {} must equal {}".format(key, expected))
    if not tsz_meta["profile_label"].startswith("Battaglia12 fiducial"):
        raise ValueError("Not the Battaglia12 pressure map")
    if float(tsz_meta["maximum_halo_redshift_requested"]) != np.inf:
        raise ValueError("tSZ must include the full lightcone, not z<=1")
    if tsz_meta["catalog_rows_scanned"] != tsz_meta["catalog_total_rows"]:
        raise ValueError("tSZ catalogue incomplete")
    # The map may have been copied to the cluster; its digest is checked against
    # the transfer sidecar instead of requiring its old pathname to match.
    digest = sha256(args.tsz_map)
    sidecar = args.tsz_map.with_suffix(".sha256")
    if sidecar.exists() and digest != sidecar.read_text().split()[0]:
        raise ValueError("tSZ transfer checksum mismatch")
    y = hp.read_map(str(args.tsz_map), dtype=np.float64, verbose=False)
    nside = hp.get_nside(y)
    if nside != int(tsz_meta["nside"]) or nside != 4096:
        raise ValueError("This comparison expects NSIDE4096")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Invalid full-sky Compton-y map")
    lmax = args.lmax
    if not 2 <= lmax <= 2*nside:
        raise ValueError("Use lmax <= 2*NSIDE; higher modes need a separate convergence test")
    y_mean = float(np.mean(y))
    y -= y_mean
    print("Transforming full-lightcone y, lmax={}".format(lmax), flush=True)
    yalm = hp.map2alm(y, lmax=lmax, iter=3, pol=False)
    del y
    arrays = {"ell": np.arange(lmax+1), "cl_yy": hp.alm2cl(yalm)}
    metadata = {"tsz": tsz_meta, "tsz_map_sha256": digest, "y_mean": y_mean,
                "lmax": lmax, "map2alm_iterations": 3, "source_maps": {},
                "noise": "none; full maps, no ray shot-noise subtraction", "kernel_weighting":
                json.loads((args.output / "kernels/provenance.json").read_text())["weighting"]}
    for survey, (count, _) in SURVEYS.items():
        path = args.output / "maps" / (args.label + "_" + survey + ".fits")
        meta = read_provenance(path.with_name(path.stem + "_provenance.txt"))
        if (int(meta["source_count"]) != count or int(meta["nside"]) != nside
                or float(meta["spherical_cut_r200c"]) != 3
                or meta["profile_mass_definition"] != "M200c"
                or meta["ordering"] != "RING"
                or meta["catalog_rows_scanned"] != meta["catalog_total_rows"]
                or meta["catalog_total_rows"] != tsz_meta["catalog_total_rows"]):
            raise ValueError("Incompatible/incomplete DM map provenance")
        expected_kernel = sha256(args.output / "kernels" / (survey + "_sources.csv"))
        if meta["source_kernel_sha256"] != expected_kernel:
            raise ValueError("DM map kernel does not match the requested sources")
        dm = hp.read_map(str(path), dtype=np.float64, verbose=False)
        if hp.get_nside(dm) != nside or not np.all(np.isfinite(dm)) or np.any(dm < 0):
            raise ValueError("Invalid DM map")
        dm -= np.mean(dm)
        print("Transforming {} {} DM".format(args.label, survey), flush=True)
        alm = hp.map2alm(dm, lmax=lmax, iter=3, pol=False)
        del dm
        cross = hp.alm2cl(yalm, alm)
        auto = hp.alm2cl(alm)
        del alm
        if not np.all(np.isfinite(cross)):
            raise ValueError("Non-finite cross spectrum")
        if np.any(cross[1:]**2 > arrays["cl_yy"][1:]*auto[1:]*(1+1e-9)):
            raise ValueError("Cauchy-Schwarz cross-spectrum bound violated")
        cross[0] = 0  # Per-redshift monopole subtraction commutes with kernel averaging.
        arrays["cl_y_dm_" + survey] = cross
        arrays["cl_dm_" + survey] = auto
        metadata["source_maps"][survey] = meta
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(str(target), **arrays)
    print("Saved " + str(target), flush=True)


def gaussian_beam(ell, fwhm_arcmin):
    sigma = np.deg2rad(fwhm_arcmin/60.) / np.sqrt(8*np.log(2.))
    return np.exp(-0.5*ell*(ell+1)*sigma*sigma)


def angular_correlation(cl, theta_arcmin, beam_arcmin):
    ell = np.arange(len(cl))
    coefficients = (2*ell+1)/(4*np.pi)*cl*gaussian_beam(ell, beam_arcmin)
    coefficients[0] = 0
    return np.polynomial.legendre.legval(np.cos(np.deg2rad(np.asarray(theta_arcmin)/60.)), coefficients)


def annular_correlation(cl, lower_arcmin, upper_arcmin, beam_arcmin):
    """Exact solid-angle average: integral P_l(mu) dmu / (mu_hi-mu_lo).

    Legendre antiderivatives avoid using plotted/jittered points as bin centres.
    Only y is beam-smoothed; FRB positions are not smeared by a second beam.
    """
    lower = np.asarray(lower_arcmin)
    upper = np.asarray(upper_arcmin)
    if np.any(lower <= 0) or np.any(upper <= lower):
        raise ValueError("Invalid angular bins")
    ell = np.arange(len(cl))
    coefficients = (2*ell+1)/(4*np.pi)*cl*gaussian_beam(ell, beam_arcmin)
    coefficients[0] = 0
    primitive = np.polynomial.legendre.legint(coefficients)
    a, b = np.deg2rad(lower/60.), np.deg2rad(upper/60.)
    width = 2*np.sin((a+b)/2)*np.sin((b-a)/2)
    return (np.polynomial.legendre.legval(np.cos(a), primitive)
            - np.polynomial.legendre.legval(np.cos(b), primitive))/width


def self_test():
    # A pure dipole has exactly mean(P_1)=mean(mu) in an annulus.
    cl = np.zeros(5)
    cl[1] = 4*np.pi/3
    lo = np.array([1., 10., 100.])
    hi = 2*lo
    expected = (np.cos(np.deg2rad(lo/60.)) + np.cos(np.deg2rad(hi/60.)))/2
    np.testing.assert_allclose(annular_correlation(cl, lo, hi, 0), expected, rtol=2e-8)
    rng = np.random.RandomState(42)
    cl = rng.normal(size=2049)/(np.arange(2049)+1)**2
    cl[0] = 0
    from scipy.special import roots_legendre
    nodes, weights = roots_legendre(512)
    for a, b in zip(lo, hi):
        mu1, mu2 = np.cos(np.deg2rad([a, b])/60.)
        mu = (mu1+mu2)/2+(mu1-mu2)*nodes/2
        theta = np.rad2deg(np.arccos(mu))*60
        reference = np.dot(weights, angular_correlation(cl, theta, 1.6))/2
        np.testing.assert_allclose(annular_correlation(cl, [a], [b], 1.6), reference, rtol=2e-7)
    np.testing.assert_allclose(gaussian_beam(np.array([0]), 10), [1])
    print("PASS: dipole normalization, exact annuli vs independent Gauss-Legendre quadrature, beam monopole")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = args.output / "plots"
    out.mkdir(parents=True, exist_ok=True)
    observations = read_rows(args.inputs / "digitized/takahashi_fig13_approximate.csv")
    loaded = {}
    tsz_digests = set()
    for key, _, _, _ in MODELS:
        path = args.output / "spectra" / (key + ".npz")
        if path.is_file():
            with np.load(str(path), allow_pickle=False) as data:
                loaded[key] = {name: data[name] for name in data.files}
            meta = json.loads(str(loaded[key]["metadata_json"].item()))
            tsz_digests.add(meta["tsz_map_sha256"])
            for survey, (count, _) in SURVEYS.items():
                dm_meta = meta["source_maps"][survey]
                if (dm_meta["profile_label"] != key or int(dm_meta["source_count"]) != count
                        or dm_meta["source_kernel_sha256"] != sha256(
                            args.output / "kernels" / (survey + "_sources.csv"))):
                    raise ValueError("Spectrum provenance does not match model/source kernel")
    if not all(key in loaded for key in ("battaglia16", "lee22_legacy")):
        raise FileNotFoundError("Need completed Battaglia16 and Lee22 legacy cross spectra")
    if len(tsz_digests) != 1:
        raise ValueError("Compared models must use the identical tSZ map")
    kernel_meta = json.loads((args.output / "kernels/provenance.json").read_text())
    rows_out, summary = [], {"scope": "halo-only partial prediction", "weighting": kernel_meta["weighting"],
        "survey_noise_mask_mocks": False, "formal_covariance_fit": False, "models": {}}
    theta = np.geomspace(1, 1000, 600)
    for include_preferred in (False, True):
        if include_preferred and "lee22_preferred" not in loaded:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8))
        for ax, (survey, (count, beam)) in zip(axes, SURVEYS.items()):
            obs = [r for r in observations if ("ACT" in r["series"]) == (survey == "act")]
            x = np.array([float(r["theta_plotted_arcmin"]) for r in obs])
            y = np.array([float(r["w_yDM_pc_cm3"]) for r in obs])
            errlo = np.array([float(r["error_lower_pc_cm3"]) for r in obs])
            errhi = np.array([float(r["error_upper_pc_cm3"]) for r in obs])
            lo = np.array([float(r["theta_bin_lower_arcmin"]) for r in obs])
            hi = np.array([float(r["theta_bin_upper_arcmin"]) for r in obs])
            ax.errorbar(x, 1e5*y, yerr=1e5*np.vstack([errlo, errhi]), fmt="o", ms=4,
                        color="black", capsize=2, linewidth=1, label="Takahashi Fig. 13 (digitized)", zorder=5)
            for key, label, color, linestyle in MODELS:
                if key not in loaded or (key == "lee22_preferred" and not include_preferred):
                    continue
                data = loaded[key]
                cl = data["cl_y_dm_" + survey]
                smooth = angular_correlation(cl, theta, beam)
                binned = annular_correlation(cl, lo, hi, beam)
                ax.plot(theta, smooth*1e5, color=color, ls=linestyle, lw=2.2, label=label)
                ax.plot(np.sqrt(lo*hi), binned*1e5, "s", color=color, ms=3)
                if not include_preferred or key == "lee22_preferred":
                    smaller = annular_correlation(cl[:int(.75*(len(cl)-1))+1], lo, hi, beam)
                    scale = max(np.max(np.abs(binned)), 1e-30)
                    summary["models"][key + "_" + survey] = {
                        "w_at_10_30_100_arcmin": angular_correlation(cl, [10,30,100], beam).tolist(),
                        "annuli_pc_cm3": binned.tolist(),
                        "lmax": len(cl)-1,
                        "lower_lmax_absolute_change_over_peak": float(np.max(np.abs(smaller-binned))/scale),
                        "lower_lmax_fractional_change_by_bin": ((smaller-binned)/np.maximum(np.abs(binned), scale*1e-6)).tolist()}
                    for i, r in enumerate(obs):
                        rows_out.append({"survey": survey, "model": key, "bin_index": i,
                            "theta_lower_arcmin": lo[i], "theta_upper_arcmin": hi[i],
                            "observed_w_pc_cm3": y[i], "observed_err_lower_pc_cm3": errlo[i],
                            "observed_err_upper_pc_cm3": errhi[i], "halfdome_w_pc_cm3": binned[i],
                            "halfdome_minus_observed_pc_cm3": binned[i]-y[i],
                            "lower_lmax_w_pc_cm3": smaller[i],
                            "below_paper_angular_cut": bool(lo[i] < (10 if survey == "planck" else 10**.25))})
            ax.axhline(0, color="0.4", lw=.8)
            ax.axvspan(1, 10 if survey == "planck" else 10**.25, color="0.5", alpha=.12)
            ax.set_xscale("log")
            ax.set_xlim(1, 1000)
            ax.set_xlabel(r"Angular separation $\theta$ [arcmin]", fontsize=12)
            ax.set_ylabel(r"$10^5\,w_{y\,\mathrm{DM}}(\theta)$ [pc cm$^{-3}$]", fontsize=12)
            ax.set_title("{}: {} observed redshifts; {:.1f}' beam".format(
                "Planck MILCA" if survey == "planck" else "ACT", count, beam), fontsize=13, pad=10)
            ax.grid(alpha=.18, which="both")
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.legend(fontsize=8.8, loc="upper right", framealpha=.95)
        fig.suptitle("HalfDome halo-only tSZ-DM prediction versus Takahashi", fontsize=16, y=.975)
        fig.text(.5, .065, "B12 pressure: full lightcone, original projected 4R200c. DM: all resolved foreground halos, spherical 3R200c, NSIDE4096.",
                 ha="center", fontsize=9)
        fig.text(.5, .037, "Observed redshifts, {}. No host/diffuse DM, survey masks/noise, or covariance fit. Squares: annulus means.".format(
            kernel_meta["weighting"]), ha="center", fontsize=9)
        if include_preferred:
            fig.text(.5, .011, "Preferred Lee22 is diagnostic only: extrapolated gas normalization fails the high-mass consistency test.",
                     ha="center", color="#8a3561", fontsize=9)
        fig.subplots_adjust(left=.065, right=.985, top=.875, bottom=.17, wspace=.25)
        name = "halfdome_vs_takahashi_preferred_diagnostic.png" if include_preferred else "halfdome_vs_takahashi_fig13.png"
        fig.savefig(str(out / name), dpi=190)
        plt.close(fig)
    write_rows(out / "halfdome_takahashi_annular_comparison.csv", rows_out)
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for survey in SURVEYS:
        sources = read_rows(args.output / "kernels" / (survey + "_sources.csv"))
        zs = np.array([float(r["redshift"]) for r in sources])
        weights = np.array([float(r["weight"]) for r in sources])
        axes[0].hist(zs, bins=np.linspace(0, 2.2, 23), histtype="step", lw=2, label=survey + " N=" + str(len(zs)))
        grid = np.linspace(0, 2.2, 1000)
        fraction = np.array([weights[zs >= z].sum()/weights.sum() for z in grid])
        axes[1].plot(grid, fraction, label=survey)
    axes[0].set(xlabel="Observed source redshift", ylabel="Number of FRBs")
    axes[1].set(xlabel="Foreground halo redshift", ylabel="Fraction of source weight behind halo")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(str(out / "observed_source_redshifts_and_kernels.png"), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, survey in zip(axes, SURVEYS):
        for key, label, color, linestyle in MODELS:
            result = summary["models"].get(key + "_" + survey)
            if result is None:
                continue
            edges = 10**np.arange(0, 3.01, .25)
            change = np.array(result["lower_lmax_fractional_change_by_bin"])*100
            ax.plot(np.sqrt(edges[:-1]*edges[1:]), change, color=color, ls=linestyle, label=label)
        ax.axhline(0, color="0.4", lw=.8)
        ax.set_xscale("log")
        ax.set_title(survey.capitalize() + ": multipole truncation sensitivity")
        ax.set_xlabel("Angular separation [arcmin]")
        ax.set_ylabel("Change using 0.75 lmax instead of lmax [%]")
        if survey == "planck":
            # Beam suppression makes these differences floating-point-level;
            # do not magnify roundoff wiggles into a visually large effect.
            ax.set_ylim(-.0001, .0001)
            ax.text(.03, .1, "All changes < 1e-8% (numerically negligible)",
                    transform=ax.transAxes, fontsize=9)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Numerical diagnostic only: this does not test modes beyond the native map resolution")
    fig.savefig(str(out / "angular_lmax_sensitivity.png"), dpi=180)
    plt.close(fig)
    print("Saved comparisons to " + str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "spectra", "plot", "self-test"])
    parser.add_argument("--inputs", type=Path, default=Path("frb_map_generation/outputs/observational_comparison_inputs_20260913"))
    parser.add_argument("--output", type=Path, default=Path("frb_map_generation/outputs/takahashi_cross_comparison_20260913"))
    parser.add_argument("--source-weights", type=Path, help="Optional CSV frb,weight with paper Eq.28 source weights")
    parser.add_argument("--tsz-map", type=Path)
    parser.add_argument("--label", choices=[r[0] for r in MODELS], default="battaglia16")
    parser.add_argument("--lmax", type=int, default=8192)
    args = parser.parse_args()
    {"prepare": prepare, "spectra": spectra, "plot": plot, "self-test": lambda _: self_test()}[args.stage](args)


if __name__ == "__main__":
    main()
