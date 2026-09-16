#!/usr/bin/env python3
"""Compare completed radius-cache repaints with the original complete maps.

Uses saved unbinned C_ell to put every result into exactly the same bins.
It does not interpolate a noisy bandpower ratio, clip negative estimates,
or confuse sampling realizations with independent cosmological skies.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.integrate import quad

from analyze_radius_dm_tests import compare_outputs
from audit_lee22_sparse_power import bands


def plot_spherical_profile_images(output):
    """Radially symmetric 2-D views of the direct-quadrature diagnostic grid.

    These are display interpolations, NOT additional painted skies or an
    independent interpolation-convergence test. Every panel shares the same
    normalized radius and every panel at a given redshift shares one colorbar.
    """
    table = np.genfromtxt(output/"preferred_profile_grid.csv", delimiter=",",
                          names=True, dtype=None, encoding="utf-8")
    profiles = [("battaglia16", "Battaglia16"), ("lee22_legacy", "Lee22 legacy, no concentration"),
                ("lee22_preferred_duffy", "Lee22 preferred + Duffy08")]
    masses = (12.5, 13., 13.5, 14., 14.5, 15., 15.5)
    coordinate = np.linspace(-3, 3, 151)
    x, y = np.meshgrid(coordinate, coordinate)
    radius = np.hypot(x, y)
    for redshift in (0., 1., 2., 3., 4.):
        at_z = table[table["redshift"] == redshift]
        max_column = at_z["spherical3_column_pc_cm3"].max()
        positive = at_z["spherical3_column_pc_cm3"] > 0
        min_column = at_z["spherical3_column_pc_cm3"][positive].min()
        norm = LogNorm(vmin=min_column, vmax=max_column)
        fig, axes = plt.subplots(7, 3, figsize=(12, 23), constrained_layout=True)
        for row, logmass in enumerate(masses):
            for col, (key, label) in enumerate(profiles):
                data = at_z[(at_z["log10_mass_msun"] == logmass) & (at_z["profile"] == key)]
                cut = data["radius_r200c"] < 3
                # Interpolate chord-mean columns, then restore exact geometric
                # chord length so the diagnostic image also vanishes at 3R.
                rr = data["radius_r200c"][cut]
                column = data["spherical3_column_pc_cm3"][cut]
                chord_mean = column/(2*np.sqrt(9-rr*rr))
                values = np.exp(np.interp(np.log(np.maximum(radius, rr.min())),
                                         np.log(rr), np.log(chord_mean)))
                values *= 2*np.sqrt(np.maximum(0, 9-radius*radius))
                displayed = np.ma.masked_where(radius >= 3, values)
                ax = axes[row, col]
                mesh = ax.imshow(displayed, extent=(-3, 3, -3, 3), origin="lower",
                                 norm=norm, cmap="magma", interpolation="nearest")
                if row == 0:
                    ax.set_title(label, fontsize=10)
                if col == 0:
                    ax.set_ylabel("log10(M/Msun)={}\ny/R200c".format(logmass), fontsize=10)
                if row == len(masses)-1:
                    ax.set_xlabel("x/R200c")
                ax.set_xticks([-3, 0, 3])
                ax.set_yticks([-3, 0, 3])
        colorbar = fig.colorbar(mesh, ax=axes, shrink=.6, pad=.02)
        colorbar.set_label(r"Observer-frame halo DM [pc cm$^{-3}$], common logarithmic scale")
        fig.suptitle("Finite spherical 3R200c: halo z={}\n2-D display of direct radial profiles; Duffy proxy and Lee22 fit extrapolations apply".format(int(redshift)), fontsize=12)
        fig.savefig(output/("spherical3_projected_halo_images_z{}.png".format(int(redshift))),
                    dpi=125, bbox_inches="tight")
        plt.close(fig)


def check_dimensionless_gas_budget(output):
    """Independent unit-free check of the Julia gas-mass diagnostic.

    M200c = (4pi/3) R200c^3 200 rho_c makes R200c, rho_c, fb and m_p
    cancel. The remaining result tests the normalization without Unitful,
    XGPaint, a numerical cosmology or any physical-unit conversion.
    These formulas repeat the coefficients deliberately as an independent
    check, not as a second production density implementation.
    """
    table = np.genfromtxt(output/"preferred_profile_grid.csv", delimiter=",",
                          names=True, dtype=None, encoding="utf-8")
    rows = table[(table["profile"] != "battaglia16") &
                 (table["radius_r200c"] == table["radius_r200c"].min())]
    worst = 0.
    for row in rows:
        mass, z = 10**row["log10_mass_msun"], row["redshift"]
        if row["profile"] == "lee22_legacy":
            ratio = mass/(10**13.61/.68)
            n0 = 6.8*(mass/1e14)**.68*(1+z)**-2.11
            xc = 7.9*(1+z)**-.67*ratio**(.47 if ratio < 1 else -.45)
            beta = 19.5*(1+z)**-.31*ratio**(.70 if ratio < 1 else -.18)
        else:
            ratio = mass/(10**13.75/.68)
            c10 = 5.71*(mass*.68/2e12)**-.084*(1+z)**-.47/10
            n0 = 15.7*ratio**.87*(1+z)**-2.09*c10**.63
            xc = 2.2*(1+z)**-.74*c10**-1.37*ratio**(-.06 if ratio < 1 else -1.45)
            beta = 7.5*(1+z)**-.39*c10**-1.11*ratio**(.24 if ratio < 1 else -1.10)
        shape_integral = quad(lambda x: x*x*(x/xc)**-.3*(1+x/xc)**-beta,
                              0, 1, epsabs=1e-12, epsrel=1e-10, points=[.04])[0]
        # n200 uses 1/X_H; mu_e/m_p = 1/0.88 for fully ionized primordial gas.
        gas_fraction = 3*n0/(.76*.88)*shape_integral
        relative = gas_fraction/row["gas_equivalent_fraction_r200c"]-1
        worst = max(worst, abs(relative))
    if worst > 1e-6 or len(rows) != 98:
        raise ValueError("Independent gas-budget check failed")
    return dict(cases=len(rows), maximum_relative_error=worst,
                result="PASS numerical identity, not a physical gas-budget approval",
                analytic_ratio="3 n0 integral_0^1[f(x) x^2 dx] / (X_H * 0.88)")


def verify_matching_halos(old, new):
    keys = ("source_redshift", "nside", "ordering", "profile_mass_definition",
            "profile_mass_units", "aperture_radius_definition", "aperture_r200c_multiplier",
            "catalog_mass_dataset", "catalog_mass_native_units", "catalog_rows_scanned",
            "catalog_total_rows", "halos_selected", "selected_mass_min_msun", "selected_mass_max_msun")
    for key in keys:
        if old.get(key) != new.get(key):
            raise ValueError("Incompatible comparison {}: {} versus {}".format(key, old.get(key), new.get(key)))


def summarize_maps(output):
    report = {}
    for path in sorted(output.glob("*.npz")):
        if not path.with_suffix(".json").exists():
            continue
        data = np.load(path)
        if "dense_cl" not in data:
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        dense, samples = data["dense_bands"], data["finite_samples"]
        std = samples.std(axis=0, ddof=1)
        report[path.stem] = dict(
            mean_dm=float(data["mean_dm"]), last_band_ell=float(data["ell"][-1]),
            last_band_dense_cl=float(dense[-1]),
            last_band_single_catalog_scatter_over_signal=float(std[-1]/dense[-1]),
            last_band_mean_minus_truth_in_sem=float((samples[:, -1].mean()-dense[-1])/(std[-1]/np.sqrt(len(samples)))),
            halos=meta["provenance"]["halos_selected"])
        validation = output.parent/"maps"/(path.stem+"_radius_validation.csv")
        if validation.exists():
            rows = np.genfromtxt(validation, delimiter=",", names=True)
            report[path.stem]["cache_max_relative_error"] = float(np.max(np.abs(rows["relative_error"])))
    return report


def compare_old_maps(output, baseline, report):
    old = np.load(baseline/"cluster_sampling_samples.npz")
    old_meta = json.loads((baseline/"cluster_sampling_summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    differences = {}
    for column, (old_key, new_key, title) in enumerate([
            ("battaglia16", "battaglia16_projected3", "Battaglia16"),
            ("lee22", "lee22_legacy_projected3", "Lee22 no concentration (legacy density)")]):
        new = np.load(output/(new_key+".npz"))
        new_meta = json.loads((output/(new_key+".json")).read_text())
        verify_matching_halos(old_meta["models"][old_key]["provenance"], new_meta["provenance"])
        if len(old[old_key+"_dense_cl"]) != len(new["dense_cl"]):
            raise ValueError("Mismatched lmax")
        ell = new["ell"]
        old_bands = bands(old[old_key+"_dense_cl"], new["ell_min"], new["ell_max"])
        new_bands = new["dense_bands"]
        percent = 100*(new_bands/old_bands-1)
        factor = ell*(ell+1)/(2*np.pi)
        axes[0, column].loglog(ell, factor*old_bands, "--", label="Previous angular-cache map", color="#a04e59")
        axes[0, column].loglog(ell, factor*new_bands, label="Corrected radius-cache map", color="#247b98")
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel(r"Complete-map $D_\ell$ [(pc cm$^{-3}$)$^2$]")
        axes[0, column].legend(fontsize=9)
        axes[1, column].semilogx(ell, percent, color="#247b98")
        axes[1, column].axhline(0, color="black", lw=.7)
        axes[1, column].set_ylabel("100 (corrected / previous - 1) [%]")
        differences[new_key] = dict(
            mean_dm_percent=100*(float(new["mean_dm"])/old_meta["models"][old_key]["mean_dm"]-1),
            highest_band_power_percent=float(percent[-1]),
            power_percent_near_ell={str(target): float(percent[np.argmin(abs(ell-target))])
                                   for target in (100, 1000, 4000, 8000)})
    for ax in axes.flat:
        ax.set_xlabel(r"Multipole $\ell$")
        ax.grid(alpha=.2)
    fig.suptitle("Numerical repaint test: same all-halo catalogue, z-source=1, NSIDE4096\nProjected 3R200c footprint and original LOS retained; no sampling noise in these curves", fontsize=12)
    fig.savefig(output/"previous_vs_radius_corrected_full_maps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report["previous_vs_corrected_projected3"] = differences


def compare_boundaries(output, report):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    results = {}
    for col, key in enumerate(("battaglia16", "lee22_legacy")):
        projected_path, spherical_path = [output/(key+"_"+kind+".npz") for kind in ("projected3", "spherical3")]
        if not projected_path.exists() or not spherical_path.exists():
            axes[0, col].text(.5, .5, "Spherical result pending", ha="center", transform=axes[0, col].transAxes)
            continue
        projected, spherical = np.load(projected_path), np.load(spherical_path)
        np.testing.assert_array_equal(projected["ell"], spherical["ell"])
        ell = projected["ell"]
        factor = ell*(ell+1)/(2*np.pi)
        for data, label in ((projected, "Projected footprint, original LOS"), (spherical, "Finite spherical 3R200c")):
            axes[0, col].loglog(ell, factor*data["dense_bands"], label=label)
        delta = 100*(spherical["dense_bands"]/projected["dense_bands"]-1)
        axes[1, col].semilogx(ell, delta)
        axes[1, col].axhline(0, color="black", lw=.7)
        axes[0, col].set_title(key.replace("_", " "))
        axes[0, col].legend(fontsize=8)
        axes[0, col].set_ylabel(r"$D_\ell$ [(pc cm$^{-3}$)$^2$]")
        axes[1, col].set_ylabel("100 (spherical / projected - 1) [%]")
        results[key] = dict(mean_dm_percent=100*(float(spherical["mean_dm"])/float(projected["mean_dm"])-1),
                            last_band_power_percent=float(delta[-1]))
    for ax in axes.flat:
        ax.set_xlabel(r"Multipole $\ell$")
        ax.grid(alpha=.2)
    fig.suptitle("Boundary experiment: same corrected radius cache and same halos\nSpherical run also uses tan(theta) screen-plane radius; this is a physical geometry change", fontsize=12)
    fig.savefig(output/"spherical_vs_projected_3r200c.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report["boundary_experiment"] = results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Downloaded cluster_results/analysis")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Completed job596898 numerical-test outputs")
    parser.add_argument("--profile-images", action="store_true", help="Also render five 7-by-3 isolated-halo profile grids")
    args = parser.parse_args()
    report = summarize_maps(args.output_dir)
    report["independent_dimensionless_gas_budget"] = check_dimensionless_gas_budget(args.output_dir)
    compare_old_maps(args.output_dir, args.baseline_dir, report)
    compare_boundaries(args.output_dir, report)
    compare_outputs(args.output_dir)
    if args.profile_images:
        plot_spherical_profile_images(args.output_dir)
    (args.output_dir/"repaint_comparison_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
