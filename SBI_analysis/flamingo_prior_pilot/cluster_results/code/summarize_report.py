"""Write the measured science report from completed, verified cluster outputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from prior_model import FIDUCIAL, HIGH, LOW, NAMES, domain_metrics, to_unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    load = lambda relative: json.loads((root / relative).read_text())
    result = load("pilot_results.json")
    verification = load("verification.json")
    assert verification["status"] == "passed"
    prior = load("audit/prior_audit.json")["groups"]
    catalogue = load("audit/actual_catalogue_prior_audit.json")["groups"]
    conditional = load("audit/conditional_prior_audit.json")
    interpolation = load("audit/interpolation_summary.json")
    quadrature = load("audit/quadrature_cost_summary.json")
    pixels = load("audit/pixel_resolution.json")["rows"]
    resources = load("resource_usage.json")
    support = load("audit/catalogue_support.json")
    examples = load("audit/extreme_parameter_examples.json")
    scaling = load("audit/amplitude_scaling_validation.json")
    scale_metrics = load("audit/fit_scale_metrics.json")["fits"]
    projected = load("audit/projected_cutoff.json")["rows"]
    variants = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")
    lines = [
        "# FLAMINGO pressure-prior pilot: completed results",
        "",
        "**The full independent rectangular prior is not ready for a larger production run.** "
        "It contains divergent or cutoff-sensitive pressure combinations, extreme thermal-content "
        "excursions, and compact profiles that the current pixel sampling cannot resolve. "
        "The fitted spectra below establish what this pilot actually reproduces.",
        "",
        "All numerical science calculations and full maps were run on the cluster. "
        "Clean spectra were fitted; the identical cached SO noise was compared afterward. "
        "The original HalfDome operators, old inference bounds, MOPED weights and trained SBI "
        "bundle were preserved.",
        "",
        "[All nine plot sets in one vector PDF](plots/tsz_prior_pilot_plots.pdf) | "
        "[Parameter table](parameters.csv) | [Clean/noisy spectra](fit_spectra.csv)",
        "",
        "## Full-map fit quality",
        "",
        "| FLAMINGO target | RMS fractional residual | Maximum bin residual | 2% RMS / 5% maximum target |",
        "|---|---:|---:|---|",
    ]
    for variant in variants:
        fit = result["fits"][variant]
        best = fit["best"]
        lines.append("| {} | {:.3f}% | {:.3f}% | {} |".format(variant,
            100*best["rms_fractional"], 100*best["max_fractional"],
            "met" if fit["meets_accuracy_target"] else "not met"))
    lines += ["", "These values use repainted Nside4096 spectra over all 40 bins "
        "(ell 80–7979). RMS is sqrt(mean((fit/target-1)^2)); the optimization objective "
        "uses log residuals. They are effective matches on the HalfDome catalogue, "
        "not unique FLAMINGO feedback measurements or SBI posteriors.",
        "", "The supplied FLAMINGO L1 maps are lensed and integrated to z=3 "
        "([product documentation](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/integrated_lightcones.html#integrated-thermal-sz-maps)). "
        "The painted HalfDome catalogue extends to z={:.6f}. Cosmology, halo/gas "
        "modelling, lensing and realization differences remain in this comparison. "
        "The fitted pressure evolution can absorb these differences, so its redshift "
        "exponents should not be interpreted as isolated feedback measurements.".format(support["z_max"]),
        "", "![Clean fits](plots/01_clean_flamingo_fits.png)",
        "", "Scale dependence of the same fits (no refitting or removal of bins):", "",
        "| Target | RMS for ell>=500 | Maximum for ell>=500 | Worst full-range bin centre |",
        "|---|---:|---:|---:|"]
    for variant in variants:
        scope = scale_metrics[variant]["ell_ge_500"]
        lines.append("| {} | {:.3f}% | {:.3f}% | {:.2f} |".format(
            variant, 100*scope["rms_fractional"], 100*scope["max_fractional"],
            scale_metrics[variant]["worst_bin_ell"]))
    lines += [
        "", "Additional ell>=1000 and ell>=2000 metrics are saved in "
        "audit/fit_scale_metrics.json. These are descriptive residuals; the pilot "
        "does not have a covariance-weighted goodness-of-fit or a multi-realization "
        "test assigning the residuals uniquely to feedback or sample variance.",
        "", "## Fitted parameters and tested ranges", "",
        "| Parameter | Lower | Upper | Battaglia12 | L1_m9 fit | fgas-8sigma fit | Mstar-1sigma fit |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for i, name in enumerate(NAMES):
        values = [LOW[i], HIGH[i], FIDUCIAL[i]]+[result["fits"][v]["best"]["theta"][i] for v in variants]
        lines.append("| "+name+" | "+" | ".join("{:.6g}".format(x) for x in values)+" |")
    lines += ["", "P0 and xc were drawn uniformly in their logarithms; the other "
        "coordinates were drawn uniformly. Alpha=1 and gamma=-0.3 remain fixed. "
        "CSV/JSON files preserve full precision.", ""]
    for variant in variants:
        theta = np.array(result["fits"][variant]["best"]["theta"])
        unit = to_unit(theta)
        edges = [name for name, x in zip(NAMES, unit) if min(x, 1-x) < .02]
        lines.append("- {}: min beta on the interpolation grid = {:.4g}; "
            "within 2% of a prior edge in unit coordinates: {}.".format(
                variant, float(domain_metrics(theta)["min_beta"]), ", ".join(edges) if edges else "none"))
    lines += ["", "A weak 0.005 unit-cube regularizer selects representatives among "
        "spectral degeneracies. Some proposal optimizers stop at their evaluation limit; "
        "the measured full-map residual, not that stopping flag, determines the fit-quality "
        "label. A boundary fit does not establish that its parameter is constrained.",
        "", "## tSZ extremes and combinations", "",
        "![tSZ range](plots/02_tsz_with_extremes.png)", "",
        "| Explicit full-map stress test | Outcome | Minimum / maximum clean-power ratio to Battaglia12 |",
        "|---|---|---|"]
    for name, record in result["extremes"].items():
        if record["status"] == "full_map_completed":
            detail = "{:.5g} / {:.5g}".format(record["power_ratio_min"], record["power_ratio_max"])
        else:
            detail = "No accepted full-map spectrum; inspect its recorded run status/log"
        lines.append("| {} | {} | {} |".format(name, record["status"], detail))
    lines += ["", "P0=1 and P0=60 are exact amplitude rescalings at fixed other "
        "parameters: y scales as P0 and clean power as P0 squared. Their power ratios "
        "are {:.6g} and {:.6g}. Shape/evolution examples use new full maps when they "
        "complete; failed examples remain explicit.".format((1/18.1)**2, (60/18.1)**2),
        "", "## Whole-prior audit", "",
        "| Condition | Interpolation grid: 65,536 draws | Actual catalogue: 65,536 draws |",
        "|---|---:|---:|"]
    for label, key in (("Divergent untruncated LOS", "los_divergent"),
                       ("Divergent infinite-volume pressure integral", "infinite_thermal_energy_divergent")):
        a, b = prior["sobol"][key], catalogue["sobol"][key]
        lines.append("| {} | {} ({:.4f}%) | {} ({:.4f}%) |".format(label, a, a/655.36, b, b/655.36))
    lines += ["", "{} draws ({:.3f}%) lose more than 1% of the central column beyond "
        "the current LOS endpoint somewhere on the diagnostic grid. The finite Y200 "
        "ratio across sampled points spans {:.4g} to {:.4g}. These extreme ratios "
        "are diagnostics, not measured exclusion thresholds.".format(
            prior["sobol"]["central_column_tail_gt_1pct"],
            prior["sobol"]["central_column_tail_gt_1pct"]/655.36,
        prior["sobol"]["min_Y200_ratio"], prior["sobol"]["max_Y200_ratio"]),
        "", "All 512 corners were also evaluated: {} have divergent LOS columns on "
        "the interpolation grid and {} at actual catalogue support. The 18 single "
        "edges and 144 pair edges are summarized separately.".format(
            prior["corners"]["los_divergent"], catalogue["corners"]["los_divergent"]),
        "", "![Prior support](plots/04_prior_support_and_conditions.png)",
        "", "![Edges and pairs](plots/05_edges_and_combinations.png)",
        "", "Concrete combinations behind the range:", "",
        "| Example | M200c [Msun] | z | Y200 / Battaglia12 | Passes optional slope screen |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("sobol_largest_Y200", "sobol_smallest_Y200", "corners_largest_admitted_Y200"):
        example = examples[name]
        lines.append("| {} | {:.5g} | {} | {:.6g} | {} |".format(name,
            example["mass_Msun"], example["z"], example["Y200_ratio"],
            example["passes_outer_slope_screen"]))
    lines += [
        "", "All nine coordinates for these cases are in "
        "[extreme_parameter_examples.json](audit/extreme_parameter_examples.json). "
        "The largest admitted corner combines high P0 and xc with mass/redshift "
        "growth of both, while decreasing beta toward high mass/redshift. "
        "It passes outer-slope convergence yet has an enormous finite pressure integral. "
        "This demonstrates why the slope screen alone is insufficient.",
        "", "## Physics of the limits", "",
        "The model has P_th/P200 = P0 (x/xc)^(-0.3) (1+x/xc)^(-beta), "
        "with x=r/R200c. Each of P0, xc and beta evolves as "
        "A0 (M200c/1e14 physical Msun)^alpha_m (1+z)^alpha_z. "
        "Electron pressure is 0.5176 times thermal pressure, and Compton y is "
        "sigma_T/(m_e c^2) times its LOS integral.",
        "",
        "Consequently the outer slope is beta(M,z)+0.3: LOS convergence needs "
        "beta>0.7, while a finite untruncated volume integral needs beta>2.7. "
        "A pivot beta0 above those thresholds does not ensure they hold at every "
        "mass and redshift. The 4R200 projected painting cutoff does not truncate "
        "the three-dimensional profile; the original LOS endpoint is 1e5 R200.",
        "", "For steep profiles, (1+x/xc)^(-beta) approaches exp(-beta*x/xc) "
        "in the inner region. The characteristic pressure extent therefore scales "
        "roughly as xc/beta, not xc alone. The finite integrated pressure scales "
        "as P0*xc^3 times a beta-dependent integral. This explains both the "
        "amplitude/size/slope degeneracy and why combinations can be much more "
        "extreme than any single parameter edge.",
        "", "![LOS cutoff](plots/07_line_of_sight_cutoff.png)",
        "", "The fixed projected painting radius imposes a separate extent limit. "
        "At M200c=1e14 Msun and z=0.5:", "",
        "| Profile | Fraction of untruncated projected flux retained inside 4R200 |",
        "|---|---:|"]
    for name in ("Battaglia12", "fit_L1_m9", "fit_fgas-8sigma", "fit_Mstar-1sigma", "xc_high"):
        row = next(r for r in projected if r["name"] == name and r["mass_Msun"] == 1e14 and r["z"] == .5)
        value = row["retained_projected_flux_fraction"]
        lines.append("| {} | {} |".format(name, "undefined infinite total" if value is None else "{:.3%}".format(value)))
    lines += [
        "", "These fractions compare intrinsic profile integrals with the model's "
        "untruncated total; they are not measured C_ell biases or observed missing "
        "gas fractions. The diagnostic is undefined when the untruncated total "
        "diverges. Wider profiles require a physical extent choice and a painting-radius "
        "convergence test as well as a pixel-resolution test. See audit/projected_cutoff.json.",
        "", "## Numerical resolution", "",
        "The pressure checks below compare production interpolation with independently "
        "validated scaled LOS quadrature. They include the exact archived radiation "
        "contribution to H(z). They are unweighted diagnostic-profile errors, not "
        "measured C_ell biases. Extreme high-mass/low-redshift grid points need not "
        "be occupied by the catalogue.", "",
        "| Case | All-grid profile-error p95 | z>=0.1 profile-error p95 |",
        "|---|---:|---:|"]
    for name in ["reference_Battaglia12"]+[result["fits"][v]["best"]["name"] for v in variants]:
        entry = interpolation[name]
        lines.append("| {} | {:.5g}% | {:.5g}% |".format(name,
            100*entry["all_grid"]["p95"], 100*entry["z_ge_0p1"]["p95"]))
    lines += ["", "A bounded quadrature diagnostic found {} native evaluation-budget "
        "hits in {} probes; all {} occurred beyond the painted 4R200 disc. "
        "At those points, the rescaled integral needed {} evaluations. Regular "
        "native/rescaled values agreed to {:.4g} relative. The timeouts therefore "
        "expose a numerical precision problem in very faint grid tails, not a "
        "physical exclusion of those pressure parameters. The temporary beta<=40 "
        "proposal-search cap is a pilot workaround, not a proposed physical upper "
        "bound on beta(M,z). See audit/quadrature_cost_summary.json and "
        "code/diagnose_quadrature.jl.".format(
            quadrature["native_budget_hits"], quadrature["probes"],
            quadrature["budget_hits_outside_painted_disc"],
            quadrature["scaled_counts_at_native_budget_hits"],
            quadrature["maximum_regular_value_relative_disagreement"]),
        "", "The archived Battaglia12 reference reveals which interpolation "
        "errors predate the wider prior. Separate direct-quadrature errors and "
        "zero-versus-positive-reference counts are saved in interpolation_summary.json.",
        "", "![Interpolation](plots/08_interpolation_accuracy.png)", "",
        "Pixel sampling at M200c=1e14 Msun and z=0.5:", "",
        "| Profile | Nside | Median sampled/continuous flux | 16th–84th percentile |",
        "|---|---:|---:|---:|"]
    for name in ("Battaglia12", "compact_steep_evolving"):
        for nside in (4096, 8192):
            row = next(r for r in pixels if r["name"] == name and r["mass_Msun"] == 1e14
                       and r["z"] == .5 and r["nside"] == nside)
            lines.append("| {} | {} | {:.5g} | {:.5g}–{:.5g} |".format(
                name, nside, row["median"], row["p16"], row["p84"]))
    lines += ["", "These are 32 reproducible actual HEALPix placements per case, "
        "not a converged population mean or a full-spectrum resolution test. "
        "Very narrow profiles are sensitive to where their cores land within pixels. "
        "An ensemble mean over uniform random placements can be unbiased in principle, "
        "but these examples do not establish convergence of rare bright alignments "
        "or of the power spectrum. "
        "A normalized Gaussian beam preserves total flux and cannot restore "
        "flux missed at painting.",
        "", "![Pixel sampling](plots/09_pixel_sampling_limits.png)",
        "", "## Lee22 and noise comparisons", "",
        "The [Lee22 pressure formula, Table 1 and Eq.12](https://arxiv.org/html/2205.01710) "
        "adds concentration dependence and an amplitude mass break. The displayed "
        "c=4.5 example uses its electron-pressure normalization and converted break-mass "
        "pivot. Grey radial shading marks its fitted radial interval; other regimes "
        "are extrapolations. The calibration approximately spans 1e13–1e14 h^-1 Msun "
        "and z<=2. It is a literature reference, not an independent "
        "FLAMINGO measurement.",
        "", "![Pressure profiles](plots/06_pressure_profiles_and_lee22.png)",
        "", "![Shared SO noise](plots/03_shared_so_noise.png)",
        "", "The exact seed-12345 mask and both SO noise arrays were verified for "
        "every completed full map. Signal-noise cross terms differ between signal "
        "maps even when the noise pixels are identical. No noisy statistic was fitted.",
        "", "Two independently painted maps differing only in P0 also verified "
        "clean C_ell proportional to P0 squared to {:.3g} maximum relative error. "
        "The noisy cross-spectrum identity, including its terms linear in signal "
        "amplitude and the unchanged noise-noise term, agreed to {:.3g} of peak "
        "noisy power.".format(scaling["maximum_clean_relative_error"],
            scaling["maximum_noisy_absolute_error_over_peak"]),
        "", "## Recommendation before a larger run", "",
        "1. Treat the current rectangle as an exploration envelope. The unrestricted "
        "product distribution is not a production-ready prior.",
        "2. Apply a coupled outer-slope condition. The optional implemented screen "
        "requires min_grid beta>=2.8 and a conservative missing-column bound <=1%. "
        "This is a mathematical/numerical condition, not an observational prior. "
        "An explicitly motivated three-dimensional pressure truncation is an "
        "alternative, but it changes the forward model and requires new validation.",
        "3. Constrain integrated pressure and profile extent jointly using the "
        "reference family and appropriate halo data. A prior in integrated Y200 "
        "and a profile-size coordinate is easier to control than independent P0, "
        "xc and beta. Spectral fits alone do not determine that physical prior.",
        "4. Validate a stable pressure-quadrature/interpolation implementation and "
        "pixel-integrated or demonstrably converged halo painting over the retained "
        "range. The present pilot intentionally preserved historical operators.",
        "5. Generate a valid new training set and retrain/validate SBI for the "
        "changed prior. Recheck whether the existing 40-bin MOPED compression "
        "retains sufficient information across the new range; changing bounds "
        "alone cannot extend the old trained posterior.",
        ""]
    retained = conditional["groups"]["sobol"]
    lines.append("The optional outer-slope screen retains {} / 65,536 draws "
        "({:.3f}%), yet its retained finite-Y200 ratios still span {:.4g}–{:.4g}. "
        "Passing that screen alone therefore does not establish production readiness.".format(
            retained["accepted"], 100*retained["fraction"],
            retained["retained_min_Y200_ratio"], retained["retained_max_Y200_ratio"]))
    lines += ["", "## Resources and reproducibility", "",
        "{} full maps passed the operator/parameter/hash checks. Maximum measured "
        "RSS among completed maps was {:.3f} GiB; median per-map elapsed time was "
        "{:.1f} s. At that pilot throughput, 1,000 serial maps would take about "
        "{:.1f} hours, or {:.0f} allocated core-hours at 26 CPUs. This estimate "
        "includes the pilot's clean/noisy spectra and profile checks, excludes "
        "proposal optimization and failed attempts, and is not a large-job benchmark.".format(
            resources["completed_full_maps"], resources["maximum_map_RSS_GiB"],
            resources["median_map_seconds"], resources["estimated_serial_hours_per_1000_maps"],
            resources["estimated_core_hours_per_1000_maps"]),
        "", "{} candidate maps did not complete; their exit codes and elapsed "
        "times are retained in resource_usage.json. The completed-map throughput "
        "estimate is not applicable to unrestricted draws containing these failures. "
        "The final jobs used mini2 after mini filled; the default PBS queue remains mini.".format(
            len(resources["incomplete_map_attempts"])),
        "", "The retained interpolation caches occupy {:.3f} GiB. The original "
        "FLAMINGO inputs, catalogue and caches remain on the cluster. Repainted "
        "pixel maps were reduced to spectra without retaining their pixel arrays. "
        "PNG, vector PDF, CSV and JSON "
        "artifacts are transferred with SHA256 verification.".format(
            resources["retained_interpolator_cache_bytes"]/1024**3),
        "", "Each retained interpolation cache occupies about {:.2f} MiB. Keeping "
        "one per model would require about {:.1f} GiB for 1,000 models, before "
        "other outputs. A larger campaign needs an explicit cache-retention policy; "
        "the RAM measurement alone is not its storage budget.".format(
            resources["median_interpolator_cache_bytes"]/1024**2,
            resources["estimated_cache_GiB_per_1000_models"]),
        "", "Created source files, purpose and execution commands are listed in "
        "[README](code/README.md). The complete file inventory is changed_files.txt. "
        "Cluster verification is in verification.json and transferred-file verification "
        "is in local_verification.json. No original pipeline or trained-model files were edited.",
        ""]
    (root / "RUN_REPORT.md").write_text("\n".join(lines))
    print("Wrote completed measured report", flush=True)


if __name__ == "__main__":
    main()
