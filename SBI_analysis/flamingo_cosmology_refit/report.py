"""Export fitted values, validation evidence and presentation-ready figures.

Run on the cluster after all three refit jobs finish. Once comparison_spectra.npz
has been exported, the figures and tables can be regenerated from that compact
result bundle without access to the original maps or CAMB.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")
PARAMETERS = ("P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc",
              "alpha_m_beta", "alpha_z_P0", "alpha_z_xc", "alpha_z_beta")
COLORS = ("#0072B2", "#D55E00", "#009E73")
DEFAULT_CAMPAIGN = Path("/lustre/work/kristero10/flamingo_tsz_comparison_20260914")
DEFAULT_PILOT = Path("/lustre/work/kristero10/flamingo_prior_pilot_20260914")
EDGES = np.append(np.arange(80, 7881, 200), 7980)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bin_cl(cl):
    ell = np.arange(7980)
    dl = cl * ell * (ell + 1) / (2 * np.pi)
    return np.array([np.average(dl[a:b], weights=2*ell[a:b]+1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])])


def selected_path(root, variant, fit):
    name = fit["best"]["name"]
    filename = variant + "_amplitude" if name == "amplitude_from_existing_map" else name
    return root / "results" / (filename + ".npz")


def prepare_spectra(root, fits, campaign, pilot):
    destination = root / "results/comparison_spectra.npz"
    if destination.exists():
        return dict(np.load(destination))
    ell = np.arange(7980)
    spectra = {"ell": np.array([np.average(ell[a:b], weights=2*ell[a:b]+1)
                               for a, b in zip(EDGES[:-1], EDGES[1:])])}
    for variant, fit in fits.items():
        selected = dict(np.load(selected_path(root, variant, fit)))
        np.testing.assert_array_equal(selected["theta"], fit["best"]["theta"])
        for key in ("target", "hd_clean", "response", "corrected_clean"):
            spectra[variant + "/" + key] = selected[key]
        old = pilot / "results" / fit["original"]["name"] / "masked_clean_cl.npy"
        spectra[variant + "/old_hd_clean"] = bin_cl(np.load(old))
        noisy = campaign / "results" / variant / "masked_noisy_cross_cl.npy"
        spectra[variant + "/observation_noisy"] = bin_cl(np.load(noisy))
    spectra["Battaglia12/hd_clean"] = bin_cl(np.load(
        campaign / "results/HalfDome/masked_clean_cl.npy"))
    np.savez(destination, **spectra)
    return spectra


def save_figure(figure, root, name):
    for extension in ("png", "pdf"):
        figure.savefig(root / "plots" / (name + "." + extension), dpi=220,
                       bbox_inches="tight")
    plt.close(figure)


def make_figures(root, spectra, fits):
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17,
                         "axes.titlesize": 17, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 12,
                         "lines.linewidth": 2.2, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    ell = spectra["ell"]
    figure, axes = plt.subplots(2, 3, figsize=(16.8, 8.0), sharex=True,
                               gridspec_kw={"height_ratios": [2.4, 1]})
    for column, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
        top, bottom = axes[:, column]
        target = spectra[variant + "/target"]
        corrected = spectra[variant + "/corrected_clean"]
        old = spectra[variant + "/old_hd_clean"]
        top.loglog(ell, 1e12*target, color="black", label="FLAMINGO")
        top.loglog(ell, 1e12*corrected, color=color, ls="--", label="New fit + cosmology")
        top.loglog(ell, 1e12*old, color="0.55", ls=":", label="Previous HalfDome fit")
        top.loglog(ell, 1e12*spectra["Battaglia12/hd_clean"],
                   color="#CC79A7", ls="-.", alpha=.8, label="Battaglia12, HalfDome")
        top.set_title(variant)
        top.legend(frameon=False, loc="lower left")
        bottom.axhline(0, color="0.3", lw=1)
        bottom.axhspan(-5, 5, color="0.8", alpha=.2)
        bottom.semilogx(ell, 100*(corrected/target-1), color=color)
        bottom.semilogx(ell, 100*(old/target-1), color="0.55", ls=":")
        bottom.set_xlabel(r"$\ell$")
        bottom.set_ylim(-10, 10)
        for axis in (top, bottom):
            axis.grid(alpha=.16)
    axes[0, 0].set_ylabel(r"$10^{12}D_\ell^{yy}$")
    axes[1, 0].set_ylabel("Residual [%]")
    figure.tight_layout(w_pad=2)
    save_figure(figure, root, "cosmology_corrected_spectra")

    references = np.load(root / "audit/reference_responses.npz")
    figure, axis = plt.subplots(figsize=(9, 5.7))
    axis.axhline(1, color="0.4", lw=1)
    axis.semilogx(ell, references["Battaglia12"], color="black", ls="--", label="Battaglia12")
    for variant, color in zip(VARIANTS, COLORS):
        axis.semilogx(ell, references[variant], color=color, label=variant)
    axis.set_xlabel(r"$\ell$")
    axis.set_ylabel(r"$D_\ell^{\mathrm{D3A}}/D_\ell^{\mathrm{HalfDome}}$")
    axis.set_ylim(.90, 1.005)
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=.2)
    figure.tight_layout()
    save_figure(figure, root, "cosmology_response_at_previous_fits")


def make_tables(root, fits, spectra):
    rows = []
    metrics = []
    for variant, fit in fits.items():
        before, after = fit["original"], fit["best"]
        amplitude = next(candidate for candidate in fit["candidates"]
                         if candidate["name"] == "amplitude_from_existing_map")
        proposal_path = root / "proposals" / (after["name"] + ".json")
        proposal = json.loads(proposal_path.read_text()) if proposal_path.exists() else None
        for name, old, new in zip(PARAMETERS, before["theta"], after["theta"]):
            rows.append(dict(variant=variant, parameter=name, previous=old,
                             cosmology_corrected=new, difference=new-old))
        reference = np.load(root / "audit/reference_responses.npz")[variant]
        old_at_new_cosmology = spectra[variant + "/old_hd_clean"] * reference
        target = spectra[variant + "/target"]
        rms_unadjusted = float(np.sqrt(np.mean((old_at_new_cosmology/target-1)**2)))
        support_path = root / "results" / (variant + "_support.json")
        support = json.loads(support_path.read_text()) if support_path.exists() else None
        metrics.append(dict(variant=variant, selected=after["name"],
            previous_rms_fractional=before["rms_fractional"],
            previous_max_abs_fractional=before["max_fractional"],
            old_theta_with_cosmology_rms_fractional=rms_unadjusted,
            corrected_rms_fractional=after["rms_fractional"],
            corrected_max_abs_fractional=after["max_fractional"],
            corrected_response_min=float(spectra[variant + "/response"].min()),
            corrected_response_max=float(spectra[variant + "/response"].max()),
            P0_ratio=after["theta"][0]/before["theta"][0],
            amplitude_only_P0=amplitude["theta"][0],
            amplitude_only_P0_ratio=amplitude["theta"][0]/before["theta"][0],
            amplitude_only_rms_fractional=amplitude["rms_fractional"],
            selected_optimizer_success=proposal["success"] if proposal else None,
            selected_optimizer_message=proposal["message"] if proposal else None,
            support=support))
    with (root / "results/parameter_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "results/comparison_summary.json", dict(
        parameter_order=PARAMETERS, variants=metrics,
        corrected_parameters={v: f["best"]["theta"] for v, f in fits.items()},
        fits_are_posterior_estimates=False, noise_used_for_fitting=False,
        original_8192_production_modified=False))

    lines = ["# Cosmology-corrected FLAMINGO spectral fits", "",
        "These are local, approximate spectral matches at FLAMINGO D3A cosmology. "
        "They are not direct halo-pressure measurements, posterior estimates, or a "
        "validated conversion between N-body realizations.", "",
        "The original 8,192-row production configuration and prior were not changed.", "",
        "| Parameter | L1_m9 | fgas-8sigma | Mstar-1sigma |",
        "|---|---:|---:|---:|"]
    for index, name in enumerate(PARAMETERS):
        values = [fits[v]["best"]["theta"][index] for v in VARIANTS]
        lines.append("| " + name + " | " + " | ".join(f"{x:.6g}" for x in values) + " |")
    lines += ["", "| Variant | Previous RMS / maximum | New RMS / maximum | Selected candidate |",
              "|---|---:|---:|---|"]
    for metric in metrics:
        lines.append("| {variant} | {a:.2f}% / {b:.2f}% | {c:.2f}% / {d:.2f}% | {selected} |".format(
            **metric, a=100*metric["previous_rms_fractional"],
            b=100*metric["previous_max_abs_fractional"],
            c=100*metric["corrected_rms_fractional"],
            d=100*metric["corrected_max_abs_fractional"]))
    lines += ["", "RMS and maximum residuals refer to the 40 clean, beam-smoothed, masked "
              "D_ell bins relative to the corresponding FLAMINGO spectrum. "
              "The new prediction is a full HalfDome map spectrum multiplied by a "
              "parameter-dependent one-plus-two-halo cosmology ratio.", "",
              "| Variant | P0 with other eight coefficients fixed | Final proposal converged | Maximum beta on interpolation domain |",
              "|---|---:|---|---:|"]
    for metric in metrics:
        maximum_beta = metric["support"]["metrics"]["beta_max"] if metric["support"] else float("nan")
        lines.append("| {variant} | {amplitude_only_P0:.6g} | {selected_optimizer_success} | {beta:.6g} |".format(
            **metric, beta=maximum_beta))
    lines += ["", "The full refits can move along parameter degeneracies, and use the current "
              "beta ceiling of 50; the older fgas-8sigma and Mstar-1sigma fits used 40, "
              "whereas the retained older L1_m9 fit used 64. "
              "Their parameter changes and fit improvements therefore cannot all be "
              "attributed to cosmology. An optimizer iteration limit is not convergence; "
              "the reported candidate is selected by its checked full-map residual.", "",
              "See ../README.md for the equations, numerical checks, assumptions, "
              "sources and implications for SBI training.", ""]
    (root / "results/RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(metrics, indent=2))


def main(root, campaign, pilot):
    (root / "plots").mkdir(exist_ok=True)
    fits = {v: json.loads((root / "results" / (v + "_fit.json")).read_text())
            for v in VARIANTS}
    spectra = prepare_spectra(root, fits, campaign, pilot)
    for variant in VARIANTS:
        np.testing.assert_allclose(spectra[variant + "/corrected_clean"],
            spectra[variant + "/hd_clean"] * spectra[variant + "/response"], rtol=1e-13)
    make_tables(root, fits, spectra)
    make_figures(root, spectra, fits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    arguments = parser.parse_args()
    main(arguments.root, arguments.campaign, arguments.pilot)
