#!/usr/bin/env python3
"""tSZ x FRB-DM angular cross-correlation with the updated Battaglia16 / Lee22 implementations.

Reuses the 100,000 source positions, the observed-redshift assignments (71 Planck / 31 ACT
redshifts of Takahashi et al. 2025) and the annular Compton-y samples of the 2026-09-14 test;
only the per-source halo DM (rays/individual_dm_updated.h5, written by
sample_halfdome_updated_sightlines.jl) is new. It adds a source plane with every source at
z = 2, the model kernel of Medlock & Nagai 2026.

Comparisons: Takahashi et al. 2025 Figure 13 (digitized; Planck MILCA 10' and ACT 1.6' beams on
y only) and the Baryon Pasting best-fit curves of Medlock & Nagai 2026 Figure 5 (digitized;
sources at z = 2, no beam). Halo-only partial predictions; no host/IGM DM, noise or masks.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_halfdome_takahashi import read_rows, write_rows, sha256
from compare_takahashi_sightlines import stratified_covariance, decode_text, EDGES

PREVIOUS = Path("frb_map_generation/outputs/takahashi_100k_20260914")
INPUTS = Path("frb_map_generation/outputs/observational_comparison_inputs_20260913")
OUTPUT = Path("frb_map_generation/outputs/tsz_dm_cross_updated_20260917")
FILTERS = {"planck": 10.0, "act": 1.6, "unbeamed": 0.0}
PLANES = {"planck": ("Planck MILCA", "10′ beam", 71), "act": ("ACT", "1.6′ beam", 31)}
PAPER_CUT = {"planck": 10.0, "act": 10 ** .25}
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", ".45"
# label: legend, colour, linestyle, linewidth, marker, in main figures
STYLE = {
    "b16_sphere1": ("Battaglia16, gas inside $R_{200c}$", BLUE, "-", 3.0, "o", True),
    "lee22_noconc_sphere1": ("Lee22 no-c, gas inside $R_{200c}$", ORANGE, "--", 3.0, "s", True),
    "b16_sphere3": ("Battaglia16, gas to $3R_{200c}$", BLUE, (0, (1.2, 1.6)), 1.9, None, True),
    "lee22_noconc_sphere3": ("Lee22 no-c, extrapolated to $3R_{200c}$", ORANGE, (0, (1.2, 1.6)), 1.9, None, True),
    "lee22_legacy_sphere3": ("Lee22 previous reading, $3R_{200c}$", GREY, (0, (3, 2)), 1.6, None, False),
    "lee22_pref_sphere1": ("Lee22 + c (diagnostic), inside $R_{200c}$", "#7B3294", (0, (4, 2)), 1.6, None, False),
}
RC = {"font.size": 15, "axes.labelsize": 18, "axes.titlesize": 17, "xtick.labelsize": 14,
      "ytick.labelsize": 14, "legend.fontsize": 13.5, "axes.linewidth": 1.1}


def column(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def analyze(args):
    out = args.output
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    positions_path = PREVIOUS / "rays/source_positions.h5"
    digest = sha256(positions_path)
    with h5py.File(str(positions_path), "r") as pos, \
            h5py.File(str(out / "rays/individual_dm_updated.h5"), "r") as dmf, \
            np.load(str(PREVIOUS / "rays/annular_y_samples.npz")) as yf:
        if decode_text(dmf.attrs["source_positions_sha256"]) != digest:
            raise ValueError("DM positions mismatch")
        if json.loads(str(yf["metadata_json"].item()))["positions_sha256"] != digest:
            raise ValueError("tSZ positions mismatch")
        if dmf.attrs["catalog_rows_scanned"] != dmf.attrs["catalog_total_rows"]:
            if not args.allow_partial_scan:
                raise ValueError("DM run did not scan the complete catalogue")
            print("WARNING: partial catalogue scan; smoke-test analysis only", flush=True)
        labels = decode_text(dmf.attrs["model_labels"]).split(",")
        planes = decode_text(dmf.attrs["source_planes"]).split(",")
        n = int(dmf.attrs["rays_per_survey"])
        y = {name: yf[name][:] for name in FILTERS}
        rows, arrays, hits = [], {}, {}
        for plane in planes:
            observed = plane in pos
            groups = pos[plane + "/observed_source_index"][:] if observed else np.zeros(n, dtype=int)
            dm = np.column_stack([dmf[plane + "/" + label][:] for label in labels])
            for name in ([plane, "unbeamed"] if observed else ["unbeamed", "planck", "act"]):
                values, covariance = stratified_covariance(dm, y[name], groups)
                errors = np.sqrt(np.maximum(0, np.diag(covariance))).reshape(values.shape)
                arrays[plane + "__" + name + "__covariance"] = covariance
                for p, label in enumerate(labels):
                    for b in range(12):
                        rows.append(dict(plane=plane, filter=name, beam_fwhm_arcmin=FILTERS[name], model=label,
                            theta_lower_arcmin=EDGES[b], theta_upper_arcmin=EDGES[b + 1],
                            theta_arcmin=np.sqrt(EDGES[b] * EDGES[b + 1]),
                            cross_pc_cm3=values[p, b], sampling_sigma_pc_cm3=errors[p, b]))
            inner = dmf[plane + "/foreground_halo_hits_r200c"][:]
            outer = dmf[plane + "/foreground_halo_hits_outer_sphere"][:]
            hits[plane] = dict(zero_hit_fraction_r200c=float(np.mean(inner == 0)),
                               zero_hit_fraction_outer_sphere=float(np.mean(outer == 0)),
                               mean_hits_r200c=float(inner.mean()), mean_hits_outer_sphere=float(outer.mean()),
                               mean_dm_by_model={label: float(dmf[plane + "/" + label][:].mean()) for label in labels})
        regression = {}
        with h5py.File(str(PREVIOUS / "rays/individual_dm.h5"), "r") as old:
            for plane in ("planck", "act"):
                for new, previous in (("b16_sphere3", "battaglia16"), ("lee22_legacy_sphere3", "lee22_legacy")):
                    a, b = dmf[plane + "/" + new][:], old[plane + "/" + previous][:]
                    regression[plane + ":" + new + "_vs_" + previous] = dict(
                        max_abs_difference_pc_cm3=float(np.max(np.abs(a - b))),
                        max_abs_difference_over_peak=float(np.max(np.abs(a - b)) / np.max(np.abs(b))),
                        mean_new=float(a.mean()), mean_previous=float(b.mean()))
        attrs = {key: (decode_text(value) if isinstance(value, bytes) else
                       (value.item() if hasattr(value, "item") else value)) for key, value in dmf.attrs.items()}
    write_rows(out / "analysis/updated_cross_estimates.csv", rows)
    np.savez_compressed(str(out / "analysis/updated_sampling_covariances.npz"), **arrays)
    summary = dict(dm_file_attributes=attrs, hits=hits, regression_against_20260914=regression,
        estimator="equal-stratum unbiased sample covariance of halo-only DM and annular y; z=2 plane is one stratum",
        errors="delete-one-source jackknife on this fixed simulated sky; no map noise, masks or cosmic variance",
        y_samples="unchanged 2026-09-14 annular Compton-y at the same positions (Battaglia12, full lightcone, 4R200c projected)")
    (out / "analysis/updated_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(hits=hits, regression=regression), indent=2))


def load_results(out):
    rows = read_rows(out / "analysis/updated_cross_estimates.csv")
    def select(plane, name, model, theta_min=0.0):
        chosen = [r for r in rows if r["plane"] == plane and r["filter"] == name and r["model"] == model
                  and float(r["theta_arcmin"]) >= theta_min]
        return column(chosen, "theta_arcmin"), column(chosen, "cross_pc_cm3"), column(chosen, "sampling_sigma_pc_cm3")
    return rows, select


def draw_model(ax, x, value, error, label, scale=1e-5):
    legend, color, ls, lw, marker, _ = STYLE[label]
    if marker:
        ax.errorbar(x, value / scale, yerr=error / scale, color=color, ls=ls, lw=lw, marker=marker, ms=6.5,
                    capsize=2.5, elinewidth=1.3, label=legend, zorder=4)
    else:
        ax.plot(x, value / scale, color=color, ls=ls, lw=lw, label=legend, zorder=3)


def style_axis(ax, xmin):
    ax.set_xscale("log")
    ax.set_xlim(xmin, 1000)
    ax.axhline(0, color=".55", lw=.9)
    ax.grid(alpha=.15, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True, length=5)
    ax.set_xlabel(r"$\theta$ [arcmin]")


def ordered_legend(fig, ax, model_labels, extra_labels):
    """Column-wise legend: Battaglia16 pair, Lee22 pair, then the reference entries."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    wanted = [STYLE[m][0] for m in model_labels] + list(extra_labels)
    fig.legend([by_label[w] for w in wanted], wanted, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.6)


def plot_takahashi(out, select, reference="previous"):
    """reference="previous": overlay the 2026-09-14 products (Battaglia16 and legacy Lee22, both with the
    3R200c sphere) as the previous implementation of each inside-R200c curve.
    reference="sphere3": overlay the recomputed updated models with gas to 3R200c instead."""
    observations = read_rows(INPUTS / "digitized/takahashi_fig13_approximate.csv")
    previous = read_rows(PREVIOUS / "analysis/sightline_comparison.csv")
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.9))
    prev_style = {"battaglia16": (BLUE, "Battaglia16, previous implementation"),
                  "lee22_legacy": (ORANGE, "Lee22 no-c, previous implementation")}
    for ax, plane in zip(axes, ("planck", "act")):
        name, beam, count = PLANES[plane]
        obs = [r for r in observations if ("ACT" in r["series"]) == (plane == "act")]
        ax.errorbar(column(obs, "theta_plotted_arcmin"), column(obs, "w_yDM_pc_cm3") / 1e-5,
                    yerr=np.vstack([column(obs, "error_lower_pc_cm3"), column(obs, "error_upper_pc_cm3")]) / 1e-5,
                    fmt="o", color="black", ms=6, capsize=2.5, lw=1.3, label="Takahashi+25 (digitized)", zorder=6)
        if reference == "previous":
            for old, (color, legend) in prev_style.items():
                rows = [r for r in previous if r["survey"] == plane and r["filter"] == plane and r["model"] == old]
                ax.plot(column(rows, "theta_arcmin"), column(rows, "cross_100k") / 1e-5, color=color,
                        ls=(0, (1.2, 1.6)), lw=2.0, label=legend, zorder=3)
        else:
            for label in ("b16_sphere3", "lee22_noconc_sphere3"):
                x, value, error = select(plane, plane, label)
                draw_model(ax, x, value, error, label)
        for label in ("b16_sphere1", "lee22_noconc_sphere1"):
            x, value, error = select(plane, plane, label)
            draw_model(ax, x, value, error, label)
        ax.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        style_axis(ax, 1)
        ax.set_title("{}: {}, {} FRB redshifts".format(name, beam, count), pad=10)
    axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if reference == "previous":
        wanted = [STYLE["b16_sphere1"][0], prev_style["battaglia16"][1], STYLE["lee22_noconc_sphere1"][0],
                  prev_style["lee22_legacy"][1], "Takahashi+25 (digitized)"]
    else:
        wanted = [STYLE[m][0] for m in ("b16_sphere1", "b16_sphere3", "lee22_noconc_sphere1", "lee22_noconc_sphere3")]
        wanted.append("Takahashi+25 (digitized)")
    fig.legend([by_label[w] for w in wanted], wanted, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.6)
    fig.subplots_adjust(left=.07, right=.985, top=.78, bottom=.13, wspace=.2)
    stem = "takahashi_fig13_updated_models" + ("" if reference == "previous" else "_vs_3r200c")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / "plots" / (stem + "." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)


def plot_medlock(out, select, plane):
    fits = read_rows(PREVIOUS / "references/medlock_fig5_best_fit_digitized.csv")
    points = read_rows(INPUTS / "digitized/medlock_nagai_fig5_approximate.csv")
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    panels = (("act", "Takahashi_ACT", "ACT"), ("planck_milca", "Takahashi_Planck_MILCA", "Planck MILCA"),
              ("planck_nilc", "Takahashi_Planck_NILC", "Planck NILC"))
    for ax, (panel, series, title) in zip(axes, panels):
        xmin = 2 if panel == "act" else 10
        pts = [r for r in points if r["series"] == series]
        ax.errorbar(column(pts, "theta_plotted_arcmin"), column(pts, "w_yDM_pc_cm3") / 1e-5,
                    yerr=np.vstack([column(pts, "error_lower_pc_cm3"), column(pts, "error_upper_pc_cm3")]) / 1e-5,
                    fmt="o", color=".6", ms=5, capsize=2, lw=1.1, label="Takahashi+25 as shown in MN26 Fig. 5", zorder=2)
        ref = [r for r in fits if r["survey"] == panel]
        xr, wr, er = column(ref, "theta_arcmin"), column(ref, "w_yDM_pc_cm3"), column(ref, "digitization_absolute_uncertainty")
        ax.plot(xr, wr / 1e-5, color="black", lw=2.6, label="Medlock & Nagai 2026 BP best fit ($z=2$)", zorder=5)
        ax.fill_between(xr, (wr - er) / 1e-5, (wr + er) / 1e-5, color=".3", alpha=.18, lw=0)
        for label in ("b16_sphere3", "lee22_noconc_sphere3", "b16_sphere1", "lee22_noconc_sphere1"):
            x, value, error = select(plane, "unbeamed", label, theta_min=xmin)
            draw_model(ax, x, value, error, label)
        style_axis(ax, xmin)
        ax.set_title(title, pad=10)
    axes[0].set_ylabel(r"$w_{y\,\mathrm{DM}}(\theta)\ \ [10^{-5}\ \mathrm{pc\,cm^{-3}}]$")
    ordered_legend(fig, axes[0], ("b16_sphere1", "b16_sphere3", "lee22_noconc_sphere1", "lee22_noconc_sphere3"),
                   ["Medlock & Nagai 2026 BP best fit ($z=2$)", "Takahashi+25 as shown in MN26 Fig. 5"])
    fig.subplots_adjust(left=.055, right=.99, top=.78, bottom=.13, wspace=.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / "plots" / ("medlock_fig5_updated_models." + ext)), dpi=200 if ext == "png" else None)
    plt.close(fig)


def plot_diagnostic(out, select):
    previous = read_rows(PREVIOUS / "analysis/sightline_comparison.csv")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col", gridspec_kw={"height_ratios": [3, 2]})
    for j, plane in enumerate(("planck", "act")):
        ax, ratio = axes[:, j]
        for label in ("lee22_legacy_sphere3", "lee22_noconc_sphere3", "b16_sphere3", "lee22_pref_sphere1",
                      "lee22_noconc_sphere1", "b16_sphere1"):
            x, value, error = select(plane, plane, label)
            legend, color, ls, lw, marker, _ = STYLE[label]
            ax.errorbar(x, value / 1e-5, yerr=error / 1e-5, color=color, ls=ls, lw=lw, marker=marker or ".", ms=5,
                        capsize=2, label=legend + " (this run)")
        for old, color, legend in (("battaglia16", BLUE, "2026-09-14 Battaglia16 3R200c"),
                                   ("lee22_legacy", GREY, "2026-09-14 Lee22 legacy 3R200c")):
            rows = [r for r in previous if r["survey"] == plane and r["filter"] == plane and r["model"] == old]
            ax.plot(column(rows, "theta_arcmin"), column(rows, "cross_100k") / 1e-5, "x", color=color, ms=9, mew=1.8,
                    label=legend, zorder=7)
        pairs = (("b16_sphere3", "b16_sphere3", "battaglia16", BLUE, "-", "B16 3R200c: rerun / 2026-09-14"),
                 ("lee22_legacy_sphere3", "lee22_legacy_sphere3", "lee22_legacy", GREY, "-", "Lee22 legacy 3R200c: rerun / 2026-09-14"),
                 ("lee22_noconc_sphere3", "lee22_legacy_sphere3", None, ORANGE, ":", "Lee22 updated / previous reading (3R200c)"),
                 ("b16_sphere1", "b16_sphere3", None, BLUE, "--", "B16: inside R200c / to 3R200c"),
                 ("lee22_noconc_sphere1", "lee22_noconc_sphere3", None, ORANGE, "--", "Lee22: inside R200c / to 3R200c"))
        for num, den, old, color, ls, legend in pairs:
            x, a, _ = select(plane, plane, num)
            if old is None:
                _, b, _ = select(plane, plane, den)
            else:
                rows = [r for r in previous if r["survey"] == plane and r["filter"] == plane and r["model"] == old]
                b = column(rows, "cross_100k")
            valid = np.abs(b) > .01 * np.max(np.abs(b))
            ratio.plot(x[valid], 100 * (a[valid] / b[valid] - 1), color=color, ls=ls, marker=".", label=legend)
        name, beam, count = PLANES[plane]
        ax.set_title("{}: {}, {} observed redshifts".format(name, beam, count))
        ax.set_ylabel(r"$w_{y\,\mathrm{DM}}\ [10^{-5}\,\mathrm{pc\,cm^{-3}}]$")
        ratio.set_ylabel("ratio $-$ 1 [%]")
        ratio.set_ylim(-100, 20)
        for a_ in (ax, ratio):
            style_axis(a_, 1)
            a_.axvspan(1, PAPER_CUT[plane], color=".5", alpha=.12, zorder=0)
        ax.legend(loc="upper right", ncol=1)
        ratio.legend(loc="lower left", ncol=1)
    fig.suptitle("Updated models versus the 2026-09-14 sightline products (same rays, same y samples)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(str(out / "plots/updated_vs_previous_diagnostic.png"), dpi=170)
    plt.close(fig)


def write_tables(out, rows, select, plane_z2):
    observations = read_rows(INPUTS / "digitized/takahashi_fig13_approximate.csv")
    fits = read_rows(PREVIOUS / "references/medlock_fig5_best_fit_digitized.csv")
    models = ("b16_sphere1", "lee22_noconc_sphere1", "b16_sphere3", "lee22_noconc_sphere3", "lee22_legacy_sphere3", "lee22_pref_sphere1")
    lines = ["# Annulus means of w_yDM in 1e-5 pc cm^-3 (HalfDome halo-only; +- = finite-source jackknife)", ""]
    table_rows = []
    for plane in ("planck", "act"):
        name, beam, count = PLANES[plane]
        obs = [r for r in observations if ("ACT" in r["series"]) == (plane == "act")]
        lines += ["## Takahashi+25 {}: {}, {} observed redshifts".format(name, beam, count), "",
                  "| annulus [arcmin] | observed | B16 inside R200c | Lee22 no-c inside R200c | B16 to 3R200c | Lee22 no-c to 3R200c | Lee22 previous reading 3R200c | Lee22+c inside R200c (diag.) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        values = {m: select(plane, plane, m) for m in models}
        for b in range(12):
            if EDGES[b] < PAPER_CUT[plane] - 1e-9:
                continue
            o = obs[b]
            cells = ["{:.2f}-{:.2f}".format(EDGES[b], EDGES[b + 1]),
                     "{:.2f} (-{:.2f}/+{:.2f})".format(float(o["w_yDM_pc_cm3"]) / 1e-5, float(o["error_lower_pc_cm3"]) / 1e-5, float(o["error_upper_pc_cm3"]) / 1e-5)]
            record = dict(plane=plane, theta_lower_arcmin=EDGES[b], theta_upper_arcmin=EDGES[b + 1],
                          observed_1e5=float(o["w_yDM_pc_cm3"]) / 1e-5)
            for m in models:
                cells.append("{:.2f} +- {:.2f}".format(values[m][1][b] / 1e-5, values[m][2][b] / 1e-5))
                record[m + "_1e5"] = values[m][1][b] / 1e-5
                record[m + "_sigma_1e5"] = values[m][2][b] / 1e-5
            lines.append("| " + " | ".join(cells) + " |")
            table_rows.append(record)
        lines.append("")
    lines += ["## Medlock & Nagai 2026 Fig. 5 kernel: all sources at z = 2, no beam (HalfDome at the annulus centres; BP best fit interpolated in log theta)", "",
              "| annulus [arcmin] | BP ACT | BP Planck MILCA | BP Planck NILC | B16 inside R200c | Lee22 no-c inside R200c | B16 to 3R200c | Lee22 no-c to 3R200c |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    values = {m: select(plane_z2, "unbeamed", m) for m in models[:4]}
    for b in range(12):
        if EDGES[b] < 10 ** .25 - 1e-9:
            continue
        theta = np.sqrt(EDGES[b] * EDGES[b + 1])
        cells = ["{:.2f}-{:.2f}".format(EDGES[b], EDGES[b + 1])]
        record = dict(plane=plane_z2, theta_lower_arcmin=EDGES[b], theta_upper_arcmin=EDGES[b + 1])
        for panel in ("act", "planck_milca", "planck_nilc"):
            ref = [r for r in fits if r["survey"] == panel]
            xr, wr = column(ref, "theta_arcmin"), column(ref, "w_yDM_pc_cm3")
            if xr.min() <= theta <= xr.max():
                w = float(np.interp(np.log10(theta), np.log10(xr), wr))
                cells.append("{:.2f}".format(w / 1e-5))
                record["bp_" + panel + "_1e5"] = w / 1e-5
            else:
                cells.append("-")
        for m in models[:4]:
            cells.append("{:.2f} +- {:.2f}".format(values[m][1][b] / 1e-5, values[m][2][b] / 1e-5))
            record[m + "_1e5"] = values[m][1][b] / 1e-5
        lines.append("| " + " | ".join(cells) + " |")
        table_rows.append(record)
    (out / "analysis/annulus_tables.md").write_text("\n".join(lines) + "\n")
    keys = sorted({k for r in table_rows for k in r}, key=lambda k: (k not in ("plane", "theta_lower_arcmin", "theta_upper_arcmin"), k))
    write_rows(out / "analysis/annulus_tables.csv", [{k: r.get(k, "") for k in keys} for r in table_rows])


def plot(args):
    out = args.output
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rows, select = load_results(out)
    planes = sorted({r["plane"] for r in rows})
    plane_z2 = [p for p in planes if p not in PLANES][0]
    plot_takahashi(out, select, reference="previous")
    plot_takahashi(out, select, reference="sphere3")
    plot_medlock(out, select, plane_z2)
    plot_diagnostic(out, select)
    write_tables(out, rows, select, plane_z2)
    print("Saved plots and tables under " + str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "plot", "all"), nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--allow-partial-scan", action="store_true", help="smoke tests only")
    args = parser.parse_args()
    if args.stage in ("analyze", "all"):
        analyze(args)
    if args.stage in ("plot", "all"):
        plot(args)


if __name__ == "__main__":
    main()
