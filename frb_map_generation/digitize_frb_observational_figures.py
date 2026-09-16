#!/usr/bin/env python3
"""Approximate plot-derived reference data, NOT author data or a covariance.

Inputs are the original figure assets from arXiv 2511.02155v2 and 2608.06455v1.
Takahashi points/error bars are read from PDF drawing paths; Medlock Figure 5
uses explicit, visually checked pixel coordinates in its source raster image.
Axis calibrations and extraction resolution are recorded for reproducibility.
No measurement is inferred from a best-fit model or from a residual panel.
"""
import argparse
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BLUE = (0., 0., 1.)
MAGENTA = (1., 0., 1.)
GREEN = (0., 1., 0.)
ORANGE = (1., .650391, 0.)


def download_source_figures(source):
    """Fetch pinned public archives; read only the four known figure members."""
    source.mkdir(parents=True, exist_ok=True)
    for paper, wanted in (("2511.02155v2", ("fig13.pdf", "fig14.pdf", "fig15.pdf")),
                          ("2608.06455v1", ("BP_Model_Fit.png",))):
        if all((source/name).exists() for name in wanted):
            continue
        archive = source/(paper+"_source.tar")
        if not archive.exists():
            with urlopen("https://arxiv.org/src/"+paper, timeout=60) as response:
                archive.write_bytes(response.read())
        with tarfile.open(str(archive), "r:*") as handle:
            for name in wanted:
                members = [m for m in handle.getmembers() if Path(m.name).name == name and m.isfile()]
                if len(members) != 1:
                    raise ValueError("Expected one source figure named " + name)
                # Never extract archive-provided directory paths or execute code.
                (source/name).write_bytes(handle.extractfile(members[0]).read())


def calibration(x_ticks, log_theta, y_ticks, w_in_1e5):
    """Fit physical coordinates to labelled ticks, not the outer plot frame."""
    return dict(x=np.polyfit(x_ticks, log_theta, 1).tolist(),
                y=np.polyfit(y_ticks, np.asarray(w_in_1e5)*1e-5, 1).tolist(),
                x_ticks=x_ticks, log10_theta_ticks=log_theta,
                y_ticks=y_ticks, w_ticks_in_1e5=w_in_1e5)


def data_row(paper, figure, series, index, xy, err_top_bottom, cal,
             coordinate_uncertainty, method, note=""):
    x, y = xy
    top, bottom = err_top_bottom
    if not top <= y <= bottom:
        raise ValueError("Error bar does not enclose marker: " + series)
    slope, intercept = cal["y"]
    theta = 10**np.polyval(cal["x"], x)
    value = slope*y + intercept
    return dict(paper=paper, figure=figure, series=series, bin_index=index,
                theta_plotted_arcmin=theta, w_yDM_pc_cm3=value,
                error_lower_pc_cm3=abs(slope)*(bottom-y),
                error_upper_pc_cm3=abs(slope)*(y-top),
                digitization_w_resolution_pc_cm3=abs(slope)*coordinate_uncertainty,
                digitization_log10_theta_resolution=abs(cal["x"][0])*coordinate_uncertainty,
                source_x=x, source_y=y, source_error_top=top, source_error_bottom=bottom,
                method=method, note=note)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def same_color(first, second):
    return first is not None and np.max(np.abs(np.array(first)-second)) < 1e-4


def vector_series(paths, color, xmin, xmax):
    markers, bars = [], []
    for path in paths:
        rect = path["rect"]
        if (same_color(path.get("fill"), color) and 4 < rect.width < 7
                and abs(rect.width-rect.height) < .02):
            center = ((rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2)
            if xmin < center[0] < xmax:
                markers.append(center)
        # Error-bar collection, excluding individual overplotted symbol strokes.
        if len(path["items"]) != 12 or path.get("width", 0) < 1:
            continue
        for item in path["items"]:
            if item[0] == "l" and abs(item[1].x-item[2].x) < .01:
                x, a, b = item[1].x, item[1].y, item[2].y
                if xmin < x < xmax:
                    bars.append((x, min(a, b), max(a, b)))
    markers.sort()
    if len(markers) != 12:
        raise ValueError("Expected 12 vector markers, got {}".format(len(markers)))
    result = []
    for x, y in markers:
        matches = [bar for bar in bars if abs(bar[0]-x) < .02]
        if len(matches) != 1:
            raise ValueError("Ambiguous error bar at x={}".format(x))
        result.append(((x, y), matches[0][1:]))
    return result


def overlay(image, rows, output, title, extent=None):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(image, extent=extent)
    for row in rows:
        x, y = row["source_x"], row["source_y"]
        ax.plot(x, y, marker="+", color="#00b8b8", ms=5, mew=.9)
        ax.plot([x, x], [row["source_error_top"], row["source_error_bottom"]],
                color="#00b8b8", alpha=.6, lw=.6)
    ax.set_axis_off()
    ax.set_title(title + "\nCyan marks: digitized centers and visible error-bar extents", fontsize=11)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_takahashi(source, output):
    import pymupdf
    planck = calibration([39.15, 86.75, 134.35, 181.95], [0, 1, 2, 3],
                         [155.20, 123.65, 92.10, 60.60, 29.05], [-2, 0, 2, 4, 6])
    act = calibration([276.75, 324.35, 371.95, 419.55], [0, 1, 2, 3],
                      [132.55, 96.35, 60.10, 23.90], [0, 4, 8, 12])
    mask = calibration([96.60, 164.60, 232.55, 300.55], [0, 1, 2, 3],
                       [186.15, 152.15, 118.10, 84.05, 50.05], [-2, 0, 2, 4, 6])
    pr4 = calibration([96.60, 164.60, 232.55, 300.55], [0, 1, 2, 3],
                      [211.70, 172.50, 133.30, 94.10, 54.95], [-2, 0, 2, 4, 6])
    configurations = {
        13: [("Planck_MILCA_71", BLUE, 34, 186, planck),
             ("ACT_31", BLUE, 272, 424, act)],
        14: [("MILCA_mask40_71", BLUE, 90, 306, mask),
             ("MILCA_mask50_66", MAGENTA, 90, 306, mask),
             ("MILCA_mask60_54", GREEN, 90, 306, mask),
             ("MILCA_mask70_42", ORANGE, 90, 306, mask)],
        15: [("Planck_PR2_MILCA", BLUE, 90, 306, pr4),
             ("Planck_PR4_no_deprojection", MAGENTA, 90, 306, pr4),
             ("Planck_PR4_CIB_deprojection", GREEN, 90, 306, pr4)]}
    all_rows, curves = [], []
    for figure, series in configurations.items():
        page = pymupdf.open(source/("fig{}.pdf".format(figure)))[0]
        paths = page.get_drawings()
        rows = []
        for name, color, xmin, xmax, cal in series:
            for i, (xy, ends) in enumerate(vector_series(paths, color, xmin, xmax)):
                row = data_row("2511.02155v2", figure, name, i, xy, ends, cal, .10,
                               "PDF_vector_axis_calibrated",
                               "Plotted x includes display offsets; use bin edges for model averaging")
                # The paper specifies twelve logarithmic bins, not the shifted
                # series marker positions. Do not silently identify the two.
                row.update(theta_bin_lower_arcmin=10**(.25*i),
                           theta_bin_upper_arcmin=10**(.25*(i+1)))
                rows.append(row)
        all_rows.extend(rows)
        write_rows(output/("takahashi_fig{}_approximate.csv".format(figure)), rows)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
        raster = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        overlay(raster, rows, output/("takahashi_fig{}_digitization_QA.png".format(figure)),
                "Takahashi Figure {}: original figure + extraction check".format(figure),
                extent=(0, page.rect.width, page.rect.height, 0))
        if figure == 13:
            for path in paths:
                color = path.get("color")
                family = ("HMx_isothermal" if same_color(color, (.75, .5, 1.)) else
                          "TNG_isothermal" if same_color(color, ORANGE) else None)
                if family is None or len(path["items"]) < 15:
                    continue
                temperature = 1e7 if path["width"] < 1 else 3e7
                cal = planck if path["rect"].x0 < 200 else act
                survey = "Planck" if cal is planck else "ACT"
                points = [item[1] for item in path["items"] if item[0] == "l"]
                points.append(path["items"][-1][2])
                for point in points:
                    curves.append(dict(survey=survey, reference=family, temperature_K=temperature,
                                       theta_arcmin=10**np.polyval(cal["x"], point.x),
                                       w_yDM_pc_cm3=np.polyval(cal["y"], point.y),
                                       role="published reference ONLY; HalfDome tSZ remains Battaglia12"))
    write_rows(output/"takahashi_fig13_temperature_reference_curves.csv", curves)
    return all_rows, {"fig13_Planck": planck, "fig13_ACT": act, "fig14": mask, "fig15": pr4}


def extract_medlock(source, output):
    image = np.array(Image.open(source/"BP_Model_Fit.png").convert("RGB"))
    if image.shape[:2] != (676, 1704):
        raise ValueError("Pixel coordinates apply only to the original 1704 x 676 figure")
    # Coordinates (x, marker y, upper error endpoint y, lower endpoint y).
    # Manually read from the source image, not from a rescaled screenshot.
    # Overlapping grey points in the left panel have lower extraction fidelity.
    sharma_x = [121, 196, 271, 347, 421]
    entries = [
        ("Sharma_DMcut500", 0, sharma_x, [249, 359, 361, 361, 360],
         [197, 316, 331, 342, 348], [301, 403, 391, 380, 372]),
        ("Sharma_DMcut750", 0, sharma_x, [237, 335, 359, 357, 361],
         [153, 268, 320, 335, 342], [321, 401, 399, 379, 379]),
        ("Sharma_DMcut1000", 0, sharma_x, [183, 279, 352, 332, 347],
         [83, 182, 295, 300, 317], [284, 376, 410, 364, 377]),
        ("Takahashi_ACT", 1, [515, 547, 578, 609, 640, 671, 703, 734, 765, 796, 826],
         [248, 304, 289, 193, 291, 353, 291, 332, 353, 377, 386],
         [85, 234, 185, 83, 216, 304, 238, 290, 317, 353, 363],
         [411, 375, 393, 301, 367, 401, 344, 374, 388, 401, 408]),
        ("Takahashi_Planck_MILCA", 2, [932, 976, 1019, 1063, 1107, 1151, 1194, 1237],
         [173, 274, 351, 343, 373, 385, 394, 399],
         [83, 201, 303, 324, 354, 371, 385, 393],
         [262, 345, 399, 364, 393, 399, 404, 405]),
        ("Takahashi_Planck_NILC", 3, [1344, 1388, 1432, 1475, 1519, 1562, 1606, 1650],
         [174, 286, 367, 343, 375, 391, 388, 383],
         [83, 219, 325, 319, 356, 381, 381, 377],
         [264, 352, 411, 365, 394, 401, 394, 390])]
    calibrations = [
        calibration([131, 384], [1, 2], [100, 152, 204, 255, 307, 358, 410], [5, 4, 3, 2, 1, 0, -1]),
        calibration([594, 718, 843], [1, 2, 3], [100, 171, 241, 312, 382], [8, 6, 4, 2, 0]),
        calibration([909, 1084, 1260], [1, 2, 3], [77, 133, 189, 244, 300, 355, 411], [6, 5, 4, 3, 2, 1, 0]),
        calibration([1320, 1496, 1672], [1, 2, 3], [118, 184, 250, 316, 382], [4, 3, 2, 1, 0])]
    rows = []
    for name, panel, xs, ys, tops, bottoms in entries:
        for i, (x, y, top, bottom) in enumerate(zip(xs, ys, tops, bottoms)):
            note = ("Some grey symbols/error bars overlap; approximate readings only" if panel == 0 else
                    "Reuses Takahashi observations; not an independent dataset")
            row = data_row("2608.06455v1", 5, name, i, (x, y), (top, bottom),
                           calibrations[panel], 3., "manual_source_raster_pixels", note)
            rows.append(row)
    write_rows(output/"medlock_nagai_fig5_approximate.csv", rows)
    overlay(image, rows, output/"medlock_nagai_fig5_digitization_QA.png",
            "Medlock & Nagai Figure 5: approximate measurement extraction")
    return rows, calibrations


def plot_reference_points(output, takahashi, medlock):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    groups = [([r for r in takahashi if r["figure"] == 13], "Takahashi Fig. 13: Planck and ACT"),
              ([r for r in takahashi if r["figure"] == 14], "Takahashi Fig. 14: Galactic masks"),
              ([r for r in takahashi if r["figure"] == 15], "Takahashi Fig. 15: Planck PR4"),
              ([r for r in medlock if r["series"].startswith("Sharma")], "Medlock Fig. 5: CHIME DM cuts"),
              ([r for r in medlock if r["series"] == "Takahashi_ACT"], "Medlock Fig. 5: ACT"),
              ([r for r in medlock if "Takahashi_Planck" in r["series"]], "Medlock Fig. 5: Planck")]
    for ax, (rows, title) in zip(axes.flat, groups):
        for label in dict.fromkeys(r["series"] for r in rows):
            selected = [r for r in rows if r["series"] == label]
            ax.errorbar([r["theta_plotted_arcmin"] for r in selected],
                        [r["w_yDM_pc_cm3"]/1e-5 for r in selected],
                        yerr=[[r[k]/1e-5 for r in selected]
                              for k in ("error_lower_pc_cm3", "error_upper_pc_cm3")],
                        fmt="o", ms=3, lw=1, capsize=2, label=label.replace("_", " "))
        ax.set_xscale("log")
        ax.axhline(0, color="black", lw=.6)
        ax.set_xlabel(r"$\theta$ [arcmin; plotted positions]")
        ax.set_ylabel(r"$w_{y\mathrm{DM}}$ [$10^{-5}$ pc cm$^{-3}$]")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    fig.suptitle("Approximate published points and visible error bars\nNo HalfDome prediction or covariance fit in this figure", fontsize=14)
    fig.savefig(output/"observational_approximate_datapoints.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pdf-tools", type=Path, help="Optional isolated PyMuPDF installation")
    parser.add_argument("--download-sources", action="store_true", help="Fetch missing pinned public figure assets")
    args = parser.parse_args()
    if args.pdf_tools:
        sys.path.insert(0, str(args.pdf_tools.resolve()))
    if args.download_sources:
        download_source_figures(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    takahashi, vector_axes = extract_takahashi(args.source_dir, args.output_dir)
    medlock, raster_axes = extract_medlock(args.source_dir, args.output_dir)
    plot_reference_points(args.output_dir, takahashi, medlock)
    sources = [args.source_dir/f for f in ("fig13.pdf", "fig14.pdf", "fig15.pdf", "BP_Model_Fit.png")]
    metadata = dict(
        source_urls=["https://arxiv.org/src/2511.02155v2", "https://arxiv.org/src/2608.06455v1"],
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        n_takahashi_points=len(takahashi), n_medlock_points=len(medlock),
        vector_axis_calibrations=vector_axes, raster_axis_calibrations=raster_axes,
        status="Approximate plot-derived data, not author-supplied measurements",
        extraction_precision="PDF: 0.10 point coordinate resolution; raster: approx 3 pixels, worse if occluded; not statistical confidence intervals",
        covariance="NOT available; visible error bars do not supply off-diagonal covariance; no formal chi-square or detection significance",
        duplicates="Planck/ACT points reused across figures are not independent observations",
        theory="Temperature curves are published references only. HalfDome tSZ uses Battaglia12 pressure; DM is halo-only, a partial prediction",
        scope="Upper-panel measurements only; lower residuals and model best-fit curves are not measured data")
    (args.output_dir/"digitization_provenance.json").write_text(json.dumps(metadata, indent=2))
    print("Saved {} Takahashi and {} Medlock plot-derived points to {}".format(
        len(takahashi), len(medlock), args.output_dir))


if __name__ == "__main__":
    main()
