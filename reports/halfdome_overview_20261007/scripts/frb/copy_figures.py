#!/usr/bin/env python3
"""Copy the final FRB / tSZ x FRB figures into reports/halfdome_overview_20261007/fig/2_frb/ (section 2 of the overview).

Each PNG is copied as is when it is at most its listed max width and under 1.5 MB, otherwise downscaled with PIL (LANCZOS) to
that width. The vector PDF beside each source is copied unchanged when it exists; a source PNG narrower than 1200 px is
replaced by a pdftoppm render of its PDF (RENDER_FROM_PDF).
Read-only on every source; writes only into the report's fig/2_frb/ folder.
Run: LD_LIBRARY_PATH=<env>/lib <env>/bin/python reports/halfdome_overview_20261007/scripts/frb/copy_figures.py
"""
from pathlib import Path
import shutil
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "reports/halfdome_overview_20261007/fig/2_frb"
O = "frb_map_generation/outputs/"
FIGS = [  # (target name, source PNG relative to the repository root, max width)
    ("01_single_halo_geometry", O + "single_halo_visual_20260917/single_halo_rays_b16_floor_mass_z0p5.png", 2000),
    ("02_tng_window_check", O + "lee22_konietzka_check_20260930/fig_konietzka_check.png", 2000),
    ("03_halo_pdf_zseries_1e13_1e14", O + "lee22_pub_figures_20261001/pub_halo_pdf_zseries.png", 2000),
    ("04_halo_pdf_zseries_1e14_1e15", O + "lee22_pub_figures_20261001/pub_halo_pdf_zseries_m1e14_1e15.png", 2000),
    ("05_halo_pdf_rays_vs_halo_model", O + "lee22_pub_figures_20261001/pub_halo_pdf_analytic.png", 2000),
    ("06_halo_pdf_diagnostic_models", O + "lee22_corrected_20260930/plots/halo_pdf_corrected_lee22_rays_analytic_tng.png", 2000),
    ("07_analytic_tng_reproduction", O + "lee22_analytic_tng_repro_20260930/fig_analytic_tng_repro.png", 2000),
    ("08_frb_source_redshift_kernels", O + "takahashi_cross_comparison_20260913/plots/observed_source_redshifts_and_kernels.png", 2000),
    ("10_takahashi_wydm_pub", O + "lee22_pub_figures_20261001/pub_takahashi.png", 2000),
    ("11_takahashi_wydm_all_pairs", O + "lee22_corrected_20260930/plots/tsz_x_frbdm_corrected_lee22_takahashi.png", 2000),
    ("09_finite_frb_realizations", O + "tsz_dm_fullsky_20260918/cluster_results/frb_map_generation/outputs/tsz_dm_fullsky_20260918/"
     "plots/realizations_selected_battaglia_residuals.png", 2000),
    ("12_yy_power_pub", O + "lee22_pub_figures_20261001/pub_yy_power.png", 1400),
    ("13_yy_power_diagnostic", O + "lee22_corrected_20260930/plots/yy_power_corrected_lee22.png", 1300),
    ("14_tsz_power_mass_split", O + "lee22_pub_figures_20261001/pub_tsz_mass_split.png", 2000),
]
RENDER_FROM_PDF = {"12_yy_power_pub": 400}   # name -> dpi for pdftoppm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src, max_w in FIGS:
        src = ROOT / src
        dst = OUT / (name + ".png")
        im = Image.open(src)
        w, h = im.size
        if w > max_w or src.stat().st_size > 1.5e6:
            nw = min(w, max_w)
            im = im.convert("RGBA") if im.mode in ("P", "LA") else im
            im.resize((nw, round(h * nw / w)), Image.LANCZOS).save(dst, optimize=True)
        else:
            shutil.copyfile(src, dst)
        pdf = src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copyfile(pdf, OUT / (name + ".pdf"))
        if name in RENDER_FROM_PDF and pdf.exists():   # source PNG narrower than 1200 px: render the vector PDF instead
            subprocess.run(["pdftoppm", "-png", "-r", str(RENDER_FROM_PDF[name]), "-singlefile", str(pdf),
                            str(OUT / (name + "_render"))], check=True)
            (OUT / (name + "_render.png")).replace(dst)
        out = Image.open(dst)
        print("%-34s %5d x %-5d -> %5d x %-5d %7.0f kB  pdf=%s  <- %s" % (
            name, w, h, out.size[0], out.size[1], dst.stat().st_size / 1e3, pdf.exists(), src.relative_to(ROOT)))


if __name__ == "__main__":
    main()
