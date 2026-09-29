#!/usr/bin/env python3
"""Part (2): TNG halo gas fractions from the literature, extracted from arXiv sources.

Sources (arXiv e-prints downloaded to the session scratchpad; vector curves read from the original figure PDFs
converted with `pdftocairo -svg`, tick marks used for calibration; numbers from tables/text are copied):
  Ayromlou+23  arXiv:2211.07659 (MNRAS 524, 5391)  Fig. 10 left (f_b_vs_M200_TNG_EAGLE_SIMBA.pdf): median
               f_b(<R200c)/f_b,cosmic vs M200c, TNG, z=0, gas+stars+BH, centrals.
               Fig. 2 top (Profile_rho_baryon_TNG.pdf): mean cumulative f_b(<R)/f_b,cosmic in 1-dex M200c bins, z=0.
               Fig. 6 (R_Closure_z.pdf): median R_c/R200c vs M200 at z = 0, 0.4, 2; with their Table 2
               (TNG, R200c: alpha = 8.08, gamma = -0.44) and eq. R_c/R200c - 1 = alpha (1+z)^gamma (1 - f_b/f_b,cos)
               this gives an INFERRED baryon fraction at z = 0.4 and 2 (valid for log M200c >= 13).
  Pillepich+18b arXiv:1707.03406 (MNRAS 475, 648)  Table 2: log10 M_stars(<R200c) = 0.74 (log10 M200c - 14) + 12.04
               (z=0, 3D, 1e13-1e15 Msun); text: total stellar mass at fixed halo mass essentially invariant z=1->0.
  Pop+22       arXiv:2205.11528  Table (tab:tabmg): SBPL fit M_gas,500 - M_500c, R500 SIM aperture, all sample, z=0:
               alpha1 = 1.671, alpha2 = 1.086, A = 13.093 (log10 Msun at X_norm = 1e14), delta = 0.19,
               X_pivot = 0.57e14 Msun. Hot gas only (T > 1e5 K, no star-forming gas), subhalo-bound gas removed,
               geometric means of 0.1-dex bins.
  Barnes+18    arXiv:1710.08420  Sec. 4.2 text: TNG300 clusters M500 > 2e14 Msun, z=0, median cumulative gas fraction
               0.86 f_b at 0.6 r500 and 0.90 f_b at r500.
  Nelson+24    arXiv:2311.06338 (TNG-Cluster) Sec. 3.1 text: f_gas,500 >~ 0.9 f_b for massive clusters, flat above
               M500c ~ 10^14.5 Msun.
  Davies+20    arXiv:1908.11380 (MNRAS 491, 4462) Sec. 3.1 text: TNG100 median f_CGM ~ 0.55 f_b below 1e12 Msun,
               minimum ~ 0.25 f_b at M200 ~ 10^12.5 Msun, rising at higher mass (z=0).
  Torkamani+26 arXiv:2601.18784 Fig. 5 (TNG gas profiles, z=0.5), integrated in this repo
               (ANALYTIC_HALO_PDF_TORKAMANI_MEDLOCK_20260925.md Sec. 5): 0.21, 0.30, 0.45, 0.57, 0.57 f_b in
               log10(M h) bins 12.5-13 ... 14.5-15 (physical-density reading; the caption says "comoving coordinates").
"""
import json
from pathlib import Path

import numpy as np

import gasfrac_lib as L
import svg_curve_extract as S

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "lee22_normalization_forensics_20260928" / "D_gas_fractions"
LIT = Path("/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/lit")
ORANGE = "rgb(100%,49.803162%,5.490112%)"


def _ticks(paths, axis):
    """Major tick positions (long 2-point black segments) along x (vertical ticks) or y (horizontal ticks)."""
    out = set()
    for p in paths:
        if p["stroke"] != "rgb(0%,0%,0%)" or len(p["xy"]) != 2 or p["width"] != 1.0:
            continue
        (x0, y0), (x1, y1) = p["xy"]
        if axis == "x" and abs(x0 - x1) < 1e-6 and abs(y0 - y1) > 11:
            out.add(round(x0, 3))
        if axis == "y" and abs(y0 - y1) < 1e-6 and abs(x0 - x1) > 11:
            out.add(round(y0, 3))
    return np.array(sorted(out))


def ayromlou_fig10():
    P = S.stroked_paths(LIT / "2211.07659" / "fbM200.svg")
    xt = _ticks(P, "x")          # includes the frame edges; interior ticks are 9..15
    yt = _ticks(P, "y")
    xin = xt[(xt > 100) & (xt < 650)]
    assert len(xin) == 7, xin                       # log M = 9 ... 15
    ax = np.polyfit(xin, np.arange(9, 16), 1)
    yin = yt[(yt > 15) & (yt < 400)]
    assert len(yin) == 8, yin                       # 1.75 ... 0 (top to bottom)
    ay = np.polyfit(yin, np.linspace(1.75, 0.0, 8), 1)
    res = np.max(np.abs(np.polyval(ax, xin) - np.arange(9, 16))), np.max(np.abs(np.polyval(ay, yin) - np.linspace(1.75, 0, 8)))
    curves = [p for p in P if p["stroke"] == ORANGE and p["width"] == 4.5 and len(p["xy"]) > 5]
    assert len(curves) == 1
    xy = curves[0]["xy"]
    lm = np.polyval(ax, xy[:, 0])
    fb = np.polyval(ay, xy[:, 1])
    return dict(log10_m200c_msun=lm.tolist(), fb_over_fbcos=fb.tolist(), tick_fit_max_residual=[float(r) for r in res])


def ayromlou_fig2():
    P = S.stroked_paths(LIT / "2211.07659" / "profTNG.svg")
    xt = _ticks(P, "x")
    xin = xt[(xt > 150) & (xt < 560)]                # R/R200c = 2, 4, 6, 8, 10
    ax = np.polyfit(xin, np.array([2, 4, 6, 8, 10.0]), 1)
    yt = _ticks(P, "y")
    yin = yt[(yt > 40) & (yt < 400)]                 # 1.0 ... 0.0
    ay = np.polyfit(yin, np.linspace(1.0, 0.0, 6), 1)
    names = {"rgb(19.607544%,80.39093%,19.607544%)": "11-12", "rgb(100%,64.704895%,0%)": "12-13",
             "rgb(100%,0%,0%)": "13-14", "rgb(54.508972%,0%,0%)": "14-15"}
    out = {}
    for p in P:
        if p["stroke"] in names and p["width"] == 3.5:
            x = np.polyval(ax, p["xy"][:, 0]); y = np.polyval(ay, p["xy"][:, 1]); o = np.argsort(x)
            out[names[p["stroke"]]] = {f"R={r}": float(np.interp(r, x[o], y[o])) for r in (0.5, 1.0, 1.5, 2.0)}
    return out


def ayromlou_fig6_inferred():
    P = S.stroked_paths(LIT / "2211.07659" / "Rcz.svg")
    xt = _ticks(P, "x"); yt = _ticks(P, "y")
    xin = xt[(xt > 150) & (xt < 660)]
    ax = np.polyfit(xin, np.arange(9, 16), 1)
    yin = yt[(yt > 50) & (yt < 400)]
    ay = np.polyfit(yin, np.linspace(15, 0, 7), 1)
    lab = {None: 0.0, "11.1,4.8": 0.4, "3,4.95": 2.0}
    out = {}
    for p in P:
        if p["stroke"] == ORANGE and len(p["xy"]) > 5:
            z = lab[p["dash"]]
            lm = np.polyval(ax, p["xy"][:, 0]); rc = np.polyval(ay, p["xy"][:, 1])
            beta = 8.08 * (1 + z) ** -0.44
            grid = [13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5]
            rcg = np.interp(grid, lm, rc, left=np.nan, right=np.nan)
            out[str(z)] = dict(log10_m200c_msun=grid, Rc_over_R200c=rcg.tolist(),
                               fb_over_fbcos_inferred=(1 - (rcg - 1) / beta).tolist(), beta=beta)
    return out


def pillepich_fstar_r200c(log10m200c):
    """M_stars(<R200c)/(f_b M200c), TNG z=0 (Pillepich+18b Table 2, 3D, fit valid 1e13-1e15 Msun)."""
    ms = 10 ** (0.74 * (np.asarray(log10m200c) - 14) + 12.04)
    return ms / 10 ** np.asarray(log10m200c) / (L.TNG["ob"] / L.TNG["om"])


def pop22_fgas500(m500):
    """Pop+22 SBPL (SIM aperture, all sample): M_gas,500 [Msun] at M500c [Msun], z=0."""
    a1, a2, A, d, xp, xn = 1.671, 1.086, 13.093, 0.19, 0.57e14, 1e14
    m500 = np.asarray(m500, float)
    y = (m500 / xn) ** ((a1 + a2) / 2) * (np.cosh(np.log10(m500 / xp) / d) / np.cosh(np.log10(xn / xp) / d)) ** (
        (a2 - a1) / 2 * d * np.log(10))
    return 10 ** A * y / m500 / (L.TNG["ob"] / L.TNG["om"])


def main():
    f10 = ayromlou_fig10()
    f2 = ayromlou_fig2()
    f6 = ayromlou_fig6_inferred()
    lm_f10 = np.array(f10["log10_m200c_msun"]); fb_f10 = np.array(f10["fb_over_fbcos"])
    grid = np.array([12.5, 13.0, 13.17, 13.5, 13.57, 13.97, 14.0, 14.37, 14.5, 14.77])
    fb_grid = np.interp(grid, lm_f10, fb_f10)
    fs = pillepich_fstar_r200c(grid)
    tng_gas_z0 = dict(log10_m200c_msun=grid.tolist(), fb_over_fbcos_Ayromlou_fig10_median=fb_grid.tolist(),
                      fstar_over_fbcos_Pillepich18b=fs.tolist(), fgas_over_fbcos=(fb_grid - fs).tolist(),
                      note="gas = median baryons (Ayromlou+23 Fig.10, linear interpolation between the plotted "
                           "0.54-dex spaced medians) minus mean stellar mass inside R200c (Pillepich+18b Table 2; "
                           "extrapolated below 1e13 Msun)")
    m500 = np.array([1e13, 2e13, 3e13, 5e13, 1e14, 2e14, 3e14])
    res = dict(
        ayromlou23_fig10_tng_median_z0=f10,
        ayromlou23_fig2_tng_mean_cumulative_z0=f2,
        ayromlou23_fig6_inferred_baryon_fraction=f6,
        tng_gas_fraction_R200c_z0=tng_gas_z0,
        pop22_fgas500_over_fb_z0=dict(M500c_msun=m500.tolist(), fgas500_over_fb=pop22_fgas500(m500).tolist(),
                                      note="geometric mean, hot non-star-forming gas, subhalo gas excluded"),
        barnes18=dict(sample="TNG300 M500>2e14 Msun, z=0, median", fgas_over_fb_at_0p6r500=0.86, fgas_over_fb_at_r500=0.90),
        nelson24_tngcluster=dict(statement="f_gas,500 >~ 0.9 f_b for massive clusters; flat above M500c ~ 10^14.5 Msun"),
        davies20=dict(sample="TNG100 centrals z=0, median f_CGM", fcgm_below_1e12=0.55, fcgm_min_at_12p5=0.25),
        torkamani26_fig5_repo_integration=dict(log10_mh_bins=["12.5-13", "13-13.5", "13.5-14", "14-14.5", "14.5-15"],
                                              fgas_R200c_over_fb=[0.21, 0.30, 0.45, 0.57, 0.57], z=0.5,
                                              caveat="physical-density reading; caption states comoving coordinates"),
    )
    (OUT / "tng_literature_gas_fractions.json").write_text(json.dumps(res, indent=1))
    print("Ayromlou Fig10 TNG median (logM, fb/fbcos):", np.round(np.c_[lm_f10, fb_f10], 3).tolist())
    print("tick residuals", f10["tick_fit_max_residual"])
    print("Fig2 mean cumulative:", json.dumps(f2))
    print("Fig6 inferred:", json.dumps({z: np.round(v["fb_over_fbcos_inferred"], 3).tolist() for z, v in f6.items()}))
    print("Fig6 Rc:", json.dumps({z: np.round(v["Rc_over_R200c"], 2).tolist() for z, v in f6.items()}))
    print("TNG gas z=0 (grid, fb, fstar, fgas):", np.round(np.c_[grid, fb_grid, fs, fb_grid - fs], 3).tolist())
    print("Pop22 fgas500/fb:", np.round(np.c_[m500 / 1e13, pop22_fgas500(m500)], 3).tolist())


if __name__ == "__main__":
    main()
