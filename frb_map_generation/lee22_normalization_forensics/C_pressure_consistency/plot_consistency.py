#!/usr/bin/env python3
"""Figures for the Lee22 internal-consistency forensics (task C). Reads the CSV/JSON products of
run_consistency.py and the extracted paper curves; recomputes profiles with lee22_models."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lee22_models as L
from run_consistency import XCOP_X, XCOP_NE, OUT, load_curves

C = L.TNG
h = C.h
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # reference categorical slots 1,2,3,7
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "legend.frameon": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def read_csv(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def fig_temperature():
    T = read_csv("temperature_grid.csv")
    B = read_csv("temperature_grid_battaglia_benchmark.csv")
    R = json.loads((OUT / "consistency_results.json").read_text())
    fam_key = [k for k in T[0].keys() if k.startswith("fit_family")][0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for col, lm in zip((S1, S2, S3), ("13.3", "13.9", "14.5")):
        rows = [r for r in T if r[fam_key] == "noc" and abs(float(r["log10_M200c_hinv"]) - float(lm)) < 1e-6]
        z = np.array([float(r["z"]) for r in rows]); t = np.array([float(r["Tnw_over_T200_within_R200"]) for r in rows])
        ax[0].plot(1 + z, t, color=col, label=f"Lee22 Tables 7+8, log M={lm}")
        rb = [r for r in B if abs(float(r["log10_M200c_hinv"]) - float(lm)) < 1e-6]
        ax[0].plot(1 + np.array([float(r["z"]) for r in rb]), [float(r["Tnw_over_T200_within_R200"]) for r in rb],
                   color=col, ls="--", label=f"B12 P_e / B16 n_e, log M={lm}")
    A = R["A_figure_forensics"]
    ax[0].plot([1, 1], [A["panel0_log10Mh_13.2-13.4"]["measured_sim_Tnw_over_T200_x0.05_1"],
                        A["panel1_log10Mh_13.8-14.0"]["measured_sim_Tnw_over_T200_x0.05_1"]],
               ls="none", marker="o", ms=8, mfc=INK, mec="#fcfcfb", mew=2,
               label="paper's own TNG stacks (Fig. 2 / Fig. 5 red curves), z=0")
    zz = np.linspace(0, 2, 50)
    ax[0].plot(1 + zz, 0.147 * (1 + zz), color=INK2, ls=":", lw=1.5, label="0.147 (1+z)$^{+1}$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xticks([1, 1.5, 2, 3]); ax[0].set_xticklabels(["1", "1.5", "2", "3"])
    ax[0].set_xlabel("1 + z"); ax[0].set_ylabel(r"$T_{e}/T_{200}$ (number-weighted, $r<R_{200c}$)")
    ax[0].set_title("Implied electron temperature", color=INK, loc="left")
    ax[0].legend(fontsize=7.2, loc="center left", bbox_to_anchor=(0.0, 0.47))
    # absolute evolution of the enclosed electron content and Y, normalised to z=0
    I = read_csv("integrated_ratios_vs_battaglia.csv")
    for col, lm in zip((S1, S2, S3), ("13.5", "14.0", "14.5")):
        rows = [r for r in I if abs(float(r["log10_M200c_msun"]) - float(lm)) < 1e-6]
        z = np.array([float(r["z"]) for r in rows])
        fl = np.array([float(r["Mgas(<R200c)/(f_b M200)_Lee_noc"]) for r in rows])
        fb16 = np.array([float(r["Mgas/(f_b M200)_B16"]) for r in rows])
        yl = np.array([float(r["Y/(P200 V200)_Lee_noc"]) for r in rows])
        ax[1].plot(1 + z, fl / fl[0], color=col, label=f"Lee22 n: $M_{{gas}}(<R_{{200c}})$, log M={lm} Msun")
        ax[1].plot(1 + z, yl / yl[0], color=col, ls="-.", label=f"Lee22 P: $Y/(P_{{200}}V_{{200}})$, log M={lm}")
        if lm == "14.0":
            ax[1].plot(1 + z, fb16 / fb16[0], color=INK2, ls="--", lw=1.5, label="B16 $M_{gas}$ (physical), log M=14")
    ax[1].plot(1 + zz, (1 + zz) ** -3, color=INK, ls=":", lw=1.2, label="(1+z)$^{-3}$")
    ax[1].plot(1 + zz, (1 + zz) ** -2, color=INK2, ls=":", lw=1.2, label="(1+z)$^{-2}$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xticks([1, 1.5, 2, 3]); ax[1].set_xticklabels(["1", "1.5", "2", "3"])
    ax[1].set_xlabel("1 + z"); ax[1].set_ylabel("value / value at z = 0 (fixed M200c)")
    ax[1].set_title("Redshift evolution in self-similar units", color=INK, loc="left")
    ax[1].legend(fontsize=7, loc="lower left", ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_temperature_and_evolution.{ext}", dpi=150)
    plt.close(fig)


def fig_observations():
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    s = np.geomspace(0.03, 1.5, 200)
    for j, (label, m200) in enumerate((("$M_{200c}=10^{14}\\,M_\\odot$", 1e14), ("$M_{200c}=10^{15}\\,h^{-1}M_\\odot$", 1e15 / h))):
        z = 0.0
        c200 = L.tng_mean_c(m200 * h, z)
        m5, s5 = L.nfw_m_ratio(c200)
        x = s * s5
        n200, P200 = C.n200(z), C.P200(m200, z) / L.KEV
        ne_lee = L.lee_nbar(x, m200 * h, z) * n200
        ne_b16 = L.b16_nbar(x, m200, z) * n200
        ne_x = np.exp(np.interp(np.log(s), np.log(XCOP_X), np.log(XCOP_NE)))
        a = ax[0, j]
        a.plot(s, ne_lee, color=S1, label="Lee22 Table 8, eq. 9 (literal)")
        a.plot(s, ne_lee * 0.6016316602183194, color=S1, ls="--", lw=1.5, label="Lee22 x 0.602 (HalfDome xgpnative)")
        a.plot(s, ne_b16, color=S2, label="Battaglia16 (XGPaint, f_b, 0.9)")
        a.plot(s, ne_x, color=S3, label="X-COP median, $M_{500}$=3.5-9e14 (Ghirardini+19)")
        a.set_yscale("log"); a.set_xscale("log")
        a.set_ylabel(r"$n_e$ [cm$^{-3}$]"); a.set_title(f"{label}, z = 0", color=INK, loc="left")
        Pe_lee = L.lee_Pbar(x, m200 * h, z) * P200
        Pe_b12 = L.b12_Pe_over_P200(x, m200, z) * P200
        Pe_upp = L.arnaud10_upp_P(s, m5 * m200, z, C)
        b = ax[1, j]
        b.plot(s, Pe_lee, color=S1, label="Lee22 Table 7")
        b.plot(s, Pe_b12, color=S2, label="Battaglia12 x 0.5176")
        b.plot(s, Pe_upp, color=S3, label="Arnaud+10 UPP")
        b.set_yscale("log"); b.set_xscale("log")
        b.set_ylabel(r"$P_e$ [keV cm$^{-3}$]"); b.set_xlabel(r"$r/R_{500c}$")
        if j == 0:
            a.legend(fontsize=7.5); b.legend(fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_physical_profiles_vs_observations.{ext}", dpi=150)
    plt.close(fig)


def fig_reference_curves():
    cur = load_curves()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    lm = 13.9
    for i, (asset, ttl) in enumerate((("fig_4.pdf", "Lee22 Fig. 5 (density), 10$^{13.8-14.0}h^{-1}$, z=0"),
                                       ("fig_2.pdf", "Lee22 Fig. 2 (pressure), 10$^{13.8-14.0}h^{-1}$, z=0"))):
        x, sim = cur[(asset, 1, "simulation")]
        _, cyan = cur[(asset, 1, "battaglia")]
        _, orange = cur[(asset, 1, "without_c_bpl")]
        a = ax[i]
        a.plot(x, sim, color=INK2, label="paper: TNG stack (red curve)")
        a.plot(x, cyan, color=S1, lw=3, alpha=0.45, label="paper: Battaglia curve as drawn (cyan)")
        if asset == "fig_4.pdf":
            m = 10 ** lm   # the paper's curve uses the h^-1 bin centre as Msun
            mm = m / 1e14
            P0 = 4e3 * mm ** 0.29; al = 0.88 * mm ** -0.03; be = 3.83 * mm ** 0.04; g = -0.2
            sx = x / 0.5
            printed = P0 * sx ** g * (1 + sx ** al) ** (-(be - g) / al)
            a.plot(x, x ** 3 * printed / (200 * C.fb), color=S1, ls="--", lw=1.5,
                   label=r"B16 as printed (no $f_b$) / (200 $f_b\rho_c$)")
            a.plot(x, x ** 3 * printed * L.X_H / (200 * L.MU_E), color=S2,
                   label=r"B16 with $f_b$, $n_e=\rho/(\mu_e m_p)$ (physical)")
            a.set_ylabel(r"$x^3\, n_e/n_{200}$")
        else:
            a.plot(x, x ** 3 * L.b12_Pth_over_P200(x, 10 ** lm, 0.0), color=S1, ls="--", lw=1.5,
                   label="B12 thermal pressure $P_{th}/P_{200}$")
            a.plot(x, x ** 3 * L.b12_Pe_over_P200(x, 10 ** lm / h, 0.0), color=S2,
                   label="B12 electron pressure 0.5176 $P_{th}$ (physical)")
            a.set_ylabel(r"$x^3\, P_e/P_{200}$")
        a.plot(x, orange, color=S3, lw=1.5, label="paper: fit without c, with BPL (orange)")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$x=r/R_{200}$")
        a.set_title(ttl, color=INK, loc="left", fontsize=9.5)
        a.legend(fontsize=7.2, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_paper_reference_curves.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_temperature()
    fig_observations()
    fig_reference_curves()
    print("figures written to", OUT)
