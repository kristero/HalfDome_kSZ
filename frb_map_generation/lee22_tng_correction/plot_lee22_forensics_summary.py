#!/usr/bin/env python3
"""Concise figures for the Lee22 normalization forensics (2026-09-29), built only from the saved outputs of the
2026-09-28 investigation (frb_map_generation/outputs/lee22_normalization_forensics_20260928/).

  fig1  what is wrong: gas inside R200c implied by the printed fit; TNG/Lee22 per halo-redshift slice with the
        Battaglia16 control and the constant 0.6016 of our production variant; Lee22's own T_e/T200; the anchors of
        the unexplained constant F_n (open question)
  fig2  FRB halo DM against TNG versus source redshift, and where the z_s = 1 halo DM comes from
  fig3  kSZ and tSZ consequences (painted HalfDome maps)
Run from the repository root."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

F = Path("frb_map_generation/outputs/lee22_normalization_forensics_20260928")
OUT = Path("frb_map_generation/outputs/lee22_tngcal_20260929/report")
C0, K = 0.2841790493880425, 2.483190184608004
XGP = 0.6016316602183194
COL = dict(tng="#1a1a1a", b16="#0072B2", xgp="#E69F00", cal="#009E73", lit="#D55E00", hyb="#CC79A7")
RC = {"font.family": "DejaVu Sans", "font.size": 12.5, "axes.labelsize": 13.5, "axes.titlesize": 13.5,
      "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11, "axes.linewidth": 1.0,
      "xtick.direction": "in", "ytick.direction": "in", "savefig.facecolor": "white", "pdf.fonttype": 42}


def c_tng(z):
    return C0 * (1 + np.minimum(z, 2.0)) ** K


def rows(path, **match):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if all(r[k] == v for k, v in match.items())]


def style(ax):
    ax.grid(color=".9", lw=.6)
    ax.tick_params(which="both", top=True, right=True)


def panel_label(ax, text):
    ax.text(.02, .97, text, transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")


def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    # (a) implied gas fraction inside R200c
    ax = axes[0, 0]
    grid = rows(F / "D_gas_fractions/lee22_gas_fractions_grid.csv", fit="noc_table8", variant="literal")
    for lm, mk in ((13.0, "o"), (13.8, "s"), (14.6, "^")):
        g = sorted((float(r["z"]), float(r["fe_R200c"])) for r in grid if float(r["log10M_h"]) == lm)
        z, fe = map(np.array, zip(*g))
        ax.plot(z, fe, color=COL["lit"], marker=mk, ms=7, lw=2)
        ax.plot(z, fe * c_tng(z), color=COL["cal"], marker=mk, ms=7, lw=2, ls="--", mfc="white")
    ax.axhspan(1, 20, color=COL["lit"], alpha=.06, lw=0)
    ax.axhline(1, color=".35", lw=1.2)
    ax.text(1.95, 7.2, "impossible", ha="right", fontsize=11, color=COL["lit"])
    ax.set(yscale="log", ylim=(.06, 10), xlim=(-.08, 2.08), xlabel="halo redshift $z$",
           ylabel=r"gas inside $R_{200c}$  /  $f_b M_{200c}$")
    leg = ax.legend(handles=[Line2D([], [], color=COL["lit"], lw=2, label="Lee22 as printed (Table 8, eq. 9)"),
                             Line2D([], [], color=COL["cal"], lw=2, ls="--", label="× TNG correction $C(z)$"),
                             Line2D([], [], color=".35", lw=1.2, label="all of the halo's baryons ($f_b M_{200c}$)")],
                    loc="lower left", frameon=False)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], color=".35", marker=m, ls="none", ms=7, label=r"$10^{%.1f}$" % v)
                       for v, m in ((13.0, "o"), (13.8, "s"), (14.6, "^"))], title=r"$M_{200c}$ [$h^{-1}M_\odot$]",
              loc="upper right", bbox_to_anchor=(1, .9), frameon=False, fontsize=10.5, title_fontsize=10.5, handletextpad=.2)
    ax.set_title("1. The printed density fit holds 2–5× all baryons at $z=0$", loc="left")
    panel_label(ax, "a")
    # (b) TNG / Lee22 per halo-redshift slice
    ax = axes[0, 1]
    sl = [r for r in rows(F / "A_tng_redshift_test/tng_vs_model_slices.csv", window="mass13to14", mode="continuous")
          if float(r["z_hi"]) <= 2.0 + 1e-9]
    zm = np.array([.5 * (float(r["z_lo"]) + float(r["z_hi"])) for r in sl])
    for key, col, mk, lab in (("lee22_literal", COL["lit"], "o", "TNG / Lee22 as printed"),
                              ("b16", COL["b16"], "s", "TNG / Battaglia16 (control)")):
        c = np.array([float(r["C_" + key]) for r in sl]); e = np.array([float(r["C_err_" + key]) for r in sl])
        ax.errorbar(zm, c, yerr=e, color=col, marker=mk, ms=6, lw=0, elinewidth=1.2, capsize=0, label=lab)
    zz = np.linspace(0, 2, 200)
    ax.plot(zz, c_tng(zz), color=COL["cal"], lw=2.2, label=r"fit $0.284\,(1+z)^{2.48}$  (= $C(z)$)")
    zc = (XGP / C0) ** (1 / K) - 1
    ax.axhline(XGP, color=COL["xgp"], lw=2.2, ls=(0, (6, 3)), label="production 0.6016 (right only at $z\\approx%.2f$ ○)" % zc)
    ax.plot([zc], [XGP], marker="o", ms=11, mfc="none", mec=".2", mew=1.5)
    ax.set(yscale="log", ylim=(.18, 40), xlim=(-.05, 2.05), xlabel="halo redshift $z$ (slices of 0.1)",
           ylabel="TNG halo electrons / model  (10$^{13}$–10$^{14}\\,M_\\odot$)")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(.03, 1))
    ax.set_title("2. Against TNG itself the error grows as $(1+z)^{2.5}$; no constant fits", loc="left")
    panel_label(ax, "b")
    # (c) Lee22's own temperature
    ax = axes[1, 0]
    tg = [r for r in csv.DictReader(open(F / "C_pressure_consistency/temperature_grid.csv"))
          if r[list(r.keys())[0]] == "noc"]
    for lm, mk in ((13.0, "o"), (13.6, "s"), (14.4, "^")):
        g = sorted((float(r["z"]), float(r["Tnw_over_T200_within_R200"])) for r in tg if abs(float(r["log10_M200c_hinv"]) - lm) < 1e-6)
        z, t = map(np.array, zip(*g))
        ax.plot(z, t, color=COL["lit"], marker=mk, ms=7, lw=2)
        ax.plot(z, t / (1 + z), color=COL["lit"], marker=mk, ms=6, lw=1.4, ls=":", mfc="white")
    ax.axhspan(.68, 1.45, color=COL["b16"], alpha=.10, lw=0)
    ax.text(1.98, 1.25, "physical: B12/B16 0.68–0.87,\nTNG300 X-ray 0.95–1.45", ha="right", va="top", fontsize=10.5, color=COL["b16"])
    ax.text(1.25, .36, "Lee22 pressure fit / density fit", fontsize=10.5, color=COL["lit"], ha="center")
    ax.text(.05, .093, "same, $a^3$ / $a^2$ bookkeeping removed", fontsize=10.5, color=COL["lit"])
    ax.legend(handles=[Line2D([], [], color=".35", marker=m, ls="none", ms=7, label=r"$10^{%.1f}$" % v)
                       for v, m in ((13.0, "o"), (13.6, "s"), (14.4, "^"))], title=r"$M_{200c}$ [$h^{-1}M_\odot$]",
              loc="center left", bbox_to_anchor=(0, .55), frameon=False, fontsize=10.5, title_fontsize=10.5, handletextpad=.2)
    ax.set(yscale="log", ylim=(.06, 2), xlim=(-.05, 2.05), xlabel="halo redshift $z$",
           ylabel=r"implied $T_e / T_{200}$ inside $R_{200c}$")
    ax.set_title("3. Lee22's own fits imply gas 5–7× too cold: density too high", loc="left")
    panel_label(ax, "c")
    # (d) open question: the constant F_n
    ax = axes[1, 1]
    anchors = [("Konietzka rays, $z\\approx0.05$", 3.12, 3.12), ("$C(z)$ extrapolated to $z=0$", 3.52, 3.52),
               ("TNG gas-fraction literature", 5.4, 5.4), ("TNG300 hot gas $f_{gas,500}$", 5.0, 5.5),
               ("TNG300 X-ray temperatures", 5.3, 7.5), ("hydrostatic, 15% non-thermal", 4.7, 6.8),
               ("hydrostatic, thermal only", 5.5, 8.0)]
    for i, (lab, lo, hi) in enumerate(anchors):
        ax.plot([lo, hi], [i, i], color=".25", lw=5 if hi > lo else 0, solid_capstyle="round")
        ax.plot([.5 * (lo + hi)], [i], marker="o", ms=8, color=".25")
    for (val, lab), dy in zip(((1 / .6774 ** 3, "$1/h^3$"), (1 / (.6774 ** 3 * .76 ** 2), "$1/(h^3X_H^2)$"), (.3089 / .0486, "$\\Omega_m/\\Omega_b$")), (0, 0, .45)):
        ax.axvline(val, color=COL["hyb"], lw=1.6, ls="--")
        ax.text(val + (-.06 if "X_H" in lab else .06), len(anchors) - .45 + dy, lab, color=COL["hyb"],
                ha="right" if "X_H" in lab else "left", fontsize=10.5)
    ax.set(xlim=(2.5, 9), ylim=(-.7, len(anchors) + .4), xlabel="extra z-independent density factor $F_n$")
    ax.set_yticks(range(len(anchors)))
    ax.set_yticklabels([a[0] for a in anchors], fontsize=10.5)
    ax.set_title("4. Open: the size and origin of the constant $F_n$", loc="left")
    panel_label(ax, "d")
    for a in axes.flat:
        style(a)
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    save(fig, "lee22_fig1_what_is_wrong")


def fig2():
    cum = rows(F / "A_tng_redshift_test/tng_vs_model_cumulative.csv", window="mass13to14", mode="continuous")
    sl = rows(F / "A_tng_redshift_test/tng_vs_model_slices.csv", window="mass13to14", mode="continuous")
    zs = np.array([float(r["z_s"]) for r in cum])
    keep = zs <= 2.0 + 1e-9
    hit = np.array([float(r["model_hit_unclustered"]) for r in cum])
    lit_slice = np.array([float(r["model_dDM_lee22_literal"]) for r in sl])
    zeff = np.array([float(r["zmean_dm_weighted_lee22_literal"]) for r in sl])
    cal_all = np.cumsum(lit_slice * c_tng(zeff))
    np.testing.assert_allclose(np.cumsum(lit_slice), [float(r["model_mean_all_lee22_literal"]) for r in cum], rtol=1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    ax = axes[0]
    tng = np.array([float(r["tng_mean_per_hit"]) for r in cum])
    tng_err = np.array([float(r["tng_mean_all_err"]) / float(r["tng_hit"]) for r in cum])
    ax.errorbar(zs[keep], tng[keep], yerr=tng_err[keep], color=COL["tng"], marker="o", ms=6, lw=0, elinewidth=1.2, zorder=5)
    series = (("lee22_literal", COL["lit"], "-", "Lee22 as printed"), ("lee22_xgpnative", COL["xgp"], (0, (6, 3)), "Lee22 × 0.6016 (production)"),
              ("b16", COL["b16"], "-.", "Battaglia16"))
    for key, col, ls, lab in series:
        ax.plot(zs[keep], np.array([float(r["model_mean_per_hit_" + key]) for r in cum])[keep], color=col, lw=2.2, ls=ls)
    ax.plot(zs[keep], (cal_all / hit)[keep], color=COL["cal"], lw=2.6)
    lab_pos = {"lee22_literal": 1.35, "lee22_xgpnative": 1.55, "b16": 1.6}
    for key, col, ls, lab in series:
        y = np.interp(lab_pos[key], zs, [float(r["model_mean_per_hit_" + key]) for r in cum])
        ax.text(lab_pos[key], y * (1.07 if key != "lee22_xgpnative" else 0.9), lab, color=col, fontsize=10.5, ha="center")
    ax.text(1.62, np.interp(1.62, zs, cal_all / hit) * 1.1, "Lee22 × $C(z)$ (TNG-calibrated)", color=COL["cal"], fontsize=10.5, ha="center")
    ax.text(.12, 100, "IllustrisTNG (Konietzka)", color=COL["tng"], fontsize=10.5)
    ax.set(yscale="log", ylim=(55, 700), xlim=(0, 2.05), xlabel="FRB source redshift $z_s$",
           ylabel="mean halo DM per hit ray  [pc cm$^{-3}$]")
    ax.set_title("Halo DM ($10^{13}$–$10^{14}\\,M_\\odot$, inside $R_{200c}$) vs source redshift", loc="left")
    ax.axvline(1.0, color=".6", lw=1, ls=":")
    ax.text(1.02, 600, "$z_s=1$: the only place\nthe 0.6016 constant is right", fontsize=10, color=".35", va="top")
    panel_label(ax, "a")
    ax = axes[1]
    s1 = [i for i, r in enumerate(sl) if float(r["z_hi"]) <= 1.0 + 1e-9]
    edges = np.array([float(sl[i]["z_lo"]) for i in s1] + [1.0])
    for vals, col, ls, lw, lab in ((np.array([float(sl[i]["tng_dDM"]) for i in s1]), COL["tng"], "-", 2.6, "IllustrisTNG"),
                                   (np.array([float(sl[i]["model_dDM_lee22_xgpnative"]) for i in s1]), COL["xgp"], (0, (6, 3)), 2.2, "Lee22 × 0.6016 (production)"),
                                   ((lit_slice * c_tng(zeff))[s1], COL["cal"], "-", 2.2, "Lee22 × $C(z)$")):
        frac = vals / vals.sum()
        ax.stairs(frac, edges, color=col, ls=ls, lw=lw, label=lab + "  (median $z$ = %.2f)" % np.interp(.5, np.cumsum(frac), edges[1:]))
    ax.set(xlim=(0, 1), ylim=(0, .26), xlabel="halo redshift $z$", ylabel="share of the halo DM at $z_s=1$ per $\\Delta z=0.1$")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Same total at $z_s=1$ from the wrong redshifts", loc="left")
    panel_label(ax, "b")
    for a in axes:
        style(a)
    fig.tight_layout(w_pad=2.2)
    save(fig, "lee22_fig2_frb_dm_redshift_test")


def fig3():
    ks = json.load(open(F / "verify_5/ksz_split_results.json"))["bands"]
    ts = json.load(open(F / "verify_5/tsz_split_results.json"))["bands"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    ax = axes[0]
    ell = np.array([.5 * (b["band"][0] + b["band"][1]) for b in ks])
    for key, col, ls, mk, lab in (("lit_over_b16", COL["lit"], "-", "o", "Lee22 as printed"),
                                  ("xgp_over_b16", COL["xgp"], (0, (6, 3)), "s", "Lee22 × 0.6016 (production)"),
                                  ("litC_frozen_over_b16", COL["cal"], "-", "D", "Lee22 × $C(z)$ (TNG-calibrated)")):
        v = np.array([b[key] for b in ks])
        ax.plot(ell, v, color=col, ls=ls, marker=mk, ms=7, lw=2.2)
        ax.text(ell[0] * .93, v[0], lab, color=col, fontsize=10.5, ha="right", va="center")
    ax.axhline(1, color=COL["b16"], lw=1.4)
    ax.text(5200, 1.08, "Battaglia16", color=COL["b16"], fontsize=10.5, ha="right")
    ax.set(xscale="log", yscale="log", xlim=(60, 6000), ylim=(.6, 30), xlabel="multipole $\\ell$", ylabel="kSZ $D_\\ell$ / Battaglia16")
    ax.set_title("kSZ: the printed density's excess is a low-$z$ effect", loc="left")
    panel_label(ax, "a")
    ax = axes[1]
    ells = np.array(sorted(int(k) for k in ts))
    for key, col, ls, mk, lab in (("ratio_L_over_B", COL["lit"], "-", "o", "Lee22 pressure, all masses"),
                                  ("ratio_hybrid_over_B", COL["hyb"], "--", "s", "Lee22 pressure, Battaglia12 above $10^{14.8}h^{-1}M_\\odot$"),
                                  ("ratio_Lhigh_over_Bhigh", ".45", ":", "^", "halos above $10^{14.8}h^{-1}M_\\odot$ only")):
        v = np.array([ts[str(l)][key] for l in ells])
        ax.plot(ells, v, color=col, ls=ls, marker=mk, ms=7, lw=2.2, label=lab)
    ax.axhline(1, color=COL["b16"], lw=1.4)
    ax.text(28, 1.07, "Battaglia12", color=COL["b16"], fontsize=10.5)
    ax.set(xscale="log", yscale="log", xlim=(25, 10000), ylim=(.2, 6), xlabel="multipole $\\ell$", ylabel="tSZ $C_\\ell^{yy}$ / Battaglia12")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("tSZ: the low-$\\ell$ excess is the high-mass extrapolation", loc="left")
    panel_label(ax, "b")
    for a in axes:
        style(a)
    fig.tight_layout(w_pad=2.2)
    save(fig, "lee22_fig3_sz_consequences")


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / (stem + "." + ext), dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / (stem + ".png"))


if __name__ == "__main__":
    with plt.rc_context(RC):
        fig1(); fig2(); fig3()
