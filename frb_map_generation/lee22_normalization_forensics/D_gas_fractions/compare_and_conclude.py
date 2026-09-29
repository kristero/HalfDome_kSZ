#!/usr/bin/env python3
"""Part (4): confront every Lee22 normalization reading with TNG's gas content.

Inputs (written by the other three scripts of this subdir):
  lee22_gas_fractions_summary.json, konietzka_implied_fe_shells.json, tng_literature_gas_fractions.json
Outputs:
  lee22_vs_tng_literature.csv      f_e(<R200c) of each Lee22 reading at the literature masses and z = 0, 0.4, 2
  correction_fits.json             K(z) = K0 (1+z)^p needed to turn the literal fit into TNG's gas content
  dm_zs_predictions.csv            cumulative mean halo DM (1e13-1e14 Msun) vs z_s: TNG vs models
  gas_fraction_forensics.png/.pdf  figure
"""
import csv
import json
from pathlib import Path

import numpy as np

import gasfrac_lib as L

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "lee22_normalization_forensics_20260928" / "D_gas_fractions"
H = L.TNG["h"]
FB = L.TNG["ob"] / L.TNG["om"]


def lee22_fe(log10m_msun, z, fit="noc", X=1.0):
    lmh = log10m_msun + np.log10(H)
    n0, xc, beta = (L.lee22_noc_params if fit == "noc" else L.lee22_c_params)(lmh, z)
    return L.F_E_PER_NBAR * 3 * float(n0) * L.shape_moment(float(xc), float(beta), X)


VARIANTS = {
    "literal": lambda z: 1.0,
    "xgpnative_0.6016": lambda z: L.XGP_NATIVE_FACTOR,
    "XH2_0.5776": lambda z: L.X_H ** 2,
    "comoving_hyp_(1+z)^3/E^2": lambda z: (1 + z) ** 3 / L.E2(L.TNG, z),
    "(1+z)^3": lambda z: (1 + z) ** 3,
    "(1+z)^3*h^2": lambda z: (1 + z) ** 3 * H ** 2,
    "(1+z)^3*h^3": lambda z: (1 + z) ** 3 * H ** 3,
    "(1+z)^3*XH^2": lambda z: (1 + z) ** 3 * L.X_H ** 2,
    "(1+z)^3*h^3*XH^2": lambda z: (1 + z) ** 3 * H ** 3 * L.X_H ** 2,
}


def tng_reference(lit):
    """TNG gas inside R200c at (log10 M200c [Msun], z): z=0 from Ayromlou Fig.10 median baryons minus Pillepich stars;
    z = 0.4 and 2 from the closure-radius inversion (Ayromlou Fig. 6 + Table 2) minus the same z=0 stellar fraction
    (Pillepich+18b: total stellar mass at fixed halo mass ~invariant z=1->0; at z=2 an assumption)."""
    import tng_literature_gas_fractions as T
    f10 = lit["ayromlou23_fig10_tng_median_z0"]
    f6 = lit["ayromlou23_fig6_inferred_baryon_fraction"]
    rows = []
    for lm in (13.0, 13.25, 13.5, 13.75, 14.0):
        fs = float(T.pillepich_fstar_r200c(lm))
        fb0 = float(np.interp(lm, f10["log10_m200c_msun"], f10["fb_over_fbcos"]))
        for z, fb in ((0.0, fb0),
                      (0.4, float(np.interp(lm, f6["0.4"]["log10_m200c_msun"], f6["0.4"]["fb_over_fbcos_inferred"]))),
                      (2.0, float(np.interp(lm, f6["2.0"]["log10_m200c_msun"], f6["2.0"]["fb_over_fbcos_inferred"])))):
            if not np.isfinite(fb):
                continue
            rows.append(dict(log10M200c_msun=lm, z=z, tng_fb=fb, tng_fstar=fs, tng_fgas=fb - fs,
                             source="Ayromlou+23 Fig.10 - Pillepich+18b" if z == 0 else
                             "Ayromlou+23 Fig.6 + Table 2 inversion - Pillepich+18b (z=0 stars)"))
    return rows


def main():
    lit = json.loads((OUT / "tng_literature_gas_fractions.json").read_text())
    shells = json.loads((OUT / "konietzka_implied_fe_shells.json").read_text())["shells"]

    # ---- 1. Lee22 readings at the literature (mass, z) points
    ref = tng_reference(lit)
    out_rows = []
    for r in ref:
        for fit in ("noc", "c"):
            fe_lit = lee22_fe(r["log10M200c_msun"], r["z"], fit)
            for v, K in VARIANTS.items():
                fe = K(r["z"]) * fe_lit
                out_rows.append(dict(r, fit=fit, variant=v, lee22_fe=fe, ratio_to_tng_gas=fe / r["tng_fgas"]))
    with open(OUT / "lee22_vs_tng_literature.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print("ratio Lee22(no-c)/TNG gas (literature) at log M200c = 13.0 .. 14.0 [Msun]:")
    for v in VARIANTS:
        for z in (0.0, 0.4, 2.0):
            rr = [x["ratio_to_tng_gas"] for x in out_rows if x["variant"] == v and x["fit"] == "noc" and x["z"] == z]
            print("  {:28s} z={:3.1f}: ".format(v, z) + " ".join("{:6.2f}".format(q) for q in rr))

    # ---- 2. correction K(z) needed, fitted as K0 (1+z)^p
    fits = {}
    zc = np.array([0.5 * (s["z_lo"] + s["z_hi"]) for s in shells])
    sel = zc < 2.0
    need = np.array([s["fe_tng_implied"] / s["lee22_noc_literal_fe"] for s in shells])
    err = np.array([s["fe_tng_implied_err"] / s["lee22_noc_literal_fe"] for s in shells])
    x = np.log(1 + zc[sel]); y = np.log(need[sel]); wts = (need[sel] / err[sel]) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * wts[:, None]))
    coef = cov @ (A.T @ (wts * y))
    resid = y - A @ coef
    chi2 = float(np.sum(wts * resid ** 2)); dof = int(sel.sum() - 2)
    scale = max(1.0, chi2 / dof)
    fits["konietzka_dm_shells_z<2"] = dict(K0=float(np.exp(coef[0])), p=float(coef[1]),
                                           p_err=float(np.sqrt(cov[1, 1] * scale)),
                                           K0_err_frac=float(np.sqrt(cov[0, 0] * scale)), chi2=chi2, dof=dof,
                                           note="K(z) = f_e(TNG DM catalogue, HMF-weighted 1e13-1e14 Msun) / "
                                                "f_e(Lee22 no-c literal, same weights)")
    # literature points (no-c)
    lx, ly = [], []
    for r in ref:
        fe = lee22_fe(r["log10M200c_msun"], r["z"], "noc")
        lx.append(np.log(1 + r["z"])); ly.append(np.log(r["tng_fgas"] / fe))
    lx, ly = np.array(lx), np.array(ly)
    c2 = np.polyfit(lx, ly, 1)
    fits["tng_literature_points"] = dict(K0=float(np.exp(c2[1])), p=float(c2[0]),
                                         rms_log_resid=float(np.std(ly - np.polyval(c2, lx))),
                                         n=len(lx), note="unweighted fit over log M200c 13-14 Msun, z = 0, 0.4, 2")
    # pure (1+z)^3 constants
    c0_dm = need[sel] / (1 + zc[sel]) ** 3
    fits["C0_for_pure_(1+z)^3"] = {
        "dm_shells_z<0.2": float(np.mean(c0_dm[zc[sel] < 0.2])),
        "dm_shells_0.4<z<0.6": float(np.mean(c0_dm[(zc[sel] > 0.4) & (zc[sel] < 0.6)])),
        "dm_shells_0.9<z<1.1": float(np.mean(c0_dm[(zc[sel] > 0.9) & (zc[sel] < 1.1)])),
        "dm_shells_1.5<z<2": float(np.mean(c0_dm[(zc[sel] > 1.5)])),
        "literature_z0_mean": float(np.mean([np.exp(v) for v, r in zip(ly, ref) if r["z"] == 0])),
        "literature_z0.4_mean": float(np.mean([np.exp(v) / 1.4 ** 3 for v, r in zip(ly, ref) if r["z"] == 0.4])),
        "literature_z2_mean": float(np.mean([np.exp(v) / 27 for v, r in zip(ly, ref) if r["z"] == 2.0])),
        "candidates": {"h^2": H ** 2, "h^3": H ** 3, "X_H^2": L.X_H ** 2, "h^3 X_H^2": H ** 3 * L.X_H ** 2,
                       "h^2 X_H^2": H ** 2 * L.X_H ** 2, "Omega_b/Omega_m": FB, "X_H^2 Omega_b/Omega_m": L.X_H ** 2 * FB},
    }
    # effective (1+z) power of candidate forms over z = 0..1
    fits["effective_power_0_to_1"] = {
        "(1+z)^3": 3.0,
        "(1+z)^3/E^2": float(np.log(8 / L.E2(L.TNG, 1.0)) / np.log(2)),
        "(1+z)^3/E": float(np.log(8 / np.sqrt(L.E2(L.TNG, 1.0))) / np.log(2)),
    }
    (OUT / "correction_fits.json").write_text(json.dumps(fits, indent=1))
    print(json.dumps(fits, indent=1))

    # ---- 3. cumulative DM predictions vs z_s
    rows = []
    for s in shells:
        row = dict(z_s=s["z_hi"], tng=s["tng_mean_cum"], b16=s["b16_cum"])
        for v in ("literal", "xgpnative", "XH2", "comoving_hyp", "(1+z)^3", "(1+z)^3*h^3*XH^2", "(1+z)^3*XH^2*fb"):
            row["lee22_noc_" + v] = s[f"lee22_noc_{v}_cum"]
        rows.append(row)
    with open(OUT / "dm_zs_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        if round(r["z_s"], 3) in (0.1, 0.3, 0.5, 1.0, 1.5, 2.0):
            print("z_s={:.1f} TNG {:6.2f} | ratios: B16 {:.3f} lit {:.3f} xgp {:.3f} XH2 {:.3f} (1+z)^3h^3XH^2 {:.3f}".format(
                r["z_s"], r["tng"], r["b16"] / r["tng"], r["lee22_noc_literal"] / r["tng"],
                r["lee22_noc_xgpnative"] / r["tng"], r["lee22_noc_XH2"] / r["tng"],
                r["lee22_noc_(1+z)^3*h^3*XH^2"] / r["tng"]))

    make_figure(ref, shells, rows, fits)


def make_figure(ref, shells, rows, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
             green="#008300", violet="#4a3aa7", red="#e34948", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
                         "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True,
                         "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    zz = np.linspace(0, 2, 81)
    styles = [("literal", C["red"], "-", "Lee22 literal (eq. 9 as printed)"),
              ("xgpnative_0.6016", C["orange"], "--", "x 0.602 (XGPaint-native, production)"),
              ("comoving_hyp_(1+z)^3/E^2", C["violet"], "-.", "x (1+z)^3/E^2 (repo comoving hyp.)"),
              ("(1+z)^3*h^3*XH^2", C["blue"], "-", "x (1+z)^3 h^3 X_H^2 (0.18)")]
    # panel A: two masses
    ax = axs[0]
    for lm, mk in ((13.25, "o"), (13.75, "s")):
        for v, col, ls, lab in styles:
            fe = np.array([lee22_fe(lm, z) * VARIANTS[v](z) for z in zz])
            ax.plot(zz, fe, color=col, ls=ls, lw=2 if lm == 13.25 else 1.2,
                    label=lab if lm == 13.25 else None)
        pts = [r for r in ref if r["log10M200c_msun"] == lm]
        ax.plot([r["z"] for r in pts], [r["tng_fgas"] for r in pts], mk, ms=8, mfc="white", mec=C["ink"], mew=1.5,
                ls="none", label=f"TNG literature, log M200c={lm}" + (" (thick)" if lm == 13.25 else " (thin)"))
    ax.axhline(1, color=C["ink2"], lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_ylim(0.05, 8); ax.set_xlim(-0.06, 2.06)
    ax.set_xlabel("z"); ax.set_ylabel(r"$f_e(<R_{200c}) = M_{\rm gas,ion}/(f_b M_{200c})$")
    ax.set_title("A. Implied gas inside R200c vs TNG (literature)", loc="left", fontsize=9.5)
    ax.text(1.98, 1.07, "cosmic baryon share", ha="right", va="bottom", fontsize=7.5, color=C["ink2"])
    ax.legend(fontsize=6.8, loc="lower left")
    # panel B: HMF-weighted shells vs TNG DM catalogue
    ax = axs[1]
    zc = np.array([0.5 * (s["z_lo"] + s["z_hi"]) for s in shells])
    m = zc < 2.0
    ax.errorbar(zc[m], [s["fe_tng_implied"] for s, k in zip(shells, m) if k],
                yerr=[s["fe_tng_implied_err"] for s, k in zip(shells, m) if k], fmt="o", ms=5, color=C["ink"],
                mfc="white", capsize=0, lw=1, label="TNG within-R200 DM (Konietzka), implied")
    for key, col, ls, lab in (("lee22_noc_literal_fe", C["red"], "-", "Lee22 literal"),
                              ("lee22_noc_xgpnative_fe", C["orange"], "--", "x 0.602 (production)"),
                              ("lee22_noc_comoving_hyp_fe", C["violet"], "-.", "x (1+z)^3/E^2"),
                              ("lee22_noc_(1+z)^3*h^3*XH^2_fe", C["blue"], "-", "x (1+z)^3 h^3 X_H^2"),
                              ("fe_b16", C["aqua"], ":", "Battaglia16 (XGPaint, incl. 0.9)")):
        ax.plot(zc[m], [s[key] for s, k in zip(shells, m) if k], color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(1, color=C["ink2"], lw=0.8, ls=":")
    ax.axhspan(0.45, 0.55, color=C["grid"], alpha=0.9, lw=0)
    ax.text(1.98, 0.43, "TNG literature, z=0, 1e13-1e14 (gas = baryons - stars)", fontsize=6.5, color=C["ink2"], va="top", ha="right")
    ax.set_yscale("log"); ax.set_ylim(0.05, 8); ax.set_xlim(0, 2)
    ax.set_xlabel("z of the shell"); ax.set_ylabel(r"HMF-weighted $f_e(<R_{200c})$, $10^{13}$-$10^{14}\,M_\odot$")
    ax.set_title("B. Per-shell electron content (same halo weights)", loc="left", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="lower left")
    # panel C: cumulative DM ratio
    ax = axs[2]
    zs = np.array([r["z_s"] for r in rows]); msk = zs <= 2.0
    for key, col, ls, lab in (("lee22_noc_literal", C["red"], "-", "Lee22 literal"),
                              ("lee22_noc_xgpnative", C["orange"], "--", "x 0.602 (production)"),
                              ("lee22_noc_(1+z)^3*h^3*XH^2", C["blue"], "-", "x (1+z)^3 h^3 X_H^2"),
                              ("b16", C["aqua"], ":", "Battaglia16")):
        ax.plot(zs[msk], [r[key] / r["tng"] for r, k in zip(rows, msk) if k], color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(1, color=C["ink2"], lw=0.8)
    ax.axvline(1, color=C["ink2"], lw=0.8, ls=":")
    ax.text(1.02, 2.6, "published PDF test\n(z_s = 1)", fontsize=7, color=C["ink2"], va="top")
    ax.set_xlim(0, 2); ax.set_ylim(0, 3.2)
    ax.set_xlabel(r"source redshift $z_s$"); ax.set_ylabel("mean halo DM over all rays, model / TNG")
    ax.set_title("C. Why z_s=1 matches: errors cancel", loc="left", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"gas_fraction_forensics.{ext}", dpi=160)
    print("figure written")


if __name__ == "__main__":
    main()
