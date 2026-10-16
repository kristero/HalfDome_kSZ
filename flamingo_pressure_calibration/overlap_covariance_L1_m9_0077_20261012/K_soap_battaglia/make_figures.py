"""Test K figures (local, from the fetched results).

  python -B make_figures.py [--results DIR] [--out DIR]
Figures (out dir, default figures/):
  fig_K_params_vs_mass_{P,G}.png   per-cell free fits (K2) vs mass at z ~ 0, 0.5, 1, 2, 3 with the refitted 'orig' laws
                                   (each fit range) and the published law (grey)                                  [main]
  fig_K_params_vs_z_{P,G}.png      the same vs z for four mass bins
  fig_K_profiles_{P,G}.png         x^3 P/P200 and x^3 rho/(f_b rho_c) at M200c = 10^12.8-10^15 and z = 0-3: published
                                   vs refitted vs SOAP per-halo-shell points (z = 0: particle stacks, Test I fits) [main]
  fig_K_ratio_to_published.png     refitted / published profile vs x
  fig_K_residuals_{P,G}.png        ln(SOAP / law) per aperture vs mass and z (published, orig R1-R3)          [main]
  fig_K_aperture_integrals.png     law / SOAP for Y(<R500c), Y(<R200c), M_gas(<R200c) vs mass at 5 redshifts   [main]
  fig_K_sigma_d.png                model-discrepancy floor per field x range x parametrisation (+ published; hatched =
                                   not converged, results/k7_fit_audit.json)
  fig_K_corr.png                   parameter correlation matrices of the R2 orig and free fits
  fig_K_law_params.png             all five profile parameters of the published, orig R3, recommended and alternative
                                   laws vs mass at z = 0, 1, 2 (thick inside the SOAP data, thin = extrapolation)  [main]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_common as KC  # noqa: E402
import k_fit_core as K  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RANGE_COL = {"R1": CAT[0], "R2": CAT[1], "R3": CAT[2]}
SEQ = ["#9ec5f0", "#5f9fe6", "#2a78d6", "#1b56a3", "#0e3263"]       # sequential blue, light -> dark (z)
GREY = "#8a8985"
INK = "#2b2b2a"
DIV = LinearSegmentedColormap.from_list("kdiv", ["#2a78d6", "#f0efec", "#e34948"])
ZS = (0.0, 0.5, 1.0, 2.0, 3.0)
PNAMES = {"B12": ("amp", "xc", "beta"), "B16": ("amp", "alpha", "beta")}
PLABEL = {"B12": {"amp": "$P_0$", "xc": "$x_c$", "beta": r"$\beta$"}, "B16": {"amp": r"$\rho_0$", "alpha": r"$\alpha$", "beta": r"$\beta$"}}
FIELD_FORM = {"P": "B12", "G": "B16"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#6b6a66", "axes.labelcolor": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight"})


def rec_keys():
    """Recommended fits (report/recommendation.json, written after reading the results)."""
    p = HERE / "report" / "recommendation.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    return {"P": d.get("tSZ", {}).get("key", "P|R3|brk_amp"), "G": d.get("kSZ_DM", {}).get("key", "G|R3|free"),
            "P_alt": d.get("tSZ_alt", {}).get("key", "P|R3|free"), "G_alt": d.get("kSZ_DM_alt", {}).get("key", "G|R3|brk_all")}


def short(key):
    f, r, p = key.split("|")
    return f"{p} {r}"


def law_phi(form, kind, theta, M, z):
    law = K.Law(form, kind)
    p = law.phi(np.asarray(theta, float), np.log(np.atleast_1d(M).astype(float)), np.log1p(np.atleast_1d(z).astype(float)))
    return p


def nearest_snaps(cells_z, zs):
    """Output redshift nearest to each requested z."""
    uz = np.unique(cells_z)
    return [float(uz[np.argmin(np.abs(uz - z))]) for z in zs]


def load(R):
    fits = json.loads((R / "k3_fits.json").read_text())
    k2 = json.loads((R / "k2_cell_fits.json").read_text())["fits"] if (R / "k2_cell_fits.json").exists() else []
    pred = np.load(R / "k3_predictions.npz") if (R / "k3_predictions.npz").exists() else None
    cells = []
    for f in sorted((R / "cells").glob("cells_*.npz")):
        cells.append(dict(np.load(f)))
    C = {k: np.concatenate([c[k] for c in cells]) for k in ("z", "snap", "cell_bin", "cell_N", "cell_mean_lnM", "shell_P", "shell_G", "xbar")} if cells else None
    prof = np.load(R / "k4_profiles.npz") if (R / "k4_profiles.npz").exists() else None
    part = np.load(R / "k4_particle_z0.npz") if (R / "k4_particle_z0.npz").exists() else None
    return fits, k2, pred, C, prof, part


def fig_params_vs_mass(field, fits, k2, out):
    form = FIELD_FORM[field]
    qs = PNAMES[form]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True)
    zs_all = np.array(sorted({d["z"] for d in k2 if d["field"] == field}))
    if not len(zs_all):
        plt.close(fig); return
    znear = [float(zs_all[np.argmin(np.abs(zs_all - z))]) for z in ZS]
    mgrid = np.linspace(12.7, 15.4, 60)
    for c, r in enumerate(KC.RANGES):
        it = fits["configs"].get(f"{field}|{r}|orig")
        for row, q in enumerate(qs):
            ax = axes[row, c]
            for i, (z, zn) in enumerate(zip(ZS, znear)):
                pts = [d for d in k2 if d["field"] == field and d["range"] == r and abs(d["z"] - zn) < 1e-6 and d.get("ok") and q in d]
                if pts:
                    lm = np.array([d["lnM_mean_nodes"] for d in pts]) / math.log(10)
                    v = np.array([abs(d[q]) for d in pts]); e = np.array([d[q + "_err_ln"] for d in pts]) * v
                    bnd = np.array([bool(d.get("at_bound")) for d in pts])
                    ok = (e < 3 * v) & ~bnd
                    ax.errorbar(lm[ok], v[ok], yerr=e[ok], fmt="o", ms=3.5, color=SEQ[i], mec="white", mew=0.5, elinewidth=0.8, capsize=0,
                                label=f"z = {zn:.2f} cells" if (row == 0 and c == 0) else None, zorder=3)
                    ax.plot(lm[bnd], v[bnd], "o", ms=3.5, mfc="none", mec=SEQ[i], mew=0.8, zorder=3,
                            label="cell fit at a bound (open)" if (row == 0 and c == 0 and i == len(ZS) - 1) else None)
                if it and "theta" in it:
                    p = law_phi(form, "orig", it["theta"], 10 ** mgrid, np.full_like(mgrid, zn))
                    ax.plot(mgrid, np.abs(p[q]), color=SEQ[i], lw=1.4, zorder=2)
                pp = law_phi(form, "orig", K.Law(form, "orig").start(), 10 ** mgrid, np.full_like(mgrid, zn))
                ax.plot(mgrid, np.abs(pp[q]), color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=1,
                        label="published (Battaglia)" if (row == 0 and c == 0 and i == 0) else None)
            if q == "amp":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{r}: {KC.RANGES[r]['label']}", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(PLABEL[form][q])
            if row == 2:
                ax.set_xlabel(r"$\log_{10} M_{200c}\ [M_\odot]$")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{'B12 pressure' if field == 'P' else 'B16 gas density'}: per-cell free fits (points, Hessian errors) and the "
                 f"refitted 'orig' law (lines) at the same z; published law dashed grey", fontsize=9.5)
    fig.savefig(out / f"fig_K_params_vs_mass_{field}.png")
    plt.close(fig)


def fig_params_vs_z(field, fits, k2, out):
    form = FIELD_FORM[field]
    qs = PNAMES[form]
    bins = ((12.9, 13.1), (13.5, 13.7), (13.9, 14.1), (14.5, 14.7))
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True)
    zgrid = np.linspace(0, 4, 60)
    for c, r in enumerate(KC.RANGES):
        it = fits["configs"].get(f"{field}|{r}|orig")
        for row, q in enumerate(qs):
            ax = axes[row, c]
            for i, (lo, hi) in enumerate(bins):
                lab = f"{lo:.1f}-{hi:.1f}"
                pts = [d for d in k2 if d["field"] == field and d["range"] == r and d["cell"].endswith(lab) and d.get("ok") and q in d]
                col = CAT[i]
                if pts:
                    zz = np.array([d["z"] for d in pts]); v = np.array([abs(d[q]) for d in pts]); e = np.array([d[q + "_err_ln"] for d in pts]) * v
                    ok = e < 3 * v
                    ax.errorbar(zz[ok], v[ok], yerr=e[ok], fmt="o", ms=3, color=col, mec="white", mew=0.4, elinewidth=0.7,
                                label=f"cells {lab}" if (row == 0 and c == 0) else None)
                    mm = 10 ** np.mean([d["lnM_mean_nodes"] for d in pts] if pts else [0]) if False else math.exp(np.mean([d["lnM_mean_nodes"] for d in pts]))
                else:
                    mm = 10 ** (0.5 * (lo + hi))
                if it and "theta" in it:
                    p = law_phi(form, "orig", it["theta"], np.full_like(zgrid, mm), zgrid)
                    ax.plot(zgrid, np.abs(p[q]), color=col, lw=1.3)
                pp = law_phi(form, "orig", K.Law(form, "orig").start(), np.full_like(zgrid, mm), zgrid)
                ax.plot(zgrid, np.abs(pp[q]), color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.8)
            if q == "amp":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{r}", fontsize=9)
            if c == 0:
                ax.set_ylabel(PLABEL[form][q])
            if row == 2:
                ax.set_xlabel("z")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{'B12 pressure' if field == 'P' else 'B16 gas density'} parameters vs z: per-cell fits (points), refitted orig "
                 "law (solid) and published law (dashed) at each bin's mean mass", fontsize=9.5)
    fig.savefig(out / f"fig_K_params_vs_z_{field}.png")
    plt.close(fig)


def soap_points(C, field, logm, z):
    """Display shells (estimator B) of the cell containing logm at the output nearest z."""
    if C is None:
        return None
    uz = np.unique(C["z"])
    zn = uz[np.argmin(np.abs(uz - z))]
    b = int(np.floor((logm - KC.MASS_EDGES[0]) / 0.2 + 1e-9))
    sel = np.nonzero((np.abs(C["z"] - zn) < 1e-9) & (C["cell_bin"] == b))[0]
    if not len(sel):
        return None
    sh = C[f"shell_{field}"][sel[0]]
    ok = np.isfinite(sh[:, 4]) & (sh[:, 6] >= 5) & (sh[:, 0] >= 0)      # skip the inner cumulative [0, x1]
    return {"xa": sh[ok, 2], "xb": sh[ok, 3], "val": sh[ok, 4], "sem": sh[ok, 5], "N": int(C["cell_N"][sel[0]]), "z": float(zn)}


def fig_profiles(field, fits, prof, C, part, out):
    if prof is None:
        return
    form = FIELD_FORM[field]
    x = prof["x"]; masses = prof["masses_log10"]; zs = prof["z"]
    fig, axes = plt.subplots(len(masses), len(zs), figsize=(2.3 * len(zs) + 1, 2.0 * len(masses) + 0.6), sharex=True)
    show = [(f"published|{field}", "published", GREY, (0, (3, 2)), 1.2)]
    for r in KC.RANGES:
        show.append((f"{field}|{r}|orig", f"orig {r}", RANGE_COL[r], "-", 1.5))
    rk = rec_keys()
    if field == "P":
        show.append((rk["P"], f"{short(rk['P'])} (recommended tSZ)", CAT[6], "-", 1.4))
        show.append((rk["P_alt"], short(rk["P_alt"]), CAT[4], "-", 0.9))
    else:
        show.append((rk["G"], f"{short(rk['G'])} (recommended kSZ/DM)", CAT[6], "-", 1.4))
        show.append((rk["G_alt"], short(rk["G_alt"]), CAT[4], "-", 0.9))
    for i, lm in enumerate(masses):
        for j, z in enumerate(zs):
            ax = axes[i, j]
            for key, lab, col, ls, lw in show:
                k = f"F__{key}"
                if k in prof.files:
                    ax.plot(x, x ** 3 * prof[k][i, j], color=col, ls=ls, lw=lw, label=lab)
            pts = soap_points(C, field, lm, z)
            if pts is not None and len(pts["val"]):
                xm = np.sqrt(np.maximum(pts["xa"], 1e-3) * pts["xb"])
                xm = np.where(pts["xa"] > 0, xm, pts["xb"] / 2 ** (1 / 3))
                xerr = np.vstack([xm - np.maximum(pts["xa"], 1e-3), pts["xb"] - xm])
                ax.errorbar(xm, xm ** 3 * pts["val"], xerr=xerr, yerr=xm ** 3 * pts["sem"], fmt="o", ms=3.2, color=INK, mfc="white",
                            elinewidth=0.7, capsize=0, label="SOAP shells (mean radii)", zorder=5)
                ax.text(0.03, 0.04, f"N={pts['N']}, z={pts['z']:.2f}", transform=ax.transAxes, fontsize=6.5, color="#52514e")
            if part is not None and abs(z) < 1e-9:
                labels = list(part["bin_labels"])
                lab = f"{lm - 0.1:.1f}-{lm + 0.1:.1f}" if False else None
                b = [k for k, l in enumerate(labels) if float(l.split("-")[0]) <= lm < float(l.split("-")[1])]
                if b:
                    e = part["edges"]; xc = np.sqrt(e[1:] * np.maximum(e[:-1], 1e-3)); xc[0] = e[1] / 2 ** (1 / 3)
                    for tag, col, ls in ((f"{field}_all", INK, "-"), (f"{field}_A2ii", "#6b6a66", ":")):
                        v = part[f"{tag}_mean"][b[0]]; res = part[f"{tag}_resolved"][b[0]].astype(bool)
                        ax.plot(xc[res], xc[res] ** 3 * v[res], color=col, ls=ls, lw=1.0,
                                label=("particles all gas" if tag.endswith("all") else "particles A2-ii"))
                tk = f"testI_{'P' if field == 'P' else 'E'}_a_Rfit4"
                if tk in part.files:
                    lo = part["testI_bins_log10M_lo"]
                    bi = [k for k, l0 in enumerate(lo) if l0 <= lm < l0 + 0.2 + 1e-9]
                    if bi and np.all(np.isfinite(part[tk][bi[0]])):
                        a0, a1, a2 = part[tk][bi[0]]
                        if field == "P":
                            Fi = K.profile(x, "B12", a0, a1, 1.0, a2, -0.3)
                        else:                       # Test I fits E = n_e mu_XGP/(rho_c f_b); E / kappa_E (1.094) ~ G
                            Fi = K.profile(x, "B16", a0, 0.5, a1, a2, -0.2) / 1.094
                        ax.plot(x, x ** 3 * Fi, color=INK, ls="-.", lw=0.9, label="Test I per-bin fit (A2-ii, R_fit 4)" + ("" if field == "P" else " / 1.094"))
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"z = {z:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"$10^{{{lm:g}}}$\n" + (r"$x^3 P_e/P_{200}$" if field == "P" else r"$x^3\rho/(f_b\rho_c)$"), fontsize=8)
            if i == len(masses) - 1:
                ax.set_xlabel(r"$x=r/R_{200c}$")
            ax.set_xlim(0.01, 4)
            yv = x ** 3 * prof[f"F__published|{field}"][i, j]
            ax.set_ylim(max(np.nanmax(yv) * 1e-3, 1e-6), np.nanmax(yv) * 4)
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out / f"fig_K_profiles_{field}.png", dpi=100)
    plt.close(fig)


def fig_ratio(fits, prof, out):
    if prof is None:
        return
    x = prof["x"]; masses = prof["masses_log10"]; zs = prof["z"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    for r_i, field in enumerate(("P", "G")):
        for c, zi in enumerate((0, 2, 3)):
            ax = axes[r_i, c]
            for i, lm in enumerate(masses):
                rec = rec_keys()[field]
                for key, ls in ((rec, "-"), (f"{field}|R3|orig", (0, (4, 2)))):
                    k = f"F__{key}"
                    if k in prof.files:
                        ax.plot(x, prof[k][i, zi] / prof[f"F__published|{field}"][i, zi], color=CAT[i], ls=ls, lw=1.2,
                                label=f"$10^{{{lm:g}}}$" if ls == "-" else None)
            ax.axhline(1, color=GREY, lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{'pressure (B12)' if field == 'P' else 'gas density (B16)'}, z = {zs[zi]:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel("refitted / published")
            if r_i == 1:
                ax.set_xlabel(r"$x=r/R_{200c}$")
    axes[0, 0].legend(fontsize=7, title=r"$M_{200c}\,[M_\odot]$", title_fontsize=7)
    rk = rec_keys()
    fig.suptitle(f"Refitted / published profile: solid = recommended laws (pressure {short(rk['P'])}, gas {short(rk['G'])}), dashed = orig R3", fontsize=9.5)
    fig.savefig(out / "fig_K_ratio_to_published.png")
    plt.close(fig)


def fig_residuals(field, pred, out):
    if pred is None:
        return
    base = f"{field}__R3"
    if base + "__lnd" not in pred.files:
        return
    names = pred[base + "__name"]; z = pred[base + "__z"]; lm = pred[base + "__cell_lnM"] / math.log(10)
    aps = [a for a in KC.AP_NAMES if np.any(names == a)]
    models = [("published", base + "__published")] + [(f"orig {r}", base + f"__fit_{r}_orig") for r in KC.RANGES]
    models = [(t, k) for t, k in models if k in pred.files]
    fig, axes = plt.subplots(len(aps), len(models), figsize=(2.6 * len(models) + 1, 1.25 * len(aps) + 1), sharex=True, sharey=True, squeeze=False)
    im = None
    for i, a in enumerate(aps):
        s = names == a
        for j, (t, k) in enumerate(models):
            ax = axes[i, j]
            res = pred[base + "__lnd"][s] - pred[k][s]
            im = ax.scatter(lm[s], z[s], c=res, cmap=DIV, vmin=-0.3, vmax=0.3, s=14, marker="s", linewidths=0)
            ax.grid(False)
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{a.replace('SO/', '').replace('IS/', '')}\nz", fontsize=7.5)
            if i == len(aps) - 1:
                ax.set_xlabel(r"$\log_{10}M_{200c}$")
            ax.text(0.98, 0.9, f"rms {np.sqrt(np.mean(res ** 2)):.3f}", transform=ax.transAxes, ha="right", fontsize=6.5, color="#52514e")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("ln(SOAP / law)")
    fig.suptitle(f"{'Pressure (Y without recent AGN)' if field == 'P' else 'Gas mass'}: residual of every usable aperture "
                 "(R3 dataset) for the published and refitted 'orig' laws", fontsize=9.5)
    fig.savefig(out / f"fig_K_residuals_{field}.png", dpi=100)
    plt.close(fig)


def fig_aperture_integrals(R, out):
    import csv
    p = R / "k4_aperture_integrals.csv"
    if not p.exists():
        return
    rows = [r for r in csv.DictReader(open(p)) if r["eval_dataset"] == "R3"]
    if not rows:
        return
    zs_all = np.unique([float(r["z"]) for r in rows])
    znear = [float(zs_all[np.argmin(np.abs(zs_all - z))]) for z in ZS]
    fig, axes = plt.subplots(len(ZS), 3, figsize=(11, 2.0 * len(ZS) + 0.8), sharex=True, sharey="col")
    for c, q in enumerate(("Y500", "Y200", "Mgas200")):
        for i, zn in enumerate(znear):
            ax = axes[i, c]
            sub = [r for r in rows if r["quantity"] == q and abs(float(r["z"]) - zn) < 1e-6]
            if not sub:
                continue
            sub.sort(key=lambda r: float(r["log10M_mean"]))
            lm = np.array([float(r["log10M_mean"]) for r in sub])
            ax.plot(lm, [float(r["ratio_published"]) for r in sub], color=GREY, ls=(0, (3, 2)), marker="o", ms=3, label="published")
            for rr in KC.RANGES:
                key = f"ratio_fit_{rr}_orig"
                if key in sub[0]:
                    ax.plot(lm, [float(r[key]) for r in sub], color=RANGE_COL[rr], marker="o", ms=3, label=f"orig {rr}")
            rk = rec_keys()
            fld = "P" if q.startswith("Y") else "G"
            extra = [rk[fld], rk[fld + "_alt"]]
            for key, lab, col in [(f"ratio_fit_{k.split('|')[1]}_{k.split('|')[2]}", short(k), CAT[6 if n == 0 else 4]) for n, k in enumerate(extra)]:
                if key in sub[0]:
                    ax.plot(lm, [float(r[key]) for r in sub], color=col, lw=1.1, label=lab)
            ax.axhline(1, color=INK, lw=0.7)
            ax.fill_between(lm, 1 - np.array([float(r["sd_boot_ln"]) for r in sub]), 1 + np.array([float(r["sd_boot_ln"]) for r in sub]),
                            color="#d9d8d3", lw=0, zorder=0)
            if i == 0:
                ax.set_title({"Y500": r"$Y(<R_{500c})$", "Y200": r"$Y(<R_{200c})$", "Mgas200": r"$M_{\rm gas}(<R_{200c})$"}[q] + ": law / SOAP", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"z = {zn:.2f}")
            if i == len(ZS) - 1:
                ax.set_xlabel(r"$\log_{10}M_{200c}$ (cell mean)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(out / "fig_K_aperture_integrals.png")
    plt.close(fig)


def fig_sigma(fits, out, R=None):
    au = Path(R or HERE / "results") / "k7_fit_audit.json"
    notconv = {k for k, c in json.loads(au.read_text())["convergence"].items() if not c["converged"]} if au.exists() else set()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    kinds = ("published",) + KC.PARAMETRIZATIONS
    for a, field in enumerate(("P", "G")):
        ax = axes[a]
        w = 0.15
        for k_i, kind in enumerate(kinds):
            vals = []
            for r in KC.RANGES:
                if kind == "published":
                    vals.append(fits["published"].get(f"{field}|{r}", {}).get("sigma_d", np.nan))
                else:
                    vals.append(fits["configs"].get(f"{field}|{r}|{kind}", {}).get("sigma_d", np.nan))
            xs = np.arange(3) + (k_i - 2) * w
            bars = ax.bar(xs, vals, width=w * 0.92, color=GREY if kind == "published" else CAT[k_i - 1], label=kind)
            for b_, r, xx, v in zip(bars, KC.RANGES, xs, vals):
                bad = kind != "published" and f"{field}|{r}|{kind}" in notconv
                if bad:                                   # least squares stopped at max_nfev: not the chi^2/nu = 1 floor
                    b_.set_hatch("////"); b_.set_alpha(0.55)
                if np.isfinite(v):
                    ax.text(xx, v, f"{v:.3f}" + ("*" if bad else ""), ha="center", va="bottom", fontsize=6, rotation=90, color="#52514e")
        ax.set_xticks(range(3)); ax.set_xticklabels(list(KC.RANGES))
        ax.set_title("pressure (B12)" if field == "P" else "gas density (B16)", fontsize=9)
        ax.set_ylabel(r"$\sigma_d$ (ln units per aperture point)")
    axes[0].legend(fontsize=7)
    if notconv:
        fig.text(0.5, -0.02, "* hatched: " + ", ".join(sorted(notconv)) + " did not converge (least squares stopped at max_nfev; "
                 "results/k7_fit_audit.json), so its value is not the chi2/nu = 1 floor", ha="center", fontsize=7, color="#52514e")
    fig.savefig(out / "fig_K_sigma_d.png")
    plt.close(fig)


def fig_corr(fits, out):
    keys = ["P|R2|orig", "P|R2|free", "G|R2|orig", "G|R2|free"]
    keys = [k for k in keys if k in fits["configs"] and "corr" in fits["configs"][k]]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.8))
    for ax, k in zip(np.atleast_1d(axes), keys):
        c = np.array(fits["configs"][k]["corr"]); n = fits["configs"][k]["names"]
        im = ax.imshow(c, cmap=DIV, vmin=-1, vmax=1)
        ax.set_xticks(range(len(n))); ax.set_xticklabels(n, rotation=90, fontsize=6)
        ax.set_yticks(range(len(n))); ax.set_yticklabels(n, fontsize=6)
        ax.set_title(k, fontsize=9); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(out / "fig_K_corr.png")
    plt.close(fig)


def fig_law_params(R, fits, out):
    """All five profile parameters of the published law, the orig R3 law and the recommended (and alternative) laws vs
    M200c at z = 0, 1, 2; thick where SOAP cells with >= 10 haloes exist at that redshift (output nearest z)."""
    cz, cb, cn = [], [], []
    for f in sorted((R / "cells").glob("cells_*.npz")):
        d = np.load(f)
        cz.append(d["z"]); cb.append(d["cell_bin"]); cn.append(d["cell_N"])
    cz, cb, cn = np.concatenate(cz), np.concatenate(cb), np.concatenate(cn)
    uz = np.unique(cz)
    rk = rec_keys()
    mgrid = np.linspace(12.7, 15.4, 120)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.8))
    for row, field in enumerate(("P", "G")):
        form = FIELD_FORM[field]
        laws = [(f"published|{field}", "published", GREY, None), (f"{field}|R3|orig", "orig R3", CAT[2], None),
                (rk[field], f"{short(rk[field])} (recommended)", CAT[6], None), (rk[field + "_alt"], short(rk[field + "_alt"]), CAT[4], None)]
        for j, q in enumerate(K.QNAMES):
            ax = axes[row, j]
            for key, lab, col, _ in laws:
                if key.startswith("published"):
                    law = K.Law(form, "orig"); th = law.start()
                else:
                    it = fits["configs"].get(key)
                    if not it or "theta" not in it:
                        continue
                    law = K.Law(form, key.split("|")[2]); th = np.array(it["theta"])
                for zi, (z, ls) in enumerate(((0.0, "-"), (1.0, (0, (5, 2))), (2.0, (0, (1, 1.5))))):
                    zn = float(uz[np.argmin(np.abs(uz - z))])
                    m = (np.abs(cz - zn) < 1e-9) & (cn >= KC.MIN_CELL_HALOES)
                    top = KC.MASS_EDGES[cb[m].max() + 1] if m.any() else 12.7
                    p = law.phi(th, np.log(10 ** mgrid), np.full_like(mgrid, math.log1p(z)))
                    v = np.abs(np.asarray(p[q], float)) * np.ones_like(mgrid)
                    ax.plot(mgrid, v, color=col, ls=ls, lw=0.9, alpha=0.9)
                    ins = mgrid <= top
                    ax.plot(mgrid[ins], v[ins], color=col, ls=ls, lw=2.0,
                            label=(f"{lab}" if zi == 0 else None) if (j == 0) else None)
            if q in ("amp", "xc", "alpha", "beta"):
                ax.set_yscale("log")
            if q == "alpha" and field == "G":
                ax.axhline(10, color=INK, lw=0.6, ls=":")
            nm = {"amp": "$P_0$" if field == "P" else r"$\rho_0$", "xc": "$x_c$", "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$|\gamma|$"}[q]
            ax.set_title(f"{'pressure' if field == 'P' else 'gas'}: {nm}", fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$\log_{10} M_{200c}\ [M_\odot]$")
        axes[row, 0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Profile parameters of the laws vs mass: z = 0 solid, z = 1 dashed, z = 2 dotted; thick = inside the SOAP data "
                 "(cells with >= 10 haloes at that z), thin = extrapolation", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_K_law_params.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fits, k2, pred, C, prof, part = load(R)
    for field in ("P", "G"):
        fig_params_vs_mass(field, fits, k2, out)
        fig_params_vs_z(field, fits, k2, out)
        fig_profiles(field, fits, prof, C, part, out)
        fig_residuals(field, pred, out)
    fig_ratio(fits, prof, out)
    fig_aperture_integrals(R, out)
    fig_sigma(fits, out, R)
    fig_corr(fits, out)
    fig_law_params(R, fits, out)
    print("figures in", out, sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
