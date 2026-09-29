#!/usr/bin/env python3
"""Figures of workstream B (CLASS-SZ verification of the B12 and Lee22 tSZ). Reads only saved outputs.
Colours: data (maps) black, Battaglia12 #0072B2, Lee22 as painted #E69F00, Lee22 corrected (M1 x(1+z)^2) #009E73,
other (catalogue-weighted / diagnostics) #CC79A7."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402

OUT = Path(C.OUT); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
BLACK, B12, L22, L22C, OTHER = "black", "#0072B2", "#E69F00", "#009E73", "#CC79A7"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.8})


def logbin(ell, cl, nper=12, lmin=20, lmax=8000):
    e = np.geomspace(lmin, lmax, int(np.log10(lmax / lmin) * nper) + 1)
    lc, dl = [], []
    for a, b in zip(e[:-1], e[1:]):
        s = (ell >= a) & (ell < b)
        if s.sum() == 0:
            continue
        L = ell[s]
        lc.append(np.sqrt(a * b)); dl.append(np.mean(L * (L + 1) * cl[s] / (2 * np.pi)))
    return np.array(lc), np.array(dl)


def dl(ell, cl):
    return ell * (ell + 1) * cl / (2 * np.pi)


def fig_profiles():
    E = np.load(OUT / "profiles" / "profile_curves.npz")
    chk = json.load(open(OUT / "profiles" / "profile_checks.json"))
    x = E["x"]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), gridspec_kw=dict(height_ratios=[2.2, 1]))
    styles = {0: "-", 2: "--", 3: ":"}
    zs = E["z"]
    for i, m in enumerate(E["M"]):
        ax, ar = axs[0, i], axs[1, i]
        for j, ls in styles.items():
            pb, pl = E[f"Pe_keV_b12_{i}_{j}"], E[f"Pe_keV_lee22_{i}_{j}"]
            ax.loglog(x, x ** 3 * pb, color=B12, ls=ls, label=f"Battaglia12, z = {zs[j]:g}" if i == 0 else None)
            ax.loglog(x, x ** 3 * pl, color=L22, ls=ls, label=f"Lee22 as painted, z = {zs[j]:g}" if i == 0 else None)
            ar.semilogx(x, pl / pb, color=L22, ls=ls)
        ax.set_title(f"M200c = {m:.0e} Msun" + ("  (above Lee22 fit range)" if m * C.PAINT['h'] > C.LEE_CLIP_HINV else ""), fontsize=9.5)
        ax.set_ylim(1e-6, 3e-2)
        ar.axhline(1, color="0.5", lw=0.8)
        ar.set_ylim(0, 2.6)
        ar.set_xlabel("r / R200c")
        ax.axvspan(1.34, 4, color="0.95", lw=0)
        ar.axvspan(1.34, 4, color="0.95", lw=0)
    axs[0, 0].set_ylabel(r"$(r/R_{200c})^3\,P_e$  [keV cm$^{-3}$]")
    axs[1, 0].set_ylabel("Lee22 / B12")
    axs[0, 0].legend(fontsize=7.5, loc="upper left", ncol=1)
    # code agreement panel
    ax = axs[0, 3]; ar = axs[1, 3]
    ells = np.array(chk["C_yell"]["b12"][0]["ell"])
    for tag, col, lab in (("b12", B12, "B12"), ("lee22_low", L22, "Lee22 (M < 10^14.8/h)"), ("lee22_high", L22, None)):
        for r in chk["C_yell"].get(tag, []):
            ax.semilogx(ells, r["ratio_class_sz_over_python"], color=col, marker="o", ms=3, lw=0.8, alpha=0.7,
                        label=lab if (r is chk["C_yell"][tag][0] and lab) else None)
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_ylim(0.99, 1.015)
    ax.set_title("CLASS-SZ / exact transform, $y_\\ell(M,z)$", fontsize=9.5)
    ax.set_xlabel(r"$\ell$"); ax.legend(fontsize=7.5, loc="upper right")
    A = chk["A_xgpaint_vs_python"]
    rb = [a["ratio"] - 1 for a in A if a["model"] == "b12"]; rl = [a["ratio"] - 1 for a in A if a["model"] == "lee22"]
    ar.plot(np.arange(len(rb)), np.array(rb) * 1e10, "o", color=B12, ms=3, label="B12")
    ar.plot(np.arange(len(rl)), np.array(rl) * 1e10, "s", color=L22, ms=3, label="Lee22")
    ar.set_ylim(-1, 1)
    ar.set_ylabel(r"XGPaint $y(\theta)$/Python $-1$ [$10^{-10}$]", fontsize=8)
    ar.set_xlabel("test point (M, z, r)"); ar.legend(fontsize=7.5)
    fig.suptitle("Pressure profiles as painted (paint cosmology h = 0.68, P_e = 0.5176 P_th for B12; Lee22 fits P_e; "
                 "grey: outside Lee22 fit range 0.04-1.34 R200c)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_profiles.png", dpi=150); fig.savefig(FIG / "fig1_profiles.pdf")
    plt.close(fig)


def fig_b12():
    d = np.load(OUT / "analysis" / "curves.npz")
    ell = d["ell"]; pw2 = d["pixwin2"]
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.3]))
    lc, m = logbin(ell, d["map_B12"]); ax.loglog(lc, 1e12 * m, "o", color=BLACK, ms=4, label="B12 map (raw, point-sampled)")
    lc, md = logbin(ell, np.where(pw2 > 0, d["map_B12"] / pw2, 0)); ax.loglog(lc, 1e12 * md, "o", mfc="none", color=BLACK, ms=4,
                                                                            label="B12 map / pixel window$^2$")
    L = ell[10:]
    ax.loglog(L, 1e12 * dl(L, d["th_CS_b12_paint_all"][10:]), color=B12, label="CLASS-SZ 1h+2h (T08 M200c, x_out=4, high precision)")
    ax.loglog(L, 1e12 * dl(L, d["th_CS_b12_paint_all_1h"][10:]), color=B12, ls="--", lw=1.2, label="CLASS-SZ 1h")
    ax.loglog(L, 1e12 * dl(L, d["th_CS_b12_paint_all_2h"][10:]), color=B12, ls=":", lw=1.2, label="CLASS-SZ 2h")
    ax.loglog(L, 1e12 * dl(L, d["th_CAT_b12"][10:]), color=OTHER, label="catalogue-weighted halo model (aperture)")
    ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$"); ax.set_ylim(0.004, 3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Battaglia12: painted HalfDome map vs CLASS-SZ (paint cosmology)", fontsize=10)
    for key, col, ls, lab in (("th_CS_b12_paint_all", B12, "-", "map / CLASS-SZ"),
                              ("th_CAT_b12", OTHER, "-", "map / catalogue-weighted"),
                              ("th_PY_aperture_b12", B12, "--", "map / Python T08 aperture")):
        lc, r = logbin(ell, d["map_B12"] / np.where(d[key] > 0, d[key], np.inf))
        th = logbin(ell, d[key])[1]; mm = logbin(ell, d["map_B12"])[1]
        ar.semilogx(lc, mm / th, color=col, ls=ls, marker="o", ms=3, label=lab)
    lc, _ = logbin(ell, d["map_B12"])
    ar.semilogx(lc, logbin(ell, d["th_PY_sphere_b12"])[1] / logbin(ell, d["th_CS_b12_paint_all"])[1], color="0.35", ls=":",
                label="Python sphere / CLASS-SZ (validation)")
    ar.semilogx(lc, logbin(ell, d["th_CSdef_b12_paint_all"])[1] / logbin(ell, d["th_CS_b12_paint_all"])[1], color="0.6",
                ls="-.", label="CLASS-SZ default precision / high precision")
    ar.axhline(1, color="0.5", lw=0.8); ar.set_ylim(0.6, 1.3)
    ar.set_xlabel(r"$\ell$"); ar.set_ylabel("ratio"); ar.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_b12_map_vs_class_sz.png", dpi=150); fig.savefig(FIG / "fig2_b12_map_vs_class_sz.pdf")
    plt.close(fig)


def fig_lee22():
    d = np.load(OUT / "analysis" / "curves.npz")
    ell = d["ell"]
    fig, (ax, ar, ar2) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.1, 1.1]))
    B = lambda k: logbin(ell, d[k])
    lc, m = B("map_L22"); ax.loglog(lc, 1e12 * m, "o", color=BLACK, ms=4, label="Lee22 map (all halos)")
    lc, m = B("map_L22calib"); ax.loglog(lc, 1e12 * m, "s", mfc="none", color=BLACK, ms=4, label="Lee22 map, calibrated range")
    lc, m = B("map_B12"); ax.loglog(lc, 1e12 * m, "^", color="0.55", ms=4, label="B12 map")
    L = ell[10:]
    for k, col, ls, lab in (("th_CSX_lee22", L22, "-", "CLASS-SZ Lee22 as painted"),
                            ("th_CS_lee22_paint_calib", L22, "--", "CLASS-SZ Lee22, calibrated range"),
                            ("th_CS_b12_paint_all", B12, "-", "CLASS-SZ B12"),
                            ("th_CSX_lee22m1", L22C, "-", "CLASS-SZ Lee22 x (1+z)$^2$ (M1)"),
                            ("th_CS_lee22m1_paint_calib", L22C, "--", "CLASS-SZ Lee22 x (1+z)$^2$, calibrated range")):
        ax.loglog(L, 1e12 * dl(L, d[k][10:]), color=col, ls=ls, label=lab)
    ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$"); ax.set_ylim(0.01, 60)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("Lee22 no-c pressure (Table 7) as painted vs CLASS-SZ", fontsize=10)
    lc = B("map_L22")[0]
    for num, den, col, ls, lab in (("map_L22", "th_CSX_lee22", L22, "-", "Lee22 map / CLASS-SZ"),
                                   ("map_L22calib", "th_CS_lee22_paint_calib", L22, "--", "Lee22 calib map / CLASS-SZ"),
                                   ("map_B12", "th_CS_b12_paint_all", B12, "-", "B12 map / CLASS-SZ"),
                                   ("map_L22", "th_CAT_lee22", OTHER, "-", "Lee22 map / catalogue-weighted")):
        ar.semilogx(lc, B(num)[1] / B(den)[1], color=col, ls=ls, marker="o", ms=3, label=lab)
    ar.axhline(1, color="0.5", lw=0.8); ar.set_ylim(0.6, 1.4); ar.set_ylabel("map / model"); ar.legend(fontsize=7.5, ncol=2)
    for num, den, col, ls, mk, lab in (("map_L22", "map_B12", BLACK, "-", "o", "maps: Lee22 / B12"),
                                       ("th_CSX_lee22", "th_CS_b12_paint_all", L22, "-", None, "CLASS-SZ: Lee22 / B12"),
                                       ("th_CAT_lee22", "th_CAT_b12", OTHER, "-", None, "catalogue-weighted: Lee22 / B12"),
                                       ("map_L22calib", "map_B12", BLACK, "--", "s", "maps: Lee22 calib / B12 (all)"),
                                       ("th_CSX_lee22m1", "th_CS_b12_paint_all", L22C, "-", None, "CLASS-SZ: M1 / B12"),
                                       ("th_CSX_hybrid", "th_CS_b12_paint_all", OTHER, ":", None, "CLASS-SZ: hybrid (B12 > 10^14.8/h) / B12")):
        ar2.semilogx(lc, B(num)[1] / B(den)[1], color=col, ls=ls, marker=mk, ms=3, label=lab)
    ar2.axhline(1, color="0.5", lw=0.8); ar2.set_yscale("log"); ar2.set_ylim(0.2, 30)
    ar2.set_ylabel("ratio to B12"); ar2.set_xlabel(r"$\ell$"); ar2.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_lee22_map_vs_class_sz.png", dpi=150); fig.savefig(FIG / "fig3_lee22_map_vs_class_sz.pdf")
    plt.close(fig)


def fig_decomp():
    res = json.load(open(OUT / "analysis" / "results.json"))
    dec = res["decomposition"]["python_T08_aperture"]
    cat = res["decomposition"]["catalogue_weighted_aperture"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    ells = list(dec.keys())
    shades = ["#FBE3B0", "#F2C166", "#E69F00", "#9A6A00"]
    for col, (src, name) in enumerate(((dec, "Tinker08 halo model"), (cat, "catalogue-weighted (85.2M halos)"))):
        for row, key in enumerate(("mass", "redshift")):
            ax = axs[row, col]
            nb = len(src[ells[0]][key]); w = 0.8 / len(ells)
            for k, L in enumerate(ells):
                vals = [r["contrib_to_ratio"] for r in src[L][key]]
                ax.bar(np.arange(nb) + (k - 1.5) * w, vals, width=w * 0.92, color=shades[k],
                       label=f"$\\ell$ = {L}: Lee22/B12 = {src[L]['ratio_total']:.2f}")
            if key == "mass":
                labs = [f"{r['m_lo']:.1e}-\n{r['m_hi']:.1e}" for r in src[ells[0]][key]]
                ax.set_xlabel("M200c bin [Msun/h, h = 0.68]  (last bin: above the Lee22 fit range)")
            else:
                labs = [f"{r['z_lo']:.2g}-{r['z_hi']:.2g}" for r in src[ells[0]][key]]
                ax.set_xlabel("redshift bin")
            ax.set_xticks(np.arange(nb)); ax.set_xticklabels(labs, fontsize=7.5)
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_ylabel(r"$[C_\ell^{\rm Lee22}(\rm bin) - C_\ell^{\rm B12}(\rm bin)]/C_\ell^{\rm B12}$")
            ax.set_title(f"{name}: {key}-bin share of Lee22/B12 - 1", fontsize=9.5)
            ax.legend(fontsize=7.5, loc="upper left" if key == "mass" else "upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_lee22_over_b12_decomposition.png", dpi=150); fig.savefig(FIG / "fig4_lee22_over_b12_decomposition.pdf")
    plt.close(fig)


def fig_joint():
    res = json.load(open(OUT / "analysis" / "results.json"))
    dec = res["decomposition"]["python_T08_aperture"]
    ells = list(dec.keys())
    mlab = ["4.5e12-1e13", "1e13-3e13", "3e13-1e14", "1e14-3e14", "3e14-6.3e14", "> 6.3e14\n(above fit)"]
    zlab = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3.9"]
    fig, axs = plt.subplots(1, len(ells), figsize=(15, 4.2), sharey=True)
    for ax, L in zip(axs, ells):
        M = np.array(dec[L]["joint_contrib_to_ratio"]["matrix"])
        v = max(0.05, np.abs(M).max())
        im = ax.imshow(M, origin="lower", cmap="PuOr_r", vmin=-v, vmax=v, aspect="auto")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if abs(M[a, b]) >= 0.005:
                    ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M[a, b]) > 0.6 * v else "black")
        ax.set_xticks(range(len(zlab))); ax.set_xticklabels(zlab, fontsize=7.5, rotation=30)
        ax.set_yticks(range(len(mlab))); ax.set_yticklabels(mlab, fontsize=7.5)
        ax.set_xlabel("redshift")
        ax.set_title(f"$\\ell$ = {L}: Lee22/B12 - 1 = {dec[L]['ratio_total'] - 1:+.2f}", fontsize=9.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axs[0].set_ylabel("M200c [Msun/h], h = 0.68")
    fig.suptitle("Where the Lee22 (as painted) - Battaglia12 difference comes from: (C_Lee22 - C_B12)(M, z bin) / C_B12(total), "
                 "Tinker08 halo model, 4 R200c aperture", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_lee22_minus_b12_mass_z.png", dpi=150); fig.savefig(FIG / "fig5_lee22_minus_b12_mass_z.pdf")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["profiles", "b12", "lee22", "decomp", "joint"]
    for w in which:
        globals()["fig_" + w]()
        print("done", w)
