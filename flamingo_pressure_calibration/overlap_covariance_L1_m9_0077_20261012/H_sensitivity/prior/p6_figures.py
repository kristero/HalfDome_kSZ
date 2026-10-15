"""P6: summary figures of the earlier results (local, light).  Reads only the tables written by p1-p5 and the stored
importance arrays.  Colours: the dataviz reference categorical palette in fixed order (light mode), line styles as a
secondary encoding; heat maps use the single-hue (blue) sequential ramp.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import prior_common as C

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2, 2, 2))]
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                     "#184f95", "#0d366b"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

P_OBS = [("yytot", 100, "tSZ yy, ℓ=100"), ("yytot", 1000, "tSZ yy, ℓ=1000"), ("yytot", 3000, "tSZ yy, ℓ=3000"),
         ("ymean", None, "mean y"), ("ydmtot_P", 1000, "tSZ×DM (P side), ℓ=1000")]
E_OBS = [("ksz1h", 100, "kSZ 1h, ℓ=100"), ("ksz1h", 1000, "kSZ 1h, ℓ=1000"), ("ksz1h", 3000, "kSZ 1h, ℓ=3000"),
         ("dm_mean", None, "mean DM"), ("dm_q16", None, "DM Q16"), ("dm_q50", None, "DM Q50"),
         ("dm_q999", None, "DM Q99.9")]


def f_map(d, key, ell):
    a = d[f"imp__{key}"]
    if ell is not None:
        a = a[:, :, int(np.flatnonzero(d["ell"] == ell)[0])]
    return a.sum(0)


def k_map(K, key, ell):
    kell = K["ell"]
    ie = None if ell is None else int(np.flatnonzero(kell == ell)[0])
    tab = {"yytot": ("J_cl_total__P4", "O_cl_total__P4"), "ymean": ("J_mean__P4", "O_mean__P4"),
           "ksz1h": ("J_ksz_cl_1h__E4", "O_ksz_cl_1h__E4"), "dm_mean": ("J_dm_mean__E4", "O_dm_mean__E4")}
    if key in tab:
        J, O = K[tab[key][0]], K[tab[key][1]]
        J = J[..., ie] if ie is not None else J
        O = O[ie] if ie is not None and np.ndim(O) else O
    elif key.startswith("dm_q"):
        qi = ["dm_q16", "dm_q50", "dm_q84", "dm_q975", "dm_q999"].index(key)
        J, O = K["J_dm_quantiles__E4"][qi], K["O_dm_quantiles__E4"][qi]
    elif key == "ydmtot_P":
        J = K["J_cross_1h_P__P4xE4"][..., ie] + K["J_cross_2h_P__P4xE4"][..., ie]
        O = K["O_cross_cl_1h__P4xE4"][ie] + K["O_cross_cl_2h__P4xE4"][ie]
    else:
        raise KeyError(key)
    return J.sum(1) / O                                     # [67]; normalisation constant irrelevant (peak-scaled)


def fig_radial():
    d = np.load(C.F_IMP)
    K = np.load(C.K_JAC)
    fe, xe = d["shell_edges"], K["x200_edges"]
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for col, obs in enumerate((P_OBS, E_OBS)):
        a = ax[0, col]
        for i, (key, ell, lab) in enumerate(obs):
            imp = f_map(d, key, ell)
            c = np.r_[0, np.cumsum(imp)] / imp.sum()
            a.plot(fe, c, color=CAT[i], ls=LS[i], lw=2, label=lab)
        a.axhline(0.9, color=INK2, lw=0.8, ls=":")
        a.text(0.05, 0.915, "90%", color=INK2, fontsize=8)
        for x in (1, 2, 3):
            a.axvline(x, color=INK2, lw=0.6, ls="--")
        a.set_xlim(0, 4)
        a.set_ylim(0, 1.05)
        a.set_xlabel(r"$x = r/R_{200c}$")
        a.set_ylabel(r"cumulative importance $\sum_{s:\,x_s<x} I_s$")
        a.set_title(("(a) pressure observables" if col == 0 else "(b) electron observables")
                    + " - FLAMINGO truth, 35 shells", fontsize=10, loc="left")
        a.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
        b = ax[1, col]
        for i, (key, ell, lab) in enumerate(obs):
            j = k_map(K, key, ell)
            w = np.log(xe[2:] / xe[1:-1])
            dens = j[1:] / w
            dens = dens / np.max(dens)
            xs = np.r_[xe[1:-1], xe[-1]]
            b.step(xs, np.r_[dens, dens[-1]], where="post", color=CAT[i], ls=LS[i], lw=1.8, label=lab)
        b.set_xscale("log")
        b.set_xlim(0.025, 4)
        b.axhline(0, color=INK2, lw=0.6)
        for x in (1, 2, 3):
            b.axvline(x, color=INK2, lw=0.6, ls="--")
        b.set_xlabel(r"$x = r/R_{200c}$")
        b.set_ylabel(r"$dI/d\ln x$ / peak")
        b.set_title(("(c) pressure" if col == 0 else "(d) electrons")
                    + " - kernel Jacobians at the 4R fits, 67 shells", fontsize=10, loc="left")
        b.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.9)
    fig.suptitle("Where the signal comes from: importance of each radial shell (all masses, z=0 calibration)",
                 fontsize=11)
    fig.savefig(C.FIGURES / "prior_importance_radial.png", dpi=150)
    fig.savefig(C.FIGURES / "prior_importance_radial.svg")
    plt.close(fig)


def fig_r90():
    r = pd.read_csv(C.TABLES / "imp_radii_F.csv")
    obs = [("yytot_100", "tSZ ℓ=100"), ("yytot_1000", "tSZ ℓ=1000"), ("yytot_3000", "tSZ ℓ=3000"),
           ("ydmtot_P_1000", "tSZ×DM P ℓ=1000"), ("ksz1h_100", "kSZ ℓ=100"), ("ksz1h_1000", "kSZ ℓ=1000"),
           ("ksz1h_3000", "kSZ ℓ=3000"), ("ydmtot_E_1000", "tSZ×DM E ℓ=1000"), ("dm_mean", "mean DM"),
           ("dm_var", "DM var"), ("dm_q16", "DM Q16"), ("dm_q50", "DM Q50"), ("dm_q999", "DM Q99.9")]
    bins = [b for b in r.masses.unique() if b not in ("all", "calibrated 13.5-15.5")]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    for p, (col, title, vmax, fmt) in enumerate((("R90", "R90: radius enclosing 90% of the importance [R200c]", 4,
                                                 "{:.2f}"),
                                                ("share_beyond_2R", "share of the importance beyond 2 R200c [%]", 80,
                                                 "{:.0f}"))):
        M = np.full((len(bins), len(obs)), np.nan)
        for i, b in enumerate(bins):
            for j, (o, _) in enumerate(obs):
                q = r[(r.masses == b) & (r.observable == o)]
                if len(q):
                    M[i, j] = q[col].iloc[0] * (100 if col.startswith("share") else 1)
        a = ax[p]
        im = a.imshow(np.clip(M, 0, None), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    v = M[i, j]
                    a.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                           color="white" if np.clip(v, 0, None) > 0.55 * vmax else INK)
        a.set_xticks(range(len(obs)))
        a.set_xticklabels([o[1] for o in obs], rotation=55, ha="right", fontsize=8)
        a.set_yticks(range(len(bins)))
        a.set_yticklabels(bins, fontsize=8)
        a.set_ylabel(r"engine mass bin, $\log_{10}(M_{200c}/M_\odot)$")
        a.set_title(title, fontsize=10, loc="left")
        a.grid(False)
        fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle("Per mass bin: how far out each observable's sensitivity extends (Test F importance maps, truth)",
                 fontsize=11)
    fig.savefig(C.FIGURES / "prior_R90_by_mass.png", dpi=150)
    fig.savefig(C.FIGURES / "prior_R90_by_mass.svg")
    plt.close(fig)


def fig_grids():
    F = pd.read_csv(C.TABLES / "grid_F_B12B16_long.csv")
    F = F[(F.objective == "a") & (F.prediction == "deterministic")]
    Cc = pd.read_csv(C.TABLES / "grid_F_controls_B12B16_long.csv")
    A2 = Cc[Cc.selection == "a2ii"]
    fams = ("tSZ", "kSZ", "DM", "tSZxDM")
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.4), constrained_layout=True)
    norm = LogNorm(vmin=1, vmax=60)
    for r_, (df, lab) in enumerate(((F, "all gas"), (A2, "A2-ii partition"))):
        for c_, fam in enumerate(fams):
            P = df.pivot_table(index="R_fit", columns="R_cut", values=f"rms_{fam}")
            a = ax[r_, c_]
            im = a.imshow(P.values, cmap=SEQ, norm=norm, aspect="auto", origin="lower")
            i0, j0 = np.unravel_index(np.nanargmin(P.values), P.shape)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P.values[i, j]
                    a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                           color="white" if v > 12 else INK, fontweight="bold" if (i, j) == (i0, j0) else None)
            a.add_patch(plt.Rectangle((j0 - .5, i0 - .5), 1, 1, fill=False, ec=CAT[7], lw=2))
            a.set_xticks(range(P.shape[1]))
            a.set_xticklabels([f"{x:g}" for x in P.columns])
            a.set_yticks(range(P.shape[0]))
            a.set_yticklabels([f"{x:g}" for x in P.index])
            a.set_xlabel(r"$R_{\rm cut}$ [$R_{200c}$]")
            a.set_ylabel(r"$R_{\rm fit}$ [$R_{200c}$]")
            a.set_title(f"{fam} RMS [%], {lab}", fontsize=9, loc="left")
            a.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="family RMS |ΔO/O| [%] (log scale); red box = minimum")
    fig.suptitle("B12/B16 refits (objective a, one mean profile per bin) against the FLAMINGO truth", fontsize=11)
    fig.savefig(C.FIGURES / "prior_fit_cut_grids_B12B16.png", dpi=150)
    fig.savefig(C.FIGURES / "prior_fit_cut_grids_B12B16.svg")
    plt.close(fig)


def fig_models():
    G = pd.read_csv(C.TABLES / "grid_G_long.csv")
    L = pd.read_csv(C.TABLES / "lee22_observables.csv")
    sel = (G.objective == "a") & (G.R_fit == 4) & (G.R_cut == 4) & (G.prediction == "S1")
    rows = []
    for model, lab in (("P0", "B12 refit (P0)"), ("P1", "B12 + power law (P1, Test F)"),
                       ("P2m", "B12 x f_tr + 2-halo, T(M) (P2m)")):
        for tgt, ev, tl in (("main", "paint", "all-gas fit x p_keep"), ("a2ii", "a2ii", "fit to A2-ii stack")):
            q = G[sel & (G.model == model) & (G.fit_target == tgt) & (G["eval"] == ev)]
            rows.append(("tSZ", lab, tl, q.rms_tSZ.iloc[0]))
    q = L[(L.field == "P") & (L.dset == "mass") & (L.R_fit == 4) & (L.model == "BPLall+O2") & (L["eval"] == "paint")
          & (L.prediction == "S1")]
    rows.append(("tSZ", "Lee22 gNFW BPLall+O2 (14 par.)", "all-gas fit x p_keep", q.rms_tSZ.iloc[0]))
    for model, lab in (("E0", "B16 refit (E0)"), ("E1", "B16 + power law (E1, Test F)"),
                       ("E2gf", "B16 x f_tr + 2-halo (E2gf)")):
        for tgt, ev, tl in (("main", "paint", "all-gas fit x p_keep"), ("a2ii", "a2ii", "fit to A2-ii stack")):
            q = G[sel & (G.model == model) & (G.fit_target == tgt) & (G["eval"] == ev)]
            rows.append(("kSZ", lab, tl, q.rms_kSZ.iloc[0]))
            rows.append(("DM", lab, tl, q.rms_DM.iloc[0]))
    q = L[(L.field == "E") & (L.dset == "mass") & (L.R_fit == 4) & (L.model == "BPLall+O2") & (L["eval"] == "paint")
          & (L.prediction == "S1")]
    rows.append(("kSZ", "Lee22 gNFW BPLall+O2 (12 par.)", "all-gas fit x p_keep", q.rms_kSZ.iloc[0]))
    rows.append(("DM", "Lee22 gNFW BPLall+O2 (12 par.)", "all-gas fit x p_keep", q.rms_DM.iloc[0]))
    df = pd.DataFrame(rows, columns=["family", "model", "target", "rms"])
    df.to_csv(C.TABLES / "model_comparison_4R_S1.csv", index=False, float_format="%.4g")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for k, fam in enumerate(("tSZ", "kSZ", "DM")):
        a = ax[k]
        sub = df[df.family == fam]
        models = list(dict.fromkeys(sub.model))
        for t_i, (tl, mk) in enumerate((("all-gas fit x p_keep", "o"), ("fit to A2-ii stack", "s"))):
            s2 = sub[sub.target == tl]
            ys = [models.index(m) + (0.12 if t_i else -0.12) for m in s2.model]
            a.scatter(s2.rms, ys, s=60, marker=mk, color=CAT[t_i], edgecolor="white", linewidth=1.5, zorder=3,
                      label=tl)
            for x, y in zip(s2.rms, ys):
                a.text(x * 1.08, y, f"{x:.2f}", va="center", fontsize=7.5, color=INK2)
        a.set_yticks(range(len(models)))
        a.set_yticklabels(models, fontsize=8)
        a.set_xscale("log")
        a.set_xlim(0.2, 12)
        a.set_xlabel(f"{fam} family RMS [%]")
        a.set_ylim(len(models) - 0.5, -0.5)
        if k == 0:
            h, l = a.get_legend_handles_labels()
            fig.legend(h, l, loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(r"Profile forms at $R_{\rm fit}=R_{\rm cut}=4R_{200c}$ vs the A2-ii truth, S1 scatter, "
                 r"objective a (Test G)", fontsize=11, x=0.02, ha="left")
    fig.savefig(C.FIGURES / "prior_model_comparison_4R.png", dpi=150)
    fig.savefig(C.FIGURES / "prior_model_comparison_4R.svg")
    plt.close(fig)


if __name__ == "__main__":
    fig_radial()
    fig_r90()
    fig_grids()
    fig_models()
    print("figures written to", C.FIGURES)
