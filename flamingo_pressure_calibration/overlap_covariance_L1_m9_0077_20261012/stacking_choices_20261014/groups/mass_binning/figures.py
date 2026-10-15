"""Figures of the mass-binning group (PNG, figures/)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbcommon import (BAND_LABELS, CAL_LABELS, EDGES_FIT, FIG, GROUP, MARKERS, PAL, TAB, WORK, X_FIT, interp_lnF,  # noqa: E402
                      sc)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})
R = json.loads((GROUP / "results.json").read_text())
A = R["analyses"]
ST = np.load(WORK / "scheme_stacks.npz")
FLAB = {"P": "pressure P_e/P200", "E": "electrons n_e (rho_c f_b units)", "G": "gas density", "T_e": "T_e/T200",
        "K": "K/K200"}
SCHEMES = ["M200c_0.1", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5", "Lee22_h"]
SLAB = {"M200c_0.1": "0.1 dex M200c", "M200c_0.3": "0.3 dex M200c", "M200c_0.5": "0.5 dex M200c",
        "M500c_0.2": "0.2 dex M500c", "M500c_0.5": "0.5 dex M500c (B24-like)", "Lee22_h": "0.2 dex h$^{-1}$M$_\\odot$ (Lee22)"}


BANDS_ALL = [(0, .1), (.1, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4)]


def shade_cal(ax):
    ax.axvspan(13.5, 15.5, color="0.92", zorder=0, lw=0)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1: scheme band ratios vs mass
bands_show = ["0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "2-3"]
fig, axs = plt.subplots(2, len(bands_show), figsize=(15, 6.4), sharex=True)
for r, f in enumerate(["P", "E"]):
    for c, bl in enumerate(bands_show):
        ax = axs[r, c]
        shade_cal(ax)
        ax.axhline(0, color="0.3", lw=0.8)
        for k, S in enumerate(SCHEMES):
            xs, ys, es = [], [], []
            for e in A["scheme_vs_reference_same_mass"][f][S]:
                v = e["bands"][bl]
                if v and v["err"] is not None and e["N"] >= 6:
                    xs.append(e["mean_logm200c"] + (k - 2.5) * 0.012)
                    ys.append(100 * (v["ratio"] - 1))
                    es.append(100 * v["err"])
            ax.errorbar(xs, ys, es, color=PAL[k], marker=MARKERS[k], ms=4, lw=1.1, capsize=0, label=SLAB[S])
        ax.set_ylim(-15, 15)
        ax.set_title(f"{f}: {bl} R200c")
        if c == 0:
            ax.set_ylabel(f"{f}: ratio - 1 [%]")
        if r == 1:
            ax.set_xlabel("mean log10 M200c of the bin [Msun]")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.suptitle("Bin schemes vs the 0.2-dex M200c reference interpolated to the same mean M200c "
             "(grey: calibrated range; error bars: paired bootstrap)", y=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 0.98))
save(fig, "fig01_scheme_band_ratios.png")

# ---------------------------------------------------------------- Fig 2: observable impacts of the schemes (2 metrics)
obs = [("P", "tSZ_yy_total_ell100", "tSZ C$_\\ell$, $\\ell$=100"), ("P", "tSZ_yy_total_ell1000", "tSZ $\\ell$=1000"),
       ("P", "tSZ_yy_total_ell3000", "tSZ $\\ell$=3000"), ("E", "kSZ_1h_ell1000", "kSZ 1h $\\ell$=1000"),
       ("E", "kSZ_1h_ell3000", "kSZ 1h $\\ell$=3000"), ("E", "DM_mean", "mean DM"), ("E", "DM_var", "DM variance")]
SCH2 = ["M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5", "M500c_0.2", "M500c_0.5", "Lee22_h"]
SLAB2 = dict(SLAB, **{"M200c_0.2": "0.2 dex M200c (reference)"})
CPI = A["scheme_impacts_continuous_painting"]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
for a, (title, getter) in enumerate((
        ("A: scheme at the reference's node masses vs reference stacks\n(reference exact by construction)",
         lambda S, f, o: (A["scheme_impacts_smooth_painting"][S][f]["impact"][o])),
        ("B: continuous painting (each halo at its own M200c)\nvs the true bin-mean profiles",
         lambda S, f, o: CPI[S][f]["vs_truth"][o]))):
    ax = axs[a]
    for i, S in enumerate(SCH2):
        for j, (f, o, ol) in enumerate(obs):
            v = getter(S, f, o)
            ax.errorbar(v["pct"], i + (j - 3) * 0.1, xerr=v["err_pct"], fmt=MARKERS[j], color=PAL[j], ms=5, lw=1,
                        label=ol if (i == 0 and a == 1) else None)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("first-order change of the observable [%]")
    ax.set_title(title)
axs[0].set_yticks(range(len(SCH2)))
axs[0].set_yticklabels([SLAB2[S] for S in SCH2])
axs[0].invert_yaxis()
axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.suptitle("Observable impact of the mass-bin scheme (calibrated bins 13.5-15.5; paired-bootstrap 1 sigma, census design)")
fig.tight_layout()
save(fig, "fig02_observable_impacts.png")

# ---------------------------------------------------------------- Fig 13: top-bin treatments and normalisation
top_ = A["top_bins"]
rows13 = [("continuous: top two pooled (27) as one node", top_["continuous_painting"]["pool_15.1_node"], "vs_reference_painted_same_way"),
          ("continuous: >=14.9 pooled (154) as one node", top_["continuous_painting"]["pool_14.9_node"], "vs_reference_painted_same_way"),
          ("continuous: top two from trend fit 14.3-15.1", top_["continuous_painting"]["trend_14.3-15.1_nodes"], "vs_reference_painted_same_way"),
          ("continuous: top two dropped, trend extrapolated", top_["continuous_painting"]["drop_top_extrapolate"], "vs_reference_painted_same_way"),
          ("continuous: top two dropped, 14.9-15.1 held flat", top_["continuous_painting"]["drop_top_flat"], "vs_reference_painted_same_way"),
          ("bin stacks: top two painted with pooled (27)", top_["impacts"]["pool_15.1_step"], None),
          ("bin stacks: top two painted with 14.9-15.1", top_["impacts"]["nearest_14.9-15.1"], None),
          ("bin stacks: bin-mean P200/R200c (0.2 dex)", {f: R["variants"]["mass_binning:binmean_norm"]["impacts"][f]["impact"]
                                                        for f in "PE"}, None)]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for i, (lab, d, key) in enumerate(rows13):
    for j, (f, o, ol) in enumerate(obs):
        v = d[f][key][o] if key else d[f][o]
        ax.errorbar(v["pct"], i + (j - 3) * 0.1, xerr=v["err_pct"], fmt=MARKERS[j], color=PAL[j], ms=5, lw=1,
                    label=ol if i == 0 else None)
ax.axvline(0, color="0.3", lw=0.8)
ax.set_yticks(range(len(rows13)))
ax.set_yticklabels([r[0] for r in rows13])
ax.invert_yaxis()
ax.set_xlim(-6.5, 5)
ax.set_xlabel("first-order change relative to the reference treated the same way [%]")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
ax.set_title("Sparse top bins (15.1-15.3: 21, 15.3-15.5: 6 haloes) and bin-mean normalisation\n"
             "(paired-bootstrap 1 sigma, census design)")
save(fig, "fig13_topbins_normalisation_impacts.png")

# ---------------------------------------------------------------- Fig 3: residual mass trend per 0.1 dex
bands_t = ["0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for a, f in enumerate(["P", "E"]):
    ax = axs[a]
    shade_cal(ax)
    ax.axhline(0, color="0.3", lw=0.8)
    for k, bl in enumerate(bands_t):
        xs, ys, es = [], [], []
        for e in A["mass_trend"][f]:
            v = e.get("slope_central_per_0p1dex", {}).get(bl) if e.get("slope_central_per_0p1dex") else None
            if v and e["bin"] not in ("15.1-15.3",):
                xs.append(e["mean_logm200c"])
                ys.append(100 * v["value"])
                es.append(100 * v["err"])
        ax.errorbar(xs, ys, es, color=PAL[k], marker=MARKERS[k], ms=4, lw=1.2, capsize=0, label=f"{bl} R200c")
    ax.set_xlabel("log10 M200c [Msun]")
    ax.set_title(f"{FLAB[f]}")
    if a == 0:
        ax.set_ylabel("d ln F per 0.1 dex of M200c [%]")
axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.suptitle("Residual mass trend of the self-similarly normalised stacks (central differences of the 0.2-dex bins)")
fig.tight_layout()
save(fig, "fig03_mass_trend.png")

# ---------------------------------------------------------------- Fig 4: wide bins vs 0.1-dex stacks at the same mass
fig, axs = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
for r, f in enumerate(["P", "E"]):
    for c, S in enumerate(["M200c_0.3", "M200c_0.5"]):
        ax = axs[r, c]
        ax.axhline(0, color="0.3", lw=0.8)
        k = 0
        Fn = {kk: ST[f"{f}__M200c_0.1__{kk}"] for kk in ("value", "m200", "strict", "boot")}
        V = {kk: ST[f"{f}__{S}__{kk}"] for kk in ("value", "m200", "strict", "boot", "labels")}
        for j, lab in enumerate(V["labels"]):
            lo = float(lab.split("-")[0])
            if lo < 13.0:
                continue
            Fi, _, _, _ = interp_lnF(Fn["m200"], Fn["value"], V["m200"][j])
            Bi, _, _, _ = interp_lnF(Fn["m200"], np.moveaxis(Fn["boot"], 1, 0), V["m200"][j])
            ratio = V["value"][j] / Fi - 1
            err = np.nanstd(V["boot"][j] / Bi, axis=0, ddof=1)
            m = V["strict"][j] & np.isfinite(ratio)
            ax.fill_between(X_FIT[m], 100 * (ratio[m] - err[m]), 100 * (ratio[m] + err[m]), color=PAL[k], alpha=0.15,
                            lw=0)
            ax.plot(X_FIT[m], 100 * ratio[m], color=PAL[k], marker=MARKERS[k], ms=3.5, lw=1.3, label=lab)
            k += 1
        ax.set_xscale("log")
        ax.set_ylim(-15, 15)
        ax.set_title(f"{f}: {SLAB[S]} stack / 0.1-dex stacks at its mean mass")
        if c == 0:
            ax.set_ylabel("ratio - 1 [%]")
        if r == 1:
            ax.set_xlabel("r / R200c")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, title="bin")
fig.tight_layout()
save(fig, "fig04_wide_vs_fine.png")

# ---------------------------------------------------------------- Fig 5: own vs bin-mean normalisation
rn = A["normalisation_own_vs_binmean"]
JS = np.load(WORK / "jensen_shells.npz")
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True)
show = [("M200c_0.2", "13.5-13.7"), ("M200c_0.2", "14.5-14.7"), ("M200c_0.5", "13.5-14.0"), ("M200c_0.5", "14.5-15.0")]
BCEN = [np.sqrt(max(lo, 0.05) * hi) for lo, hi in BANDS_ALL]
for a, f in enumerate(["P", "E", "T_e", "K"]):
    ax = axs[a]
    ax.axhline(0, color="0.3", lw=0.8)
    for k, (S, lab) in enumerate(show):
        e = [x for x in rn["bins"][S] if x["bin"] == lab][0]
        xs, ys, es = [], [], []
        for bl, xc in zip(BAND_LABELS, BCEN):
            v = e["fields"][f][bl]
            if v and v["err"] is not None:
                xs.append(xc)
                ys.append(100 * (v["ratio"] - 1))
                es.append(100 * v["err"])
        ax.errorbar(xs, ys, es, color=PAL[k], marker=MARKERS[k], ms=5, lw=0, elinewidth=1.2, capsize=0,
                    label=f"{lab} ({'0.2' if S == 'M200c_0.2' else '0.5'} dex): measured")
        pure, second, model, beta, alpha, resolved = JS[f"{f}__{S}__{lab}"]
        m = resolved.astype(bool) & np.isfinite(second)
        ax.plot(X_FIT[m], 100 * second[m], color=PAL[k], ls="-", lw=1.3)
        ax.plot(X_FIT[m], 100 * pure[m], color=PAL[k], ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("r / R200c (points: band centres)")
    ax.set_title(f"{FLAB[f]}")
    if a == 0:
        ax.set_ylabel("bin-mean normalised / own normalised - 1 [%]")
h_, l_ = axs[3].get_legend_handles_labels()
h_ += [plt.Line2D([], [], color="0.3", ls="-", lw=1.3), plt.Line2D([], [], color="0.3", ls=":", lw=1.2)]
l_ += ["prediction incl. residual mass trend\n(2nd order: Jensen + covariance + curvature)", "pure Jensen term only"]
axs[3].legend(h_, l_, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.suptitle("Stacking in physical units and scaling by the bin's mean-mass P200, R200c instead of each halo's own "
             "(paired-bootstrap 1 sigma)")
fig.tight_layout()
save(fig, "fig05_normalisation_smearing.png")

# ---------------------------------------------------------------- Fig 6: M200c vs M500c
lm2, lm5 = TAB["logm200c"], TAB["logm500c"]
lc = np.log10(TAB["concentration_200c"])
cal = (lm2 >= 13.5) & np.isfinite(lc)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axs[0]
sc_ = ax.scatter(lc[cal], (lm2 - lm5)[cal], c=lm2[cal], s=5, cmap="viridis", alpha=0.6, lw=0)
cb = fig.colorbar(sc_, ax=ax)
cb.set_label("log10 M200c [Msun]")
ax.set_xlabel("log10 c200c (SOAP)")
ax.set_ylabel("log10 M200c - log10 M500c")
ax.set_title("M500c/M200c is set by concentration (haloes >= 10^13.5)")
ax = axs[1]
comp = A["m500_vs_m200"]["composition"]
for k, (S, bl) in enumerate([("M500c_0.2", "0.25-0.5"), ("M500c_0.2", "0.5-1"), ("M500c_0.2", "1-1.5")]):
    xs, ys, es = [], [], []
    for e, sv in zip(comp[S], A["scheme_vs_reference_same_mass"]["P"][S]):
        v = sv["bands"][bl]
        if v and v["err"] is not None and e["N"] >= 30 and e["mean_logm200c"] >= 13.5:
            xs.append(e["dlog_c200c_vs_M200c_bins_at_same_mean_m200"])
            ys.append(100 * (v["ratio"] - 1))
            es.append(100 * v["err"])
    ax.errorbar(xs, ys, es, fmt=MARKERS[k], color=PAL[k], ms=5, lw=1, label=f"P, {bl} R200c")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("mean log c200c of the M500c bin - M200c bins at the same mean M200c")
ax.set_ylabel("M500c-binned P / reference at same M200c - 1 [%]")
ax.set_title("Binning in M500c selects more concentrated haloes")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.tight_layout()
save(fig, "fig06_m500_vs_m200.png")

# ---------------------------------------------------------------- Fig 7: precision vs N
prec = A["sample_size"]["precision"]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
bands_p = ["0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "2-3", "3-4"]
for a, f in enumerate(["P", "E"]):
    ax = axs[a]
    for k, bl in enumerate(bands_p):
        for lab, ls in (("14.1-14.3", "-"), ("14.5-14.7", "--")):
            Ns, ys = [], []
            for N, d in prec[lab].items():
                v = d[f][bl]
                if v and int(N) <= 200:
                    Ns.append(int(N))
                    ys.append(100 * v["sd_inf"])
            ax.plot(Ns, ys, color=PAL[k], marker=MARKERS[k], ms=4, ls=ls, lw=1.2,
                    label=f"{bl} R200c" if lab == "14.1-14.3" else None)
    for p in (1, 3, 5):
        ax.axhline(p, color="0.45", lw=0.8, ls=":")
        ax.text(205, p, f"{p}%", va="center", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("haloes per bin N")
    ax.set_title(f"{FLAB[f]}: solid 14.1-14.3, dashed 14.5-14.7")
    if a == 0:
        ax.set_ylabel("error of the band stack (SD, FPC-corrected) [%]")
axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.suptitle("Precision of an N-halo stack (400 random subsamples per N; scales as 1/sqrt(N))")
fig.tight_layout()
save(fig, "fig07_precision_vs_N.png")

# ---------------------------------------------------------------- Fig 8: N needed for 3% vs mass (analytic sigma1)
nn = A["N_needed_analytic"]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
mids = {lab: float(np.mean([float(x) for x in lab.split("-")])) for lab in nn}
for a, f in enumerate(["P", "E"]):
    ax = axs[a]
    shade_cal(ax)
    for k, bl in enumerate(bands_p):
        xs, ys = [], []
        for lab, d in nn.items():
            v = d[f][bl]
            if v:
                xs.append(mids[lab])
                ys.append(v["N_3pct"])
        ax.plot(xs, ys, color=PAL[k], marker=MARKERS[k], ms=4, lw=1.2, label=f"{bl} R200c")
    ax.step([mids[lab] for lab in nn], [nn[lab]["N"] for lab in nn], where="mid", color="0.15", lw=1.8,
            label="haloes in the sample")
    ax.set_yscale("log")
    ax.set_xlabel("log10 M200c [Msun] (0.2-dex bins)")
    ax.set_title(f"{FLAB[f]}: haloes needed for 3% (1 sigma)")
    if a == 0:
        ax.set_ylabel("N = (sigma_1 / 0.03)^2")
axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.tight_layout()
save(fig, "fig08_N_needed.png")

# ---------------------------------------------------------------- Fig 9: covariance stability
cov = A["sample_size"]["covariance"]
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
for k, lab in enumerate(cov):
    for f, ls in (("P", "-"), ("E", "--")):
        e = cov[lab][f]["byN"]
        Ns = [int(N) for N in e]
        axs[0].plot(Ns, [e[N]["mean_sd_offdiag_inf"] for N in e], color=PAL[k], marker=MARKERS[k], ms=4, ls=ls,
                    label=f"{lab} (p={cov[lab]['p_shells']})" if f == "P" else None)
        axs[1].plot(Ns, [e[N]["second_eigvec_cos_median"] for N in e], color=PAL[k], marker=MARKERS[k], ms=4, ls=ls)
        Nh = [int(N) for N in e if e[N]["precision_trace_ratio_sd"] is not None]
        axs[2].plot(Nh, [e[str(N)]["precision_trace_ratio_sd"] for N in Nh], color=PAL[k], marker=MARKERS[k], ms=4,
                    ls=ls)
for ax in axs:
    ax.axvline(30, color="0.4", lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("haloes N")
axs[0].set_ylabel("mean SD of a correlation coefficient")
axs[0].set_title("element noise of the shell correlation matrix")
axs[1].set_ylabel("|cos| between 2nd eigenvectors (N vs full)")
axs[1].set_title("stability of the inner-outer (2nd) mode")
axs[2].set_ylabel("SD of tr(H_N C)/p (Hartlap-corrected)")
axs[2].set_yscale("log")
axs[2].set_title("noise of the inverse covariance (GLS)")
axs[0].text(31, axs[0].get_ylim()[1] * 0.95, "Lee22: >29", fontsize=8, va="top", color="0.3")
axs[0].legend(loc="upper right", frameon=False, title="bin (solid P, dashed E)")
fig.suptitle("Covariance of ln F on Lee22's grid (strict-resolved shells) from N-halo subsamples")
fig.tight_layout()
save(fig, "fig09_covariance_stability.png")

# ---------------------------------------------------------------- Fig 10: observable scatter vs N per bin
osc = A["sample_size"]["observable_scatter"]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
Ns = [int(N) for N in osc if int(N) <= 100]
Nf = [100, 200, 300]
for j, (f, o, ol) in enumerate(obs):
    ax.plot(Ns, [osc[str(N)][f][o]["sd_pct"] for N in Ns], color=PAL[j], marker=MARKERS[j], ms=5, lw=1.3, label=ol)
    ax.plot(Nf, [osc[str(N)][f][o]["sd_pct"] for N in Nf], color=PAL[j], marker=MARKERS[j], ms=5, lw=1.0, ls="--",
            mfc="white")
amp = osc["25"]["P"]["tSZ_yy_total_ell1000"]["sd_pct"] * 5.0
nn_ = np.array([10, 300])
ax.plot(nn_, amp / np.sqrt(nn_), color="0.4", ls=":", lw=1, label=f"{amp:.1f}% / sqrt(N)")
ax.text(170, 2.4, "open markers: N close to the bin size\n(finite-population limited, lower limits)", fontsize=8,
        color="0.3", ha="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("haloes per calibrated 0.2-dex bin N (bins with fewer haloes kept whole)")
ax.set_ylabel("SD of the observable over realisations [%]")
ax.set_title("Observable error from a finite number of haloes per bin (400 realisations, first order)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
save(fig, "fig10_observable_scatter_vs_N.png")

# ---------------------------------------------------------------- Fig 11: top bins
top = A["top_bins"]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
centres = [np.sqrt(max(lo, 0.05) * hi) for lo, hi in [(0, .1), (.1, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2),
                                                       (2, 3), (3, 4)]]
for r, f in enumerate(["P", "E"]):
    for c, lab in enumerate(["15.1-15.3", "15.3-15.5"]):
        ax = axs[r, c]
        ax.axhline(0, color="0.3", lw=0.8)
        series = [("trend fit 14.3-15.1", top["trend"][f][lab]["trend_over_measured"]),
                  ("pooled >= 15.1 (27)", top["pooled"]["pool_15.1"][f]["vs_bins"][lab]),
                  ("pooled >= 14.9 (154)", top["pooled"]["pool_14.9"][f]["vs_bins"][lab])]
        for k, (sl, d) in enumerate(series):
            xs, ys, es = [], [], []
            for bl, xc in zip(BAND_LABELS, centres):
                v = d.get(bl)
                if v and v["err"] is not None:
                    xs.append(xc * (1 + 0.04 * (k - 1)))
                    ys.append(100 * (v["ratio"] - 1))
                    es.append(100 * v["err"])
            ax.errorbar(xs, ys, es, color=PAL[k], marker=MARKERS[k], ms=5, lw=1.2, capsize=0, label=sl)
        ax.set_xscale("log")
        ax.set_ylim(-60, 60)
        nb = top["individual"][lab]["N"]
        ax.set_title(f"{f}, bin {lab} ({nb} haloes): replacement / measured stack")
        if c == 0:
            ax.set_ylabel("ratio - 1 [%]")
        if r == 1:
            ax.set_xlabel("r / R200c (band centres)")
axs[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.tight_layout()
save(fig, "fig11_top_bins.png")

# ---------------------------------------------------------------- Fig 12: share of each observable per bin
shares = top["bin_shares"]
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(CAL_LABELS))
for j, (f, o, ol) in enumerate(obs):
    ax.plot(x, [100 * shares[f][lab][o] for lab in CAL_LABELS], color=PAL[j], marker=MARKERS[j], ms=5, lw=1.3,
            label=ol)
ax.set_xticks(x)
ax.set_xticklabels(CAL_LABELS, rotation=45, ha="right")
ax.set_xlabel("calibrated 0.2-dex M200c bin")
ax.set_ylabel("share of the calibrated-bin response [%]")
ax.set_title("Where each observable is calibrated (response to a uniform 1% change of one bin)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
save(fig, "fig12_bin_shares.png")

# ---------------------------------------------------------------- Fig 14: interpolant test (linear vs PCHIP in log M)
IT = A["interpolant_test"]
WID = ["M200c_0.1", "M200c_0.2", "M200c_0.3", "M200c_0.5"]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.4), sharey=True)
for a, (it, title) in enumerate((("linear", "ln F linear in log M200c between nodes"),
                                 ("pchip", "ln F monotone cubic (PCHIP) in log M200c"))):
    ax = axs[a]
    for i, S in enumerate(WID):
        for j, (f, o, ol) in enumerate(obs):
            v = IT[it][S][f]["vs_truth"][o]
            ax.errorbar(v["pct"], i + (j - 3) * 0.1, xerr=v["err_pct"], fmt=MARKERS[j], color=PAL[j], ms=5, lw=1,
                        label=ol if (i == 0 and a == 1) else None)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlim(-5, 2)
    ax.set_xlabel("continuous painting vs true bin means [%]")
    ax.set_title(title)
axs[0].set_yticks(range(len(WID)))
axs[0].set_yticklabels([SLAB2[S] for S in WID])
axs[0].invert_yaxis()
axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.suptitle("Bin width and the mass interpolant: every calibrated halo painted at its own M200c "
             "(paired-bootstrap 1 sigma)")
fig.tight_layout()
save(fig, "fig14_interpolant_test.png")
