"""Test G report figures: the fitted profiles against the stacked profiles (1D), per calibrated mass bin.

Models are evaluated with Test G's own code (g_models.Model.nodes, g_selection.SelModel.evaluate) at the fitted
parameters stored in results/fits.npz and results/selection.npz; the shell averages reproduce the stored fit
shells (results/fits.npz S35) to machine precision (checked below). Stacks: inclusion-weighted means of the
per-halo profiles (Test F's profiles.npz), error of the mean = SD / sqrt(N_eff), halo band = weighted 16-84%.

  python -B make_report_figures.py          (run from anywhere; G_MODE=full is the default)
Outputs: report/figures/*.png (+ .pdf for the LaTeX report) and report/figure_numbers.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GDIR = HERE.parent
sys.path.insert(0, str(GDIR))

import g_common as GC  # noqa: E402
import g_models as GM  # noqa: E402
import g_physics as PH  # noqa: E402
import g_selection as GS  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
COL = {"P0": "#2a78d6", "E0": "#2a78d6", "P1": "#eb6834", "E1": "#eb6834", "P2m": "#b07a00", "E2gf": "#1baf7a",
       "BPL": "#4a3aa7"}
LABEL = {"P0": "P0: B12 refitted to 4R (30 par.)", "P1": "P1: B12 + A(x/2)$^{-s}$, Test F (50)",
         "P2m": "P2m: B12$\\,f_{\\rm tr}$ + 2-halo gas, T(M) (44)", "E0": "E0: B16 refitted to 4R (30)",
         "E1": "E1: B16 + A(x/2)$^{-s}$, Test F (50)", "E2gf": "E2gf: B16$\\,f_{\\rm tr}$ + linear 2-halo (41)",
         "BPL_P": "BPLall+O2: Lee22 gNFW, broken power laws (14)", "BPL_E": "BPLall+O2: Lee22 gNFW, broken power laws (12)"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": "#b9b8b0", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.titlesize": 8.5, "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.4,
                     "savefig.facecolor": SURFACE, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

pr = dict(np.load(GC.FBASE / "profiles.npz"))
prep = GC.load_prepare()
fz = np.load(GC.RESULTS / "fits.npz")
keys = [str(k) for k in fz["keys"]]
pars = json.loads(str(fz["params_json"]))
sel_results = json.loads(str(np.load(GC.RESULTS / "selection.npz")["results_json"]))
bhalo = np.load(GC.B_HALOS)
conc = np.asarray(bhalo["concentration_200c"], float)[pr["halo_rows"]]
lh = PH.LinearHalo()
x35, wq35 = PH.quad_nodes(GC.F_EDGES)
XS = PH.shell_radius(GC.F_EDGES)                       # volume-weighted shell radius
CTX = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"]), "b": prep["bin_b"], "logm": prep["bin_logm"],
       "b_pivot": float(prep["b_pivot"])}
T35 = prep["T2h"][:, :GC.N_F]
NB = GC.N_CAL
numbers = {}


# ----------------------------------------------------------------------------- data
def stack(field, sel, c):
    b = GC.CAL_BINS[c]
    mean = pr[f"mean{field}_{sel}"][b]
    err = pr[f"sd{field}_{sel}"][b] / np.sqrt(pr[f"neff_{sel}"][b])
    rows = np.flatnonzero(pr["mass_bin"] == b)
    w = pr["weight"][rows]
    vals = pr[f"{field}_{sel}"][rows]
    lo, hi = np.full(GC.N_F, np.nan), np.full(GC.N_F, np.nan)
    for s in range(GC.N_F):
        v = vals[:, s]
        ok = np.isfinite(v)
        if ok.sum() < 5:
            continue
        o = np.argsort(v[ok])
        cw = np.cumsum(w[ok][o]) / w[ok].sum()
        lo[s], hi[s] = np.interp([0.16, 0.84], cw, v[ok][o])
    base = pr["base_mask"][b] & (mean > 0)
    return mean, err, lo, hi, base


def g_model(key, cut=None, paint=False):
    """Shell averages (bound, outer) [NB, 35] of a Test G fit; uncut by default."""
    if cut is None and not paint:
        f, m, t, o, rf = GC.parse_key(key)
        model = GM.get(m)
        pb, pg = np.array(pars[key]["pb"], float), np.array(pars[key]["pg"], float)
        bnd, out = model.nodes(pb, pg, CTX, x35, T35)
        Bs, Os = np.sum(bnd * wq35[None], -1), np.sum(out * wq35[None], -1)
        assert np.allclose(Bs + Os, fz["S35"][keys.index(key)], rtol=1e-12, atol=0), key
        return Bs, Os
    j = list(fz["r_cut"]).index(cut)
    i = keys.index(key)
    if paint:
        return fz["Bp"][i, j, :, :GC.N_F], fz["Op"][i, j, :, :GC.N_F]
    return fz["B"][i, j, :, :GC.N_F], fz["O"][i, j, :, :GC.N_F]


def sel_model(field, name="BPLall", outer="O2", r_fit=4.0, lik="floor"):
    rec = [r for r in sel_results if r["field"] == field and r["dset"] == "mass" and r["model"] == f"{name}+{outer}"
           and abs(float(r["R_fit"]) - r_fit) < 1e-9][0]
    theta = np.array(rec["fits"][lik]["theta"], float)
    cells = GS.Cells(pr, field, "mass", conc, lh, gls=False)
    spec = [s for s in GS.model_specs(field, "mass") if s[0] == name][0]
    model = GS.build_model(field, spec, outer, cells.n)
    inner, outer_s = model.evaluate(theta, cells, x35, wq35, GC.N_F, cells.T2h, {"ebar": CTX["ebar"], "kappa_P": CTX["kappa_P"]},
                                    jac=False, parts=True)
    lnS, _ = model.evaluate(theta, cells, x35, wq35, GC.N_F, cells.T2h, {"ebar": CTX["ebar"], "kappa_P": CTX["kappa_P"]},
                            jac=False)
    assert np.allclose(np.log(inner + outer_s), lnS)
    mask = cells.masks(r_fit)
    return inner, outer_s, mask, dict(zip(rec["names"], theta.tolist())), rec


# ----------------------------------------------------------------------------- figure 1/2: profiles per bin
def profile_figure(field, models, fname, title):
    fig = plt.figure(figsize=(7.3, 10.3))
    outer_gs = fig.add_gridspec(5, 2, hspace=0.42, wspace=0.24, left=0.095, right=0.985, top=0.935, bottom=0.05)
    curves = {}
    for name, key in models:
        if name == "BPL":
            inner, outer_s, mask, theta, rec = sel_model(field)
            curves[name] = inner + outer_s
            numbers[f"{field}_BPLall_O2_theta"] = theta
        else:
            Bs, Os = g_model(key)
            curves[name] = Bs + Os
    stats = {name: [] for name, _ in models}
    axes_pairs = []
    for c in range(NB):
        mean, err, lo, hi, base = stack(field, "main", c)
        fitm = base & GC.x_hi_le(GC.F_EDGES, 4.0)
        g = outer_gs[c // 2, c % 2].subgridspec(2, 1, height_ratios=[2.1, 1], hspace=0.05)
        ax, axr = fig.add_subplot(g[0]), fig.add_subplot(g[1])
        ok = mean > 0
        ax.errorbar(XS[fitm], mean[fitm], yerr=err[fitm], fmt="o", ms=2.6, color=INK, lw=0.8, label="stack, fitted shells", zorder=5)
        unf = ok & ~fitm
        ax.plot(XS[unf], mean[unf], "o", ms=2.6, mfc="white", mec=INK, mew=0.7, label="stack, unresolved (not fitted)", zorder=5)
        band = fitm & np.isfinite(lo) & (lo > 0)
        ax.fill_between(XS[band], lo[band], hi[band], color="#e7e6e0", lw=0, label="haloes, 16-84%", zorder=1)
        for name, key in models:
            S = curves[name][c]
            lab = LABEL[f"BPL_{field}"] if name == "BPL" else LABEL[name]
            ax.plot(XS, S, color=COL[name], lw=1.3, label=lab, zorder=4)
            ratio = S / mean
            axr.plot(XS[fitm], ratio[fitm], color=COL[name], lw=1.2)
            rr = np.log(ratio[fitm])
            stats[name].append({"bin": GC.CAL_LABELS[c], "rms_lt1": float(np.sqrt(np.mean(rr[XS[fitm] < 1] ** 2))),
                                "rms_1_2": float(np.sqrt(np.mean(rr[(XS[fitm] >= 1) & (XS[fitm] < 2)] ** 2))),
                                "rms_2_4": float(np.sqrt(np.mean(rr[XS[fitm] >= 2] ** 2)))})
        axr.fill_between(XS[fitm], 1 - err[fitm] / mean[fitm], 1 + err[fitm] / mean[fitm], color="#cfcec6", lw=0)
        axr.axhline(1, color=INK2, lw=0.7)
        for a in (ax, axr):
            a.set_xscale("log")
            a.set_xlim(0.03, 4.3)
            a.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
            a.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
            a.minorticks_off()
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        axr.set_yscale("log")
        axr.set_ylim(0.7, 1.43)
        axr.set_yticks([0.8, 1.0, 1.25])
        axr.set_yticklabels(["0.8", "1", "1.25"])
        axr.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for v in (0.9, 1.1):
            axr.axhline(v, color=MUTED, lw=0.5, ls=":")
        ax.set_title(f"log M$_{{200c}}$ {GC.CAL_LABELS[c].replace('-', '–')}  (N = {int(np.sum(pr['mass_bin'] == GC.CAL_BINS[c]))})", loc="left")
        if c % 2 == 0:
            ax.set_ylabel("P$_e$ / P$_{200}$" if field == "P" else "$\\mu$ n$_e$ / (f$_b \\rho_{\\rm crit}$)")
            axr.set_ylabel("ratio")
        if c >= NB - 2:
            axr.set_xlabel("r / R$_{200c}$")
        if c == 0:
            handles, labels = ax.get_legend_handles_labels()
        axes_pairs.append((ax, axr))
    fig.align_ylabels([a for pair in axes_pairs[::2] for a in pair])
    order = [labels.index(l) for l in labels if l.startswith("stack")] + \
            [labels.index(l) for l in labels if l.startswith("haloes")] + \
            [labels.index(l) for l in labels if not l.startswith(("stack", "haloes"))]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3, fontsize=7.0,
               bbox_to_anchor=(0.53, 0.997), handlelength=1.8, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=170)
    plt.close(fig)
    numbers[f"{field}_shell_rms_by_bin"] = stats


profile_figure("P", [("P0", "P|P0|main|a|4"), ("P1", "P|P1|main|a|4"), ("P2m", "P|P2m|main|a|4"), ("BPL", None)],
               "profiles_pressure",
               "All-gas stacks (recently AGN-heated gas removed); every model fitted to 4 R$_{200c}$ (objective a). "
               "Lower panels: model / stack; grey band = error of the stacked mean.")
profile_figure("E", [("E0", "E|E0|main|a|4"), ("E1", "E|E1|main|a|4"), ("E2gf", "E|E2gf|main|a|4"), ("BPL", None)],
               "profiles_electrons",
               "All-gas stacks (recently AGN-heated gas removed); every model fitted to 4 R$_{200c}$ (objective a). "
               "Lower panels: model / stack; grey band = error of the stacked mean.")


# ----------------------------------------------------------------------------- figure 3: components of P2m and E2gf
def components_fine(key, c, x):
    f, m, t, o, rf = GC.parse_key(key)
    model = GM.get(m)
    pb, pg = np.array(pars[key]["pb"], float)[c:c + 1], np.array(pars[key]["pg"], float)
    b = GC.CAL_BINS[c]
    rows = np.flatnonzero(pr["mass_bin"] == b)
    T = PH.two_halo_template(lh, pr["m200c_msun"][rows], pr["r200c_mpc"][rows], pr["weight"][rows], x[None, :])
    ctx = dict(CTX, b=np.asarray(CTX["b"])[c:c + 1], logm=np.asarray(CTX["logm"])[c:c + 1])
    bound, outer = model.nodes(pb, pg, ctx, x[None, :], T[None])
    core, _ = GM.get("P0" if f == "P" else "E0").nodes(pb[:, :3], np.zeros(0), ctx, x[None, :], T[None])
    return core[0, 0], bound[0, 0], outer[0, 0], float(pb[0, 3])


SHOW = [0, 3, 5, 8]
xf = np.geomspace(0.05, 6.0, 500)
fig, axes = plt.subplots(2, len(SHOW), figsize=(7.3, 4.6), sharex=True)
comp_numbers = {}
for row, (field, key) in enumerate((("P", "P|P2m|main|a|4"), ("E", "E|E2gf|main|a|4"))):
    for k, c in enumerate(SHOW):
        ax = axes[row, k]
        core, bound, outer, xt = components_fine(key, c, xf)
        mean, err, lo, hi, base = stack(field, "main", c)
        r_m = float(prep["r200m_over_r200c"][c])
        ax.axvspan(1.0 * r_m, 1.5 * r_m, color="#efe9d6", lw=0, label="1.0-1.5 R$_{200m}$ (splashback band)")
        ok = mean > 0
        ax.plot(XS[ok], (mean * XS**2)[ok], "o", ms=2.4, color=INK, label="stack", zorder=5)
        ax.plot(xf, core * xf**2, color=COL["P0"], lw=1.0, ls=":", label="core alone (no transition)")
        ax.plot(xf, bound * xf**2, color=COL["P0"], lw=1.3, ls="--", label="bound = core $\\times f_{\\rm tr}$")
        ax.plot(xf, outer * xf**2, color=COL["P1"], lw=1.3, ls="-.", label="outer (2-halo gas)")
        ax.plot(xf, (bound + outer) * xf**2, color=COL["P2m" if field == "P" else "E2gf"], lw=1.6, label="total model")
        ax.axvline(xt, color=INK2, lw=0.8, ls=(0, (4, 2)))
        ax.text(xt * 0.93, 0.95, f"x$_t$ = {xt:.2f}", transform=ax.get_xaxis_transform(), fontsize=6.6, color=INK2,
                ha="right", va="top")
        ax.axvline(4.0, color=MUTED, lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 0.3, 1, 4])
        ax.set_xticklabels(["0.1", "0.3", "1", "4"])
        ax.minorticks_off()
        top = np.nanmax((mean * XS**2)[ok])
        ax.set_ylim(top * (3e-4 if field == "P" else 4e-3), top * 3)
        ax.set_title(f"{'P2m' if field == 'P' else 'E2gf'}, log M {GC.CAL_LABELS[c].replace('-', '–')}", loc="left",
                     fontsize=7.6)
        if k == 0:
            ax.set_ylabel("x$^2$ P / P$_{200}$" if field == "P" else "x$^2\\,\\mu$ n$_e$/(f$_b\\rho_{\\rm crit}$)")
        if row == 1:
            ax.set_xlabel("x = r / R$_{200c}$")
        # where the outer term takes over
        cross = xf[np.argmax(outer > bound)] if np.any(outer > bound) else np.nan
        comp_numbers[f"{field}|{GC.CAL_LABELS[c]}"] = {"x_t": xt, "x_outer_exceeds_bound": float(cross),
                                                       "R200m_over_R200c": r_m}
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"components.{ext}", dpi=170)
plt.close(fig)
numbers["components"] = comp_numbers


# ----------------------------------------------------------------------------- figure 4: painting target ratios
fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9), sharey=True)
ramp = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, NB))
paint_numbers = {}
for ax, (field, key) in zip(axes, (("P", "P|P2m|main|a|4"), ("E", "E|E2gf|main|a|4"))):
    Bp, Op = g_model(key, cut=4.0, paint=True)
    rms = []
    for c in range(NB):
        mean, err, lo, hi, base = stack(field, "a2ii", c)
        m = base & (mean > 0)
        ratio = (Bp[c] + Op[c]) / mean
        ax.plot(XS[m], ratio[m], color=ramp[c], lw=1.1, label=GC.CAL_LABELS[c].replace("-", "–"))
        rr = np.log(ratio[m])
        rms.append({"bin": GC.CAL_LABELS[c], "rms_lt1": float(np.sqrt(np.mean(rr[XS[m] < 1] ** 2))),
                    "rms_2_4": float(np.sqrt(np.mean(rr[XS[m] >= 2] ** 2)))})
    paint_numbers[field] = rms
    ax.axhline(1, color=INK2, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 1.6)
    ax.set_yticks([0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5])
    ax.set_yticklabels(["0.7", "0.8", "0.9", "1", "1.1", "1.25", "1.5"])
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("r / R$_{200c}$")
    ax.set_title("pressure: P2m $\\times\\,p_{\\rm keep}$ / A2-ii stack" if field == "P"
                 else "electrons: E2gf $\\times\\,p_{\\rm keep}$ / A2-ii stack", loc="left")
axes[0].set_ylabel("painted model / A2-ii stack")
axes[1].legend(fontsize=6.5, ncol=2, loc="lower left", title="log M$_{200c}$", title_fontsize=6.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"painting_ratio.{ext}", dpi=170)
plt.close(fig)
numbers["painting_ratio_rms"] = paint_numbers

(HERE / "figure_numbers.json").write_text(json.dumps(GC.plain(numbers), indent=1))
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
