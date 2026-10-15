#!/usr/bin/env python
"""Figures and report tables of Test F from the compact results of run_test_f.py (runs locally).

  python -B make_figures.py [--results F_observables/results] [--out F_observables/figures]
Reads only <results>/*.npz|json and <results>/tables/*.csv (never work/).  Writes PNG (150 dpi) + SVG
figures and report tables (CSV + Markdown) to --out.
Style: analysis/plotting.py (categorical order #2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948,
diverging blue-grey(#f0efec)-red with symmetric limits for signed differences, one y-axis per panel).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analysis import plotting as PT  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SEL_ELL = (100, 300, 1000, 3000, 8000)
PRIMARY_ELL = (100, 300, 1000, 3000)
Q_NAMES = ("q16", "q50", "q84", "q975", "q999")
Q_LABELS = {"q16": "Q16", "q50": "Q50", "q84": "Q84", "q975": "Q97.5", "q999": "Q99.9"}
FAMILIES = {"tSZ": [f"yytot_{e}" for e in PRIMARY_ELL], "kSZ": [f"ksz1h_{e}" for e in PRIMARY_ELL],
            "DM": ["dm_mean", "dm_var"] + [f"dm_{q}" for q in Q_NAMES], "tSZxDM": [f"ydmtot_{e}" for e in PRIMARY_ELL]}
FAMILY_LABEL = {"tSZ": "tSZ auto", "kSZ": "kSZ auto (1h)", "DM": "DM PDF", "tSZxDM": "tSZ x DM"}
BLUE = LinearSegmentedColormap.from_list("blue_ramp", ["#eef4fc", "#0d366b"])
BAND_LABELS = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4")


def plain_log_axis(ax, which="both"):
    """Plain decimal tick labels on log axes (no 2x10^1 notation)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axes = []
    if which in ("both", "x"):
        axes.append(ax.xaxis)
    if which in ("both", "y"):
        axes.append(ax.yaxis)
    for a in axes:
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(NullFormatter())


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def write_md(path, header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")


class Results:
    def __init__(self, root: Path):
        self.root = root
        self.tables = root / "tables"
        self.scores = read_csv(self.tables / "F_model_scores.csv")
        self.errors = read_csv(self.tables / "F_errors_long.csv")
        self.cv = read_csv(self.tables / "F5_cv_scores.csv")
        self.cv_long = read_csv(self.tables / "F5_cv_errors_long.csv")
        self.controls = read_csv(self.tables / "F_controls_scores.csv")
        self.imp_summary = json.loads((root / "importance_summary.json").read_text())
        self.imp = np.load(root / "importance_F.npz")
        self.spectra = np.load(root / "model_spectra.npz")
        self.models = np.load(root / "models_main.npz")
        self.prof = np.load(root / "profiles.npz")
        self.truth_summary = json.loads((root / "truth_summary.json").read_text())
        self.run = json.loads((root / "run_summary.json").read_text()) if (root / "run_summary.json").exists() else {}
        self.manifest = json.loads((root / "run_manifest.json").read_text()) if (root / "run_manifest.json").exists() else {}

    def score(self, model, prediction="deterministic"):
        for r in self.scores:
            if r["model"] == model and r["prediction"] == prediction:
                return r
        return None

    def error(self, model, observable, prediction="deterministic", key="rel_error"):
        for r in self.errors:
            if r["model"] == model and r["prediction"] == prediction and r["observable"] == observable:
                return fnum(r[key])
        return np.nan

    def grid(self, objective="a"):
        rows = [r for r in self.scores if r["model"].startswith(f"grid_{objective}_") and r["prediction"] == "deterministic"]
        rf = sorted({fnum(r["R_fit"]) for r in rows})
        rc = sorted({fnum(r["R_cut"]) for r in rows})
        return rows, rf, rc


# ============================================================================ F1
def fig_importance_bands(R: Results, out: Path):
    rows = ([f"yytot_{e}" for e in SEL_ELL] + [f"ksz1h_{e}" for e in SEL_ELL] + ["dm_mean", "dm_var"]
            + [f"dm_{q}" for q in Q_NAMES] + [f"ydmtot_P_{e}" for e in SEL_ELL] + [f"ydmtot_E_{e}" for e in SEL_ELL])
    rows = [r for r in rows if r in R.imp_summary]
    M = np.array([R.imp_summary[r]["bands"] for r in rows]) * 100
    cal = np.array([R.imp_summary[r]["calibrated_share"] for r in rows]) * 100
    lim = max(np.nanmax(np.abs(M)), 1e-9)
    fig, ax = plt.subplots(figsize=(9.6, 0.34 * len(rows) + 1.6), constrained_layout=True)
    im = ax.imshow(M, cmap=PT.DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color=PT.TEXT)
    ax.set_xticks(range(len(BAND_LABELS)))
    ax.set_xticklabels([f"{b}" for b in BAND_LABELS])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{pretty(r)}  [cal {c:.0f}%]" for r, c in zip(rows, cal)], fontsize=7.5)
    ax.set_xlabel("radial band [R200c]")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("normalised importance [%] (all haloes; sums to 100)")
    ax.set_title("F1: where each observable comes from (measured FLAMINGO truth, scatter included)")
    PT.save(fig, out / "F1_importance_bands")


MODEL_NAMES = {"flex_gnfw5_a": "gNFW (5 free) (a) 4/4", "flex_gnfw5_d": "gNFW (5 free) (d) 4/4",
               "flex_env_a": "B12/B16 + outer power law (a) 4/4", "flex_env_d": "B12/B16 + outer power law (d) 4/4",
               "flex_spline_a": "B12/B16 x spline (a) 4/4", "flex_spline_d": "B12/B16 x spline (d) 4/4",
               "tab_stack": "measured mean stack (ceiling)", "tab_pooled": "stack, partial pooling in mass",
               "trunc_flux": "2R fit, edge matched to Y/N_e(<4R)", "trunc_obs": "2R fit, edge matched to 1h(1000)/DM var",
               "direct_bin": "direct observable fit, per bin", "direct_trend": "direct observable fit, log-M trend",
               "grid_a_Rf2_Rc4": "B12/B16 (a) 2/4  [current]"}


def pretty_model(m: str) -> str:
    base, _, pred = m.partition("|")
    if base in MODEL_NAMES:
        name = MODEL_NAMES[base]
    elif base.startswith("grid_"):
        _, o, rf, rc = base.split("_")
        name = f"B12/B16 ({o}) {rf[2:]}/{rc[2:]}"
    elif base.startswith("selected_a_"):
        name = f"(a) config selected on training ({base[11:]})"
    else:
        name = base
    return name + (f" + {pred.replace('_', ' ')}" if pred else "")


def pretty(key: str) -> str:
    key = key.replace("ydmtot_P", "y x DM (P side)").replace("ydmtot_E", "y x DM (E side)").replace("ydmtot", "y x DM total")
    key = key.replace("yytot", "yy total").replace("ksz1h", "kSZ 1h").replace("dm_var", "DM variance")
    key = key.replace("dm_mean", "mean DM")
    for q, lab in Q_LABELS.items():
        key = key.replace(f"dm_{q}", f"DM {lab}")
    if "_" in key and key.split("_")[-1].isdigit():
        head, ell = key.rsplit("_", 1)
        key = f"{head}, ell {ell}"
    return key


def fig_importance_mass(R: Results, out: Path):
    data = read_csv(R.tables / "F1_importance_mass.csv")
    obs = [o for o in ([f"yytot_{e}" for e in SEL_ELL] + [f"ksz1h_{e}" for e in SEL_ELL] + ["dm_mean", "dm_var"]
                       + [f"dm_{q}" for q in Q_NAMES] + [f"ydmtot_P_{e}" for e in SEL_ELL])
           if any(r["observable"] == o for r in data)]
    bins = []
    for r in data:
        if r["mass_bin"] not in bins:
            bins.append(r["mass_bin"])
    M = np.zeros((len(obs), len(bins)))
    for r in data:
        if r["observable"] in obs:
            M[obs.index(r["observable"]), bins.index(r["mass_bin"])] = fnum(r["normalised_importance"]) * 100
    calib = {r["mass_bin"]: r["calibrated"] == "True" for r in data}
    lim = max(np.nanmax(np.abs(M)), 1e-9)
    fig, ax = plt.subplots(figsize=(11.5, 0.34 * len(obs) + 1.8), constrained_layout=True)
    im = ax.imshow(M, cmap=PT.DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 0.5:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=PT.TEXT)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b + ("*" if calib.get(b) else "") for b in bins], rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(obs)))
    ax.set_yticklabels([pretty(o) for o in obs], fontsize=7.5)
    ax.set_xlabel("log10 M200c [Msun] bin  (* calibrated: measured FLAMINGO profiles; others: published fallback)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("normalised importance [%]")
    ax.set_title("F1: which haloes carry each observable")
    PT.save(fig, out / "F1_importance_mass")


def fig_importance_profiles(R: Results, out: Path):
    edges = R.imp["shell_edges"]
    lo = np.maximum(edges[:-1], edges[1] / 1.6)
    x = np.sqrt(lo * edges[1:])
    dln = np.log(edges[1:] / lo)
    ell = R.imp["ell"]
    idx = [int(np.argmin(np.abs(ell - e))) for e in SEL_ELL]
    panels = [("tSZ auto: yy total", "imp__yytot", [(f"ell {e}", i) for e, i in zip(SEL_ELL, idx)]),
              ("kSZ auto: 1h", "imp__ksz1h", [(f"ell {e}", i) for e, i in zip(SEL_ELL, idx)]),
              ("DM: mean, variance, PDF quantiles", None, None),
              ("tSZ x DM (electron side)", "imp__ydmtot_E", [(f"ell {e}", i) for e, i in zip(SEL_ELL, idx)])]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), constrained_layout=True, sharex=True)
    for ax, (title, key, series) in zip(axes.ravel(), panels):
        if key is not None:
            M = R.imp[key]
            for k, (lab, i) in enumerate(series):
                y = M[..., i].sum(axis=0) / dln * 100
                ax.plot(x, y, "-", color=PT.CATEGORICAL[k], lw=1.6, label=lab)
        else:
            items = [("mean DM", "imp__dm_mean"), ("DM variance", "imp__dm_var")] + \
                [(f"DM {Q_LABELS[q]}", f"imp__dm_{q}") for q in Q_NAMES]
            for k, (lab, kk) in enumerate(items):
                if kk in R.imp.files:
                    ax.plot(x, R.imp[kk].sum(axis=0) / dln * 100, "-", color=PT.CATEGORICAL[k], lw=1.6, label=lab)
        ax.axhline(0, color=PT.TEXT_2, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel("importance per unit ln r [%]")
        ax.legend(fontsize=7.5, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("r / R200c")
    fig.suptitle("F1: radial importance profiles at the measured truth (all mass bins; area under each curve = 100%)",
                 fontsize=10)
    PT.save(fig, out / "F1_importance_profiles")


# ============================================================================ F2
def fig_grid(R: Results, out: Path, objective="a", suffix=""):
    rows, rf, rc = R.grid(objective)
    if not rows:
        return None
    fams = list(FAMILIES)
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.06)
    best = {}
    for ax, fam in zip(axes, fams):
        M = np.full((len(rf), len(rc)), np.nan)
        for r in rows:
            M[rf.index(fnum(r["R_fit"])), rc.index(fnum(r["R_cut"]))] = fnum(r[f"rms_{fam}{suffix}"]) * 100
        norm = LogNorm(vmin=max(np.nanmin(M), 0.05), vmax=np.nanmax(M))
        im = ax.imshow(M, cmap=BLUE, aspect="auto", origin="lower", norm=norm)
        i, j = np.unravel_index(np.nanargmin(M), M.shape)
        best[fam] = (rf[i], rc[j], float(M[i, j]))
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                v = M[a, b]
                ax.text(b, a, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                        color="#ffffff" if norm(v) > 0.55 else PT.TEXT)
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=PT.CATEGORICAL[1], lw=2.0))
        ax.set_xticks(range(len(rc)))
        ax.set_xticklabels([f"{v:g}" for v in rc])
        ax.set_yticks(range(len(rf)))
        ax.set_yticklabels([f"{v:g}" for v in rf])
        ax.set_xlabel("R_cut [R200c]")
        ax.set_ylabel("R_fit [R200c]")
        ax.grid(False)
        ax.set_title(f"{FAMILY_LABEL[fam]}: RMS |dO/O| [%]", fontsize=9)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
    what = "vs the deterministic truth mean" if suffix else "vs the TRUTH (scatter included)"
    fig.suptitle(f"F2: objective ({objective}) fits, hard cut; observable error {what}; box = best", fontsize=10)
    PT.save(fig, out / f"F2_grid_{objective}{suffix or ''}")
    return best


def pareto(points):
    pts = np.asarray(points)
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        dominated = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        keep[i] = not dominated.any()
    return keep


def fig_pareto(R: Results, out: Path, objective="a"):
    rows, rf, rc = R.grid(objective)
    if not rows:
        return []
    fams = list(FAMILIES)
    P = np.array([[fnum(r[f"rms_{f}"]) * 100 for f in fams] for r in rows])
    keep = pareto(P)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    marks = ("o", "s", "^", "D", "v", "P", "X", "*")
    for ax, (a, b) in zip(axes, ((0, 2), (1, 3))):
        for k, cut in enumerate(rc):
            sel = [i for i, r in enumerate(rows) if fnum(r["R_cut"]) == cut]
            ax.scatter(P[sel, a], P[sel, b], s=26, color=PT.CATEGORICAL[k % 8], marker=marks[k % 8],
                       label=f"R_cut {cut:g}", edgecolor="none", alpha=0.9)
        front = np.flatnonzero(keep)
        ax.scatter(P[front, a], P[front, b], s=90, facecolor="none", edgecolor=PT.TEXT, lw=1.0, label="Pareto front (4 families)")
        offsets = {1.0: (5, -9), 2.0: (5, 4), 4.0: (6, -3)}
        for i, r in enumerate(rows):
            if fnum(r["R_cut"]) == max(rc) and fnum(r["R_fit"]) in offsets:
                ax.annotate(f"R_fit {fnum(r['R_fit']):g}", (P[i, a], P[i, b]), fontsize=7, xytext=offsets[fnum(r["R_fit"])],
                            textcoords="offset points", color=PT.TEXT_2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_axis(ax)
        ax.set_xlabel(f"{FAMILY_LABEL[fams[a]]} RMS |dO/O| [%]")
        ax.set_ylabel(f"{FAMILY_LABEL[fams[b]]} RMS |dO/O| [%]")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"F2: (R_fit, R_cut) trade-offs, objective ({objective}), errors vs the truth; R_cut = 4 points with "
                 "R_fit = 1, 2 (current), 4 labelled; circled = Pareto front over the four families", fontsize=10)
    PT.save(fig, out / f"F2_pareto_{objective}")
    return [(fnum(rows[i]["R_fit"]), fnum(rows[i]["R_cut"]), *P[i]) for i in np.flatnonzero(keep)]


# ============================================================================ F3
OBJ_LABEL = {"a": "(a) current diagonal", "c": "(c) GLS + shrinkage + sigma_d", "d": "(d) importance-diagonal",
             "e": "(e) observable metric K", "f": "(f) hybrid C^-1 + lambda K"}


def fig_objectives(R: Results, out: Path):
    configs = [(2.0, 4.0), (3.0, 4.0), (4.0, 4.0)]
    metrics = [("P_rms_inner_lt_1R", "pressure RMS <1R [%]"), ("P_rms_1_to_2R", "pressure RMS 1-2R [%]"),
               ("P_rms_outer_2_to_4R", "pressure RMS 2-4R [%]"), ("E_rms_inner_lt_1R", "electrons RMS <1R [%]"),
               ("E_rms_1_to_2R", "electrons RMS 1-2R [%]"), ("E_rms_outer_2_to_4R", "electrons RMS 2-4R [%]")] + \
        [(f"rms_{f}", f"{FAMILY_LABEL[f]} |dO/O| [%]") for f in FAMILIES]
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.6), constrained_layout=True)
    table = []
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for k, o in enumerate("acdef"):
            ys, xs = [], []
            for i, (rf, rc) in enumerate(configs):
                r = R.score(f"grid_{o}_Rf{rf:g}_Rc{rc:g}")
                if r is None:
                    continue
                v = fnum(r[key]) * (100 if key.startswith("rms_") else 1)
                xs.append(i + (k - 2) * 0.09)
                ys.append(v)
            ax.plot(xs, ys, "-o", color=PT.CATEGORICAL[k], ms=4.5, lw=1.0, label=OBJ_LABEL[o])
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([f"{a:g}/{b:g}" for a, b in configs])
        ax.set_xlabel("R_fit / R_cut [R200c]")
        ax.set_title(title, fontsize=8.8)
        ax.set_yscale("log")
        plain_log_axis(ax, "y")
    axes[0, 0].legend(fontsize=7, loc="best")
    for o in "acdef":
        for rf, rc in configs:
            r = R.score(f"grid_{o}_Rf{rf:g}_Rc{rc:g}")
            if r is not None:
                table.append([OBJ_LABEL[o], f"{rf:g}/{rc:g}"] + [f"{fnum(r[k]):.2f}" for k, _ in metrics[:6]]
                             + [f"{fnum(r[k]) * 100:.2f}" for k, _ in metrics[6:]])
    fig.suptitle("F3: can upweighting the important shells extend the fit? (profile RMS on resolved shells; observable "
                 "errors vs the truth)", fontsize=10)
    PT.save(fig, out / "F3_objectives")
    write_md(out / "F3_objectives.md", ["objective", "R_fit/R_cut"] + [t for _, t in metrics], table)


def fig_lambda(R: Results, out: Path):
    rows = [r for r in R.scores if r["model"].startswith("lam_") and r["prediction"] == "deterministic"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6), constrained_layout=True)
    for ax, fam in zip(axes, FAMILIES):
        for k, rf in enumerate((2.0, 3.0, 4.0)):
            pts = sorted((float(r["model"].split("_l")[-1]), fnum(r[f"rms_{fam}"]) * 100) for r in rows
                         if r["model"].startswith(f"lam_Rf{rf:g}_"))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=PT.CATEGORICAL[k], ms=4, label=f"R_fit {rf:g}")
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
        ax.set_xlabel("lambda (K trace-normalised to C^-1)")
        ax.set_ylabel("RMS |dO/O| [%]")
        ax.set_title(FAMILY_LABEL[fam], fontsize=9)
        ax.legend(fontsize=7.5)
    fig.suptitle("F3: hybrid objective r^T C^-1 r + lambda r^T K r, R_cut = 4", fontsize=10)
    PT.save(fig, out / "F3_lambda_scan")


# ============================================================================ F4 / F6
def method_list(R: Results):
    models = []
    for r in R.scores:
        m = r["model"]
        if r["prediction"] != "deterministic" or m in models:
            continue
        if m.startswith(("flex_", "tab_", "trunc_", "direct_")) or m in (
                "grid_a_Rf2_Rc4", "grid_a_Rf4_Rc4", "grid_c_Rf2_Rc4", "grid_d_Rf3_Rc4", "grid_e_Rf3_Rc4", "grid_f_Rf3_Rc4",
                "grid_a_Rf3_Rc4", "grid_c_Rf4_Rc4", "grid_d_Rf4_Rc4", "grid_f_Rf4_Rc4"):
            models.append(m)
    return models


def fig_methods(R: Results, out: Path, prediction="deterministic"):
    models = [m for m in method_list(R) if R.score(m, prediction) is not None]
    obs = sum((FAMILIES[f] for f in FAMILIES), [])
    M = np.array([[R.error(m, o, prediction) * 100 for o in obs] for m in models])
    if not M.size:
        return
    lim = 5.0
    fig, ax = plt.subplots(figsize=(16.0, 0.36 * len(models) + 2.2), constrained_layout=True)
    im = ax.imshow(np.clip(M, -lim, lim), cmap=PT.DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.2, color=PT.TEXT)
    ax.set_xticks(range(len(obs)))
    ax.set_xticklabels([pretty(o) for o in obs], rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([pretty_model(m) for m in models], fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label(f"(model - truth)/truth [%], {prediction} prediction (colour clipped at +-{lim:.0f}%; values printed)")
    ax.set_title(f"F4/F6: method x observable error against the truth ({prediction})")
    if prediction != "deterministic":
        fig.text(0.01, -0.01, "Grey cells: the DM PDF of this scatter variant was computed only for the B12/B16 (a) 2/4 and "
                 "4/4, (c) 2/4, stack and B12/B16 + outer power law (a) models (9-node lognormal amplitude mixture).",
                 fontsize=8, color=PT.TEXT_2)
    PT.save(fig, out / f"F4_methods_{prediction}")
    rows = []
    for m in models:
        r = R.score(m, prediction)
        rows.append([m] + [f"{fnum(r[f'rms_{f}']) * 100:.2f}" for f in FAMILIES]
                    + [f"{fnum(r[f'rms_{f}_vs_mean']) * 100:.2f}" for f in FAMILIES])
    write_md(out / f"F4_methods_{prediction}.md", ["model"] + [f"{f} vs truth [%]" for f in FAMILIES]
             + [f"{f} vs truth mean [%]" for f in FAMILIES], rows)


def fig_cv(R: Results, out: Path):
    if not R.cv:
        return
    fams = list(FAMILIES)
    cvm = [r for r in R.cv if r.get("kind") in ("fixed method", "training-selected (a) config")
           and not r["model"].startswith("grid_a_") or r["model"] in ("grid_a_Rf2_Rc4", "grid_a_Rf4_Rc4")]
    names = [r["model"] for r in cvm]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 0.32 * len(names) + 2.0), constrained_layout=True, sharey=True)
    for ax, fam in zip(axes, fams):
        cv_vals = [fnum(r[f"rms_{fam}"]) * 100 for r in cvm]
        ins = []
        for n in names:
            s = R.score(n) if not n.startswith("selected_") else None
            ins.append(fnum(s[f"rms_{fam}"]) * 100 if s else np.nan)
        y = np.arange(len(names))
        ax.scatter(ins, y, s=22, color=PT.CATEGORICAL[0], label="in-sample (full sample)", zorder=3)
        ax.scatter(cv_vals, y, s=22, color=PT.CATEGORICAL[1], marker="D", label="out-of-fold (spatial CV)", zorder=3)
        for i in range(len(names)):
            if np.isfinite(ins[i]) and np.isfinite(cv_vals[i]):
                ax.plot([ins[i], cv_vals[i]], [i, i], color=PT.GRID, lw=1.0, zorder=1)
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
        ax.set_xlabel("RMS |dO/O| [%]")
        ax.set_title(FAMILY_LABEL[fam], fontsize=9)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([pretty_model(n) for n in names], fontsize=7.5)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("F5: spatial-block cross-validation (out-of-fold predictions vs the full truth)", fontsize=10)
    PT.save(fig, out / "F5_cv")
    rows = [[r["model"], r.get("kind", "")] + [f"{fnum(r[f'rms_{f}']) * 100:.2f}" for f in fams] for r in R.cv]
    write_md(out / "F5_cv.md", ["model", "kind"] + [f"{f} CV [%]" for f in fams], rows)


def fig_spectra(R: Results, out: Path, models=None):
    s = R.spectra
    names = list(s["spectra_names"])
    ell = s["ell"]
    models = models or ["grid_a_Rf2_Rc4", "grid_a_Rf4_Rc4", "flex_env_d", "flex_gnfw5_d", "direct_bin", "tab_stack"]
    models = [m for m in models if np.any(s["models"] == m)]
    panels = [("yytot", "tSZ auto (yy total)"), ("ksz1h", "kSZ 1h"), ("dmcltot", "DM-field C_ell total"),
              ("ydmtot", "tSZ x DM total")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True, sharex=True)
    handles = []
    for ax, (key, title) in zip(axes.ravel(), panels):
        k = names.index(key)
        t = s["truth"][k]
        for i, m in enumerate(models):
            for pred, ls in (("deterministic", "-"), ("lognormal_1R", "--")):
                sel = np.flatnonzero((s["models"] == m) & (s["predictions"] == pred))
                if len(sel):
                    line, = ax.plot(ell, (s["values"][sel[0], k] / t - 1) * 100, ls, color=PT.CATEGORICAL[i % 8], lw=1.4,
                                    label=pretty_model(m) if pred == "deterministic" else None)
                    if key == "yytot" and pred == "deterministic":
                        handles.append(line)
        line, = ax.plot(ell, (s["truth_mean"][k] / t - 1) * 100, ":", color=PT.TEXT, lw=1.6,
                        label="truth mean (no halo-to-halo scatter)")
        if key == "yytot":
            handles.append(line)
        ax.axhline(0, color=PT.TEXT_2, lw=0.8)
        ax.axvspan(8000, ell.max(), color=PT.GRID, alpha=0.35, lw=0)
        ax.set_ylim(*{"yytot": (-11, 6), "ksz1h": (-10, 5), "dmcltot": (-10, 6), "ydmtot": (-22, 8)}[key])
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("model / truth - 1 [%]")
    for ax in axes[-1]:
        ax.set_xlabel("multipole ell")
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color=PT.TEXT_2, ls="-", label="deterministic (one profile per bin)"),
                Line2D([], [], color=PT.TEXT_2, ls="--", label="+ lognormal amplitude scatter (sigma from Y, N_e(<1R))")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Spectra relative to the truth (measured per-halo profiles to 4 R200c, scatter included); shaded: "
                 "ell > 8000 (unresolved-core dominated; the gNFW cusp leaves the frame there)", fontsize=10)
    PT.save(fig, out / "F6_spectra_ratio")


def fig_profiles(R: Results, out: Path, models=None):
    labels = list(R.models["labels"])
    edges = R.prof["f_edges"]
    lo = np.maximum(edges[:-1], edges[1] / 1.6)
    x = np.sqrt(lo * edges[1:])
    models = [m for m in (models or ["grid_a_Rf2_Rc4", "grid_a_Rf4_Rc4", "flex_env_d", "flex_gnfw5_d", "direct_bin"])
              if m in labels]
    n_cal = R.models["P"].shape[1]
    cal = list(range(5, 15)) if n_cal == 10 else [10, 11][:n_cal]          # full: 13.5-15.4; smoke: 14.5-14.9
    picks = [c for c in range(len(cal)) if cal[c] in (5, 10, 12)] or list(range(min(2, len(cal))))
    fig, axes = plt.subplots(2, len(picks), figsize=(4.6 * len(picks), 6.8), constrained_layout=True, squeeze=False)
    for col, c in enumerate(picks):
        b = cal[c]
        base = R.prof["base_mask"][b]
        for row, (field, key) in enumerate((("pressure", "P"), ("electrons", "E"))):
            ax = axes[row, col]
            truth = R.prof[f"mean{key}_main"][b]
            for i, m in enumerate(models):
                model = R.models[key][labels.index(m)][c][:len(truth)]
                y = np.where(truth > 0, model / np.where(truth > 0, truth, 1) - 1, np.nan) * 100
                ax.plot(x, np.where(base, y, np.nan), "-", color=PT.CATEGORICAL[i], lw=1.4, label=pretty_model(m))
                ax.plot(x, np.where(~base, y, np.nan), ":", color=PT.CATEGORICAL[i], lw=1.0)
            ax.axhline(0, color=PT.TEXT_2, lw=0.8)
            ax.set_xscale("log")
            plain_log_axis(ax, "x")
            ax.set_ylim(-60, 60)
            ax.set_title(f"{field}, log M {b * 0.2 + 12.5:.1f}-{b * 0.2 + 12.7:.1f}", fontsize=9)
            ax.set_ylabel("model / measured mean - 1 [%]")
            if row == 1:
                ax.set_xlabel("r / R200c")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle("Profile residuals against the measured inclusion-weighted mean (solid: resolved shells, dotted: "
                 "unresolved core, excluded from every fit)", fontsize=10)
    PT.save(fig, out / "F6_profile_residuals")


def best_tables(R: Results, out: Path):
    """Best (R_fit, R_cut) per observable and per family for every objective grid; Pareto sets."""
    rows = []
    for o in "acdef":
        grid, rf, rc = R.grid(o)
        if not grid:
            continue
        for fam in FAMILIES:
            for key in ("", "_vs_mean"):
                vals = [(fnum(r[f"rms_{fam}{key}"]), r) for r in grid]
                v, r = min(vals, key=lambda t: t[0])
                rows.append([o, fam, "truth" if not key else "truth mean", f"{fnum(r['R_fit']):g}", f"{fnum(r['R_cut']):g}",
                             f"{v * 100:.2f}"])
    write_md(out / "F2_best_per_family.md", ["objective", "family", "against", "R_fit", "R_cut", "RMS |dO/O| [%]"], rows)
    per_obs = []
    grid, rf, rc = R.grid("a")
    names = [r["model"] for r in grid]
    for fam, obs in FAMILIES.items():
        for o in obs:
            vals = [(abs(R.error(m, o)), m) for m in names]
            v, m = min(vals)
            per_obs.append([o, m.replace("grid_a_", ""), f"{v * 100:.2f}"])
    write_md(out / "F2_best_per_observable_a.md", ["observable", "best config (objective a)", "|dO/O| [%]"], per_obs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.results).resolve()
    out = Path(args.out).resolve() if args.out else HERE / ("figures" if root.name == "results" else f"figures_{root.name}")
    out.mkdir(parents=True, exist_ok=True)
    PT.apply_style()
    R = Results(root)
    fig_importance_bands(R, out)
    fig_importance_mass(R, out)
    fig_importance_profiles(R, out)
    bests = {o: fig_grid(R, out, o) for o in "acdef"}
    bests_mean = {o: fig_grid(R, out, o, "_vs_mean") for o in "a"}
    front = fig_pareto(R, out, "a")
    fig_objectives(R, out)
    fig_lambda(R, out)
    fig_methods(R, out, "deterministic")
    fig_methods(R, out, "lognormal_1R")
    fig_cv(R, out)
    fig_spectra(R, out)
    fig_profiles(R, out)
    best_tables(R, out)
    (out / "summary.json").write_text(json.dumps({"best_grid": bests, "best_grid_vs_mean": bests_mean,
                                                  "pareto_a": front}, indent=1, default=float) + "\n")
    print(f"figures and tables in {out}")


if __name__ == "__main__":
    main()
