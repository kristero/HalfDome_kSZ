"""Figures for the 'Appendix: XGPaint edits' section (2026-10-07).

Reads only saved data:
  - this directory: fork_vs_upstream_29ec48a.csv, shape_functions_m1e14_z0p5.csv (measure_xgpaint_edits.jl)
  - frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit/{v1c_itp_vs_direct_grid.csv, v4b_cl_*.npz}
    (same data and selection as that directory's v6_plot.py, split into two separate figures)
  - frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit/painted_profile_vs_independent_los.csv
Writes PNGs into fig/6_xgpaint/.
"""
import csv
import datetime as dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(REPORT, "fig", "6_xgpaint")
REPO = "/home/cbllover/HalfDome"
VA = f"{REPO}/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
AA = f"{REPO}/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit"
os.makedirs(FIG, exist_ok=True)

# reference categorical palette (dataviz skill, light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
RED = "#e34948"          # reserved here for "broken state" marks only
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 150, "figure.facecolor": "white",
})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ 1. fork lineage (ordinal: one column per commit)
def fig_lineage():
    S = 1.55  # lane spacing
    lane_y = {"up": 3 * S, "ad": 2 * S, "main": 1 * S, "cl": 0.0}
    lane_name = {3 * S: "upstream\nWebSky-CITA\nmain", 2 * S: "adrianbayer\n(upstream/\nadrian)", 1 * S: "fork\nmain",
                 0.0: "fork\ncluster"}
    col = {3 * S: INK2, 2 * S: ORANGE, 1 * S: BLUE, 0.0: AQUA}
    commits = [  # hash, date, lane, note (wrapped to ~14 characters)
        ("29ec48a", "2025-08-30", "up", "last upstream\ncommit used"),
        ("f16a7f3", "2026-02-09", "main", "PowerLawParam;\n15 kwargs"),
        ("9f37b1e", "2026-02-27", "ad", "tau: f_b,\nphysical col.,\nexponent sign"),
        ("077b849", "2026-02-27", "ad", "comment only"),
        ("b459728", "2026-03-02", "main", "f_b in ne2d\n(superseded)"),
        ("b924ab1", "2026-03-02", "main", "merge Adrian;\nB12 sign bug"),
        ("e67a8a1", "2026-04-30", "main", "B12 sign fix,\n0.9, FRB,\nslices"),
        ("cbf5cbd", "2026-07-01", "cl", "coarse grids,\nrtol 1e-3"),
        ("71646e2", "2026-07-13", "main", "M2_TO_PC_CM3\nvia Unitful"),
        ("ce1d239", "2026-07-13", "main", "pad 128,\nrtol 1e-12,\ntree 3b57cac"),
        ("5dd0b57", "2026-07-13", "cl", "merge;\nunparseable"),
        ("07fc686", "2026-09-24", "cl", "parses;\ndrops N_logθ"),
        ("35f79ad", "2026-09-24", "cl", "restores\nN_logθ; tree\n= ce1d239"),
        ("1660e08", "2026-09-25", "cl", "tests only"),
    ]
    idx = {h: i for i, (h, *_rest) in enumerate(commits)}
    n = len(commits)
    fig, ax = plt.subplots(figsize=(18, 7.6))
    for y, name in lane_name.items():
        ax.hlines(y, -0.6, n - 0.4, color=col[y], lw=1.0, alpha=0.30)
        ax.text(-0.85, y, name, color=INK, fontsize=10.5, ha="right", va="center", weight="bold")
    edges = [("29ec48a", "f16a7f3"), ("29ec48a", "9f37b1e"), ("9f37b1e", "077b849"), ("077b849", "b924ab1"),
             ("f16a7f3", "b459728"), ("b459728", "b924ab1"), ("b924ab1", "e67a8a1"), ("e67a8a1", "71646e2"),
             ("71646e2", "ce1d239"), ("f16a7f3", "cbf5cbd"), ("cbf5cbd", "5dd0b57"), ("ce1d239", "5dd0b57"),
             ("5dd0b57", "07fc686"), ("07fc686", "35f79ad"), ("35f79ad", "1660e08")]
    for a, b in edges:
        ia, ib = idx[a], idx[b]
        ya, yb = lane_y[commits[ia][2]], lane_y[commits[ib][2]]
        same = ya == yb
        ax.plot([ia, ib], [ya, yb], color=col[yb] if same else INK2, lw=2.2 if same else 1.1,
                ls="-" if same else (0, (3, 2)), alpha=0.9 if same else 0.7, zorder=1)
    # broken states (red reserved for these two windows)
    ax.fill_between([idx["b924ab1"], idx["e67a8a1"]], S - 0.12, S + 0.12, color=RED, alpha=0.35, lw=0, zorder=2)
    ax.fill_between([idx["5dd0b57"], idx["07fc686"]], -0.12, 0.12, color=RED, alpha=0.35, lw=0, zorder=2)
    for i, (h, date, lane, note) in enumerate(commits):
        y = lane_y[lane]
        ax.plot([i], [y], "o", ms=12, color=col[y], mec="white", mew=2, zorder=5)
        ax.text(i, y + 0.20, f"{h}\n{date}", fontsize=8.8, color=INK, ha="center", va="bottom", family="monospace")
        ax.text(i, y - 0.22, note, fontsize=8.6, color=INK2, ha="center", va="top")
    ax.text(n - 0.4, -1.25, "red band = broken state of that branch: b924ab1..e67a8a1 B12 exponent (1+x)^(+β); "
            "5dd0b57..07fc686 source does not parse\n"
            "tag validated-tsz256-20260922 points at 35f79ad (tree 3b57cac, byte-identical to ce1d239); "
            "pinned by the tSZ 64k bundle tsz64k-flat-prior-v1",
            fontsize=9.2, color=INK, ha="right", va="top")
    ax.set_xlim(-2.4, n - 0.3)
    ax.set_ylim(-1.9, 3 * S + 0.75)
    ax.axis("off")
    ax.set_title("kristero/XGPaint.jl: every fork commit on top of WebSky-CITA 29ec48a (one column per commit, in commit order)",
                 loc="left", fontsize=13, color=INK)
    save(fig, "01_fork_lineage.png")


# ------------------------------------------------------------------ 2. tau fork / upstream
def fig_tau_ratio():
    rows = list(csv.DictReader(open(os.path.join(HERE, "fork_vs_upstream_29ec48a.csv"))))
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    zz = np.linspace(0.0, 3.1, 300)
    fb = 0.049 / 0.31
    ax.plot(zz, 0.9 * fb * (1 + zz) ** 2, color=INK2, lw=1.5, ls="--", label=r"0.9 f$_b$ (1+z)$^2$ (from the code diff)")
    for M, c, mk in (("1.000e+13", BLUE, "o"), ("1.000e+14", ORANGE, "s"), ("1.000e+15", AQUA, "^")):
        r = [x for x in rows if x["M_msun"] == M]
        z = np.array([float(x["z"]) for x in r]); t = np.array([float(x["tau_ratio"]) for x in r])
        ax.plot(z, t, mk, ms=8, color=c, mec="white", mew=1.2, ls="none",
                label=fr"measured, $M_{{200c}}$ = 10$^{{{int(round(np.log10(float(M))))}}}\,M_\odot$, x = 0.1-3.5")
    ax.axhline(1, color=INK, lw=0.8)
    zc = np.sqrt(1 / (0.9 * fb)) - 1
    ax.annotate(f"equal at z = {zc:.2f}", xy=(zc, 1), xytext=(zc + 0.25, 0.55), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("halo redshift z")
    ax.set_ylabel(r"$\tau$(fork 35f79ad) / $\tau$(upstream 29ec48a)")
    ax.set_title("Battaglia16 optical depth: fork over upstream, same halo and angle", loc="left", fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left")
    save(fig, "02_tau_fork_over_upstream.png")


# ------------------------------------------------------------------ 3. B12 sign bug (3-D shape)
def fig_sign_bug():
    r = list(csv.DictReader(open(os.path.join(HERE, "shape_functions_m1e14_z0p5.csv"))))
    x = np.array([float(a["x"]) for a in r])
    up = np.array([float(a["b12_upstream"]) for a in r]); fk = np.array([float(a["b12_fork"]) for a in r])
    bug = np.array([float(a["b12_b924ab1"]) for a in r])
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.plot(x, up, color=BLUE, lw=4, alpha=0.45, label="upstream 29ec48a: (β-γ)/α with β = γ - αβ$_{B}$")
    ax.plot(x, fk, color=BLUE, lw=1.6, ls="--", label="fork e67a8a1+: -(β+γ)/α with β = αβ$_{B}$ - γ")
    ax.plot(x, bug, color=RED, lw=2, label="b924ab1 to e67a8a1: -(β+γ)/α with β = -γ - αβ$_{B}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"x = r / R$_{200c}$")
    ax.set_ylabel("3-D shape  (x/x$_c$)$^γ$ [1+(x/x$_c$)$^α$]$^{exponent}$")
    ax.set_title(r"Battaglia12 pressure shape, $M_{200c}=10^{14}\,M_\odot$, z = 0.5", loc="left", fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.text(0.5, 3e-4, "upstream and fork overlap exactly\n(max |ratio - 1| = 0)", fontsize=10, color=INK)
    save(fig, "04_b12_sign_bug_shape.png")


# ------------------------------------------------------------------ 4/5. low-z cache error (v6_plot.py data, separated)
def fig_lowz():
    R = list(csv.DictReader(open(f"{VA}/v1c_itp_vs_direct_grid.csv")))
    styles = {"0.100": ":", "1.000": "--", "3.900": "-"}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for mod, colr, lab in (("b12_allz", BLUE, "Battaglia12 cache"), ("lee22_allz", ORANGE, "Lee22 Table 7 cache")):
        for xv, ls in styles.items():
            rr = [r for r in R if r["map"] == mod and r["log10M"] == "14.0000" and r["x"] == xv]
            z = np.array([float(r["z"]) for r in rr]); e = np.abs([float(r["rel"]) for r in rr]); o = np.argsort(z)
            ax.plot(z[o], np.maximum(e[o], 1e-10), ls=ls, color=colr, lw=2, label=f"{lab}, x = {float(xv):.1f}")
    ax.axvline(0.0206, color=INK2, lw=1, ls="-.")
    ax.text(0.0225, 3e-8, "first cache z cell\n(0.001-0.0206) ends", fontsize=9, color=INK2)
    ax.axvspan(0.1, 2.0, color="#efeeea", zorder=0)
    ax.text(0.3, 3e-3, "range tested by the\nfirst audit (z = 0.1-2)", fontsize=9, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-10, 50)
    ax.set_xlabel("halo redshift z")
    ax.set_ylabel(r"|y$_{\rm interpolator}$ / y$_{\rm direct\ LOS}$ - 1|")
    ax.set_title(r"Production y(θ) caches vs direct XGPaint LOS, $M_{200c}=10^{14}\,M_\odot$", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    save(fig, "05_lowz_cache_error.png")

    fig, ax = plt.subplots(figsize=(10, 6.0))
    for mod, colr, lab in (("b12", BLUE, "Battaglia12 all-z map"), ("lee22", ORANGE, "Lee22 all-z map")):
        dd = np.load(f"{VA}/v4b_cl_{mod}.npz"); ell = dd["ell"]
        r = dd["cl_stored"] / dd["cl_corrected"]
        rs = np.convolve(r, np.ones(5) / 5, mode="same")
        ax.plot(ell[3:-3], rs[3:-3], color=colr, lw=2, label=lab)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(4, 2000)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$C_\ell$(stored) / $C_\ell$(z < 0.1 halos re-evaluated exactly)")
    ax.set_title("Effect of the low-z cache error on the stored y maps (NSIDE 1024, 5-ℓ running mean)", loc="left", fontsize=12)
    ax.legend(fontsize=10)
    save(fig, "06_lowz_cl_effect.png")


# ------------------------------------------------------------------ 6. painted interpolator vs independent LOS (z >= 0.1)
def fig_painted():
    R = list(csv.DictReader(open(f"{AA}/painted_profile_vs_independent_los.csv")))
    fig, ax = plt.subplots(figsize=(10, 6.0))
    seen = set()
    for r_model, colr, lab in (("b12", BLUE, "Battaglia12"), ("lee22", ORANGE, "Lee22 Table 7")):
        keys = sorted({(r["logM"], r["z"]) for r in R if r["model"] == r_model})
        for k in keys:
            rr = [r for r in R if r["model"] == r_model and (r["logM"], r["z"]) == k]
            x = np.array([float(r["x"]) for r in rr])
            e = np.array([float(r["y_itp"]) / float(r["y_indep_matched"]) - 1 for r in rr])
            o = np.argsort(x)
            ax.plot(x[o], 1e4 * e[o], color=colr, lw=1.1, alpha=0.75,
                    label=f"{lab}: {len(keys)} (M, z)" if r_model not in seen else None)
            seen.add(r_model)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(4.0, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"x = r$_\perp$ / R$_{200c}$")
    ax.set_ylabel(r"10$^4$ × (interpolator / independent LOS - 1)")
    ax.set_title("Painted interpolators vs an independent LOS integral (z = 0.1-2, log M = 13-15)", loc="left", fontsize=12)
    ax.legend(fontsize=10, loc="lower left")
    save(fig, "07_painted_vs_independent.png")


if __name__ == "__main__":
    fig_lineage()
    fig_tau_ratio()
    fig_sign_bug()
    fig_lowz()
    fig_painted()
