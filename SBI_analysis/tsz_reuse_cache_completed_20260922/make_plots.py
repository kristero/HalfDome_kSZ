"""Publication-sized figures; 100-ell averaging is for display only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results.json").read_text())
SPECTRA = np.load(ROOT / "spectra.npz")
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
LABELS = ["Battaglia12", "FL_L1_m9", "compact", "extended_shallow"]
NAMES = ["Battaglia12", "FLAMINGO fit", "Compact / faint", "Bright / shallow"]
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
                    "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
                    "savefig.dpi": 210, "svg.fonttype": "none"})


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(PLOTS / (name + "." + suffix), bbox_inches="tight")
    plt.close(fig)


def timing(group, identifier):
    return next(t for t in DATA["timings"] if t["group"] == group and t["id"] == identifier)


def error(group, identifier, label):
    return next(e for e in DATA["errors"] if e["group"] == group and e["task"] == identifier and e["label"] == label)


def speed():
    controls = [("original_exterior", 0, "Four catalogue passes"),
                ("original_exterior", 1, "Read once; four painters"),
                ("original_exterior", 2, "Shared geometry"),
                ("original_exterior", 8, "Shared geometry repeat"),
                ("balanced", 0, "Shared + balanced blocks")]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    values = [timing(g, i)["painting_seconds"] / 60 for g, i, _ in controls]
    bars = ax.barh(np.arange(len(values)), values, color=["#666666", "#999999", "#3279a8", "#91b6cf", "#d17824"])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=15)
    ax.set(yticks=np.arange(len(values)), yticklabels=[n for _, _, n in controls],
           xlabel="Painting time for four skies [min]", xlim=(0, max(values) * 1.14))
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    save(fig, "painting_time")


def cache():
    controls = [("common_reference", 2, "Default: 512 / 256 / 128"),
                ("common_reference", 3, "Half theta: 256 / 256 / 128"),
                ("common_reference", 4, "Half z: 512 / 128 / 128"),
                ("common_reference", 5, "Half mass: 512 / 256 / 64"),
                ("common_reference", 6, "Half all: 256 / 128 / 64"),
                ("common_reference", 7, "Quarter all: 128 / 64 / 32"),
                ("smooth_exterior", 0, "Smooth boundary; default"),
                ("smooth_exterior", 1, "Smooth boundary; half all")]
    values = np.array([[error(g, i, label)["distance9"] for label in LABELS] for g, i, _ in controls])
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    im = ax.imshow(np.maximum(values, 1e-12), norm=LogNorm(1e-12, 1e2), cmap="cividis", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center", fontsize=14,
                    color="white" if values[i, j] < 1e-5 else "black")
    ax.set(xticks=np.arange(4), xticklabels=["Battaglia12", "FLAMINGO", "Compact", "Bright"],
           yticks=np.arange(len(controls)), yticklabels=[n for _, _, n in controls])
    fig.colorbar(im, ax=ax, label="Local MOPED shift / noise", ticks=[1e-12, 1e-8, 1e-4, 1, 100])
    save(fig, "cache_common_reference")
    ell = SPECTRA["ell"]
    groups = np.array_split(np.arange(len(ell)), 79)
    x = np.array([ell[g].mean() for g in groups])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    for label, title, ax in zip(LABELS, NAMES, axes.flat):
        reference = SPECTRA["smooth_002__" + label]
        for key, name, color in [("original_002", "Default", "#666666"),
                                  ("original_006", "Half all", "#bb6a24"),
                                  ("original_007", "Quarter all", "#99495d"),
                                  ("smooth_001", "Smooth; half all", "#167698")]:
            value = SPECTRA[key + "__" + label]
            ratio = np.array([1e6 * ((value[g].mean() / reference[g].mean()) - 1) for g in groups])
            ax.plot(x, ratio, label=name, color=color, lw=1.5)
        ax.axhline(0, color="black", lw=.6); ax.grid(alpha=.2); ax.set(title=title, xscale="log")
    for ax in axes[-1]: ax.set_xlabel(r"$\ell$")
    for ax in axes[:, 0]: ax.set_ylabel(r"$\Delta D_\ell / D_\ell$ [ppm]")
    axes[0, 0].legend(fontsize=11)
    save(fig, "cache_spectral_residuals")


def resolution():
    ell = SPECTRA["ell"]
    groups = np.array_split(np.arange(len(ell)), 79)
    x = np.array([ell[g].mean() for g in groups])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for j, (label, title) in enumerate(zip([LABELS[0], LABELS[1], LABELS[3]], [NAMES[0], NAMES[1], NAMES[3]])):
        key = {"Battaglia12": "original_012", "extended_shallow": "original_011", "FL_L1_m9": "reference16384"}[label]
        fine = SPECTRA[key + "__" + label]
        for prefix, name, color, linestyle in [("centre4096", "4096", "#bc6728", "-"),
                                               ("original_002", "8192", "#247daa", "--"),
                                               (key, "16384 reference", "#222222", ":")]:
            value = SPECTRA[prefix + "__" + label]
            avg = np.array([value[g].mean() for g in groups])
            ratio = np.array([100 * (value[g].mean() / fine[g].mean() - 1) for g in groups])
            axes[0, j].loglog(x, avg, label=name, color=color, ls=linestyle, lw=2)
            axes[1, j].semilogx(x, ratio, color=color, ls=linestyle, lw=1.5)
        axes[0, j].set_title(title)
        axes[1, j].set_xlabel(r"$\ell$"); axes[1, j].axhline(0, color="black", lw=.6)
    axes[0, 0].set_ylabel(r"$\widetilde D_\ell^{yy}$")
    axes[1, 0].set_ylabel("Difference [%]")
    axes[0, 1].legend(loc="lower left", fontsize=12)
    for ax in axes.flat: ax.grid(alpha=.2)
    save(fig, "resolution_spectra")


def pixels():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, label, title in zip(axes, [LABELS[0], LABELS[3]], [NAMES[0], NAMES[3]]):
        for corrected, name, color in [(False, "Child mean", "#bc6728"), (True, "Mean / parent window", "#247daa")]:
            rows = [r for r in DATA["pixel_quadrature"] if r["label"] == label and
                    r.get("common_reference_nside") == 16384 and r["dewindow"] == corrected]
            rows.sort(key=lambda r: r["fine_nside"])
            ax.plot([4, 16], [r["distance9"] for r in rows], "o-", label=name, color=color, lw=2)
        ax.set(title=title, xticks=[4, 16], xlabel="Samples per 4096 pixel", yscale="log")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Local MOPED shift / noise")
    axes[0].legend(loc="best", fontsize=11)
    save(fig, "pixel_refinement")


if __name__ == "__main__":
    speed(); cache(); resolution(); pixels()
    print("Wrote five PNG figures and five editable SVG figures.")
