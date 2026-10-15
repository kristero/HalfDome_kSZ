"""Figures for the importance kernels (reads importance.npz only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("plot_style", HERE/"plot_style.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

FIGURES = HERE/"figures"
PANELS = (("P4__cl_1h", "yy 1-halo, P4"), ("P4__cl_2h", "yy linear 2-halo, P4"), ("P4__cl_total", "yy total, P4"),
          ("P2__cl_1h", "yy 1-halo, P2"), ("P2__cl_2h", "yy linear 2-halo, P2"), ("P2__cl_total", "yy total, P2"),
          ("E4__ksz_cl_1h", "kSZ 1-halo, E4"), ("E4__ksz_cl_2h_approx", "kSZ 2-halo (approx.), E4"),
          ("E4__dm_cl_1h", "DM-field 1-halo, E4"),
          ("P4xE4__cross_1h_P", "y x DM 1-halo, pressure side"), ("P4xE4__cross_1h_E", "y x DM 1-halo, electron side"),
          ("E4__dm_pdf", "DM one-point statistics, E4"))


def norm_for(values):
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if low < -.5:
        limit = max(abs(low), abs(high))
        return TwoSlopeNorm(0., -limit, limit), ST.DIVERGING, True
    return Normalize(0., high), ST.BLUE_RAMP, False


def panel_values(data, key):
    if key == "E4__dm_pdf":
        mean = data["normalised__E4__dm_mean"].sum(axis=1)[:, None]
        var = data["normalised__E4__dm_var"].sum(axis=1)[:, None]
        quant = data["normalised__E4__dm_quantiles"].sum(axis=1)
        return 100*np.c_[mean, var, quant], ["mean", "var"]+[f"Q{100*q:g}" for q in data["quantile_levels"]]
    return 100*data[f"normalised__{key}"].sum(axis=1), [f"{int(e)}" for e in data["ell"]]


def heatmap(data):
    values = [panel_values(data, key)[0] for key, _ in PANELS]
    norm, cmap, signed = norm_for(np.concatenate([v.ravel() for v in values]))
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 17), layout="constrained", sharey=True)
    shells = list(data["shell_labels"])
    for i, (ax, (key, title), grid) in enumerate(zip(axes.ravel(), PANELS, values)):
        columns = panel_values(data, key)[1]
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.grid(False)
        for s_ in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[s_, c]
                dark = (abs(norm(v)-.5) > .32) if signed else norm(v) > .62
                ax.text(c, s_, f"{v:.0f}" if abs(v) >= 9.95 else f"{v:.1f}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if dark else ST.INK)
        ax.set_xticks(range(len(columns)), columns, fontsize=9)
        ax.set_yticks(range(len(shells)), shells, fontsize=9)
        ax.set_title(title)
        if i % 3 == 0:
            ax.set_ylabel(r"Shell [$R_{200c}$]")
        if i >= 9:
            ax.set_xlabel(r"Multipole $\ell$" if key != "E4__dm_pdf" else "Statistic (mean, variance, quantiles)")
    bar = fig.colorbar(image, ax=axes, shrink=.45, pad=.01)
    bar.set_label("Normalised shell sensitivity [%] (each column sums to 100%)")
    bar.outline.set_visible(False)
    fig.suptitle(r"Which radii drive each observable? Shell importance summed over halo mass (4R cut; columns: $\ell$ or statistic)")
    return ST.save(fig, FIGURES, "importance_heatmap")


MASS_PANELS = (("P4__cl_total", 0, r"yy total, $\ell=100$, P4"), ("P4__cl_total", 2, r"yy total, $\ell=1000$, P4"),
               ("P4__cl_total", 3, r"yy total, $\ell=3000$, P4"), ("P4__cl_total", 4, r"yy total, $\ell=8000$, P4"),
               ("E4__ksz_cl_1h", 3, r"kSZ 1-halo, $\ell=3000$, E4"), ("P4xE4__cross_1h_E", 2, r"y x DM 1h, $\ell=1000$, electron side"),
               ("E4__dm_mean", None, "mean DM, E4"), ("E4__dm_var", None, "DM variance, E4"),
               ("E4__dm_quantiles", 1, "DM median (Q50), E4"), ("E4__dm_quantiles", 4, "DM Q99.9, E4"))


def mass_shell(data):
    """Joint (shell, mass bin) maps, each scaled to its own maximum |value|."""
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.4), layout="constrained", sharex=True, sharey=True)
    bins = [b.split("-")[0] if "-" in b else "15.5+" for b in data["mass_bin_labels"]]
    shells = list(data["shell_labels"])
    for ax, (key, column, title) in zip(axes.ravel(), MASS_PANELS):
        values = data[f"normalised__{key}"]
        grid = 100*(values[..., column] if column is not None else values)
        peak = np.max(np.abs(grid))
        image = ax.imshow(grid/peak, origin="lower", aspect="auto", cmap=ST.DIVERGING, vmin=-1, vmax=1)
        ax.grid(False)
        top = np.unravel_index(np.argmax(grid), grid.shape)
        ax.set_title(f"{title}\nmax {grid[top]:.1f}% at {shells[top[0]]} R, bin {bins[top[1]]}", fontsize=10.5)
    for ax in axes[1]:
        ax.set_xticks(range(len(bins)), bins, rotation=90, fontsize=8.5)
        ax.set_xlabel(r"Mass bin, lower edge $\log_{10}M_{200c}$")
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(shells)), shells, fontsize=9)
        ax.set_ylabel(r"Shell [$R_{200c}$]")
    bar = fig.colorbar(image, ax=axes, shrink=.7, pad=.01)
    bar.set_label("Sensitivity relative to the panel maximum (each panel sums to 100%)")
    bar.outline.set_visible(False)
    fig.suptitle("Which haloes and which radii matter: joint shell x mass-bin importance (no halos below 12.81)")
    return ST.save(fig, FIGURES, "importance_mass_shell")


def flux_and_mass(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.8))
    centres = data["flux_logmass_centres"]
    shells = list(data["shell_labels"])
    for ax, name, label in ((axes[0, 0], "P4", "Y(<4R) shell fractions, pressure 4R fit"),
                            (axes[0, 1], "E4", "N$_e$(<4R) shell fractions, electrons 4R fit")):
        fractions = 100*data[f"flux_fraction__{name}"]
        fallback = data[f"flux_source__{name}"] != "fit"
        bottom = np.zeros(len(centres))
        for s in range(fractions.shape[1]):
            ax.bar(centres, fractions[:, s], width=.17, bottom=bottom, color=ST.ORDINAL8[s],
                   edgecolor=ST.SURFACE, linewidth=1., label=shells[s])
            bottom += fractions[:, s]
        edge = centres[fallback & (centres < 14)].max()+.1
        ax.axvspan(centres.min()-.15, edge, color=ST.MIDPOINT, zorder=0, lw=0)
        ax.text(.5*(centres.min()-.15+edge), 103, "published\nfallback", ha="center", va="bottom", fontsize=9,
                color=ST.INK_SECONDARY)
        ax.set(xlabel=r"$\log_{10}(M_{200c}/M_\odot)$ (0.2-dex bin centre)", ylabel="Share of 4R integral [%]",
               title=label, ylim=(0, 118))
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(title=r"Shell [$R_{200c}$]", ncol=4, loc="upper right", fontsize=8.5, title_fontsize=9)
    bins = list(data["mass_bin_labels"])
    x = np.arange(len(bins))
    ax = axes[1, 0]
    for i, l in enumerate((0, 2, 3, 4)):
        ax.plot(x, 100*data["normalised__P4__cl_total"][..., l].sum(axis=0), color=ST.CATEGORICAL[i], marker="o", ms=4,
                label=rf"yy total, $\ell={int(data['ell'][l])}$")
    ax.set(ylabel="Share of the observable [%]", title="Which haloes drive the tSZ power (P4)")
    ax = axes[1, 1]
    series = (("E4__dm_mean", None, "mean DM"), ("E4__dm_var", None, "DM variance"),
              ("E4__dm_quantiles", 1, "DM median"), ("E4__dm_quantiles", 4, "DM Q99.9"),
              ("E4__ksz_cl_1h", 3, r"kSZ 1h, $\ell=3000$"))
    for i, (key, column, label) in enumerate(series):
        values = data[f"normalised__{key}"]
        values = values[..., column] if column is not None else values
        ax.plot(x, 100*values.sum(axis=0), color=ST.CATEGORICAL[i], marker="o", ms=4, label=label)
    ax.set(title="Which haloes drive DM and kSZ (E4)")
    for ax in axes[1]:
        ax.set_xticks(x, [b.split("-")[0] if "-" in b else b for b in bins], rotation=90, fontsize=8.5)
        ax.set_xlabel(r"Mass bin, lower edge $\log_{10}M_{200c}$")
        ax.legend(fontsize=9)
        ST.zero_line(ax)
    axes[1, 1].set_ylabel("Share of the observable [%]")
    fig.tight_layout()
    return ST.save(fig, FIGURES, "importance_flux_mass")


def main():
    ST.apply()
    data = dict(np.load(HERE/"importance.npz"))
    heatmap(data)
    mass_shell(data)
    flux_and_mass(data)


if __name__ == "__main__":
    main()
