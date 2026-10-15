"""Shared figure conventions for kernels/ and C_sigmoid/ (matplotlib, PNG+SVG).

Categorical colours in fixed order (never cycled), a one-hue sequential blue
ramp, a blue-grey-red diverging map with symmetric limits, text in neutral
inks, 1.75 pt lines, recessive solid grid, one y-axis per panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
ORDINAL8 = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
MIDPOINT = "#f0efec"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#cde2fb", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", MIDPOINT, "#e34948"])
LINE = 1.75


def apply():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.linewidth": LINE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": .6, "grid.linestyle": "-",
        "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "svg.fonttype": "none"})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = directory/f"{name}.{extension}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def zero_line(ax):
    ax.axhline(0., color=INK_SECONDARY, lw=.8, zorder=1)
