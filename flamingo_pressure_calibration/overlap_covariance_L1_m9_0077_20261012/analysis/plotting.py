"""Shared figure style: fixed categorical order, diverging map, neutral text, PNG+SVG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#dcdad5"
SURFACE = "#ffffff"
MISSING = "#d9d7d2"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])
DIVERGING.set_bad(MISSING)

OBJECTIVE_COLOURS = {"a": CATEGORICAL[0], "b": CATEGORICAL[1], "c": CATEGORICAL[2]}
OBJECTIVE_LABELS = {"a": "(a) diagonal SD/mean, 5% floor", "b": "(b) diagonal error on mean",
                    "c": "(c) GLS + shrinkage + sigma_d"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT_2,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titlecolor": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "none", "axes.prop_cycle": matplotlib.cycler(color=list(CATEGORICAL)),
    })


def save(fig, path_without_suffix: Path | str, dpi: int = 150) -> list[str]:
    path = Path(path_without_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in (".png", ".svg"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=dpi, bbox_inches="tight", metadata={"Date": None} if suffix == ".svg" else None)
        out.append(str(target))
    plt.close(fig)
    return out
