#!/usr/bin/env python3
"""Figure: halo DM per crossing vs halo redshift (1e13-1e14), TNG vs smooth-profile models."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs" / "lee22_normalization_forensics_20260928" / "A_tng_redshift_test"
rows = list(csv.DictReader(open(OUT / "per_crossing_dm_by_slice_mass13to14.csv")))
z = np.array([0.5 * (float(r["z_lo"]) + float(r["z_hi"])) for r in rows])
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for key, color, lab, dx, mk in (("tng", INK, "TNG (Konietzka ray increments)", 0.0, "o"),
                                ("lee22_xgpnative", "#8a8984", "Lee22 x 0.602 (XGPaint-native)", -0.012, "s"),
                                ("lee22_corrected", "#1baf7a", "Lee22 literal x fitted C(z)", 0.012, "D"),
                                ("b16", "#eb6834", "Battaglia16", 0.024, "^")):
    med = np.array([float(r[key + "_median"]) for r in rows])
    lo = np.array([float(r[key + "_p16"]) for r in rows])
    hi = np.array([float(r[key + "_p84"]) for r in rows])
    ax.errorbar(z + dx, med, [med - lo, hi - med], fmt=mk + "-", color=color, ms=5, lw=1.8, elinewidth=1.0, capsize=2,
                label=lab)
ax.set_yscale("log")
ax.set_xlabel("halo redshift (slice centre)")
ax.set_ylabel("DM per halo crossing [pc cm$^{-3}$]\nmedian and 16-84 %")
ax.set_title("1e13-1e14 Msun: DM per crossing inside R200c, by halo redshift", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / "per_crossing_dm_by_slice_mass13to14.{}".format(ext), dpi=160)
print("ok")
