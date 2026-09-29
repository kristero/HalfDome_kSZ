"""One-figure summary of the redshift test: s_model - s_FLAMINGO (s = dln/dln(1+z) at fixed M200c bin, z <= 2) for
Y(<R200c)/(P200 R200^3) and the electron fraction f_e(<R200c), against the M1 code-level predictions -2 (pressure) and
-3 (density). Reads OUT/pressure/pressure_trends_raw.csv and OUT/density/density_trends.csv; writes OUT/fig_summary_redshift_test.png."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common_c import OUT, BLUE, ORANGE, GREEN, PINK, BLACK  # noqa: E402

pt = [e for e in csv.DictReader(open(OUT / "pressure" / "pressure_trends_raw.csv")) if float(e["zmax"]) == 2.0 and e["x"] == "Y200"]
dt = [e for e in csv.DictReader(open(OUT / "density" / "density_trends.csv")) if float(e["zmax"]) == 2.0 and e["x"] == "f_e"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, rows, title, models, pred in (
        (axes[0], pt, r"pressure: $Y(<R_{200c})/(P_{200}R_{200}^3)$",
         [("s_B12", BLUE, "o", "Battaglia+12"), ("s_Lee22", ORANGE, "s", "Lee22 no-c (as printed = as painted)"),
          ("s_Lee22_a2", GREEN, "D", r"Lee22 $\times(1+z)^2$"), ("s_FLfit", PINK, "v", "FLAMINGO 9-coeff. fit")], -2.0),
        (axes[1], dt, r"density: electron fraction $f_e(<R_{200c})$",
         [("s_B16", BLUE, "o", "Battaglia+16 (XGPaint)"), ("s_Lee22_literal", ORANGE, "s", "Lee22 no-c printed / x0.6016"),
          ("s_Lee22_tngcal", GREEN, "D", r"Lee22 $\times$C(z) = 0.284(1+z)$^{2.48}$"),
          ("s_Lee22_literal_plus3_M1", PINK, "v", r"Lee22 printed $\times(1+z)^3$ (M1)")], -3.0)):
    lm = np.array([0.5 * (float(e["log10_m_lo"]) + float(e["log10_m_hi"])) for e in rows])
    sfl = np.array([float(e["s_FLAMINGO"]) for e in rows])
    err = np.array([float(e["s_FLAMINGO_err"]) for e in rows])
    ax.axhspan(-0.05, 0.05, color="0.85")
    ax.axhline(0, color=BLACK, lw=1)
    ax.axhline(pred, color=ORANGE, lw=1, ls=":")
    ax.text(13.04, pred + 0.1, f"M1 code-level prediction {pred:+.0f}", color=ORANGE, fontsize=8.5, ha="left")
    for key, color, marker, label in models:
        d = np.array([float(e[key]) for e in rows]) - sfl
        ax.errorbar(lm, d, err, fmt=marker + "-", color=color, ms=5, lw=1.4, capsize=2, label=label)
    ax.axvline(14.8 - np.log10(0.681), color="0.5", lw=0.8, ls="--")
    ax.text(14.8 - np.log10(0.681) - 0.02, -3.55, "Lee22 max mass\n(10$^{14.8}$ h$^{-1}$)", fontsize=7.5, ha="right", color="0.4")
    ax.set_title(title)
    ax.set_xlabel(r"log$_{10}$ M200c [Msun] (0.2-dex bin centre)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="lower left" if pred == -2.0 else "center right")
axes[0].set_ylabel(r"$s_{model} - s_{FLAMINGO}$,  s = dln/dln(1+z), 0 $\leq$ z $\leq$ 2")
axes[0].set_ylim(-3.8, 1.0)
fig.suptitle("Redshift evolution of Lee22 relative to FLAMINGO L2p8_m9 (same haloes; error bars = FLAMINGO slope error)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_summary_redshift_test.png", dpi=130)
