#!/usr/bin/env python3
"""Which exponent convention reproduces Lee22's own drawn fit curves? Paper curves: vector paths of Fig. 2 (P_e) and Fig. 5
(n_e) from the arXiv source (no-c + BPL fit = Tables 7/8), z = 0, bins 10^13.2-13.4 and 10^13.8-14.0 h^-1 Msun. Candidates for
the outer factor [1 + (x/xc)^alpha]^E with the tabulated slope b (beta or beta'): E = -b (arXiv eq. 10-11 as printed, our
implementation), E = -(b - gamma)/alpha and E = -(b + gamma)/alpha (b read as an asymptotic slope), each evaluated at the bin
centre and averaged over the bin (0.02-dex mass nodes)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lee22_normalization_forensics/C_pressure_consistency"))
import lee22_models as lm  # noqa: E402
CURVES = ROOT / "outputs/lee22_normalization_forensics_20260928/C_pressure_consistency/lee22_fig2_fig5_vector_curves.csv"
G, A = -0.3, 1.0
CONV = {"-b (as printed; ours)": lambda b: -b, "-(b-gamma)/alpha": lambda b: -(b - G) / A, "-(b+gamma)/alpha": lambda b: -(b + G) / A}
d = np.genfromtxt(CURVES, delimiter=",", names=True, dtype=None, encoding="utf-8")
rows = []
for q, par in (("x3_Pe_over_P200", lm.lee_P_noc_params), ("x3_ne_over_n200", lm.lee_n_noc_params)):
    for panel, (lo, hi) in enumerate(((13.2, 13.4), (13.8, 14.0))):
        s = (d["quantity"] == q) & (d["panel"] == panel) & (d["curve"] == "without_c_bpl")
        x, v = d["x_r_over_R200"][s].astype(float), d["value"][s].astype(float)
        ok = (x > 0.04) & (x < 1.34)
        x, v = x[ok], v[ok]
        for name, E in CONV.items():
            for mode in ("centre", "bin mean"):
                masses = [10 ** (0.5 * (lo + hi))] if mode == "centre" else 10 ** np.linspace(lo + 0.01, hi - 0.01, 10)
                model = np.mean([x ** 3 * par(m, 0.0)[0] * (x / par(m, 0.0)[1]) ** G * (1 + (x / par(m, 0.0)[1]) ** A) ** E(par(m, 0.0)[2])
                                 for m in masses], axis=0)
                r = model / v
                rows.append((q.split("_")[1], "%.1f-%.1f" % (lo, hi), name, mode, float(np.median(np.abs(r - 1))), float(r.min()), float(r.max())))
print("%-6s %-9s %-22s %-8s  median|model/paper-1|   min    max" % ("qty", "bin", "exponent", "mass"))
for r in rows:
    print("%-6s %-9s %-22s %-8s  %8.3f            %.3f  %.3f" % r)
