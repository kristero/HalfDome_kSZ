"""Read-only check: fractional bootstrap error of the stacked mean on x200 sub-shells vs their parent fit-grid shell.

Reads web_variants.json (ref = fit grid, grids_scaling:x200).  Calibrated bins, shells resolved on both grids.
Splits the x200 sub-shells into slivers (< 0.02 dex) and the rest.  Writes nothing (stdout only).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = json.loads((HERE / "web_variants.json").read_text())
R = json.loads((HERE / "results.json").read_text())
CAL = R["calibrated_bins"]
rec = {(r["variant"], r["field"], r["bin"]): r for r in W["records"]}
for f in ("P", "E"):
    thin, wide = [], []
    for b in CAL:
        a, x = rec[("ref", f, b)], rec[("grids_scaling:x200", f, b)]
        fe, xe = np.array(a["edges"]), np.array(x["edges"])
        fa = np.array(a["err"]) / np.array(a["value"])
        fx = np.array(x["err"]) / np.array(x["value"])
        for i in range(len(xe) - 1):
            if xe[i] <= 0 or not x["resolved"][i]:
                continue
            j = int(np.searchsorted(fe, xe[i] + 1e-9) - 1)
            if not a["resolved"][j]:
                continue
            if abs(fe[j] - xe[i]) < 1e-6 and abs(fe[j + 1] - xe[i + 1]) < 1e-6:
                continue
            dex = np.log10(xe[i + 1] / xe[i])
            (thin if dex < 0.02 else wide).append(fx[i] / fa[j])
    for nm, v in (("slivers < 0.02 dex", thin), ("other split shells", wide)):
        v = np.array(v)
        print(f, nm, "n", len(v), "median", round(float(np.median(v)), 3), "min", round(float(v.min()), 3),
              "max", round(float(v.max()), 3))
