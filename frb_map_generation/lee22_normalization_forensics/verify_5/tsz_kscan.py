#!/usr/bin/env python3
"""verify_5: exponent scan C_P = (1+z)^k on the Lee22 pressure (catalogue 1-halo), in-range and all masses:
max/min of Lee22p/B12 over l = 300..8000 and 30..8000."""
import json
from pathlib import Path
import numpy as np
OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5")
d = np.load(OUT / "indep_halo_sums_tables.npz")
ells, N, M, z = d["ells"], d["N"], d["M"], d["z"]
lo = M < 10 ** 14.8 / 0.68
yb, yl = d["y_b12"], d["y_lee22p"]
res = {}
for sel, tag in ((lo, "in_range"), (np.ones_like(lo), "all_masses")):
    B = ((N * sel)[:, None] * yb ** 2).sum(0)
    rows = []
    for k in np.arange(0, 3.01, 0.25):
        L = ((N * sel * (1 + z) ** (2 * k))[:, None] * yl ** 2).sum(0)
        r = L / B
        m1 = (ells >= 300); m2 = ells >= 30
        rows.append(dict(k=float(k), r300=float(r[ells == 300][0]), r1000=float(r[ells == 1000][0]), r3000=float(r[ells == 3000][0]),
                         r8000=float(r[ells == 8000][0]), maxmin_300_8000=float(r[m1].max() / r[m1].min()),
                         maxmin_30_8000=float(r[m2].max() / r[m2].min())))
    res[tag] = rows
    print(tag)
    for row in rows:
        print(" k=%.2f r300=%.3f r1000=%.3f r3000=%.3f r8000=%.3f maxmin300-8000=%.3f maxmin30-8000=%.3f" % tuple(row.values()))
(OUT / "tsz_kscan.json").write_text(json.dumps(res, indent=1))
