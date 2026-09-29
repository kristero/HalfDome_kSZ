#!/usr/bin/env python3
"""Summary CSV: effective constant corrections each observable 'sees' for a given C(z), and the headline numbers."""
import csv
import json
import numpy as np
import analysis as A

res = json.loads((A.OUT / "budget_results.json").read_text())
k0, k1 = res["konietzka_zs_test"]["fits_zs_0p1_to_2"]["C0(1+z)^k"]["params"]
c3 = res["konietzka_zs_test"]["fits_zs_0p1_to_2"]["C0(1+z)^3"]["params"][0]
d = A.load("validation")
lit = A.spectra(d, "tau_lee22_literal", "ksz"); lp = A.spectra(d, "y_lee22p", "tsz")
cands = {"TNG fit {:.3f}(1+z)^{:.2f}".format(k0, k1): lambda z: k0 * (1 + z) ** k1,
         "TNG fit {:.3f}(1+z)^3".format(c3): lambda z: c3 * (1 + z) ** 3}
# FRB weight at z_s = 1 (paint cosmology)
zf, w, _ = A.frb_weight(A.S.PAINT_COSMO, "lee22_literal", 1.0)
rows = []
for name, C in cands.items():
    row = dict(correction=name, frb_zs1_equiv_const=float(A.trapz(C(zf) * w, zf) / A.trapz(w, zf)))
    for l in (300, 1000, 3000, 5000):
        row["ksz_l{}_equiv_const".format(l)] = float(np.sqrt(A.at(lit, A.cl_total(lit, C), l) / A.at(lit, A.cl_total(lit), l)))
    rows.append(row)
for kp in (2.0, 3.0):
    C = lambda z, kp=kp: (1 + z) ** kp
    row = dict(correction="pressure (1+z)^{:.0f} (times C0')".format(kp))
    for l in (300, 1000, 3000, 8000):
        row["tsz_l{}_equiv_const".format(l)] = float(np.sqrt(A.at(lp, A.cl_total(lp, C), l) / A.at(lp, A.cl_total(lp), l)))
    rows.append(row)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "correction", k))
with open(A.OUT / "equivalent_constant_corrections.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
print(json.dumps(rows, indent=1))
