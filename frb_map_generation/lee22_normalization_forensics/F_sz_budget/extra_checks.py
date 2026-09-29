#!/usr/bin/env python3
"""Grid-convergence check (tables_validation_fine) and 1R200c-sphere kSZ variant (tables_validation_ksz1r)."""
import json
import numpy as np
import analysis as A

res = json.loads((A.OUT / "budget_results.json").read_text())
k0, k1 = res["konietzka_zs_test"]["fits_zs_0p1_to_2"]["C0(1+z)^k"]["params"]
C = lambda z: k0 * (1 + z) ** k1
out = {}
for tag in ("", "_fine", "_ksz1r"):
    A.TAG = tag
    d = A.load("validation")
    lit, b16 = A.spectra(d, "tau_lee22_literal", "ksz"), A.spectra(d, "tau_b16", "ksz")
    lp, b12 = A.spectra(d, "y_lee22p", "tsz"), A.spectra(d, "y_b12", "tsz")
    row = dict(meta=dict(nz=d["meta"]["nz"], nm=d["meta"]["nm"], ksz_sphere=d["meta"]["ksz_sphere_r200c"]))
    for l in (300, 1000, 3000, 5000):
        Dl_ = lambda s, c=None: A.at(s, A.Dl(s, A.cl_total(s, c)), l)
        row[l] = dict(ksz_literal=Dl_(lit), ksz_xgp=Dl_(lit) * A.XGP ** 2, ksz_b16=Dl_(b16), ksz_tngC=Dl_(lit, C),
                      ksz_xgp_over_b16=Dl_(lit) * A.XGP ** 2 / Dl_(b16), ksz_tngC_over_b16=Dl_(lit, C) / Dl_(b16),
                      tsz_ratio=Dl_(lp) / Dl_(b12), zmed_ksz_literal=A.zbudget(lit, l)[0]["z_median"],
                      zmed_ksz_tngC=A.zbudget(lit, l, C)[0]["z_median"])
    out[tag or "baseline"] = row
(A.OUT / "extra_checks.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v["meta"])
    for l in (300, 1000, 3000, 5000):
        print("  ", l, {a: round(b, 4) for a, b in v[l].items()})
