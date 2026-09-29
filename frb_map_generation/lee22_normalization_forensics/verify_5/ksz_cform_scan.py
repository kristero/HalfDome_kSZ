#!/usr/bin/env python3
"""verify_5: kSZ 1-halo (catalogue) ratio to B16 vs l for several density corrections C(z) on literal Lee22,
and painted-equivalent D3000 = painted literal D3000 x (catalogue C-corrected / catalogue literal)."""
import json
from pathlib import Path
import numpy as np
OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5")
d = np.load(OUT / "indep_halo_sums_tables.npz")
ells, V2, z = d["ells"], d["V2"], d["z"]
w = V2
tl, tb = d["tau_lit"], d["tau_b16"]
B = (w[:, None] * tb ** 2).sum(0); L = (w[:, None] * tl ** 2).sum(0)
PAINTED_LIT_D3000 = 2.4548176584114616
forms = {"literal": lambda z: 1 + 0 * z, "const_0.6016": lambda z: 0.6016316602183194 + 0 * z,
         "C_TNG_0.2842(1+z)^2.483": lambda z: 0.2841790493880425 * (1 + z) ** 2.483190184608004,
         "0.18(1+z)^3": lambda z: 0.18 * (1 + z) ** 3, "0.2387(1+z)^3": lambda z: 0.23866882396943453 * (1 + z) ** 3,
         "0.3601(1+z)^3/E^2": None}
from astropy.cosmology import FlatLambdaCDM
cos = FlatLambdaCDM(H0=67.74, Om0=0.3089)
forms["0.3601(1+z)^3/E^2"] = lambda z: 0.3601255521382468 * (1 + z) ** 3 / cos.efunc(z) ** 2
res = {}
i3 = int(np.where(ells == 3000)[0][0])
for k, f in forms.items():
    Cz = f(z)
    Lc = ((w * Cz ** 2)[:, None] * tl ** 2).sum(0)
    r = Lc / B
    res[k] = dict(ratio_to_b16={int(l): float(v) for l, v in zip(ells, r)},
                  r300_over_r3000=float(r[ells == 300][0] / r[ells == 3000][0]),
                  painted_equiv_D3000=float(PAINTED_LIT_D3000 * Lc[i3] / L[i3]))
    print("%-26s" % k, "ratio l=300/1000/3000/5000: %.2f %.2f %.2f %.2f" % tuple(r[np.isin(ells, [300, 1000, 3000, 5000])]),
          " r300/r3000=%.2f  painted-equiv D3000=%.3f" % (res[k]["r300_over_r3000"], res[k]["painted_equiv_D3000"]))
(OUT / "ksz_cform_scan.json").write_text(json.dumps(res, indent=1))
