"""kT_e/kT200 at the shell containing x = 0.5: FLAMINGO (sum n_e kT V / sum N_e from the idark products) vs the printed
Lee22 P/n (X_H/mu x Pbar/nbar; the M1 volume factors cancel only at z = 0). kT200 = G M200c mu m_p/(2 R200c), mu = 0.588.
Output: OUTV/v2b_temperature.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as L  # noqa: E402

DENS = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/C_flamingo/idark_density/products")
ST = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/notebook_cache/stacks")
OUTV = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_C_flamingo")
G, MSUN, MPC, MP = 6.67430e-11, 1.98841e30, 3.0856775814913673e22, 1.67262192369e-27
X_H = 0.76
MU = 4 / (3 + 5 * X_H)
res = {}
for s in (78, 58, 38):
    d = np.load(DENS / f"L2p8_m9_{s:04d}_density_products.npz")
    st = np.load(ST / f"L2p8_m9_{s:04d}_stack_products.npz")
    z = float(d["redshift"])
    e = d["edges"]
    k = int(np.searchsorted(e, 0.5) - 1)
    m, r = d["m200c_msun"], d["r200_mpc"]
    kt200 = G * m * MSUN * MU * MP / (2 * r * MPC)
    te = d["pe3"][:, k] / d["ne3"][:, k] / kt200
    lm = np.log10(m)
    sel = lm >= 13.6
    xm = np.sqrt(e[k] * e[k + 1])
    lee = X_H / MU * L.lee_Pbar(xm, m[sel] * 0.681, z) / L.lee_nbar(xm, m[sel] * 0.681, z)
    res[str(round(z, 2))] = {"shell": [float(e[k]), float(e[k + 1])], "n": int(sel.sum()),
                             "FL_Te_over_T200_median": float(np.nanmedian(te[sel])),
                             "Lee22_printed_implied_median": float(np.median(lee)),
                             "Lee22_M1_corrected_implied_median (x (1+z)^-1)": float(np.median(lee) / (1 + z))}
(OUTV / "v2b_temperature.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
