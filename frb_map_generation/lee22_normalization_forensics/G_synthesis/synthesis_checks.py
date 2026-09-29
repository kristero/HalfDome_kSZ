"""Synthesizer cross-checks (2026-09-28).
1. Lee22 Table 7+8 (no-c) implied M_gas(<R200c)/(f_b M200) and T_e/T200 (number weighted), and T/T200
   after removing the code-predicted a^3 (density) and a^2 (pressure) bookkeeping factors.
2. Lee22-pressure / B12 tSZ 1-halo ratio split at the Lee22 fit edge M200c = 10^14.8 h^-1 Msun, from
   F_sz_budget's Tinker08 halo-model mass budgets (budget_curves.npz dlnM are normalized distributions).
3. Konietzka TNG300 1e13-1e14 Msun halo DM stats vs z_s.
"""
import json
import numpy as np
from scipy.integrate import quad

ROOT = "/home/cbllover/HalfDome"
OUT = ROOT + "/frb_map_generation/outputs/lee22_normalization_forensics_20260928/G_synthesis"
XH, MU = 0.76, 0.59
T = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def bpl(A, am, amp, az, M, Mc, z):
    s = M / Mc
    return A * (1 + z) ** az * (s ** am if s < 1 else s ** amp)

def gnfw(x, A, xc, b):
    return A * (x / xc) ** -0.3 * (1 + x / xc) ** (-b)

lines = ["# 1. Lee22 no-c fits: log10(M h), z, <n/n200>(<R200c), Mgas/(fb M200), T/T200, T/T200 x (1+z)^-1"]
for lm in [13.0, 13.61, 14.0, 14.5]:
    for z in [0.0, 0.5, 1.0, 2.0]:
        M = 10 ** lm
        n0 = bpl(6.8, 0.68, 0.68, -2.11, M, 10 ** 13.61, z)
        xcn = bpl(7.9, 0.47, -0.45, -0.67, M, 10 ** 13.61, z)
        bn = bpl(19.5, 0.70, -0.18, -0.31, M, 10 ** 13.61, z)
        P0 = bpl(2.8, 1.08, 0.80, -1.89, M, 10 ** 13.60, z)
        xcp = bpl(2.10, -0.35, -0.35, 0.53, M, 10 ** 13.60, z)
        bp = bpl(9.4, -0.06, -0.06, 0.60, M, 10 ** 13.60, z)
        In = quad(lambda x: x * x * gnfw(x, n0, xcn, bn), 0, 1)[0]
        Ip = quad(lambda x: x * x * gnfw(x, P0, xcp, bp), 0, 1)[0]
        t = Ip / In * XH / MU
        lines.append(f"{lm:.2f} {z:.1f} {3*In:.3f} {3*In*2/(1+XH)/XH:.2f} {t:.3f} {t/(1+z):.3f}")

F = ROOT + "/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget/"
f = np.load(F + "budget_curves.npz")
d = json.load(open(F + "budget_results.json"))["validation"]["tsz"]
lnM = f["validation_lnM"]; lM = lnM / np.log(10)
cut = 14.8 - np.log10(0.68)  # painting cosmology h
m = lM <= cut
lines.append("# 2. tSZ 1-halo Lee22p/B12: ell, all masses, M<=10^14.8/h, M>10^14.8/h, Lee22p frac above")
for l in ["300", "1000", "3000", "8000"]:
    a = f[f"validation_lee22p_{l}_dlnM"]; b = f[f"validation_b12_{l}_dlnM"]
    fa = T(a[m], lnM[m]) / T(a, lnM); fb = T(b[m], lnM[m]) / T(b, lnM)
    A = d[l]["D_lee22p_1h"]; B = d[l]["D_b12_1h"]
    lines.append(f"{l} {A/B:.2f} {A*fa/(B*fb):.2f} {A*(1-fa)/(B*(1-fb)):.2f} {1-fa:.2f}")

K = ROOT + "/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/"
zs = np.load(K + "redshifts_DMhalo.npy"); arr = np.load(K + "DMhalo_r200_mass13to14.npy")
lines.append("# 3. Konietzka 13-14: z_s, hit fraction, mean per hit, all-ray mean")
for z in [0.1, 0.3, 0.5, 1.0, 2.0]:
    i = np.argmin(abs(zs - z)); v = arr[i]; h = v > 0
    lines.append(f"{zs[i]:.2f} {h.mean():.4f} {v[h].mean():.1f} {v.mean():.2f}")

open(OUT + "/synthesis_checks.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
