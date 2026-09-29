#!/usr/bin/env python
"""Verifier: recompute the halo-model tSZ ratios of workstream D (Lee22 printed x (1+z)^2 vs printed vs B12) from the
F_sz_budget 'validation' tables with an independent integration (no szhm import). Also report the table mass units."""
import numpy as np
T = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget/tables_validation.npz"
d = dict(np.load(T))
print({k: d[k].shape for k in d})
z, M, ell = d["z"], d["M"], d["ell"]
print(f"M range {M.min():.3e}-{M.max():.3e} (docstring: physical Msun 6.56e12-3.79e15); z {z.min():.3f}-{z.max():.2f}")
lnM = np.log(M)
W = d["chi"] ** 2 / d["Hc"]
k = (ell[None, :] + 0.5) / d["chi"][:, None]
Pl = d["D"][:, None] ** 2 * np.exp(np.interp(np.log(k), np.log(d["k_lin"]), np.log(d["p_lin0"])))
def C(key, w, mmax=np.inf, two=True):
    X = d[key] * w[:, None, None] * (M <= mmax)[None, :, None]
    one = np.trapz(W[:, None] * np.trapz(d["dn"][:, :, None] * X ** 2, lnM, axis=1), z, axis=0)
    if not two:
        return one
    Ib = np.trapz((d["dn"] * d["bias"])[:, :, None] * X, lnM, axis=1)
    return one + np.trapz(W[:, None] * Ib ** 2 * Pl, z, axis=0)
one = np.ones_like(z); w2 = (1 + z) ** 2
p, c, b = C("y_lee22p", one), C("y_lee22p", w2), C("y_b12", one)
for l in (300, 1000, 3000):
    i = np.argmin(np.abs(ell - l))
    print(f"l={ell[i]:.0f}: corr/printed {c[i]/p[i]:.3f}  corr/B12 {c[i]/b[i]:.3f}  printed/B12 {p[i]/b[i]:.3f}")
for mmax in (1e14, 10 ** 14.8 / 0.6774, np.inf):
    c1, b1 = C("y_lee22p", w2, mmax, False), C("y_b12", one, mmax, False)
    print(f"M<={mmax:.3g}: 1h corr/B12 " + ", ".join(f"{c1[np.argmin(np.abs(ell-l))]/b1[np.argmin(np.abs(ell-l))]:.3f}" for l in (300, 1000, 3000)))
# where in z does the 1-halo come from for the printed lee22 at l=3000
i = np.argmin(np.abs(ell - 3000))
dC = W * np.trapz(d["dn"] * d["y_lee22p"][:, :, i] ** 2, lnM, axis=1)
cum = np.concatenate([[0], np.cumsum(0.5 * (dC[1:] + dC[:-1]) * np.diff(z))]); cum /= cum[-1]
print("printed Lee22 1h at l=3000: z_med", np.interp(0.5, cum, z), " share from z>2:", 1 - np.interp(2, z, cum))
dC = W * np.trapz(d["dn"] * (w2[:, None] * d["y_lee22p"][:, :, i]) ** 2, lnM, axis=1)
cum = np.concatenate([[0], np.cumsum(0.5 * (dC[1:] + dC[:-1]) * np.diff(z))]); cum /= cum[-1]
print("corrected Lee22 1h at l=3000: z_med", np.interp(0.5, cum, z), " share from z>2:", 1 - np.interp(2, z, cum))
