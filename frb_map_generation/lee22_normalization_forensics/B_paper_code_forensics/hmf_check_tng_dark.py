#!/usr/bin/env python3
"""Check the redshift dependence of the Tinker08 systematic: TNG300-1-Dark full-sky light-cone halo
counts (FoF/SUBFIND M200c, from the existing tng300_lee22_1r200c_20260921 run) per z-slice versus the
Tinker08 prediction, and the same for the N_e-weighted (Lee22 literal) sum that enters <DM>."""
import sys, json
from pathlib import Path
import numpy as np, h5py
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "x"]
import zresolved_mean_dm_test as zt  # noqa
hm, C = zt.hm, zt.C
f = h5py.File(HERE.parents[1] / "outputs/tng300_lee22_1r200c_20260921/tng_lightcone.h5", "r")
m = f["mass_msun"][:]; z = f["redshift"][:]
lo, hi = 1.4762326542663123e13, 1e14
sel = (m >= lo) & (m < hi)
m, z = m[sel], z[sel]
edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
lm = np.linspace(np.log10(lo), np.log10(hi), 41)
# N_e weights for the lightcone halos (interpolate in log M on a grid per slice)
out = []
for a, b in zip(edges[:-1], edges[1:]):
    s = (z >= a) & (z < b)
    zz = np.linspace(a, b, 9)
    vol = []; nt = []; net = []
    for zi in zz:
        dn = hm.tinker08_dndlnm(C, 10 ** lm, zi)
        ne = np.array([zt.n_electrons_literal(10 ** x, zi) for x in lm])
        dV = 4 * np.pi * C.chi(zi)[0] ** 2 * hm.C_KMS / (100 * C.h * C.E(zi))   # Mpc^3 per unit z (full sky)
        nt.append(np.trapz(dn, lm * np.log(10)) * dV)
        net.append(np.trapz(dn * ne, lm * np.log(10)) * dV)
    N_t = np.trapz(nt, zz); NE_t = np.trapz(net, zz)
    zc = 0.5 * (a + b)
    ne_halo = np.array([zt.n_electrons_literal(mm, zc) for mm in np.geomspace(lo, hi, 41)])
    NE_lc = np.sum(np.interp(np.log(m[s]), np.log(np.geomspace(lo, hi, 41)), ne_halo))
    out.append(dict(z_lo=a, z_hi=b, N_lightcone=int(s.sum()), N_tinker=float(N_t), ratio_counts=float(s.sum() / N_t),
                    ratio_Ne_weighted=float(NE_lc / NE_t)))
    print(out[-1])
json.dump(out, open(zt.OUT / "hmf_check_tng_dark.json", "w"), indent=1)
