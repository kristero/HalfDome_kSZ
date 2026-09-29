#!/usr/bin/env python
"""Verifier: mass dependence of f_xe = N_e(x_e) / N_e(full ionisation) in the direct raw-snapshot sums (all 4 boxes),
stacked per x bin, and its effect on the density amplitude over 0.04-1.34 (geometric mean of f_xe)."""
import glob, os
import numpy as np
DD = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/direct"
MUE, MP, MSUN = 2 / 1.76, 1.67262192e-24, 1.98847e33
for new in (90, 74, 60, 52):
    acc = {}
    for fn in glob.glob(os.path.join(DD, f"direct_CV_*_{new:03d}.npz")):
        d = np.load(fn)
        M = np.log10(d["M200_code"] * 1e10 / float(d["h"]))
        xe = d["x_edges"]; xc = np.sqrt(xe[1:-1] * xe[2:]); z = float(d["z"])
        vol = (xe[2:] ** 3 - xe[1:-1] ** 3)[None, :] * d["R200_code"][:, None] ** 3
        for lab, s in (("12.5-13.5", (M >= 12.5) & (M < 13.5)), ("13.5-14.5", (M >= 13.5) & (M < 14.5)), ("pooled>=12.5", M >= 12.5)):
            ne = d["true_Ne"][s, 1:] / vol[s]; nf = d["true_Mgas"][s, 1:] * MSUN / (MUE * MP) / vol[s]
            a = acc.setdefault(lab, [0, 0, 0]); a[0] = a[0] + ne.sum(0) * 0 + np.nansum(ne / (nf.mean(1)[:, None] + 1e-300) * 0, 0) + ne.sum(0); a[1] = a[1] + nf.sum(0); a[2] += s.sum()
    sl = (xc >= 0.04) & (xc <= 1.34)
    out = []
    for lab, (n1, n2, N) in acc.items():
        f = n1 / n2
        out.append(f"{lab} (N={N}): f_xe x=0.1 {np.interp(0.1, xc, f):.3f}, x=0.3 {np.interp(0.3, xc, f):.3f}, geo-mean 0.04-1.34 {np.exp(np.mean(np.log(f[sl]))):.3f}")
    print(f"z={z:.2f}: " + " | ".join(out))
