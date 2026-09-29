#!/usr/bin/env python
"""Verifier: independent 27-box stack of the CAMELS precomputed (illstack) profiles vs printed Lee22, with a different
bin-average treatment from the workstream: per halo and per coarse comoving bin, ratio R = data_shell / <model>_shell
(model volume-averaged over the same shell), log-interpolated in x; stacked data(x) = sum_i R_i(x) m_i(x), model sum_i m_i(x).

Units (own): r [ckpc/h] -> x = r / Group_R_Crit200 (both comoving); rho_phys = Profiles[0] * 1e10 Msun h^2 / (a^3 kpc^3);
P_th,phys = Profiles[1] * 1e10 (km/s -> cm/s)^2 * same;  P_e = 0.5176 P_th ; n_e,full = rho / (mu_e m_p);
n_e = n_e,full * f_xe(x, z) with f_xe measured here from the direct raw-snapshot sums (all halos >= 10^12.5 Msun).
CAMELS: Om 0.3, Ob 0.049, h 0.6711.  z from the illstack red_dict (differences to header z < 2e-3).
Output: v2_precomputed_trend.txt / .csv
"""
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM

G, MP, MSUN, KPC = 6.6743e-8, 1.67262192e-24, 1.98847e33, 3.08567758e21
XH, MUE = 0.76, 2 / 1.76
h, Om, Ob = 0.6711, 0.3, 0.049
fb = Ob / Om
PD = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/camels_profiles"
DD = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/direct"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_D_tng_camels"
ZOLD = {10: 2.00259, 11: 1.86243, 12: 1.72882, 13: 1.60144, 14: 1.48001, 15: 1.36424, 16: 1.25388, 17: 1.14868,
        18: 1.04838, 19: 0.95276, 20: 0.86161, 21: 0.77471, 22: 0.69187, 23: 0.61290, 24: 0.53761, 25: 0.46584,
        26: 0.39741, 27: 0.33218, 28: 0.27, 29: 0.21072, 30: 0.15420, 31: 0.10033, 32: 0.04896, 33: 0.0}
MB = [(12.5, 13.0), (13.0, 13.5), (13.5, 14.0)]
XG = np.logspace(np.log10(0.04), np.log10(1.34), 25)


def rhoc(z):
    H0 = 100 * h * 1e5 / (1e3 * KPC)
    return 3 * H0 ** 2 * (Om * (1 + z) ** 3 + 1 - Om) / (8 * np.pi * G)


def shell_avg(fun, xlo, xhi, n=12):
    u = (np.arange(n) + 0.5) / n
    return sum(fun((xlo ** 3 + uu * (xhi ** 3 - xlo ** 3)) ** (1 / 3)) for uu in u) / n


# ---- f_xe(x, z) from direct raw sums (my own): stacked N_e / (M_gas/(mu_e m_p)) in fine x bins
fxe = {}
for new in (90, 74, 60, 52, 44):
    num = den = 0
    for fn in glob.glob(os.path.join(DD, f"direct_CV_*_{new:03d}.npz")):
        d = np.load(fn)
        M = d["M200_code"] * 1e10 / float(d["h"])
        s = np.log10(M) >= 12.5
        R = d["R200_code"][s] * float(d["a"]) / float(d["h"]) * KPC
        xe = d["x_edges"]
        vol = 4 * np.pi / 3 * R[:, None] ** 3 * (xe[None, 2:] ** 3 - xe[None, 1:-1] ** 3)
        n200 = 200 * rhoc(float(d["z"])) * fb / (XH * MP)
        num = num + np.sum(d["true_Ne"][s, 1:] / vol / n200, 0)
        den = den + np.sum(d["true_Mgas"][s, 1:] * MSUN / (MUE * MP) / vol / n200, 0)
        xc = np.sqrt(xe[1:-1] * xe[2:])
        zz = float(d["z"])
    fxe[round(zz, 2)] = np.interp(np.log(XG), np.log(xc), num / den)


def fxe_at(z):
    k = min(fxe, key=lambda q: abs(q - z))
    return fxe[k]


bs = np.log(1e6) / 25
edges = 0.1 * np.exp(bs * np.arange(26))
rec = []
for box in range(27):
    for old, z in ZOLD.items():
        fn = os.path.join(PD, f"CV_{box}", f"IllustrisTNG_CV_{box}_{old:03d}.hdf5")
        with h5py.File(fn, "r") as f:
            Pr, n, Mc, Rc = f["Profiles"][:], f["n"][:], f["Group_M_Crit200"][:], f["Group_R_Crit200"][:]
        a = 1 / (1 + z)
        dens = 1e10 * MSUN * h ** 2 / (a ** 3 * KPC ** 3)
        M = Mc * 1e10 / h
        Rp = Rc * a / h * KPC
        n200 = 200 * rhoc(z) * fb / (XH * MP)
        for j in np.where(np.log10(M) >= 12.5)[0]:
            xlo, xhi = edges[:-1] / Rc[j], edges[1:] / Rc[j]
            ok = (n[0, j] >= 1) & (Pr[0, j] > 0) & (Pr[1, j] > 0) & (xhi > 0.02) & (xlo < 2.0)
            if ok.sum() < 3:
                continue
            P200 = 200 * G * M[j] * MSUN * rhoc(z) * fb / (2 * Rp[j])
            mh = M[j] * 0.6774
            pe = 0.5176 * Pr[1, j] * dens * 1e10 / P200
            nf = Pr[0, j] * dens / (MUE * MP) / n200
            mP = shell_avg(lambda x: LM.lee_Pbar(x, mh, z, "noc"), xlo, xhi)
            mN = shell_avg(lambda x: LM.lee_nbar(x, mh, z, "noc"), xlo, xhi)
            mB = shell_avg(lambda x: LM.b12_Pe_over_P200(x, M[j], z), xlo, xhi)
            xc = np.sqrt(xlo * xhi)[ok]

            def itp(r):
                return np.exp(np.interp(np.log(XG), np.log(xc), np.log(r[ok]), left=np.nan, right=np.nan))
            rec.append(dict(box=box, old=old, z=z, M=M[j], RP=itp(pe / mP), RN=itp(nf / mN), RB=itp(pe / mB),
                            mP=LM.lee_Pbar(XG, mh, z, "noc"), mN=LM.lee_nbar(XG, mh, z, "noc"),
                            mB=LM.b12_Pe_over_P200(XG, M[j], z)))
print(len(rec), "halo-snapshots")
lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


def amp(S, which, z=None, fx=False):
    R = np.array([r["R" + which] for r in S])
    m = np.array([r["m" + which] for r in S])
    data = np.nansum(R * m, 0)
    mod = np.nansum(np.where(np.isfinite(R), m, 0), 0)
    rr = data / mod
    if fx:
        rr = rr * fxe_at(z)
    return np.exp(np.nanmean(np.log(rr))), rr


rows = ["logM_lo,boxes,old,z,nhalo,amp_P,amp_P_over_1pz2,amp_n_fxe,Fn,amp_P_over_B12"]
for label, boxes in (("all 27", set(range(27))), ("no CV_2", set(range(27)) - {2}), ("CV_0,1,3", {0, 1, 3})):
    say(f"\n==== precomputed, boxes {label} ====")
    for lo, hi in MB:
        zz, aP, aN = [], [], []
        for old, z in ZOLD.items():
            S = [r for r in rec if r["old"] == old and lo <= np.log10(r["M"]) < hi and r["box"] in boxes]
            if len(S) < 3:
                continue
            p, rp = amp(S, "P")
            nn, rn = amp(S, "N", z, fx=True)
            b, rb = amp(S, "B")
            zz.append(z); aP.append(p); aN.append(nn)
            rows.append(f"{lo},{label},{old},{z},{len(S)},{p:.4f},{p / (1 + z) ** 2:.4f},{nn:.5f},{1 / (nn / (1 + z) ** 3):.3f},{b:.4f}")
            if old in (33, 25, 18, 14, 10):
                iq = [np.argmin(np.abs(XG - q)) for q in (0.1, 0.3, 0.5, 1.0)]
                say(f"  logM {lo}-{hi} z={z:.3f} N={len(S)}: P/printed amp {p:.3f}, /(printed(1+z)^2) {p / (1 + z) ** 2:.3f}, "
                    f"/B12 amp {b:.3f} [x=.1,.3,.5,1: {', '.join(f'{np.log10(rb[i]):+.3f}' for i in iq)} dex]; "
                    f"n/printed amp {nn:.4f} -> F_n {(1 + z) ** 3 / nn:.2f}")
        zz = np.array(zz)
        if len(zz) >= 3:
            kP, cP = np.polyfit(np.log10(1 + zz), np.log10(aP), 1)
            kN, cN = np.polyfit(np.log10(1 + zz), np.log10(aN), 1)
            say(f"  FIT logM {lo}-{hi} ({len(zz)} snaps, z<={zz.max():.2f}): P/printed = {10 ** cP:.3f} (1+z)^{kP:.2f}; "
                f"n/printed = {10 ** cN:.3f} (1+z)^{kN:.2f}")
say("\nf_xe used (x=0.1, 0.3, 1): " + "; ".join(f"z={k}: {np.interp(0.1, XG, v):.3f}, {np.interp(0.3, XG, v):.3f}, {np.interp(1.0, XG, v):.3f}"
                                              for k, v in sorted(fxe.items())))
with open(os.path.join(OUT, "v2_precomputed_trend.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "v2_precomputed_trend.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
