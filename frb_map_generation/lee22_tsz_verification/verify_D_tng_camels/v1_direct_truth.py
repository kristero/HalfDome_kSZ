#!/usr/bin/env python
"""Independent re-derivation (verifier) of workstream D key numbers from the RAW-snapshot direct profile sums
(direct/direct_CV_<b>_<new>.npz; per-halo per-x-bin SUMS of N_e, P_e V, P_th V, M_gas, written on idark).

Own units / conventions (do not use camels_common / profile_tools):
  * header h, Om, OL, Ob, a, z from each npz.  f_b = Ob/Om.
  * M200c [Msun] = Group_M_Crit200 * 1e10 / h ; R200c,phys [cm] = Group_R_Crit200 [ckpc/h] * a / h * kpc
  * rho_crit(z) [g/cm^3] = 3 H0^2 E^2 / (8 pi G), H0 = 100 h km/s/Mpc  (PHYSICAL)
  * n200 = 200 rho_crit f_b / (X_H m_p) ;  P200 = 200 G M200 rho_crit f_b / (2 R200,phys) with SUBFIND R200 (not recomputed)
  * shell volumes PHYSICAL from x_edges * R200,phys
  * Lee22 printed (Table 7/8 no-c) evaluated at M_h = M200c[Msun] * 0.6774 (TNG h), physical x; model shell-averaged
    over each x bin (volume weighted, 8 sub-points).
Stacks: ratio of halo means (sum data / sum model) per x bin; amplitude = geometric mean over bins with centres in
0.04-1.34.  Also: Lee22-bookkeeping emulation / truth, Yhat, f_gas, f_e.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM  # validated Lee22/B12 transcription (formulas only)

G = 6.6743e-8
MP = 1.67262192e-24
MSUN = 1.98847e33
KPC = 3.08567758e21
XH = 0.76
MUE = 2 / (1 + XH)
DDIR = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/direct"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_D_tng_camels"
MB = [(12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]


def shell_avg(fun, xlo, xhi, n=8):
    """volume-weighted average of fun(x) over [xlo, xhi] (arrays), midpoint rule in x^3"""
    u = (np.arange(n) + 0.5) / n
    acc = 0.0
    for uu in u:
        x = (xlo ** 3 + uu * (xhi ** 3 - xlo ** 3)) ** (1 / 3)
        acc = acc + fun(x)
    return acc / n


recs = []
for fn in sorted(glob.glob(os.path.join(DDIR, "direct_CV_*_*.npz"))):
    d = np.load(fn)
    box = int(os.path.basename(fn).split("_")[2])
    new = int(os.path.basename(fn).split("_")[3][:3])
    h, a, z, Om, OL, Ob = (float(d[k]) for k in ("h", "a", "z", "Om", "OL", "Ob"))
    fb = Ob / Om
    H0 = 100 * h * 1e5 / (1e3 * KPC)
    rhoc = 3 * H0 ** 2 * (Om * (1 + z) ** 3 + OL) / (8 * np.pi * G)
    n200 = 200 * rhoc * fb / (XH * MP)
    M = d["M200_code"] * 1e10 / h
    R = d["R200_code"] * a / h * KPC
    # consistency: SUBFIND R200 vs rho_crit definition
    Rdef = (3 * M * MSUN / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
    P200 = 200 * G * M * MSUN * rhoc * fb / (2 * R)
    xe = d["x_edges"]
    xlo, xhi = xe[1:-1], xe[2:]          # skip centre sphere bin 0
    xc = np.sqrt(xlo * xhi)
    for j in range(len(M)):
        if np.log10(M[j]) < 12.5:
            continue
        vol = 4 * np.pi / 3 * (R[j] ** 3) * (xhi ** 3 - xlo ** 3)
        ne = d["true_Ne"][j, 1:] / vol
        pe = d["true_PeV"][j, 1:] / vol
        pth = d["true_PthV"][j, 1:] / vol
        rho = d["true_Mgas"][j, 1:] * MSUN / vol
        mh = M[j] * 0.6774
        mn = shell_avg(lambda x: LM.lee_nbar(x, mh, z, "noc"), xlo, xhi)
        mp = shell_avg(lambda x: LM.lee_Pbar(x, mh, z, "noc"), xlo, xhi)
        mb12 = shell_avg(lambda x: LM.b12_Pe_over_P200(x, M[j], z), xlo, xhi)
        # emulation (Lee22 bookkeeping): code-unit sums / comoving shells / code normalisation
        en = d["lee_ne"][j, 1:] / float(d["lee_n200_code"])
        ep = d["lee_pe"][j, 1:] / d["lee_P200_code"][j]
        exc = np.sqrt(d["lee_xlo"][j, 1:] * d["lee_xhi"][j, 1:])
        # Yhat and f_gas inside R200c
        Yhat = d["in200_PeV"][j] / (P200[j] * 4 * np.pi / 3 * R[j] ** 3)
        Yhat_th = 0.5176 * d["in200_PthV"][j] / (P200[j] * 4 * np.pi / 3 * R[j] ** 3)
        fgas = d["in200_Mgas"][j] / (fb * M[j])
        fe = d["in200_Ne"][j] * MUE * MP / MSUN / (fb * M[j])
        # models integrated: 3 int_0^1 f x^2 dx
        xx = np.logspace(-5, 0, 4000)
        Yl = 3 * np.trapz(LM.lee_Pbar(xx, mh, z, "noc") * xx ** 2, xx)
        Yb = 3 * np.trapz(LM.b12_Pe_over_P200(xx, M[j], z) * xx ** 2, xx)
        fl = 3 * np.trapz(LM.lee_nbar(xx, mh, z, "noc") * xx ** 2, xx) * MUE / XH
        recs.append(dict(box=box, new=new, z=z, M=M[j], Rratio=R[j] / Rdef[j], ne=ne / n200, pe=pe / P200[j],
                         pth=0.5176 * pth / P200[j], nfull=rho / (MUE * MP) / n200, mn=mn, mp=mp, mb12=mb12,
                         en=np.interp(np.log(xc), np.log(exc), en), ep=np.interp(np.log(xc), np.log(exc), ep),
                         Yhat=Yhat, Yhat_th=Yhat_th, fgas=fgas, fe=fe, Yl=Yl, Yb=Yb, fl=fl))

xc = np.sqrt(np.load(glob.glob(os.path.join(DDIR, "*.npz"))[0])["x_edges"][1:-1] *
             np.load(glob.glob(os.path.join(DDIR, "*.npz"))[0])["x_edges"][2:])
AMP = (xc >= 0.04) & (xc <= 1.34)
Q = [np.argmin(np.abs(np.log(xc / q))) for q in (0.1, 0.3, 0.5, 1.0)]
lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


say(f"quote bins at x = {', '.join(f'{xc[i]:.3f}' for i in Q)}")
say(f"SUBFIND R200 / R200(rho_crit def): min {min(r['Rratio'] for r in recs):.5f} max {max(r['Rratio'] for r in recs):.5f}")
news = sorted(set(r["new"] for r in recs), reverse=True)
rows = ["new,z,logM_lo,logM_hi,boxes,nhalo,amp_Pe_over_lee_printed,amp_ne_over_lee_printed,Pe_truth_over_printed_1pz2,"
        "Fn_printed_1pz3_over_truth,amp_Pe_over_B12,emul_over_true_n,emul_over_true_P,a3,a2,Yhat,Yhat_lee_printed,Yhat_b12,"
        "fgas,fe,fe_lee_printed,Pe_over_0.5176Pth_x0.3"]
for excl in (None, 2):
    say(f"\n==== direct truth, boxes {'0-3' if excl is None else '0,1,3 (CV_2 excluded)'} ====")
    for new in news:
        for lo, hi in MB:
            S = [r for r in recs if r["new"] == new and lo <= np.log10(r["M"]) < hi and r["box"] != excl]
            if len(S) < 2:
                continue
            z = np.mean([r["z"] for r in S])

            def st(k):
                return np.nanmean(np.array([r[k] for r in S]), 0)
            rp = st("pe") / st("mp")
            rn = st("ne") / st("mn")
            rb = st("pe") / st("mb12")
            ampP = np.exp(np.nanmean(np.log(rp[AMP])))
            ampN = np.exp(np.nanmean(np.log(rn[AMP])))
            ampB = np.exp(np.nanmean(np.log(rb[AMP])))
            sl = (xc >= 0.1) & (xc <= 1.5)
            emn = np.nanmean((st("en") / st("ne"))[sl])
            emp = np.nanmean((st("ep") / st("pe"))[sl])
            aa = 1 / (1 + z)
            Yh, Yl, Yb = st("Yhat"), st("Yl"), st("Yb")
            fg, fe, fl = st("fgas"), st("fe"), st("fl")
            pr = st("pe")[Q[1]] / st("pth")[Q[1]]
            nb = len(set(r["box"] for r in S))
            say(f"z={z:.3f} logM {lo}-{hi}: {len(S)} halos/{nb} boxes | P_e/P200 / Lee22 printed amp={ampP:.3f} "
                f"[x=0.1,0.3,0.5,1: {', '.join(f'{rp[i]:.3f}' for i in Q)}]; /(printed(1+z)^2)={ampP * aa ** 2:.3f}; /B12 amp={ampB:.3f} | "
                f"n_e/n200 / printed amp={ampN:.4f}, F_n=printed(1+z)^3/truth={1 / (ampN * aa ** 3):.2f} | emul/true n {emn:.4f} "
                f"(a^3 {aa ** 3:.4f}) P {emp:.4f} (a^2 {aa ** 2:.4f}) | Yhat {Yh:.3f} lee {Yl:.3f} B12 {Yb:.3f} | f_gas {fg:.3f} "
                f"f_e {fe:.3f} lee f_e {fl:.2f} | Pe/(0.5176Pth) {pr:.4f}")
            if excl is None:
                rows.append(f"{new},{z:.4f},{lo},{hi},{nb},{len(S)},{ampP:.4f},{ampN:.5f},{ampP * aa ** 2:.4f},"
                            f"{1 / (ampN * aa ** 3):.3f},{ampB:.4f},{emn:.4f},{emp:.4f},{aa ** 3:.4f},{aa ** 2:.4f},{Yh:.4f},"
                            f"{Yl:.4f},{Yb:.4f},{fg:.4f},{fe:.4f},{fl:.3f},{pr:.4f}")
    # power-law fits to the direct amplitudes
    for lo, hi in MB[:3]:
        zz, rp, rn = [], [], []
        for new in news:
            S = [r for r in recs if r["new"] == new and lo <= np.log10(r["M"]) < hi and r["box"] != excl]
            if len(S) < 3:
                continue
            st = lambda k: np.nanmean(np.array([r[k] for r in S]), 0)  # noqa: E731
            zz.append(np.mean([r["z"] for r in S]))
            rp.append(np.exp(np.nanmean(np.log((st("pe") / st("mp"))[AMP]))))
            rn.append(np.exp(np.nanmean(np.log((st("ne") / st("mn"))[AMP]))))
        if len(zz) >= 3:
            zz = np.array(zz)
            kp, cp = np.polyfit(np.log10(1 + zz), np.log10(rp), 1)
            kn, cn = np.polyfit(np.log10(1 + zz), np.log10(rn), 1)
            say(f"  fit logM {lo}-{hi} ({len(zz)} z): P_e/printed = {10 ** cp:.3f} (1+z)^{kp:.2f};  n_e/printed = {10 ** cn:.3f} (1+z)^{kn:.2f}")

with open(os.path.join(OUT, "v1_direct_truth.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
with open(os.path.join(OUT, "v1_direct_truth.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
