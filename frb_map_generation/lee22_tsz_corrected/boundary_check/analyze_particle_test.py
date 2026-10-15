#!/usr/bin/env python
"""Summarise camels_wrap_particle_test.py outputs (CV_0, snapshots 090 / 044 = z 0 / 2):
 1. the 'wrap' recomputation reproduces the stored production direct profiles;
 2. 'nowrap'/'wrap' per shell versus the geometric surrogate f_in (validates camels_boundary_sensitivity.py);
 3. effect of a missing wrap on the stacked amplitudes <data>/<Lee22 printed> (geometric mean 0.04-1.34 R200c, the
    F_n / F_P estimator of analyze_stacks.py), on Y(<R200c) and N_e(<R200c), and on the Lee22-bookkeeping emulation.
Output: OUT/particle_test_summary.txt / .json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/D_tng_camels")
import camels_common as cc  # noqa: E402
import profile_tools as pt  # noqa: E402

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007"
XG = cc.x_grid(31, 0.03, 3.0)
AMP_SL = (XG >= 0.04) & (XG <= 1.34)
C = cc.CAMELS
lines, res = [], {}


def say(s=""):
    print(s, flush=True)
    lines.append(s)


for fn in sorted(glob.glob(os.path.join(OUT, "particle_test_CV_0_*.npz"))):
    d = np.load(fn)
    snap = os.path.basename(fn).split("_")[-1].split(".")[0]
    z, h = float(d["z"]), float(d["h"])
    a = 1 / (1 + z)
    rk = dict(zip(d["repro_keys"].tolist(), d["repro_vals"].tolist()))
    M = d["M200_code"] * 1e10 / h
    R = d["R200_code"] * a / h
    t = d["dmin_face_over_R200"]
    r = {"z": z, "n_halos": int(len(M)), "repro_wrap_vs_stored": rk, "n_raw_outside": int(d["n_raw_outside"])}
    say(f"== CV_0 snapshot {snap} (z = {z:.3f}): {len(M)} halos with M200c >= 1e12 Msun; raw cells outside [0, L): "
        f"{int(d['n_raw_outside'])}")
    say("   wrap recomputation vs stored production file, max |ratio - 1|: " +
        ", ".join(f"{k} {v:.1e}" for k, v in rk.items() if not k.endswith("mismatch")) +
        "; empty/non-empty bin mismatches: " + str(sum(int(v) for k, v in rk.items() if k.endswith("mismatch"))))
    say("   halos whose sphere crosses a face at 1 / 1.34 / 2.5 / 5 R200c: " +
        " / ".join(str(int(np.sum(t < x))) for x in (1.0, 1.34, 2.5, 5.0)))
    # 2. surrogate check, per shell, bins with >= 30 cells in the wrapped run
    xe = d["x_edges"]
    for key in ("Ne", "PeV"):
        w, n = d[f"true_{key}_wrap"], d[f"true_{key}_nowrap"]
        fin = d["finx_true"]
        m = (d["true_n_wrap"] >= 30) & (fin < 0.99) & (w > 0)
        rat = n[m] / w[m]
        dev = rat - fin[m]
        r[f"surrogate_{key}"] = dict(n_bins=int(m.sum()), mean_nowrap_over_wrap=float(rat.mean()), mean_fin=float(fin[m].mean()),
                                     mean_dev=float(dev.mean()), rms_dev=float(np.sqrt(np.mean(dev ** 2))))
        say(f"   shells crossing a face (f_in < 0.99, >= 30 cells), {key}: {m.sum()} bins; <nowrap/wrap> = {rat.mean():.3f}, "
            f"<f_in> = {fin[m].mean():.3f}; nowrap/wrap - f_in: mean {dev.mean():+.3f}, rms {np.sqrt(np.mean(dev ** 2)):.3f}")
    # Lee22-emulation shells
    for key in ("ne", "pe"):
        w, n, fin = d[f"lee_{key}_wrap"], d[f"lee_{key}_nowrap"], d["finx_lee"]
        m = (d["lee_npart_wrap"] >= 30) & (fin < 0.99) & (w > 0)
        rat = n[m] / w[m]
        r[f"surrogate_lee_{key}"] = dict(n_bins=int(m.sum()), mean_dev=float((rat - fin[m]).mean()),
                                         rms_dev=float(np.sqrt(np.mean((rat - fin[m]) ** 2))))
    # 3. stacked amplitudes
    xc = np.sqrt(xe[2:] * xe[1:-1])
    q = xe[2] / xe[1]
    lm = np.log10(M)
    for lo, hi in ((12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (12.5, 14.5)):
        s = np.where((lm >= lo) & (lm < hi))[0]
        if len(s) < 2:
            continue
        out = {}
        for mode in ("wrap", "nowrap", "surr"):
            nb, Pb = [], []
            for j in s:
                re = xe * R[j] * cc.KPC
                vol = 4 * np.pi / 3 * (re[1:] ** 3 - re[:-1] ** 3)
                src = "wrap" if mode == "surr" else mode
                fac = d["finx_true"][j] if mode == "surr" else 1.0
                ok = d[f"true_n_{src}"][j] > 0
                for arr, key, norm in ((nb, "Ne", C.n200(z)), (Pb, "PeV", C.P200(M[j], z))):
                    v = d[f"true_{key}_{src}"][j] * fac / vol / norm
                    okk = ok[1:] & (v[1:] > 0)
                    arr.append(pt.to_grid(xc, pt.point_values(xc, v[1:], q, okk), XG))
            nb, Pb = np.array(nb), np.array(Pb)
            mn = np.array([cc.LM.lee_nbar(XG, M[j] * cc.H_TNG, z, "noc") for j in s])
            mp = np.array([cc.LM.lee_Pbar(XG, M[j] * cc.H_TNG, z, "noc") for j in s])
            amp_n = float(np.exp(np.nanmean(np.log(np.nanmean(nb, 0)[AMP_SL] / np.nanmean(mn, 0)[AMP_SL]))))
            amp_p = float(np.exp(np.nanmean(np.log(np.nanmean(Pb, 0)[AMP_SL] / np.nanmean(mp, 0)[AMP_SL]))))
            src = "wrap" if mode == "surr" else mode
            fac = 1.0
            Y = float(np.sum(d[f"in200_PeV_{src}"][s]))
            Ne = float(np.sum(d[f"in200_Ne_{src}"][s]))
            out[mode] = dict(amp_n=amp_n, amp_P=amp_p, Y200_sum=Y, Ne200_sum=Ne)
        rr = dict(n_halos=int(len(s)), n_cross_1p34=int(np.sum(t[s] < 1.34)), n_cross_2p5=int(np.sum(t[s] < 2.5)),
                  amp_n_nowrap_over_wrap=out["nowrap"]["amp_n"] / out["wrap"]["amp_n"],
                  amp_P_nowrap_over_wrap=out["nowrap"]["amp_P"] / out["wrap"]["amp_P"],
                  amp_n_surrogate_over_wrap=out["surr"]["amp_n"] / out["wrap"]["amp_n"],
                  amp_P_surrogate_over_wrap=out["surr"]["amp_P"] / out["wrap"]["amp_P"],
                  Y200_nowrap_over_wrap=out["nowrap"]["Y200_sum"] / out["wrap"]["Y200_sum"],
                  Ne200_nowrap_over_wrap=out["nowrap"]["Ne200_sum"] / out["wrap"]["Ne200_sum"],
                  amp_n_wrap=out["wrap"]["amp_n"], amp_P_wrap=out["wrap"]["amp_P"])
        r[f"logM_{lo}_{hi}"] = rr
        say(f"   logM {lo}-{hi}: {len(s)} halos ({rr['n_cross_1p34']} cross at 1.34 R200c, {rr['n_cross_2p5']} at 2.5): "
            f"amplitude <data>/<Lee22> wrap n {out['wrap']['amp_n']:.3f} P {out['wrap']['amp_P']:.3f}; nowrap/wrap: n "
            f"{rr['amp_n_nowrap_over_wrap']:.4f}, P {rr['amp_P_nowrap_over_wrap']:.4f} (surrogate n "
            f"{rr['amp_n_surrogate_over_wrap']:.4f}, P {rr['amp_P_surrogate_over_wrap']:.4f}); "
            f"sum Y(<R200c) {rr['Y200_nowrap_over_wrap']:.4f}, sum N_e(<R200c) {rr['Ne200_nowrap_over_wrap']:.4f}")
    res[snap] = r

# redshift-slope change from the two particle-level redshifts (CV_0 only)
if "090" in res and "044" in res:
    for lo, hi in ((12.5, 13.0), (13.0, 13.5), (12.5, 14.5)):
        k = f"logM_{lo}_{hi}"
        if k in res["090"] and k in res["044"]:
            for qq in ("n", "P"):
                d0 = np.log(res["090"][k][f"amp_{qq}_nowrap_over_wrap"])
                d2 = np.log(res["044"][k][f"amp_{qq}_nowrap_over_wrap"])
                dk = (d2 - d0) / np.log(1 + res["044"]["z"])
                res.setdefault("dk_two_point", {})[f"{qq}_{lo}_{hi}"] = float(dk)
                say(f"two-point (z=0, z=2) slope change from a missing wrap, {qq} logM {lo}-{hi}: dk = {dk:+.4f}")
with open(os.path.join(OUT, "particle_test_summary.json"), "w") as f:
    json.dump(res, f, indent=1)
with open(os.path.join(OUT, "particle_test_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
