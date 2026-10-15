#!/usr/bin/env python
"""How much would a periodic-boundary error have changed OUR CAMELS numbers (F_n, F_P, redshift slopes k)?

Re-implements, read-only, the amplitude / F / k pipeline of lee22_tsz_verification/D_tng_camels
(build_halo_tables.py -> analyze_stacks.py) and runs it twice:
  baseline : the production (periodic) profiles, must reproduce analysis_summary.txt;
  nowrap   : every halo's binned gas sums multiplied by f_in(shell) (geom.py), the fraction of the shell inside the
             primary 25 Mpc/h box, i.e. what a code without minimum-image / ghost particles would have measured for an
             isotropic halo (surrogate validated at the particle level by camels_wrap_particle_test.py).
Also tabulates how many halos have their x R200c sphere crossing a box face.

Positions: precomputed (27 boxes) use GroupCM (the only position in the illstack files; median offset from GroupPos
0.08 R200c), direct (4 boxes) use GroupPos.

Output: OUT/camels_boundary_sensitivity.json, OUT/camels_face_fractions.csv, OUT/camels_boundary_sensitivity.txt
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python \
     camels_boundary_sensitivity.py
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/D_tng_camels")
import geom  # noqa: E402
import camels_common as cc  # noqa: E402  (read-only use)
import profile_tools as pt  # noqa: E402  (read-only use)

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007"
os.makedirs(OUT, exist_ok=True)
LBOX = 25000.0                   # ckpc/h (CAMELS; illstack hard-codes box = 25000.)
XG = cc.x_grid(31, 0.03, 3.0)
LOGM_MIN = 12.5
C = cc.CAMELS
AMP_SL = (XG >= 0.04) & (XG <= 1.34)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


# ------------------------------------------------------------------ precomputed (illstack) 27-box halo table, 2 modes
def precomputed_tables():
    q = np.exp(np.log(1e6) / 25)
    bs = np.log(1e6) / 25
    edges_code = 0.1 * np.exp(bs * np.arange(26))                         # ckpc/h
    rec = {k: [] for k in ("box", "old", "z", "M200", "R200c", "dmin", "nbar_b", "Pbar_b", "nbar_n", "Pbar_n")}
    for box in range(27):
        for old in range(10, 34):
            fn = os.path.join(cc.PROFDIR, f"CV_{box}", f"IllustrisTNG_CV_{box}_{old:03d}.hdf5")
            if not os.path.exists(fn):
                continue
            pc = cc.load_precomputed(box, old)
            with h5py.File(fn, "r") as f:
                cm = np.stack([f["GroupCMx"][:], f["GroupCMy"][:], f["GroupCMz"][:]], 1).astype(np.float64)
                Rcode = f["Group_R_Crit200"][:].astype(np.float64)
            z = pc["z"]
            sel = np.where(np.log10(pc["M200"]) >= LOGM_MIN)[0]
            for k in sel:
                M, R = pc["M200"][k], pc["R200"][k]
                xc = pc["r_kpc"] / R
                nb = cc.ne_n200_full_ion(pc["rho"][k], z)
                Pb = cc.PE_OVER_PTH * pc["Pth"][k] / C.P200(M, z)
                ok = (pc["ncell"][k] > 0) & (pc["rho"][k] > 0)
                fin = geom.f_in_shells(cm[k], edges_code, LBOX)
                for tag, fac in (("b", np.ones_like(fin)), ("n", fin)):
                    rec["nbar_" + tag].append(pt.to_grid(xc, pt.point_values(xc, nb * fac, q, ok), XG))
                    rec["Pbar_" + tag].append(pt.to_grid(xc, pt.point_values(xc, Pb * fac, q, ok & (Pb > 0)), XG))
                for key, v in (("box", box), ("old", old), ("z", z), ("M200", M), ("R200c", Rcode[k]),
                               ("dmin", np.min(np.minimum(cm[k], LBOX - cm[k])))):
                    rec[key].append(v)
        print(f"box {box}: {len(rec['M200'])} halo-snapshots", flush=True)
    return {k: np.array(v) for k, v in rec.items()}


def direct_tables():
    import glob
    files = sorted(glob.glob(os.path.join(cc.DIRECTDIR, "direct_CV_*_*.npz")))
    rec = {k: [] for k in ("box", "new", "z", "M200", "R200c", "dmin", "nbar_b", "Pbar_b", "nbar_n", "Pbar_n",
                           "nfull_b", "nfull_n")}
    for fn in files:
        d = np.load(fn)
        b = int(os.path.basename(fn).split("_")[2])
        new = int(os.path.basename(fn).split("_")[3].split(".")[0])
        z, a, h = float(d["z"]), float(d["a"]), float(d["h"])
        M200 = d["M200_code"] * 1e10 / h
        R200 = d["R200_code"] * a / h
        xe = d["x_edges"]
        xc = np.sqrt(xe[2:] * xe[1:-1])
        q = xe[2] / xe[1]
        for j in np.where(np.log10(M200) >= LOGM_MIN)[0]:
            M, R = M200[j], R200[j]
            re = xe * R * cc.KPC
            vol = 4 * np.pi / 3 * (re[1:] ** 3 - re[:-1] ** 3)
            n200, P200 = C.n200(z), C.P200(M, z)
            fin = geom.f_in_shells(d["GroupPos"][j].astype(float), xe * d["R200_code"][j], LBOX)
            okn = d["true_n"][j] > 0
            prof = {"nbar": d["true_Ne"][j] / vol / n200, "Pbar": d["true_PeV"][j] / vol / P200,
                    "nfull": d["true_Mgas"][j] * cc.MSUN / (cc.MU_E * cc.M_P) / vol / n200}
            for tag, fac in (("b", np.ones_like(fin)), ("n", fin)):
                for key, v in prof.items():
                    v = v * fac
                    ok = okn[1:] & (v[1:] > 0)
                    rec[f"{key}_{tag}"].append(pt.to_grid(xc, pt.point_values(xc, v[1:], q, ok), XG))
            for key, v in (("box", b), ("new", new), ("z", z), ("M200", M), ("R200c", d["R200_code"][j]),
                           ("dmin", np.min(np.minimum(d["GroupPos"][j], LBOX - d["GroupPos"][j])))):
                rec[key].append(v)
    return {k: np.array(v) for k, v in rec.items()}


P = precomputed_tables()
D = direct_tables()
say(f"precomputed halo-snapshots (logM >= {LOGM_MIN}): {len(P['M200'])}; direct: {len(D['M200'])}")


def lee_models(M, z):
    n = np.array([cc.LM.lee_nbar(XG, M[k] * cc.H_TNG, z[k], "noc") for k in range(len(M))])
    p = np.array([cc.LM.lee_Pbar(XG, M[k] * cc.H_TNG, z[k], "noc") for k in range(len(M))])
    return n, p


PMn, PMp = lee_models(P["M200"], P["z"])
DMn, DMp = lee_models(D["M200"], D["z"])
LOGM_P = np.log10(P["M200"])
LOGM_D = np.log10(D["M200"])

# ElectronAbundance correction f_xe(x, z) exactly as analyze_stacks (from the direct table) -- the no-wrap factor
# cancels in this ratio (same cells), so it is evaluated per mode
Zd = np.array([np.round(np.mean(D["z"][D["new"] == n]), 3) for n in D["new"]])


def fxe_table(tag):
    out = {}
    for zz in sorted(set(Zd)):
        s = Zd == zz
        out[zz] = np.nanmean(D["nbar_" + tag][s], 0) / np.nanmean(D["nfull_" + tag][s], 0)
    return out


FXE = {t: fxe_table(t) for t in ("b", "n")}


def fxe_at(z, tag):
    keys = np.array(sorted(FXE[tag]))
    k = keys[np.argmin(np.abs(keys - z))]
    f = FXE[tag][k].copy()
    return np.where(np.isfinite(f), f, 1.0)


olds = sorted(set(P["old"]))
zs = np.array([float(np.mean(P["z"][P["old"] == o])) for o in olds])


def ratio_series(q, lo, hi, tag, box_excl=None):
    out = np.full(len(olds), np.nan)
    for i, o in enumerate(olds):
        s = (P["old"] == o) & (LOGM_P >= lo) & (LOGM_P < hi)
        if box_excl is not None:
            s &= P["box"] != box_excl
        if s.sum() < 3:
            continue
        if q == "n":
            f = np.array([fxe_at(z, tag) for z in P["z"][s]])
            data, mod = P["nbar_" + tag][s] * f, PMn[s]
        else:
            data, mod = P["Pbar_" + tag][s], PMp[s]
        r = np.nanmean(data, 0)[AMP_SL] / np.nanmean(mod, 0)[AMP_SL]
        out[i] = np.exp(np.nanmean(np.log(r)))
    return out


def fit_k(r, z):
    ok = np.isfinite(r) & (r > 0) & (z <= 2.1)
    if ok.sum() < 3:
        return np.nan, np.nan
    c = np.polyfit(np.log10(1 + z[ok]), np.log10(r[ok]), 1)
    return float(c[0]), float(10 ** c[1])


res = {"precomputed_27box": {}, "direct_4box": {}}
say("\n== precomputed 27-box pipeline (analyze_stacks.py 'redshift trend' and 'normalisation after M1'):")
say("   k = exponent of <data>/<Lee22 printed> ~ (1+z)^k, z<=2; F(z) = printed x (1+z)^p / truth at z = 0, 0.47, 1.05, 1.48, 2.00")
for q, p, name in (("n", 3, "n_e/n200 (F_n)"), ("P", 2, "P_e/P200 (F_P)")):
    for lo, hi in cc.MASS_BINS:
        row = {}
        for tag in ("b", "n"):
            r = ratio_series(q, lo, hi, tag)
            k, c0 = fit_k(r, zs)
            F = [1.0 / (r[olds.index(o)] / (1 + zs[olds.index(o)]) ** p) for o in cc.TARGETS.values()]
            row[tag] = dict(k=k, c0=c0, F=F)
        # jackknife scatter of k (baseline) for scale
        jk = [fit_k(ratio_series(q, lo, hi, "b", box_excl=b), zs)[0] for b in range(27)]
        ek = float(np.sqrt(26 / 27 * np.nansum((np.array(jk) - np.nanmean(jk)) ** 2)))
        dF = [row["n"]["F"][i] / row["b"]["F"][i] - 1 for i in range(5)]
        res["precomputed_27box"][f"{q}_{lo}"] = dict(baseline=row["b"], nowrap=row["n"], k_jk_err=ek, dF_frac=dF,
                                                     dk=row["n"]["k"] - row["b"]["k"])
        say(f"  {name:16s} logM {lo}-{hi}: baseline k = {row['b']['k']:.3f} (+-{ek:.3f} jk), F = " +
            ", ".join(f"{v:.3f}" for v in row["b"]["F"]) + f" | nowrap k = {row['n']['k']:.3f} (dk = "
            f"{row['n']['k'] - row['b']['k']:+.3f}), F/F_base - 1 = " + ", ".join(f"{v:+.4f}" for v in dF))

say("\n== direct 4-box pipeline (analyze_stacks.py 'cross-check with the DIRECT profiles only'):")
zd_list = sorted(set(Zd))
for q, p, name in (("n", 3, "n_e/n200"), ("P", 2, "P_e/P200")):
    for lo, hi in cc.MASS_BINS[:3]:
        row = {}
        for tag in ("b", "n"):
            rr = []
            for zz in zd_list:
                sd = (Zd == zz) & (LOGM_D >= lo) & (LOGM_D < hi)
                if sd.sum() < 3:
                    rr.append(np.nan)
                    continue
                data = D[("nbar_" if q == "n" else "Pbar_") + tag][sd]
                mod = (DMn if q == "n" else DMp)[sd]
                qq = np.nanmean(data, 0)[AMP_SL] / np.nanmean(mod, 0)[AMP_SL]
                rr.append(float(np.exp(np.nanmean(np.log(qq)))))
            rr = np.array(rr)
            k, c0 = fit_k(rr, np.array(zd_list))
            row[tag] = dict(ratio=rr.tolist(), k=k, c0=c0)
        res["direct_4box"][f"{q}_{lo}"] = dict(baseline=row["b"], nowrap=row["n"],
                                              dratio_frac=(np.array(row["n"]["ratio"]) / np.array(row["b"]["ratio"]) - 1).tolist())
        say(f"  {name} logM {lo}-{hi}: baseline ratio(z) = " + ", ".join(f"{v:.3f}" for v in row["b"]["ratio"]) +
            f" -> {row['b']['c0']:.3f} (1+z)^{row['b']['k']:.2f};  nowrap/baseline - 1 = " +
            ", ".join(f"{v:+.4f}" for v in res["direct_4box"][f"{q}_{lo}"]["dratio_frac"]) +
            f";  nowrap k = {row['n']['k']:.3f}")

# ------------------------------------------------------------------ face-crossing fractions
rows = ["sample,logM_lo,logM_hi,z_target,nhalo,frac_R200c,frac_1p34R200c,frac_2p5R200c,frac_3R200c"]
say("\n== fraction of halos whose x R200c sphere crosses a face of the 25 Mpc/h box (precomputed set, GroupCM):")
for zt, old in cc.TARGETS.items():
    for lo, hi in cc.MASS_BINS:
        s = (P["old"] == old) & (LOGM_P >= lo) & (LOGM_P < hi)
        if s.sum() == 0:
            continue
        fr = [float(np.mean(P["dmin"][s] < x * P["R200c"][s])) for x in (1.0, 1.34, 2.5, 3.0)]
        rows.append(f"precomputed,{lo},{hi},{zt},{s.sum()},{fr[0]:.4f},{fr[1]:.4f},{fr[2]:.4f},{fr[3]:.4f}")
        say(f"  z={zt:.1f} logM {lo}-{hi}: {s.sum():4d} halos; crossing at 1 / 1.34 / 2.5 / 3 R200c: " +
            " / ".join(f"{v:.3f}" for v in fr))
s_all = LOGM_P >= 13.5
say("  all z, logM >= 13.5: " + " / ".join(f"{np.mean(P['dmin'][s_all] < x * P['R200c'][s_all]):.3f}" for x in (1.0, 1.34, 2.5, 3.0)))
for zz in zd_list:
    s = Zd == zz
    fr = [float(np.mean(D["dmin"][s] < x * D["R200c"][s])) for x in (1.0, 1.34, 2.5, 3.0)]
    rows.append(f"direct,{LOGM_MIN},16,{zz},{s.sum()},{fr[0]:.4f},{fr[1]:.4f},{fr[2]:.4f},{fr[3]:.4f}")

with open(os.path.join(OUT, "camels_face_fractions.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
with open(os.path.join(OUT, "camels_boundary_sensitivity.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
with open(os.path.join(OUT, "camels_boundary_sensitivity.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
