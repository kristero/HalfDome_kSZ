#!/usr/bin/env python
"""Impact of the group_particles low-face pair-dropping bug on Lee22-style stacked profiles (2026-10-08).

Inputs : impact/replica/replica_CV_<box>_<snap>.npz (camels_bug_replica.py; C++-verified, see cpp_vs_replica/*.json)
Outputs: impact/impact_summary.json, impact/impact_summary.txt, impact/tables/*.csv

(a) CAMELS (L = 25 Mpc/h), measured: stacked n_e and P_e profiles in Lee22's 128 shells, buggy / fixed per shell and
    integrated over 0.04-1.34 R200c, per redshift and mass bin, for the actual box faces (translation 0) and 16
    random periodic translations, plus the exact expectation over random translations (expL).
(b) TNG300 (L = 205 Mpc/h): the same expectation with the CAMELS gas but L_TNG300 (expT), compared with the isotropic
    analytic law; analytic expectation and realisation scatter per Lee22 mass bin and redshift with TNG300 R200c and
    halo counts from a Tinker08 (200c) mass function; propagated to the Lee22 GNFW parameters and their (1+z)
    exponents, to F_n, and to the mean DM per halo hit.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bugmodel as bm  # noqa: E402
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import lee22_models as LM  # noqa: E402

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/impact"
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
NB = bm.NB
XE = bm.x_edges_nominal()
XC = np.sqrt(np.maximum(XE[:-1], 0.03 / 1.035) * XE[1:])        # geometric bin centres (bin 0: nominal)
FIT = (XC >= 0.04) & (XC <= 1.34)                               # Lee22 fit range (bins 9..109)
LT = bm.L_TNG300
lines = []
J = {}


def say(s=""):
    print(s, flush=True)
    lines.append(s)


# ------------------------------------------------------------------ load
reps = {}
for f in sorted(glob.glob(os.path.join(OUT, "replica", "replica_CV_*_*.npz"))):
    tag = os.path.basename(f)[8:-4]               # CV_<box>_<snap>
    box, snap = int(tag.split("_")[1]), tag.split("_")[2]
    reps[(box, snap)] = dict(np.load(f))
SNAPS = sorted({s for _, s in reps}, reverse=True)
ZS = {s: float(np.mean([reps[k]["z"] for k in reps if k[1] == s])) for s in SNAPS}
say(f"replica files: {len(reps)} ({', '.join(f'CV_{b}_{s}' for b, s in sorted(reps))})")

# ------------------------------------------------------------------ C++ validation
val = []
for f in sorted(glob.glob(os.path.join(OUT, "cpp_vs_replica", "*.json"))):
    d = json.load(open(f))
    tag = os.path.basename(f)[:-5]
    row = {"run": tag}
    for v in ("buggy", "plain", "perfix", "naive"):
        if v in d:
            row[v] = (d[v]["groups_and_order_identical"], d[v]["pe"]["pairs_cpp"], d[v]["pe"]["pairs_replica"],
                      d[v]["pe"]["sum_abs_count_mismatch"] + d[v]["ne"]["sum_abs_count_mismatch"])
    lp = d.get("lost_pairs", {})
    row["lost_cpp"], row["lost_rep"], row["lost_groups"] = lp.get("cpp_total"), lp.get("replica_total"), lp.get("groups_with_loss_cpp")
    row["lost_max_group_diff"] = lp.get("max_abs_group_diff")
    row["buggy_gt_plain_shells"] = lp.get("cpp_buggy_minus_plain_negative_shells")
    for k in ("plain_vs_perfix", "plain_vs_naive"):
        if k in d:
            row[k] = (d[k]["sum_abs_count_diff"], d[k]["max_rel_pe"])
    val.append(row)
nrun = len(val)
tot_lost_cpp = int(sum(r["lost_cpp"] or 0 for r in val))
tot_lost_rep = int(sum(r["lost_rep"] or 0 for r in val))
mism = int(sum(r["buggy"][3] + r["plain"][3] for r in val if "buggy" in r and "plain" in r))
ident = all(r["buggy"][0] and r["plain"][0] for r in val if "buggy" in r)
pp = [r["plain_vs_perfix"] for r in val if "plain_vs_perfix" in r]
pn = [r["plain_vs_naive"] for r in val if "plain_vs_naive" in r]
say("\n== C++ group_particles (c3fe9b8 y_prof/ne_prof) vs Python replica on the real CAMELS snapshots")
say(f"   {nrun} C++ runs (box/snapshot/translation); groups and order identical: {ident}; pairs lost (buggy vs plain) "
    f"C++ {tot_lost_cpp} vs replica {tot_lost_rep}; max per-group difference "
    f"{max(r['lost_max_group_diff'] or 0 for r in val)}; shells where buggy > plain: {sum(r['buggy_gt_plain_shells'] or 0 for r in val)}")
m0 = [r["buggy"][3] + r["plain"][3] for r in val if r["run"].endswith("_s0") and "buggy" in r]
ms = [r["buggy"][3] + r["plain"][3] for r in val if not r["run"].endswith("_s0") and "buggy" in r]
say(f"   |shell-count differences| C++ vs replica (bin-edge float flips; buggy+plain, n_e+P_e programs): translation 0: "
    f"{min(m0)}-{max(m0)} per run ({len(m0)} runs); translated boxes: {min(ms) if ms else 0}-{max(ms) if ms else 0} per run "
    f"({len(ms)} runs; the replica bins the untranslated float32 distances, the C++ the re-rounded translated ones)")
say(f"   plain vs perfix: summed |count diff| {sum(a for a, _ in pp)}, max rel P_e diff {max([b for _, b in pp] + [0]):.1e}; "
    f"plain vs naive: {sum(a for a, _ in pn)}, {max([b for _, b in pn] + [0]):.1e}  ({len(pp)} runs)")
J["validation"] = {"runs": nrun, "groups_identical": ident, "lost_pairs_cpp": tot_lost_cpp, "lost_pairs_replica": tot_lost_rep,
                   "shell_count_mismatches": mism, "mismatch_range_s0": [int(min(m0)), int(max(m0))],
                   "mismatch_range_translated": [int(min(ms)) if ms else 0, int(max(ms)) if ms else 0], "plain_vs_perfix_count_diff": int(sum(a for a, _ in pp)),
                   "plain_vs_naive_count_diff": int(sum(a for a, _ in pn)), "per_run": val}


# ------------------------------------------------------------------ helpers
def logm(R):
    return np.log10(R["M200_code"] * 1e10)        # h^-1 Msun


def norm(R, q):
    """per-halo normalisation of the shell sums: profile = sum/vol, divided by n200 (common) or P200 (per halo)"""
    if q == "ne":
        return R["vol"]
    return R["vol"] * R["P200_code"][:, None]


def stack(keys, q, sel, which, s=None):
    """mean over the selected halos (all boxes of one snapshot) of sum/vol/norm; which in fix, bug, expL, expT"""
    num = []
    for k in keys:
        R = reps[k]
        m = sel(R)
        if not m.any():
            continue
        a = R[f"{which}_{q}"]
        if which == "bug":
            a = a[s]
        num.append(a[m] / norm(R, q)[m])
    if not num:
        return None, 0
    a = np.concatenate(num)
    return a.mean(0), len(a)


def integ(keys, q, sel, which, s=None):
    """halo-mean of the 0.04-1.34 R200c volume integral normalised by n200 R200^3 or P200 R200^3"""
    vals = []
    for k in keys:
        R = reps[k]
        m = sel(R)
        if not m.any():
            continue
        a = R[f"{which}_{q}"]
        if which == "bug":
            a = a[s]
        nrm = R["R200_code"] ** 3 * (1.0 if q == "ne" else R["P200_code"])
        vals.append(a[m][:, FIT].sum(1) / nrm[m])
    return np.concatenate(vals).mean() if vals else np.nan


def iso_kept_table(L):
    """interpolator for the isotropic expected kept fraction vs r/L"""
    t = np.linspace(0, 0.2, 2001)
    return t, bm.iso_expected_kept(t * L, L)


TI, KI = iso_kept_table(1.0)


def iso_kept(r, L):
    return np.interp(np.asarray(r) / L, TI, KI)


def iso_kept_bins(R200, L):
    """(nh, NB) shell-averaged isotropic expected kept fraction for halos with comoving R200 [ckpc/h]"""
    out = np.empty((len(R200), NB))
    for i, Rc in enumerate(R200):
        _, _, _, _, rlo, rhi = bm.lee22_bins(np.float32(Rc))
        out[i] = bm.shell_average(lambda r: iso_kept(r, L), rlo, rhi)
    return out


MBINS = [(12.0, 12.5), (12.5, 13.0), (13.0, 14.2)] + [(round(13.0 + 0.2 * i, 1), round(13.2 + 0.2 * i, 1)) for i in range(6)]

# ------------------------------------------------------------------ (a) CAMELS measured stacks
say("\n== (a) CAMELS-TNG CV (L = 25 Mpc/h), real snapshots: buggy/fixed of the stacked Lee22-style profiles")
say("   stack = halo mean of n_e/n200 or P_e/P200 per shell; 'integral' = halo mean of the 0.04-1.34 R200c volume integral")
say("   (N_e/(n200 R200^3), Y/(P200 R200^3)); s0 = actual box faces; shifts = 16 random periodic translations (min / mean / max);")
say("   expL = exact expectation over random translations of this box (same halos)")
rows = []
Ja = {}
xs_report = [0.1, 0.5, 1.0, 1.34, 2.0, 2.5]
ix_rep = [int(np.argmin(np.abs(XC - x))) for x in xs_report]
for s_ in SNAPS:
    keys = [k for k in reps if k[1] == s_]
    boxes = sorted(b for b, _ in keys)
    for lo, hi in MBINS:
        def sel(R, lo=lo, hi=hi):
            lm = logm(R)
            return (lm >= lo) & (lm < hi)
        nh = sum(int(sel(reps[k]).sum()) for k in keys)
        if nh == 0:
            continue
        for q in ("ne", "pe"):
            fix, _ = stack(keys, q, sel, "fix")
            NS = reps[keys[0]]["bug_ne"].shape[0]
            rat = np.array([stack(keys, q, sel, "bug", s)[0] / fix for s in range(NS)])        # (NS, NB)
            expL = stack(keys, q, sel, "expL")[0] / fix
            expT = stack(keys, q, sel, "expT")[0] / fix
            isoT = np.concatenate([reps[k][f"fix_{q}"][sel(reps[k])] / norm(reps[k], q)[sel(reps[k])]
                                   * iso_kept_bins(reps[k]["R200_code"][sel(reps[k])], LT) for k in keys]).mean(0) / fix
            I_fix = integ(keys, q, sel, "fix")
            I_rat = np.array([integ(keys, q, sel, "bug", s) / I_fix for s in range(NS)])
            I_expL = integ(keys, q, sel, "expL") / I_fix
            I_expT = integ(keys, q, sel, "expT") / I_fix
            amp = np.exp(np.nanmean(np.log(rat[:, FIT]), axis=1))
            nhit0 = sum(int(np.any(reps[k]["bug_n"][0][sel(reps[k])] != reps[k]["fix_n"][sel(reps[k])], axis=1).sum()) for k in keys)
            key = f"z{ZS[s_]:.2f}_m{lo:.1f}-{hi:.1f}_{q}"
            Ja[key] = {"z": ZS[s_], "boxes": boxes, "nhalo": nh, "halos_with_dropped_pairs_s0": nhit0,
                       "ratio_s0_at_x": dict(zip(map(str, xs_report), rat[0, ix_rep].round(5).tolist())),
                       "ratio_shifts_min_at_x": dict(zip(map(str, xs_report), rat[1:, ix_rep].min(0).round(5).tolist())),
                       "ratio_shifts_mean_at_x": dict(zip(map(str, xs_report), rat[1:, ix_rep].mean(0).round(5).tolist())),
                       "expL_at_x": dict(zip(map(str, xs_report), expL[ix_rep].round(5).tolist())),
                       "expT_at_x": dict(zip(map(str, xs_report), expT[ix_rep].round(6).tolist())),
                       "integral_0.04-1.34_s0": round(float(I_rat[0]), 5),
                       "integral_shifts_min_mean_max": [round(float(I_rat[1:].min()), 5), round(float(I_rat[1:].mean()), 5), round(float(I_rat[1:].max()), 5)],
                       "integral_expL": round(float(I_expL), 5), "integral_expT": round(float(I_expT), 6),
                       "amplitude_0.04-1.34_s0": round(float(amp[0]), 5),
                       "amplitude_shifts_min_mean_max": [round(float(amp[1:].min()), 5), round(float(amp[1:].mean()), 5), round(float(amp[1:].max()), 5)]}
            for k_ in range(NB):
                rows.append(f"{ZS[s_]:.3f},{lo},{hi},{q},{nh},{k_},{XE[k_]:.5f},{XE[k_ + 1]:.5f},{rat[0, k_]:.6f},"
                            f"{rat[1:, k_].min():.6f},{rat[1:, k_].mean():.6f},{rat[1:, k_].max():.6f},{expL[k_]:.6f},{expT[k_]:.7f},{isoT[k_]:.7f},"
                            f"{np.percentile(rat[1:, k_], 16):.6f},{np.percentile(rat[1:, k_], 84):.6f}")
            if q == "ne":
                say(f"  z={ZS[s_]:.2f} logM {lo:.1f}-{hi:.1f}: {nh} halos in boxes {boxes} ({nhit0} lose pairs at s0)")
            sv = Ja[key]
            say(f"     {q}: s0 ratio at x=0.1/0.5/1/1.34/2.5: " + "/".join(f"{rat[0, i]:.4f}" for i in ix_rep if XC[i] != XC[ix_rep[4]]) +
                f" | integral s0 {I_rat[0]:.4f}, shifts {I_rat[1:].min():.4f}/{I_rat[1:].mean():.4f}/{I_rat[1:].max():.4f}, expL {I_expL:.4f}, expT {I_expT:.5f}"
                f" | amplitude s0 {amp[0]:.4f}, shifts {amp[1:].min():.4f}/{amp[1:].mean():.4f}/{amp[1:].max():.4f}")
# realisation scatter over the 16 translations vs the independent-halo prediction sqrt(sum w^2 Var_h)/sum w,
# Var_h = 3 [r/(12L) - (r/4L)^2] for an isotropic shell of comoving radius r = x R200c (used for the TNG300 sigma below)
say("\n   realisation scatter over the 16 translations (std of the stacked loss) vs the independent-halo prediction:")
Jsc = {}
for s_ in SNAPS:
    keys = [k for k in reps if k[1] == s_]
    for lo, hi in ((12.0, 13.0), (13.0, 14.2)):
        def sel(R, lo=lo, hi=hi):
            lm = logm(R)
            return (lm >= lo) & (lm < hi)
        if sum(int(sel(reps[k]).sum()) for k in keys) == 0:
            continue
        for q in ("ne", "pe"):
            out_ = []
            for xq in (0.5, 1.34, 2.5):
                kq = int(np.argmin(np.abs(XC - xq)))
                W, V, Bs = [], [], []
                for k in keys:
                    R = reps[k]
                    m = sel(R)
                    if not m.any():
                        continue
                    w = R[f"fix_{q}"][m, kq] / norm(R, q)[m, kq]
                    r = XC[kq] * R["R200_code"][m]
                    L = float(R["L"])
                    W.append(w)
                    V.append(3 * (r / (12 * L) - (r / (4 * L)) ** 2))
                    Bs.append(R[f"bug_{q}"][1:, m, kq] / norm(R, q)[m, kq][None, :])
                W, V, Bs = np.concatenate(W), np.concatenate(V), np.concatenate(Bs, axis=1)
                loss_s = 1 - Bs.sum(1) / W.sum()
                pred = np.sqrt(np.sum(W ** 2 * V)) / W.sum()
                out_.append((xq, float(loss_s.std(ddof=1)), float(pred)))
            Jsc[f"z{ZS[s_]:.2f}_m{lo}-{hi}_{q}"] = out_
            say(f"     z={ZS[s_]:.2f} logM {lo}-{hi} {q}: " + ", ".join(f"x={a}: std {b * 100:.2f}% vs pred {c * 100:.2f}%" for a, b, c in out_))
J["scatter_check"] = Jsc
with open(os.path.join(TAB, "camels_stack_ratio_per_shell.csv"), "w") as f:
    f.write("z,logM_lo,logM_hi,q,nhalo,shell,x_lo,x_hi,ratio_s0,ratio_shift_min,ratio_shift_mean,ratio_shift_max,expL,expT,isoT,ratio_shift_p16,ratio_shift_p84\n")
    f.write("\n".join(rows) + "\n")
J["camels_stacks"] = Ja

# ------------------------------------------------------------------ CAMELS: shift of a (1+z)^k amplitude fit caused by the bug
say("\n   (1+z)^k exponent shift from the bug in CAMELS (amplitude = geometric mean of stacked buggy/fixed over 0.04-1.34,")
say("    fitted as (1+z)^dk over the snapshots present; s0 = actual faces, translations = same translation index at every z)")
Jdk = {}
for lo, hi in ((12.0, 12.5), (12.5, 13.0), (13.0, 14.2)):
    for q in ("ne", "pe"):
        zz, amps = [], []
        for s_ in SNAPS:
            a_ = Ja.get(f"z{ZS[s_]:.2f}_m{lo:.1f}-{hi:.1f}_{q}")
            if a_ is None:
                continue
            keys = [k for k in reps if k[1] == s_]

            def sel(R, lo=lo, hi=hi):
                lm = logm(R)
                return (lm >= lo) & (lm < hi)
            fix, _ = stack(keys, q, sel, "fix")
            NS = reps[keys[0]]["bug_ne"].shape[0]
            amps.append([np.exp(np.nanmean(np.log((stack(keys, q, sel, "bug", s_i)[0] / fix)[FIT]))) for s_i in range(NS)])
            zz.append(ZS[s_])
        if len(zz) < 2:
            continue
        amps = np.array(amps)            # (nz, NS)
        dks = np.array([np.polyfit(np.log1p(zz), np.log(amps[:, i]), 1)[0] for i in range(amps.shape[1])])
        Jdk[f"m{lo}-{hi}_{q}"] = {"z": zz, "amp_s0": amps[:, 0].round(5).tolist(), "dk_s0": float(dks[0]),
                                  "dk_translations_min_max": [float(dks[1:].min()), float(dks[1:].max())],
                                  "dk_translations_rms": float(np.sqrt(np.mean(dks[1:] ** 2)))}
        say(f"     logM {lo}-{hi} {q}: z = {', '.join(f'{v:.2f}' for v in zz)}: amplitude s0 = {', '.join(f'{v:.4f}' for v in amps[:, 0])}; "
            f"dk s0 = {dks[0]:+.4f}; translations {dks[1:].min():+.4f} .. {dks[1:].max():+.4f} (rms {np.sqrt(np.mean(dks[1:] ** 2)):.4f})")
J["camels_dk"] = Jdk

# ------------------------------------------------------------------ per-halo: realised loss vs isotropic cap geometry
say("\n== per halo and shell: realised loss 1 - bug/fix vs the isotropic cap prediction for the actual centre and faces")
pairs = []          # (pred, meas_ne, meas_pe, r/d)
DIRS_CAP = bm.fibonacci_dirs(4000)
for k, R in reps.items():
    L = float(R["L"])
    NS = R["bug_ne"].shape[0]
    for s in range(NS):
        for j in range(len(R["ID"])):
            c = R["gpos_shifted"][s, j]
            rhi = R["r_hi"][j]
            dl = c.min()
            if rhi[-1] <= dl:
                continue
            kk = np.where(R["r_hi"][j] > dl)[0]
            kk = kk[R["fix_n"][j, kk] >= 30]
            if len(kk) == 0:
                continue
            pred = 1 - bm.shell_average(lambda r: bm.iso_cap_kept(c, r, L, DIRS_CAP), R["r_lo"][j, kk], R["r_hi"][j, kk])
            mne = 1 - R["bug_ne"][s, j, kk] / R["fix_ne"][j, kk]
            mpe = 1 - np.where(R["fix_pe"][j, kk] > 0, R["bug_pe"][s, j, kk] / np.where(R["fix_pe"][j, kk] > 0, R["fix_pe"][j, kk], 1), 1)
            rd = np.sqrt(R["r_lo"][j, kk] * R["r_hi"][j, kk]) / max(dl, 1e-3)
            for a_ in zip(pred, mne, mpe, rd):
                pairs.append(a_)
pairs = np.array(pairs)
Jc = {}
if len(pairs):
    say(f"   {len(pairs)} (halo, shell, translation) entries with >= 30 cells and a shell reaching a low face")
    for lo, hi in ((1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)):
        m = (pairs[:, 3] >= lo) & (pairs[:, 3] < hi)
        if m.sum() < 5:
            continue
        Jc[f"r/d {lo}-{hi}"] = [int(m.sum()), float(pairs[m, 0].mean()), float(pairs[m, 1].mean()), float(pairs[m, 2].mean()),
                                float(np.std(pairs[m, 1] - pairs[m, 0])), float(np.std(pairs[m, 2] - pairs[m, 0]))]
        say(f"   r/d {lo:>4}-{hi:<5}: n={m.sum():5d}  predicted loss {pairs[m, 0].mean():.4f}  measured n_e {pairs[m, 1].mean():.4f}, "
            f"P_e {pairs[m, 2].mean():.4f}  (rms meas-pred: n_e {np.std(pairs[m, 1] - pairs[m, 0]):.3f}, P_e {np.std(pairs[m, 2] - pairs[m, 0]):.3f})")
J["cap_check"] = Jc

# ------------------------------------------------------------------ (b) TNG300 rescaling check with the CAMELS gas
say("\n== (b) CAMELS gas rescaled to the TNG300 box: expectation over translations with L = 205 Mpc/h (expT) vs the")
say("   isotropic analytic law for the same halos (E[kept] = <prod_i(1 - max(0,-r n_i)/L)>_directions ~ 1 - 3r/(4L))")
Jb = {}
for s_ in SNAPS:
    keys = [k for k in reps if k[1] == s_]
    for lo, hi in ((12.0, 13.0), (13.0, 14.2)):
        def sel(R, lo=lo, hi=hi):
            lm = logm(R)
            return (lm >= lo) & (lm < hi)
        nh = sum(int(sel(reps[k]).sum()) for k in keys)
        if nh == 0:
            continue
        for q in ("ne", "pe"):
            fixs, iso_s, expT_s, expL_s, isoL_s = [], [], [], [], []
            for k in keys:
                R = reps[k]
                m = sel(R)
                if not m.any():
                    continue
                f_ = R[f"fix_{q}"][m] / norm(R, q)[m]
                fixs.append(f_)
                iso_s.append(f_ * iso_kept_bins(R["R200_code"][m], LT))
                isoL_s.append(f_ * iso_kept_bins(R["R200_code"][m], float(R["L"])))
                expT_s.append(R[f"expT_{q}"][m] / norm(R, q)[m])
                expL_s.append(R[f"expL_{q}"][m] / norm(R, q)[m])
            F = np.concatenate(fixs).mean(0)
            lossT = 1 - np.concatenate(expT_s).mean(0) / F
            lossI = 1 - np.concatenate(iso_s).mean(0) / F
            lossL = 1 - np.concatenate(expL_s).mean(0) / F
            lossIL = 1 - np.concatenate(isoL_s).mean(0) / F
            sel_x = FIT & (XC >= 0.1)
            rT = float(np.sum(lossT[sel_x]) / np.sum(lossI[sel_x]))
            rL = float(np.sum(lossL[sel_x]) / np.sum(lossIL[sel_x]))
            scale = float(np.sum(lossL[sel_x]) / np.sum(lossT[sel_x]))
            key = f"z{ZS[s_]:.2f}_m{lo:.1f}-{hi:.1f}_{q}"
            Jb[key] = {"nhalo": nh, "loss_expT_at_x": dict(zip(map(str, xs_report), lossT[ix_rep].round(6).tolist())),
                       "loss_iso_TNG_at_x": dict(zip(map(str, xs_report), lossI[ix_rep].round(6).tolist())),
                       "expT_over_iso_0.1-1.34": round(rT, 4), "expL_over_isoL_0.1-1.34": round(rL, 4),
                       "loss_L25_over_loss_L205_0.1-1.34": round(scale, 3)}
            say(f"  z={ZS[s_]:.2f} logM {lo}-{hi} {q} ({nh} halos): loss with L_TNG at x=0.5/1/1.34/2.5: "
                + "/".join(f"{lossT[ix_rep[i]] * 100:.3f}%" for i in (1, 2, 3, 5))
                + f"; isotropic law: " + "/".join(f"{lossI[ix_rep[i]] * 100:.3f}%" for i in (1, 2, 3, 5))
                + f"; ratio (0.1-1.34) {rT:.3f} (L=25 Mpc/h: {rL:.3f}); loss(L=25)/loss(L=205) = {scale:.2f} (205/25 = 8.20)")
J["tng300_rescaling_check"] = Jb

# ------------------------------------------------------------------ TNG300 analytic per Lee22 bin
say("\n== TNG300 (L = 205 Mpc/h), analytic expected bias of the stacked mean profile (isotropic law, validated above)")
ZL = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]      # TNG snapshots 99 91 84 78 72 67 59 50 40 33
TNG = LM.TNG


def r200c_ckpch(mh, z):
    """comoving R200c [ckpc/h] for M200c [h^-1 Msun] in the TNG cosmology (rho_crit0 = 277.5 h^2 Msun/kpc^3)"""
    return (3 * mh / (4 * np.pi * 200 * 277.5 * TNG.E2(z))) ** (1 / 3) * (1 + z)


try:
    import hmf
    mf = hmf.MassFunction(hmf_model="Tinker08", mdef_model="SOCritical", mdef_params={"overdensity": 200},
                          cosmo_params={"Om0": 0.3089, "Ob0": 0.0486, "H0": 67.74}, sigma_8=0.8159, n=0.9667,
                          Mmin=12.5, Mmax=15.5, dlog10m=0.01, transfer_model="EH")
    HMF_OK = True
except Exception as e:      # pragma: no cover
    print("hmf unavailable:", e)
    HMF_OK = False
VOL = 205.0 ** 3
Lee_bins = [(round(13.0 + 0.2 * i, 1), round(13.2 + 0.2 * i, 1)) for i in range(6)]
trows = []
Jt = {}
xfit = np.exp(np.linspace(np.log(0.04), np.log(1.34), 27))      # Lee22's 27 fit bins (log spaced, assumed)
for z in ZL:
    if HMF_OK:
        mf.update(z=z)
        lm_grid, dndlog10m = np.log10(mf.m), mf.dndlog10m
    for lo, hi in Lee_bins:
        if HMF_OK:
            m = (lm_grid >= lo) & (lm_grid < hi)
            N = float(np.trapz(dndlog10m[m], lm_grid[m]) * VOL)
            Rbar = float(np.trapz(dndlog10m[m] * r200c_ckpch(10 ** lm_grid[m], z), lm_grid[m]) / np.trapz(dndlog10m[m], lm_grid[m]))
        else:
            N = np.nan
            Rbar = r200c_ckpch(10 ** (0.5 * (lo + hi)), z)
        _, _, _, _, rlo, rhi = bm.lee22_bins(np.float32(Rbar))
        loss = 1 - bm.shell_average(lambda r: iso_kept(r, LT), rlo, rhi)
        # realisation scatter of the stacked mean over N independent halo positions: Var_h ~ 3[r/(12L) - (r/4L)^2]
        rc = np.sqrt(np.maximum(rlo, 1e-6) * rhi)
        sig = np.sqrt(3 * (rc / (12 * LT) - (rc / (4 * LT)) ** 2) / max(N, 1))
        touched = 1 - (1 - np.minimum(1.34 * Rbar / LT, 1)) ** 3
        bfit = 1 - iso_kept(xfit * Rbar, LT)
        # bias of the 0.04-1.34 R200c volume integrals (N_e, Y) with the Lee22 model profile as weight
        xi = np.exp(np.linspace(np.log(0.04), np.log(1.34), 400))
        kept_i = iso_kept(xi * Rbar, LT)
        integ_bias = {}
        for q, parf in (("ne", LM.lee_n_noc_params), ("pe", LM.lee_P_noc_params)):
            A0_, xc_, be_ = parf(10 ** (0.5 * (lo + hi)), z)
            w = A0_ * LM.gnfw(xi, xc_, be_) * xi ** 3
            integ_bias[q] = float(np.trapz(w * kept_i, np.log(xi)) / np.trapz(w, np.log(xi)) - 1)
        Jt[f"z{z}_m{lo:.1f}"] = {"N_halos_TNG300_tinker08": round(N, 1), "R200c_mean_ckpch": round(Rbar, 1),
                                 "integral_0.04-1.34_bias_ne": round(integ_bias["ne"], 7),
                                 "integral_0.04-1.34_bias_pe": round(integ_bias["pe"], 7),
                                 "bias_at_x": dict(zip(map(str, xs_report), (-loss[ix_rep]).round(6).tolist())),
                                 "sigma_realisation_at_x": dict(zip(map(str, xs_report), sig[ix_rep].round(6).tolist())),
                                 "fraction_halos_low_face_within_1.34R200": round(float(touched), 5),
                                 "bias_mean_over_fit_range": round(float(-np.mean(bfit)), 6)}
        for k_ in range(NB):
            trows.append(f"{z},{lo},{hi},{N:.1f},{Rbar:.2f},{k_},{XE[k_]:.5f},{XE[k_ + 1]:.5f},{-loss[k_]:.7f},{sig[k_]:.7f}")
with open(os.path.join(TAB, "tng300_expected_bias_per_shell.csv"), "w") as f:
    f.write("z,logM_lo,logM_hi,N_halos,R200c_mean_ckpch,shell,x_lo,x_hi,expected_bias,sigma_realisation\n")
    f.write("\n".join(trows) + "\n")
say("   bias = <buggy>/<fixed> - 1 of the stacked mean; sigma = 1-sigma realisation scatter for N halos at random positions")
say("   z     logM bin    N(TNG300)  R200c[ckpc/h]  bias x=0.04   x=0.5     x=1.0     x=1.34    x=2.5   | sigma x=1.34 | f(low face<1.34R) | integral 0.04-1.34 n_e, P_e")
for z in ZL:
    for lo, hi in Lee_bins:
        t = Jt[f"z{z}_m{lo:.1f}"]
        b = t["bias_at_x"]
        b04 = -float(1 - bm.shell_average(lambda r: iso_kept(r, LT), *[a[[9]] for a in bm.lee22_bins(np.float32(t["R200c_mean_ckpch"]))[4:6]])[0])
        t["bias_at_x"]["0.04"] = round(b04, 7)
        if z in (0.0, 0.5, 1.0, 2.0):
            say(f"   {z:<4}  {lo:.1f}-{hi:.1f}  {t['N_halos_TNG300_tinker08']:9.0f}  {t['R200c_mean_ckpch']:9.0f}     "
                f"{b04 * 100:+.4f}%  {b['0.5'] * 100:+.3f}%  {b['1.0'] * 100:+.3f}%  {b['1.34'] * 100:+.3f}%  {b['2.5'] * 100:+.3f}% | "
                f"{t['sigma_realisation_at_x']['1.34'] * 100:.3f}%     | {t['fraction_halos_low_face_within_1.34R200'] * 100:.2f}%            | "
                f"{t['integral_0.04-1.34_bias_ne'] * 100:+.3f}%, {t['integral_0.04-1.34_bias_pe'] * 100:+.3f}%"
                + ("" if t["N_halos_TNG300_tinker08"] >= 30 else "   (N < 30: not in Lee22)"))
for lab, keys_ in (("all 10 z x 6 mass bins", list(Jt)), ("bins with N >= 30 (Lee22's criterion)", [k for k in Jt if Jt[k]["N_halos_TNG300_tinker08"] >= 30])):
    allb = np.array([[Jt[k]["bias_at_x"]["0.04"], Jt[k]["bias_at_x"]["1.34"], Jt[k]["bias_at_x"]["2.5"], Jt[k]["integral_0.04-1.34_bias_ne"],
                      Jt[k]["integral_0.04-1.34_bias_pe"], Jt[k]["sigma_realisation_at_x"]["1.34"], Jt[k]["bias_mean_over_fit_range"]] for k in keys_])
    say(f"   range over {lab} ({len(keys_)}): x=0.04 {allb[:, 0].max() * 100:+.4f}% .. {allb[:, 0].min() * 100:+.4f}%; "
        f"x=1.34 {allb[:, 1].max() * 100:+.3f}% .. {allb[:, 1].min() * 100:+.3f}%; x=2.5 {allb[:, 2].max() * 100:+.3f}% .. {allb[:, 2].min() * 100:+.3f}%; "
        f"integral n_e {allb[:, 3].max() * 100:+.3f}% .. {allb[:, 3].min() * 100:+.3f}%, P_e {allb[:, 4].max() * 100:+.3f}% .. {allb[:, 4].min() * 100:+.3f}%; "
        f"sigma(x=1.34) {allb[:, 5].min() * 100:.3f}% .. {allb[:, 5].max() * 100:.3f}%")
    J[f"tng300_range_{'all' if 'all' in lab else 'N30'}"] = {"x0.04": [float(allb[:, 0].min()), float(allb[:, 0].max())],
        "x1.34": [float(allb[:, 1].min()), float(allb[:, 1].max())], "x2.5": [float(allb[:, 2].min()), float(allb[:, 2].max())],
        "integral_ne": [float(allb[:, 3].min()), float(allb[:, 3].max())], "integral_pe": [float(allb[:, 4].min()), float(allb[:, 4].max())],
        "sigma_x1.34": [float(allb[:, 5].min()), float(allb[:, 5].max())]}
J["tng300_expected"] = Jt

# ------------------------------------------------------------------ propagation to the Lee22 GNFW fits
say("\n== propagation to Lee22's GNFW fits (derived): refit of n0, xc', beta' (gamma=-0.3, alpha=1) to Lee22 model x (1 - expected loss)")
say("   on 27 log-spaced points in 0.04-1.34 R200c (equal weights in ln); then the (1+z) exponents over the 10 snapshots")


def fit_gnfw(x, y, p0):
    def res(p):
        return np.log(np.exp(p[0]) * LM.gnfw(x, np.exp(p[1]), p[2])) - np.log(y)
    r = least_squares(res, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return r.x, float(np.max(np.abs(r.fun)))


Jp = {}
for q, parf in (("ne", LM.lee_n_noc_params), ("pe", LM.lee_P_noc_params)):
    dk = {"A0": [], "xc": [], "beta": []}
    drows = []
    for lo, hi in Lee_bins:
        mh = 10 ** (0.5 * (lo + hi))
        P_true, P_bug = [], []
        for z in ZL:
            A0, xc, beta = parf(mh, z)
            y = A0 * LM.gnfw(xfit, xc, beta)
            p_true, r_true = fit_gnfw(xfit, y, [np.log(A0), np.log(xc), beta])
            b = 1 - iso_kept(xfit * r200c_ckpch(mh, z), LT)
            p_bug, r_bug = fit_gnfw(xfit, y * (1 - b), p_true)
            P_true.append(p_true)
            P_bug.append(p_bug)
            drows.append((z, lo, p_bug[0] - p_true[0], p_bug[1] - p_true[1], (p_bug[2] - p_true[2]) / p_true[2], r_bug, float(b.max())))
        P_true, P_bug = np.array(P_true), np.array(P_bug)
        lz = np.log1p(np.array(ZL))
        for i, nm in enumerate(("A0", "xc", "beta")):
            yt = P_true[:, i] if nm != "beta" else np.log(P_true[:, i])
            yb = P_bug[:, i] if nm != "beta" else np.log(P_bug[:, i])
            if nm == "xc":
                pass
            kt = np.polyfit(lz, yt, 1)[0]
            kb = np.polyfit(lz, yb, 1)[0]
            dk[nm].append(kb - kt)
    d = np.array(drows)
    Jp[q] = {"dlnA0_range": [float(d[:, 2].min()), float(d[:, 2].max())], "dlnxc_range": [float(d[:, 3].min()), float(d[:, 3].max())],
             "dbeta_over_beta_range": [float(d[:, 4].min()), float(d[:, 4].max())], "max_abs_residual_ln": float(d[:, 5].max()),
             "dk_(1+z)_exponent_A0": [float(min(dk["A0"])), float(max(dk["A0"]))], "dk_xc": [float(min(dk["xc"])), float(max(dk["xc"]))],
             "dk_beta": [float(min(dk["beta"])), float(max(dk["beta"]))]}
    say(f"   {q}: d ln A0 {d[:, 2].min():+.5f}..{d[:, 2].max():+.5f}, d ln xc' {d[:, 3].min():+.5f}..{d[:, 3].max():+.5f}, "
        f"d beta'/beta' {d[:, 4].min():+.5f}..{d[:, 4].max():+.5f}, residual after refit <= {d[:, 5].max():.5f} (ln); "
        f"(1+z)-exponent shifts: A0 {min(dk['A0']):+.5f}..{max(dk['A0']):+.5f}, xc' {min(dk['xc']):+.5f}..{max(dk['xc']):+.5f}, "
        f"beta' {min(dk['beta']):+.5f}..{max(dk['beta']):+.5f}")
J["lee22_param_shifts"] = Jp

# ------------------------------------------------------------------ amplitude in (1+z): bias of k and of F_n, F_P
say("\n== redshift trend of the bias (geometric mean over 0.04-1.34 R200c, TNG300, analytic) and power-law exponent shift")
Jk = {}
for lo, hi in Lee_bins:
    mh = 10 ** (0.5 * (lo + hi))
    amp = np.array([np.exp(np.mean(np.log(iso_kept(xfit * r200c_ckpch(mh, z), LT)))) for z in ZL])
    k = np.polyfit(np.log1p(ZL), np.log(amp), 1)[0]
    Jk[f"m{lo:.1f}"] = {"amplitude_z": dict(zip(map(str, ZL), amp.round(6).tolist())), "dk": float(k),
                        "factor_z1": float(amp[ZL.index(1.0)] / amp[0])}
    say(f"   logM {lo:.1f}-{hi:.1f}: <buggy/fixed>_0.04-1.34 = {amp[0]:.5f} (z=0) .. {amp[ZL.index(1.0)]:.5f} (z=1) .. {amp[-1]:.5f} (z=2); "
        f"dk = {k:+.5f}; factor z=1 vs z=0: {amp[ZL.index(1.0)] / amp[0]:.5f}  (vs (1+z)^2 = 4, (1+z)^3 = 8)")
J["redshift_trend"] = Jk

# ------------------------------------------------------------------ mean DM per halo hit (Konietzka comparison)
say("\n== mean DM per halo hit (sphere-truncated, projected): N_e-weighted bias of the Lee22 density inside the truncation")
say("   (the painted profile is the Lee22 FIT: ratio of the GNFW refit to the biased stack over the refit to the unbiased")
say("    stack, N_e-weighted inside the truncation sphere; beyond 1.34 R200c this is the fits' extrapolation)")
Jd = {}
xx = np.exp(np.linspace(np.log(1e-3), np.log(4.0), 4000))
for lm_ in (13.0, 13.5, 14.0):
    mh = 10 ** lm_
    for z in (0.1, 0.5, 1.0, 2.0):
        A0, xc, beta = LM.lee_n_noc_params(mh, z)
        y = A0 * LM.gnfw(xfit, xc, beta)
        p_t, _ = fit_gnfw(xfit, y, [np.log(A0), np.log(xc), beta])
        p_b, _ = fit_gnfw(xfit, y * iso_kept(xfit * r200c_ckpch(mh, z), LT), p_t)
        n_t = np.exp(p_t[0]) * LM.gnfw(xx, np.exp(p_t[1]), p_t[2])
        n_b = np.exp(p_b[0]) * LM.gnfw(xx, np.exp(p_b[1]), p_b[2])
        for xt in (1.0, 2.0, 4.0):
            m = xx <= xt
            Jd[f"m{lm_}_z{z}_xt{xt}"] = float(np.trapz(n_b[m] * xx[m] ** 3, np.log(xx[m])) / np.trapz(n_t[m] * xx[m] ** 3, np.log(xx[m])) - 1)
dms = np.array(list(Jd.values()))
d1 = [v for k, v in Jd.items() if k.endswith("xt1.0")]
say(f"   logM 13-14, z 0.1-2, truncation 1-4 R200c: N_e (= mean DM per hit) of the biased fit / unbiased fit - 1 = "
    f"{dms.max() * 100:+.3f}% .. {dms.min() * 100:+.3f}%  (1 R200c: {max(d1) * 100:+.3f}% .. {min(d1) * 100:+.3f}%)")
J["dm_per_hit_bias"] = Jd

with open(os.path.join(OUT, "impact_summary.json"), "w") as f:
    json.dump(J, f, indent=1, default=float)
with open(os.path.join(OUT, "impact_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}/impact_summary.json, impact_summary.txt, tables/")
