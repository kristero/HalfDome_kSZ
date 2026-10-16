"""Moment-observable assembly of Test J: per-bin moments of a definition / treatment / configuration, the fallback of
the configuration (j_fallback), and Test F's in-basis forms (f_basis.Basis, read-only, with its fallback attributes
replaced in this process).

For a configuration with calibrated bins c (profiles F_c,s, per-halo second moments <F_s G_t>_c) and fallback haloes:
  C^1h(ell)  = sum_c sum_st K_c[s, t, ell] <F_s F_t>_c + fallback 1h          (yy: P P, kSZ / DM field: E E, y x DM: P E)
  C^2h(ell)  = sum_z P_lin(k_ell, z)/V_z a_z [W_fb(z, ell) + sum_cs F_c,s W_c,s(z, ell)]^2
  mean DM    = sum_cs E_c,s m_c,s + fallback;   Var(DM) = sum_c sum_st Q_c[s, t] <E_s E_t>_c + fallback
(K, W, m, Q: Test F's kernels on Test H's fine basis; a_z = 1 for tSZ, sigma_v,lin^2/c^2 for the kSZ 2-halo term,
s_DM(z)^2 for the DM field; definitions in F_observables/f_basis.py and H_sensitivity/REPORT.md H.1).
"""
from __future__ import annotations

import j_common as JC
import j_fallback as JF
import numpy as np

FC, FB = JC.FC, JC.FB
SEL_KEY = {"a2ii": "E_a2ii", "hb": "E_hb", "all": "E_all"}
SCALAR = ("ymean", "dm_mean", "dm_var")
OBS_KEYS = ("yy1h", "yy2h", "yytot", "ymean", "ksz1h", "ksz2h", "ksztot", "dm_mean", "dm_var", "dmcl1h", "dmcl2h",
            "dmcltot", "ydm1h", "ydm2h", "ydmtot")


def moments(pr, sel="a2ii", cut=4.0, treat="T1", config="C14", sub82=False, ptreat="T1"):
    """Moments of the J bins [N_J, ...]: P (A2-ii), E (selection ``sel`` cut in 3D at ``cut``), SPP, SEE, SPE.
    Bins outside the configuration's support are zeroed (they are painted by the fallback instead)."""
    jb = JC.J_BINS
    ek = SEL_KEY[sel]
    P = np.array(pr[f"mean_P_a2ii_{ptreat}"][jb], copy=True)
    E = np.array(pr[f"mean_{ek}_{treat}"][jb], copy=True)
    SPP = np.array(pr[f"S_P_a2ii_{ptreat}"][jb], copy=True)
    SEE = np.array(pr[f"S_{ek}_{treat}"][jb], copy=True)
    SPE = np.array(pr[f"SPE_{ek}_{treat}"][jb], copy=True)
    if sub82 and 1 in jb:
        c = jb.index(1)
        P[c], E[c] = pr["sub82_mean_P_a2ii"], pr[f"sub82_mean_{ek}"]
        SPP[c], SEE[c], SPE[c] = pr["sub82_S_P_a2ii"], pr[f"sub82_S_{ek}"], pr[f"sub82_SPE_{ek}"]
    E = JC.cut_profile(E, cut)
    SEE = JC.cut_moment(SEE, cut)
    SPE = JC.cut_moment_side(SPE, cut, "E")
    keep = np.zeros(JC.N_J, bool)
    keep[[jb.index(b) for b in JC.SUPPORT_BINS[config]]] = True
    for arr in (P, E, SPP, SEE, SPE):
        arr[~keep] = 0.0
    return dict(P=P, E=E, SPP=SPP, SEE=SEE, SPE=SPE)


def evaluate(B, mom, config, cut):
    """truth (scatter-inclusive), truthmean (bin-mean profiles) and truthcal (calibrated haloes only)."""
    JF.install(B, config, cut)
    full = B.observables(**mom)
    mean = B.observables(mom["P"], mom["E"])
    cal = B.observables(**mom, cal_only=True)
    return {"truth": full, "truthmean": mean, "truthcal": cal}


def per_bin(B, mom):
    """Each calibrated bin's own 1-halo / linear terms (exact, additive over bins): [N_J(, ell)]."""
    P, E = B.pad(mom["P"]), B.pad(mom["E"])
    SPP, SEE, SPE = (B.pad2(mom[k]) for k in ("SPP", "SEE", "SPE"))
    return {"dm_mean": np.einsum("cs,cs->c", E, B.dm_mean),
            "ymean": np.einsum("cs,cs->c", P, B.mean_y),
            "dm_var": np.einsum("cst,cst->c", B.K_dmvar, SEE),
            "ksz1h": np.einsum("cstl,cst->cl", B.K_ksz1h, SEE, optimize=True),
            "dmcl1h": np.einsum("cstl,cst->cl", B.K_dm1h, SEE, optimize=True),
            "yy1h": np.einsum("cstl,cst->cl", B.K_yy1h, SPP, optimize=True),
            "ydm1h": np.einsum("cstl,cst->cl", B.K_ydm1h, SPE, optimize=True)}


def fallback_per_bin(B):
    """Fallback haloes' own 1-halo / linear terms per engine bin [16(, ell)] from the installed fallback pieces
    (Euler: sum_s J_lin[b, s] = the bin's term; sum_s J_quad[b, s] = 2 x the bin's term)."""
    fb = B.fb
    return {"dm_mean": fb["J_dm_mean"].sum(axis=1), "ymean": fb["J_mean_y"].sum(axis=1),
            "dm_var": fb["J_dm_var"].sum(axis=1) / 2, "ksz1h": fb["J_ksz1h"].sum(axis=1) / 2,
            "dmcl1h": fb["J_dm1h"].sum(axis=1) / 2, "yy1h": fb["J_yy1h"].sum(axis=1) / 2,
            "ydm1h": fb["J_ydm1h_P"].sum(axis=1)}


def scalars(o):
    out = {}
    for k, v in o.items():
        if np.ndim(v) == 0:
            out[k] = float(v)
        else:
            for e, i in zip(FC.SEL_ELL, FC.SEL_INDEX):
                out[f"{k}_{e}"] = float(v[i])
    return out
