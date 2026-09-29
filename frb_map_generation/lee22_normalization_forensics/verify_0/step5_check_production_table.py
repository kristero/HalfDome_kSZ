#!/usr/bin/env python3
"""verify_0 / step 5: check the production HalfDome sphere-profile table (lee22_noconc_xgpnative) against an
independent Python evaluation of Lee22 eq. 9 x 0.6016316602 (chord inside the R200c sphere, observer frame),
and quantify the two other production deviations from the printed formula:
  (a) radiation in XGPaint's rho_crit(z) vs the printed rho_cr(z) = rho_cr0 [Om(1+z)^3 + OL];
  (b) the production shape clip (x'_c, beta' frozen above 10^14.8/h Msun).
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).parent))
from step1_params_from_tex import t8, bpl  # noqa: E402

ROOT = Path("/home/cbllover/HalfDome/frb_map_generation")
OUT = ROOT / "outputs/lee22_normalization_forensics_20260928/verify_0"
K = {k: float(v) for k, v in (l.strip().split("=") for l in open(OUT / "step2_constants.txt"))}
MSUN, G, MP, ME, MPC, PC_CM = K["M_sun_kg"], K["G_SI"], K["m_p_kg"], K["m_e_kg"], K["Mpc_m"], K["pc_cm"]
XH = 0.76
Oc, Ob, h = 0.261, 0.049, 0.68
Om = Oc + Ob; fb = Ob / Om
RATIO = 0.9 * XH * MP / (ME + 2 * MP / (1 + XH))
M2_TO_PC_CM3 = 1e-4 * PC_CM  # 1 m^-2 = 1e-4 cm^-2; DM[pc cm^-3] = N[cm^-2]/pc[cm]  -> factor 1e-4 / pc_cm
M2_TO_PC_CM3 = 1e-4 / PC_CM


def rho_crit(z, radiation):
    H0 = 100e3 * h / MPC
    Or = 0.0
    if radiation:
        Og = 4.48131e-7 * 2.7255 ** 4 / h ** 2
        Or = Og * (1 + 3.046 * (7 / 8) * (4 / 11) ** (4 / 3))
    return 3 * H0 ** 2 / (8 * np.pi * G) * (Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + (1 - Om - Or))


def params(M, z, clip):
    A0, am, _, az = t8["n_0"][:4]
    Mcut = 10 ** t8["log10Mcut_hinv"] / h
    n0 = A0 * (M / Mcut) ** am * (1 + z) ** az
    Ms = min(M, clip)
    r = Ms / Mcut
    A0x, amx, ampx, azx = t8["x'_c"]
    A0b, amb, ampb, azb = t8["beta'"]
    return n0, float(A0x * bpl(r, amx, ampx) * (1 + z) ** azx), float(A0b * bpl(r, amb, ampb) * (1 + z) ** azb)


def chord_mean(x, M, z, radiation, clip, factor):
    n0, xc, bp = params(M, z, clip)
    rc = rho_crit(z, radiation)
    n200 = 200 * rc * fb / (XH * MP)
    R200 = (3 * M * MSUN / (4 * np.pi * 200 * rc)) ** (1 / 3)
    L = np.sqrt(1 - x * x)
    f = lambda l: (np.hypot(x, l) / xc) ** -0.3 * (1 + np.hypot(x, l) / xc) ** (-bp)
    val, _ = quad(f, 0, L, epsabs=0, epsrel=1e-12, limit=200)
    dm = factor * n0 * n200 * 2 * val * R200 * M2_TO_PC_CM3 / (1 + z)
    return dm / (2 * L)


f = h5py.File(ROOT / "outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5", "r")
lm = f["log10_m200c_msun"][:]; zz = f["redshift"][:]; xx = f["x_r200c"][:]
tab = f["lee22_noconc_xgpnative/chord_mean_dm_pc_cm3"]
rng = np.random.default_rng(12345)
idx = [(int(rng.integers(0, 700)), int(rng.integers(0, 100)), int(rng.integers(0, 200))) for _ in range(300)]
idx += [(699, 99, 0), (0, 0, 199), (500, 50, 100), (690, 10, 150), (650, 90, 20)]
clip = 10 ** 14.8 / h
dev_rad, dev_print, clip_effects = [], [], []
for (j, i, k) in idx:
    M = 10 ** lm[j]; z = zz[i]; x = xx[k]
    t = tab[j, i, k]
    mine_rad = chord_mean(x, M, z, True, clip, RATIO)
    mine_print = chord_mean(x, M, z, False, clip, RATIO)
    dev_rad.append(t / mine_rad - 1)
    dev_print.append((z, t / mine_print - 1))
    if M > clip:
        noclip = chord_mean(x, M, z, True, np.inf, RATIO)
        clip_effects.append((lm[j], z, x, t / noclip))
dev_rad = np.array(dev_rad)
dp = np.array(dev_print)
res = dict(n_points=len(idx), table_over_indep_radiation_rho=dict(min=float(dev_rad.min()), max=float(dev_rad.max())),
           table_over_indep_printed_rho=dict(min=float(dp[:, 1].min()), max=float(dp[:, 1].max())),
           literal_equivalent_factor=RATIO)
# clip effect on a grid above the clip mass
ce = []
for lgM in (15.0, 15.3, 15.6, 15.9):
    for z in (0.1, 0.5, 0.9):
        for x in (0.05, 0.3, 0.7):
            M = 10 ** lgM
            ce.append(dict(log10M=lgM, z=z, x=x, clipped_over_unclipped=chord_mean(x, M, z, True, clip, 1) / chord_mean(x, M, z, True, np.inf, 1)))
res["shape_clip_effect_chord_mean"] = ce
res["random_nodes_above_clip"] = [dict(log10M=float(a), z=float(b), x=float(c), table_over_unclipped=float(d)) for a, b, c, d in clip_effects[:10]]
json.dump(res, open(OUT / "step5_production_table_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("shape_clip_effect_chord_mean", "random_nodes_above_clip")}, indent=1))
for c in ce:
    print(c)
print(res["random_nodes_above_clip"])
