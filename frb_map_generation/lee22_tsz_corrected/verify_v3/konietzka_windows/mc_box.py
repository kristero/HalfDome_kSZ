#!/usr/bin/env python3
"""Independent Monte Carlo of the 1e14-1e15 hit fraction over TNG300-size box realisations (Poisson + linear
cosmic variance with Tinker10 bias), plus hit expectation for [9e12,1e13) slices (c=9 test)."""
import json
import numpy as np
from scipy import stats
import hmf_lib as H

rng = np.random.default_rng(12345)
DC = 1.686


def tinker10_bias(M, z):
    Omz = H.Om * (1 + z) ** 3 / H.E(z) ** 2
    y = np.log10(200 / Omz)
    A = 1 + 0.24 * y * np.exp(-(4 / y) ** 4); a = 0.44 * y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4); c = 2.4
    nu = DC / (H.sigma_M(M) * H.growth(z))
    return 1 - A * nu ** a / (nu ** a + DC ** a) + B * nu ** b + C * nu ** c


R_eq = (3 * H.V_BOX / (4 * np.pi)) ** (1 / 3)
sig_box0 = H.norm * H.sigma_R(R_eq)[0]
out = {"sigma_lin_box_z0": float(sig_box0)}
lo, hi = 1e14, 1e15
snaps = {}
for z in H.SNAPS:
    lm = np.linspace(np.log(lo), np.log(hi), 300)
    M = np.exp(lm)
    dn = H.dndlnM_200c(M, z) * np.gradient(lm)
    snaps[z] = dict(N=dn.sum() * H.V_BOX * 0.944, p=dn / dn.sum(), area=np.pi * H.r200c_com(M, z) ** 2,
                    beff=float((dn * tinker10_bias(M, z)).sum() / dn.sum()), sig=sig_box0 * H.growth(z))
n_mc = 20000
for z_s in (0.1, 0.5, 1.0):
    ints = H.intervals(z_s, "z")
    delta = rng.standard_normal(n_mc)
    u = rng.random(n_mc)
    nb = np.zeros(n_mc)
    for z, a, b in ints:
        s = snaps[z]
        mean_n = s["N"] * np.clip(1 + s["beff"] * s["sig"] * delta, 0, None)
        k = stats.poisson.ppf(u, mean_n).astype(int)
        idx_all = rng.choice(len(s["p"]), size=k.sum(), p=s["p"])
        cs = np.concatenate([[0], np.cumsum(s["area"][idx_all])])
        ends = np.cumsum(k)
        tot = cs[ends] - cs[ends - k]
        nb += tot / H.V_BOX * (b - a)
    hit = 1 - np.exp(-nb)
    out[str(z_s)] = dict(median=float(np.median(hit)), mean=float(hit.mean()), p0005=float(np.quantile(hit, 5e-4)),
                         p9995=float(np.quantile(hit, 1 - 5e-4)), min=float(hit.min()), rel_std=float(hit.std() / hit.mean()),
                         beff_z0=snaps[0.0]["beff"])
    print(z_s, {kk: (round(100 * v, 3) if kk not in ("rel_std", "beff_z0") else round(v, 3)) for kk, v in out[str(z_s)].items()})
# c = 9 test: fraction of rays crossing a [9e12, 1e13) halo (x0.8 TNG abundance); the identity 10to14 = 10to12 + 13to14
# would fail on these rays if the windows were [lo, 9 hi)
for z_s in (0.1, 0.5, 1.0):
    nbs = sum(H.window(z, 9e12, 1e13, 50)[1] * (b - a) for z, a, b in H.intervals(z_s, "z"))
    out[f"hit_9e12_1e13_zs{z_s}"] = float(1 - np.exp(-0.8 * nbs))
    print("rays crossing a [9e12,1e13) halo, z_s =", z_s, f"{100 * (1 - np.exp(-0.8 * nbs)):.2f}%  -> {int(120000 * (1 - np.exp(-0.8 * nbs)))} rays")
json.dump(out, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v3_20261001/konietzka_windows/mc_box.json", "w"), indent=1)
