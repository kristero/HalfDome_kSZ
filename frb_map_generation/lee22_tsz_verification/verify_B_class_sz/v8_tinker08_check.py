"""Independent Tinker+08 M200c mass function (own implementation of Tinker et al. 2008 Table 2 + eq. 5-8, Delta_m =
200/Omega_m(z), linear interpolation of the parameters in log Delta) from CLASS-SZ's saved linear P(k) table, compared
with CLASS-SZ's own T08M200c dn/dlnM table (b12_paint_all.npz).
P_lin(k, z) is stored at k = (ell+1/2)/chi(z) for the 135 CLASS-SZ ells: each z slice is rescaled to z = 0 with my own
LCDM growth factor and the slices are merged into one P0(k)."""
import json
import numpy as np

CS = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/class_sz/"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/"
d = np.load(CS + "b12_paint_all.npz", allow_pickle=True)
z = d["tab_z"]; lnm = d["tab_lnm_msunh"]; dn_cs = d["tab_dndlnm_h3mpc3"]; pk = d["tab_pklin_mpch3"]; da = d["tab_da_mpc"]
cl = np.load(CS + "b12_paint_all.npz", allow_pickle=True)["ell"]
h = 0.68; Om = 0.31
chi_h = da * (1 + z) * h                                      # Mpc/h
def growth(zz):
    a = 1 / (1 + np.atleast_1d(zz)); out = []
    for ai in a:
        aa = np.linspace(1e-5, ai, 20000); E = np.sqrt(Om / aa ** 3 + 1 - Om)
        out.append(2.5 * Om * np.sqrt(Om / ai ** 3 + 1 - Om) * np.trapz(1 / (aa * E) ** 3, aa))
    return np.array(out)
D0 = growth(0.0)[0]
Dz = growth(z) / D0
kk = []; pp = []
for j in range(z.size):
    k = (cl + 0.5) / chi_h[j]
    kk.append(k); pp.append(pk[j] / Dz[j] ** 2)
kk = np.concatenate(kk); pp = np.concatenate(pp); o = np.argsort(kk); kk, pp = kk[o], pp[o]
# consistency of the rescaled slices (scatter of P0 in narrow k bins)
kg = np.logspace(np.log10(kk.min()) + 0.05, np.log10(kk.max()) - 0.05, 400)
P0 = np.array([np.median(pp[(kk > kq / 1.02) & (kk < kq * 1.02)]) for kq in kg])
scat = np.array([np.std(np.log(pp[(kk > kq / 1.02) & (kk < kq * 1.02)])) for kq in kg])
rho_m = 2.775e11 * Om                                          # h^2 Msun / Mpc^3 -> (Msun/h)/(Mpc/h)^3
def sigma(Mh):
    R = (3 * Mh / (4 * np.pi * rho_m)) ** (1 / 3)
    x = kg[None, :] * R[:, None]
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(np.trapz(kg ** 3 * P0 * W ** 2 / (2 * np.pi ** 2), np.log(kg), axis=1))
TAB = np.array([[200, 0.186, 1.47, 2.57, 1.19], [300, 0.200, 1.52, 2.25, 1.27], [400, 0.212, 1.56, 2.05, 1.34],
                [600, 0.218, 1.61, 1.87, 1.45], [800, 0.248, 1.87, 1.59, 1.58], [1200, 0.255, 2.13, 1.51, 1.80],
                [1600, 0.260, 2.30, 1.46, 1.97], [2400, 0.260, 2.53, 1.44, 2.24], [3200, 0.260, 2.66, 1.41, 2.44]])
def t08(Mh, zq):
    Omz = Om * (1 + zq) ** 3 / (Om * (1 + zq) ** 3 + 1 - Om)
    Dm = 200 / Omz
    ld = np.log10(Dm)
    A, a, b, c = [np.interp(ld, np.log10(TAB[:, 0]), TAB[:, i]) for i in range(1, 5)]
    alpha = 10 ** (-(0.75 / np.log10(Dm / 75)) ** 1.2)
    A = A * (1 + zq) ** -0.14; a = a * (1 + zq) ** -0.06; b = b * (1 + zq) ** -alpha
    s = sigma(Mh) * growth(zq)[0] / D0
    lns = np.log(s); dlns = np.gradient(lns, np.log(Mh))
    f = A * ((s / b) ** -a + 1) * np.exp(-c / s ** 2)
    return f * rho_m / Mh * (-dlns)
M8 = 4 * np.pi / 3 * 8.0 ** 3 * rho_m
res = {"sigma8 from rescaled P_lin (expect 0.8159)": float(sigma(np.array([M8]))[0]), "P0 slice scatter (rms ln) median, max": [float(np.median(scat)), float(np.max(scat))], "comparison": []}
Mh = np.exp(lnm)
for zq in [0.05, 0.3, 1.0, 2.0]:
    j = int(np.argmin(abs(z - zq)))
    mine = t08(Mh, z[j])
    for mq in [1e13, 1e14, 3e14, 6.3e14, 1e15, 2e15]:
        i = int(np.argmin(abs(Mh - mq)))
        res["comparison"].append({"z": float(z[j]), "M_msunh": float(Mh[i]), "mine/CLASS-SZ": float(mine[i] / dn_cs[j, i])})
print(json.dumps(res, indent=1))
json.dump(res, open(OUT + "v8_tinker08_check.json", "w"), indent=1)
