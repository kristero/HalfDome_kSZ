"""HalfDome vs WebSky halo LOS velocity rms vs redshift at matched mass, and both vs linear theory (2026-10-02,
ksz_debug/velocities). Inputs: catalogue_sample.npz (HalfDome), websky_sample.npz (idark strided sample), linear_sigma_v.npz.
WebSky: x,y,z [comoving Mpc], v [km/s], R_TH [Mpc]; M200m = 4pi/3 rho_m R_TH^3 (rho_m = 2.775e11 Om h^2 Msun/Mpc^3).
HalfDome: Position [Mpc/h], Velocity [km/s], halo_mass_m200m / m200c [Msun/h]."""
import numpy as np, camb, json
from scipy.interpolate import RegularGridInterpolator as RGI
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities/"
h, Om = 0.68, 0.31
rho_m = 2.775e11 * Om * h**2
pars = camb.set_params(H0=100 * h, ombh2=0.049 * h * h, omch2=(Om - 0.049) * h * h, mnu=0)
bg = camb.get_background(pars)
zgrid = np.linspace(0, 5, 5001); chig = bg.comoving_radial_distance(zgrid)

ws = np.load(O + "websky_sample.npz")["data"].astype(float)
xw, vw, Rw = ws[:, :3], ws[:, 3:6], ws[:, 6]
rw = np.linalg.norm(xw, axis=1); zw = np.interp(rw, chig, zgrid)
vlos_w = (xw * vw).sum(1) / rw
M200m_w = 4 * np.pi / 3 * rho_m * Rw**3          # Msun
lagw = ws[:, 7:10]; disp = np.linalg.norm(xw - lagw, axis=1)

hd = np.load(O + "catalogue_sample.npz")
P = hd["Position"].astype(float); V = hd["Velocity"].astype(float); zh = hd["redshift"].astype(float)
vlos_h = (P * V).sum(1) / np.linalg.norm(P, axis=1)
M200m_h = hd["halo_mass_m200m"] / h; M200c_h = hd["halo_mass_m200c"] / h; lfof = hd["log10_m_rfof"].astype(float) - np.log10(h)

L = np.load(O + "linear_sigma_v.npz")
fpk = RGI((L["z"], L["logM_msun_h"]), L["sigma_v1d_tophat_peak"], bounds_error=False, fill_value=None)
lin_w = fpk(np.c_[zw, np.log10(M200m_w * h)])
lin_h = fpk(np.c_[zh, lfof + np.log10(h)])

print("WebSky sample: N=%d  z range %.2f-%.2f  log10 M200m [Msun] range %.2f-%.2f" % (len(zw), zw.min(), zw.max(), np.log10(M200m_w.min()), np.log10(M200m_w.max())))
print("WebSky |v|/v_los rms ratio overall: %.3f; mean v = %s" % (np.sqrt(np.mean((vw**2).sum(1)) / np.mean(vlos_w**2)), vw.mean(0).round(1)))
rms = lambda x: np.sqrt(np.mean(x**2))
zbins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]
mbins = [(12.8, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.5)]   # log10 M200m [Msun]
res = []
print("\n z bin    | WebSky v_los rms (N) / HalfDome v_los rms (N) / ratio HD/WS   per log10 M200m[Msun] bin;  WS/lin  HD/lin (all M>12.8)")
for zl, zhh in zbins:
    sw = (zw >= zl) & (zw < zhh); sh = (zh >= zl) & (zh < zhh)
    line = "%4.2f-%4.2f |" % (zl, zhh)
    row = {"z": [zl, zhh], "bins": []}
    for ml, mh in mbins:
        a = sw & (np.log10(M200m_w) >= ml) & (np.log10(M200m_w) < mh); b = sh & (np.log10(M200m_h) >= ml) & (np.log10(M200m_h) < mh)
        if a.sum() > 200 and b.sum() > 200:
            line += " %5.1f(%6d)/%5.1f(%6d)/%5.3f |" % (rms(vlos_w[a]), a.sum(), rms(vlos_h[b]), b.sum(), rms(vlos_h[b]) / rms(vlos_w[a]))
            row["bins"].append([ml, mh, rms(vlos_w[a]), int(a.sum()), rms(vlos_h[b]), int(b.sum())])
        else:
            line += "  %29s |" % "-"
    a = sw & (M200m_w > 10**12.8); b = sh & (M200m_h > 10**12.8)
    line += "  %5.3f  %5.3f" % (rms(vlos_w[a]) / rms(lin_w[a]), rms(vlos_h[b]) / rms(lin_h[b]))
    row["ws_over_lin"] = rms(vlos_w[a]) / rms(lin_w[a]); row["hd_over_lin"] = rms(vlos_h[b]) / rms(lin_h[b])
    print(line); res.append(row)
# matched-mass mean power ratio, HD/WS, weighted by HalfDome counts per cell
num = den = 0.0
for row in res:
    for ml, mh, vw_, nw, vh_, nh in row["bins"]:
        num += nh * vh_**2; den += nh * vw_**2
print("\nHalfDome-count-weighted <v_los^2> ratio HD/WS (matched z, M200m cells) = %.3f -> velocity ratio %.3f" % (num / den, np.sqrt(num / den)))
# WebSky scaling of velocity with a: fit ratio to linear theory vs a
a_ = np.array([1 / (1 + 0.5 * (r["z"][0] + r["z"][1])) for r in res]); wl = np.array([r["ws_over_lin"] for r in res]); hl = np.array([r["hd_over_lin"] for r in res])
print("fit WS/lin ~ a^p: p=%.3f amp=%.3f ; HD/lin ~ a^p: p=%.3f amp=%.3f" % (*np.polyfit(np.log(a_), np.log(wl), 1)[::1] * [1, 1],
      *np.polyfit(np.log(a_), np.log(hl), 1)))
print("WebSky displacement |x - x_lag| rms by z:", [(r["z"], round(float(rms(disp[(zw >= r['z'][0]) & (zw < r['z'][1])])), 2)) for r in res])
json.dump(res, open(O + "websky_halfdome_vlos_rms.json", "w"), indent=1)
