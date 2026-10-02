"""Summary table: HalfDome (full catalogue) and WebSky (full catalogue) v_los rms vs z in M200m bins, linear theory
(top-hat + peak, unsmoothed), power-law fits of the ratio to linear theory in a. Writes velocity_summary.csv. 2026-10-02."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities/"
w = np.load(O + "websky_velocity_hist.npz"); d = np.load(O + "halfdome_velocity_hist.npz"); L = np.load(O + "linear_sigma_v.npz")
LE = w["logm200m_edges"]; ZE = w["z_edges"]
fpk = RGI((L["z"], L["logM_msun_h"]), L["sigma_v1d_tophat_peak"], bounds_error=False, fill_value=None)
def agg(n, s, ml, mh, zl, zh):
    m = (LE[:-1] >= ml - 1e-9) & (LE[:-1] < mh - 1e-9); z = (ZE[:-1] >= zl - 1e-9) & (ZE[:-1] < zh - 1e-9)
    N = n[m][:, z].sum(); return (np.sqrt(s[m][:, z].sum() / N) if N > 100 else np.nan), N
zb = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 3.8)]
rows = []
for ml, mh in ((13.0, 13.5), (14.0, 14.5)):
    for zl, zh in zb:
        zc = 0.5 * (zl + zh); a = 1 / (1 + zc)
        vh, nh = agg(d["m200m_count"], d["m200m_sum_vlos2"], ml, mh, zl, zh); vw, nw = agg(w["count"], w["sum_vlos2"], ml, mh, zl, zh)
        lin = fpk([[zc, 0.5 * (ml + mh) + np.log10(0.68)]])[0]; unsm = np.interp(zc, L["z"], L["sigma_v1d_unsmoothed"])
        rows.append((ml, mh, zl, zh, a, vh, nh, vw, nw, lin, unsm))
rows = np.array(rows)
np.savetxt(O + "velocity_summary.csv", rows, delimiter=",", fmt="%.5g",
           header="log10M200m_lo,log10M200m_hi,z_lo,z_hi,a_mid,HD_vlos_rms_kms,HD_N,WS_vlos_rms_kms,WS_N,lin_tophat_peak_1D_kms,lin_unsmoothed_1D_kms")
print("M200m bin   z       a     HD     WS     lin_THpk  lin_unsm  HD/WS  HD/lin  WS/lin  sqrt(3)? 1/sqrt(a)?")
for r in rows:
    print("%4.1f-%4.1f %4.2f-%4.2f %5.3f %6.1f %6.1f %8.1f %8.1f   %5.3f  %5.3f  %5.3f  %5.3f %5.3f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[7], r[9], r[10], r[5] / r[7], r[5] / r[9], r[7] / r[9], np.sqrt(3), 1 / np.sqrt(r[4])))
for ml in (13.0, 14.0):
    s = (rows[:, 0] == ml) & np.isfinite(rows[:, 5]) & np.isfinite(rows[:, 7])
    for name, col in (("HD", 5), ("WS", 7)):
        p, lnA = np.polyfit(np.log(rows[s, 4]), np.log(rows[s, col] / rows[s, 9]), 1)
        print("M200m %.1f-%.1f: %s/lin_THpk = %.3f a^%.3f" % (ml, ml + 0.5, name, np.exp(lnA), p))
    p, lnA = np.polyfit(np.log(rows[s, 4]), np.log(rows[s, 5] / rows[s, 7]), 1)
    print("M200m %.1f-%.1f: HD/WS = %.3f a^%.3f" % (ml, ml + 0.5, np.exp(lnA), p))
