"""verify_2 / test T1: redshift scalings implied by the Lee22 tables themselves.

For each mass (h^-1 Msun) and z in the fitted range, compute
  I_n = <n_e/n200>(<R200c)        (Table 8 no-c; Table 3 with c)
  I_P = Y_sph(<R200c)/(P200 V200) (Table 7 no-c; Table 1 with c)
  T-ratio = I_P / I_n  (proportional to the electron-number-weighted T_e / T200)
and log-slopes d ln / d ln(1+z) over z = 0..2 (least squares on 21 points). Also B12 pressure for
reference. Also evaluate the slopes of the radial profiles at fixed x (0.1, 0.5, 1.0).
Predictions of claim C3: I_n ~ (1+z)^-3 x true, I_P ~ (1+z)^-2 x true, T-ratio ~ (1+z)^+1 x true.
"""
import json
import numpy as np
import v2_lee22_lib as L

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_2/"
zs = np.linspace(0, 2, 21)
lz = np.log1p(zs)
masses = [13.0, 13.2, 13.4, 13.61, 13.8, 14.0]


def slope(y):
    return np.polyfit(lz, np.log(y), 1)[0]


def conc(mh, z, dz=-0.3):
    # power law through the two TNG mean concentrations quoted in Lee22 sec 2.3 (z=0)
    c0 = 5.65 * (mh / 10 ** 13.1) ** (np.log10(4.53 / 5.65))
    return c0 * (1 + z) ** dz


res = {"no_c": {}, "with_c": {}, "B12": {}}
lines = []
lines.append("# T1: Lee22 table-implied redshift scalings (d ln / d ln(1+z), z = 0..2)")
lines.append("# log10(M h) | I_n slope | I_P slope | T-ratio slope | I_n(z=0) I_n(z=1) | I_P(z=0) I_P(z=1) | T(z=0) T(z=1) T(z=2) T*(1+z)^-1 at z=0,1,2")
for lm in masses:
    mh = 10 ** lm
    In = np.array([L.vol_mean(L.TABLE8_N, mh, z) for z in zs])
    Ip = np.array([L.vol_mean(L.TABLE7_P, mh, z) for z in zs])
    T = Ip / In
    # T in units of T200 = G M mu m_p/(2 R200) with mu = 0.588 (fully ionized X_H=0.76):
    # P200/n200 = G M X_H m_p/(2R)  ->  kT_e/kT200 = (I_P/I_n) * X_H/mu
    mu = 4.0 / (3 + 5 * L.XH)
    Tt = T * L.XH / mu
    d = dict(In_slope=slope(In), Ip_slope=slope(Ip), T_slope=slope(T), In=In.tolist(), Ip=Ip.tolist(),
             T_over_T200=Tt.tolist())
    res["no_c"][str(lm)] = d
    i1 = 10  # z = 1
    lines.append(f"{lm:5.2f} | {d['In_slope']:+.3f} | {d['Ip_slope']:+.3f} | {d['T_slope']:+.3f} | "
                 f"{In[0]:.3f} {In[i1]:.3f} | {Ip[0]:.3f} {Ip[i1]:.3f} | {Tt[0]:.3f} {Tt[i1]:.3f} {Tt[-1]:.3f} | "
                 f"{Tt[0]:.3f} {Tt[i1]/2:.3f} {Tt[-1]/3:.3f}")

lines.append("# with concentration (Tables 3 and 1), c(M,z) = TNG-mean power law x (1+z)^dz; dz = 0, -0.3, -0.5")
for dz in (0.0, -0.3, -0.5):
    for lm in masses:
        mh = 10 ** lm
        In = np.array([L.vol_mean(L.TABLE3_N, mh, z, conc(mh, z, dz)) for z in zs])
        Ip = np.array([L.vol_mean(L.TABLE1_P, mh, z, conc(mh, z, dz)) for z in zs])
        res["with_c"][f"{lm}_dz{dz}"] = dict(In_slope=slope(In), Ip_slope=slope(Ip), T_slope=slope(Ip / In))
        lines.append(f"dz={dz:+.1f} {lm:5.2f} | I_n {slope(In):+.3f} | I_P {slope(Ip):+.3f} | T {slope(Ip/In):+.3f}")

lines.append("# B12 (AGN) Y/(P200 V200) slope for comparison (M in physical Msun = M_h / h)")
for lm in masses:
    m = 10 ** lm / L.H
    y = np.array([L.b12_vol_mean(m, z) for z in zs])
    res["B12"][str(lm)] = dict(slope=slope(y), y0=y[0], y1=y[10])
    lines.append(f"{lm:5.2f} | B12 I_P slope {slope(y):+.3f} | I_P(0)={y[0]:.3f} I_P(1)={y[10]:.3f}")

lines.append("# fixed-x slopes of the no-c profiles (density, pressure) at x = 0.1, 0.5, 1.0")
for lm in masses:
    mh = 10 ** lm
    row = []
    for x in (0.1, 0.5, 1.0):
        gn = np.array([L.gnfw(x, *L.shape(L.TABLE8_N, mh, z)) for z in zs])
        gp = np.array([L.gnfw(x, *L.shape(L.TABLE7_P, mh, z)) for z in zs])
        row.append(f"x={x}: n {slope(gn):+.2f} P {slope(gp):+.2f}")
    lines.append(f"{lm:5.2f} | " + " | ".join(row))

txt = "\n".join(lines)
print(txt)
open(OUT + "T1_table_scalings.txt", "w").write(txt + "\n")
json.dump(res, open(OUT + "T1_table_scalings.json", "w"), indent=1)
