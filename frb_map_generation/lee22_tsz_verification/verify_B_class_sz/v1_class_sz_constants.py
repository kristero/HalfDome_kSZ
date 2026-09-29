"""Verify independently the two CLASS-SZ constants that decide P_e vs P_th for the B12 profile.

class_sz.c (v0.1.70.post40):
  P_200 = m/r200c * f_b * 2.61051e-18 * (100 h E)^2        [eV cm^-3], m in Msun/h, r200c in Mpc/h, H in km/s/Mpc
  sigmaT_over_mec2_times_50eV_per_cm3_times_Tcmb = 283.2980000259841/0.5176 * T_cmb/2.725
Thermal Battaglia P200 = 200 G M rho_c f_b / (2 R200) = (300/(8 pi)) H^2 f_b M / R.
"""
import json
import numpy as np

G = 6.67430e-8
MSUN = 1.98847e33
MPC = 3.0856775814913673e24
EV = 1.602176634e-12
SIGMA_T = 6.6524587321e-25          # cm^2
MEC2_EV = 510998.95

# P200_th / [(m/r) f_b H^2] with m/r in Msun/Mpc (h cancels), H in km/s/Mpc
H_unit = 1e5 / MPC                   # 1 km/s/Mpc in s^-1
coef_th = 300.0 / (8 * np.pi) * H_unit ** 2 * MSUN / MPC / EV   # eV cm^-3
coef_e = 0.5176 * coef_th
# y factor: sigma_T/(m_e c^2) x 50 eV cm^-3 x 1 Mpc x T_cmb[muK]
yfac = SIGMA_T / MEC2_EV * 50.0 * MPC * 2.725e6
out = {
    "P200_thermal_constant_eV_cm3": coef_th,
    "P200_electron_constant_0p5176_eV_cm3": coef_e,
    "class_sz_constant": 2.61051e-18,
    "class_sz / thermal": 2.61051e-18 / coef_th,
    "class_sz / (0.5176 thermal)": 2.61051e-18 / coef_e,
    "sigmaT/mec2*50eV/cm3*Mpc*Tcmb[muK]": yfac,
    "class_sz 283.298/0.5176": 283.2980000259841 / 0.5176,
    "ratio y factor": (283.2980000259841 / 0.5176) / yfac,
}
print(json.dumps(out, indent=1))
json.dump(out, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_B_class_sz/v1_class_sz_constants.json", "w"), indent=1)
