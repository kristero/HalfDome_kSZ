"""FLAMINGO definition checks on the 109 cached z = 0 particle cutouts (flamingo_pressure_calibration/notebook_cache/particles/):
P_e/P_total (hot gas, T > 1e6 K, volume weighted), ComptonY = sigma_T/(m_e c^2) n_e k T m/rho, and N_e m_p / M_gas (mu_e^-1).
Writes OUT/density/check_pe_fraction.json."""
import glob
import json
import numpy as np
from scipy.constants import m_p, m_e, c, physical_constants
from common_c import OUT, FPC

SIGMA_T = physical_constants["Thomson cross section"][0]
MPC, MSUN = 3.085677581491367e22, 1.98841e30
pe_frac, y_check, inv_mue = [], [], []
for f in sorted(glob.glob(str(FPC / "notebook_cache" / "particles" / "*.npz"))):
    d = np.load(f)
    v = d["mass_msun"].astype(float) * MSUN / (d["density_msun_mpc3"].astype(float) * MSUN / MPC**3)   # m^3
    hot = d["temperature_k"] > 1e6
    pe_frac.append(np.sum(d["pe_pa"][hot] * v[hot]) / np.sum(d["p_total_pa"][hot].astype(float) * v[hot]))
    y_check.append(np.sum(SIGMA_T / (m_e * c**2) * d["pe_pa"] * v / MPC**2) / np.sum(d["y_mpc2"]))
    inv_mue.append(np.sum(d["ne_m3"] * v) * m_p / np.sum(d["mass_msun"].astype(float) * MSUN))
res = {"n_haloes": len(pe_frac),
       "Pe_over_Ptotal_hot_median": float(np.median(pe_frac)), "Pe_over_Ptotal_hot_range": [float(min(pe_frac)), float(max(pe_frac))],
       "B12_XGPaint_convention": 0.5176,
       "y_from_nekT_over_ComptonY_median": float(np.median(y_check)), "y_check_range": [float(min(y_check)), float(max(y_check))],
       "Ne_mp_over_Mgas_median": float(np.median(inv_mue)), "fully_ionised_primordial_X0.752": (1 + 0.752) / 2}
(OUT / "density").mkdir(parents=True, exist_ok=True)
(OUT / "density" / "check_pe_fraction.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
