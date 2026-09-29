"""verify_3: gas mass inside R200c implied by the B16 density fit as printed (rho_gas = rho_crit rho_fit,
no f_b) and with f_b, B16 cosmology f_b = 0.043/0.25 = 0.172."""
import numpy as np, json, os
from scipy.integrate import quad
def b16_fit(x, m, z=0.0):
    m14 = m / 1e14
    rho0 = 4e3 * m14 ** 0.29 * (1 + z) ** -0.66
    al = 0.88 * m14 ** -0.03 * (1 + z) ** 0.19
    be = 3.83 * m14 ** 0.04 * (1 + z) ** -0.025
    u = x / 0.5
    return rho0 * u ** -0.2 * (1 + u ** al) ** (-(be + 0.2) / al)
out = {}
for lm in [13.0, 13.5, 14.0, 14.5, 15.0]:
    for z in [0.0, 1.0]:
        mg = quad(lambda x: 3 * x ** 2 * b16_fit(x, 10 ** lm, z) / 200, 0, 1)[0]   # M_gas/M200 as printed
        out[f"logM={lm},z={z}"] = dict(Mgas_over_M200_printed=mg, Mgas_over_fbM200_printed_fbB16=mg / 0.172,
                                       Mgas_over_fbM200_if_times_fb=mg)
        print(lm, z, round(mg, 3), round(mg / 0.172, 2))
json.dump(out, open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3/b16_gas_mass.json", "w"), indent=1)
