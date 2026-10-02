"""Linear-theory halo velocity rms (1-D = LOS, and 3-D), sigma_v^2 = (f a H)^2 int P_lin(k,z) W^2(k R_L) dk / (2 pi^2) / 3 (1-D),
for the HalfDome and WebSky cosmologies, R_L = Lagrangian radius of M [Msun/h]."""
import numpy as np, camb, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/halo_catalogue')
from tinker08 import Tinker
def vrms(t, z, M=None):
    p = t.p; om = p['om']
    E = np.sqrt(om * (1 + z) ** 3 + 1 - om); a = 1 / (1 + z)
    pk = np.array([np.interp(z, t.zgrid, t.pk[:, j]) for j in range(len(t.kh))])
    # growth rate f from CAMB growth: numerical d ln D / d ln a using sigma8(z) ratio
    f = (om * (1 + z) ** 3 / E ** 2) ** 0.55          # growth rate (Linder gamma = 0.55, <1% for LCDM)
    W = 1.0
    if M is not None:
        R = (3 * M / (4 * np.pi * t.rho_m)) ** (1 / 3); x = t.kh * R; W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    I = np.trapz(t.kh * pk * W ** 2, np.log(t.kh)) / (2 * np.pi ** 2)
    v3 = f * a * 100 * E * np.sqrt(I)
    return v3 / np.sqrt(3), v3, f
if __name__ == '__main__':
    for name in ['halfdome', 'websky']:
        t = Tinker(name)
        for z in [0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5]:
            r = [vrms(t, z, M) for M in (None, 1e13, 1e14)]
            print(name, z, ' '.join(f'{x[0]:.0f}/{x[1]:.0f}' for x in r), f'f={r[0][2]:.3f}')
