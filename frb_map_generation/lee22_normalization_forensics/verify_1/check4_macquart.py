"""verify_1 check 4: Konietzka total DM (IGM+halo) vs Macquart relation, to test that the TNG catalogue
includes the 1/(1+z) factor (slope of DM/Macquart vs z should be ~0)."""
import numpy as np, sys
from scipy import integrate
sys.path.insert(0, '.')
import lee22_verify_lib as L
K = '/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/'
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_1/'
zs = np.load(K + 'redshifts_DMhalo.npy'); a = np.load(K + 'DMall_IGM_halo.npy')
H0 = 100 * L.H * 1e5 / L.MPC_CM
ne0 = 3 * H0**2 / (8 * np.pi * L.G_CGS) * L.OB / L.MP_G * (L.XH + (1 - L.XH) / 2)  # cm^-3, fully ionized H+He
cH0 = 2.99792458e10 / H0 / L.PC_CM  # pc
lines = []
for i, z in enumerate(zs):
    I = integrate.quad(lambda x: (1 + x) / np.sqrt(L.E2(x)), 0, z)[0]
    mac = ne0 * cH0 * I
    Ino = integrate.quad(lambda x: (1 + x)**2 / np.sqrt(L.E2(x)), 0, z)[0]  # if 1/(1+z) omitted
    if i in [0, 2, 4, 9, 14, 19, 29, 39, 54]:
        lines.append(f'z_s={z:.1f}  <DM_TNG>={a[i].mean():8.1f}  Macquart(f_d=1)={mac:8.1f}  ratio={a[i].mean()/mac:.4f}   ratio if no 1/(1+z)={a[i].mean()/(ne0*cH0*Ino):.4f}')
r = np.array([a[i].mean() / (ne0 * cH0 * integrate.quad(lambda x: (1 + x) / np.sqrt(L.E2(x)), 0, z)[0]) for i, z in enumerate(zs)])
k = np.polyfit(np.log(1 + zs), np.log(r), 1)[0]
lines.append(f'slope of TNG/Macquart vs (1+z) over z_s=0.1-5.5: {k:.4f}; range {r.min():.4f}-{r.max():.4f}')
print('\n'.join(lines)); open(OUT + 'check4_macquart.txt', 'w').write('\n'.join(lines) + '\n')
