"""Outside-group estimate: effect on the HalfDome B16 one-halo kSZ of WebSky's halo-map compensation (Stein+20 Sec 3.2.1:
subtract the kSZ of a uniform sphere with the halo's mass and position and a radius at overdensity 3 [mean], electrons
f_b M (1 - Yp/2)/m_p), evaluated with the HalfDome catalogue histogram and the signed a(ell; M, z) table."""
from common import *
import h5py
with h5py.File(OUTD + 'onehalo_cell_table_signed.h5', 'r') as f:
    ell = f['ell'][:]; A = np.transpose(f['a'][:], (2, 1, 0)); A0 = f['a0'][:].T; S0 = f['s0_per_M'][:].T; CHI = f['chi_mpc'][:]
    lt = f['logm200c_msun_centres'][:]; zt = f['z_centres'][:]
d = np.load(OUTD + 'halfdome_catalogue_pass.npz'); nc, vc = d['hist_n_c'], d['hist_v2_c']
le = d['logm_edges']; lmc = 0.5 * (le[1:] + le[:-1])
i0 = int(round((lt[0] - lmc[0]) / 0.025)); nm = len(lt); nz = nc.shape[1]
V2 = vc[i0:i0 + nm, :nz] / C ** 2
rmean = d['acc_ratio'] / np.maximum(d['acc_n'], 1)        # <M200c/M200m> per 0.1 z
RHO_M0 = 2.775e11 * 0.31 * 0.68 ** 2
def W(x): x = np.maximum(x, 1e-6); return np.where(x < 1e-3, 1 - x ** 2 / 10, 3 * (np.sin(x) - x * np.cos(x)) / x ** 3)
LIDX = [list(ell).index(L) for L in (300.0, 1000.0, 2000.0, 3000.0, 5000.0)]
out = {}
for label, m_def, ffree in (('sphere mass = M200m, x0.9 (consistent with XGPaint ne2d)', 'm', 0.9),
                             ('sphere mass = M200m, no 0.9', 'm', 1.0), ('sphere mass = M200c, x0.9', 'c', 0.9)):
    cl_b = np.zeros(len(ell)); cl_c = np.zeros(len(ell)); qs = []
    for j in range(nz):
        im = np.nonzero(V2[:, j] > 0)[0]
        if len(im) == 0: continue
        ratio = rmean[min(int(zt[j] / 0.1), len(rmean) - 1)] if m_def == 'm' else 1.0
        Mm = 10 ** lt[im] / ratio                                  # M200m [Msun]
        as0 = S0[im, j] / ratio * ffree                            # S0 was computed for sphere mass = M200c of the cell
        R3 = (3 * Mm / (4 * np.pi * 3 * RHO_M0)) ** (1 / 3)        # comoving Mpc
        th = R3 / CHI[j]
        ah = A[:, im, j]
        ac = ah - as0[None, :] * W(ell[:, None] * th[None, :])
        cl_b += (ah ** 2 * V2[im, j]).sum(1); cl_c += (ac ** 2 * V2[im, j]).sum(1)
        if abs(zt[j] - 0.5125) < 1e-6:
            for lm in (13.0125, 13.5125, 14.0125, 14.5125):
                k = np.argmin(abs(lt[im] - lm)); qs.append(f'logM200c {lt[im][k]:.2f}: q = N_sphere/N_halo(<4R200c) = {as0[k] / A0[im[k], j]:.3f}, theta_s/theta200c = {th[k] / (10 ** 0 * f_th[im[k], j]) if False else th[k]:.2e} rad')
    print(label); print('   ' + '\n   '.join(qs))
    print('   D_ell base   ' + ' '.join(f'{dl(ell[i], cl_b[i] / (4 * np.pi)):.4f}' for i in LIDX))
    print('   D_ell comp.  ' + ' '.join(f'{dl(ell[i], cl_c[i] / (4 * np.pi)):.4f}' for i in LIDX) + '   ratio comp/base ' + ' '.join(f'{cl_c[i] / cl_b[i]:.3f}' for i in LIDX))
    out[label] = cl_c / cl_b
