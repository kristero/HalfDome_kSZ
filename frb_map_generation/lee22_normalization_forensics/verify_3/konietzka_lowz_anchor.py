"""verify_3: independent re-derivation of the low-z Konietzka (TNG300) anchor for F_n.

All-ray mean halo DM for sources at z_s (no FFT needed):
  <DM>(z_s) = int_0^zs dz (1+z) (c/H) int dlnM dn/dlnM(comoving) N_e(M,z)
with N_e the electrons inside the R200c sphere. Halo mass function: Tinker08 from the `hmf`
package (independent of the repo's engine), 200c, TNG300 Planck15 cosmology, CAMB transfer.
Window 1e13-1e14 Msun (M200c, physical Msun, as assumed for the catalogue).
Outputs: implied electron-equivalent gas fraction f_e = N_e mu_e m_p/(f_b M) of TNG halos
(mass-function weighted) and C = TNG/Lee22-literal, F_n^K = 1/C for z_s = 0.1, 0.2.
"""
import numpy as np, json, os
from scipy import integrate
import hmf
from astropy.cosmology import FlatLambdaCDM
import lee22_c4_lib as L

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3"
KDIR = "/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/"
zs_arr = np.load(KDIR + "redshifts_DMhalo.npy")
tng = {}
for name in ["mass13to14", "mass12to14"]:
    a = np.load(KDIR + f"DMhalo_r200_{name}.npy", mmap_mode="r")
    tng[name] = {float(round(zs_arr[i], 3)): float(np.asarray(a[i]).mean()) for i in range(3)}

cosmo = FlatLambdaCDM(H0=100 * L.H, Om0=L.OM, Ob0=L.OB, Tcmb0=2.7255)
C_KMS = 299792.458
windows = {"mass13to14": (13.0, 14.0), "mass12to14": (12.0, 14.0)}
zgrid = np.linspace(0.0025, 0.1975, 40)
lnm = np.linspace(np.log(1e12), np.log(1e14), 161)
m = np.exp(lnm)
dndlnm = np.zeros((len(zgrid), len(m)))
mf = hmf.MassFunction(cosmo_model=cosmo, sigma_8=0.8159, n=0.9667, hmf_model="Tinker08",
                      mdef_model="SOCritical", mdef_params={"overdensity": 200},
                      transfer_model="CAMB", Mmin=11.5, Mmax=14.5, dlog10m=0.01)
for iz, z in enumerate(zgrid):
    mf.update(z=z)
    # hmf masses are in Msun/h and dndlnm in h^3 Mpc^-3
    dndlnm[iz] = np.interp(lnm, np.log(mf.m / L.H), mf.dndlnm * L.H ** 3)


def electrons_literal(mmsun, z):
    """electrons inside R200c for Lee22 literal (n_e = nhat n200)"""
    nbar = np.array([L.vol_int(lambda x: L.nhat(x, mi * L.H, z), 1e-4, 1.0, n=800) for mi in mmsun])
    V = 4 / 3 * np.pi * (L.r200c_mpc(mmsun, z) * L.MPC_CM) ** 3
    return nbar * L.n200_cm3(z) * V


def electrons_fe1(mmsun):
    return L.FB * mmsun * L.MSUN_G / (L.MU_E * L.M_P_G)


res = {"tng_all_ray_mean_dm": tng}
lit = np.array([electrons_literal(m, z) for z in zgrid])   # (nz, nm)
fe1 = electrons_fe1(m)[None, :]
for wname, (lo, hi) in windows.items():
    sel = (lnm >= np.log(10 ** lo)) & (lnm <= np.log(10 ** hi))
    for zs in [0.1, 0.2]:
        zsel = zgrid < zs
        dz = zgrid[1] - zgrid[0]
        kern = (1 + zgrid) * C_KMS / (100 * L.H * L.E(zgrid))            # Mpc
        conv = 1e6 / L.MPC_CM ** 3                                       # (Mpc^-3 * Mpc) -> pc cm^-3
        def dm(Ne):
            inner = integrate.simpson(dndlnm[:, sel] * Ne[:, sel], x=lnm[sel], axis=1)
            return float(np.sum((kern * inner)[zsel]) * dz * conv)
        dm_fe1 = dm(np.broadcast_to(fe1, lit.shape))
        dm_lit = dm(lit)
        t = tng[wname][zs]
        res[f"{wname}|zs={zs}"] = dict(TNG_all_ray_mean=t, model_fe1=dm_fe1, model_literal=dm_lit,
                                       implied_fe_TNG=t / dm_fe1, implied_fe_literal=dm_lit / dm_fe1,
                                       C_TNG_over_literal=t / dm_lit, Fn_K=dm_lit / t)
        print(wname, zs, res[f"{wname}|zs={zs}"])
json.dump(res, open(os.path.join(OUT, "konietzka_lowz_anchor.json"), "w"), indent=1)
