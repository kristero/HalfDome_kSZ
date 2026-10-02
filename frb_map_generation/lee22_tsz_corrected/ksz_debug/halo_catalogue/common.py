import numpy as np, h5py, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/halo_catalogue')
OUTD = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/halo_catalogue/'
T2 = 2.7255e6 ** 2; C = 299792.458; HPAINT = 0.68
def load_table():
    with h5py.File(OUTD + 'onehalo_cell_table.h5', 'r') as f:
        # Julia column-major -> h5py reversed axes: a2 is (z, logm, ell)
        return dict(ell=f['ell'][:], logm=f['logm200c_msun_centres'][:], zc=f['z_centres'][:],
                    a2=np.transpose(f['a2'][:], (2, 1, 0)), th=np.transpose(f['theta200c_rad'][:]))
def dl(ell, cl): return ell * (ell + 1) * cl / (2 * np.pi) * T2
def nfw_mu(x): return np.log(1 + x) - x / (1 + x)
def convert_delta(c_in, delta_in, delta_out, nit=60):
    """NFW: given concentration c_in w.r.t. delta_in (same reference density), return (M_out/M_in, R_out/R_in)."""
    # mean density inside r (units of reference): delta_in * mu(c y)/mu(c) / y^3 = delta_out -> solve y
    lo, hi = 1e-3 * np.ones_like(c_in), 1e3 * np.ones_like(c_in)
    for _ in range(nit):
        y = np.sqrt(lo * hi)
        dens = delta_in * nfw_mu(c_in * y) / nfw_mu(c_in) / y ** 3
        hi = np.where(dens < delta_out, y, hi); lo = np.where(dens >= delta_out, y, lo)
    y = np.sqrt(lo * hi)
    return nfw_mu(c_in * y) / nfw_mu(c_in), y
def om_z(z, om=0.31): return om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om)
def m200c_over_m200m_c7(z, om=0.31):
    """WebSky: NFW with c200m = 7; M200c / M200m (200c = 200/Om(z) times mean)."""
    r, _ = convert_delta(np.atleast_1d(7.0 * np.ones_like(np.atleast_1d(z), dtype=float)), 200.0, 200.0 / om_z(np.atleast_1d(z), om))
    return r
def duffy_c200c(Mh, z):   # Duffy+08 full sample, M200c in Msun/h
    return 5.71 * (Mh / 2e12) ** -0.084 * (1 + z) ** -0.47
def m200c_over_m200m_duffy(Mh_c, z, om=0.3089):
    c = duffy_c200c(Mh_c, z)
    r, _ = convert_delta(np.atleast_1d(c), 200.0 / om_z(np.atleast_1d(z), om), 200.0)   # crit->mean: delta_in=200/Om(z) in mean units
    return 1 / r
