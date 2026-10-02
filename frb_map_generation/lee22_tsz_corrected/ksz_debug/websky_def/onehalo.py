"""Catalogue one-halo kSZ engine (Python re-implementation of ksz/poisson_mass_split.jl), exact flat-sky:
   C_ell = (1/4pi) sum_cells sum_i (v_i/c)^2 |tau_ell(M,z)|^2,  tau_ell = sigma_T/D_A^2 int_0^{X R200c} 4 pi r^2 n_e(r) j0(k r) dr,
   k = ell/D_A (physical).  (The 2-D Fourier transform of a projected, spherically truncated 3-D profile is the 3-D FT.)
Profile: Battaglia (2016) AGN 200c gas density as in XGPaint BattagliaTauProfile:
   rho_gas = f_b rho_crit(z) P0 (x/xc)^g [1+(x/xc)^a]^(-(b+g)/a),  n_e = 0.9 rho_gas / (1.1364 m_p)  ('xgp')
Options reproduce candidate WebSky conventions (see variants in run_variants.py)."""
import numpy as np
from scipy.integrate import quad
MSUN = 1.98847e30; MPC = 3.0856775814913673e22; MP = 1.67262192e-27; ME = 9.1093837e-31
SIGT = 6.6524587321e-29; C_KMS = 299792.458
XH = 0.76
M_PER_E = ME + 2*XH/(XH+1)*MP + (1-XH)/(2*(1+XH))*4*MP      # XGPaint ne2d mass per electron (kg)

class Cosmo:
    def __init__(s, h=0.68, Om=0.31, Ob=0.049):
        s.h, s.Om, s.Ob = h, Om, Ob; s.OL = 1 - Om
        zz = np.linspace(0, 6, 60001); Ez = np.sqrt(Om*(1+zz)**3 + s.OL)
        chi = np.concatenate([[0], np.cumsum(0.5*(1/Ez[1:] + 1/Ez[:-1])*np.diff(zz))]) * C_KMS/(100*h)
        s._z, s._chi = zz, chi
        s.rhoc0 = 2.77536627e11 * h**2          # Msun / Mpc^3
    def E(s, z): return np.sqrt(s.Om*(1+z)**3 + s.OL)
    def chi(s, z): return np.interp(z, s._z, s._chi)
    def DA(s, z): return s.chi(z)/(1+z)
    def rhoc(s, z): return s.rhoc0*s.E(z)**2   # physical Msun/Mpc^3
    def Omz(s, z): return s.Om*(1+z)**3/s.E(z)**2

def b16_params(M, z, exp_form='xgp'):
    m = M/1e14; z1 = 1+z
    P0 = 4e3*m**0.29*z1**-0.66; a = 0.88*m**-0.03*z1**0.19; b = 3.83*m**0.04*z1**-0.025; g = -0.2; xc = 0.5
    return P0, a, b, g, xc

def b16_shape(x, M, z, exp_form='xgp'):
    P0, a, b, g, xc = b16_params(M, z)
    xb = x/xc
    e = (b+g)/a if exp_form == 'xgp' else (b-g)/a      # 'printed' = B16 eq. A1 as printed
    return P0 * xb**g * (1 + xb**a)**(-e)

def tau_ell_table(cos, logM, z, ells, X=4.0, ne_conv='xgp', exp_form='xgp', nr=600, comp=None, comp_mass=None, comp_match=False):
    """tau_ell for arrays logM[n], z[n] (cells) -> (n, nell). ne_conv: 'xgp' (0.9 rho_gas/m_per_e), 'websky' ((1-Y/2) rho_gas/m_p),
    'meanb' (F = rho_fit relative to mean baryon density: n_e = (1-Y/2) rho_b(z) rho_fit/m_p).
    comp: None or overdensity Delta for a compensating uniform sphere (radius where mean density = Delta*rho_m(z)) with
    electron number f_b*comp_mass*(electrons per gas mass of the same convention). Returns tau_ell (dimensionless, sr)."""
    M = 10**np.asarray(logM, float); z = np.asarray(z, float); ells = np.asarray(ells, float)
    fb = cos.Ob/cos.Om
    rhoc = cos.rhoc(z)                                   # Msun/Mpc^3 physical
    R200 = (3*M/(4*np.pi*200*rhoc))**(1/3)               # Mpc physical
    DA = cos.DA(z)                                       # Mpc physical
    # radial grid in x = r/R200c, log spaced + 0
    Xarr = np.broadcast_to(np.asarray(X, float), M.shape)
    out = np.zeros((len(M), len(ells)))
    if ne_conv == 'xgp':   epm = 0.9*MSUN/M_PER_E        # electrons per Msun of gas
    else:                  epm = (1 - 0.24/2)*MSUN/MP
    for i in range(len(M)):
        x = np.geomspace(1e-5, Xarr[i], nr)
        shape = b16_shape(x, M[i], z[i], exp_form)       # rho_fit
        if ne_conv == 'meanb':
            rho_gas = cos.Ob*cos.rhoc0*(1+z[i])**3 * shape   # Msun/Mpc^3 physical, F = rho_fit
        else:
            rho_gas = fb*rhoc[i]*shape
        ne = rho_gas*epm                                  # electrons / Mpc^3
        r = x*R200[i]
        k = ells/DA[i]                                    # 1/Mpc
        kr = np.outer(k, r)
        j0 = np.sinc(kr/np.pi)
        integrand = 4*np.pi*r**2*ne * j0                 # (nell, nr)
        Ne_k = np.trapz(integrand*r, np.log(r), axis=1)   # electrons (FT at k)
        Ne_k += 4*np.pi*ne[0]*r[0]**3/3                  # tiny inner sphere
        Ne_k0 = np.trapz(4*np.pi*r**3*ne, np.log(r)) + 4*np.pi*ne[0]*r[0]**3/3
        tau = SIGT*Ne_k/(DA[i]*MPC)**2
        if comp is not None:
            Mc = comp_mass[i]
            rhom = cos.Om*cos.rhoc0*(1+z[i])**3
            R3 = (3*Mc/(4*np.pi*comp*rhom))**(1/3)
            Nsph = fb*Mc*epm
            if comp_match:                                   # sphere electrons = profile electrons (zero net tau at ell=0)
                Nsph = Ne_k0
            q = k*R3
            W = np.where(q < 1e-3, 1 - q**2/10, 3*(np.sin(q) - q*np.cos(q))/np.maximum(q, 1e-30)**3)
            tau = tau - SIGT*Nsph*W/(DA[i]*MPC)**2
        out[i] = tau
    return out

def m200c_over_m200m_table(cos, cm=7.0):
    def mfun(x): return np.log1p(x) - x/(1+x)
    zt = np.linspace(0, 5, 501); r = np.empty_like(zt); xs = np.linspace(0.5, 7.0, 200001)
    for i, zz in enumerate(zt):
        g = mfun(xs)/xs**3 - mfun(cm)/cm**3/cos.Omz(zz)
        j = np.where(np.diff(np.sign(g)))[0][0]; xx = xs[j] - g[j]*(xs[j+1]-xs[j])/(g[j+1]-g[j])
        r[i] = mfun(xx)/mfun(cm)
    return zt, r

def onehalo_cl(cos, logm_edges, z_edges, count, sumv2, ells, select=None, **kw):
    lm = 0.5*(logm_edges[1:]+logm_edges[:-1]); zc = 0.5*(z_edges[1:]+z_edges[:-1])
    im, iz = np.nonzero(count > 0)
    LM, Z, W = lm[im], zc[iz], sumv2[im, iz]
    if select is not None:
        s = select(LM, Z); LM, Z, W = LM[s], Z[s], W[s]
    comp_mass = kw.pop('comp_mass_fn', None)
    Xfn = kw.pop('X_fn', None)
    if Xfn is not None: kw['X'] = Xfn(LM, Z)
    if comp_mass is not None: kw['comp_mass'] = comp_mass(LM, Z)
    t = tau_ell_table(cos, LM, Z, ells, **kw)
    return (W[:, None]*t**2).sum(0)/(4*np.pi), (LM, Z, W, t)
