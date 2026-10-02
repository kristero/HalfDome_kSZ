"""Tinker+2008 halo mass function (dn/dlnM, M in Msun/h, comoving (Mpc/h)^-3) for any Delta_mean, with CAMB linear P(k).
cosmologies: HalfDome (TNG: h 0.6774, Om 0.3089, Ob 0.0486, s8 0.8159, ns 0.9667) and WebSky (h 0.68, Om 0.31, Ob 0.049, s8 0.81, ns 0.965)."""
import numpy as np, camb
COSMO = {'halfdome': dict(h=0.6774, om=0.3089, ob=0.0486, s8=0.8159, ns=0.9667),
         'websky': dict(h=0.68, om=0.31, ob=0.049, s8=0.81, ns=0.965),
         'painter': dict(h=0.68, om=0.31, ob=0.049, s8=0.81, ns=0.965)}
_D = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200.])
_A = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
_a = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
_b = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
_c = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

class Tinker:
    def __init__(self, name, zmax=4.7, nz=48):
        p = COSMO[name]; self.p = p; h = p['h']
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100 * h, ombh2=p['ob'] * h * h, omch2=(p['om'] - p['ob']) * h * h, mnu=0.0, num_massive_neutrinos=0, omk=0)
        pars.InitPower.set_params(ns=p['ns'], As=2.1e-9)
        self.zs = np.linspace(0, zmax, nz)
        pars.set_matter_power(redshifts=list(self.zs[::-1]), kmax=200.0)
        pars.NonLinear = camb.model.NonLinear_none
        res = camb.get_results(pars)
        s8 = res.get_sigma8()[-1]
        kh, z, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints=2000)
        self.kh = kh; self.pk = pk * (p['s8'] / s8) ** 2; self.zgrid = np.array(z)
        self.rho_m = 2.775e11 * p['om']        # (Msun/h) / (Mpc/h)^3 comoving
        self.res = res
    def sigma(self, M, z):
        R = (3 * M / (4 * np.pi * self.rho_m)) ** (1 / 3)
        iz = np.argmin(np.abs(self.zgrid - z))
        pk = np.array([np.interp(z, self.zgrid, self.pk[:, j]) for j in range(len(self.kh))]) if abs(self.zgrid[iz] - z) > 1e-6 else self.pk[iz]
        x = np.outer(R, self.kh); W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
        lnk = np.log(self.kh)
        return np.sqrt(np.trapz(self.kh ** 3 * pk * W ** 2, lnk, axis=1) / (2 * np.pi ** 2))
    def om_z(self, z):
        om = self.p['om']; return om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om)
    def dndlnm(self, M, z, delta, ref='mean', zcap=None):
        """M [Msun/h]; delta relative to 'mean' or 'crit'; returns dn/dlnM [(Mpc/h)^-3]."""
        D = delta if ref == 'mean' else delta / self.om_z(z)
        lD = np.log10(D)
        A0 = np.interp(lD, np.log10(_D), _A); a0 = np.interp(lD, np.log10(_D), _a)
        b0 = np.interp(lD, np.log10(_D), _b); c0 = np.interp(lD, np.log10(_D), _c)
        ze = z if zcap is None else min(z, zcap)
        alpha = 10 ** (-(0.75 / np.log10(D / 75.0)) ** 1.2)
        A = A0 * (1 + ze) ** -0.14; a = a0 * (1 + ze) ** -0.06; b = b0 * (1 + ze) ** -alpha
        lnM = np.log(M)
        s = self.sigma(M, z)
        fs = A * ((s / b) ** -a + 1) * np.exp(-c0 / s ** 2)
        dlnsinv = -np.gradient(np.log(s), lnM)
        return fs * self.rho_m / M * dlnsinv
    def chi(self, z):
        return self.res.comoving_radial_distance(z) * self.p['h']    # Mpc/h
