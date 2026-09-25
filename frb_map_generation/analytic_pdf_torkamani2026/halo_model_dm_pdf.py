#!/usr/bin/env python3
"""Analytical halo-model one-point PDF of the halo DM of FRB sight-lines, following
Torkamani et al. 2026 (Baryonification III, arXiv:2601.18784), Sec. 2.3:

  unclustered (their eq. 31):  P_u(lam) = exp{ int dz dM  c chi^2/H  dn/dM  Y(M, z, lam) },
                               Y(M, z, lam) = int 2 pi theta dtheta [exp(i lam DM(M, z, theta)) - 1]
  clustered   (their eq. 46):  P_cl(lam) = P_u(lam) exp{ 1/2 int dz H/c D^2(z) alpha^2(z, lam) int k dk/(2 pi) P_lin(k) },
                               alpha(z, lam) = int dM b(M, z) c chi^2/H dn/dM Y(M, z, lam)

(the c's make their chi^2/H and H dimensionally explicit). Instead of evaluating Y on a
lambda grid, the single-halo DM distribution is deposited on a fine linear DM lattice,

  Lambda(D) dD = sum_cells dN/dOmega * [solid angle of the halo disc with DM in dD],

so that int dM dz (dN/dOmega) Y = FT[Lambda] - Nbar, with Nbar the mean number of halos a
ray crosses, and the compound-Poisson PDF is one inverse real FFT. Each halo's area between two
consecutive impact-parameter samples is spread uniformly over the DM interval they span (exact
piecewise-linear area-vs-DM interpolation), so the lattice is smooth down to its spacing h.
The DM = 0 delta (rays that miss every halo) has weight exp(-Nbar) (unclustered) or
exp(-Nbar + <A^2(lam->inf)>/2) (clustered) and is removed before normalizing, exactly like the
simulation PDFs that are normalized over rays with DM > 0.

Units: masses in physical Msun, distances in Mpc (no h), DM in pc cm^-3.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import integrate, interpolate

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 has only trapz

C_KMS = 299792.458
RHO_CRIT0_H2 = 2.77536627e11          # Msun Mpc^-3 h^-2
DELTA_C = 1.686
MPC_CM = 3.0856775814913673e24
PC_CM = 3.0856775814913673e18
MSUN_G = 1.98847e33
M_P_G = 1.67262192369e-24


@dataclass
class Cosmology:
    name: str
    omega_m: float
    omega_b: float
    h: float
    sigma8: float
    n_s: float
    _cache: dict = field(default_factory=dict, repr=False)

    def E(self, z):
        return np.sqrt(self.omega_m * (1 + z) ** 3 + 1 - self.omega_m)

    def H_over_c(self, z):
        """H(z)/c in Mpc^-1."""
        return 100 * self.h * self.E(z) / C_KMS

    def chi(self, z):
        """Comoving distance [Mpc]."""
        z = np.atleast_1d(np.asarray(z, float))
        grid = self._cache.get("chi_grid")
        if grid is None:
            zz = np.linspace(0, 6, 60001)
            ch = integrate.cumulative_trapezoid(1 / self.H_over_c(zz), zz, initial=0)
            grid = self._cache["chi_grid"] = (zz, ch)
        return np.interp(z, *grid)

    def rho_crit(self, z):
        """Physical critical density [Msun Mpc^-3]."""
        return RHO_CRIT0_H2 * self.h ** 2 * self.E(z) ** 2

    def rho_m0(self):
        """Comoving mean matter density [Msun Mpc^-3]."""
        return RHO_CRIT0_H2 * self.h ** 2 * self.omega_m

    def omega_m_z(self, z):
        return self.omega_m * (1 + z) ** 3 / self.E(z) ** 2

    def growth(self, z):
        """Linear growth D(z)/D(0) for flat LCDM (Heath 1977 integral)."""
        def unnorm(zz):
            a = 1 / (1 + zz)
            val, _ = integrate.quad(lambda x: 1 / (x * self.E(1 / x - 1)) ** 3, 1e-8, a, limit=200)
            return 2.5 * self.omega_m * self.E(zz) * val
        z = np.atleast_1d(np.asarray(z, float))
        d0 = unnorm(0.0)
        return np.array([unnorm(zz) / d0 for zz in z])

    # ---- linear power spectrum (CAMB, z = 0, sigma8-normalized) ----
    def power(self):
        if "power" not in self._cache:
            import camb
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=100 * self.h, ombh2=self.omega_b * self.h ** 2,
                               omch2=(self.omega_m - self.omega_b) * self.h ** 2, mnu=0.0, omk=0)
            pars.InitPower.set_params(ns=self.n_s, As=2.1e-9)
            pars.set_matter_power(redshifts=[0.0], kmax=300.0)
            pars.NonLinear = camb.model.NonLinear_none
            res = camb.get_results(pars)
            kh, _, pk = res.get_matter_power_spectrum(minkh=1e-5, maxkh=250.0, npoints=2000)
            k = kh * self.h                      # 1/Mpc
            p = pk[0] / self.h ** 3              # Mpc^3
            s8 = np.sqrt(self._sigma2_R(k, p, 8.0 / self.h))
            p = p * (self.sigma8 / s8) ** 2
            self._cache["power"] = (k, p)
        return self._cache["power"]

    @staticmethod
    def _sigma2_R(k, p, R):
        x = np.outer(np.atleast_1d(R), k)
        w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
        return _trapz(k ** 3 * p * w ** 2 / (2 * np.pi ** 2), np.log(k), axis=-1)

    def sigma_M(self, mass):
        """sigma(M, z=0) of the linear field smoothed with a top hat of Lagrangian radius."""
        mass = np.atleast_1d(np.asarray(mass, float))
        key = "sigma_spline"
        if key not in self._cache:
            k, p = self.power()
            lm = np.linspace(6, 17, 441)
            R = (3 * 10 ** lm / (4 * np.pi * self.rho_m0())) ** (1 / 3)
            s = np.sqrt(self._sigma2_R(k, p, R))
            self._cache[key] = interpolate.CubicSpline(lm, np.log(s))
        return np.exp(self._cache[key](np.log10(mass)))

    def dlnsigma_inv_dlnM(self, mass):
        mass = np.atleast_1d(np.asarray(mass, float))
        self.sigma_M(mass[:1])
        spl = self._cache["sigma_spline"]
        return -spl(np.log10(mass), 1) / np.log(10)

    def projected_variance(self):
        """S = int k dk/(2 pi) P_lin(k, z=0)  [Mpc]  (Torkamani eq. 46)."""
        k, p = self.power()
        return _trapz(k * k * p / (2 * np.pi), np.log(k))


HALFDOME = Cosmology("HalfDome", omega_m=0.31, omega_b=0.049, h=0.68, sigma8=0.81, n_s=0.965)
PLANCK15_TNG = Cosmology("Planck15 (TNG)", omega_m=0.3089, omega_b=0.0486, h=0.6774, sigma8=0.8159, n_s=0.9667)

# ---------------------------------------------------------------- Tinker08 HMF, Tinker10 bias
_T08_DELTA = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200], float)
_T08_A = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
_T08_a = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
_T08_b = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
_T08_c = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
_T08_SPL = [interpolate.CubicSpline(np.log10(_T08_DELTA), v) for v in (_T08_A, _T08_a, _T08_b, _T08_c)]


def tinker08_dndlnm(cosmo, mass, z, delta_crit=200.0):
    """Comoving dn/dlnM [Mpc^-3] for M_{Delta c} (Tinker et al. 2008, eqs. 3, 5-8, Table 2)."""
    mass = np.atleast_1d(np.asarray(mass, float))
    delta_m = delta_crit / cosmo.omega_m_z(z)
    ld = np.log10(delta_m)
    A0, a0, b0, c0 = (float(s(ld)) for s in _T08_SPL)
    alpha = 10 ** (-(0.75 / np.log10(delta_m / 75.0)) ** 1.2)
    A = A0 * (1 + z) ** -0.14
    a = a0 * (1 + z) ** -0.06
    b = b0 * (1 + z) ** -alpha
    sig = cosmo.sigma_M(mass) * cosmo.growth(z)[0]
    f = A * ((sig / b) ** -a + 1) * np.exp(-c0 / sig ** 2)
    return f * cosmo.rho_m0() / mass * cosmo.dlnsigma_inv_dlnM(mass)


def tinker10_bias(cosmo, mass, z, delta_crit=200.0):
    """Tinker et al. 2010 eq. 6 / Table 2 bias for M_{Delta c}."""
    mass = np.atleast_1d(np.asarray(mass, float))
    y = np.log10(delta_crit / cosmo.omega_m_z(z))
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4
    nu = DELTA_C / (cosmo.sigma_M(mass) * cosmo.growth(z)[0])
    return 1 - A * nu ** a / (nu ** a + DELTA_C ** a) + B * nu ** b + C * nu ** c


def r200c_mpc(cosmo, mass, z):
    return (3 * mass / (4 * np.pi * 200 * cosmo.rho_crit(z))) ** (1 / 3)


# ---------------------------------------------------------------- BFC hot gas (Torkamani Table 1)
BFC_TNG_FIDUCIAL = dict(eta=0.017, deta=0.229, Nstar=0.0074, ciga=0.0093, theta_c=0.231,
                        log10Mc=12.86, mu=0.721, delta=5.47, alpha=1.0, gamma=1.5,
                        Mstar_hinv=2.5e11, zeta=1.376, eps0=4.0, eps1=0.5)
X_H_TNG = 0.76
ELECTRONS_PER_MP = (1 + X_H_TNG) / 2    # fully ionized H + He: n_e = 0.88 rho / m_p (TNG's own X_H)


def dutton_maccio14_c200c(cosmo, mass, z):
    b = -0.101 + 0.026 * z
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
    return 10 ** (a + b * np.log10(mass * cosmo.h / 1e12))


def bfc_fractions(cosmo, mass, par=BFC_TNG_FIDUCIAL):
    mh = mass * cosmo.h
    ratio = mh / par["Mstar_hinv"]
    fstar = par["Nstar"] / (ratio ** -par["zeta"] + ratio ** par["eta"])
    fcga = par["Nstar"] / (ratio ** -par["zeta"] + ratio ** (par["eta"] + par["deta"]))
    figa = par["ciga"] * fcga
    fhga = cosmo.omega_b / cosmo.omega_m - fstar - figa
    return fhga, fstar, figa


def bfc_chord_mean_dm(cosmo, log10m, z, x, par=BFC_TNG_FIDUCIAL, n_gl=96):
    """g(x) = DM_sphere / (2 sqrt(1-x^2)) [pc cm^-3] of the BFC hot gas inside the R200c sphere,
    for halos log10m (array) at redshift z and impact parameters x = b/R200c (array).
    rho_hga = rho0 [1+u/theta_c]^-beta [1+(u/eps)^gamma]^(-delta/gamma), u = r/R200c, normalized so the
    total hot gas mass to infinity is f_hga M_tot (Schneider et al. 2025, eqs. 2.10-2.20)."""
    mass = 10 ** np.asarray(log10m, float)
    c = dutton_maccio14_c200c(cosmo, mass, z)
    nu = DELTA_C / (cosmo.sigma_M(mass) * cosmo.growth(z)[0])
    eps = par["eps0"] + par["eps1"] * nu
    beta = 3 * (mass * cosmo.h / 10 ** par["log10Mc"]) ** par["mu"] / (1 + (mass * cosmo.h / 10 ** par["log10Mc"]) ** par["mu"])
    fhga = bfc_fractions(cosmo, mass, par)[0]
    # radial integrals on a log grid in u = r/R200c
    u = np.union1d(np.logspace(-6, 3.5, 6000), [1.0])        # u = 1 exactly (M200 normalization)
    lu = np.log(u)
    ynfw = 1 / ((u[None] * c[:, None]) * (1 + u[None] * c[:, None]) ** 2 * (1 + (u[None] / eps[:, None]) ** 2) ** 2)
    m_inf = _trapz(ynfw * u ** 3, lu, axis=1)
    inside = u <= 1
    m_200 = _trapz((ynfw * u ** 3)[:, inside], lu[inside], axis=1)
    mtot_over_m200 = m_inf / m_200
    yhga = lambda uu: ((1 + uu / par["theta_c"]) ** -beta[:, None] *
                       (1 + (uu / eps[:, None]) ** par["gamma"]) ** (-par["delta"] / par["gamma"]))
    norm = _trapz(yhga(u[None]) * u ** 3, lu, axis=1)
    r200 = r200c_mpc(cosmo, mass, z)                              # physical Mpc
    rho0 = fhga * mtot_over_m200 * mass / (4 * np.pi * r200 ** 3 * norm)   # Msun / Mpc^3 (physical)
    ne0 = rho0 * MSUN_G / MPC_CM ** 3 * ELECTRONS_PER_MP / M_P_G           # cm^-3
    x = np.asarray(x, float)
    L = np.sqrt(np.clip(1 - x ** 2, 0, None))
    t, w = np.polynomial.legendre.leggauss(n_gl)
    t = 0.5 * (t + 1); w = 0.5 * w
    l = L[:, None] * t[None]                                      # (nx, n_gl)
    r = np.sqrt(x[:, None] ** 2 + l ** 2)
    prof = yhga(r.reshape(1, -1)).reshape(len(mass), len(x), n_gl)
    mean_col = (prof * w).sum(-1)                                 # (1/L) int_0^L y dl
    fac = ne0 * r200 * MPC_CM / PC_CM / (1 + z)                   # pc cm^-3 per unit chord length (R200c)
    return fac[:, None] * mean_col, dict(c=c, eps=eps, beta=beta, fhga=fhga, mtot_over_m200=mtot_over_m200)


def bfc_gas_fraction_inside_r200(cosmo, log10m, z, par=BFC_TNG_FIDUCIAL):
    """Hot-gas mass inside R200c over M200c (for sanity checks)."""
    mass = 10 ** np.asarray(log10m, float)
    c = dutton_maccio14_c200c(cosmo, mass, z)
    nu = DELTA_C / (cosmo.sigma_M(mass) * cosmo.growth(z)[0])
    eps = par["eps0"] + par["eps1"] * nu
    q = mass * cosmo.h / 10 ** par["log10Mc"]
    beta = 3 * q ** par["mu"] / (1 + q ** par["mu"])
    fhga = bfc_fractions(cosmo, mass, par)[0]
    u = np.union1d(np.logspace(-6, 3.5, 6000), [1.0]); lu = np.log(u)
    ynfw = 1 / ((u[None] * c[:, None]) * (1 + u[None] * c[:, None]) ** 2 * (1 + (u[None] / eps[:, None]) ** 2) ** 2)
    ratio = _trapz(ynfw * u ** 3, lu, axis=1) / _trapz((ynfw * u ** 3)[:, u <= 1], lu[u <= 1], axis=1)
    y = (1 + u[None] / par["theta_c"]) ** -beta[:, None] * (1 + (u[None] / eps[:, None]) ** 1.5) ** (-par["delta"] / 1.5)
    frac_in = _trapz((y * u ** 3)[:, u <= 1], lu[u <= 1], axis=1) / _trapz(y * u ** 3, lu, axis=1)
    return fhga * ratio * frac_in


# ---------------------------------------------------------------- HalfDome profile tables (Julia)
class HalfDomeSphereTable:
    """g(x; M, z) of the HalfDome sphere products, tabulated by tabulate_halfdome_sphere_profiles.jl on the
    quadrature nodes (log10 M = 9.005 ... 15.995 step 0.01, z = 0.005 ... 0.995 step 0.01)."""

    def __init__(self, path, model):
        import h5py
        with h5py.File(path, "r") as f:
            self.log10m = np.asarray(f["log10_m200c_msun"][:])
            self.z = np.asarray(f["redshift"][:])
            self.x = np.asarray(f["x_r200c"][:])
            self.g = np.asarray(f[model + "/chord_mean_dm_pc_cm3"][:])      # (nm, nz, nx)
            self.theta200 = np.asarray(f[model + "/theta200c_rad"][:])      # (nm, nz)
            self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        assert self.g.shape == (len(self.log10m), len(self.z), len(self.x)), self.g.shape
        self.model = model

    def zi(self, z):
        i = int(np.argmin(np.abs(self.z - z)))
        assert abs(self.z[i] - z) < 1e-9, (z, self.z[i])
        return i


# ---------------------------------------------------------------- lattice engine
@dataclass
class Lattice:
    h: float = 0.002          # pc cm^-3
    n: int = 2 ** 24          # DM_max = h n = 33554 pc cm^-3
    dropped: float = 0.0      # expected crossings per ray with DM beyond the lattice (dropped; must stay negligible)
    max_dropped: float = 1e-6   # 0.1 of a ray in a 120k-ray sample

    @property
    def dm_max(self):
        return self.h * self.n


def impact_samples(n_inner=700, n_outer=900, x_min=1e-4, x_split=0.5):
    """Monotone impact-parameter edges x in [0, 1]: log-spaced in the core, uniform in the chord half-length
    s = sqrt(1-x^2) near the edge (where DM ~ s), so both the central cusp and the DM -> 0 edge are resolved."""
    inner = np.logspace(np.log10(x_min), np.log10(x_split), n_inner)
    s = np.linspace(np.sqrt(1 - x_split ** 2), 0.0, n_outer + 1)[1:]
    outer = np.sqrt(1 - s ** 2)
    return np.concatenate([[0.0], inner, outer])


def deposit_difference(lattice, dm_edges, weight_sets):
    """Deposit, for a batch of halo cells, the solid-angle measure between consecutive impact-parameter
    edges uniformly over the DM interval they span.  dm_edges: (ncell, nedge) DM at the edges (decreasing
    in x, last = 0); weight_sets: list of (ncell, nedge-1) arrays (expected number of halos per ray in each
    annulus, e.g. unweighted and bias-weighted), all sharing the same positions.
    Returns, per weight set, the *difference* array dd (length n + 2) whose cumulative sum times h is the
    lattice cell mass (expected number of crossings with DM in [k h, (k+1) h)). A uniform density rho on
    [lo, hi) is a step +rho at lo and -rho at hi, split between the two cells around each end by the
    sub-cell fraction, which is exact per cell; intervals narrower than 2h become point masses at their
    mid-point (+m/h in its cell, -m/h in the next). Differences add linearly, so the cumulative sum is taken
    once per accumulated sum (deposit_to_cells)."""
    h, n = lattice.h, lattice.n
    hi = dm_edges[:, :-1].ravel().copy(); lo = dm_edges[:, 1:].ravel().copy()
    over = hi >= lattice.dm_max - 2 * h
    weight_sets = [np.asarray(w, float).ravel() for w in weight_sets]
    if np.any(over):
        # annuli whose DM exceeds the lattice (Lee22 cusps of the rarest, most massive cells): drop them if their
        # total weight per ray is negligible, so they cannot alias; otherwise refuse
        lost = float(weight_sets[0][over].sum())
        lattice.dropped += lost
        if lattice.dropped > lattice.max_dropped:
            raise ValueError("halo DM exceeds the lattice range with non-negligible weight {:.3g} (max DM {:.1f})".format(
                lattice.dropped, hi.max()))
        weight_sets = [np.where(over, 0.0, w) for w in weight_sets]
        hi[over] = 0.0; lo[over] = 0.0
    width = hi - lo
    wide = width > 2 * h
    inv = np.where(wide, 1.0 / np.where(wide, width, 1.0), 0.0)
    klo = np.floor(lo / h).astype(np.int64); flo = lo / h - klo
    khi = np.floor(hi / h).astype(np.int64); fhi = hi / h - khi
    kpm = np.floor(0.5 * (hi + lo) / h).astype(np.int64)
    idx = np.concatenate([klo[wide], klo[wide] + 1, khi[wide], khi[wide] + 1, kpm[~wide], kpm[~wide] + 1])
    out = []
    for w in weight_sets:
        rho = w * inv
        vals = np.concatenate([rho[wide] * (1 - flo[wide]), rho[wide] * flo[wide],
                               -rho[wide] * (1 - fhi[wide]), -rho[wide] * fhi[wide],
                               w[~wide] / h, -w[~wide] / h])
        out.append(np.bincount(idx, weights=vals, minlength=n + 2)[:n + 2])
    return out


def deposit_to_cells(lattice, dd):
    return np.cumsum(dd)[:lattice.n] * lattice.h


def deposit_halos(lattice, dm_edges, weights):
    """Lattice cell masses for one weight set (see deposit_difference)."""
    return deposit_to_cells(lattice, deposit_difference(lattice, dm_edges, [weights])[0])


@dataclass
class PDFResult:
    lattice: Lattice
    p_unclustered: np.ndarray
    p_clustered: np.ndarray
    zero_unclustered: float
    zero_clustered: float
    nbar: float
    mean_dm_all_rays: float
    info: dict

    def binned(self, edges, clustered=True, normalize_range=True):
        """Probability density in the given DM bins, normalized over rays with DM inside [edges[0], edges[-1]]
        (the convention of the simulation histograms)."""
        p = self.p_clustered if clustered else self.p_unclustered
        cdf = np.concatenate([[0.0], np.cumsum(p)])
        grid = np.arange(len(cdf)) * self.lattice.h
        c = np.interp(edges, grid, cdf)
        prob = np.diff(c)
        total = prob.sum() if normalize_range else 1.0
        return prob / (total * np.diff(edges)), prob

    def stats(self, clustered=True):
        p = (self.p_clustered if clustered else self.p_unclustered).copy()
        zero = self.zero_clustered if clustered else self.zero_unclustered
        p[0] -= zero
        centers = (np.arange(len(p)) + 0.5) * self.lattice.h
        hit = p.sum()
        cdf = np.cumsum(p) / hit
        out = dict(hit_fraction=hit, mean_dm_hit=float((p * centers).sum() / hit),
                   median_dm_hit=float(np.interp(0.5, cdf, centers)))
        for t in (1, 3, 10, 30):
            out["frac_hit_below_{}".format(t)] = float(np.interp(t, centers, cdf))
        return out


def compound_poisson_pdf(lattice, slices, clustering_dz=0.05, cosmo=None, verbose=False):
    """slices: iterable of dicts with keys z, dz, dm_edges (ncell, nedge), area_weights (ncell, nedge-1),
    bias (ncell,), nbar (float), nbar_b (float). Returns PDFResult with unclustered and clustered lattices."""
    n = lattice.n
    total = np.zeros(n + 2)
    nbar = 0.0
    mean_all = 0.0
    S = cosmo.projected_variance()
    log_cl = np.zeros(n // 2 + 1, complex)
    zero_cl_term = 0.0
    n_groups = 0
    group = None

    def flush(g):
        # alpha(z, lam) integrated over the group: A = FT[bias-weighted deposit] - bias-weighted Nbar
        zc = g["z_w"] / g["dz"]
        amp = cosmo.H_over_c(zc) * cosmo.growth(zc)[0] ** 2 * S / g["dz"]
        A = np.fft.rfft(deposit_to_cells(lattice, g["dep_b"])) - g["nbar_b"]
        return 0.5 * amp * A * A, 0.5 * amp * g["nbar_b"] ** 2

    for s in slices:                       # slices must arrive ordered in z
        dd, dd_b = deposit_difference(lattice, s["dm_edges"], [s["area_weights"], s["area_weights"] * s["bias"][:, None]])
        total += dd
        nbar += s["nbar"]
        mean_all += s["mean_dm"]
        gkey = int(np.floor(s["z"] / clustering_dz + 1e-9))
        if group is not None and group["key"] != gkey:
            a, b = flush(group); log_cl += a; zero_cl_term += b; n_groups += 1
            group = None
        if group is None:
            group = dict(key=gkey, dep_b=np.zeros(n + 2), nbar_b=0.0, z_w=0.0, dz=0.0)
        group["dep_b"] += dd_b
        group["nbar_b"] += s["nbar_b"]
        group["z_w"] += s["z"] * s["dz"]; group["dz"] += s["dz"]
        if verbose:
            print("  z = {:.3f}: Nbar so far {:.4f}".format(s["z"], nbar), flush=True)
    if group is not None:
        a, b = flush(group); log_cl += a; zero_cl_term += b; n_groups += 1
    total = deposit_to_cells(lattice, total)
    log_u = np.fft.rfft(total) - nbar
    p_u = np.fft.irfft(np.exp(log_u), n)
    p_cl = np.fft.irfft(np.exp(log_u + log_cl), n)
    info = dict(n_clustering_groups=n_groups, projected_variance_mpc=S, dropped_crossings_per_ray=lattice.dropped)
    return PDFResult(lattice, p_u, p_cl, float(np.exp(-nbar)), float(np.exp(-nbar + zero_cl_term)), nbar, mean_all, info)
