"""Linear-theory LOS velocity dispersion sigma_v(z) from the saved CLASS power table.

tab_pklin_mpch3[z, ell] holds P_lin(k=(ell+1/2)/chi(z), z) in (Mpc/h)^3 (the
class_sz Limber table used by lightcone_analytic). Each row is scaled to z=0
with the linear growth D(z) of the same flat cosmology (Omega_m=0.31, h=0.68,
no massive neutrinos, radiation included in H) and the rows are merged into
P0(k). Checks: rows collapse onto one P0(k), and sigma_8 of P0 equals the
CLASS value 0.8159. Then

  sigma_v,1D(z) = f(z) a H(z) D(z) [int P0(k) dk / (6 pi^2)]^(1/2)

(one Cartesian component of the linear 3D velocity dispersion).
"""
from __future__ import annotations

import json

import numpy as np
from scipy import integrate, interpolate

OMEGA_M = .31
H_LITTLE = .68
N_UR = 3.046
T_CMB_K = 2.7255


def omega_radiation():
    gamma = 2.4728e-5*(T_CMB_K/2.7255)**4/H_LITTLE**2      # Omega_gamma h^2 = 2.4728e-5
    return gamma*(1+.22710731766*N_UR)


def expansion(z):
    """E(z) = H(z)/H0 for flat LCDM with radiation."""
    r = omega_radiation()
    return np.sqrt(OMEGA_M*(1+z)**3+r*(1+z)**4+(1-OMEGA_M-r))


def growth(z):
    """Linear growth D(z)/D(0) and f = dlnD/dlna from the standard ODE in ln a."""
    r = omega_radiation()
    def rhs(lna, y):
        a = np.exp(lna)
        e2 = OMEGA_M*a**-3+r*a**-4+(1-OMEGA_M-r)
        dlne2 = (-3*OMEGA_M*a**-3-4*r*a**-4)/e2
        d, dprime = y
        return [dprime, -(2+.5*dlne2)*dprime+1.5*OMEGA_M*a**-3/e2*d]
    a0 = 1e-3
    solution = integrate.solve_ivp(rhs, (np.log(a0), 0.), [a0, a0], rtol=1e-10, atol=1e-13,
                                   dense_output=True)
    lna = -np.log1p(np.asarray(z, float))
    d, dprime = solution.sol(lna)
    d0 = solution.sol(0.)[0]
    return d/d0, dprime/d


def merged_power(arrays):
    """P0(k) [(Mpc/h)^3] on a log-k grid [h/Mpc] from all positive table rows."""
    z = arrays["tab_z"]
    ell = arrays["ell"]
    chi_h = arrays["tab_da_mpc"]*(1+z)*H_LITTLE
    d, _ = growth(z)
    k = (ell[None, :]+.5)/chi_h[:, None]
    p0 = arrays["tab_pklin_mpch3"]/d[:, None]**2
    use = arrays["tab_pklin_mpch3"] > 0
    grid = np.geomspace(k[use].min(), min(k[use].max(), 50.), 400)
    logk, logp = np.log(k[use]), np.log(p0[use])
    order = np.argsort(logk)
    logk, logp = logk[order], logp[order]
    # Median of all row estimates in narrow log-k bins (rows overlap heavily).
    edges = np.log(np.geomspace(grid[0]/1.01, grid[-1]*1.01, 401))
    which = np.searchsorted(edges, logk)-1
    median = np.array([np.median(logp[which == i]) if np.any(which == i) else np.nan for i in range(400)])
    centres = .5*(edges[1:]+edges[:-1])
    good = np.isfinite(median)
    # Row-to-row scatter about the merged curve: the collapse check, restricted to
    # k <= 10 h/Mpc (beyond it some low-z rows hold numerical-noise tails).
    spline = interpolate.UnivariateSpline(centres[good], median[good], s=0, k=3)
    scatter = (logp-spline(logk))[np.exp(logk) <= 10.]
    return dict(k=np.exp(centres[good]), p0=np.exp(median[good]), spline=spline,
                collapse_rms=float(np.sqrt(np.mean(scatter**2))),
                collapse_p99=float(np.percentile(np.abs(scatter), 99)),
                k_range=[float(np.exp(logk[0])), float(np.exp(logk[-1]))])


def integrals(power, n_s=.9667):
    """sigma_8 and int P0 dk with power-law extensions beyond the table range."""
    logk = np.log(power["k"])
    lo, hi = logk[0], logk[-1]
    fine = np.linspace(lo, hi, 20000)
    p = np.exp(power["spline"](fine))
    kk = np.exp(fine)
    # Low k: P ~ k^n_s; high k: local log slope of the last decade.
    low = np.exp(power["spline"](lo))*power["k"][0]/(n_s+1)
    slope = (power["spline"](hi)-power["spline"](hi-np.log(10.)))/np.log(10.)
    high = -np.exp(power["spline"](hi))*power["k"][-1]/(slope+1) if slope < -1 else np.nan
    p_dk = integrate.trapezoid(p*kk, fine)+low+high
    x = 8.*kk
    window = 3*(np.sin(x)-x*np.cos(x))/x**3
    sigma8 = np.sqrt(integrate.trapezoid(kk**3*p*window**2, fine)/(2*np.pi**2))
    return dict(int_P_dk_Mpc_h2=float(p_dk), low_k_tail_fraction=float(low/p_dk),
                high_k_tail_fraction=float(high/p_dk), high_k_log_slope=float(slope), sigma8=float(sigma8))


def sigma_v_los(z, arrays):
    """1D linear velocity dispersion [km/s] at redshifts z, plus diagnostics."""
    power = merged_power(arrays)
    moments = integrals(power)
    d, f = growth(z)
    hubble = 100*H_LITTLE*expansion(np.asarray(z))
    rms_displacement = np.sqrt(moments["int_P_dk_Mpc_h2"]/(6*np.pi**2))/H_LITTLE     # Mpc
    sigma = f/(1+np.asarray(z))*hubble*d*rms_displacement
    diagnostics = dict(collapse_rms_dex_k_le_10=power["collapse_rms"]/np.log(10.),
                       collapse_p99_abs_dex_k_le_10=power["collapse_p99"]/np.log(10.),
                       table_k_range_h_per_Mpc=power["k_range"], **moments,
                       sigma8_class=float(json.loads(str(arrays["metadata_json"]))["sigma8"]),
                       definition="sigma_v,1D = f a H D sqrt(int P0 dk/(6 pi^2)); P0 merged from the CLASS Limber table rows")
    return sigma, diagnostics
