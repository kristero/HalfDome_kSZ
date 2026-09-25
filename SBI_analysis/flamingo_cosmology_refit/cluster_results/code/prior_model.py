"""Pressure conventions and proposed prior, independent of trained emulators.

P0 is total thermal pressure / P200. Electron pressure is 0.5176 times
thermal pressure, matching the archived XGPaint campaign. beta is the raw
GNFW exponent; the asymptotic outer slope is beta + 0.3.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, betainc

NAMES = ("P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
         "alpha_z_P0", "alpha_z_xc", "alpha_z_beta")
LOW = np.array([1., .1, 2.8, -.2, -.6, -.2, -4.5, -1.5, -.5])
HIGH = np.array([60., 4., 16., 1.5, .4, .4, .5, 2., 1.5])
FIDUCIAL = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
KEYS = ("battaglia_P0_amp", "battaglia_x_c_amp", "battaglia_beta_amp",
        "battaglia_P0_alpha_m", "battaglia_x_c_alpha_m", "battaglia_beta_alpha_m",
        "battaglia_P0_alpha_z", "battaglia_x_c_alpha_z", "battaglia_beta_alpha_z")
G = 6.67430e-11
MSUN = 1.98847e30
MPC = 3.085677581491367e22
SIGMA_T = 6.6524587321e-29
MEC2 = 8.1871057769e-14
H = .68
OM = .31
OB = .049
ELECTRON_FRACTION = .5176
OMEGA_PHOTON = 4.48131e-7*2.7255**4/H**2
OMEGA_RADIATION = OMEGA_PHOTON*(1+3.04*(7/8)*(4/11)**(4/3))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def from_unit(u):
    """Unit-cube coordinates: logarithmic P0/xc; linear other parameters."""
    u = np.asarray(u)
    result = LOW + u*(HIGH-LOW)
    result[..., :2] = np.exp(np.log(LOW[:2])+u[..., :2]*np.log(HIGH[:2]/LOW[:2]))
    return result


def to_unit(theta):
    theta = np.asarray(theta)
    result = (theta-LOW)/(HIGH-LOW)
    result[..., :2] = np.log(theta[..., :2]/LOW[:2])/np.log(HIGH[:2]/LOW[:2])
    return result


def parameters(theta, mass, z):
    """Return P0, xc, beta at physical M200c in Msun and redshift z."""
    theta = np.asarray(theta)
    m = np.asarray(mass)/1e14
    zp = 1+np.asarray(z)
    return tuple(theta[..., i, None]*m**theta[..., i+3, None]
                 * zp**theta[..., i+6, None] for i in range(3))


def domain_metrics(theta, logmass=(12., 15.7), redshift=(.001, 5.)):
    # Power laws attain extrema at these four corners; a dense grid is
    # unnecessary for the positivity and asymptotic-slope conditions.
    mass = 10**np.array([logmass[0], logmass[0], logmass[1], logmass[1]])
    z = np.array([redshift[0], redshift[1], redshift[0], redshift[1]])
    p0, xc, beta = parameters(theta, mass, z)
    return dict(min_beta=beta.min(axis=-1), max_beta=beta.max(axis=-1),
                min_xc=xc.min(axis=-1), max_xc=xc.max(axis=-1),
                min_p0=p0.min(axis=-1), max_p0=p0.max(axis=-1))


def pressure_shape(radius, xc, beta):
    q = np.asarray(radius)/xc
    return np.exp(-.3*np.log(q)-beta*np.log1p(q))


def central_column(p0, xc, beta, cutoff=np.inf):
    """Analytic central dimensionless LOS integral, or finite direct integral.

    Returns integral P/P200 d(l/R200). Infinity is intentional for a
    divergent untruncated column and is never sent to the production painter.
    """
    if beta > .7:
        total = 2*p0*xc*np.exp(betaln(.7, beta-.7))
        if np.isinf(cutoff):
            return total
        return total*betainc(.7, beta-.7, cutoff/(cutoff+xc))
    if np.isinf(cutoff):
        return np.inf
    # Substitution r=t^(1/0.7) removes the integrable central cusp.
    limit = cutoff**.7
    f = lambda t: xc**.3/.7*(1+t**(1/.7)/xc)**(-beta)
    return 2*p0*quad(f, 0, limit, epsabs=0, epsrel=1e-9, limit=300)[0]


def finite_pressure_integral(p0, xc, beta, radius=1., order=96):
    """Integral_0^radius P/P200 x^2 dx, with no infinite-volume assumption."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    # Log radius in units of xc resolves compact cores at extreme beta/xc.
    # A fixed linear-radius rule would miss those cores and bias the audit.
    lower = np.log(1e-12)
    upper = np.log(radius/np.asarray(xc))
    half_width = (upper-lower)/2
    t = lower+(nodes+1)*half_width[..., None]
    values = np.exp(2.7*t-np.asarray(beta)[..., None]*np.logaddexp(0, t))
    return np.asarray(p0)*np.asarray(xc)**3*half_width*np.sum(values*weights, axis=-1)


def expansion_e2(z):
    """H(z)^2/H0^2, including XGPaint's default photons and massless neutrinos."""
    zp = 1+np.asarray(z)
    return OM*zp**3+OMEGA_RADIATION*zp**4+1-OM-OMEGA_RADIATION


def physical_scales(mass, z, include_radiation=True):
    mass = np.asarray(mass)
    z = np.asarray(z)
    e2 = expansion_e2(z) if include_radiation else OM*(1+z)**3+1-OM
    hz = H*100*1000/MPC*np.sqrt(e2)
    rho = 3*hz*hz/(8*np.pi*G)
    r200 = (3*mass*MSUN/(4*np.pi*200*rho))**(1/3)
    p200 = G*mass*MSUN*200*rho*(OB/OM)/(2*r200)
    return r200, p200


def bin_cl(cl):
    cl = np.asarray(cl, dtype=np.float64)
    ell = np.arange(len(cl))
    dl = ell*(ell+1)*cl/(2*np.pi)
    edges = np.append(np.arange(80, 7881, 200), 7980)
    values, centres = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        e = ell[left:right]
        weights = 2*e+1
        values.append(np.average(dl[left:right], weights=weights))
        centres.append(np.average(e, weights=weights))
    return np.array(centres), np.array(values)


def comparison_metrics(prediction, target):
    ratio = np.asarray(prediction)/np.asarray(target)
    return dict(rms_fractional=float(np.sqrt(np.mean((ratio-1)**2))),
                max_fractional=float(np.max(abs(ratio-1))),
                rms_log=float(np.sqrt(np.mean(np.log(ratio)**2))))
