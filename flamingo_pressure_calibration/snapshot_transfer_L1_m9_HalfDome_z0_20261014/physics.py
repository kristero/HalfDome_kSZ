"""Physical units, cosmology transfer and Battaglia shapes for this study."""
import json
from pathlib import Path
import numpy as np
from scipy.constants import G, c, electron_mass, proton_mass, physical_constants

MPC_M = 3.08567758e22
MPC_CM = MPC_M * 100.
MSUN_KG = 1.98841e30
SIGMA_T = physical_constants['Thomson cross section'][0]
T_CMB = 2.7255
MU_E_KG = (electron_mass + 2 * proton_mass / 1.76) / .9
Y_TO_P_VOLUME = electron_mass * c**2 / (SIGMA_T * MPC_M)
NE_VOLUME_TO_TAU_AREA = SIGMA_T * 1e4 * MPC_CM
SOURCE = dict(h=.681, omega_m=.304611 + .0013890794316214667,
              omega_b=.0486, neutrino_mass_ev=.06)
TARGET = dict(h=.6774, omega_m=.3089, omega_b=.0486, neutrino_mass_ev=0.)
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393])
B16 = np.array([4e3, .88, 3.83, .29, -.03, .04])
BOUNDS = {'pressure': ([.001, .01, .5], [1e4, 10., 20.]),
          'electrons': ([.1, .1, .5], [1e7, 10., 20.])}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.temporary')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def critical_density(cosmology):
    """Critical density at z=0, in physical solar masses per Mpc cubed."""
    h_si = 100 * cosmology['h'] * 1000 / MPC_M
    return 3 * h_si**2 / (8 * np.pi * G) * MPC_M**3 / MSUN_KG


def scales(mass, cosmology=TARGET):
    """R200c [Mpc], P200 [Pa], electron normalization [cm^-3]."""
    mass = np.asarray(mass, float)
    rho_c = critical_density(cosmology)
    fb = cosmology['omega_b'] / cosmology['omega_m']
    radius = (mass / (4 * np.pi / 3 * 200 * rho_c))**(1/3)
    p200 = G * mass * MSUN_KG * 200 * rho_c * MSUN_KG / MPC_M**3 * fb / (2 * radius * MPC_M)
    # The fitted B16 electron shape is mu_XGP n_e / (rho_c f_b).
    ne0 = rho_c * fb * MSUN_KG / MPC_CM**3 / MU_E_KG
    return radius, p200, np.full_like(mass,ne0)


def shape(field, x, pars):
    x = np.maximum(np.asarray(x), 1e-12)
    p, a, b = np.asarray(pars)
    if field == 'pressure':
        q = x / a
        return .5176 * p * q**(-.3) * (1 + q)**(-b)
    q = x / .5
    return p * q**(-.2) * (1 + q**a)**(-(b - .2) / a)


def battaglia(field, mass):
    p = B12 if field == 'pressure' else B16
    return p[:3] * (np.asarray(mass)[..., None] / 1e14)**p[3:6]


def cosmology_ratios():
    fb = (TARGET['omega_b']/TARGET['omega_m']) / (SOURCE['omega_b']/SOURCE['omega_m'])
    h = TARGET['h'] / SOURCE['h']
    return dict(radius=h**(-2/3), pressure=fb*h**(8/3), electrons=fb*h**2,
                projected_y=fb*h**2, tau_and_dm=fb*h**(4/3),
                integrated_Y=fb*h**(2/3))


def map_observables(sums, pixel_area):
    """Convert [Y, n_e V, n_e v_z V] deposits to y,tau,DM,DeltaT."""
    y = sums[0] / pixel_area
    tau = NE_VOLUME_TO_TAU_AREA * sums[1] / pixel_area
    dm = 1e6 * sums[1] / pixel_area
    dt = -T_CMB * 1e6 / (c / 1000) * NE_VOLUME_TO_TAU_AREA * sums[2] / pixel_area
    return np.asarray([y, tau, dm, dt])
