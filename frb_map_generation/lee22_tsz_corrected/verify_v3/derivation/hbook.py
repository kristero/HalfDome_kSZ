"""Independent unit bookkeeping of group_particles ne_prof.cpp / y_prof.cpp / dens_prof.cpp, and candidate F_n values.
Numeric code values are computed for a toy physical halo by explicit unit conversion, not by algebra."""
import numpy as np, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency')
import lee22_models as L
from scipy.integrate import quad

h = 0.6774; XH = 0.76; Om = 0.3089; Ob = 0.0486; OL = 0.6911; fb = Ob / Om
G_cgs = 6.67430e-8; mp = 1.67262192369e-24; Msun = 1.98847e33; kpc = 3.0856775814913673e21; kms = 1e5
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# --- 1. mprot check
mprot_code = 0.6774 * 1.6726219 / 1.98847 * 1e-67     # as in ne_prof.cpp
mprot_true = 1.6726219e-27 / 1.98847e30 / 1e10 * h      # m_p in units of 1e10 Msun/h
P('mprot code', mprot_code, 'm_p in 1e10 Msun/h', mprot_true, 'ratio', mprot_code / mprot_true)
P('GNewton check [kpc (km/s)^2 / 1e10 Msun]:', G_cgs * 1e10 * Msun / kpc / kms**2, 'code 4.30091e4')
rc0 = 3 * (100 * h * kms / (1e3 * kpc))**2 / (8 * np.pi * G_cgs)   # g/cm^3 physical
rc0_code = rc0 / (1e10 * Msun / h) * (kpc / h)**3
P('rho_crit_0 in 1e10 Msun/h/(kpc/h)^3:', rc0_code, 'code 2.775e-8')

# --- 2. toy halo at redshift z: physical quantities in cgs, then convert to code numbers explicitly
for z in [0.0, 0.47, 1.05, 1.48]:
    a = 1 / (1 + z)
    rho_c = rc0 * (Om / a**3 + OL)
    M = 10**13.75 * Msun
    R = (3 * M / (4 * np.pi * 200 * rho_c))**(1 / 3)          # physical cm
    # a shell from 0.5R to 0.6R containing gas mass Mg with electron abundance x and specific energy u
    r1, r2 = 0.5 * R, 0.6 * R
    Vphys = 4 * np.pi / 3 * (r2**3 - r1**3)
    Mg = 1e12 * Msun; x = 1.158; u = (500 * kms)**2
    Ne_true = x * XH * Mg / mp
    ne_true = Ne_true / Vphys                                  # cm^-3
    Pe_true = 4 * x * XH / (1 + 3 * XH + 4 * XH * x) * (2 / 3) * Mg * u / Vphys   # erg/cm^3
    n200_true = 200 * rho_c * fb / (XH * mp)                   # cm^-3, as printed
    P200_true = 200 * G_cgs * M * rho_c * fb / (2 * R)
    # code numbers: lengths ckpc/h, masses 1e10 Msun/h, energies (km/s)^2
    L_code = lambda r: r / kpc * h / a
    m_code = Mg / (1e10 * Msun) * h
    M_code = M / (1e10 * Msun) * h
    R_code = L_code(R)
    V_code = 4 * np.pi / 3 * (L_code(r2)**3 - L_code(r1)**3)
    e_code = u / kms**2
    # ne_prof
    ne_code = x * XH * m_code / mprot_code / V_code
    # y_prof
    Y_code = 4 * x * XH / (1 + 3 * XH + 4 * XH * x) * (2 / 3) * m_code * e_code
    Pe_code = Y_code / V_code
    rhoc_code = 2.775e-8 * (Om / a**3 + OL)
    P200_code = 100 * 4.30091e4 * M_code * rhoc_code * Ob / Om / R_code
    # external n200 options
    n200_kpc_phys = n200_true * kpc**3                  # per physical kpc^3, h-free
    n200_codeunits = 200 * rhoc_code * fb / (XH * mprot_code)   # per (physical kpc/h)^3, same helper
    ne_true_kpc = ne_true * kpc**3
    P('z=%.2f a=%.4f' % (z, a))
    P('  ne_code / ne_true[kpc^-3] =', ne_code / ne_true_kpc, ' (a/h)^3 =', (a / h)**3)
    P('  ratio(ne_code/n200 h-free phys) / (ne/n200)_true =', (ne_code / n200_kpc_phys) / (ne_true / n200_true), '  a^3/h^3 =', a**3 / h**3)
    P('  ratio(ne_code/n200 same code helper) / true =', (ne_code / n200_codeunits) / (ne_true / n200_true), '  a^3 =', a**3)
    P('  Pe_code / Pe_true[1e10Msun (km/s)^2/kpc^3] =', Pe_code / (Pe_true / (1e10 * Msun) / kms**2 * kpc**3), ' a^3/h^2 =', a**3 / h**2)
    P('  P200_code / P200_true =', P200_code / (P200_true / (1e10 * Msun) / kms**2 * kpc**3), ' a/h^2 =', a / h**2)
    P('  (Pe/P200)_code / true =', (Pe_code / P200_code) / (Pe_true / P200_true), ' a^2 =', a**2)
    # dens_prof (gas mass density), then various external conversions
    rho_code = m_code / V_code
    rho_true_code_units = Mg / Vphys / (1e10 * Msun) * kpc**3     # 1e10 Msun / kpc^3
    P('  dens_prof rho_code / rho_true[1e10Msun/kpc^3] =', rho_code / rho_true_code_units, ' a^3/h^2 =', a**3 / h**2)

# --- 3. candidates
mue = 2 / (1 + XH); mu = 4 / (3 + 5 * XH)
cands = {
 '1/h^3': h**-3, 'h^-4': h**-4, 'h^-2': h**-2,
 '1/(XH h^3)': 1 / (XH * h**3), 'mu_e/(XH h^3)': mue / (XH * h**3), '1/(mu h^3)': 1 / (mu * h**3),
 '1/(h^3 XH^2)': 1 / (h**3 * XH**2), '1/(h^2 XH^3)': 1 / (h**2 * XH**3), 'Om/Ob': Om / Ob,
 '1/(h^3 XH^3)': 1 / (h**3 * XH**3), 'x_max/(XH h^3)': (1 + (1 - XH) / (2 * XH)) / (XH * h**3),
 '1/(h^4 XH)': 1 / (h**4 * XH), '1/(h^3 XH) mu_e^2': mue**2 / (XH * h**3),
}
meas = np.array([5.4, 5.6, 5.5, 5.1]); meas5 = np.array([5.4, 5.6, 5.5, 5.1, 5.9])
P('\nmeasured F_n mean(4)=%.3f  mean(5)=%.3f  range %.1f-%.1f' % (meas.mean(), meas5.mean(), meas5.min(), meas5.max()))
for k, v in sorted(cands.items(), key=lambda kv: kv[1]):
    P('  %-20s %.3f   /mean4 = %.3f  /mean5 = %.3f  /min = %.3f /max = %.3f' % (k, v, v / meas.mean(), v / meas5.mean(), v / meas5.min(), v / meas5.max()))

# --- 4. implied gas fraction from the printed fit (z=0), printed n200, full ionisation
P('\nImplied f_gas(<R200c)/f_b from the printed Lee22 no-c density fit, n_e = nbar * n200_printed, gas fully ionised:')
for lm in [13.5, 13.75, 14.0]:
    mh = 10**lm * h
    I = quad(lambda xx: 3 * xx**2 * L.lee_nbar(xx, mh, 0.0), 0, 1, limit=200)[0]
    fimp = mue / XH * I
    P('  log M200c/Msun=%.2f  3int nbar x^2 = %.3f  implied f_gas/f_b = %.3f ; /5.57 -> %.3f ; /4.81 -> %.3f ; /6.36 -> %.3f' % (lm, I, fimp, fimp / 5.57, fimp / 4.81, fimp / 6.36))
open('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v3_20261001/derivation/hbook.txt', 'w').write('\n'.join(out) + '\n')
