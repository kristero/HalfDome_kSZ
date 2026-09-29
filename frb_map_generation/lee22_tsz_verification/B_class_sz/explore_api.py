#!/usr/bin/env python3
"""One-off probe of the classy_sz 0.1.70.post40 accessor units (P_lin, dn/dlnM, bias, y_ell, P200) and of the
neutrino override. Usage: explore_api.py [massless|cosmo_model]"""
import faulthandler, sys, time
import numpy as np
faulthandler.enable()
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C
from classy_sz import Class

mode = sys.argv[1] if len(sys.argv) > 1 else "massless"
pars = C.class_sz_base(C.PAINT, output="tSZ_1h,tSZ_2h,dndlnM", massless_nu=(mode == "massless"))
pars.update(C.b12_class_sz_pars())
c = Class()
for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
    c.pars.pop(k, None)
c.set(pars)
print("pars after set:", {k: c.pars.get(k) for k in ("N_ur", "N_ncdm", "m_ncdm", "deg_ncdm", "cosmo_model")}, flush=True)
t0 = time.time(); c.compute(); print("compute", time.time() - t0, flush=True)
h = c.h()
print("h", h, "Omega_m", c.Omega_m(), "f_b", c.get_f_b(), "sigma8", c.sigma8(), "Neff", c.Neff(), flush=True)
z = 0.5; k = 0.1; m = 1e14
print("pk_lin classy (k 1/Mpc, Mpc^3):", c.pk_lin(k, z), flush=True)
print("get_pk_lin_at_k_and_z(k=0.1):", c.get_pk_lin_at_k_and_z(k, z), " vs pk_lin(k*h)*h^3", c.pk_lin(k * h, z) * h ** 3, flush=True)
print("dndlnM(z, 1e14 Msun/h)", c.get_dndlnM_at_z_and_M(z, m), flush=True)
print("bias", c.get_first_order_bias_at_z_and_m(z, m), flush=True)
print("P_delta b12 (m=1e14 Msun/h, z)", c.get_P_delta_at_m_and_z_b12(m, z), flush=True)
print("P/P200 at x=0.5", c.get_pressure_P_over_P_delta_at_x_M_z_b12_200c(0.5, m, z), flush=True)
print("r200c(1e14)", c.get_r_delta_of_m_delta_at_z(200, m, z), flush=True)
print("rho_crit(z)", c.get_rho_crit_at_z(z), flush=True)
print("D_A(z) Mpc", c.angular_distance(z), " H(z) 1/Mpc", c.Hubble(z), flush=True)
for l in (100.0, 1000.0, 3000.0):
    print("dydzdlnm l=", l, c.get_dydzdlnm_at_z_and_m(z, m, l), flush=True)
r = c.cl_sz()
ell = np.asarray(r["ell"]); i = np.argmin(abs(ell - 3000))
print("ell n", ell.size, ell[:3], ell[-3:], "1h(3000)", r["1h"][i], "2h", r["2h"][i], flush=True)
c.struct_cleanup(); c.empty()
