"""Python tau(theta) (projected and 4 R200c sphere) for comparison with xgpaint_reference_values.jl output."""
import sys
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/painting_geometry")
import numpy as np
from onehalo_variants import *
from scipy.integrate import quad
for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0), (3e13, 2.0)):
    P0, xc, al, be, ga = b16_params(M, z)
    R = r200c_m(M, z); d = dA(z) * MPC; th = np.arctan(R / d)
    A = NE_PER_KG * f_b * rho_c(z) * P0
    for fr in (0.1, 0.5, 1.0, 2.0, 3.5):
        pr = 2 * quad(lambda l: gnfw(np.hypot(fr, l), xc, al, be, ga), 0, 1e5, limit=500, points=[1, 10, 100])[0]
        L = np.sqrt(16 - fr**2)
        sp = 2 * quad(lambda l: gnfw(np.hypot(fr, l), xc, al, be, ga), 0, L, limit=500)[0]
        print("M=%g z=%g theta200=%.10g f=%g tau_proj=%.8e tau_sphere4=%.8e" % (M, z, th, fr, SIGMA_T * A * R * pr, SIGMA_T * A * R * sp))
