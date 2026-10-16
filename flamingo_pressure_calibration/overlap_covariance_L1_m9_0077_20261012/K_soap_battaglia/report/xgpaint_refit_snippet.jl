# Test K: Battaglia laws refitted to FLAMINGO L1_m9 SOAP-HBT (results/k_params.json); XGPaint branch 'cluster'
# (checked: report/xgpaint_snippet_check.json, Julia 1.12.2 + /home/kn18001/.julia/dev/XGPaint).
# Masses in Msun (no h), pivot 1e14 Msun, x = r/R200c.  The profiles scale with the painter's f_b = Omega_b/(Omega_b+Omega_c)
# (fitted with FLAMINGO's f_b = 0.0486/0.306 = 0.15882; HalfDome's 0.0486/0.3089 = 0.15733 paints 0.9906 of it).
using XGPaint
Omega_b, Omega_c, h = 0.0486, 0.3089 - 0.0486, 0.6774      # HalfDome lightcone cosmology

# tSZ_recommended: P|R3|brk_amp  (sigma_d = 0.0735)
model_tSZ_recommended = XGPaint.BreakModel(; Omega_c=Omega_c, Omega_b=Omega_b, h=h, alpha_break=0.1705458673, M_break=1.066301902e+14, 
    P0_amp=3.768591428, P0_alpha_m=0.8734859711, P0_alpha_z=1.273020111,
    x_c_amp=1.376261012, x_c_alpha_m=-0.5005433699, x_c_alpha_z=-0.5068782532,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0,
    beta_amp=5.625150206, beta_alpha_m=-0.1121537912, beta_alpha_z=0.05611912463,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)

# tSZ_published_form: P|R3|orig  (sigma_d = 0.0852)
model_tSZ_published_form = XGPaint.Battaglia16ThermalSZProfile(; Omega_c=Omega_c, Omega_b=Omega_b, h=h, 
    P0_amp=3.706751285, P0_alpha_m=0.9527808103, P0_alpha_z=1.221554847,
    x_c_amp=1.254348043, x_c_alpha_m=-0.3580825557, x_c_alpha_z=-0.4145351884,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0,
    beta_amp=5.320689611, beta_alpha_m=0.0008185780002, beta_alpha_z=0.1142784895,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)

# gas_recommended: G|R3|free  (sigma_d = 0.0652)
model_gas_recommended = XGPaint.BattagliaTauProfilePhysical(; Omega_c=Omega_c, Omega_b=Omega_b, h=h, 
    P0_amp=169.6309533, P0_alpha_m=0.8189640054, P0_alpha_z=0.6503373445,
    x_c_amp=0.6305462983, x_c_alpha_m=-0.3426882647, x_c_alpha_z=-0.3093766085,
    alpha_amp=3.651666429, alpha_alpha_m=0.6216398595, alpha_alpha_z=1.125701422,
    beta_amp=2.372904328, beta_alpha_m=0.04584424198, beta_alpha_z=0.03856412465,
    gamma_amp=-0.8509139444, gamma_alpha_m=-0.1042450083, gamma_alpha_z=0.3869529378)
# gas density -> particle electrons with XGPaint (n_e = 0.9 rho_gas/mu_e): P0_amp = 169.6309533 / (0.9 r), r = particle-measured
# [M_gas/mu_e]/N_e at R200c (k4_electron_factors.json): 1.0433 (10^12.5-13.5) -> P0_amp = 180.6617044; 1.0185 (10^13.5-14.5) -> 185.0470581; 1.0166 (>= 10^14.5) -> 185.4048182

# gas_published_form: G|R3|orig  (sigma_d = 0.1258)
model_gas_published_form = XGPaint.BattagliaTauProfilePhysical(; Omega_c=Omega_c, Omega_b=Omega_b, h=h, 
    P0_amp=8380.49951, P0_alpha_m=-0.2911916093, P0_alpha_z=-0.8045932279,
    x_c_amp=0.5, x_c_alpha_m=0.0, x_c_alpha_z=0.0,
    alpha_amp=0.4521043232, alpha_alpha_m=0.235889789, alpha_alpha_z=0.7294637325,
    beta_amp=2.906195836, beta_alpha_m=0.08226887786, beta_alpha_z=0.2786447643,
    gamma_amp=-0.2, gamma_alpha_m=0.0, gamma_alpha_z=0.0)
# gas density -> particle electrons with XGPaint (n_e = 0.9 rho_gas/mu_e): P0_amp = 8380.49951 / (0.9 r), r = particle-measured
# [M_gas/mu_e]/N_e at R200c (k4_electron_factors.json): 1.0433 (10^12.5-13.5) -> P0_amp = 8925.466112; 1.0185 (10^13.5-14.5) -> 9142.121466; 1.0166 (>= 10^14.5) -> 9159.796355

# gas_smooth_alternative (G|R3|brk_all): broken power laws are not an XGPaint type; evaluate with k_fit_core.Law

