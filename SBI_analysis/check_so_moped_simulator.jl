# Lightweight environment/parameter preflight. No maps or halo catalogue are read.
using XGPaint, Healpix, HDF5, Interpolations, TOML

model = Battaglia16ThermalSZProfile(
    Omega_c=0.261, Omega_b=0.049, h=0.68,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758,
    x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0,
)
@assert model.P0.amp == 18.1
@assert model.x_c.amp == 0.497
@assert model.beta.amp == 4.35
@assert model.P0.alpha_m == 0.154
@assert model.P0.alpha_z == -0.758
@assert model.x_c.alpha_m == -0.00865
@assert model.x_c.alpha_z == 0.731
@assert model.beta.alpha_m == 0.0393
@assert model.beta.alpha_z == 0.415
TOML.print(stdout, Dict("xgpaint_source" => pathof(XGPaint), "julia_version" => string(VERSION),
                        "parameter_constructor_check" => true))
