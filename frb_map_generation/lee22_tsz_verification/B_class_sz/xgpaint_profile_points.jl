# XGPaint Compton-y of the production Battaglia12 and Lee22 profiles at a few (M200c, z, theta) points, plus the
# geometry XGPaint uses (R200c physical, D_A, rho_crit, theta200), for the independent Python check in
# profile_checks.py. Models are constructed exactly as the production painter does
# (paint_halfdome_battaglia12_tsz_map.jl: Battaglia16ThermalSZProfile with Omega_c=0.261, Omega_b=0.049, h=0.68;
# lee2022_tsz_pressure_profile.jl: Lee2022ThermalSZProfile with the same cosmology and the 10^14.8/h shape clip).
# Run: JULIA_DEPOT_PATH=/home/kn18001/.julia /home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia xgpaint_profile_points.jl OUT.csv
using XGPaint
using Unitful, UnitfulAstro
include("/home/cbllover/HalfDome/frb_map_generation/lee2022_tsz_pressure_profile.jl")

const OMEGA_C, OMEGA_B, HV = 0.261, 0.049, 0.68
b12 = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=HV,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758, x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0, beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)
lee = Lee2022ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=HV)

out = length(ARGS) > 0 ? ARGS[1] : "xgpaint_profile_points.csv"
open(out, "w") do io
    println(io, "model,M200c_msun,z,x_over_theta200,theta_rad,y,R200_mpc,DA_mpc,rho_crit_g_cm3,theta200_rad,f_b")
    for (name, model) in (("b12", b12), ("lee22", lee))
        for M in (1.0e13, 1.0e14, 1.0e15), z in (0.1, 0.5, 1.0, 2.0)
            Mu = M * XGPaint.M_sun
            r200 = XGPaint.R_Δ(model, Mu, z, 200)
            th200 = XGPaint.angular_size(model, r200, z)
            da = ustrip(u"Mpc", XGPaint.angular_diameter_dist(model.cosmo, z))
            rc = ustrip(u"g/cm^3", XGPaint.ρ_crit(model, z))
            for x in (0.05, 0.3, 1.0, 2.0, 3.9)
                th = x * th200
                y = name == "b12" ? XGPaint.compton_y(model, th, Mu, z) : model(th, M, z)
                println(io, join((name, M, z, x, th, Float64(y), ustrip(u"Mpc", r200), da, rc, th200, model.f_b), ","))
            end
        end
    end
end
println("wrote ", out)
