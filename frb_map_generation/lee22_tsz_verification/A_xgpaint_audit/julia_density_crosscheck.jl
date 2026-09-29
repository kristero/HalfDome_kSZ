# Workstream A, goal 3 cross-check: evaluate the PRODUCTION Julia density code (lee2022_frb_dm_profile.jl,
# included read-only; XGPaint unchanged) at table grid points, for comparison with profile_tables.npz
# (make_profile_tables.py reads the CSV written here).
using XGPaint
using Unitful, UnitfulAstro
using Printf
const XG = XGPaint
include("/home/cbllover/HalfDome/frb_map_generation/lee2022_frb_dm_profile.jl")

const OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit"
const H = 0.68
clip = 10.0^14.8 / H
lit = Lee2022NoConcentrationDMProfile(normalization=:literal, n0_pivot=:mcut, shape_clip_mass_msun=clip, redshift_scaling=:physical)
ne2d = Lee2022NoConcentrationDMProfile(normalization=:xgpaint_ne2d, n0_pivot=:mcut, shape_clip_mass_msun=clip, redshift_scaling=:physical)
tng = Lee2022NoConcentrationDMProfile(normalization=:literal, n0_pivot=:mcut, shape_clip_mass_msun=clip, redshift_scaling=:tng_calibrated)
pipe = Battaglia16DensityDMProfile(Lee2022XGPaintDMProfile(ne2d))     # the XGPaint get_params/ne2d route used in production
b16 = Battaglia16DensityDMProfile(XG.HaloDMProfile(XG.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)))
variants = [("lee22_literal", lit), ("lee22_xgpaint_ne2d", ne2d), ("lee22_xgpaint_pipeline", pipe), ("lee22_tngcal", tng),
            ("b16_xgpaint", b16)]

xgrid = exp.(range(log(0.01), log(5.0), length=60))
open(joinpath(OUT, "julia_production_density_crosscheck.csv"), "w") do io
    println(io, "variant,log10M,z,x,ne_m3,rho_crit_xgpaint_kg_m3")
    for (name, model) in variants, lm in (12.5, 13.5, 14.5, 15.5), z in (0.0, 0.5, 1.0, 3.0), k in (1, 21, 41, 60)
        x = xgrid[k]
        ne = halo_electron_density_m3(model, x, 10.0^lm, z)
        rho = Float64(ustrip(uconvert(u"kg/m^3", XG.ρ_crit(model isa Battaglia16DensityDMProfile ? model.inner : model, z))))
        @printf(io, "%s,%.2f,%.2f,%.12e,%.12e,%.12e\n", name, lm, z, x, ne, rho)
    end
end
println("normalization factor (xgpaint_ne2d) = ", lee2022_normalization_factor(ne2d))
println("C(z=1) tng_calibrated = ", lee2022_redshift_scaling_factor(tng, 1.0), "  C(z=3) = ", lee2022_redshift_scaling_factor(tng, 3.0))
println("wrote julia_production_density_crosscheck.csv")
