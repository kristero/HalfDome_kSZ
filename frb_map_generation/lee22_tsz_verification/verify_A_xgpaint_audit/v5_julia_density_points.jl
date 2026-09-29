# Verifier A: production density code (lee2022_frb_dm_profile.jl, included read-only) at points NOT used by the workstream.
using XGPaint, Unitful, UnitfulAstro, Printf
const XG = XGPaint
include("/home/cbllover/HalfDome/frb_map_generation/lee2022_frb_dm_profile.jl")
const O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
clip = 10.0^14.8 / 0.68
lit = Lee2022NoConcentrationDMProfile(normalization=:literal, n0_pivot=:mcut, shape_clip_mass_msun=clip, redshift_scaling=:physical)
ne2d = Lee2022NoConcentrationDMProfile(normalization=:xgpaint_ne2d, n0_pivot=:mcut, shape_clip_mass_msun=clip, redshift_scaling=:physical)
tng = Lee2022NoConcentrationDMProfile(normalization=:literal, n0_pivot=:mcut, shape_clip_mass_msun=clip, redshift_scaling=:tng_calibrated)
b16 = Battaglia16DensityDMProfile(XG.HaloDMProfile(XG.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)))
open(joinpath(O, "v5_julia_density_points.csv"), "w") do io
    println(io, "variant,log10M,z,x,ne_m3,rho_crit_kg_m3")
    for (name, model) in (("lee22_literal", lit), ("lee22_xgpaint_ne2d", ne2d), ("lee22_tngcal", tng), ("b16_xgpaint", b16)),
        lm in (13.0, 14.0, 14.9, 15.25), z in (0.25, 0.75, 2.0, 3.0), x in (0.05, 0.5, 2.0)
        ne = halo_electron_density_m3(model, x, 10.0^lm, z)
        rho = Float64(ustrip(uconvert(u"kg/m^3", XG.ρ_crit(model isa Battaglia16DensityDMProfile ? model.inner : model, z))))
        @printf(io, "%s,%.2f,%.2f,%.4f,%.15e,%.15e\n", name, lm, z, x, ne, rho)
    end
end
