#!/usr/bin/env julia
# Projected (2-D) electron column profiles, full line of sight (to 1e5 R200c), rest frame, for the
# Battaglia16 gas-density fit (XGPaint parameters, ne2d electrons) and the Lee22 no-concentration fit
# in the reading used for the TNG comparison (XGPaint-native normalization, M_cut pivot, fit-range
# shape clip), on a grid of masses, redshifts and impact parameters. Also writes the 3-D density.
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))
using Unitful, UnitfulAstro
const PS = ProfileSupport

function projected_profile_main()
    options = parse_options(ARGS)
    output = abspath(option(options, "output_dir", joinpath(@__DIR__, "outputs", "projected_profiles_20260917")))
    mkpath(output)
    masses = parse.(Float64, split(option(options, "masses_msun", "7.327e12,3e13,1e14,1e15"), ','))
    redshifts = parse.(Float64, split(option(options, "redshifts", "0.1,0.5,1.0,2.0"), ','))
    impacts = 10 .^ range(-2.0, log10(5.0); length=70)
    clip = 10.0^14.8 / H_VALUE
    base = (dm_profile="battaglia16", lee2022_concentration_mode="none", lee2022_normalization="literal",
            lee2022_n0_pivot="legacy_1e14", lee2022_concentration_source="duffy2008",
            lee2022_shape_mass_clip_msun=Inf, lee2022_redshift_scaling="physical",
            halo_boundary="projected", dm_aperture_r200_multiplier=1.0)
    b16 = PS.Battaglia16DensityDMProfile(PS.dm_profile_runtime_configuration(base).model)
    lee = PS.dm_profile_runtime_configuration(merge(base, (dm_profile="lee2022", lee2022_normalization="xgpaint_ne2d",
        lee2022_n0_pivot="mcut", lee2022_shape_mass_clip_msun=clip))).model
    models = (("battaglia16", b16), ("lee22_noconc", lee))
    path = joinpath(output, "projected_profiles.csv")
    open(path, "w") do io
        println(io, "model,mass_msun,redshift,r200c_mpc,theta200c_arcmin,impact_r200c,column_rest_pc_cm3,column_inside_r200c_rest_pc_cm3,ne_3d_cm3,mass_cut_msun,n0")
        for (label, model) in models, z in redshifts, mass in masses
            r200c = XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200)
            r200c_mpc = Float64(ustrip(uconvert(u"Mpc", r200c)))
            theta200c = Float64(XGPaint.angular_size(model, r200c, z)) * 180 / pi * 60
            mcut, n0 = NaN, NaN
            if model isa PS.AbstractLee2022DMProfile
                p = PS.lee2022_parameters(model, mass, z)
                mcut, n0 = p.mass_cut_msun, p.n0
            end
            for x in impacts
                # chord_dm_pc_cm3 returns observer-frame DM (1/(1+z)); multiply back for the rest-frame column
                full = PS.chord_dm_pc_cm3(model, x, PS.LEE2022_LOS_MAX_R200C, mass, z) * (1 + z)
                inside = x < 1 ? PS.chord_dm_pc_cm3(model, x, sqrt(1 - x^2), mass, z) * (1 + z) : 0.0
                ne3d = PS.halo_electron_density_m3(model, x, mass, z) * 1e-6
                println(io, join((label, mass, z, r200c_mpc, theta200c, x, full, inside, ne3d, mcut, n0), ','))
            end
        end
    end
    println("Saved ", path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    projected_profile_main()
end
