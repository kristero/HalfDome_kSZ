#!/usr/bin/env julia
# 3-D and projected (line-of-sight, Compton-y) electron-pressure profiles for the Battaglia12 fiducial
# thermal-pressure fit (the coefficients used for the HalfDome Compton-y map) and the Lee22 no-concentration
# pressure fit (arXiv v1 Table 7 = MNRAS Table A1, XGPaint mapping), on a grid of masses, redshifts and
# radii/impact parameters. Companion to compute_projected_profile_grid.jl (electron density).
include(joinpath(@__DIR__, "paint_halfdome_battaglia12_tsz_map.jl"))
using Unitful, UnitfulAstro

const KEV_PER_CM3_PER_PA = 6.241509074e9  # 1 Pa = 1 J/m^3 -> keV/cm^3

function pressure_profile_main()
    options = Support.parse_options(ARGS)
    output = abspath(Support.option(options, "output_dir", joinpath(@__DIR__, "outputs", "pressure_profiles_20260919")))
    mkpath(output)
    masses = parse.(Float64, split(Support.option(options, "masses_msun", "7.327e12,3e13,1e14,1e15"), ','))
    redshifts = parse.(Float64, split(Support.option(options, "redshifts", "0.1,0.5,1.0,2.0"), ','))
    radii = 10 .^ range(-2.0, log10(5.0); length=70)

    b12 = build_battaglia12_fiducial_model()
    lee = Lee2022ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H_VALUE)
    models = (("battaglia12", b12), ("lee22_noconc", lee))

    path = joinpath(output, "pressure_profiles.csv")
    open(path, "w") do io
        println(io, "model,mass_msun,redshift,r200c_mpc,theta200c_arcmin,x_r200c,y_projected,pe_3d_kev_cm3,pe_over_p200_x3,p200_kev_cm3,p0_thermal,xc,beta")
        for (label, model) in models, z in redshifts, mass in masses
            M = mass * XGPaint.M_sun
            r200c = XGPaint.R_Δ(model, M, z, 200)
            r200c_mpc = Float64(ustrip(uconvert(u"Mpc", r200c)))
            theta200c_arcmin = Float64(XGPaint.angular_size(model, r200c, z)) * 180 / pi * 60
            par = XGPaint.get_params(model, M, z)
            p200_pa = Float64(ustrip(uconvert(u"Pa",
                200 * XGPaint.constants.G * M * XGPaint.ρ_crit(model, z) * model.f_b / 2 / r200c)))
            p200_kev_cm3 = p200_pa * KEV_PER_CM3_PER_PA
            for x in radii
                theta = x * Float64(XGPaint.angular_size(model, r200c, z))
                y = Float64(XGPaint.compton_y(model, theta, M, z))
                shape = XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ)
                pe_over_p200 = 0.5176 * Float64(par.P₀) * shape
                pe_3d_kev_cm3 = p200_kev_cm3 * pe_over_p200
                println(io, join((label, mass, z, r200c_mpc, theta200c_arcmin, x, y, pe_3d_kev_cm3,
                    pe_over_p200 * x^3, p200_kev_cm3, Float64(par.P₀), Float64(par.xc), Float64(par.β)), ','))
            end
        end
    end
    println("Saved ", path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    pressure_profile_main()
end
