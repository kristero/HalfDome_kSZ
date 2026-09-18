#!/usr/bin/env julia
# Reproduce Lee et al. 2022 (arXiv:2205.01710) Figure 5's own "Battaglia (2016)" and
# "Simulation"/best-fit curves, using XGPaint's actual generalized_nfw/get_params machinery
# for the profile shape on both sides, with two normalization choices identified by digitizing
# the published figure directly (see LEE22_FIG5_SHAPE_CHECK_20260919.md, amendment 2):
#
#   Battaglia (2016) cyan curve: XGPaint's own Battaglia16 gas-density fit and electron
#     conversion (0.9 / m_per_e for ionized H+He), but WITHOUT the f_b = Omega_b/Omega_m
#     factor XGPaint applies for a *physical* gas density. I.e. Lee+2022 plotted Battaglia's
#     rho0-normalized gNFW shape times rho_crit(z) directly as a gas density.
#   Lee22 red/"this work" curves: n_e = n0 f(x) n200 read completely literally against their
#     own eq. (9) n200 = 200 rho_crit(z) Omega_b / (X_H m_p Omega_m) -- XGPaint's own 0.9/m_per_e
#     electron accounting is NOT applied here; this is the `:literal` normalization, which is
#     already this codebase's default for Lee2022NoConcentrationDMProfile.
#
# Also verifies numerically that this file's `lee2022_dimensionless_density` (the literal
# x^gamma (1+x^alpha)^(-beta') form) is bit-identical to calling XGPaint.generalized_nfw with
# the beta = alpha*beta' - gamma substitution used for Battaglia -- i.e. both profiles' radial
# shape function is, and can be evaluated as, the same XGPaint building block.
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))
using Unitful, UnitfulAstro
const PS = ProfileSupport

const BASE_CONFIG = (dm_profile="battaglia16", lee2022_concentration_mode="none", lee2022_normalization="literal",
    lee2022_n0_pivot="legacy_1e14", lee2022_concentration_source="duffy2008",
    lee2022_shape_mass_clip_msun=Inf, lee2022_redshift_scaling="physical",
    halo_boundary="projected", dm_aperture_r200_multiplier=1.0)

function shape_equivalence_check()
    par = PS.lee2022_no_concentration_parameters(1.0e14, 0.5, 0.6774)
    worst = 0.0
    for x in (0.02, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0)
        direct = PS.lee2022_dimensionless_density(x, par.x_c, par.alpha, par.beta_prime, par.gamma)
        via_xgpaint = XGPaint.generalized_nfw(x, par.x_c, par.alpha, par.alpha * par.beta_prime - par.gamma, par.gamma)
        worst = max(worst, abs(direct - via_xgpaint) / direct)
    end
    println("Lee22 shape: max relative difference vs XGPaint.generalized_nfw over x=0.02-3.0: ", worst)
    return worst
end

"""Battaglia16 gas density, XGPaint params and electron conversion, WITHOUT f_b -- the reading
that (per the digitized Fig. 5 check) matches Lee+2022's own plotted 'Battaglia (2016)' curve."""
function battaglia16_ne_m3_no_fb(inner, x::Real, mass_msun::Real, z::Real)
    par = XGPaint.get_params(inner, mass_msun * XGPaint.M_sun, z)
    rho_crit = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(inner, z)
    rho_gas_no_fb = par.P₀ * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ) * rho_crit
    me = XGPaint.constants.ElectronMass
    mH = XGPaint.constants.ProtonMass
    xH = 0.76
    factor = me + (2xH / (xH + 1)) * mH + ((1 - xH) / (2(1 + xH))) * 4mH
    return Float64(ustrip(uconvert(u"m^-3", 0.9 * rho_gas_no_fb / factor)))
end

function fig5_match_main()
    options = parse_options(ARGS)
    output = abspath(option(options, "output_dir", joinpath(@__DIR__, "outputs", "fig5_match_20260919")))
    mkpath(output)

    worst = shape_equivalence_check()
    worst < 1.0e-10 || error("Lee22 shape mismatch vs XGPaint.generalized_nfw exceeds tolerance")

    masses = (3.0e13, 1.17e14)
    z = 0.0
    impacts = 10 .^ range(-2.0, log10(5.0); length=90)
    clip = 10.0^14.8 / H_VALUE

    b16 = PS.dm_profile_runtime_configuration(BASE_CONFIG).model  # XGPaint.HaloDMProfile
    lee_literal = PS.dm_profile_runtime_configuration(merge(BASE_CONFIG, (dm_profile="lee2022",
        lee2022_normalization="literal", lee2022_n0_pivot="mcut",
        lee2022_shape_mass_clip_msun=clip))).model

    path = joinpath(output, "fig5_match.csv")
    open(path, "w") do io
        println(io, "curve,mass_msun,redshift,impact_r200c,ne_over_n200_x3")
        for mass in masses, x in impacts
            n200 = Float64(ustrip(uconvert(u"cm^-3",
                200 * getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(b16, z) * (0.049 / 0.31) /
                (0.76 * XGPaint.constants.ProtonMass))))
            ne_b = battaglia16_ne_m3_no_fb(b16, x, mass, z) * 1.0e-6
            ne_l = PS.halo_electron_density_m3(lee_literal, x, mass, z) * 1.0e-6
            println(io, join(("battaglia16_no_fb", mass, z, x, ne_b / n200 * x^3), ','))
            println(io, join(("lee22_literal", mass, z, x, ne_l / n200 * x^3), ','))
        end
    end
    println("Saved ", path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    fig5_match_main()
end
