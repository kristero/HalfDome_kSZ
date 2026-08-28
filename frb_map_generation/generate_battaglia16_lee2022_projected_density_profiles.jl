#!/usr/bin/env julia

# Generate the exact radial data used by the separate two-dimensional
# Battaglia16-versus-Lee22 projected electron-density consistency plot.
#
# The quantity is physical projected electron column density N_e, not
# observer-frame DM. Battaglia16 is evaluated through XGPaint.ne2d. The Lee22
# implementation returns DM_obs, so N_e is recovered with the (1+z_halo)
# factor and converted from pc cm^-3 to cm^-2. Both use physical M200c/R200c.
# A 3R200c setting is a projected aperture; both LOS quadratures are untruncated.

using Unitful
using UnitfulAstro
using XGPaint

include(joinpath(@__DIR__, "lee2022_frb_dm_profile.jl"))

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB

function string_option(key::AbstractString, default::AbstractString)
    prefix_dash = "--" * String(key) * "="
    prefix_plain = String(key) * "="
    for argument in ARGS
        startswith(argument, prefix_dash) &&
            return split(argument, "="; limit=2)[2]
        startswith(argument, prefix_plain) &&
            return split(argument, "="; limit=2)[2]
    end
    return String(default)
end

float_option(key, default) =
    parse(Float64, string_option(key, string(default)))
int_option(key, default) =
    parse(Int, string_option(key, string(default)))

function radius_200c(model, mass_msun::Real, redshift::Real)
    radius_function = getfield(XGPaint, Symbol("R_", Char(0x0394)))
    return radius_function(
        model,
        Float64(mass_msun) * XGPaint.M_sun,
        Float64(redshift),
        200,
    )
end

function theta_200c(model, mass_msun::Real, redshift::Real)
    radius = radius_200c(model, mass_msun, redshift)
    return Float64(XGPaint.angular_size(model, radius, Float64(redshift)))
end

function battaglia_projected_electron_column_cm2(
    model,
    x_r200c::Real,
    mass_msun::Real,
    redshift::Real,
)
    theta = max(
        Float64(x_r200c) * theta_200c(model, mass_msun, redshift),
        eps(Float64),
    )
    column = XGPaint.ne2d(
        model,
        theta,
        Float64(mass_msun) * XGPaint.M_sun,
        Float64(redshift),
    )
    return Float64(ustrip(u"cm^-2", column))
end

function lee2022_projected_electron_column_cm2(
    model,
    x_r200c::Real,
    mass_msun::Real,
    redshift::Real,
)
    z = Float64(redshift)
    theta = max(
        Float64(x_r200c) * theta_200c(model, mass_msun, z),
        eps(Float64),
    )
    observer_dm = Float64(model(theta, Float64(mass_msun), z))
    physical_column = observer_dm * (1.0 + z) * u"pc/cm^3"
    return Float64(ustrip(u"cm^-2", physical_column))
end

function validate_inputs(
    mass_msun,
    redshift,
    minimum_radius_r200c,
    extent_r200c,
    radial_points,
)
    isfinite(mass_msun) && mass_msun > 0.0 ||
        error("mass_msun must be finite and positive.")
    isfinite(redshift) && redshift > 0.0 ||
        error("redshift must be finite and positive.")
    isfinite(minimum_radius_r200c) && minimum_radius_r200c > 0.0 ||
        error("minimum_radius_r200c must be finite and positive.")
    isfinite(extent_r200c) && extent_r200c > minimum_radius_r200c ||
        error("extent_r200c must exceed minimum_radius_r200c.")
    radial_points >= 64 || error("radial_points must be at least 64.")
end

function profile_models()
    battaglia = XGPaint.BattagliaTauProfile(
        Omega_c=OMEGAC,
        Omega_b=OMEGAB,
        h=H_VALUE,
    )
    lee2022 = Lee2022NoConcentrationDMProfile(
        Omega_c=OMEGAC,
        Omega_b=OMEGAB,
        h=H_VALUE,
    )
    return battaglia, lee2022
end

function profile_consistency_self_test()
    mass_msun = 1.0e14
    redshift = 0.5
    battaglia, lee2022 = profile_models()
    battaglia_r200c = radius_200c(battaglia, mass_msun, redshift)
    lee_r200c = radius_200c(lee2022, mass_msun, redshift)
    radius_ratio = Float64(ustrip(lee_r200c / battaglia_r200c))
    @assert isapprox(radius_ratio, 1.0; rtol=1.0e-12)

    battaglia_column = battaglia_projected_electron_column_cm2(
        battaglia, 1.0, mass_msun, redshift,
    )
    lee_column = lee2022_projected_electron_column_cm2(
        lee2022, 1.0, mass_msun, redshift,
    )
    @assert isfinite(battaglia_column) && battaglia_column > 0.0
    @assert isfinite(lee_column) && lee_column > 0.0
    println("PASS: common M200c/R200c geometry and finite projected electron columns.")
    println("  Battaglia16 N_e(R200c)=$(battaglia_column) cm^-2")
    println("  Lee22 no-concentration N_e(R200c)=$(lee_column) cm^-2")
end

function write_radial_table(path, radii, battaglia_column, lee_column)
    open(path, "w") do io
        println(
            io,
            "r_perp_over_r200c,battaglia16_ne_column_cm2," *
            "lee2022_no_concentration_ne_column_cm2,lee_over_battaglia," *
            "percent_difference_lee_relative_to_battaglia",
        )
        for index in eachindex(radii)
            ratio = lee_column[index] / battaglia_column[index]
            percent = 100.0 * (ratio - 1.0)
            println(
                io,
                join(
                    (
                        radii[index],
                        battaglia_column[index],
                        lee_column[index],
                        ratio,
                        percent,
                    ),
                    ',',
                ),
            )
        end
    end
end

function main()
    output_dir = string_option(
        "output-dir",
        joinpath(
            @__DIR__,
            "outputs",
            "battaglia16_lee2022_2d_profile_consistency",
        ),
    )
    mass_msun = float_option("mass-msun", 1.0e14)
    redshift = float_option("redshift", 0.5)
    minimum_radius_r200c = float_option("minimum-radius-r200c", 1.0e-3)
    extent_r200c = float_option("extent-r200c", 3.0)
    radial_points = int_option("radial-points", 400)
    validate_inputs(
        mass_msun,
        redshift,
        minimum_radius_r200c,
        extent_r200c,
        radial_points,
    )
    mkpath(output_dir)

    battaglia, lee2022 = profile_models()
    battaglia_r200c = radius_200c(battaglia, mass_msun, redshift)
    lee_r200c = radius_200c(lee2022, mass_msun, redshift)
    battaglia_r200c_kpc = Float64(ustrip(u"kpc", battaglia_r200c))
    lee_r200c_kpc = Float64(ustrip(u"kpc", lee_r200c))
    isapprox(battaglia_r200c_kpc, lee_r200c_kpc; rtol=1.0e-12) ||
        error(
            "The models produced inconsistent R200c values: " *
            "$(battaglia_r200c_kpc) and $(lee_r200c_kpc) kpc.",
        )

    radii = 10.0 .^ collect(range(
        log10(minimum_radius_r200c),
        log10(extent_r200c);
        length=radial_points,
    ))
    battaglia_column = [
        battaglia_projected_electron_column_cm2(
            battaglia, radius, mass_msun, redshift,
        )
        for radius in radii
    ]
    lee_column = [
        lee2022_projected_electron_column_cm2(
            lee2022, radius, mass_msun, redshift,
        )
        for radius in radii
    ]
    all(isfinite, battaglia_column) && all(>(0.0), battaglia_column) ||
        error("Battaglia16 projected column profile contains invalid values.")
    all(isfinite, lee_column) && all(>(0.0), lee_column) ||
        error("Lee22 projected column profile contains invalid values.")

    csv_path = joinpath(
        output_dir,
        "battaglia16_vs_lee2022_projected_electron_density_radial.csv",
    )
    metadata_path = joinpath(
        output_dir,
        "battaglia16_vs_lee2022_projected_electron_density_provenance.txt",
    )
    write_radial_table(csv_path, radii, battaglia_column, lee_column)
    open(metadata_path, "w") do io
        println(io, "quantity=physical projected electron column density N_e")
        println(io, "unit=cm^-2")
        println(io, "mass_definition=M200c")
        println(io, "radius_definition=R200c")
        println(io, "mass_msun=$(mass_msun)")
        println(io, "halo_redshift=$(redshift)")
        println(io, "r200c_kpc=$(battaglia_r200c_kpc)")
        println(io, "minimum_radius_r200c=$(minimum_radius_r200c)")
        println(io, "projected_aperture_r200c=$(extent_r200c)")
        println(io, "line_of_sight_integration=untruncated XGPaint GNFW quadrature")
        println(io, "battaglia_path=XGPaint.ne2d(BattagliaTauProfile)")
        println(
            io,
            "lee2022_path=Lee2022NoConcentrationDMProfile converted " *
            "from DM_obs to physical N_e",
        )
        println(io, "lee2022_concentration_mode=none")
        println(io, "lee2022_radial_fit_range_r200c=0.04,1.34")
        println(io, "lee2022_3r200c_status=radial extrapolation")
        println(io, "radial_points=$(radial_points)")
    end
    println("Saved exact projected-density profile data:")
    println("  $(csv_path)")
    println("  $(metadata_path)")
end

if any(
    argument -> argument == "--self-test" || argument == "self-test",
    ARGS,
)
    profile_consistency_self_test()
else
    main()
end
