#!/usr/bin/env julia

# Generate the exact radial data used by the separate two-dimensional
# Battaglia16-versus-Lee22 projected electron-density consistency plot.
#
# The quantity is physical projected electron column density N_e, not
# observer-frame DM. Battaglia16 uses XGPaint.ne2d with a physical-radius
# profile, while Lee22 is projected directly in R_perp/R200c. Both paths work at
# z=0 and use the same physical M200c/R200c geometry.
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

projected_radius(model, x_r200c, mass_msun, redshift) =
    Float64(x_r200c) * radius_200c(model, mass_msun, redshift)

function battaglia_projected_electron_column_cm2(
    model,
    x_r200c::Real,
    mass_msun::Real,
    redshift::Real,
)
    radius = projected_radius(model, x_r200c, mass_msun, redshift)
    column = XGPaint.ne2d(
        model,
        radius,
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
    emitted_column = lee2022_projected_electron_column_pc_cm3(
        model, Float64(x_r200c), Float64(mass_msun), z,
    )
    physical_column = emitted_column * u"pc/cm^3"
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
    isfinite(redshift) && redshift >= 0.0 ||
        error("redshift must be finite and nonnegative.")
    isfinite(minimum_radius_r200c) && minimum_radius_r200c > 0.0 ||
        error("minimum_radius_r200c must be finite and positive.")
    isfinite(extent_r200c) && extent_r200c > minimum_radius_r200c ||
        error("extent_r200c must exceed minimum_radius_r200c.")
    radial_points >= 64 || error("radial_points must be at least 64.")
end

function profile_models()
    battaglia = XGPaint.BattagliaTauProfilePhysical(
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
        println(io, "battaglia_path=XGPaint.ne2d(BattagliaTauProfilePhysical)")
        println(io, "lee2022_path=direct dimensionless LOS quadrature in R_perp/R200c")
        println(io, "lee2022_concentration_mode=none")
        println(io, "lee2022_radial_fit_range_r200c=0.04,1.34")
        println(io, "lee2022_3r200c_status=radial extrapolation")
        println(io, "radial_points=$(radial_points)")
    end
    println("Saved exact projected-density profile data:")
    println("  $(csv_path)")
    println("  $(metadata_path)")
end

const DEFAULT_GRID_LOGMASSES = [12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]
const DEFAULT_GRID_REDSHIFTS = [0.0, 1.0, 2.0, 3.0, 4.0]
const GRID_RADIAL_NAME = "battaglia16_vs_lee2022_mass_redshift_grid_radial.csv"
const GRID_SUMMARY_NAME = "battaglia16_vs_lee2022_mass_redshift_grid_summary.csv"
const GRID_PROVENANCE_NAME = "battaglia16_vs_lee2022_mass_redshift_grid_provenance.txt"

function float_list_option(key::AbstractString, defaults)
    raw = string_option(key, join(defaults, ','))
    values = Float64[]
    for item in split(raw, ',')
        stripped = strip(item)
        isempty(stripped) || push!(values, parse(Float64, stripped))
    end
    isempty(values) && error("$(key) must contain at least one value.")
    all(isfinite, values) || error("$(key) contains a non-finite value.")
    return values
end

function validate_grid_inputs(log_masses, redshifts, minimum_radius, extent, radial_points)
    all(value -> value > 0.0, log_masses) || error("log-masses must be positive.")
    all(value -> value >= 0.0, redshifts) || error("redshifts must be nonnegative.")
    length(unique(log_masses)) == length(log_masses) || error("log-masses contains duplicates.")
    length(unique(redshifts)) == length(redshifts) || error("redshifts contains duplicates.")
    validate_inputs(10.0^first(log_masses), first(redshifts), minimum_radius, extent, radial_points)
end

function evaluate_profile_pair(battaglia, lee2022, radii, mass_msun, redshift)
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
        error("Battaglia16 produced a non-finite or nonpositive column at M=$(mass_msun), z=$(redshift).")
    all(isfinite, lee_column) && all(>(0.0), lee_column) ||
        error("Lee22 produced a non-finite or nonpositive column at M=$(mass_msun), z=$(redshift).")
    return battaglia_column, lee_column
end

function grid_main()
    output_dir = string_option(
        "output-dir",
        joinpath(@__DIR__, "outputs", "battaglia16_lee2022_mass_redshift_grid"),
    )
    log_masses = sort(float_list_option("log-masses", DEFAULT_GRID_LOGMASSES))
    redshifts = sort(float_list_option("redshifts", DEFAULT_GRID_REDSHIFTS))
    minimum_radius = float_option("minimum-radius-r200c", 1.0e-4)
    extent = float_option("extent-r200c", 3.0)
    radial_points = int_option("radial-points", 180)
    validate_grid_inputs(log_masses, redshifts, minimum_radius, extent, radial_points)
    mkpath(output_dir)

    radii = 10.0 .^ collect(range(
        log10(minimum_radius), log10(extent); length=radial_points,
    ))
    battaglia, lee2022 = profile_models()
    radial_path = joinpath(output_dir, GRID_RADIAL_NAME)
    summary_path = joinpath(output_dir, GRID_SUMMARY_NAME)
    provenance_path = joinpath(output_dir, GRID_PROVENANCE_NAME)
    parsec_cm = Float64(ustrip(u"cm", 1u"pc"))
    fit_mass_min_msun = LEE2022_FIT_MASS_MIN_HINV_MSUN / H_VALUE
    fit_mass_max_msun = LEE2022_FIT_MASS_MAX_HINV_MSUN / H_VALUE

    open(radial_path, "w") do radial_io
        println(
            radial_io,
            "log10_mass_msun,mass_msun,redshift,r200c_kpc,r_perp_over_r200c," *
            "inside_lee22_mass_fit,inside_lee22_radius_fit," *
            "battaglia16_ne_column_cm2,lee2022_no_concentration_ne_column_cm2," *
            "lee_over_battaglia,percent_difference_lee_relative_to_battaglia",
        )
        open(summary_path, "w") do summary_io
            println(
                summary_io,
                "log10_mass_msun,mass_msun,redshift,r200c_kpc,inside_lee22_mass_fit," *
                "battaglia16_dm_obs_max_pc_cm3,lee2022_dm_obs_max_pc_cm3," *
                "ratio_min_over_radius,ratio_max_over_radius," *
                "ratio_at_0p04r200c,ratio_at_1r200c,ratio_at_1p34r200c,ratio_at_3r200c",
            )
            for redshift in redshifts, log_mass in log_masses
                mass_msun = 10.0^log_mass
                r200c = radius_200c(battaglia, mass_msun, redshift)
                lee_r200c = radius_200c(lee2022, mass_msun, redshift)
                r200c_kpc = Float64(ustrip(u"kpc", r200c))
                radius_ratio = Float64(ustrip(lee_r200c / r200c))
                isapprox(radius_ratio, 1.0; rtol=1.0e-12) ||
                    error("R200c mismatch at logM=$(log_mass), z=$(redshift): ratio=$(radius_ratio).")
                battaglia_column, lee_column = evaluate_profile_pair(
                    battaglia, lee2022, radii, mass_msun, redshift,
                )
                ratio = lee_column ./ battaglia_column
                inside_mass_fit = fit_mass_min_msun <= mass_msun <= fit_mass_max_msun
                for index in eachindex(radii)
                    inside_radius_fit =
                        LEE2022_FIT_RADIUS_MIN_R200C <= radii[index] <=
                        LEE2022_FIT_RADIUS_MAX_R200C
                    println(
                        radial_io,
                        join((
                            log_mass, mass_msun, redshift, r200c_kpc, radii[index],
                            Int(inside_mass_fit), Int(inside_radius_fit),
                            battaglia_column[index], lee_column[index], ratio[index],
                            100.0 * (ratio[index] - 1.0),
                        ), ','),
                    )
                end
                comparison_radii = (0.04, 1.0, 1.34, 3.0)
                direct_ratios = map(comparison_radii) do radius
                    lee2022_projected_electron_column_cm2(
                        lee2022, radius, mass_msun, redshift,
                    ) / battaglia_projected_electron_column_cm2(
                        battaglia, radius, mass_msun, redshift,
                    )
                end
                battaglia_dm_max = maximum(battaglia_column) / parsec_cm / (1.0 + redshift)
                lee_dm_max = maximum(lee_column) / parsec_cm / (1.0 + redshift)
                println(
                    summary_io,
                    join((
                        log_mass, mass_msun, redshift, r200c_kpc, Int(inside_mass_fit),
                        battaglia_dm_max, lee_dm_max, minimum(ratio), maximum(ratio),
                        direct_ratios...,
                    ), ','),
                )
                println(
                    "Completed direct physical projections: log10(M200c/Msun)=$(log_mass), " *
                    "z=$(redshift), max DM_obs Battaglia=$(battaglia_dm_max), Lee=$(lee_dm_max) pc cm^-3",
                )
            end
        end
    end

    open(provenance_path, "w") do io
        println(io, "quantity=physical projected electron column density N_e")
        println(io, "unit=cm^-2")
        println(io, "mass_definition=M200c")
        println(io, "radius_definition=R200c")
        println(io, "log10_mass_msun=$(join(log_masses, ','))")
        println(io, "redshifts=$(join(redshifts, ','))")
        println(io, "minimum_radius_r200c=$(minimum_radius)")
        println(io, "projected_aperture_r200c=$(extent)")
        println(io, "radial_points=$(radial_points)")
        println(io, "cache_or_interpolator_used=false")
        println(io, "battaglia_path=XGPaint.ne2d(BattagliaTauProfilePhysical)")
        println(io, "lee2022_path=direct dimensionless LOS quadrature in R_perp/R200c")
        println(io, "line_of_sight_integration=untruncated XGPaint GNFW quadrature")
        println(io, "lee2022_concentration_mode=none")
        println(io, "lee2022_fit_mass_hinv_msun=1e13,10^14.8")
        println(io, "lee2022_fit_mass_physical_msun=$(fit_mass_min_msun),$(fit_mass_max_msun)")
        println(io, "lee2022_fit_redshift=0,4")
        println(io, "lee2022_fit_radius_r200c=0.04,1.34")
        println(io, "requested_points_outside_fit_are_explicit_extrapolations=true")
    end
    println("Saved mass-redshift projection-grid diagnostics:")
    println("  $(radial_path)")
    println("  $(summary_path)")
    println("  $(provenance_path)")
end

if any(
    argument -> argument == "--self-test" || argument == "self-test",
    ARGS,
)
    profile_consistency_self_test()
elseif any(
    argument -> argument == "--grid-test" || argument == "grid-test",
    ARGS,
)
    grid_main()
else
    main()
end
