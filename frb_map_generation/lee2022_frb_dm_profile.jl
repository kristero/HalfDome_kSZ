# Electron-density FRB profile from Lee et al. (2022), MNRAS 517, 420.
#
# This file is included by generate_halfdome_z1_dm_mass_windows.jl after
# XGPaint and Unitful are available. It deliberately lives outside XGPaint so
# that the installed package and the authoritative Battaglia cache are not
# modified.

using Unitful
using UnitfulAstro

const LEE2022_PROFILE_REFERENCE = "Lee et al. 2022, MNRAS 517, 420, Appendix A, Table A2"
const LEE2022_PROFILE_DOI = "10.1093/mnras/stac2602"
const LEE2022_NO_CONCENTRATION_MODEL_FAMILY =
    "lee2022_table_a2_no_concentration_m200c_profile_owned_los_v2"
const LEE2022_NO_CONCENTRATION_CACHE_SIGNATURE =
    "lee2022_table_a2_no_concentration_v2|M200c|R200c|alpha=1|gamma=-0.3|XH=0.76|profile_owned_los|observer=1/(1+z)"
const LEE2022_FIT_MASS_MIN_HINV_MSUN = 1.0e13
const LEE2022_FIT_MASS_MAX_HINV_MSUN = 10.0^14.8
const LEE2022_FIT_RADIUS_MIN_R200C = 0.04
const LEE2022_FIT_RADIUS_MAX_R200C = 1.34
const LEE2022_DIRECT_PROFILE_SANITY_MAX_PC_CM3 = 1.0e6
const LEE2022_VALIDATION_LOG10_MASSES_MSUN = (12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5)
const LEE2022_VALIDATION_REDSHIFTS = (0.0, 1.0, 2.0, 3.0, 4.0)
const LEE2022_VALIDATION_RADII_R200C = (0.001, 0.04, 1.0, 1.34, 3.0)
const LEE2022_LOS_MAX_R200C = 1.0e5
const LEE2022_LOS_RELATIVE_TOLERANCE = 1.0e-8
const LEE2022_REFERENCE_DM_LOGM12P5_Z0_X0P001 = 156.46630756838437
const LEE2022_PARSEC_IN_CM = Float64(ustrip(uconvert(u"cm", 1.0u"pc")))

"""No-concentration electron-density fit in Appendix A, Table A2."""
struct Lee2022NoConcentrationDMProfile{T,C} <: XGPaint.AbstractGNFW{T}
    omega_b::T
    omega_m::T
    hydrogen_mass_fraction::T
    cosmo::C
end

function Lee2022NoConcentrationDMProfile(
    ; Omega_c::T=0.261, Omega_b::T=0.049, h::T=0.68,
      hydrogen_mass_fraction::T=0.76,
) where {T<:Real}
    omega_m = Omega_b + Omega_c
    omega_m > zero(T) || error("Omega_b + Omega_c must be positive.")
    zero(T) < Omega_b < omega_m || error("Omega_b must be between zero and Omega_m.")
    zero(T) < hydrogen_mass_fraction <= one(T) || error(
        "hydrogen_mass_fraction must lie in (0, 1].",
    )
    cosmo = XGPaint.get_cosmology(T; h=h, Neff=T(3.046), OmegaM=omega_m)
    return Lee2022NoConcentrationDMProfile{T,typeof(cosmo)}(
        Omega_b, omega_m, hydrogen_mass_fraction, cosmo,
    )
end

@inline function lee2022_broken_mass_factor(
    mass_msun::Real,
    mass_cut_msun::Real,
    slope_below::Real,
    slope_above::Real,
)
    ratio = mass_msun / mass_cut_msun
    return ratio < 1 ? ratio^slope_below : ratio^slope_above
end

"""
Return the Appendix-A2 density parameters for physical M200c in Msun.

The unbroken n0 scaling uses the 1e14 Msun pivot in equation (11). The x_c'
and beta' broken power laws use the fitted M_cut=10^13.61 h^-1 Msun in equation
(12), converted to physical Msun before comparison with the HalfDome mass.
"""
function lee2022_no_concentration_parameters(
    mass_m200c_msun::Real,
    redshift::Real,
    little_h::Real,
)
    mass = Float64(mass_m200c_msun)
    z = Float64(redshift)
    h = Float64(little_h)
    isfinite(mass) && mass > 0 || error("M200c must be finite and positive.")
    isfinite(z) && z >= 0 || error("Redshift must be finite and nonnegative.")
    isfinite(h) && h > 0 || error("little h must be finite and positive.")

    one_plus_z = 1.0 + z
    mass_cut_msun = 10.0^13.61 / h
    n0 = 6.8 * (mass / 1.0e14)^0.68 * one_plus_z^(-2.11)
    x_c = 7.9 * one_plus_z^(-0.67) * lee2022_broken_mass_factor(
        mass, mass_cut_msun, 0.47, -0.45,
    )
    beta_prime = 19.5 * one_plus_z^(-0.31) * lee2022_broken_mass_factor(
        mass, mass_cut_msun, 0.70, -0.18,
    )
    return (
        n0=n0,
        x_c=x_c,
        alpha=1.0,
        beta_prime=beta_prime,
        gamma=-0.3,
        mass_cut_msun=mass_cut_msun,
    )
end

@inline function lee2022_dimensionless_density(
    radius_r200c::Real,
    x_c::Real,
    alpha::Real,
    beta_prime::Real,
    gamma::Real,
)
    scaled_radius = Float64(radius_r200c) / Float64(x_c)
    scaled_radius > 0.0 || error("Lee2022 scaled radius must be positive.")
    return scaled_radius^Float64(gamma) *
           (1.0 + scaled_radius^Float64(alpha))^(-Float64(beta_prime))
end

function lee2022_quadgk_function()
    if isdefined(XGPaint, :quadgk)
        return getfield(XGPaint, :quadgk)
    end
    if isdefined(XGPaint, :QuadGK)
        quadgk_module = getfield(XGPaint, :QuadGK)
        isdefined(quadgk_module, :quadgk) &&
            return getfield(quadgk_module, :quadgk)
    end
    error("The active XGPaint environment does not provide QuadGK.quadgk.")
end

function lee2022_dimensionless_los(
    x_perpendicular::Real,
    x_c::Real,
    alpha::Real,
    beta_prime::Real,
    gamma::Real;
    los_max_r200c::Real=LEE2022_LOS_MAX_R200C,
    relative_tolerance::Real=LEE2022_LOS_RELATIVE_TOLERANCE,
)
    x = Float64(x_perpendicular)
    los_max = Float64(los_max_r200c)
    rtol = Float64(relative_tolerance)
    x > 0.0 || error("Lee2022 projected radius must be positive.")
    isfinite(los_max) && los_max > 0.0 ||
        error("Lee2022 LOS maximum must be finite and positive.")
    isfinite(rtol) && 0.0 < rtol < 1.0 ||
        error("Lee2022 LOS relative tolerance must lie in (0, 1).")
    x_squared = x^2
    integrand(y) = lee2022_dimensionless_density(
        sqrt(y^2 + x_squared), x_c, alpha, beta_prime, gamma,
    )
    integral, estimated_error = lee2022_quadgk_function()(
        integrand, 0.0, los_max; rtol=rtol, order=9,
    )
    projected = 2.0 * Float64(integral)
    isfinite(projected) && projected > 0.0 || error(
        "Lee2022 LOS quadrature returned $(projected); " *
        "estimated error=$(estimated_error).",
    )
    return projected
end

"""Physical projected electron column at `R_perp/R200c`, in pc cm^-3 units."""
function lee2022_projected_electron_column_pc_cm3(
    model::Lee2022NoConcentrationDMProfile{T},
    x_perpendicular::Real,
    mass_m200c_msun::Real,
    redshift::Real,
) where {T}
    x = T(x_perpendicular)
    mass = T(mass_m200c_msun)
    z = T(redshift)
    x > zero(T) || error("R_perp/R200c must be positive.")
    mass > zero(T) || error("M200c must be positive.")
    z >= zero(T) || error("Redshift must be nonnegative.")

    parameters = lee2022_no_concentration_parameters(mass, z, model.cosmo.h)
    mass_with_units = mass * XGPaint.M_sun
    r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(
        model, mass_with_units, z, 200,
    )
    dimensionless_los = lee2022_dimensionless_los(
        x,
        parameters.x_c,
        parameters.alpha,
        parameters.beta_prime,
        parameters.gamma,
    )

    rho_critical = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(model, z)
    n200 = 200 * rho_critical /
           (model.hydrogen_mass_fraction * XGPaint.constants.ProtonMass) *
           (model.omega_b / model.omega_m)
    electron_column = parameters.n0 * dimensionless_los * n200 * r200c
    electron_column_cm2 = Float64(ustrip(uconvert(u"cm^-2", electron_column)))
    return T(electron_column_cm2 / LEE2022_PARSEC_IN_CM)
end

"""Evaluate observer-frame halo DM in pc cm^-3 at angular radius `theta`."""
function (model::Lee2022NoConcentrationDMProfile{T})(
    theta_rad,
    mass_m200c_msun,
    redshift,
) where {T}
    theta = T(theta_rad)
    mass = T(mass_m200c_msun)
    z = T(redshift)
    theta > zero(T) || error("Angular radius must be positive for the log-radius profile grid.")
    mass > zero(T) || error("M200c must be positive.")
    z >= zero(T) || error("Redshift must be nonnegative.")

    mass_with_units = mass * XGPaint.M_sun
    r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, mass_with_units, z, 200)
    theta200c = XGPaint.angular_size(model, r200c, z)
    x_perpendicular = theta / theta200c
    emitted_dm = lee2022_projected_electron_column_pc_cm3(
        model, x_perpendicular, mass, z,
    )
    return T(emitted_dm / (one(T) + z))
end

function lee2022_no_concentration_provenance(model::Lee2022NoConcentrationDMProfile)
    return Dict{String,Any}(
        "lee2022_reference" => LEE2022_PROFILE_REFERENCE,
        "lee2022_doi" => LEE2022_PROFILE_DOI,
        "lee2022_density_table" => "Appendix A, Table A2",
        "lee2022_concentration_mode" => "none",
        "lee2022_profile_alpha" => 1.0,
        "lee2022_profile_gamma" => -0.3,
        "lee2022_hydrogen_mass_fraction" => model.hydrogen_mass_fraction,
        "lee2022_fit_mass_min_hinv_msun" => LEE2022_FIT_MASS_MIN_HINV_MSUN,
        "lee2022_fit_mass_max_hinv_msun" => LEE2022_FIT_MASS_MAX_HINV_MSUN,
        "lee2022_fit_radius_min_r200c" => LEE2022_FIT_RADIUS_MIN_R200C,
        "lee2022_fit_radius_max_r200c" => LEE2022_FIT_RADIUS_MAX_R200C,
        "lee2022_radial_extrapolation" => "profile evaluated beyond fitted 1.34R200c for the requested 3R200c comparison",
        "lee2022_low_mass_extrapolation" => "profile evaluated below fitted 1e13 h^-1 Msun where selected by the HalfDome windows",
        "lee2022_mass_cut_hinv_msun" => 10.0^13.61,
        "lee2022_n0_A0" => 6.8,
        "lee2022_n0_alpha_m" => 0.68,
        "lee2022_n0_alpha_z" => -2.11,
        "lee2022_xc_A0" => 7.9,
        "lee2022_xc_alpha_m_below" => 0.47,
        "lee2022_xc_alpha_m_above" => -0.45,
        "lee2022_xc_alpha_z" => -0.67,
        "lee2022_beta_prime_A0" => 19.5,
        "lee2022_beta_prime_alpha_m_below" => 0.70,
        "lee2022_beta_prime_alpha_m_above" => -0.18,
        "lee2022_beta_prime_alpha_z" => -0.31,
        "lee2022_direct_profile_sanity_max_pc_cm3" =>
            LEE2022_DIRECT_PROFILE_SANITY_MAX_PC_CM3,
        "lee2022_direct_validation_log10_masses_msun" =>
            join(LEE2022_VALIDATION_LOG10_MASSES_MSUN, ","),
        "lee2022_direct_validation_redshifts" =>
            join(LEE2022_VALIDATION_REDSHIFTS, ","),
        "lee2022_direct_validation_radii_r200c" =>
            join(LEE2022_VALIDATION_RADII_R200C, ","),
        "lee2022_los_integrator" =>
            "profile-owned explicit Lee22 GNFW integrand with QuadGK",
        "lee2022_los_max_r200c" => LEE2022_LOS_MAX_R200C,
        "lee2022_los_relative_tolerance" =>
            LEE2022_LOS_RELATIVE_TOLERANCE,
        "lee2022_xgpaint_private_los_helper_used" => false,
        "lee2022_column_to_dm_conversion" =>
            "N_e[cm^-2] divided by parsec[cm]",
    )
end

function validate_lee2022_direct_profile_grid(
    model::Lee2022NoConcentrationDMProfile;
    sanity_max_pc_cm3::Real=LEE2022_DIRECT_PROFILE_SANITY_MAX_PC_CM3,
)
    sanity_max = Float64(sanity_max_pc_cm3)
    isfinite(sanity_max) && sanity_max > 0 || error(
        "Lee2022 direct-profile sanity maximum must be finite and positive.",
    )
    minimum_value = Inf
    maximum_value = -Inf
    point_count = 0
    for logmass in LEE2022_VALIDATION_LOG10_MASSES_MSUN
        mass = 10.0^logmass
        for redshift in LEE2022_VALIDATION_REDSHIFTS
            previous_value = Inf
            for radius in LEE2022_VALIDATION_RADII_R200C
                value = Float64(lee2022_projected_electron_column_pc_cm3(
                    model, radius, mass, redshift,
                ))
                isfinite(value) && value > 0 || error(
                    "Non-positive/non-finite Lee2022 direct column at " *
                    "log10(M200c/Msun)=$(logmass), z=$(redshift), " *
                    "R_perp/R200c=$(radius): $(value) pc cm^-3",
                )
                value <= sanity_max || error(
                    "Unphysical Lee2022 direct column $(value) pc cm^-3 exceeds " *
                    "$(sanity_max) at log10(M200c/Msun)=$(logmass), " *
                    "z=$(redshift), R_perp/R200c=$(radius).",
                )
                value <= previous_value * (1 + 1.0e-10) || error(
                    "Lee2022 projected column is not radially non-increasing at " *
                    "log10(M200c/Msun)=$(logmass), z=$(redshift), " *
                    "R_perp/R200c=$(radius): $(value) > $(previous_value).",
                )
                minimum_value = min(minimum_value, value)
                maximum_value = max(maximum_value, value)
                previous_value = value
                point_count += 1
            end
        end
    end
    return (
        minimum_pc_cm3=minimum_value,
        maximum_pc_cm3=maximum_value,
        point_count=point_count,
    )
end

function run_lee2022_no_concentration_profile_self_test()
    model = Lee2022NoConcentrationDMProfile()
    mass_cut = 10.0^13.61 / model.cosmo.h
    at_cut = lee2022_no_concentration_parameters(mass_cut, 0.0, model.cosmo.h)
    @assert isapprox(at_cut.x_c, 7.9; rtol=2e-14)
    @assert isapprox(at_cut.beta_prime, 19.5; rtol=2e-14)

    below = lee2022_no_concentration_parameters(mass_cut * (1 - 1e-9), 0.5, model.cosmo.h)
    above = lee2022_no_concentration_parameters(mass_cut * (1 + 1e-9), 0.5, model.cosmo.h)
    @assert isapprox(below.x_c, above.x_c; rtol=2e-9)
    @assert isapprox(below.beta_prime, above.beta_prime; rtol=2e-9)

    dm = model(1.0e-4, 1.0e14, 0.5)
    @assert isfinite(dm) && dm > 0
    r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(
        model, 1.0e14 * XGPaint.M_sun, 0.5, 200,
    )
    x = 1.0e-4 / XGPaint.angular_size(model, r200c, 0.5)
    emitted_column = lee2022_projected_electron_column_pc_cm3(model, x, 1.0e14, 0.5)
    @assert isapprox(emitted_column, dm * 1.5; rtol=2.0e-12)
    z0_column = lee2022_projected_electron_column_pc_cm3(model, 1.0, 1.0e14, 0.0)
    @assert isfinite(z0_column) && z0_column > 0
    reference_dm = lee2022_projected_electron_column_pc_cm3(
        model, 0.001, 10.0^12.5, 0.0,
    )
    @assert isapprox(
        reference_dm, LEE2022_REFERENCE_DM_LOGM12P5_Z0_X0P001; rtol=2.0e-8,
    )
    validation = validate_lee2022_direct_profile_grid(model)
    @assert validation.point_count ==
        length(LEE2022_VALIDATION_LOG10_MASSES_MSUN) *
        length(LEE2022_VALIDATION_REDSHIFTS) *
        length(LEE2022_VALIDATION_RADII_R200C)
    println("PASS: Lee2022 parameters, mass-break continuity, angular/physical projection, and z=0 projection.")
    println("  spot DM(theta=1e-4 rad, M200c=1e14 Msun, z=0.5)=$(dm) pc cm^-3")
    println("  emitted column(x=R200c, M200c=1e14 Msun, z=0)=$(z0_column) pc cm^-3")
    println(
        "  regression DM(x=0.001, log10(M200c/Msun)=12.5, z=0)=" *
        "$(reference_dm) pc cm^-3",
    )
    println(
        "  direct validation grid: $(validation.point_count) points, range=" *
        "[$(validation.minimum_pc_cm3), $(validation.maximum_pc_cm3)] pc cm^-3",
    )
    return dm
end
