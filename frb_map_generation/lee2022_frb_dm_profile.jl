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
const LEE2022_NO_CONCENTRATION_MODEL_FAMILY = "lee2022_table_a2_no_concentration_m200c"
const LEE2022_NO_CONCENTRATION_CACHE_SIGNATURE =
    "lee2022_table_a2_no_concentration_v1|M200c|R200c|alpha=1|gamma=-0.3|XH=0.76|observer=1/(1+z)"
const LEE2022_FIT_MASS_MIN_HINV_MSUN = 1.0e13
const LEE2022_FIT_MASS_MAX_HINV_MSUN = 10.0^14.8
const LEE2022_FIT_RADIUS_MIN_R200C = 0.04
const LEE2022_FIT_RADIUS_MAX_R200C = 1.34

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
    beta_standard = parameters.gamma - parameters.alpha * parameters.beta_prime
    dimensionless_los = XGPaint._nfw_profile_los_quadrature(
        x,
        parameters.x_c,
        parameters.alpha,
        beta_standard,
        parameters.gamma,
    )

    rho_critical = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(model, z)
    n200 = 200 * rho_critical /
           (model.hydrogen_mass_fraction * XGPaint.constants.ProtonMass) *
           (model.omega_b / model.omega_m)
    electron_column = parameters.n0 * dimensionless_los * n200 * r200c
    return T(ustrip(uconvert(u"pc/cm^3", electron_column)))
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
    println("PASS: Lee2022 parameters, mass-break continuity, angular/physical projection, and z=0 projection.")
    println("  spot DM(theta=1e-4 rad, M200c=1e14 Msun, z=0.5)=$(dm) pc cm^-3")
    println("  emitted column(x=R200c, M200c=1e14 Msun, z=0)=$(z0_column) pc cm^-3")
    return dm
end
