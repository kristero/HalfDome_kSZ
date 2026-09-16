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

abstract type AbstractLee2022DMProfile{T} <: XGPaint.AbstractGNFW{T} end

# Option conventions shared by both Lee22 models (see LEE22_IMPLEMENTATION_CHECK_20260916.md):
#   normalization      :literal          n_e = n0 f(x) n200 with n200 from eq. (9) as printed
#                      :baryon_fraction  n_e = (Omega_b/Omega_m) n0 f(x) n200; makes the fits
#                                        physically consistent with XGPaint's Battaglia16 (which
#                                        carries the same f_b factor) and with TNG gas fractions
#   n0_pivot           :legacy_1e14      no-concentration n0 uses the eq. (11) 1e14 Msun pivot
#                      :mcut             all parameters share M_cut, the literal eq. (12) form
#   concentration_source :duffy2008      Duffy08 median NFW c200c
#                        :tng_mean       power law through the two TNG mean concentrations quoted
#                                        in Lee22 section 2.3 (5.65 at 10^13.1, 4.53 at 10^14.1 h^-1 Msun)
#   shape_clip_mass_msun  Inf            extrapolate every mass power law
#                         M (Msun)       freeze the shape parameters x_c and beta' at this mass
#   redshift_scaling   :physical         n200 uses rho_crit(z) (eq. 9 read literally)
#                      :comoving_hypothesis  multiply by (1+z)^3 / E^2(z): the conversion needed if
#                                        the fitted n_e were comoving densities normalized by the z=0
#                                        critical density; this reproduces the fitted alpha_z of n0 and
#                                        is a hypothesis about the paper's bookkeeping, not a documented fact
const LEE2022_NORMALIZATIONS = (:literal, :baryon_fraction)
const LEE2022_REDSHIFT_SCALINGS = (:physical, :comoving_hypothesis)
const LEE2022_N0_PIVOTS = (:legacy_1e14, :mcut)
const LEE2022_CONCENTRATION_SOURCES = (:duffy2008, :tng_mean)
const LEE2022_FIT_MASS_MAX_MSUN_AT_H068 = LEE2022_FIT_MASS_MAX_HINV_MSUN / 0.68
const LEE2022_TNG_MEAN_C_PIVOT_HINV_MSUN = 10.0^13.1
const LEE2022_TNG_MEAN_C_AT_PIVOT = 5.65
const LEE2022_TNG_MEAN_C_SLOPE = log10(4.53 / 5.65)  # per dex between the two quoted bins
const LEE2022_TNG_MEAN_C_REDSHIFT_SLOPE = -0.47      # borrowed from Duffy08; TNG evolution not quoted

"""No-concentration fit (arXiv v1 Table 8 = MNRAS Table A2); options select the conventions."""
struct Lee2022NoConcentrationDMProfile{T,C} <: AbstractLee2022DMProfile{T}
    omega_b::T
    omega_m::T
    hydrogen_mass_fraction::T
    cosmo::C
    normalization::Symbol
    n0_pivot::Symbol
    shape_clip_mass_msun::Float64
    redshift_scaling::Symbol
end

"""Preferred density fit: Lee22 arXiv Table 3, with a mean-concentration proxy.

This is not a replacement for the measured per-halo Klypin concentrations in
Lee22; no concentration scatter is applied. The mass pivot follows the common
M_cut printed in equation (12). Normalization, concentration source and the
optional shape-parameter mass clip are selected by the option fields.
"""
struct Lee2022ConcentrationDMProfile{T,C} <: AbstractLee2022DMProfile{T}
    omega_b::T
    omega_m::T
    hydrogen_mass_fraction::T
    cosmo::C
    normalization::Symbol
    concentration_source::Symbol
    shape_clip_mass_msun::Float64
    redshift_scaling::Symbol
end

const LEE2022_CONCENTRATION_MODEL_FAMILY = "lee2022_preferred_duffy2008_eq12_v1"
const LEE2022_CONCENTRATION_CACHE_SIGNATURE =
    "lee2022_arxiv_table3|eq12_common_mcut|duffy2008_full_nfw_200c|no_scatter|M200c|R200c|alpha=1|gamma=-0.3|XH=0.76|profile_owned_los|observer=1/(1+z)"

function Lee2022ConcentrationDMProfile(;
    normalization::Symbol=:literal,
    concentration_source::Symbol=:duffy2008,
    shape_clip_mass_msun::Real=Inf,
    redshift_scaling::Symbol=:physical,
    kwargs...,
)
    normalization in LEE2022_NORMALIZATIONS || error("Unknown Lee22 normalization $(normalization)")
    redshift_scaling in LEE2022_REDSHIFT_SCALINGS || error("Unknown Lee22 redshift scaling $(redshift_scaling)")
    concentration_source in LEE2022_CONCENTRATION_SOURCES ||
        error("Unknown Lee22 concentration source $(concentration_source)")
    shape_clip_mass_msun > 0 || error("Lee22 shape clip mass must be positive (or Inf)")
    base = Lee2022NoConcentrationDMProfile(; kwargs...)
    return Lee2022ConcentrationDMProfile(
        base.omega_b, base.omega_m, base.hydrogen_mass_fraction, base.cosmo,
        normalization, concentration_source, Float64(shape_clip_mass_msun), redshift_scaling,
    )
end

"""Mean TNG concentration proxy from the two sample means quoted in Lee22 section 2.3.

c = 5.65 (M h / 10^13.1)^(-0.096) (1+z)^(-0.47). The z=0 anchors are the paper's own
Rockstar/Klypin means, so this matches the paper's concentration definition better
than an NFW-fit relation; the redshift slope is borrowed from Duffy08.
"""
function lee2022_tng_mean_concentration(mass_msun::Real, redshift::Real, little_h::Real)
    isfinite(mass_msun) && mass_msun > 0 || error("Invalid TNG-mean concentration mass")
    isfinite(redshift) && redshift >= 0 || error("Invalid TNG-mean concentration redshift")
    return LEE2022_TNG_MEAN_C_AT_PIVOT *
           (mass_msun * little_h / LEE2022_TNG_MEAN_C_PIVOT_HINV_MSUN)^LEE2022_TNG_MEAN_C_SLOPE *
           (1 + redshift)^LEE2022_TNG_MEAN_C_REDSHIFT_SLOPE
end

function lee2022_concentration(source::Symbol, mass_msun::Real, redshift::Real, little_h::Real)
    source == :duffy2008 && return duffy2008_c200c(mass_msun, redshift, little_h)
    source == :tng_mean && return lee2022_tng_mean_concentration(mass_msun, redshift, little_h)
    error("Unknown Lee22 concentration source $(source)")
end

"""Duffy08 Table 1: full-sample NFW 200-critical fit over z=0..2.

Input mass is physical Msun; the published pivot is 2e12 Msun/h.
Beyond its calibrated mass/redshift range this is an explicit extrapolation.
"""
function duffy2008_c200c(mass_msun::Real, redshift::Real, little_h::Real)
    isfinite(mass_msun) && mass_msun > 0 || error("Invalid Duffy2008 mass")
    isfinite(redshift) && redshift >= 0 || error("Invalid Duffy2008 redshift")
    isfinite(little_h) && little_h > 0 || error("Invalid Duffy2008 h")
    return 5.71 * (mass_msun * little_h / 2.0e12)^(-0.084) *
           (1 + redshift)^(-0.47)
end

function lee2022_concentration_parameters(mass_msun::Real, z::Real, h::Real;
                                         concentration=duffy2008_c200c(mass_msun, z, h),
                                         shape_clip_mass_msun::Real=Inf)
    isfinite(mass_msun) && mass_msun > 0 || error("Invalid Lee22 mass")
    isfinite(z) && z >= 0 || error("Invalid Lee22 redshift")
    isfinite(h) && h > 0 || error("Invalid Lee22 h")
    isfinite(concentration) && concentration > 0 || error("Invalid concentration")
    mass_cut = 10.0^13.75 / h
    c10 = concentration / 10
    # Shape parameters may be frozen at the fit's upper mass limit; the amplitude n0
    # and the concentration always use the true halo mass.
    mass_shape = min(Float64(mass_msun), Float64(shape_clip_mass_msun))
    # Eq. (12) explicitly shares M_cut among n0, xc and beta. For n0,
    # the absent second slope means the same exponent on both sides.
    n0 = 15.7 * (mass_msun / mass_cut)^0.87 * (1 + z)^(-2.09) * c10^0.63
    xc = 2.2 * (1 + z)^(-0.74) * c10^(-1.37) *
         lee2022_broken_mass_factor(mass_shape, mass_cut, -0.06, -1.45)
    beta = 7.5 * (1 + z)^(-0.39) * c10^(-1.11) *
           lee2022_broken_mass_factor(mass_shape, mass_cut, 0.24, -1.10)
    return (n0=n0, x_c=xc, alpha=1.0, beta_prime=beta, gamma=-0.3,
            mass_cut_msun=mass_cut, concentration=concentration)
end

lee2022_parameters(model::Lee2022NoConcentrationDMProfile, mass, z) =
    lee2022_no_concentration_parameters(mass, z, model.cosmo.h;
        n0_pivot=model.n0_pivot, shape_clip_mass_msun=model.shape_clip_mass_msun)
lee2022_parameters(model::Lee2022ConcentrationDMProfile, mass, z) =
    lee2022_concentration_parameters(mass, z, model.cosmo.h;
        concentration=lee2022_concentration(model.concentration_source, mass, z, model.cosmo.h),
        shape_clip_mass_msun=model.shape_clip_mass_msun)

"""Overall factor multiplying the literal eq. (9) normalization."""
lee2022_normalization_factor(model::AbstractLee2022DMProfile) =
    model.normalization == :baryon_fraction ? model.omega_b / model.omega_m : one(model.omega_b)

"""Redshift-dependent factor: 1 for the literal physical reading, (1+z)^3/E^2(z) for the
comoving-bookkeeping hypothesis (E^2 = rho_crit(z)/rho_crit(0))."""
function lee2022_redshift_scaling_factor(model::AbstractLee2022DMProfile, z::Real)
    model.redshift_scaling == :physical && return 1.0
    rho = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))
    e2 = Float64(ustrip(rho(model, z) / rho(model, zero(z))))
    return (1 + Float64(z))^3 / e2
end

function lee2022_variant_tokens(model::AbstractLee2022DMProfile)
    tokens = String[]
    model.normalization == :baryon_fraction && push!(tokens, "norm=fb")
    if model isa Lee2022NoConcentrationDMProfile && model.n0_pivot == :mcut
        push!(tokens, "n0pivot=mcut")
    end
    if model isa Lee2022ConcentrationDMProfile && model.concentration_source != :duffy2008
        push!(tokens, "c=" * String(model.concentration_source))
    end
    if isfinite(model.shape_clip_mass_msun)
        push!(tokens, "shapeclip=1e$(round(log10(model.shape_clip_mass_msun), digits=4))Msun")
    end
    model.redshift_scaling == :comoving_hypothesis && push!(tokens, "zscale=comoving_hyp")
    return tokens
end

"""Cache signature; identical to the historical strings when every option is legacy."""
function lee2022_cache_signature(model::AbstractLee2022DMProfile)
    base = model isa Lee2022ConcentrationDMProfile ?
        LEE2022_CONCENTRATION_CACHE_SIGNATURE : LEE2022_NO_CONCENTRATION_CACHE_SIGNATURE
    tokens = lee2022_variant_tokens(model)
    return isempty(tokens) ? base : base * "|" * join(tokens, "|")
end

function lee2022_model_family(model::AbstractLee2022DMProfile)
    base = model isa Lee2022ConcentrationDMProfile ?
        LEE2022_CONCENTRATION_MODEL_FAMILY : LEE2022_NO_CONCENTRATION_MODEL_FAMILY
    tokens = lee2022_variant_tokens(model)
    isempty(tokens) && return base
    return base * "_" * join([replace(replace(t, "=" => "-"), "." => "p") for t in tokens], "_")
end

function lee2022_option_provenance(model::AbstractLee2022DMProfile)
    entries = Dict{String,Any}(
        "lee2022_normalization" => String(model.normalization),
        "lee2022_normalization_factor" => Float64(lee2022_normalization_factor(model)),
        "lee2022_shape_clip_mass_msun" => model.shape_clip_mass_msun,
        "lee2022_redshift_scaling" => String(model.redshift_scaling),
        "lee2022_variant_tokens" => join(lee2022_variant_tokens(model), "|"),
        "lee2022_cache_signature" => lee2022_cache_signature(model),
        "lee2022_model_family" => lee2022_model_family(model),
    )
    if model isa Lee2022NoConcentrationDMProfile
        entries["lee2022_n0_pivot"] = String(model.n0_pivot)
    else
        entries["lee2022_concentration_source"] = String(model.concentration_source)
    end
    return entries
end

function Lee2022NoConcentrationDMProfile(
    ; Omega_c::T=0.261, Omega_b::T=0.049, h::T=0.68,
      hydrogen_mass_fraction::T=0.76,
      normalization::Symbol=:literal,
      n0_pivot::Symbol=:legacy_1e14,
      shape_clip_mass_msun::Real=Inf,
      redshift_scaling::Symbol=:physical,
) where {T<:Real}
    normalization in LEE2022_NORMALIZATIONS || error("Unknown Lee22 normalization $(normalization)")
    redshift_scaling in LEE2022_REDSHIFT_SCALINGS || error("Unknown Lee22 redshift scaling $(redshift_scaling)")
    n0_pivot in LEE2022_N0_PIVOTS || error("Unknown Lee22 n0 pivot $(n0_pivot)")
    shape_clip_mass_msun > 0 || error("Lee22 shape clip mass must be positive (or Inf)")
    omega_m = Omega_b + Omega_c
    omega_m > zero(T) || error("Omega_b + Omega_c must be positive.")
    zero(T) < Omega_b < omega_m || error("Omega_b must be between zero and Omega_m.")
    zero(T) < hydrogen_mass_fraction <= one(T) || error(
        "hydrogen_mass_fraction must lie in (0, 1].",
    )
    cosmo = XGPaint.get_cosmology(T; h=h, Neff=T(3.046), OmegaM=omega_m)
    return Lee2022NoConcentrationDMProfile{T,typeof(cosmo)}(
        Omega_b, omega_m, hydrogen_mass_fraction, cosmo,
        normalization, n0_pivot, Float64(shape_clip_mass_msun), redshift_scaling,
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
    little_h::Real;
    n0_pivot::Symbol=:legacy_1e14,
    shape_clip_mass_msun::Real=Inf,
)
    mass = Float64(mass_m200c_msun)
    z = Float64(redshift)
    h = Float64(little_h)
    isfinite(mass) && mass > 0 || error("M200c must be finite and positive.")
    isfinite(z) && z >= 0 || error("Redshift must be finite and nonnegative.")
    isfinite(h) && h > 0 || error("little h must be finite and positive.")

    one_plus_z = 1.0 + z
    mass_cut_msun = 10.0^13.61 / h
    mass_shape = min(mass, Float64(shape_clip_mass_msun))
    n0_pivot_msun = n0_pivot == :mcut ? mass_cut_msun : 1.0e14
    n0 = 6.8 * (mass / n0_pivot_msun)^0.68 * one_plus_z^(-2.11)
    x_c = 7.9 * one_plus_z^(-0.67) * lee2022_broken_mass_factor(
        mass_shape, mass_cut_msun, 0.47, -0.45,
    )
    beta_prime = 19.5 * one_plus_z^(-0.31) * lee2022_broken_mass_factor(
        mass_shape, mass_cut_msun, 0.70, -0.18,
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
    model::AbstractLee2022DMProfile{T},
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

    parameters = lee2022_parameters(model, mass, z)
    if model isa Lee2022ConcentrationDMProfile
        parameters.beta_prime - parameters.gamma > 1 || error(
            "Lee22 preferred density extrapolation has a divergent untruncated LOS at " *
            "M200c=$(mass), z=$(z), outer slope=$(parameters.beta_prime-parameters.gamma). " *
            "A projected aperture does not repair this; specify a physical outer boundary first.",
        )
    end
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
    electron_column = lee2022_normalization_factor(model) *
                      lee2022_redshift_scaling_factor(model, z) *
                      parameters.n0 * dimensionless_los * n200 * r200c
    electron_column_cm2 = Float64(ustrip(uconvert(u"cm^-2", electron_column)))
    return T(electron_column_cm2 / LEE2022_PARSEC_IN_CM)
end

"""Evaluate observer-frame halo DM in pc cm^-3 at angular radius `theta`."""
function (model::AbstractLee2022DMProfile{T})(
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

function lee2022_concentration_provenance(model::Lee2022ConcentrationDMProfile)
    return merge(lee2022_option_provenance(model), Dict{String,Any}(
        "lee2022_reference" => "Lee22 arXiv:2205.01710v1 Table 3, equations 9, 10, 12",
        "lee2022_doi" => LEE2022_PROFILE_DOI,
        "lee2022_concentration_mode" => "duffy2008",
        "lee2022_concentration_prescription" => model.concentration_source == :duffy2008 ?
            "Duffy08 Table 1 full NFW 200c z=0..2: 5.71*(M*h/2e12)^-0.084*(1+z)^-0.47" :
            "Lee22 sec 2.3 TNG means: 5.65*(M*h/10^13.1)^(log10(4.53/5.65))*(1+z)^-0.47",
        "lee2022_concentration_caveat" => "mean proxy; not measured per-halo Klypin concentrations; no scatter",
        "lee2022_mass_pivot" => "common 10^13.75 Msun/h for all parameters, equation 12",
        "lee2022_profile_alpha" => 1.0,
        "lee2022_profile_gamma" => -0.3,
        "lee2022_hydrogen_mass_fraction" => model.hydrogen_mass_fraction,
        "lee2022_fit_mass_min_hinv_msun" => LEE2022_FIT_MASS_MIN_HINV_MSUN,
        "lee2022_fit_mass_max_hinv_msun" => LEE2022_FIT_MASS_MAX_HINV_MSUN,
        "lee2022_fit_radius_min_r200c" => LEE2022_FIT_RADIUS_MIN_R200C,
        "lee2022_fit_radius_max_r200c" => LEE2022_FIT_RADIUS_MAX_R200C,
        "lee2022_fit_limits_are_selection_cuts" => false,
        "lee2022_gas_mass_renormalized" => false,
        "lee2022_los_max_r200c" => LEE2022_LOS_MAX_R200C,
        "lee2022_los_relative_tolerance" => LEE2022_LOS_RELATIVE_TOLERANCE,
    ))
end

function lee2022_no_concentration_provenance(model::Lee2022NoConcentrationDMProfile)
    return merge(lee2022_option_provenance(model), Dict{String,Any}(
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
    ))
end

function validate_lee2022_direct_profile_grid(
    model::AbstractLee2022DMProfile;
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

# ---------------------------------------------------------------------------
# Profile-owned Battaglia16 density and the spherical (chord-limited) boundary
# ---------------------------------------------------------------------------

"""Battaglia16 gas density evaluated from XGPaint's own parameters and normalization
(rho_gas = P0 gNFW f_b rho_crit; n_e via XGPaint's ne2d composition), but with the
line-of-sight integral owned by this file so it can be truncated at a sphere."""
struct Battaglia16DensityDMProfile{T,C,P} <: XGPaint.AbstractGNFW{T}
    cosmo::C
    inner::P
end

function Battaglia16DensityDMProfile(inner::XGPaint.HaloDMProfile{T,C}) where {T,C}
    return Battaglia16DensityDMProfile{T,C,typeof(inner)}(inner.cosmo, inner)
end

const B16_DENSITY_CACHE_SIGNATURE =
    "battaglia16_xgpaint_params|rho_gas=P0*gnfw*f_b*rho_crit|ne2d_composition|M200c|R200c|profile_owned_los|observer=1/(1+z)"
const B16_DENSITY_MODEL_FAMILY = "battaglia16_profile_owned_los_v1"

"""Physical 3-D electron density [m^-3] at x = r/R200c for either density family."""
function halo_electron_density_m3(model::Battaglia16DensityDMProfile, x::Real, mass_msun::Real, z::Real)
    inner = model.inner
    par = XGPaint.get_params(inner, mass_msun * XGPaint.M_sun, z)
    rho_crit = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(inner, z)
    rho_gas = par.P₀ * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ) * inner.f_b * rho_crit
    me = XGPaint.constants.ElectronMass
    mH = XGPaint.constants.ProtonMass
    xH = 0.76
    factor = me + (2xH / (xH + 1)) * mH + ((1 - xH) / (2(1 + xH))) * 4mH
    return Float64(ustrip(uconvert(u"m^-3", 0.9 * rho_gas / factor)))
end

function halo_electron_density_m3(model::AbstractLee2022DMProfile, x::Real, mass_msun::Real, z::Real)
    parameters = lee2022_parameters(model, mass_msun, z)
    rho_critical = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(model, z)
    n200 = 200 * rho_critical / (model.hydrogen_mass_fraction * XGPaint.constants.ProtonMass) *
           (model.omega_b / model.omega_m)
    n200_m3 = Float64(ustrip(uconvert(u"m^-3", n200)))
    return lee2022_normalization_factor(model) * lee2022_redshift_scaling_factor(model, z) *
           parameters.n0 * n200_m3 *
           lee2022_dimensionless_density(x, parameters.x_c, parameters.alpha, parameters.beta_prime, parameters.gamma)
end

const M2_TO_PC_CM3_LOCAL = Float64(ustrip(uconvert(u"pc*cm^-3", 1.0u"m^-2")))

"""Observer-frame DM [pc cm^-3] through a chord of half-length `lmax` (in R200c units) at
projected radius `x` (R200c units), integrating the profile-owned 3-D density."""
function chord_dm_pc_cm3(model, x::Real, lmax::Real, mass_msun::Real, z::Real; rtol=1.0e-8)
    lmax > 0 || return 0.0
    r200c_m = Float64(ustrip(uconvert(u"m",
        getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, mass_msun * XGPaint.M_sun, z, 200))))
    integral, _ = lee2022_quadgk_function()(
        l -> halo_electron_density_m3(model, sqrt(x^2 + l^2), mass_msun, z), 0.0, Float64(lmax);
        rtol=rtol, order=9,
    )
    return 2 * integral * r200c_m * M2_TO_PC_CM3_LOCAL / (1 + z)
end

"""Projected (long LOS) evaluation, same convention as XGPaint's HaloDMProfile."""
function (model::Battaglia16DensityDMProfile{T})(theta_rad, mass_msun, redshift) where {T}
    theta = Float64(theta_rad); mass = Float64(mass_msun); z = Float64(redshift)
    theta > 0 || error("Angular radius must be positive.")
    r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, mass * XGPaint.M_sun, z, 200)
    x = theta / XGPaint.angular_size(model, r200c, z)
    return T(chord_dm_pc_cm3(model, x, LEE2022_LOS_MAX_R200C, mass, z))
end

"""Chord-mean wrapper for a spherical boundary of radius `sphere_r200c` R200c.

The callable returns g(theta, M, z) = DM_sphere / (2 L_max), the mean electron
column per unit chord half-length, which is smooth and positive everywhere (it tends
to the edge density inside R200c/(1+z) units and stays there outside the sphere), so it
can be cached and log-interpolated safely. The generator multiplies the interpolated
g by the exact chord factor 2 L_max = 2 X sqrt(1 - (theta/theta_max)^2) per ray, which
gives DM -> 0 continuously at the boundary instead of the projected hard floor.
"""
struct SphericalChordDMProfile{T,C,P} <: XGPaint.AbstractGNFW{T}
    cosmo::C
    inner::P
    sphere_r200c::Float64
end

function SphericalChordDMProfile(inner::XGPaint.AbstractGNFW{T}, sphere_r200c::Real) where {T}
    isfinite(sphere_r200c) && sphere_r200c > 0 || error("Sphere radius must be positive")
    return SphericalChordDMProfile{T,typeof(inner.cosmo),typeof(inner)}(inner.cosmo, inner, Float64(sphere_r200c))
end

spherical_chord_half_length(x::Real, sphere::Real) = x < sphere ? sqrt(sphere^2 - x^2) : 0.0

"""Exact chord factor used by the generator: 2 L_max at angular radius theta, given the
aperture edge theta_max = angular_size(X R200c)."""
@inline function spherical_chord_factor(theta::Real, theta_max::Real, sphere::Real)
    ratio = Float64(theta) / Float64(theta_max)
    ratio >= 1 && return 0.0
    return 2 * Float64(sphere) * sqrt(1 - ratio^2)
end

function (model::SphericalChordDMProfile{T})(theta_rad, mass_msun, redshift) where {T}
    theta = Float64(theta_rad); mass = Float64(mass_msun); z = Float64(redshift)
    theta > 0 || error("Angular radius must be positive.")
    r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, mass * XGPaint.M_sun, z, 200)
    x = theta / XGPaint.angular_size(model, r200c, z)
    sphere = model.sphere_r200c
    lmax = spherical_chord_half_length(x, sphere)
    if lmax < 1.0e-6
        # At or beyond the edge: the chord-mean tends to the edge density; keep the
        # cached function continuous and positive (the generator never uses g there).
        r200c_m = Float64(ustrip(uconvert(u"m", r200c)))
        return T(halo_electron_density_m3(model.inner, sphere, mass, z) * r200c_m * M2_TO_PC_CM3_LOCAL / (1 + z))
    end
    return T(chord_dm_pc_cm3(model.inner, x, lmax, mass, z) / (2 * lmax))
end

"""Direct spherical DM (no cache) for validation."""
function spherical_dm_pc_cm3(model::SphericalChordDMProfile, theta_rad, mass_msun, redshift)
    r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, Float64(mass_msun) * XGPaint.M_sun, Float64(redshift), 200)
    x = Float64(theta_rad) / XGPaint.angular_size(model, r200c, Float64(redshift))
    return chord_dm_pc_cm3(model.inner, x, spherical_chord_half_length(x, model.sphere_r200c), Float64(mass_msun), Float64(redshift))
end

profile_cache_signature(model::AbstractLee2022DMProfile) = lee2022_cache_signature(model)
profile_cache_signature(::Battaglia16DensityDMProfile) = B16_DENSITY_CACHE_SIGNATURE
profile_cache_signature(model::SphericalChordDMProfile) =
    profile_cache_signature(model.inner) * "|boundary=sphere$(model.sphere_r200c)R200c|cache=chord_mean"
profile_model_family(model::AbstractLee2022DMProfile) = lee2022_model_family(model)
profile_model_family(::Battaglia16DensityDMProfile) = B16_DENSITY_MODEL_FAMILY
profile_model_family(model::SphericalChordDMProfile) =
    profile_model_family(model.inner) * "_sphere" * replace(string(model.sphere_r200c), "." => "p") * "r200c_chordmean"

profile_provenance(model::Lee2022NoConcentrationDMProfile) = lee2022_no_concentration_provenance(model)
profile_provenance(model::Lee2022ConcentrationDMProfile) = lee2022_concentration_provenance(model)
profile_provenance(::Battaglia16DensityDMProfile) = Dict{String,Any}(
    "battaglia16_density_source" => "XGPaint BattagliaTauProfile parameters; rho_gas = P0 gNFW f_b rho_crit(z); ne2d composition",
    "battaglia16_los_max_r200c" => LEE2022_LOS_MAX_R200C,
)
function profile_provenance(model::SphericalChordDMProfile)
    return merge(profile_provenance(model.inner), Dict{String,Any}(
        "halo_boundary" => "spherical",
        "halo_boundary_sphere_r200c" => model.sphere_r200c,
        "halo_boundary_cache_quantity" => "chord-mean electron column g = DM_sphere/(2 L_max); exact chord factor applied per ray",
        "halo_boundary_impact_parameter" => "b/R200c = theta/theta200c with theta200c = atan(R200c/D_A)",
    ))
end

function run_spherical_boundary_self_test()
    b16_inner = XGPaint.HaloDMProfile(XGPaint.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68))
    b16 = Battaglia16DensityDMProfile(b16_inner)
    # 1. Profile-owned projected B16 reproduces XGPaint's HaloDMProfile.
    for (mass, z, xb) in ((1.0e14, 0.5, 0.01), (1.0e14, 0.5, 1.0), (7.327e12, 0.2, 0.5), (1.0e15, 1.0, 2.0))
        r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(b16, mass * XGPaint.M_sun, z, 200)
        theta = xb * XGPaint.angular_size(b16, r200c, z)
        ours = b16(theta, mass, z); ref = b16_inner(theta, mass, z)
        isapprox(ours, ref; rtol=1.0e-5) || error("B16 density LOS mismatch: $(ours) vs XGPaint $(ref) at M=$(mass) z=$(z) x=$(xb)")
    end
    # 2. Spherical chord: DM_sphere <= DM_projected, continuous to zero at the edge, and the
    #    chord-mean times the exact chord factor reproduces the direct spherical quadrature.
    sph = SphericalChordDMProfile(b16, 1.0)
    for (mass, z) in ((7.327e12, 0.5), (1.0e14, 0.5), (1.0e15, 0.2))
        r200c = getfield(XGPaint, Symbol("R_", Char(0x0394)))(b16, mass * XGPaint.M_sun, z, 200)
        theta200c = XGPaint.angular_size(b16, r200c, z)
        previous = Inf
        for xb in (0.01, 0.5, 0.9, 0.99, 0.999)
            theta = xb * theta200c
            direct = spherical_dm_pc_cm3(sph, theta, mass, z)
            rebuilt = sph(theta, mass, z) * spherical_chord_factor(theta, theta200c, 1.0)
            isapprox(direct, rebuilt; rtol=1.0e-9) || error("chord-mean reconstruction mismatch $(direct) vs $(rebuilt)")
            direct <= b16(theta, mass, z) * (1 + 1.0e-9) || error("spherical DM exceeds projected DM")
            direct < previous || error("spherical DM not decreasing outward")
            previous = direct
        end
        edge_g = sph(theta200c * (1 + 1.0e-9), mass, z)
        near_g = sph(theta200c * 0.9995, mass, z)
        isapprox(edge_g, near_g; rtol=5.0e-2) || error("chord-mean not continuous at the edge: $(near_g) vs $(edge_g)")
        spherical_chord_factor(theta200c * 1.0001, theta200c, 1.0) == 0.0 || error("chord factor must vanish outside")
    end
    lee = SphericalChordDMProfile(Lee2022NoConcentrationDMProfile(normalization=:baryon_fraction, n0_pivot=:mcut), 1.0)
    v = spherical_dm_pc_cm3(lee, 1.0e-4, 1.0e14, 0.5)
    isfinite(v) && v > 0 || error("Lee22 spherical DM invalid")
    println("PASS: profile-owned Battaglia16 LOS matches XGPaint; spherical chord-mean cache reconstruction, monotonicity and edge continuity.")
    return nothing
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
    # Option conventions: baryon-fraction factor, common M_cut pivot, shape clipping,
    # and the TNG-mean concentration anchors.
    fb_model = Lee2022NoConcentrationDMProfile(normalization=:baryon_fraction)
    @assert isapprox(fb_model(1.0e-4, 1.0e14, 0.5) / dm, 0.049 / 0.31; rtol=1e-12)
    pivot_model = Lee2022NoConcentrationDMProfile(n0_pivot=:mcut)
    @assert isapprox(lee2022_parameters(pivot_model, mass_cut, 0.0).n0, 6.8; rtol=2e-14)
    @assert isapprox(pivot_model(1.0e-4, 1.0e14, 0.5) / dm,
                     (1.0e14 / mass_cut)^0.68; rtol=1e-10)
    clip_model = Lee2022NoConcentrationDMProfile(shape_clip_mass_msun=LEE2022_FIT_MASS_MAX_MSUN_AT_H068)
    edge = lee2022_parameters(clip_model, LEE2022_FIT_MASS_MAX_MSUN_AT_H068, 0.3)
    beyond = lee2022_parameters(clip_model, 10 * LEE2022_FIT_MASS_MAX_MSUN_AT_H068, 0.3)
    @assert edge.x_c == beyond.x_c && edge.beta_prime == beyond.beta_prime
    @assert isapprox(beyond.n0 / edge.n0, 10.0^0.68; rtol=1e-12)
    @assert isapprox(lee2022_tng_mean_concentration(10.0^13.1 / 0.68, 0.0, 0.68), 5.65; rtol=1e-12)
    @assert isapprox(lee2022_tng_mean_concentration(10.0^14.1 / 0.68, 0.0, 0.68), 4.53; rtol=1e-12)
    pref = Lee2022ConcentrationDMProfile(normalization=:baryon_fraction, concentration_source=:tng_mean,
                                         shape_clip_mass_msun=LEE2022_FIT_MASS_MAX_MSUN_AT_H068)
    for logm in (14.0, 15.0, 15.6), zz in (0.0, 0.5, 1.0)
        pp = lee2022_parameters(pref, 10.0^logm, zz)
        @assert pp.beta_prime - pp.gamma > 1 "clipped preferred fit must keep a convergent LOS"
        @assert isfinite(pref(1.0e-4, 10.0^logm, zz))
    end
    zmodel = Lee2022NoConcentrationDMProfile(redshift_scaling=:comoving_hypothesis)
    @assert isapprox(lee2022_redshift_scaling_factor(zmodel, 0.0), 1.0; rtol=1e-12)
    e2_half = 0.31 * 1.5^3 + 0.69
    @assert isapprox(zmodel(1.0e-4, 1.0e14, 0.5) / dm, 1.5^3 / e2_half; rtol=2e-3)
    @assert lee2022_cache_signature(model) == LEE2022_NO_CONCENTRATION_CACHE_SIGNATURE
    @assert lee2022_cache_signature(pref) != LEE2022_CONCENTRATION_CACHE_SIGNATURE
    @assert lee2022_model_family(Lee2022ConcentrationDMProfile()) == LEE2022_CONCENTRATION_MODEL_FAMILY
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
