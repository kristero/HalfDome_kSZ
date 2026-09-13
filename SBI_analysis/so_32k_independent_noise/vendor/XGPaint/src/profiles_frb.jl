abstract type AbstractFRBProfile{T} <: AbstractBattagliaTauProfile{T} end

const M2_TO_PC_CM3 = ustrip(u"pc*cm^-3", 1u"m^-2")


struct HaloDMProfile{T,C} <: AbstractFRBProfile{T}
    f_b::T
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
end

function HaloDMProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=4.0e3, P0_alpha_m::T=0.29, P0_alpha_z::T=-0.66,
        x_c_amp::T=0.5, x_c_alpha_m::T=0.0, x_c_alpha_z::T=0.0,
        alpha_amp::T=0.88, alpha_alpha_m::T=-0.03, alpha_alpha_z::T=0.19,
        beta_amp::T=3.83, beta_alpha_m::T=0.04, beta_alpha_z::T=-0.025,
        gamma_amp::T=-0.2, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0) where {T <: Real}
    OmegaM = Omega_b + Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, Neff=3.046, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    return HaloDMProfile{T, typeof(cosmo)}(f_b, cosmo, P0, x_c, alpha, beta, gamma)
end

function HaloDMProfile(tau_model::BattagliaTauProfile{T,C}) where {T,C}
    return HaloDMProfile{T,C}(tau_model.f_b, tau_model.cosmo, tau_model.P0, tau_model.x_c,
        tau_model.alpha, tau_model.beta, tau_model.gamma)
end

BattagliaFRBProfile(; kwargs...) = HaloDMProfile(; kwargs...)

function object_size(model::HaloDMProfile{T,C}, physical_size, z) where {T,C}
    d_A = angular_diameter_dist(model.cosmo, z)
    phys_siz_unitless = T(ustrip(uconvert(unit(d_A), physical_size)))
    d_A_unitless = T(ustrip(d_A))
    return atan(phys_siz_unitless, d_A_unitless)
end

"""
    compute_DM(model, r, M_200c, z)

Compute the observed halo DM contribution in `pc/cm^3` directly from the
projected electron column density.
"""
function compute_DM(model::AbstractFRBProfile{T}, r, M_200c, z) where T
    electron_column = ustrip(u"m^-2", ne2d(model, r, M_200c, z))
    return T(electron_column * M2_TO_PC_CM3 / (one(z) + z))
end

# direct evaluation follows the rest of the profile API: mass is in Msun and
# the returned value is a plain number in pc/cm^3
(model::AbstractFRBProfile)(r, M_200c, z) = compute_DM(model, r, M_200c * M_sun, z)
