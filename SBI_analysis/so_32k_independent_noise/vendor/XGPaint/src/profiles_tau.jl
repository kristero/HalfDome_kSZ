
# can be used with either angular (default) or physical units
abstract type AbstractBattagliaTauProfile{T} <: AbstractGNFW{T} end

# default is radian angle radial units
struct BattagliaTauProfile{T,C} <: AbstractBattagliaTauProfile{T}
    f_b::T  # Omega_b / Omega_c = 0.0486 / 0.2589
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
end

# alternative: physical units in the radial direction
struct BattagliaTauProfilePhysical{T,C} <: AbstractBattagliaTauProfile{T}
    f_b::T  # Omega_b / Omega_c = 0.0486 / 0.2589
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
end


function BattagliaTauProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=4.0e3, P0_alpha_m::T=0.29, P0_alpha_z::T=-0.66,
        x_c_amp::T=0.5, x_c_alpha_m::T=0.0, x_c_alpha_z::T=0.0,
        alpha_amp::T=0.88, alpha_alpha_m::T=-0.03, alpha_alpha_z::T=0.19,
        beta_amp::T=3.83, beta_alpha_m::T=0.04, beta_alpha_z::T=-0.025,
        gamma_amp::T=-0.2, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, Neff=3.046, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    return BattagliaTauProfile{T, typeof(cosmo)}(f_b, cosmo, P0, x_c, alpha, beta, gamma)
end


function BattagliaTauProfilePhysical(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=4.0e3, P0_alpha_m::T=0.29, P0_alpha_z::T=-0.66,
        x_c_amp::T=0.5, x_c_alpha_m::T=0.0, x_c_alpha_z::T=0.0,
        alpha_amp::T=0.88, alpha_alpha_m::T=-0.03, alpha_alpha_z::T=0.19,
        beta_amp::T=3.83, beta_alpha_m::T=0.04, beta_alpha_z::T=-0.025,
        gamma_amp::T=-0.2, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, Neff=3.046, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    return BattagliaTauProfilePhysical{T, typeof(cosmo)}(f_b, cosmo, P0, x_c, alpha, beta, gamma)
end

function get_params(model::AbstractBattagliaTauProfile{T}, M_200c, z) where T
	z₁ = z + 1
	m = M_200c / (1e14M_sun)
    P₀ = powerlaw_value(model.P0, m, z₁)
	α = powerlaw_value(model.alpha, m, z₁)
	β = powerlaw_value(model.beta, m, z₁)
	xc = powerlaw_value(model.x_c, m, z₁)
    γ = powerlaw_value(model.gamma, m, z₁)
    return (xc=T(xc), α=T(α), β=T(β), γ=T(γ), P₀=T(P₀))
end

# if angular, return the R200 size in radians
function object_size(model::BattagliaTauProfile{T,C}, physical_size, z) where {T,C}
    d_A = angular_diameter_dist(model.cosmo, z)
    phys_siz_unitless = T(ustrip(uconvert(unit(d_A), physical_size)))
    d_A_unitless = T(ustrip(d_A))
    return atan(phys_siz_unitless, d_A_unitless)
end

# if physical, return the R200 size in Mpc
function object_size(::BattagliaTauProfilePhysical{T,C}, physical_size, z) where {T,C}
    return physical_size
end

# returns a density, which we can check against Msun/Mpc² 
function rho_2d(model::AbstractBattagliaTauProfile, r, m200c, z)
    par = get_params(model, m200c, z)
    r200c = R_Δ(model, m200c, z, 200)   # this is physical units
    X = r / object_size(model, r200c, z)  # either ang/ang or phys/phys
    rho_crit = ρ_crit(model, z)   # this is physical units
    rho_fit = par.P₀ * XGPaint._nfw_profile_los_quadrature(X, par.xc, par.α, par.β, par.γ)   # dimensionless _nfw_profile_los_quadrature comes from profiles_y.jl TODO move
    return rho_fit * model.f_b * rho_crit * r200c # mistake in battaglia 2016: need f_b to convert from m to gas. note r200c due to integral over X
end

function ne2d(model::AbstractBattagliaTauProfile, r, m200c, z)
    me = constants.ElectronMass
    mH = constants.ProtonMass
    mHe = 4mH
    xH = 0.76
    nH_ne = 2xH / (xH + 1)
    nHe_ne = (1 - xH)/(2 * (1 + xH))
    factor = (me + nH_ne*mH + nHe_ne*mHe)
    result = rho_2d(model, r, m200c, z)  # (Msun) / (Mpc)^2
    return 0.9 * result / factor
end

# r is either a physical or angular radius. unitful does not do little h, so physical radius 
# is always in Mpc, and angular radius is always in radians.
function compute_tau(model, r, m200c, z)
    return constants.ThomsonCrossSection * ne2d(model, r, m200c, z) + 0  # ensure unitless
end


# evaluation functions never take Unitful inputs, only raw numbers
# mass is in Msun
function (tau_model::AbstractBattagliaTauProfile)(r, m200c, z)
    return compute_tau(tau_model, r, m200c * M_sun, z)
end
