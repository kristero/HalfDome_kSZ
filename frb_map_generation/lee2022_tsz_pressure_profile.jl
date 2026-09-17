# Lee et al. 2022 (arXiv:2205.01710 v1 Table 7 = MNRAS 517, 420 Table A1) no-concentration electron-pressure
# fit as an XGPaint thermal-SZ profile. Only get_params (and the y evaluation helpers XGPaint types
# for its own Battaglia profile) are provided; XGPaint's LOS quadrature, interpolator and painter run unchanged.
#
#   Lee22 eq. (10):  P_e / P200 = P0 (x/x_c)^gamma [1 + (x/x_c)^alpha]^(-beta),   alpha = 1, gamma = -0.3
#   Lee22 eq. (9):   P200 = 200 G M200 rho_cr(z) Omega_b / (2 R200 Omega_m)      (the Battaglia P200)
#   Table 7 (BPL in P0): P0 = 2.8 (M/M_cut)^1.08 [M < M_cut] or ^0.80 [M > M_cut] (1+z)^-1.89,
#                        x_c = 2.10 (M/M_cut)^-0.35 (1+z)^0.53,  beta = 9.4 (M/M_cut)^-0.06 (1+z)^0.60,
#                        log10 M_cut = 13.60 h^-1 Msun.  Fit range 0.04-1.34 R200, 1e13-10^14.8 h^-1 Msun, z <= 2.
#
#   XGPaint (profiles_y.jl): y = P_e_factor * 0.5176 * [G M200 200 rho_cr f_b / 2] * P0_XG * LOS(x; xc, alpha, beta_XG, gamma)
#   with generalized_nfw exponent -(beta_XG + gamma)/alpha.  Therefore
#       P0_XG = P0_Lee / 0.5176   (Lee22 fits the electron pressure directly; XGPaint converts thermal -> electron)
#       beta_XG = alpha * beta_Lee - gamma  (= beta_Lee + 0.3)
#   Shape parameters (x_c, beta) are frozen above the fit-range mass (same treatment as the Lee22 density fit);
#   P0 keeps its fitted high-mass slope 0.80.

const LEE22_PRESSURE_MCUT_HINV_MSUN = 10.0^13.60
const LEE22_PRESSURE_FIT_MASS_MAX_HINV_MSUN = 10.0^14.8
const LEE22_PRESSURE_ELECTRON_TO_THERMAL = 0.5176   # XGPaint's P_e = 0.5176 P_th

struct Lee2022ThermalSZProfile{T,C} <: XGPaint.AbstractGNFW{T}
    f_b::T
    cosmo::C
    shape_clip_mass_msun::Float64
end

function Lee2022ThermalSZProfile(; Omega_c::T=0.261, Omega_b::T=0.049, h::T=0.68,
                                 shape_clip_mass_msun::Real=LEE22_PRESSURE_FIT_MASS_MAX_HINV_MSUN / 0.68) where {T<:Real}
    OmegaM = Omega_b + Omega_c
    cosmo = XGPaint.get_cosmology(T; h=h, Neff=T(3.046), OmegaM=OmegaM)
    return Lee2022ThermalSZProfile{T,typeof(cosmo)}(Omega_b / OmegaM, cosmo, Float64(shape_clip_mass_msun))
end

"""Lee22 Table 7 parameters (P0, x_c, beta in the paper's convention) for physical M200c in Msun."""
function lee2022_pressure_parameters(model::Lee2022ThermalSZProfile, mass_msun::Real, z::Real)
    mass = Float64(mass_msun); z1 = 1.0 + Float64(z)
    mcut = LEE22_PRESSURE_MCUT_HINV_MSUN / Float64(model.cosmo.h)
    ratio = mass / mcut
    p0 = 2.8 * (ratio < 1 ? ratio^1.08 : ratio^0.80) * z1^(-1.89)
    shape_ratio = min(mass, model.shape_clip_mass_msun) / mcut
    x_c = 2.10 * shape_ratio^(-0.35) * z1^0.53
    beta = 9.4 * shape_ratio^(-0.06) * z1^0.60
    return (P0=p0, x_c=x_c, alpha=1.0, beta=beta, gamma=-0.3, mass_cut_msun=mcut)
end

function XGPaint.get_params(model::Lee2022ThermalSZProfile{T}, M_200c, z) where {T}
    p = lee2022_pressure_parameters(model, Float64(M_200c / XGPaint.M_sun), z)
    return (xc=T(p.x_c), α=T(p.alpha), β=T(p.alpha * p.beta - p.gamma), γ=T(p.gamma),
            P₀=T(p.P0 / LEE22_PRESSURE_ELECTRON_TO_THERMAL))
end

function XGPaint.dimensionless_P_profile_los(model::Lee2022ThermalSZProfile{T}, r, M_200c, z) where {T}
    par = XGPaint.get_params(model, M_200c, z)
    r200 = XGPaint.R_Δ(model, M_200c, z, 200)
    x = r / XGPaint.angular_size(model, r200, z)
    return par.P₀ * XGPaint._nfw_profile_los_quadrature(x, par.xc, par.α, par.β, par.γ)
end

XGPaint.compton_y(model::Lee2022ThermalSZProfile, r, M_200c, z) =
    XGPaint.P_e_los(model, r, M_200c, z) * XGPaint.P_e_factor + 0

(model::Lee2022ThermalSZProfile)(r, M_200c, z) = XGPaint.compton_y(model, r, M_200c * XGPaint.M_sun, z)

# Same per-(M, z) fast path as XGPaint's Battaglia profile (used by profile_grid / build_interpolator)
function XGPaint.prepare_profile_slice(model::Lee2022ThermalSZProfile{T}, mass, redshift) where {T}
    mass_with_units = mass * XGPaint.M_sun
    par = XGPaint.get_params(model, mass_with_units, redshift)
    r200 = XGPaint.R_Δ(model, mass_with_units, redshift, 200)
    theta_scale = T(XGPaint.angular_size(model, r200, redshift))
    amplitude = T((LEE22_PRESSURE_ELECTRON_TO_THERMAL * XGPaint.constants.G * mass_with_units * 200 *
                   XGPaint.ρ_crit(model, redshift) * model.f_b / 2 * XGPaint.P_e_factor * par.P₀) + 0)
    return (; theta_scale, xc=par.xc, alpha=par.α, beta=par.β, gamma=par.γ, amplitude)
end

@inline function XGPaint.evaluate_profile_slice(::Lee2022ThermalSZProfile, prepared, theta, mass, redshift)
    return prepared.amplitude * XGPaint._nfw_profile_los_quadrature(theta / prepared.theta_scale,
        prepared.xc, prepared.alpha, prepared.beta, prepared.gamma)
end

lee2022_pressure_provenance(model::Lee2022ThermalSZProfile) = Dict{String,Any}(
    "profile_label" => "Lee22 no-concentration electron pressure (arXiv v1 Table 7 = MNRAS Table A1)",
    "xgpaint_profile_type" => "Lee2022ThermalSZProfile (get_params wrapper; XGPaint LOS/interpolator/painter unchanged)",
    "lee22_pressure_P0" => "2.8 (M/M_cut)^1.08 [M<M_cut] or ^0.80 [M>M_cut] (1+z)^-1.89",
    "lee22_pressure_xc" => "2.10 (M/M_cut)^-0.35 (1+z)^0.53",
    "lee22_pressure_beta" => "9.4 (M/M_cut)^-0.06 (1+z)^0.60",
    "lee22_pressure_alpha" => 1.0, "lee22_pressure_gamma" => -0.3,
    "lee22_pressure_mcut_hinv_msun" => LEE22_PRESSURE_MCUT_HINV_MSUN,
    "lee22_pressure_mcut_msun" => LEE22_PRESSURE_MCUT_HINV_MSUN / Float64(model.cosmo.h),
    "lee22_pressure_shape_clip_mass_msun" => model.shape_clip_mass_msun,
    "lee22_pressure_P200" => "200 G M200 rho_cr(z) f_b / (2 R200) (Battaglia definition; eq. 9)",
    "lee22_pressure_electron_conversion" => "fit is P_e/P200; XGPaint P0 = P0_Lee/0.5176 so that 0.5176 P_th = P_e",
    "lee22_pressure_exponent" => "beta_XG = alpha beta - gamma (XGPaint exponent -(beta+gamma)/alpha = -beta_Lee)",
    "lee22_pressure_fit_range" => "0.04-1.34 R200; 1e13-10^14.8 h^-1 Msun; 20 TNG300 snapshots to z = 2",
    "lee22_pressure_aperture_note" => "painted with the same XGPaint projected aperture as the Battaglia12 map",
)

"""Self-test: exponent identity, an independent y integral from eq. (9)-(10), and the interpolator fast path."""
function run_lee2022_tsz_self_test(model::Lee2022ThermalSZProfile; quad=nothing)
    quad === nothing && error("Pass a quadgk function")
    for (mass, z, x) in ((1.0e13, 0.1, 0.5), (1.0e14, 0.5, 1.0), (5.0e14, 1.5, 0.05), (2.0e15, 0.3, 2.0))
        p = lee2022_pressure_parameters(model, mass, z)
        par = XGPaint.get_params(model, mass * XGPaint.M_sun, z)
        lee = p.P0 * (x / p.x_c)^p.gamma * (1 + (x / p.x_c)^p.alpha)^(-p.beta)
        xg = par.P₀ * LEE22_PRESSURE_ELECTRON_TO_THERMAL * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ)
        isapprox(lee, xg; rtol=1e-12) || error("Lee22 pressure exponent/normalization mapping failed: $(lee) vs $(xg)")
        # independent Compton-y: y = P_e_factor * P200 * R200 * P0 * 2 * int_0^inf f(sqrt(x^2 + l^2)) dl
        M = mass * XGPaint.M_sun
        r200 = XGPaint.R_Δ(model, M, z, 200)
        p200 = 200 * XGPaint.constants.G * M * XGPaint.ρ_crit(model, z) * model.f_b / (2 * r200)
        los = 2 * quad(l -> (sqrt(x^2 + l^2) / p.x_c)^p.gamma * (1 + (sqrt(x^2 + l^2) / p.x_c)^p.alpha)^(-p.beta),
                       0.0, 1.0e5; rtol=1e-10, order=9)[1]
        y_direct = Float64(XGPaint.P_e_factor * p200 * r200 * p.P0 * los + 0)   # + 0 strips the (cancelled) units, as in XGPaint
        theta = x * Float64(XGPaint.angular_size(model, r200, z))
        y_model = Float64(model(theta, mass, z))
        isapprox(y_direct, y_model; rtol=1e-6) || error("Lee22 pressure y mismatch: direct $(y_direct) vs XGPaint path $(y_model)")
        prepared = XGPaint.prepare_profile_slice(model, mass, z)
        y_slice = Float64(XGPaint.evaluate_profile_slice(model, prepared, theta, mass, z))
        isapprox(y_slice, y_model; rtol=1e-9) || error("Lee22 pressure interpolator fast path mismatch")
    end
    println("PASS: Lee22 pressure -> XGPaint mapping (P0/0.5176, beta+0.3), independent eq. 9-10 Compton-y integral, fast path")
    return nothing
end
