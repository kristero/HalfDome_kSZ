# Measure the numerical effect of the kristero/XGPaint.jl fork edits against upstream WebSky-CITA 29ec48a.
#
# Loads the fork (dev checkout /home/kn18001/.julia/dev/XGPaint, src tree = tag validated-tsz256-20260922)
# and evaluates, next to it, the upstream 29ec48a code of the functions the fork changed, transcribed
# verbatim from `git show 29ec48a:src/profiles_tau.jl` (l.33-42, 69-95) and `git show 29ec48a:src/profiles_y.jl`
# (l.32-60). The helpers both versions share (rho_crit, R_Delta, object_size, angular_size, constants) are
# unchanged between 29ec48a and the fork (git diff 29ec48a 1660e08 touches only XGPaint.jl, model.jl
# [additions], profiles.jl, profiles_frb.jl, profiles_tau.jl, profiles_y.jl), so they are taken from the fork.
#
# Writes CSV files next to this script. Light: a few thousand quadratures, well under a minute.
using XGPaint, Unitful, UnitfulAstro, Printf
const XG = XGPaint
const OUT = @__DIR__

module Up29ec48a
using XGPaint, Unitful, UnitfulAstro
import XGPaint: ρ_crit, R_Δ, object_size, constants, M_sun
const quadgk = XGPaint.QuadGK.quadgk
# ---- upstream profiles_y.jl l.32-39
function generalized_nfw(x, xc, α, β, γ)
    x̄ = x / xc
    return x̄^γ * (1 + x̄^α)^((β - γ) / α)
end
# ---- upstream profiles_y.jl l.42-52 (Battaglia12 parameters, Sigurd's conversion)
function get_params_y(::Type{T}, M_200, z) where T
	z₁ = z + 1
	m = M_200 / (1e14M_sun)
	P₀ = 18.1 * m^0.154 * z₁^-0.758
	xc = 0.497 * m^-0.00865 * z₁^0.731
	β = 4.35 * m^0.0393 * z₁^0.415
	α = 1
    γ = -0.3
    β = γ - α * β  # Sigurd's conversion from Battaglia to standard NFW
    return (xc=T(xc), α=T(α), β=T(β), γ=T(γ), P₀=T(P₀))
end
# ---- upstream profiles_y.jl l.54-60
function _nfw_profile_los_quadrature(x, xc, α, β, γ; zmax=1e5, rtol=eps(), order=9)
    x² = x^2
    scale = 1e9
    integral, err = quadgk(y -> scale * generalized_nfw(√(y^2 + x²), xc, α, β, γ),
                      0.0, zmax, rtol=rtol, order=order)
    return 2integral / scale
end
# ---- upstream profiles_tau.jl l.33-42
function get_params_tau(::Type{T}, M_200c, z) where T
	z₁ = z + 1
	m = M_200c / (1e14M_sun)
    P₀ = 4.e3 * m^0.29 * z₁^(-0.66)
	α =  0.88 * m^(-0.03) * z₁^0.19
	β = -3.83 * m^0.04 * z₁^(-0.025)
	xc = 0.5
    γ = -0.2
    return (xc=T(xc), α=T(α), β=T(β), γ=T(γ), P₀=T(P₀))
end
# ---- upstream profiles_tau.jl l.44-46, 69-95
function ρ_crit_comoving_h⁻²(model, z)
    return  (ρ_crit(model, z) ) / (1+z)^3 / model.cosmo.h^2
end
function rho_2d(model, r, m200c, z)
    par = get_params_tau(Float64, m200c, z)
    r200c = R_Δ(model, m200c, z, 200)
    X = r / object_size(model, r200c, z)  # either ang/ang or phys/phys
    rho_crit = ρ_crit_comoving_h⁻²(model, z)  # need to sort this out, it's all in comoving...
    result = par.P₀ * _nfw_profile_los_quadrature(X, par.xc, par.α, par.β, par.γ)
    return result * rho_crit * (r200c * (1+z))
end
function ne2d(model, r, m200c, z)
    me = constants.ElectronMass
    mH = constants.ProtonMass
    mHe = 4mH
    xH = 0.76
    nH_ne = 2xH / (xH + 1)
    nHe_ne = (1 - xH)/(2 * (1 + xH))
    factor = (me + nH_ne*mH + nHe_ne*mHe) / model.cosmo.h^2
    result = rho_2d(model, r, m200c, z)  # (Msun/h) / (Mpc/h)^2
    return result / factor
end
compute_tau(model, r, m200c, z) = constants.ThomsonCrossSection * ne2d(model, r, m200c, z) + 0
# upstream y: P_e_los/P_th_los/compton_y are unchanged; only dimensionless_P_profile_los differs through get_params
function dimensionless_P_profile_los(model, r, M_200, z)
    par = get_params_y(Float64, M_200, z)
    R_200 = R_Δ(model, M_200, z, 200)
    x = r / XGPaint.angular_size(model, R_200, z)
    return par.P₀ * _nfw_profile_los_quadrature(x, par.xc, par.α, par.β, par.γ)
end
# upstream r200c_comoving (profiles_tau.jl l.48-51)
function r200c_comoving(model, M_200c, z)
    rho_crit = ρ_crit(model, z) / (1+z)^3
    return cbrt(M_200c/ (4π/3 * rho_crit * 200))
end
end # module
const Up = Up29ec48a

# HalfDome painter cosmology used by the production y/tau/DM painters
tau = XG.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
tsz = XG.Battaglia16ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
dmp = XG.BattagliaFRBProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
fb = Float64(tau.f_b)
θ200(model, M, z) = Float64(XG.angular_size(model, XG.R_Δ(model, M * XG.M_sun, z, 200), z))

masses = (1e13, 1e14, 1e15)
zs = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
xs = (0.1, 0.5, 1.0, 2.0, 3.5)

# 1. tau, y and DM: fork / upstream at the same (theta, M, z)
open(joinpath(OUT, "fork_vs_upstream_29ec48a.csv"), "w") do io
    println(io, "M_msun,z,x,tau_fork,tau_up,tau_ratio,expected_0p9_fb_1pz2,yshape_ratio,dm_fork,dm_v04_fallback,dm_ratio")
    for M in masses, z in zs, x in xs
        θ = x * θ200(tau, M, z); Mu = M * XG.M_sun
        tf = Float64(tau(θ, M, z)); tu = Float64(Up.compute_tau(tau, θ, Mu, z))
        yf = Float64(XG.dimensionless_P_profile_los(tsz, θ, Mu, z)); yu = Float64(Up.dimensionless_P_profile_los(tsz, θ, Mu, z))
        dmf = Float64(dmp(θ, M, z))
        # public-v0.4 fallback of generate_halfdome_z1_dm_mass_windows.jl l.80-107: tau / (sigma_T[cm^2] pc[cm] (1+z))
        dmu = tu / (6.6524587321e-25 * 3.0856775814913673e18 * (1 + z))
        @printf(io, "%.3e,%.3f,%.2f,%.10e,%.10e,%.10f,%.10f,%.12f,%.10e,%.10e,%.10f\n",
                M, z, x, tf, tu, tf / tu, 0.9 * fb * (1 + z)^2, yf / yu, dmf, dmu, dmf / dmu)
    end
end

# 2. 3-D shape functions at M = 1e14 Msun, z = 0.5 (for the sign-bug figure)
open(joinpath(OUT, "shape_functions_m1e14_z0p5.csv"), "w") do io
    println(io, "x,b12_upstream,b12_fork,b12_b924ab1,b16_upstream,b16_fork")
    Mu = 1e14 * XG.M_sun; z = 0.5
    pf = XG.get_params(tsz, Mu, z); pu = Up.get_params_y(Float64, Mu, z)
    βraw = 4.35 * 1.0^0.0393 * 1.5^0.415
    βbug = -pf.γ - pf.α * βraw                       # b924ab1: beta = -gamma - alpha*beta_raw
    tf_ = XG.get_params(tau, Mu, z); tu_ = Up.get_params_tau(Float64, Mu, z)
    for lx in range(log10(0.01), log10(10.0); length=121)
        x = 10.0^lx
        @printf(io, "%.6e,%.10e,%.10e,%.10e,%.10e,%.10e\n", x,
                Up.generalized_nfw(x, pu.xc, pu.α, pu.β, pu.γ), XG.generalized_nfw(x, pf.xc, pf.α, pf.β, pf.γ),
                XG.generalized_nfw(x, pf.xc, pf.α, βbug, pf.γ),
                Up.generalized_nfw(x, tu_.xc, tu_.α, tu_.β, tu_.γ), XG.generalized_nfw(x, tf_.xc, tf_.α, tf_.β, tf_.γ))
    end
end

# 3. b924ab1 projected y: buggy / correct LOS integral (rtol 1e-12 instead of b924ab1's eps(): the integrand grows as x^4)
open(joinpath(OUT, "b924ab1_los_ratio.csv"), "w") do io
    println(io, "M_msun,z,x,los_bug_over_correct")
    for M in masses, z in (0.1, 0.5, 1.0, 2.0), x in (0.1, 1.0, 3.5)
        p = XG.get_params(tsz, M * XG.M_sun, z)
        βraw = (p.β + p.γ) / p.α                      # invert beta = alpha*beta_raw - gamma
        good = XG._nfw_profile_los_quadrature(x, p.xc, p.α, p.β, p.γ)
        bad = XG._nfw_profile_los_quadrature(x, p.xc, p.α, -p.γ - p.α * βraw, p.γ)
        @printf(io, "%.3e,%.2f,%.2f,%.6e\n", M, z, x, bad / good)
    end
end

# 4. quadrature settings: cbf5cbd (rtol 1e-3, order 7), upstream/b924ab1 (eps(), 9), ce1d239/HEAD (1e-12, 9)
open(joinpath(OUT, "quadrature_settings.csv"), "w") do io
    println(io, "profile,M_msun,z,x,rel_cbf5cbd_vs_head,rel_eps_vs_head")
    for (name, model) in (("B12_y", tsz), ("B16_tau", tau)), M in masses, z in (0.05, 0.5, 2.0), x in (0.02, 0.1, 0.5, 1.0, 2.0, 3.9)
        p = XG.get_params(model, M * XG.M_sun, z)
        head = XG._nfw_profile_los_quadrature(x, p.xc, p.α, p.β, p.γ)
        c7 = XG._nfw_profile_los_quadrature(x, p.xc, p.α, p.β, p.γ; rtol=1e-3, order=7)
        ep = XG._nfw_profile_los_quadrature(x, p.xc, p.α, p.β, p.γ; rtol=eps(), order=9)
        @printf(io, "%s,%.3e,%.2f,%.2f,%.3e,%.3e\n", name, M, z, x, c7 / head - 1, ep / head - 1)
    end
end

# 5. constants and identities
open(joinpath(OUT, "constants_and_identities.txt"), "w") do io
    println(io, "M2_TO_PC_CM3 (fork, 71646e2+) = ", XG.M2_TO_PC_CM3)
    println(io, "M2_TO_PC_CM3 (e67a8a1 hard-coded) = 4.87e5; ratio e67a8a1/correct = ", 4.87e5 / XG.M2_TO_PC_CM3)
    println(io, "f_b (Omega_b 0.049 / Omega_m 0.31) = ", fb)
    me = XG.constants.ElectronMass; mp = XG.constants.ProtonMass
    mpe = ustrip(u"kg", me + 2mp / 1.76); println(io, "m_per_e / m_p = ", mpe / ustrip(u"kg", mp))
    println(io, "XGPaint n_e = 0.9 rho/m_per_e = ", 0.9 * ustrip(u"kg", mp) / mpe, " rho/m_p")
    worst = 0.0
    for M in masses, z in zs
        a = Up.r200c_comoving(tau, M * XG.M_sun, z); b = XG.r200c_comoving(tau, M * XG.M_sun, z)
        worst = max(worst, abs(ustrip(u"Mpc", a) / ustrip(u"Mpc", b) - 1))
    end
    println(io, "r200c_comoving upstream (profiles_tau.jl) vs fork (model.jl): max rel diff = ", worst)
end
println("done")
