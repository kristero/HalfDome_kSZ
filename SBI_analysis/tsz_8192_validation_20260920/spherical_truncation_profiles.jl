# Chord-limited (spherical) truncation for XGPaint gNFW profiles.
#
# XGPaint's Battaglia16ThermalSZProfile and BattagliaTauProfile integrate the 3-D
# profile along the full line of sight (`_nfw_profile_los_quadrature`, 0..1e5 R200c)
# and the painter only cuts the projected disc at theta_max = X R200c. So a ray at
# projected radius x < X also collects gas at 3-D radii r > X R200c ("projected"
# truncation). The FRB pipeline was changed on 2026-09-16 to count only gas inside
# the X R200c sphere (chord-limited LOS, DM -> 0 at the edge). This file provides
# the same fix for tSZ (Compton-y) and kSZ (tau) so the two can be compared.
#
# ChordMeanProfile caches g = value_sphere / (2 L), L = sqrt(X^2 - x^2) in R200c
# units, which is positive and continuous (tends to the 3-D profile value at the
# edge), so XGPaint's log interpolator works unchanged. The painter multiplies the
# interpolated g by the exact chord factor 2 X sqrt(1 - (theta/theta_max)^2).
module SphericalTruncation

using XGPaint
const quadgk = XGPaint.QuadGK.quadgk

export ChordMeanProfile, chord_factor, theta_r200c, spherical_value_direct

struct ChordMeanProfile{T,C,P} <: XGPaint.AbstractGNFW{T}
    cosmo::C
    inner::P
    sphere::Float64      # X: sphere radius in units of R200c
end

function ChordMeanProfile(inner::XGPaint.AbstractGNFW{T}, sphere::Real) where {T}
    isfinite(sphere) && sphere > 0 || error("sphere radius must be positive")
    return ChordMeanProfile{T,typeof(inner.cosmo),typeof(inner)}(inner.cosmo, inner, Float64(sphere))
end

@inline log1pexp_stable(v) = max(v, 0.0) + log1p(exp(-abs(v)))

"""Log of r times the gNFW shape, including the LOS-transform Jacobian."""
@inline function log_radial_integrand(logr, xc, α, β, γ)
    # XGPaint's internal beta is the asymptotic outer slope. For tSZ its
    # get_params converts beta_raw to alpha*beta_raw-gamma.
    exponent = (β + γ) / α
    return (1 + γ) * logr - γ * log(xc) -
           exponent * log1pexp_stable(α * (logr - log(xc)))
end

"""Peak log amplitude over the finite radial interval; no outer-slope cut."""
function log_integrand_peak(loglo, loghi, xc, α, β, γ)
    peak = log_radial_integrand(loghi, xc, α, β, γ)
    isfinite(loglo) && (peak = max(peak, log_radial_integrand(loglo, xc, α, β, γ)))
    if 1 + γ > 0 && β > 1
        stationary = log(xc) + log((1 + γ) / (β - 1)) / α
        at = clamp(stationary, loglo, loghi)
        peak = max(peak, log_radial_integrand(at, xc, α, β, γ))
    end
    return peak
end

"""Finite chord integral with normalized positive quadrature.

For x>0 use l=x*sinh(u), r=x*cosh(u), dl=r*du. At x=0 integrate
in log(r), which handles the integrable central cusp without evaluating it.
The peak is only a numerical scale; its factor is restored in the result.
"""
function checked_quadrature(f, lo, hi; rtol=1.0e-10, maxevals=4096)
    value, error = quadgk(f, lo, hi; rtol=rtol, atol=0.0, order=9, maxevals=maxevals)
    isfinite(value) && value > 0 && isfinite(error) && error <= rtol*value ||
        throw(ErrorException("Finite-chord quadrature failed: value=$value error=$error rtol=$rtol maxevals=$maxevals"))
    return value
end

function chord_quadrature(x, xc, α, β, γ, L; rtol=1.0e-10, maxevals=4096)
    L <= 0 && return 0.0
    x >= 0 && xc > 0 && α > 0 || error("invalid gNFW geometry/scale")
    γ > -1 || x > 0 || error("central LOS is divergent for gamma <= -1")
    loghi = log(hypot(x, L))
    loglo = x == 0 ? -Inf : log(x)
    peak = log_integrand_peak(loglo, loghi, xc, α, β, γ)
    if x == 0
        central = logr -> exp(log_radial_integrand(logr, xc, α, β, γ) - peak)
        I = checked_quadrature(central, -Inf, loghi; rtol=rtol, maxevals=maxevals)
    else
        offset = u -> exp(log_radial_integrand(log(x) + log(cosh(u)), xc, α, β, γ) - peak)
        I = checked_quadrature(offset, 0.0, asinh(L/x); rtol=rtol, maxevals=maxevals)
    end
    return exp(log(2I) + peak)
end

"""Chord-mean h(x) = chord integral / (2L) inside the sphere; the 3-D profile value
outside (continuous at x = X, never used by the painter there)."""
function chord_mean(x, xc, α, β, γ, X)
    if x < X
        L = sqrt((X - x) * (X + x))
        return chord_quadrature(x, xc, α, β, γ, L) / (2L)
    end
    return XGPaint.generalized_nfw(x, xc, α, β, γ)
end

# Generic fallback preserves the existing tau/DM wrappers. The tSZ methods
# below deliberately avoid evaluating any old LOS integral for normalization.
function projection_amplitude(inner, mass, z, θ200, xc, α, β, γ)
    return Float64(inner(θ200, mass, z)) /
           XGPaint._nfw_profile_los_quadrature(1.0, xc, α, β, γ)
end
function projection_amplitude(inner::Union{XGPaint.Battaglia16ThermalSZProfile,
                                          XGPaint.BreakModel}, mass, z, args...)
    return Float64(XGPaint.prepare_profile_slice(inner, mass, z).amplitude)
end

"""Angular radius of R200c (radians) for a halo of mass `mass` [Msun] at redshift z."""
function theta_r200c(model, mass, z)
    r200 = XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200)
    return Float64(XGPaint.angular_size(model, r200, z))
end

"""Exact chord factor 2 L_max at angle theta given theta_max = X theta200c."""
@inline function chord_factor(theta, theta_max, X)
    ratio = theta / theta_max
    ratio >= 1 && return 0.0
    return 2X * sqrt(1 - ratio^2)
end

# XGPaint's profile_grid calls prepare_profile_slice once per (M, z) and
# evaluate_profile_slice per theta; both are extended for this type.
function XGPaint.prepare_profile_slice(m::ChordMeanProfile, mass, z)
    inner = m.inner
    Mu = mass * XGPaint.M_sun
    par = XGPaint.get_params(inner, Mu, z)
    θ200 = theta_r200c(inner, mass, z)
    xc, α, β, γ = Float64(par.xc), Float64(par.α), Float64(par.β), Float64(par.γ)
    A = projection_amplitude(inner, mass, z, θ200, xc, α, β, γ)
    return (; θ200, xc, α, β, γ, A, X=m.sphere)
end

@inline function XGPaint.evaluate_profile_slice(::ChordMeanProfile, p, theta, mass, z)
    x = Float64(theta) / p.θ200
    return p.A * chord_mean(x, p.xc, p.α, p.β, p.γ, p.X)
end

# Direct callable: the cached quantity g(theta, M, z).
function (m::ChordMeanProfile{T})(theta, mass, z) where {T}
    p = XGPaint.prepare_profile_slice(m, mass, z)
    return T(XGPaint.evaluate_profile_slice(m, p, theta, mass, z))
end

"""Direct spherical value (no cache): A * chord integral inside the X R200c sphere."""
function spherical_value_direct(m::ChordMeanProfile, theta, mass, z)
    p = XGPaint.prepare_profile_slice(m, mass, z)
    x = Float64(theta) / p.θ200
    x >= p.X && return 0.0
    return p.A * chord_quadrature(x, p.xc, p.α, p.β, p.γ, sqrt((p.X-x)*(p.X+x)))
end

end # module
