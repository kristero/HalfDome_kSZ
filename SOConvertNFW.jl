module M200Convert

export m200m_to_m200c, m200m_to_m200c_arrays

# Units: H in (km/s)/Mpc, R in Mpc, M in Msun
const G = 4.30091e-9  # (Mpc * (km/s)^2) / Msun
const π = Base.MathConstants.pi

# -------- Cosmology (your values) --------
const Ωb0 = 0.049
const Ωc0 = 0.31 - Ωb0
const Ωm0 = 0.31
const h   = 0.68
const H0  = 100.0 * h   # km/s/Mpc
const Ωk0 = 0.0
const ΩΛ0 = 1.0 - Ωm0 - Ωk0
R_is_comoving = true

@inline E_z(z) = sqrt(Ωm0*(1+z)^3 + Ωk0*(1+z)^2 + ΩΛ0)
@inline H_z(z) = H0 * E_z(z)

@inline function rho_crit_z(z; R_is_comoving::Bool=false)
    ρc = 3.0 * H_z(z)^2 / (8.0 * π * G)  # Msun / Mpc^3 (physical)
    return R_is_comoving ? ρc / (1+z)^3 : ρc
end

@inline function Omega_m_z(z)
    return Ωm0*(1+z)^3 / (E_z(z)^2)
end

@inline function rho_mean_z(z; R_is_comoving::Bool=false)
    ρm = Omega_m_z(z) * rho_crit_z(z; R_is_comoving=R_is_comoving)
    return ρm
end

# -------- NFW helpers --------
@inline f_nfw(y) = log1p(y) - y/(1+y)

@inline function R_from_M(M, Δ, ρref)
    return (3.0*M / (4.0*π*Δ*ρref))^(1/3)
end

function bisect_root(f, a::Float64, b::Float64; rtol=1e-10, maxiter=200)
    fa, fb = f(a), f(b)
    if fa == 0.0; return a end
    if fb == 0.0; return b end
    if sign(fa) == sign(fb)
        error("Root not bracketed (try wider bracket). f(a)=$fa, f(b)=$fb")
    end
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in 1:maxiter
        mid = 0.5*(lo + hi)
        fmid = f(mid)
        if abs(hi-lo) / max(abs(mid), 1.0) < rtol || fmid == 0.0
            return mid
        end
        if sign(fmid) == sign(flo)
            lo, flo = mid, fmid
        else
            hi, fhi = mid, fmid
        end
    end
    return 0.5*(lo + hi)
end

# -------- Concentration model (replaceable) --------
# Duffy-style form: c = A * (M/Mp)^B * (1+z)^C
# This is a reasonable default; tune A,B,C or swap to your preferred relation.
@inline function c200m_model(M200m, z; A=10.14, B=-0.081, C=-1.01, Mp=2e12/h)
    # Mp is often quoted in Msun/h; since your M is Msun, use Mp in Msun:
    return A * (M200m / Mp)^B * (1+z)^C
end

# -------- Core: M200m -> M200c --------
function m200m_to_m200c(M200m::Real, z::Real;
                        R_is_comoving::Bool=false,
                        c200m::Union{Nothing,Real}=nothing)

    ρm = rho_mean_z(z; R_is_comoving=R_is_comoving)
    ρc = rho_crit_z(z; R_is_comoving=R_is_comoving)

    c_in = c200m === nothing ? c200m_model(M200m, z) : Float64(c200m)

    # Input radius and scale radius
    R200m = R_from_M(M200m, 200.0, ρm)
    r_s   = R200m / c_in

    # Solve for x = R200c / R200m using:
    # f(c*x)/x^3 = (200*ρc)/(200*ρm) * f(c) = (ρc/ρm)*f(c)
    target = (ρc/ρm) * f_nfw(c_in)
    eqn(x) = f_nfw(c_in*x)/x^3 - target

    x = bisect_root(eqn, 1e-4, 1e2)

    M200c = M200m * f_nfw(c_in*x) / f_nfw(c_in)

    return M200c
end

# Arrays (broadcast)
function m200m_to_m200c_arrays(M200m::AbstractArray, z::AbstractArray;
                               R_is_comoving::Bool=false,
                               c200m::Union{Nothing,AbstractArray}=nothing)
    @assert size(M200m) == size(z)
    if c200m === nothing
        return m200m_to_m200c.(M200m, z; R_is_comoving=R_is_comoving)
    else
        @assert size(c200m) == size(M200m)
        return m200m_to_m200c.(M200m, z; R_is_comoving=R_is_comoving, c200m=c200m)
    end
end

end # module
