using XGPaint
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation
const ST = SphericalTruncation

tsz = Battaglia16ThermalSZProfile(Omega_c=0.2603, Omega_b=0.0486, h=0.6774)
tau = BattagliaTauProfile(Omega_c=0.2603, Omega_b=0.0486, h=0.6774)
X = 4.0
for (name, inner) in (("tSZ", tsz), ("tau", tau))
    m = ChordMeanProfile(inner, X)
    println("== ", name)
    for (M, z) in ((7e12, 0.1), (1e13, 0.5), (1e14, 0.5), (1e15, 0.3), (3e15, 1.5))
        θ200 = theta_r200c(inner, M, z); θmax = X * θ200
        prev = Inf
        for x in (1e-3, 0.1, 0.5, 1.0, 2.0, 3.0, 3.9, 3.999)
            θ = x * θ200
            proj = Float64(inner(θ, M, z))
            direct = spherical_value_direct(m, θ, M, z)
            rebuilt = m(θ, M, z) * chord_factor(θ, θmax, X)
            isapprox(direct, rebuilt; rtol=1e-8) || error("rebuild mismatch $direct vs $rebuilt at M=$M z=$z x=$x")
            direct <= proj * (1 + 1e-9) || error("sphere > projected at M=$M z=$z x=$x")
            direct < prev || error("not decreasing at x=$x")
            prev = direct
            x in (1e-3, 1.0, 3.0, 3.9) && println("  M=$M z=$z x=$x  proj=$(round(proj, sigdigits=5))  sphere/proj=$(round(direct/proj, digits=4))")
        end
        # continuity of the cached g at the edge and positivity outside
        g_in = m(0.9999 * θmax, M, z); g_edge = m(θmax, M, z); g_out = m(1.001 * θmax, M, z)
        isapprox(g_in, g_edge; rtol=2e-2) || error("g not continuous at edge: $g_in vs $g_edge")
        g_out > 0 || error("g not positive outside")
        chord_factor(θmax * 1.0000001, θmax, X) == 0.0 || error("chord factor must vanish outside")
    end
end
# grid dispatch test on a tiny grid
m = ChordMeanProfile(tsz, X)
lt, zs, lM, A = XGPaint.profile_grid(m; N_z=4, N_logM=4, N_logtheta=32, logtheta_min=-12.0, logtheta_max=-3.0)
all(isfinite, A) && all(A .> 0) || error("grid has nonpositive/nonfinite values")
θ = exp(lt[10]); Mg = 10^lM[2]; zg = zs[3]
isapprox(A[10, 3, 2], m(θ, Mg, zg); rtol=1e-10) || error("grid value differs from direct evaluation")
println("grid dispatch OK: size ", size(A))
# XGPaint gas density for comparison with class_sz's B16 gas profile (M = 1e14 Msun/h physical mass = 1e14/h)
Mphys = 1e14 / 0.6774; z = 0.5
par = XGPaint.get_params(tau, Mphys * XGPaint.M_sun, z)
println("XGPaint tau params at M=1e14/h Msun, z=0.5: ", par)
for x in (0.1, 0.5, 1.0, 2.0, 3.0, 4.0)
    rho_over_rhocrit = par.P₀ * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ) * tau.f_b
    println("  x=$x rho_gas/rho_crit = ", rho_over_rhocrit, "   (without f_b: ", rho_over_rhocrit / tau.f_b, ")")
end
println("PASS")
