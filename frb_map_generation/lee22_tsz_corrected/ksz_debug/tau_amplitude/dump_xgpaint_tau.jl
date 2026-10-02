# Dump XGPaint BattagliaTauProfile internals and tau(theta) values (full LOS, 4 R200c sphere direct, painter cache x chord
# factor) for a grid of (M, z) so they can be compared with an independent numpy implementation (tau_amplitude debug, 2026-10-02).
# Reads only; the painter cache is used through a private copy in the debug output directory.
using XGPaint, Unitful, UnitfulAstro, Printf
include("/home/cbllover/HalfDome/truncation_comparison/spherical_truncation_profiles.jl")
using .SphericalTruncation

const OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/tau_amplitude"
const SRC_CACHE = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/caches/tau_b16_sphere4p0_chordmean_interpolator.jld2"
const CACHE = joinpath(OUT, "tau_b16_sphere4p0_chordmean_interpolator_COPY.jld2")
isfile(CACHE) || cp(SRC_CACHE, CACHE)

model = BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
X = 4.0
cm = ChordMeanProfile(model, X)
itp = build_interpolator(cm; cache_file=CACHE, overwrite=false, verbose=false)

const MPC_M = ustrip(u"m", 1u"Mpc")
sigmaT = ustrip(u"m^2", XGPaint.constants.ThomsonCrossSection)
me = ustrip(u"kg", XGPaint.constants.ElectronMass); mp = ustrip(u"kg", XGPaint.constants.ProtonMass)
println("sigmaT_m2=", sigmaT, " me_kg=", me, " mp_kg=", mp, " Msun_kg=", ustrip(u"kg", XGPaint.M_sun), " f_b=", model.f_b)
println("cosmo=", model.cosmo)

open(joinpath(OUT, "xgpaint_tau_dump.csv"), "w") do io
    println(io, "M_msun,z,xc,alpha,beta,gamma,P0,rho_crit_kgm3,r200c_mpc,dA_mpc,theta200_rad,x,tau_full,tau_sphere_direct,tau_cache_x_chord,rho2d_kgm2,ne2d_m2")
    for M in (1e13, 1e14, 1e15), z in (0.1, 0.5, 1.0, 2.0), x in (0.05, 0.2, 0.5, 1.0, 2.0, 3.5)
        Mu = M * XGPaint.M_sun
        par = XGPaint.get_params(model, Mu, z)
        rc = ustrip(u"kg/m^3", XGPaint.ρ_crit(model, z))
        r200 = ustrip(u"Mpc", XGPaint.R_Δ(model, Mu, z, 200))
        dA = ustrip(u"Mpc", XGPaint.angular_diameter_dist(model.cosmo, z))
        th200 = theta_r200c(model, M, z)
        θ = x * th200
        tfull = Float64(model(θ, M, z))
        tsph = spherical_value_direct(cm, θ, M, z)
        tcache = itp(θ, M, z) * chord_factor(θ, X * th200, X)
        r2 = ustrip(u"kg/m^2", XGPaint.rho_2d(model, θ, Mu, z))
        n2 = ustrip(u"m^-2", XGPaint.ne2d(model, θ, Mu, z))
        @printf(io, "%.6e,%.3f,%.8g,%.8g,%.8g,%.8g,%.8g,%.10e,%.10e,%.10e,%.10e,%.3f,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                M, z, par.xc, par.α, par.β, par.γ, par.P₀, rc, r200, dA, th200, x, tfull, tsph, tcache, r2, n2)
    end
end
println("wrote ", joinpath(OUT, "xgpaint_tau_dump.csv"))
