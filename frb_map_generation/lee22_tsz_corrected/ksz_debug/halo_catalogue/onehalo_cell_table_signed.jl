# Signed-a variant of onehalo_cell_table.jl (+ a(ell=0), WebSky uniform-sphere electron count / D_A^2, chi) for the compensation estimate.
# Per-halo one-halo kSZ weight table for the halo_catalogue debug (2026-10-02): for each (log10 M200c [Msun], z) cell centre,
# a(ell; M, z) = 2pi int tau_B16,4R200c-sphere(theta) J0(ell theta) theta dtheta (flat-sky Hankel, exactly as
# ksz/poisson_mass_split.jl, same read-only B16 chord-mean cache), so C_ell^1h = (1/4pi) sum_cells sum_i (v_los,i/c)^2 a^2.
# Also theta_200c(M, z). Output: outputs/lee22_ksz_debug_20261002/halo_catalogue/onehalo_cell_table_signed.h5
using XGPaint, HDF5, Interpolations, Unitful, UnitfulAstro
using Base.Threads
include("/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz/ksz_models.jl")
const besselj0 = XGPaint.SpecialFunctions.besselj0
cache = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/caches/tau_b16_sphere4p0_chordmean_interpolator.jld2"
isfile(cache) || error("missing cache")
ip = build_interpolator(ChordMeanProfile(MODELS[1][2], X); cache_file=cache, overwrite=false, verbose=false)
ref = MODELS[1][2]
theta_min = exp(first(first(ip.itp.ranges)))
logm = collect(12.0125:0.025:15.6875); zc = collect(0.0125:0.025:4.6875)
ells = vcat(exp10.(range(log10(30.0), log10(8192.0); length=80)), [300.0, 1000.0, 2000.0, 3000.0, 5000.0])
NTHETA = 3000
const sigmaT_m2 = Float64(ustrip(u"m^2", XGPaint.constants.ThomsonCrossSection)); const MSUN_KG = 1.98847e30; const MP_KG = 1.67262192e-27
A1 = zeros(Float64, length(ells), length(logm), length(zc)); TH = zeros(length(logm), length(zc)); A0 = zeros(length(logm), length(zc)); S0 = zeros(length(logm), length(zc)); CHI = zeros(length(zc))
cells = [(i, j) for i in eachindex(logm), j in eachindex(zc)]
@threads :static for c in eachindex(cells)
    i, j = cells[c]; M = exp10(logm[i]); z = zc[j]
    θ200 = theta_r200c(ref, M, z); TH[i, j] = θ200
    θmax = X * θ200
    lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ); θ = exp.(lnθ)
    f = [Float64(ip(θ[k], M, z)) * chord_factor(θ[k], θmax, X) for k in 1:NTHETA]
    w = f .* θ .^ 2 .* dln; A0[i, j] = 2pi * sum(w)
    dA = ustrip(u"m", XGPaint.angular_diameter_dist(ref.cosmo, z)); CHI[j] = ustrip(u"Mpc", XGPaint.angular_diameter_dist(ref.cosmo, z)) * (1 + z)
    S0[i, j] = sigmaT_m2 * ref.f_b * (M * MSUN_KG) * 0.88 / MP_KG / dA^2     # WebSky uniform sphere, per unit M (no 0.9)
    for (il, ℓ) in enumerate(ells)
        a = 2pi * sum(w .* besselj0.(ℓ .* θ))
        A1[il, i, j] = a
    end
end
out = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/halo_catalogue/onehalo_cell_table_signed.h5"
h5open(out, "w") do f
    f["ell"] = ells; f["logm200c_msun_centres"] = logm; f["z_centres"] = zc; f["a"] = A1; f["a0"] = A0; f["s0_per_M"] = S0; f["chi_mpc"] = CHI; f["theta200c_rad"] = TH
    attrs(f)["note"] = "B16 tau, 4 R200c sphere, chord-mean cache of lee22_ksz_20261002; a^2 per halo (v/c = 1); C_ell = sum v2 a^2 / 4pi"
end
println("done")
