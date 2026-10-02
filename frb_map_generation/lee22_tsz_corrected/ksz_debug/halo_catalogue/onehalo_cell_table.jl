# Per-halo one-halo kSZ weight table for the halo_catalogue debug (2026-10-02): for each (log10 M200c [Msun], z) cell centre,
# a(ell; M, z) = 2pi int tau_B16,4R200c-sphere(theta) J0(ell theta) theta dtheta (flat-sky Hankel, exactly as
# ksz/poisson_mass_split.jl, same read-only B16 chord-mean cache), so C_ell^1h = (1/4pi) sum_cells sum_i (v_los,i/c)^2 a^2.
# Also theta_200c(M, z). Output: outputs/lee22_ksz_debug_20261002/halo_catalogue/onehalo_cell_table.h5
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
A2 = zeros(Float64, length(ells), length(logm), length(zc)); TH = zeros(length(logm), length(zc))
cells = [(i, j) for i in eachindex(logm), j in eachindex(zc)]
@threads :static for c in eachindex(cells)
    i, j = cells[c]; M = exp10(logm[i]); z = zc[j]
    θ200 = theta_r200c(ref, M, z); TH[i, j] = θ200
    θmax = X * θ200
    lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ); θ = exp.(lnθ)
    f = [Float64(ip(θ[k], M, z)) * chord_factor(θ[k], θmax, X) for k in 1:NTHETA]
    w = f .* θ .^ 2 .* dln
    for (il, ℓ) in enumerate(ells)
        a = 2pi * sum(w .* besselj0.(ℓ .* θ))
        A2[il, i, j] = a^2
    end
end
out = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/halo_catalogue/onehalo_cell_table.h5"
h5open(out, "w") do f
    f["ell"] = ells; f["logm200c_msun_centres"] = logm; f["z_centres"] = zc; f["a2"] = A2; f["theta200c_rad"] = TH
    attrs(f)["note"] = "B16 tau, 4 R200c sphere, chord-mean cache of lee22_ksz_20261002; a^2 per halo (v/c = 1); C_ell = sum v2 a^2 / 4pi"
end
println("done")
