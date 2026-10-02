# B16 catalogue one-halo kSZ power of the HalfDome painting with (a) the HalfDome catalogue velocities (sum (v_los/c)^2 per
# (M200c, z) cell, as poisson_mass_split.jl) and (b) the same halos given WebSky's <v_los^2> at matched M200m and z
# (weights x vel2_ratio_ws_over_hd.h5). Same chord-mean cache (copied), 4 R200c sphere and flat-sky Hankel transform as
# ksz/poisson_mass_split.jl. Also (c) the one-halo term restricted to the Stein+2020 halo-map selection M200c > 1e13 Msun
# (all z), for reference. 2026-10-02, ksz_debug/velocities.
using XGPaint, HDF5, Interpolations, Unitful, UnitfulAstro
using Base.Threads
include("/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz/ksz_models.jl")
const besselj0 = XGPaint.SpecialFunctions.besselj0
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/velocities"
SRC = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002"
cache = joinpath(OUT, "tau_b16_sphere4p0_chordmean_interpolator_COPY.jld2")
isfile(cache) || cp(joinpath(SRC, "caches", "tau_b16_sphere4p0_chordmean_interpolator.jld2"), cache)
model = MODELS[1][2]; @assert MODELS[1][1] == "b16"
ip = build_interpolator(ChordMeanProfile(model, X); cache_file=cache, overwrite=false, verbose=false)
theta_min = exp(first(first(ip.itp.ranges)))
logm_edges, z_edges, count, sum_v2 = h5open(joinpath(SRC, "maps", "halo_mass_redshift_histogram.h5"), "r") do f
    read(f["logm_edges"]), read(f["z_edges"]), read(f["count"]), read(f["sum_vlos_over_c_squared"])
end
ratio = h5open(joinpath(OUT, "vel2_ratio_ws_over_hd.h5"), "r") do f; read(f["ratio_v2"]); end
@assert size(ratio) == size(count)
ells = vcat([300.0, 1000.0, 2000.0, 3000.0, 5000.0], exp10.(range(log10(100.0), log10(8000.0); length=30)))
NTHETA = 3000
cells = [(im, iz) for im in axes(count, 1), iz in axes(count, 2) if count[im, iz] > 0]
println("cells $(length(cells)) halos $(sum(count)) threads $(nthreads())")
cl = zeros(Float64, length(ells), 4, nthreads() + 8)   # 1 HD vel, 2 WS vel, 3 HD vel M200c>1e13, 4 WS vel M200c>1e13
@threads :static for ci in eachindex(cells)
    im, iz = cells[ci]; tid = threadid()
    lm = 0.5 * (logm_edges[im] + logm_edges[im + 1]); M = exp10(lm); z = 0.5 * (z_edges[iz] + z_edges[iz + 1])
    θmax = X * theta_r200c(model, M, z)
    lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ); θ = exp.(lnθ)
    f = [Float64(ip(θ[j], M, z)) * chord_factor(θ[j], θmax, X) for j in 1:NTHETA]
    for (il, ℓ) in enumerate(ells)
        a = 2pi * sum(f .* besselj0.(ℓ .* θ) .* θ .^ 2 .* dln)
        w1 = sum_v2[im, iz] * a^2 / (4pi); w2 = w1 * ratio[im, iz]
        cl[il, 1, tid] += w1; cl[il, 2, tid] += w2
        if lm >= 13.0
            cl[il, 3, tid] += w1; cl[il, 4, tid] += w2
        end
    end
end
clb = dropdims(sum(cl; dims=3); dims=3)
T2 = 2.7255e6^2
h5open(joinpath(OUT, "one_halo_velocity_reweight_b16.h5"), "w") do f
    f["ell"] = ells; f["cl_1h_hdvel"] = clb[:, 1]; f["cl_1h_wsvel"] = clb[:, 2]
    f["cl_1h_hdvel_m200c_gt_1e13"] = clb[:, 3]; f["cl_1h_wsvel_m200c_gt_1e13"] = clb[:, 4]
end
for il in 1:5
    ℓ = ells[il]; D(c) = ℓ * (ℓ + 1) * c / (2pi) * T2
    println("ell $(Int(ℓ)): D_1h HDvel=$(round(D(clb[il,1]); digits=4)) WSvel=$(round(D(clb[il,2]); digits=4)) ratio WS/HD=$(round(clb[il,2]/clb[il,1]; digits=4)) | ",
            "M200c>1e13: HDvel=$(round(D(clb[il,3]); digits=4)) WSvel=$(round(D(clb[il,4]); digits=4))")
end
