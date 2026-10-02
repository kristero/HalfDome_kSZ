# Catalogue one-halo (Poisson) kSZ power of the painted maps, split by halo mass (2026-10-02):
#   C_ell^{1h}(bin) = (1/4pi) sum_{cells in bin} sum_i (v_los,i/c)^2 |tau_ell(M, z)|^2,
# with the same chord-mean caches and 4 R200c spherical truncation as paint_ksz_variants.jl and the (log10 M, z) histogram it
# writes. Flat-sky Hankel transform per (M, z) cell (as truncation_comparison/compute_catalogue_poisson_terms.jl). The one-halo
# term is additive in mass; the velocity-correlation (two-halo) part of the map is not included.
# Usage: julia -t 20 poisson_mass_split.jl [--output-dir=...]
using XGPaint, HDF5, Interpolations, Unitful, UnitfulAstro
using Base.Threads
include(joinpath(@__DIR__, "ksz_models.jl"))
const besselj0 = XGPaint.SpecialFunctions.besselj0
getarg(key, default) = (for a in ARGS; startswith(a, "--$key=") && return split(a, "="; limit=2)[2]; end; default)
output_dir = getarg("output-dir", joinpath(FMG, "outputs", "lee22_ksz_20261002"))
names = Tuple(m[1] for m in MODELS)
itps = [build_interpolator(ChordMeanProfile(m[2], X); cache_file=joinpath(output_dir, "caches", "tau_$(m[1])_sphere4p0_chordmean_interpolator.jld2"),
        overwrite=false, verbose=false) for m in MODELS]
ref = MODELS[1][2]
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in itps)
logm_edges, z_edges, count, sum_v2 = h5open(joinpath(output_dir, "maps", "halo_mass_redshift_histogram.h5"), "r") do f
    read(f["logm_edges"]), read(f["z_edges"]), read(f["count"]), read(f["sum_vlos_over_c_squared"])
end
const MASS_BINS = [12.0, 13.0, 13.5, 14.0, 14.5, 15.0, 15.7]       # log10 M200c [Msun]
ells = exp10.(range(log10(30.0), log10(8192.0); length=80))
NTHETA = 3000
cells = [(im, iz) for im in axes(count, 1), iz in axes(count, 2) if count[im, iz] > 0]
println("occupied cells $(length(cells)); halos $(sum(count)); threads $(nthreads())")
nb = length(MASS_BINS) - 1
cl = zeros(Float64, length(ells), nb, length(names), Threads.nthreads() + 8)
@threads :static for ci in eachindex(cells)
    im, iz = cells[ci]; tid = threadid()
    lm = 0.5 * (logm_edges[im] + logm_edges[im + 1]); M = exp10(lm); z = 0.5 * (z_edges[iz] + z_edges[iz + 1])
    b = searchsortedlast(MASS_BINS, lm); (1 <= b <= nb) || continue
    θmax = X * theta_r200c(ref, M, z)
    lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ); θ = exp.(lnθ)
    cf = [chord_factor(t, θmax, X) for t in θ]
    for (k, ip) in enumerate(itps)
        f = [Float64(ip(θ[j], M, z)) * cf[j] for j in 1:NTHETA]
        for (il, ℓ) in enumerate(ells)
            a = 2pi * sum(f .* besselj0.(ℓ .* θ) .* θ .^ 2 .* dln)
            cl[il, b, k, tid] += sum_v2[im, iz] * a^2 / (4pi)
        end
    end
end
clb = dropdims(sum(cl; dims=4); dims=4)
h5open(joinpath(output_dir, "ksz_poisson_mass_split.h5"), "w") do f
    f["ell"] = collect(ells); f["log10_mass_bin_edges_msun"] = MASS_BINS
    for (k, n) in enumerate(names)
        f["cl_1h_$(n)"] = clb[:, :, k]          # (ell, mass bin), C_ell of Delta T / T
    end
    attrs(f)["note"] = "catalogue one-halo kSZ term per mass bin; flat-sky Hankel; weights sum (v_los/c)^2; 4 R200c sphere"
end
T2 = 2.7255e6^2
for (k, n) in enumerate(names), L in (300.0, 1000.0, 3000.0)
    il = argmin(abs.(ells .- L)); tot = sum(clb[il, :, k])
    println("$n ell~$(round(Int, ells[il])): D_ell 1h = $(round(ells[il] * (ells[il] + 1) * tot / (2pi) * T2; digits=4)) muK^2; fractions ",
            join([string(round(clb[il, b, k] / tot; digits=3)) for b in 1:nb], " "))
end
