# Catalogue-based Poisson (one-halo) terms of the painted maps: C_ell^{1h} = (1/4pi) sum_i w_i |a_ell,i|^2
# with w_i = 1 (Compton-y) or (v_los,i/c)^2 (kSZ), using the same interpolators and truncations as the
# painter and the (log M, z) histogram written by the painter. Flat-sky Hankel transform per cell.
# Usage: julia -t N compute_catalogue_poisson_terms.jl <histogram.h5> <output.h5>
using XGPaint, HDF5, Interpolations
using Base.Threads
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation
const besselj0 = XGPaint.SpecialFunctions.besselj0

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603; const X = 4.0
hist_file, out_file = ARGS[1], ARGS[2]
cache_dir = joinpath(@__DIR__, "caches")
tsz = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
tau = BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
load(name, model) = build_interpolator(model; cache_file=joinpath(cache_dir, "$(name)_interpolator.jld2"), overwrite=false, verbose=false)
itps = (load("tsz_projected", tsz), load("tsz_sphere4p0_chordmean", ChordMeanProfile(tsz, X)),
        load("tau_projected", tau), load("tau_sphere4p0_chordmean", ChordMeanProfile(tau, X)))
is_sphere = (false, true, false, true)
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in itps)

logm_edges, z_edges, count, sum_v2 = h5open(hist_file, "r") do f
    read(f["logm_edges"]), read(f["z_edges"]), read(f["count"]), read(f["sum_vlos_over_c_squared"])
end
ells = exp10.(range(1.0, log10(8192.0); length=90))
NTHETA = 4000
cells = [(im, iz) for im in axes(count, 1), iz in axes(count, 2) if count[im, iz] > 0]
println("occupied cells: $(length(cells)) of $(length(count)); halos: $(sum(count)); threads $(nthreads())")
cl = zeros(Float64, length(ells), 4, Threads.maxthreadid())
@threads :static for ci in eachindex(cells)
    im, iz = cells[ci]; tid = threadid()
    M = exp10(0.5 * (logm_edges[im] + logm_edges[im + 1])); z = 0.5 * (z_edges[iz] + z_edges[iz + 1])
    θ200 = theta_r200c(tsz, M, z); θmax = X * θ200
    lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ)
    θ = exp.(lnθ)
    f = [Float64(itps[k](θ[j], M, z)) * (is_sphere[k] ? chord_factor(θ[j], θmax, X) : 1.0) for j in 1:NTHETA, k in 1:4]
    w = (Float64(count[im, iz]), Float64(count[im, iz]), sum_v2[im, iz], sum_v2[im, iz])
    for (il, ℓ) in enumerate(ells)
        j0 = besselj0.(ℓ .* θ) .* θ .^ 2 .* dln
        for k in 1:4
            a = 2pi * sum(@view(f[:, k]) .* j0)
            cl[il, k, tid] += w[k] * a^2 / (4pi)
        end
    end
end
cl_tot = dropdims(sum(cl; dims=3); dims=3)
h5open(out_file, "w") do f
    f["ell"] = collect(ells)
    f["cl_1h_y_projected"] = cl_tot[:, 1]; f["cl_1h_y_sphere"] = cl_tot[:, 2]
    f["cl_1h_ksz_projected"] = cl_tot[:, 3]; f["cl_1h_ksz_sphere"] = cl_tot[:, 4]
    attrs(f)["note"] = "Poisson term of the painted maps from the (logM,z) histogram; flat-sky Hankel; kSZ weights sum (v_los/c)^2 per cell; units: C_ell of Compton-y and of Delta T/T"
end
dl(il, k) = ells[il] * (ells[il] + 1) * cl_tot[il, k] / (2pi)
il = argmin(abs.(ells .- 3000))
println("at ell~$(round(ells[il])): 1e12 D_ell^yy proj=$(1e12*dl(il,1)) sphere=$(1e12*dl(il,2));  D_ell^kSZ [muK^2] proj=$(dl(il,3)*2.7255e6^2) sphere=$(dl(il,4)*2.7255e6^2)")
println("saved $out_file")
