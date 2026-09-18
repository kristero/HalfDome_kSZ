# One-halo (Poisson) auto and y x DM cross terms of the painted maps from the (log M, z) histogram:
# C_ell^{AB,1h} = (1/4pi) sum_i a_ell,i b_ell,i, for y (Battaglia12), DM Battaglia16 and DM Lee22 no-c,
# each projected and spherical (X = 4). Flat-sky Hankel transforms, valid for ell >~ 200.
using XGPaint, HDF5, Interpolations
using Base.Threads
hist_file, out_file = ARGS[1], ARGS[2]
empty!(ARGS)   # the FRB generator parses ARGS when included
include(joinpath(@__DIR__, "dm_profiles_common.jl"))
const besselj0 = XGPaint.SpecialFunctions.besselj0
cache_dir = joinpath(@__DIR__, "caches")
tsz = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
load(name, model) = build_interpolator(model; cache_file=joinpath(cache_dir, "$(name)_interpolator.jld2"), overwrite=false, verbose=false)
dmm = dm_models()
names = ("y_projected", "y_sphere", "dm_b16_projected", "dm_b16_sphere", "dm_lee22noc_projected", "dm_lee22noc_sphere")
itps = (load("tsz_projected", tsz), load("tsz_sphere4p0_chordmean", ChordMeanProfile(tsz, SPHERE_R200C)),
        load(DM_CACHE_NAMES[1], dmm[1]), load(DM_CACHE_NAMES[2], dmm[2]), load(DM_CACHE_NAMES[3], dmm[3]), load(DM_CACHE_NAMES[4], dmm[4]))
is_sphere = (false, true, false, true, false, true)
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in itps)
logm_edges, z_edges, count = h5open(hist_file, "r") do f; read(f["logm_edges"]), read(f["z_edges"]), read(f["count"]); end
ells = exp10.(range(1.0, log10(8192.0); length=90)); NTHETA = 4000
cells = [(im, iz) for im in axes(count, 1), iz in axes(count, 2) if count[im, iz] > 0]
println("occupied cells: $(length(cells)); halos: $(sum(count)); threads $(nthreads())")
NP = length(names)
acc = zeros(Float64, length(ells), NP, NP, Threads.maxthreadid())
@threads :static for ci in eachindex(cells)
    im, iz = cells[ci]; tid = threadid()
    M = exp10(0.5 * (logm_edges[im] + logm_edges[im + 1])); z = 0.5 * (z_edges[iz] + z_edges[iz + 1])
    θmax = SPHERE_R200C * theta_r200c(tsz, M, z)
    lnθ = range(log(max(theta_min, 1.0e-9)), log(θmax); length=NTHETA); dln = step(lnθ); θ = exp.(lnθ)
    f = [Float64(itps[k](θ[j], M, z)) * (is_sphere[k] ? chord_factor(θ[j], θmax, SPHERE_R200C) : 1.0) for j in 1:NTHETA, k in 1:NP]
    w = Float64(count[im, iz]); a = zeros(NP)
    for (il, ℓ) in enumerate(ells)
        j0 = besselj0.(ℓ .* θ) .* θ .^ 2 .* dln
        for k in 1:NP; a[k] = 2pi * sum(@view(f[:, k]) .* j0); end
        for k in 1:NP, l in 1:NP; acc[il, k, l, tid] += w * a[k] * a[l] / (4pi); end
    end
end
cl = dropdims(sum(acc; dims=4); dims=4)
h5open(out_file, "w") do f
    f["ell"] = collect(ells)
    for k in 1:NP, l in k:NP; f["cl_1h_$(names[k])_x_$(names[l])"] = cl[:, k, l]; end
    attrs(f)["note"] = "Poisson terms from the (logM,z) histogram; flat-sky Hankel; y dimensionless, DM in pc cm^-3"
end
il = argmin(abs.(ells .- 3000)); dl(k, l) = ells[il] * (ells[il] + 1) * cl[il, k, l] / (2pi)
println("ell~$(round(ells[il])): D_ell y x DM_b16 proj=$(dl(1,3)) sph=$(dl(2,4));  y x DM_lee22 proj=$(dl(1,5)) sph=$(dl(2,6));  DM_b16 auto proj=$(dl(3,3)) DM_lee22 auto proj=$(dl(5,5))")
println("saved $out_file")
