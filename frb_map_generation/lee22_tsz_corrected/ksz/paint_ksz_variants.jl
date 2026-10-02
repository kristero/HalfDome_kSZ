# Halo kSZ (Delta T / T) maps of the full HalfDome lightcone for three electron-density models, painted in one catalogue pass
# (2026-10-02):
#   b16        Battaglia16 AGN density (XGPaint BattagliaTauProfile, ne2d convention)
#   lee22pipe  Lee22 no-c density through the XGPaint pipeline (ne2d reading of the printed fit: literal x 0.6016)
#   lee22corr  Lee22 no-c density corrected: literal x [h_TNG (1+z)]^3 X_H^2 (derived)
# Map = -tau * v_los/c with tau = sigma_T * integral n_e dl over the gas inside the 4 R200c sphere (chord-limited LOS, as
# truncation_comparison/ and the 2026-09-19 WebSky comparison); catalogue velocities projected on the line of sight. HalfDome
# cosmology (h = 0.68, Omega_b = 0.049, Omega_c = 0.261); catalogue M200c [Msun/h] / 0.68. Lee22 shape parameters frozen above
# 10^14.8 h^-1 Msun as in the production FRB/tSZ products. Also writes the (log10 M, z) histogram with summed (v_los/c)^2 for the
# catalogue one-halo (Poisson) term (poisson_mass_split.jl).
# Usage: julia -t 20 paint_ksz_variants.jl [--catalog=...] [--nside=4096] [--max-halos=0] [--chunk-size=1000000]
#        [--output-dir=...] [--self-test-only]
using Dates, HDF5, Healpix, Interpolations, XGPaint, Unitful, UnitfulAstro
using Base.Threads
include(joinpath(@__DIR__, "ksz_models.jl"))

getarg(key, default) = (for a in ARGS; startswith(a, "--$key=") && return split(a, "="; limit=2)[2]; end; default)
catalog = getarg("catalog", "/home/cbllover/HalfDome/lightcone_100.hdf5")
nside = parse(Int, getarg("nside", "4096"))
max_halos = parse(Int, getarg("max-halos", "0"))
chunk_size = parse(Int, getarg("chunk-size", "1000000"))
output_dir = getarg("output-dir", joinpath(FMG, "outputs", "lee22_ksz_20261002"))
self_test_only = "--self-test-only" in ARGS
cache_dir = joinpath(output_dir, "caches"); mkpath(cache_dir); mkpath(joinpath(output_dir, "maps"))
ref = MODELS[1][2]

# ---------------------------------------------------------------- self-tests (before any painting)
const M2_TO_PC = ustrip(u"pc*cm^-3", 1u"m^-2")
sigmaT = Float64(ustrip(u"m^2", XGPaint.constants.ThomsonCrossSection))
worst_tau_dm = 0.0; worst_ratio = 0.0
for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0), (3e13, 2.0)), f in (0.3, 1.0, 2.5)
    θ = f * theta_r200c(ref, M, z)
    for name in ("lee22pipe", "lee22corr")
        tau_m = Dict(MODELS)[name]
        τ = Float64(tau_m(θ, M, z)); dm = Float64(DM_REF[name](θ, M, z))       # dm in pc cm^-3 includes 1/(1+z)
        global worst_tau_dm = max(worst_tau_dm, abs(τ / (sigmaT * dm * (1 + z) / M2_TO_PC) - 1))
    end
    r = Float64(Dict(MODELS)["lee22corr"](θ, M, z)) / Float64(Dict(MODELS)["lee22pipe"](θ, M, z))
    global worst_ratio = max(worst_ratio, abs(r / ((0.6774 * (1 + z))^3 * 0.76^2 / 0.6016316602183194) - 1))
end
println("self-test: Lee22 tau = sigma_T (1+z) DM: max rel dev $(worst_tau_dm); corrected/pipeline = [h(1+z)]^3 X_H^2/0.6016: max rel dev $(worst_ratio)")
(worst_tau_dm < 1e-8 && worst_ratio < 1e-8) || error("Lee22 tau self-test failed")

itps = Dict{String,Any}()
for (name, model) in MODELS
    cm = ChordMeanProfile(model, X)
    itps[name] = build_interpolator(cm; cache_file=joinpath(cache_dir, "tau_$(name)_sphere4p0_chordmean_interpolator.jld2"),
                                    overwrite=false, verbose=false)
    dev = maximum(abs(itps[name](f * theta_r200c(ref, M, z), M, z) / cm(f * theta_r200c(ref, M, z), M, z) - 1)
                  for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0)), f in (0.1, 0.7, 2.0))
    println("cache $(name): max |cache/direct - 1| = $(dev)")
    dev < 2e-3 || error("cache check failed for $(name)")
end
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in values(itps))
logm_lo = maximum(first(ip.itp.ranges[3]) for ip in values(itps)); logm_hi = minimum(last(ip.itp.ranges[3]) for ip in values(itps))
z_hi = minimum(last(ip.itp.ranges[2]) for ip in values(itps))
println("theta_min = $(theta_min) rad; cache log10 M $(logm_lo)-$(logm_hi); z max $(z_hi)")
self_test_only && (println("self-test only: done"); exit(0))

# ---------------------------------------------------------------- painting
resolution = Healpix.Resolution(nside)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
names = [m[1] for m in MODELS]
maps = [HealpixMap{Float64,RingOrder}(nside) for _ in names]
for m in maps; fill!(m.pixels, 0.0); end
pix = [m.pixels for m in maps]
itp_list = [itps[n] for n in names]

function paint_batch!(masses, redshifts, positions, vlos_over_c)
    updates = zeros(Int, Threads.nthreads() + 8)
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        x, y, zc = positions[1, i], positions[2, i], positions[3, i]
        d = sqrt(x^2 + y^2 + zc^2); ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz); θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        M = masses[i]; z = redshifts[i]; v = vlos_over_c[i]
        θmax = min(X * theta_r200c(ref, M, z), pi)
        rs, re = XGPaint.get_relevant_rings(workspace.res, θc, θmax)
        n = 0
        for ring in rs:re
            r1, r2 = XGPaint.get_ring_disc_ranges(workspace, ring, θc, φc, θmax)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((r1, r2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    θ = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0))
                    θ < θmax || continue
                    cf = chord_factor(θ, θmax, X); θe = max(θ, theta_min)
                    for k in eachindex(pix)
                        pix[k][gp] -= v * itp_list[k](θe, M, z) * cf     # Delta T / T = - tau v_los / c
                    end
                    n += 1
                end
            end
        end
        updates[tid] += n
    end
    return sum(updates)
end

const LOGM_EDGES = collect(12.0:0.025:15.7); const Z_EDGES = collect(0.0:0.025:4.0)
hist_n = zeros(Int, length(LOGM_EDGES) - 1, length(Z_EDGES) - 1); hist_v2 = zeros(Float64, size(hist_n))
t0 = time(); total_updates = 0; nsel = 0; vrms_acc = 0.0
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mass_ds = f["halo_mass_m200c"]; vel_ds = f["Velocity"]; z_ds = f["redshift"]
    N = size(pos_ds, 2); last_row = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last_row
        idx = start:min(start + chunk_size - 1, last_row)
        pos = Float64.(pos_ds[:, idx]); vel = Float64.(vel_ds[:, idx])
        masses = Float64.(mass_ds[idx]) ./ H; zs = Float64.(z_ds[idx])
        keep = isfinite.(masses) .& (masses .> 0) .& isfinite.(zs) .& (zs .> 0) .& vec(all(isfinite, pos; dims=1)) .& vec(all(isfinite, vel; dims=1))
        pos = pos[:, keep]; vel = vel[:, keep]; masses = masses[keep]; zs = zs[keep]
        r = sqrt.(vec(sum(pos .^ 2; dims=1))); vlos_c = vec(sum(pos .* vel; dims=1)) ./ r ./ C_KMS
        all(logm_lo .<= log10.(masses) .<= logm_hi) || error("mass outside cache range in chunk $start")
        all(zs .<= z_hi) || error("redshift outside cache range in chunk $start")
        for j in eachindex(masses)
            im = searchsortedlast(LOGM_EDGES, log10(masses[j])); iz = searchsortedlast(Z_EDGES, zs[j])
            hist_n[im, iz] += 1; hist_v2[im, iz] += vlos_c[j]^2
        end
        global total_updates += paint_batch!(masses, zs, pos, vlos_c)
        global nsel += length(masses); global vrms_acc += sum(abs2, vlos_c .* C_KMS)
        println("  rows $(first(idx)):$(last(idx))  halos=$nsel  pixel updates=$total_updates  elapsed=$(round(time() - t0; digits=0)) s"); flush(stdout)
    end
end
for (k, name) in enumerate(names)
    all(isfinite, maps[k].pixels) || error("non-finite pixels in $name")
    path = joinpath(output_dir, "maps", "halfdome_ksz_$(name)_sphere4p0_nside$(nside).fits")
    Healpix.saveToFITS(maps[k], "!" * path, typechar="D")
    println("saved $path  rms=$(sqrt(sum(abs2, maps[k].pixels) / length(maps[k].pixels)))")
end
h5open(joinpath(output_dir, "maps", "halo_mass_redshift_histogram.h5"), "w") do f
    f["logm_edges"] = LOGM_EDGES; f["z_edges"] = Z_EDGES; f["count"] = hist_n; f["sum_vlos_over_c_squared"] = hist_v2
    attrs(f)["mass"] = "log10 M200c [Msun], catalogue halo_mass_m200c / 0.68"
end
open(joinpath(output_dir, "maps", "provenance.txt"), "w") do io
    for (k, v) in sort(collect(Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "catalog" => catalog, "nside" => nside,
        "halos_painted" => nsel, "pixel_updates_per_map" => total_updates, "elapsed_seconds" => round(time() - t0; digits=1),
        "julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(), "h" => H, "Omega_b" => OMEGA_B, "Omega_c" => OMEGA_C,
        "truncation" => "gas inside the $(X) R200c sphere (chord-mean cache x 2X sqrt(1-(theta/theta_max)^2))",
        "models" => "b16: BattagliaTauProfile; lee22pipe: Lee22 no-c, normalization=xgpaint_ne2d (x0.6016), physical; lee22corr: literal x [0.6774(1+z)]^3 0.76^2",
        "lee22_shape_clip_msun" => SHAPE_CLIP, "map" => "Delta T / T = -tau v_los/c", "self_test_tau_dm" => worst_tau_dm,
        "self_test_corr_over_pipe" => worst_ratio, "vlos_rms_kms" => sqrt(vrms_acc / max(nsel, 1)), "theta_min_rad" => theta_min,
    )); by=first)
        println(io, "$k=$v")
    end
end
println("done in $(round(time() - t0; digits=1)) s")
