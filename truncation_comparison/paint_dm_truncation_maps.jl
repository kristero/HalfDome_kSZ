# Paint HalfDome full-sky halo-DM maps [pc cm^-3] for Battaglia16 and Lee22 no-c (XGPaint-native),
# each with the projected (previous) and the spherical 4 R200c truncation, in one pass over the
# catalogue, with the same geometry as paint_truncation_comparison_maps.jl.
using Dates, HDF5, Healpix, Interpolations, XGPaint
using Base.Threads
include(joinpath(@__DIR__, "dm_profiles_common.jl"))

function getarg(key, default); for a in ARGS; startswith(a, "--$key=") && return split(a, "="; limit=2)[2]; end; return default; end
catalog = getarg("catalog", "/home/cbllover/HalfDome/lightcone_100.hdf5")
nside = parse(Int, getarg("nside", "4096")); max_halos = parse(Int, getarg("max-halos", "0"))
chunk_size = parse(Int, getarg("chunk-size", "1000000")); output_dir = getarg("output-dir", joinpath(@__DIR__, "maps")); tag = getarg("tag", "")
cache_dir = joinpath(@__DIR__, "caches"); mkpath(output_dir)
const DM_SANITY_MAX = 1.0e6

models = dm_models()
itps = Tuple(build_interpolator(m; cache_file=joinpath(cache_dir, "$(n)_interpolator.jld2"), overwrite=false, verbose=false) for (n, m) in zip(DM_CACHE_NAMES, models))
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in itps)
logm_lo = minimum(first(ip.itp.ranges[3]) for ip in itps); logm_hi = maximum(last(ip.itp.ranges[3]) for ip in itps); z_hi = minimum(last(ip.itp.ranges[2]) for ip in itps)
geom = dm_inner_profiles().b16
println("theta_min=$(theta_min); cache log10 M range $(logm_lo)-$(logm_hi); z max $(z_hi)")
for (M, z) in ((1e13, 0.5), (1e14, 0.5), (1e15, 0.3))
    θ = 0.7 * theta_r200c(geom, M, z)
    println("  spot M=$M z=$z: ", join(["$(n) cache/direct=$(round(itps[k](θ, M, z) / models[k](θ, M, z), digits=7))" for (k, n) in enumerate(DM_CACHE_NAMES)], "  "))
end

resolution = Healpix.Resolution(nside); workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
maps = [HealpixMap{Float64,RingOrder}(nside) for _ in 1:4]; for m in maps; fill!(m.pixels, 0.0); end
pix = Tuple(m.pixels for m in maps)

function paint_batch!(masses, redshifts, positions)
    updates = zeros(Int, Threads.maxthreadid())
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        x, y, zc = Float64(positions[1, i]), Float64(positions[2, i]), Float64(positions[3, i]); d = sqrt(x^2 + y^2 + zc^2)
        ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz); θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        M = Float64(masses[i]); z = Float64(redshifts[i])
        θmax = min(SPHERE_R200C * theta_r200c(geom, M, z), pi)
        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, θc, θmax); n = 0
        for ring in ring_start:ring_stop
            r1, r2 = XGPaint.get_ring_disc_ranges(workspace, ring, θc, φc, θmax); first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((r1, r2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    θ = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0)); θ < θmax || continue
                    cf = chord_factor(θ, θmax, SPHERE_R200C); θe = max(θ, theta_min)
                    v1 = itps[1](θe, M, z); v2 = itps[2](θe, M, z) * cf; v3 = itps[3](θe, M, z); v4 = itps[4](θe, M, z) * cf
                    (0.0 <= v1 <= DM_SANITY_MAX && 0.0 <= v3 <= DM_SANITY_MAX) || error("DM sanity failure at M=$M z=$z theta=$θ: $v1 $v3")
                    pix[1][gp] += v1; pix[2][gp] += v2; pix[3][gp] += v3; pix[4][gp] += v4; n += 1
                end
            end
        end
        updates[tid] += n
    end
    return sum(updates)
end

t0 = time(); total_updates = 0; nsel = 0
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mass_ds = f["halo_mass_m200c"]; z_ds = f["redshift"]
    N = size(pos_ds, 2); last = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last
        stop = min(start + chunk_size - 1, last); idx = start:stop
        pos = Float64.(pos_ds[:, idx]); masses = Float64.(mass_ds[idx]) ./ H; zs = Float64.(z_ds[idx])
        keep = isfinite.(masses) .& (masses .> 0) .& isfinite.(zs) .& (zs .> 0) .& vec(all(isfinite, pos; dims=1))
        pos = pos[:, keep]; masses = masses[keep]; zs = zs[keep]
        all(logm_lo .<= log10.(masses) .<= logm_hi) || error("mass outside cache range"); all(zs .<= z_hi) || error("z outside cache range")
        global total_updates += paint_batch!(masses, zs, pos); global nsel += length(masses)
        println("  rows $start:$stop  halos=$nsel  pixel updates=$total_updates  elapsed=$(round(time() - t0; digits=0)) s"); flush(stdout)
    end
end
elapsed = time() - t0
names = ("dm_b16_projected", "dm_b16_sphere", "dm_lee22noc_projected", "dm_lee22noc_sphere")
for (name, m) in zip(names, maps)
    all(isfinite, m.pixels) || error("non-finite pixels in $name")
    path = joinpath(output_dir, "halfdome_$(name)_nside$(nside)_r200cx4$(tag).fits"); Healpix.saveToFITS(m, "!" * path, typechar="D")
    println("saved $path  mean=$(sum(m.pixels)/length(m.pixels)) pc/cm3  max=$(maximum(m.pixels))")
end
open(joinpath(output_dir, "provenance_dm$(tag).txt"), "w") do io
    for (k, v) in sort(collect(Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "catalog" => catalog, "nside" => nside,
        "halos_painted" => nsel, "pixel_updates_per_map" => total_updates, "elapsed_seconds" => round(elapsed; digits=1), "julia_threads" => Threads.nthreads(),
        "h" => H, "Omega_b" => OMEGA_B, "Omega_c" => OMEGA_C, "mass_definition" => "M200c, halo_mass_m200c / h -> physical Msun (h = 0.6774, matching the y maps; FRB products used 0.68)",
        "aperture" => "theta_max = $(SPHERE_R200C) R200c for both truncations", "units" => "observer-frame DM [pc cm^-3]",
        "dm_b16" => "XGPaint HaloDMProfile(BattagliaTauProfile) = Battaglia16 AGN density, ne2d electrons, /(1+z)",
        "dm_lee22noc" => "Lee22 no-concentration fit (Table A2), XGPaint-native reading (P0 = 200 n0, ne2d electrons), M_cut n0 pivot, shape clip at 10^14.8/h Msun, via Lee2022XGPaintDMProfile",
        "projected_truncation" => "infinite LOS, disc cut only (previous code)", "spherical_truncation" => "chord-mean cache times 2 X sqrt(1-(theta/theta_max)^2)",
        "interpolator_caches" => join([joinpath(cache_dir, "$(n)_interpolator.jld2") for n in DM_CACHE_NAMES], ";"), "theta_min_rad" => theta_min,
    )); by=first); println(io, "$k=$v"); end
end
println("done in $(round(elapsed; digits=1)) s")
