# Truncation-radius series for the HalfDome Compton-y map (spherical / chord-limited truncation).
#
# One pass over the HalfDome lightcone paints six NSIDE maps that share halos, pixels and profile:
#   y_sphere{1,2,3,4,5}  gas inside the N R200c sphere only (ChordMeanProfile + exact chord factor)
#   y_projected4         XGPaint's original convention: infinite LOS, disc cut at 4 R200c (WebSky's choice)
# then computes each map's angular power spectrum (Healpix.anafast, niter=0, raw; the pixel window is
# applied downstream) and writes CSV spectra, FITS maps and a provenance file.
#
# Usage: julia -t 26 paint_radius_series.jl --catalog=... --output-dir=... --cache-dir=... [--nside=4096]
#        [--max-halos=0] [--chunk-size=1000000] [--lmax=8192] [--save-maps=true]
using Dates, HDF5, Healpix, Interpolations, XGPaint
using Base.Threads
include(isfile(joinpath(@__DIR__, "spherical_truncation_profiles.jl")) ? joinpath(@__DIR__, "spherical_truncation_profiles.jl") : joinpath(@__DIR__, "..", "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603
const RADII = (1.0, 2.0, 3.0, 4.0, 5.0)
const PROJECTED_RADIUS = 4.0

function getarg(key, default)
    for a in ARGS
        startswith(a, "--$key=") && return split(a, "="; limit=2)[2]
    end
    return default
end
catalog = getarg("catalog", "/home/cbllover/HalfDome/lightcone_100.hdf5")
nside = parse(Int, getarg("nside", "4096"))
max_halos = parse(Int, getarg("max-halos", "0"))
chunk_size = parse(Int, getarg("chunk-size", "1000000"))
lmax = parse(Int, getarg("lmax", "8192"))
save_maps = lowercase(getarg("save-maps", "true")) in ("true", "1", "yes")
output_dir = getarg("output-dir", joinpath(@__DIR__, "output"))
cache_dir = getarg("cache-dir", joinpath(@__DIR__, "caches"))
use_cache = lowercase(getarg("use-cache", "true")) in ("true", "1", "yes")   # false: build in memory (no JLD2 needed)
mkpath(output_dir); mkpath(cache_dir)
lmax = min(lmax, 3nside - 1)

tsz = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
function load_or_build(name, model)
    file = use_cache ? joinpath(cache_dir, "$(name)_interpolator.jld2") : ""
    t0 = time(); existed = use_cache && isfile(file)
    itp = build_interpolator(model; cache_file=file, N_logtheta=512, pad=128, logM_max=15.7,
                             overwrite=!existed, verbose=false)
    println("interpolator $(name): $(existed ? "loaded" : "built") in $(round(time() - t0; digits=1)) s"); flush(stdout)
    return itp
end
itp_sphere = Tuple(load_or_build("tsz_sphere$(replace(string(X), "." => "p"))_chordmean", ChordMeanProfile(tsz, X)) for X in RADII)
itp_proj = load_or_build("tsz_projected", tsz)
all_itps = (itp_sphere..., itp_proj)
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in all_itps)
logm_lo = minimum(first(ip.itp.ranges[3]) for ip in all_itps); logm_hi = maximum(last(ip.itp.ranges[3]) for ip in all_itps)
z_hi = minimum(last(ip.itp.ranges[2]) for ip in all_itps)
println("theta_min=$(theta_min) rad; cache log10 M range $(logm_lo)-$(logm_hi); z max $(z_hi)")
for (M, z) in ((1e13, 0.5), (1e14, 0.5), (1e15, 0.3))
    θ = 0.7 * theta_r200c(tsz, M, z)
    checks = ["proj $(round(itp_proj(θ, M, z) / tsz(θ, M, z); digits=7))"]
    for (k, X) in enumerate(RADII)
        push!(checks, "sphere$(X) $(round(itp_sphere[k](θ, M, z) / ChordMeanProfile(tsz, X)(θ, M, z); digits=7))")
    end
    println("  cache/direct spot M=$M z=$z: ", join(checks, "  "))
end
flush(stdout)

resolution = Healpix.Resolution(nside)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
names = [["y_sphere$(Int(X))" for X in RADII]; "y_projected$(Int(PROJECTED_RADIUS))"]
maps = [HealpixMap{Float64,RingOrder}(nside) for _ in names]
for m in maps; fill!(m.pixels, 0.0); end
pix = Tuple(m.pixels for m in maps)
const NR = length(RADII)
outer_radius = max(maximum(RADII), PROJECTED_RADIUS)

function paint_batch!(masses, redshifts, positions)
    updates = zeros(Int, Threads.maxthreadid()); hits = zeros(Int, NR + 1, Threads.maxthreadid())
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        x, y, zc = Float64(positions[1, i]), Float64(positions[2, i]), Float64(positions[3, i]); d = sqrt(x^2 + y^2 + zc^2)
        ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz); θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        M = Float64(masses[i]); z = Float64(redshifts[i])
        θ200 = theta_r200c(tsz, M, z)
        θouter = min(outer_radius * θ200, pi)
        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, θc, θouter); n = 0
        for ring in ring_start:ring_stop
            r1, r2 = XGPaint.get_ring_disc_ranges(workspace, ring, θc, φc, θouter); first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((r1, r2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    θ = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0)); θ < θouter || continue
                    θe = max(θ, theta_min); n += 1
                    for (k, X) in enumerate(RADII)
                        θmax = X * θ200
                        θ < θmax || continue
                        pix[k][gp] += itp_sphere[k](θe, M, z) * chord_factor(θ, θmax, X); hits[k, tid] += 1
                    end
                    if θ < PROJECTED_RADIUS * θ200
                        pix[NR + 1][gp] += itp_proj(θe, M, z); hits[NR + 1, tid] += 1
                    end
                end
            end
        end
        updates[tid] += n
    end
    return sum(updates), vec(sum(hits; dims=2))
end

t0 = time(); total_visits = 0; total_hits = zeros(Int, NR + 1); nsel = 0
mmin = Inf; mmax = -Inf; zmin = Inf; zmax = -Inf
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mass_ds = f["halo_mass_m200c"]; z_ds = f["redshift"]
    N = size(pos_ds, 2); last = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last
        stop = min(start + chunk_size - 1, last); idx = start:stop
        pos = Float64.(pos_ds[:, idx]); masses = Float64.(mass_ds[idx]) ./ H; zs = Float64.(z_ds[idx])
        keep = isfinite.(masses) .& (masses .> 0) .& isfinite.(zs) .& (zs .> 0) .& vec(all(isfinite, pos; dims=1))
        pos = pos[:, keep]; masses = masses[keep]; zs = zs[keep]
        all(logm_lo .<= log10.(masses) .<= logm_hi) || error("mass outside cache range in chunk $start")
        all(zs .<= z_hi) || error("redshift outside cache range in chunk $start")
        visits, hits = paint_batch!(masses, zs, pos)
        global total_visits += visits; global total_hits .+= hits; global nsel += length(masses)
        global mmin = min(mmin, minimum(masses)); global mmax = max(mmax, maximum(masses))
        global zmin = min(zmin, minimum(zs)); global zmax = max(zmax, maximum(zs))
        println("  rows $start:$stop  halos=$nsel  pixel visits=$total_visits  elapsed=$(round(time() - t0; digits=0)) s"); flush(stdout)
    end
end
paint_seconds = time() - t0
println("painting done in $(round(paint_seconds; digits=1)) s; pixel updates per map: ", Dict(zip(names, total_hits))); flush(stdout)

# spectra (raw; pixel window applied downstream)
t1 = time()
cls = Vector{Vector{Float64}}(undef, length(maps))
for (k, m) in enumerate(maps)
    all(isfinite, m.pixels) || error("non-finite pixels in $(names[k])")
    cls[k] = anafast(m; lmax=lmax, niter=0)
    println("  anafast $(names[k]): mean y=$(sum(m.pixels)/length(m.pixels))  D_ell(min(3000,lmax))=$(let l=min(3000,lmax); l*(l+1)*cls[k][l+1]/(2pi) end)  ($(round(time() - t1; digits=0)) s)"); flush(stdout)
end
open(joinpath(output_dir, "raw_cl_radius_series.csv"), "w") do io
    println(io, "ell," * join(names, ","))
    for l in 0:lmax
        println(io, string(l) * "," * join((string(cls[k][l + 1]) for k in eachindex(names)), ","))
    end
end
if save_maps
    for (k, m) in enumerate(maps)
        Healpix.saveToFITS(m, "!" * joinpath(output_dir, "halfdome_$(names[k])_nside$(nside).fits"), typechar="D")
    end
end
open(joinpath(output_dir, "provenance.txt"), "w") do io
    for (k, v) in sort(collect(Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "host" => gethostname(), "julia" => string(VERSION),
        "catalog" => catalog, "nside" => nside, "lmax" => lmax, "halos_painted" => nsel, "max_halos_option" => max_halos,
        "pixel_visits_outer_disc" => total_visits, "pixel_updates_per_map" => join(["$(n)=$(h)" for (n, h) in zip(names, total_hits)], ";"),
        "paint_seconds" => round(paint_seconds; digits=1), "anafast_seconds" => round(time() - t1; digits=1), "julia_threads" => Threads.nthreads(),
        "h" => H, "Omega_b" => OMEGA_B, "Omega_c" => OMEGA_C,
        "mass_definition" => "M200c, catalogue halo_mass_m200c / h -> physical Msun",
        "selected_mass_min_msun" => mmin, "selected_mass_max_msun" => mmax, "selected_redshift_min" => zmin, "selected_redshift_max" => zmax,
        "profile" => "Battaglia16ThermalSZProfile defaults (Battaglia12 AGN-feedback pressure fit, Delta=200c), Compton-y",
        "spherical_maps" => "y_sphereN: pressure integrated only inside the N R200c sphere (chord-mean cache x exact chord factor 2N sqrt(1-(theta/theta_max)^2)), N in " * string(RADII),
        "projected_map" => "y_projected4: XGPaint/WebSky convention, infinite line of sight, painted inside the 4 R200c disc",
        "spectra" => "Healpix.anafast(map; lmax, niter=0), raw C_ell; pixel window NOT applied here",
        "interpolator_caches" => (use_cache ? cache_dir : "none (built in memory)"), "theta_min_rad" => theta_min,
    )); by=first); println(io, "$k=$v"); end
end
println("done: $(output_dir)")
