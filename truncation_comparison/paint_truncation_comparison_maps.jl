# Paint HalfDome full-sky Compton-y and kSZ (Delta T / T) maps with the original
# projected-disc truncation (XGPaint default: profile integrated along the full
# LOS, painted inside theta_max = X R200c) and with the spherical truncation
# (only gas inside the X R200c sphere; chord-limited LOS). All four maps are
# painted in one pass over the catalogue so they share halos, geometry and
# velocities exactly.
#
# Usage: julia -t 20 paint_truncation_comparison_maps.jl --catalog=... --nside=4096
#        [--max-halos=0] [--chunk-size=1000000] [--output-dir=maps] [--overwrite]
using Dates, HDF5, Healpix, Interpolations, XGPaint
using Base.Threads
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603
const SPHERE_R200C = 4.0
const C_KMS = 299_792.458

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
output_dir = getarg("output-dir", joinpath(@__DIR__, "maps"))
tag = getarg("tag", "")
overwrite = "--overwrite" in ARGS
cache_dir = joinpath(@__DIR__, "caches")
mkpath(output_dir)

tsz = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
tau = BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
load(name, model) = build_interpolator(model; cache_file=joinpath(cache_dir, "$(name)_interpolator.jld2"),
                                       overwrite=false, verbose=false)
itp_y_proj = load("tsz_projected", tsz)
itp_y_sph = load("tsz_sphere4p0_chordmean", ChordMeanProfile(tsz, SPHERE_R200C))
itp_t_proj = load("tau_projected", tau)
itp_t_sph = load("tau_sphere4p0_chordmean", ChordMeanProfile(tau, SPHERE_R200C))
theta_min = maximum(exp(first(first(ip.itp.ranges))) for ip in (itp_y_proj, itp_y_sph, itp_t_proj, itp_t_sph))
logm_lo = minimum(first(ip.itp.ranges[3]) for ip in (itp_y_proj, itp_y_sph, itp_t_proj, itp_t_sph))
logm_hi = maximum(last(ip.itp.ranges[3]) for ip in (itp_y_proj, itp_y_sph, itp_t_proj, itp_t_sph))
z_hi = minimum(last(ip.itp.ranges[2]) for ip in (itp_y_proj, itp_y_sph, itp_t_proj, itp_t_sph))
println("theta_min=$(theta_min) rad; cache log10 M range $(logm_lo)-$(logm_hi); z max $(z_hi)")

# spot check of the caches against direct evaluation
for (M, z) in ((1e13, 0.5), (1e14, 0.5), (1e15, 0.3))
    θ200 = theta_r200c(tsz, M, z); θ = 0.7θ200
    println("  spot M=$M z=$z: y_proj cache/direct=", itp_y_proj(θ, M, z) / tsz(θ, M, z),
            "  y_sph cache/direct=", itp_y_sph(θ, M, z) / ChordMeanProfile(tsz, SPHERE_R200C)(θ, M, z),
            "  tau_proj=", itp_t_proj(θ, M, z) / tau(θ, M, z),
            "  tau_sph=", itp_t_sph(θ, M, z) / ChordMeanProfile(tau, SPHERE_R200C)(θ, M, z))
end

resolution = Healpix.Resolution(nside)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
maps = Dict(name => HealpixMap{Float64,RingOrder}(nside) for name in
            ("y_projected", "y_sphere", "ksz_projected", "ksz_sphere"))
for m in values(maps); fill!(m.pixels, 0.0); end
y_proj = maps["y_projected"].pixels; y_sph = maps["y_sphere"].pixels
k_proj = maps["ksz_projected"].pixels; k_sph = maps["ksz_sphere"].pixels

function paint_batch!(masses, redshifts, positions, vlos_over_c)
    updates = zeros(Int, Threads.maxthreadid())
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        x, y, zc = Float64(positions[1, i]), Float64(positions[2, i]), Float64(positions[3, i])
        d = sqrt(x^2 + y^2 + zc^2)
        ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz)
        θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        M = Float64(masses[i]); z = Float64(redshifts[i]); v = Float64(vlos_over_c[i])
        θ200 = theta_r200c(tsz, M, z)
        θmax = min(SPHERE_R200C * θ200, pi)
        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, θc, θmax)
        n = 0
        for ring in ring_start:ring_stop
            r1, r2 = XGPaint.get_ring_disc_ranges(workspace, ring, θc, φc, θmax)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((r1, r2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    θ = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0))
                    θ < θmax || continue
                    cf = chord_factor(θ, θmax, SPHERE_R200C)
                    θe = max(θ, theta_min)
                    yp = itp_y_proj(θe, M, z); ys = itp_y_sph(θe, M, z) * cf
                    tp = itp_t_proj(θe, M, z); ts = itp_t_sph(θe, M, z) * cf
                    y_proj[gp] += yp; y_sph[gp] += ys
                    k_proj[gp] -= v * tp; k_sph[gp] -= v * ts    # Delta T / T = - tau v_los / c
                    n += 1
                end
            end
        end
        updates[tid] += n
    end
    return sum(updates)
end

# (log10 M, z) histogram of painted halos with summed (v_los/c)^2, for the catalogue Poisson term
const LOGM_EDGES = collect(12.0:0.025:15.7); const Z_EDGES = collect(0.0:0.025:4.0)
hist_n = zeros(Int, length(LOGM_EDGES) - 1, length(Z_EDGES) - 1); hist_v2 = zeros(Float64, size(hist_n))
t0 = time(); total_updates = 0; nsel = 0
mmin = Inf; mmax = -Inf; zmin = Inf; zmax = -Inf; vrms_acc = 0.0
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mass_ds = f["halo_mass_m200c"]; vel_ds = f["Velocity"]; z_ds = f["redshift"]
    N = size(pos_ds, 2)
    last = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last
        stop = min(start + chunk_size - 1, last)
        idx = start:stop
        pos = Float64.(pos_ds[:, idx]); vel = Float64.(vel_ds[:, idx])
        masses = Float64.(mass_ds[idx]) ./ H          # Msun/h -> physical Msun (M200c)
        zs = Float64.(z_ds[idx])
        keep = isfinite.(masses) .& (masses .> 0) .& isfinite.(zs) .& (zs .> 0) .& vec(all(isfinite, pos; dims=1)) .& vec(all(isfinite, vel; dims=1))
        pos = pos[:, keep]; vel = vel[:, keep]; masses = masses[keep]; zs = zs[keep]
        r = sqrt.(vec(sum(pos .^ 2; dims=1)))
        vlos_c = vec(sum(pos .* vel; dims=1)) ./ r ./ C_KMS
        all(logm_lo .<= log10.(masses) .<= logm_hi) || error("mass outside cache range in chunk $start")
        all(zs .<= z_hi) || error("redshift outside cache range in chunk $start")
        for j in eachindex(masses)
            im = searchsortedlast(LOGM_EDGES, log10(masses[j])); iz = searchsortedlast(Z_EDGES, zs[j])
            (1 <= im <= size(hist_n, 1) && 1 <= iz <= size(hist_n, 2)) || error("halo outside histogram range")
            hist_n[im, iz] += 1; hist_v2[im, iz] += vlos_c[j]^2
        end
        global total_updates += paint_batch!(masses, zs, pos, vlos_c)
        global nsel += length(masses)
        global mmin = min(mmin, minimum(masses)); global mmax = max(mmax, maximum(masses))
        global zmin = min(zmin, minimum(zs)); global zmax = max(zmax, maximum(zs))
        global vrms_acc += sum(abs2, vlos_c .* C_KMS)
        println("  rows $start:$stop  halos=$nsel  pixel updates=$total_updates  elapsed=$(round(time() - t0; digits=0)) s"); flush(stdout)
    end
end
elapsed = time() - t0
for (name, m) in maps
    all(isfinite, m.pixels) || error("non-finite pixels in $name")
    path = joinpath(output_dir, "halfdome_$(name)_nside$(nside)_r200cx4$(tag).fits")
    Healpix.saveToFITS(m, "!" * path, typechar="D")
    println("saved $path  mean=$(sum(m.pixels)/length(m.pixels))  min=$(minimum(m.pixels))  max=$(maximum(m.pixels))")
end
h5open(joinpath(output_dir, "halo_mass_redshift_histogram$(tag).h5"), "w") do f
    f["logm_edges"] = LOGM_EDGES; f["z_edges"] = Z_EDGES; f["count"] = hist_n; f["sum_vlos_over_c_squared"] = hist_v2
end
open(joinpath(output_dir, "provenance$(tag).txt"), "w") do io
    for (k, v) in sort(collect(Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "catalog" => catalog, "nside" => nside, "halos_painted" => nsel, "max_halos_option" => max_halos,
        "pixel_updates_per_map" => total_updates, "elapsed_seconds" => round(elapsed; digits=1),
        "julia_threads" => Threads.nthreads(),
        "h" => H, "Omega_b" => OMEGA_B, "Omega_c" => OMEGA_C,
        "mass_definition" => "M200c, catalogue halo_mass_m200c / h -> physical Msun",
        "selected_mass_min_msun" => mmin, "selected_mass_max_msun" => mmax,
        "selected_mass_min_msun_h" => mmin * H, "selected_mass_max_msun_h" => mmax * H,
        "selected_redshift_min" => zmin, "selected_redshift_max" => zmax,
        "aperture" => "theta_max = $(SPHERE_R200C) R200c for both truncations (XGPaint compute_thetamax default mult=4)",
        "projected_truncation" => "XGPaint _nfw_profile_los_quadrature: LOS integral to 1e5 R200c, disc cut at theta_max only (previous code)",
        "spherical_truncation" => "gas inside the $(SPHERE_R200C) R200c sphere only: chord-mean cache times 2 X sqrt(1-(theta/theta_max)^2) per pixel",
        "tsz_profile" => "Battaglia16ThermalSZProfile defaults (Battaglia12 fiducial pressure), Compton-y",
        "ksz_profile" => "BattagliaTauProfile defaults (Battaglia16 AGN feedback density, Delta=200c), map = -tau * v_los/c (dimensionless Delta T/T)",
        "velocity" => "catalogue Velocity [km/s], projected on the unit position vector; rms v_los = $(sqrt(vrms_acc / max(nsel, 1))) km/s",
        "interpolator_caches" => join([joinpath(cache_dir, "$(n)_interpolator.jld2") for n in ("tsz_projected", "tsz_sphere4p0_chordmean", "tau_projected", "tau_sphere4p0_chordmean")], ";"),
        "theta_min_rad" => theta_min,
    )); by=first)
        println(io, "$k=$v")
    end
end
println("done in $(round(elapsed; digits=1)) s")
