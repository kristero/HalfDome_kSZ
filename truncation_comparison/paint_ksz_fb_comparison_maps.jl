# Paint the halo kSZ map with and without the f_b (Omega_b/Omega_m) factor on the Battaglia16
# gas density, both with the validated spherical (4 R200c) truncation used in
# TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md. Both maps share the same halos, geometry, and
# catalogue velocities exactly (one pass over the catalogue).
#
# The f_b factor enters BattagliaTauProfile's rho_2d purely as a multiplicative constant
# (rho_gas = rho_fit * f_b * rho_crit, XGPaint profiles_tau.jl) with no interaction with mass,
# redshift, radius, or the truncation geometry. So "without f_b" is obtained by dividing the
# already-truncated, already-velocity-weighted per-pixel kSZ contribution by f_b in the same
# accumulation loop -- mathematically identical to painting with a from-scratch "no f_b" profile
# object, without needing a second XGPaint profile type or a second interpolator cache.
#
# Usage: julia -t N paint_ksz_fb_comparison_maps.jl --catalog=... --nside=4096 [--tag=...]
using Dates, HDF5, Healpix, Interpolations, XGPaint
using Base.Threads
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603
const F_B = OMEGA_B / (OMEGA_B + OMEGA_C)
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
cache_dir = joinpath(@__DIR__, "caches")
mkpath(output_dir); mkpath(cache_dir)

println("F_B = Omega_b/(Omega_b+Omega_c) = ", F_B, "  (1/F_B = ", 1 / F_B, ")")
tau = BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
load(name, model) = build_interpolator(model; cache_file=joinpath(cache_dir, "$(name)_interpolator.jld2"),
                                       overwrite=false, verbose=false)
itp_t_sph = load("tau_sphere4p0_chordmean", ChordMeanProfile(tau, SPHERE_R200C))
theta_min = exp(first(first(itp_t_sph.itp.ranges)))
logm_lo, logm_hi = first(itp_t_sph.itp.ranges[3]), last(itp_t_sph.itp.ranges[3])
z_hi = last(itp_t_sph.itp.ranges[2])
println("theta_min=$(theta_min) rad; cache log10 M range $(logm_lo)-$(logm_hi); z max $(z_hi)")

for (M, z) in ((1e13, 0.5), (1e14, 0.5), (1e15, 0.3))
    θ200 = theta_r200c(tau, M, z); θ = 0.7θ200
    println("  spot M=$M z=$z: tau_sph cache/direct=", itp_t_sph(θ, M, z) / ChordMeanProfile(tau, SPHERE_R200C)(θ, M, z))
end

resolution = Healpix.Resolution(nside)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
maps = Dict(name => HealpixMap{Float64,RingOrder}(nside) for name in ("ksz_sphere_with_fb", "ksz_sphere_no_fb"))
for m in values(maps); fill!(m.pixels, 0.0); end
k_with = maps["ksz_sphere_with_fb"].pixels; k_without = maps["ksz_sphere_no_fb"].pixels

function paint_batch!(masses, redshifts, positions, vlos_over_c)
    updates = zeros(Int, Threads.nthreads())
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        x, y, zc = Float64(positions[1, i]), Float64(positions[2, i]), Float64(positions[3, i])
        d = sqrt(x^2 + y^2 + zc^2)
        ux, uy, uz = x / d, y / d, zc / d
        θc, φc = Healpix.vec2ang(ux, uy, uz)
        θc = Float64(θc); φc = mod(Float64(φc), 2pi)
        M = Float64(masses[i]); z = Float64(redshifts[i]); v = Float64(vlos_over_c[i])
        θ200 = theta_r200c(tau, M, z)
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
                    ts = itp_t_sph(θe, M, z) * cf
                    k_with[gp] -= v * ts             # Delta T/T = -tau v_los/c, with f_b (production)
                    k_without[gp] -= (v * ts) / F_B  # exact linear rescale: rho_gas / f_b removed
                    n += 1
                end
            end
        end
        updates[tid] += n
    end
    return sum(updates)
end

t0 = time(); total_updates = 0; nsel = 0
h5open(catalog, "r") do f
    pos_ds = f["Position"]; mass_ds = f["halo_mass_m200c"]; vel_ds = f["Velocity"]; z_ds = f["redshift"]
    N = size(pos_ds, 2)
    last_row = max_halos == 0 ? N : min(max_halos, N)
    for start in 1:chunk_size:last_row
        stop = min(start + chunk_size - 1, last_row)
        idx = start:stop
        pos = Float64.(pos_ds[:, idx]); vel = Float64.(vel_ds[:, idx])
        masses = Float64.(mass_ds[idx]) ./ H
        zs = Float64.(z_ds[idx])
        keep = isfinite.(masses) .& (masses .> 0) .& isfinite.(zs) .& (zs .> 0) .& vec(all(isfinite, pos; dims=1)) .& vec(all(isfinite, vel; dims=1))
        pos = pos[:, keep]; vel = vel[:, keep]; masses = masses[keep]; zs = zs[keep]
        r = sqrt.(vec(sum(pos .^ 2; dims=1)))
        vlos_c = vec(sum(pos .* vel; dims=1)) ./ r ./ C_KMS
        all(logm_lo .<= log10.(masses) .<= logm_hi) || error("mass outside cache range in chunk $start")
        all(zs .<= z_hi) || error("redshift outside cache range in chunk $start")
        global total_updates += paint_batch!(masses, zs, pos, vlos_c)
        global nsel += length(masses)
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
open(joinpath(output_dir, "provenance_fb$(tag).txt"), "w") do io
    println(io, "created_utc=", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"))
    println(io, "catalog=", catalog, "  nside=", nside, "  halos_painted=", nsel, "  elapsed_s=", round(elapsed; digits=1))
    println(io, "f_b=", F_B, "  1/f_b=", 1 / F_B)
    println(io, "ksz_sphere_with_fb: BattagliaTauProfile defaults (Omega_b=", OMEGA_B, ", Omega_c=", OMEGA_C, "), 4 R200c spherical truncation, map = -tau*v_los/c")
    println(io, "ksz_sphere_no_fb: identical pipeline, per-pixel accumulation divided by f_b (exact rescale of the physically-normalized tau)")
end
println("done in $(round(elapsed; digits=1)) s")
