#!/usr/bin/env julia

# Paint several HalfDome Compton-y maps in one catalogue pass (2026-09-30):
#   b12              Battaglia12 fiducial pressure (production profile and cache)
#   lee22p           Lee22 no-c Table 7 pressure as printed (production profile and cache)
#   lee22p_m1        Lee22 x (1+z)^2: the group_particles bookkeeping correction (true P = printed x (1+z)^2,
#                    verified against CAMELS-TNG and FLAMINGO in LEE22_TSZ_VERIFICATION_20260929.md)
#   lee22p_m1_b12hi  Lee22 x (1+z)^2 below the Lee22 fit limit 10^14.8 h^-1 Msun, Battaglia12 above it
#   lee22p_m1_calib  Lee22 x (1+z)^2 only for halos inside the Lee22 calibration ranges (1e13-10^14.8 h^-1 Msun, z <= 2)
# Geometry, aperture (4 R200c projected, external), mass conversion and the per-pixel evaluation are those of
# paint_halfdome_battaglia12_tsz_map.jl, so halos at z >= --exact-below-z reproduce the production maps bit for bit.
# Low-z fix: the production caches interpolate linearly in z with dz = 0.0196, which is too coarse where theta200
# ~ 1/d_A (verifier of workstream A: outskirt errors up to x35 at z < 0.02, percents at z = 0.02-0.05). Halos below
# --exact-below-z therefore get a per-halo radial table from XGPaint's fill_profile_slice! (the function that fills
# the cache nodes) at their exact mass and redshift, interpolated in log-theta only.

using Dates
using HDF5
using Healpix
using Interpolations
using Random
using XGPaint

module MatchedMapSupport
include(joinpath(@__DIR__, "..", "paint_halfdome_matched_profile_dm_map.jl"))
end
const Support = MatchedMapSupport
const ProfileSupport = Support.ProfileSupport
include(joinpath(@__DIR__, "..", "lee2022_tsz_pressure_profile.jl"))

const H_VALUE = 0.68
const OMEGA_B = 0.049
const OMEGA_C = 0.261
const LEE22_FIT_MAX_MSUN = 10.0^14.8 / H_VALUE        # 9.278784477649915e14, the production shape-clip mass
const LEE22_FIT_MIN_MSUN = 1.0e13 / H_VALUE           # 1.4705882352941176e13
const LEE22_FIT_ZMAX = 2.0
const ALL_MAPS = ("b12", "lee22p", "lee22p_m1", "lee22p_m1_b12hi", "lee22p_m1_calib")

build_b12() = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H_VALUE,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758, x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0, beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)

"""Weights (w_b12, w_lee22) of the two base profiles in map `name` for a halo of physical M200c `m` at redshift `z`."""
function map_weights(name::AbstractString, m::Float64, z::Float64)
    name == "b12" && return (1.0, 0.0)
    name == "lee22p" && return (0.0, 1.0)
    name == "lee22p_m1" && return (0.0, (1 + z)^2)
    name == "lee22p_m1_b12hi" && return m < LEE22_FIT_MAX_MSUN ? (0.0, (1 + z)^2) : (1.0, 0.0)
    if name == "lee22p_m1_calib"
        inside = LEE22_FIT_MIN_MSUN <= m < LEE22_FIT_MAX_MSUN && z <= LEE22_FIT_ZMAX
        return inside ? (0.0, (1 + z)^2) : (0.0, 0.0)
    end
    error("unknown map $(name)")
end

"""Per-halo radial table: log-theta nodes from max(theta_min, 1e-5 theta200) to theta_max, step `dlog`."""
struct ExactTable
    logtheta::Vector{Float64}
    y::Vector{Float64}
end

function ExactTable(model, mass, z, theta_min, theta200, theta_max; dlog=0.02)
    lo = log(max(theta_min, 1.0e-5 * theta200))
    hi = log(theta_max) + 1.0e-9
    n = max(8, ceil(Int, (hi - lo) / dlog) + 1)
    logtheta = collect(range(lo, hi, length=n))
    y = zeros(n)
    XGPaint.fill_profile_slice!(y, model, logtheta, mass, z)
    return ExactTable(logtheta, y)
end

function (t::ExactTable)(theta)
    lt = clamp(log(theta), first(t.logtheta), last(t.logtheta))
    k = clamp(searchsortedlast(t.logtheta, lt), 1, length(t.logtheta) - 1)
    f = (lt - t.logtheta[k]) / (t.logtheta[k + 1] - t.logtheta[k])
    y0, y1 = t.y[k], t.y[k + 1]
    return (y0 > 0 && y1 > 0) ? exp((1 - f) * log(y0) + f * log(y1)) : (1 - f) * y0 + f * y1
end

theta200_rad(model, mass, z) = Float64(XGPaint.angular_size(model, XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200), z))

"""Paint one batch of halos into all maps (threads over halos, one lock per HEALPix ring)."""
function paint_batch!(maps, names, workspace, ring_locks, itp_b12, itp_lee, model_b12, model_lee, tmin_b12, tmin_lee,
                      aperture, masses, redshifts, positions, exact_below_z; sanity_max=1.0)
    nthreads = Support.thread_capacity()
    updates = zeros(Int64, nthreads); exact_halos = zeros(Int64, nthreads)
    nmap = length(names)
    Threads.@threads :static for h in eachindex(masses)
        tid = Threads.threadid()
        x, y, w = Float64(positions[1, h]), Float64(positions[2, h]), Float64(positions[3, h])
        d = sqrt(x^2 + y^2 + w^2)
        isfinite(d) && d > 0 || error("invalid position for halo $(h)")
        ux, uy, uz = x / d, y / d, w / d
        ctheta, cphi = Healpix.vec2ang(ux, uy, uz)
        ctheta = Float64(ctheta); cphi = mod(Float64(cphi), 2pi)
        m = Float64(masses[h]); z = Float64(redshifts[h])
        wts = ntuple(k -> map_weights(names[k], m, z), nmap)
        need_b = any(t -> t[1] > 0, wts); need_l = any(t -> t[2] > 0, wts)
        (need_b || need_l) || continue
        theta_max = min(ProfileSupport.compute_theta_max_r200c_external(itp_b12, m, z, aperture), pi)
        isfinite(theta_max) && theta_max > 0 || error("invalid theta_max for M=$(m), z=$(z)")
        exact = z < exact_below_z
        t200 = exact ? theta200_rad(model_b12, m, z) : 0.0
        fb = exact && need_b ? ExactTable(model_b12, m, z, tmin_b12, t200, theta_max) : nothing
        fl = exact && need_l ? ExactTable(model_lee, m, z, tmin_lee, t200, theta_max) : nothing
        exact_halos[tid] += exact
        r1, r2 = XGPaint.get_relevant_rings(workspace.res, ctheta, theta_max)
        n_upd = 0
        for ring in r1:r2
            rng1, rng2 = XGPaint.get_ring_disc_ranges(workspace, ring, ctheta, cphi, theta_max)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for lp in Iterators.flatten((rng1, rng2))
                    gp = first_pixel + lp - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, gp)
                    theta = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0))
                    theta < theta_max || continue
                    yb = !need_b ? 0.0 : exact ? fb(max(theta, tmin_b12)) : Float64(itp_b12(max(theta, tmin_b12), m, z))
                    yl = !need_l ? 0.0 : exact ? fl(max(theta, tmin_lee)) : Float64(itp_lee(max(theta, tmin_lee), m, z))
                    (isfinite(yb) && isfinite(yl) && yb >= 0 && yl >= 0) || error("invalid y at M=$(m), z=$(z), theta=$(theta)")
                    for k in 1:nmap
                        c = wts[k][1] * yb + wts[k][2] * yl
                        c == 0.0 && continue
                        c <= sanity_max || error("y=$(c) above sanity maximum at M=$(m), z=$(z)")
                        maps[k].pixels[gp] += c
                    end
                    n_upd += 1
                end
            end
        end
        updates[tid] += n_upd
    end
    return sum(updates), sum(exact_halos)
end

load_itp(model, path) = (isfile(path) || error("cache not found: $(path)"); build_interpolator(model, cache_file=path, overwrite=false))

"""Self-test: (1) exact tables vs fill_profile_slice! at random angles; (2) the table-free path equals the production
generic path (Support.paint_batch_external_r200c!) bit for bit for halos at z >= exact_below_z."""
function self_test(catalog, itp_b12, itp_lee, model_b12, model_lee, tmin_b12, tmin_lee, aperture, exact_below_z, rows)
    rng = MersenneTwister(20260930)
    worst = 0.0
    for trial in 1:300
        m = 10.0^(12.5 + 3.0 * rand(rng)); z = 10.0^(log10(0.0021) + rand(rng) * (log10(exact_below_z) - log10(0.0021)))
        tmax = ProfileSupport.compute_theta_max_r200c_external(itp_b12, m, z, aperture)
        t200 = theta200_rad(model_b12, m, z)
        for (model, tmin) in ((model_b12, tmin_b12), (model_lee, tmin_lee))
            tab = ExactTable(model, m, z, tmin, t200, tmax)
            lts = log(max(tmin, 1e-4 * t200)) .+ rand(rng, 12) .* (log(tmax) - log(max(tmin, 1e-4 * t200)))
            ref = zeros(length(lts)); XGPaint.fill_profile_slice!(ref, model, lts, m, z)
            for i in eachindex(lts)
                ref[i] > 0 || continue
                worst = max(worst, abs(tab(exp(lts[i])) / ref[i] - 1))
            end
        end
    end
    println("self-test exact tables: max relative error vs fill_profile_slice! = $(worst) (300 halos x 2 profiles x 12 angles)")
    worst < 2e-3 || error("exact-table self-test failed")
    # bit-for-bit reproduction of the production path on catalogue rows (z >= exact_below_z only), NSIDE 512
    nside = 512
    res = Healpix.Resolution(nside); ws = XGPaint.HealpixRingProfileWorkspace{Float64}(res)
    locks = [ReentrantLock() for _ in eachindex(ws.ring_thetas)]
    pos, mass, zz = h5open(catalog, "r") do f
        Float64.(f["Position"][:, rows]), Float64.(f["halo_mass_m200c"][rows]) ./ H_VALUE, Float64.(f["redshift"][rows])
    end
    keep = (mass .> 0) .& (zz .>= exact_below_z)
    count(keep) > 0 || error("self-test rows contain no halo at z >= $(exact_below_z)")
    pos, mass, zz = pos[:, keep], mass[keep], zz[keep]
    names = ["b12", "lee22p", "lee22p_m1"]
    newmaps = [HealpixMap{Float64,RingOrder}(nside) for _ in names]; foreach(m -> fill!(m.pixels, 0.0), newmaps)
    paint_batch!(newmaps, names, ws, locks, itp_b12, itp_lee, model_b12, model_lee, tmin_b12, tmin_lee, aperture, mass, zz,
                 pos, exact_below_z)
    for (k, itp, tmin) in ((1, itp_b12, tmin_b12), (2, itp_lee, tmin_lee))
        ref = HealpixMap{Float64,RingOrder}(nside); fill!(ref.pixels, 0.0)
        Support.paint_batch_external_r200c!(ref, ws, locks, itp, tmin, aperture, mass, zz, pos;
                                            contribution_sanity_max=1.0, quantity_label="y", quantity_units="")
        # summation order over halos differs between threads, so compare to rounding
        diff = maximum(abs.(newmaps[k].pixels .- ref.pixels)) / maximum(ref.pixels)
        println("self-test production path ($(names[k]), $(count(keep)) halos): max |new - production| / max = $(diff)")
        diff < 1e-12 || error("production-path self-test failed for $(names[k])")
    end
    # lee22p_m1 must equal lee22p with each halo weighted by (1+z)^2; with a single redshift it is a constant factor
    hbig = argmax(mass)
    one = HealpixMap{Float64,RingOrder}(nside); two = HealpixMap{Float64,RingOrder}(nside)
    fill!(one.pixels, 0.0); fill!(two.pixels, 0.0)
    paint_batch!([one, two], ["lee22p", "lee22p_m1"], ws, locks, itp_b12, itp_lee, model_b12, model_lee, tmin_b12, tmin_lee,
                 aperture, mass[hbig:hbig], zz[hbig:hbig], pos[:, hbig:hbig], exact_below_z)
    ratio = sum(two.pixels) / sum(one.pixels)
    println("self-test (1+z)^2 weight: ratio $(ratio) vs $((1 + zz[hbig])^2)")
    abs(ratio / (1 + zz[hbig])^2 - 1) < 1e-12 || error("(1+z)^2 weight self-test failed")
    println("self-test OK")
end

function main()
    options = Support.parse_options(ARGS)
    catalog = abspath(Support.option(options, "halfdome_path", "/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5"))
    nside = Support.int_option(options, "nside", 4096)
    aperture = Support.float_option(options, "aperture_r200c", 4.0)
    exact_below_z = Support.float_option(options, "exact_below_z", 0.1)
    chunk_size = Support.int_option(options, "chunk_size", 100_000)
    b12_cache = abspath(Support.option(options, "b12_cache", ""))
    lee_cache = abspath(Support.option(options, "lee22_cache", ""))
    outdir = abspath(Support.option(options, "output_dir", ""))
    names = String.(split(Support.option(options, "maps", join(ALL_MAPS, ",")), ","))
    all(in(ALL_MAPS), names) || error("unknown map in $(names); allowed $(ALL_MAPS)")
    overwrite = Support.bool_option(options, "overwrite", false)
    rows_opt = Support.option(options, "rows", "")
    tag = Support.option(options, "map_tag", "")

    model_b12 = build_b12(); model_lee = Lee2022ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H_VALUE)
    run_lee2022_tsz_self_test(model_lee; quad=ProfileSupport.lee2022_quadgk_function())
    itp_b12 = load_itp(model_b12, b12_cache); itp_lee = load_itp(model_lee, lee_cache)
    tmin_b12 = ProfileSupport.compute_theta_min_local(itp_b12); tmin_lee = ProfileSupport.compute_theta_min_local(itp_lee)
    bounds_b12 = ProfileSupport.interpolator_logmass_bounds(itp_b12); bounds_lee = ProfileSupport.interpolator_logmass_bounds(itp_lee)
    println("caches: B12 $(b12_cache) logM $(bounds_b12) theta_min $(tmin_b12); Lee22 $(lee_cache) logM $(bounds_lee) theta_min $(tmin_lee)")
    println("exact per-halo tables below z = $(exact_below_z); maps: $(join(names, ", ")); threads $(Threads.nthreads())")

    if Support.bool_option(options, "painter_self_test", false)
        a, b = parse.(Int, split(Support.option(options, "painter_self_test_rows", "60000001:60050000"), ":"))
        self_test(catalog, itp_b12, itp_lee, model_b12, model_lee, tmin_b12, tmin_lee, aperture, exact_below_z, a:b)
        Support.bool_option(options, "painter_self_test_only", false) && return
    end

    isempty(outdir) && error("--output-dir is required")
    mkpath(outdir)
    paths = [joinpath(outdir, "$(n)_compton_y_allz_nside$(nside)_m200c_r200cx4_lowzexact$(tag).fits") for n in names]
    for p in paths
        isfile(p) && !overwrite && error("output exists: $(p) (pass --overwrite=true)")
    end
    res = Healpix.Resolution(nside)
    maps = [HealpixMap{Float64,RingOrder}(nside) for _ in names]; foreach(m -> fill!(m.pixels, 0.0), maps)
    ws = XGPaint.HealpixRingProfileWorkspace{Float64}(res)
    locks = [ReentrantLock() for _ in eachindex(ws.ring_thetas)]
    t0 = time(); scanned = 0; selected = 0; upd = 0; nexact = 0; total = 0; batch = 0
    h5open(catalog, "r") do f
        P = f["Position"]; M = f["halo_mass_m200c"]; Z = f["redshift"]
        total = size(P, 2)
        first_row, last_row = isempty(rows_opt) ? (1, total) : Tuple(parse.(Int, split(rows_opt, ":")))
        for s in first_row:chunk_size:last_row
            batch += 1
            idx = s:min(s + chunk_size - 1, last_row)
            pos = Float64.(P[:, idx]); mass = Float64.(M[idx]) ./ H_VALUE; zz = Float64.(Z[idx])
            scanned += length(idx)
            keep = isfinite.(mass) .& isfinite.(zz) .& (mass .> 0) .& (zz .> 0)
            any(keep) || continue
            pos, mass, zz = pos[:, keep], mass[keep], zz[keep]
            ProfileSupport.validate_profile_masses_in_cache(mass, bounds_b12)
            ProfileSupport.validate_profile_masses_in_cache(mass, bounds_lee)
            u, e = paint_batch!(maps, names, ws, locks, itp_b12, itp_lee, model_b12, model_lee, tmin_b12, tmin_lee, aperture,
                                mass, zz, pos, exact_below_z)
            selected += length(mass); upd += u; nexact += e
            if batch % 20 == 0 || last(idx) == last_row
                println("  rows $(first(idx)):$(last(idx)) halos $(selected) exact $(nexact) pixel updates $(upd) " *
                        "elapsed $(round(time() - t0, digits=1)) s"); flush(stdout)
            end
        end
    end
    elapsed = time() - t0
    for (k, n) in enumerate(names)
        all(isfinite, maps[k].pixels) && minimum(maps[k].pixels) >= 0 || error("invalid pixels in $(n)")
        Healpix.saveToFITS(maps[k], "!" * paths[k], typechar="D")
        Support.write_provenance(splitext(paths[k])[1] * "_provenance.txt", Dict{String,Any}(
            "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
            "map" => n, "painter" => "frb_map_generation/lee22_tsz_corrected/paint_tsz_variants.jl",
            "definition" => Dict("b12" => "Battaglia12 fiducial pressure", "lee22p" => "Lee22 no-c Table 7 pressure as printed",
                "lee22p_m1" => "Lee22 x (1+z)^2 (bookkeeping correction)",
                "lee22p_m1_b12hi" => "Lee22 x (1+z)^2 for M200c < 10^14.8/h Msun, Battaglia12 above",
                "lee22p_m1_calib" => "Lee22 x (1+z)^2 for 1e13/h <= M200c < 10^14.8/h Msun and z <= 2 only")[n],
            "catalogue" => catalog, "catalog_total_rows" => total, "catalog_rows_scanned" => scanned,
            "halos_selected" => selected, "halos_exact_tables" => nexact, "exact_below_z" => exact_below_z,
            "exact_table" => "XGPaint.fill_profile_slice! at the halo's M200c and z, log-theta step 0.02 from max(theta_min, 1e-5 theta200) to theta_max, log-log linear",
            "b12_cache" => b12_cache, "lee22_cache" => lee_cache, "aperture_r200c" => aperture,
            "aperture_definition" => "projected, external, R200c from physical M200c", "profile_mass" => "halo_mass_m200c / h (physical Msun)",
            "lee22_fit_max_msun" => LEE22_FIT_MAX_MSUN, "lee22_fit_min_msun" => LEE22_FIT_MIN_MSUN,
            "h" => H_VALUE, "Omega_b" => OMEGA_B, "Omega_c" => OMEGA_C, "nside" => nside, "ordering" => "RING",
            "beam" => "none", "pixel_window" => "none (point samples)", "julia_threads" => Threads.nthreads(),
            "map_mean_y" => sum(maps[k].pixels) / length(maps[k].pixels), "map_max_y" => maximum(maps[k].pixels),
            "pixel_updates" => upd, "elapsed_seconds" => elapsed))
        println("saved $(paths[k]) mean y $(sum(maps[k].pixels) / length(maps[k].pixels))")
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
