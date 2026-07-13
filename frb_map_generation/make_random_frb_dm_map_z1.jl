if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# Random hostless FRBs at a configurable fixed redshift, using the same XGPaint DM profile
# path as the existing FRB code. Foreground halos are strictly limited to
# 0 <= z <= frb_redshift.

using XGPaint
using Healpix
using Interpolations
using HDF5
using Random
using Statistics
using Base.Threads
using Plots

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const OMEGAM = OMEGAB + OMEGAC
const C_KMS = 299_792.458
const DEFAULT_FRB_REDSHIFT = 1.0

const compute_theta_max_local =
    isdefined(XGPaint, Symbol("compute_θmax")) ?
    getfield(XGPaint, Symbol("compute_θmax")) :
    error("XGPaint does not define compute_θmax.")

thread_capacity() = isdefined(Base.Threads, :maxthreadid) ? Base.Threads.maxthreadid() : Base.Threads.nthreads()

function get_string_arg(key, default; env=nothing)
    if env !== nothing
        env_names = env isa AbstractString ? (env,) : env
        for env_name in env_names
            if haskey(ENV, env_name)
                return String(ENV[env_name])
            end
        end
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for arg in ARGS
        if startswith(arg, prefix1)
            return String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, prefix2)
            return String(split(arg, "=", limit=2)[2])
        end
    end
    return String(default)
end

function get_int_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Int, value)
    return Int(default)
end

function get_float_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Float64, value)
    return Float64(default)
end

function parse_bool_arg(value)
    value_norm = lowercase(strip(String(value)))
    if value_norm in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif value_norm in ("0", "false", "f", "no", "n", "off")
        return false
    end
    error("Could not parse boolean value $(repr(value)).")
end

function get_bool_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse_bool_arg(value)
    return Bool(default)
end

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

code_root() = @__DIR__
repo_root() = basename(code_root()) == "frb_map_generation" ? dirname(code_root()) : code_root()

function resolve_repo_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(repo_root(), path))
end

function resolve_hdf5_catalog_from_directory(dir::AbstractString)
    candidates = filter(readdir(dir; join=true)) do entry
        isfile(entry) && lowercase(splitext(entry)[2]) in (".h5", ".hdf5")
    end
    isempty(candidates) && error("halfdome_path=$(dir) is a directory, but it contains no HDF5 catalog.")

    for preferred in ("lightcone_100.hdf5", "lightcone_100.h5", "halos.hdf5", "halos.h5")
        matches = filter(entry -> lowercase(basename(entry)) == preferred, candidates)
        length(matches) == 1 && return only(matches)
    end

    length(candidates) == 1 && return only(candidates)
    error("halfdome_path=$(dir) contains multiple HDF5 files. Pass the exact catalog file.")
end

function resolve_halfdome_catalog_path(path::AbstractString)
    isempty(path) && error("halfdome_path cannot be empty.")

    trial_paths = String[]
    if isabspath(path)
        push!(trial_paths, String(path))
    else
        push!(trial_paths, normpath(joinpath(repo_root(), path)))
        push!(trial_paths, normpath(joinpath(code_root(), path)))
        push!(trial_paths, normpath(joinpath(pwd(), path)))
        push!(trial_paths, normpath(joinpath(repo_root(), "halfdome", path)))
        push!(trial_paths, normpath(joinpath(repo_root(), "HalfDome", path)))
    end

    unique_trial_paths = unique(trial_paths)
    for candidate in unique_trial_paths
        isfile(candidate) && return candidate
        isdir(candidate) && return resolve_hdf5_catalog_from_directory(candidate)
    end

    error("Could not find HalfDome catalog $(repr(path)). Tried: $(join(unique_trial_paths, ", "))")
end

function make_chi_and_z_of_chi_itp(; omegam=OMEGAM, h_value=H_VALUE, z1=0.0, z2=6.0, nz=100_000)
    local_h0 = 100.0 * h_value
    h_of_z(z) = local_h0 * sqrt(omegam * (1.0 + z)^3 + 1.0 - omegam)
    dchidz(z) = C_KMS / h_of_z(z)

    z_grid = collect(range(z1, z2; length=nz))
    dz = z_grid[2] - z_grid[1]
    chi_grid = similar(z_grid)
    chi_grid[1] = 0.0

    chi_sum = 0.0
    @inbounds for i in 2:length(z_grid)
        chi_sum += 0.5 * (dchidz(z_grid[i - 1]) + dchidz(z_grid[i])) * dz
        chi_grid[i] = chi_sum
    end

    chi_of_z_itp = linear_interpolation(z_grid, chi_grid; extrapolation_bc=Line())
    z_of_chi_itp = linear_interpolation(chi_grid, z_grid; extrapolation_bc=Line())
    return chi_of_z_itp, z_of_chi_itp
end

function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = OMEGAM * one_plus_z^3
    omegam_z = ez_num / (ez_num + 1.0 - OMEGAM)
    return m200m * omegam_z^0.35
end

function compute_redshift_and_mass(x, y, z, radius, z_of_chi_itp, rho_m)
    n = length(x)
    redshift = Vector{Float64}(undef, n)
    halo_mass = Vector{Float64}(undef, n)
    mass_prefactor = (4.0 * pi / 3.0) * rho_m

    @threads for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        ri = Float64(radius[i])
        chi = sqrt(xi * xi + yi * yi + zi * zi)
        zi_redshift = z_of_chi_itp(chi)
        redshift[i] = zi_redshift
        halo_mass[i] = m200m_to_m200c_scalar(mass_prefactor * ri^3, zi_redshift)
    end

    return redshift, halo_mass
end

function stream_catalog_batches(
    process_batch!::F,
    catalog_source::AbstractString,
    halfdome_path::AbstractString,
    websky_path::AbstractString,
    chunkN::Int,
    z_of_chi_itp,
    rho_m
) where {F}
    if catalog_source == "halfdome"
        return h5open(halfdome_path, "r") do h5
            pos_ds = h5["Position"]
            mass_ds = h5["halo_mass_m200c"]
            redshift_ds = h5["redshift"]
            total_halo_count = size(pos_ds, 2)

            for batch_start in 1:chunkN:total_halo_count
                batch_stop = min(batch_start + chunkN - 1, total_halo_count)
                idx = batch_start:batch_stop
                pos = pos_ds[:, idx]
                x = @view pos[1, :]
                y = @view pos[2, :]
                z = @view pos[3, :]
                halo_mass = Float64.(mass_ds[idx]) ./ H_VALUE
                redshift = Float64.(redshift_ds[idx])
                process_batch!(batch_start, x, y, z, halo_mass, redshift)
            end

            return total_halo_count
        end
    end

    return open(websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        rthmax = read(io, Float32)
        redshiftbox = read(io, Float32)
        println("WebSky header: total_halo_count=$(total_halo_count), RTHmax=$(rthmax), redshiftbox=$(redshiftbox)")

        buf = Matrix{Float32}(undef, 10, chunkN)
        batch_start = 1
        nleft = total_halo_count

        while nleft > 0
            nthis = min(chunkN, nleft)
            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            radius = @view cat[7, :]
            redshift, halo_mass = compute_redshift_and_mass(x, y, z, radius, z_of_chi_itp, rho_m)

            process_batch!(batch_start, x, y, z, halo_mass, redshift)
            batch_start += nthis
            nleft -= nthis
        end

        return total_halo_count
    end
end

function selected_halo_mask(halo_mass, redshift; z_max::Float64, mass_min::Float64, mass_max::Float64)
    keep = isfinite.(redshift) .& isfinite.(halo_mass) .& (redshift .>= 0.0) .& (redshift .<= z_max)
    keep .&= halo_mass .> 0.0
    mass_min > 0.0 && (keep .&= halo_mass .>= mass_min)
    isfinite(mass_max) && (keep .&= halo_mass .< mass_max)
    return keep
end

function xgpaint_paint_function()
    if isdefined(Main, :paint!)
        return getfield(Main, :paint!)
    elseif isdefined(XGPaint, :paint!)
        return getfield(XGPaint, :paint!)
    end
    error("paint! is not available in this Julia/XGPaint environment.")
end

function positions_to_ra_dec(x, y, z)
    n = length(x)
    ras = Vector{Float64}(undef, n)
    decs = Vector{Float64}(undef, n)

    @threads for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        radius = sqrt(xi * xi + yi * yi + zi * zi)
        radius > 0.0 || error("Encountered halo position with zero radius.")
        theta, phi = Healpix.vec2ang(xi / radius, yi / radius, zi / radius)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function paint_full_foreground_map!(
    dm_map,
    workspace,
    dm_model_interp,
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    z_of_chi_itp,
    rho_m;
    z_max::Float64,
    mass_min::Float64,
    mass_max::Float64,
    progress_every::Int,
)
    halos_passing_cuts = 0
    halos_painted = 0
    z_min_seen = Inf
    z_max_seen = -Inf
    mass_min_seen = Inf
    mass_max_seen = -Inf
    paint_fn = xgpaint_paint_function()
    chunk_index = Ref(0)

    total_halo_count = stream_catalog_batches(
        catalog_source,
        halfdome_path,
        websky_path,
        chunkN,
        z_of_chi_itp,
        rho_m
    ) do batch_start, x, y, z, halo_mass, redshift
        chunk_index[] += 1
        keep = selected_halo_mask(halo_mass, redshift; z_max=z_max, mass_min=mass_min, mass_max=mass_max)
        selected_count = count(keep)
        selected_count == 0 && return

        xs = Float64.(x[keep])
        ys = Float64.(y[keep])
        zs = Float64.(z[keep])
        masses = Float64.(halo_mass[keep])
        redshifts = Float64.(redshift[keep])

        halos_passing_cuts += selected_count
        z_min_seen = min(z_min_seen, minimum(redshifts))
        z_max_seen = max(z_max_seen, maximum(redshifts))
        mass_min_seen = min(mass_min_seen, minimum(masses))
        mass_max_seen = max(mass_max_seen, maximum(masses))

        ras, decs = positions_to_ra_dec(xs, ys, zs)
        perm = sortperm(decs)
        paint_fn(
            dm_map,
            workspace,
            dm_model_interp,
            masses[perm],
            redshifts[perm],
            ras[perm],
            decs[perm],
        )

        halos_painted += selected_count
        if progress_every > 0 && chunk_index[] % progress_every == 0
            println("  foreground chunk $(chunk_index[]): catalog row start $(batch_start), painted so far $(halos_painted)")
            flush(stdout)
        end
    end

    halos_painted > 0 || error("No halos passed the requested foreground cuts for the full foreground map.")
    return (
        total_halo_count=total_halo_count,
        halos_passing_cuts=halos_passing_cuts,
        halos_painted=halos_painted,
        z_min=z_min_seen,
        z_max=z_max_seen,
        mass_min=mass_min_seen,
        mass_max=mass_max_seen,
    )
end

function collect_selected_halo_limits(
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    z_of_chi_itp,
    rho_m;
    z_max,
    mass_min,
    mass_max
)
    selected_count = Ref(0)
    z_min_seen = Ref(Inf)
    z_max_seen = Ref(-Inf)
    mass_min_seen = Ref(Inf)
    mass_max_seen = Ref(-Inf)

    total_halo_count = stream_catalog_batches(
        catalog_source,
        halfdome_path,
        websky_path,
        chunkN,
        z_of_chi_itp,
        rho_m
    ) do _batch_start, _x, _y, _z, halo_mass, redshift
        keep = selected_halo_mask(halo_mass, redshift; z_max=z_max, mass_min=mass_min, mass_max=mass_max)
        any(keep) || return
        selected_redshift = redshift[keep]
        selected_mass = halo_mass[keep]
        selected_count[] += length(selected_redshift)
        z_min_seen[] = min(z_min_seen[], minimum(selected_redshift))
        z_max_seen[] = max(z_max_seen[], maximum(selected_redshift))
        mass_min_seen[] = min(mass_min_seen[], minimum(selected_mass))
        mass_max_seen[] = max(mass_max_seen[], maximum(selected_mass))
    end

    selected_count[] > 0 || error("No halos passed the foreground cuts 0 <= z <= $(z_max).")
    return (
        total_halo_count=total_halo_count,
        selected_count=selected_count[],
        z_min=z_min_seen[],
        z_max=z_max_seen[],
        mass_min=mass_min_seen[],
        mass_max=mass_max_seen[]
    )
end

function make_log_edges(min_value::Float64, max_value::Float64, nbins::Int)
    min_value > 0.0 || error("Log histogram lower limit must be positive.")
    max_value > 0.0 || error("Log histogram upper limit must be positive.")
    log_min = log10(min_value)
    log_max = log10(max_value)
    if log_min == log_max
        log_min -= 0.5
        log_max += 0.5
    end
    return 10 .^ range(log_min, log_max; length=nbins + 1)
end

function bin_index(value::Real, edges::AbstractVector{<:Real})
    value_f = Float64(value)
    if !isfinite(value_f) || value_f < Float64(first(edges)) || value_f > Float64(last(edges))
        return 0
    end
    idx = searchsortedlast(edges, value_f)
    idx == 0 && return 0
    return min(idx, length(edges) - 1)
end

function update_histograms!(redshift_counts, mass_counts, z_mass_counts, redshifts, masses, redshift_edges, mass_edges)
    @inbounds for i in eachindex(redshifts)
        zidx = bin_index(redshifts[i], redshift_edges)
        midx = bin_index(masses[i], mass_edges)
        zidx == 0 && continue
        midx == 0 && continue
        redshift_counts[zidx] += 1
        mass_counts[midx] += 1
        z_mass_counts[zidx, midx] += 1
    end
    return nothing
end

function accumulate_halo_center_maps!(count_hp, mass_hp, x, y, z, masses)
    res = count_hp.resolution
    @inbounds for i in eachindex(masses)
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        r = sqrt(xi * xi + yi * yi + zi * zi)
        theta, phi = Healpix.vec2ang(xi / r, yi / r, zi / r)
        pix = Healpix.ang2pixRing(res, theta, phi)
        count_hp.pixels[pix] += 1.0
        mass_hp.pixels[pix] += Float64(masses[i])
    end
    return nothing
end

function compute_theta_min_local(model)
    if hasproperty(model, :itp)
        itp = getproperty(model, :itp)
        if hasproperty(itp, :ranges)
            return exp(Float64(first(first(getproperty(itp, :ranges)))))
        end
    end
    return eps(Float64)
end

function pixel_centers_to_ra_dec(res, pixels)
    ras = Vector{Float64}(undef, length(pixels))
    decs = Vector{Float64}(undef, length(pixels))

    @threads for i in eachindex(pixels)
        vx, vy, vz = Healpix.pix2vecRing(res, pixels[i])
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function ra_dec_to_unit_vectors(ras, decs)
    ux = Vector{Float64}(undef, length(ras))
    uy = Vector{Float64}(undef, length(ras))
    uz = Vector{Float64}(undef, length(ras))

    @threads for i in eachindex(ras)
        ra = Float64(ras[i])
        dec = Float64(decs[i])
        cosdec = cos(dec)
        ux[i] = cosdec * cos(ra)
        uy[i] = cosdec * sin(ra)
        uz[i] = sin(dec)
    end

    return ux, uy, uz
end

function build_frb_pixel_lookup(frb_pixels)
    order = sortperm(frb_pixels)
    return Int.(frb_pixels[order]), Int.(order)
end

function add_if_frb_pixel!(
    local_dm,
    local_hits::Base.RefValue{Int},
    sorted_frb_pixels,
    sorted_frb_indices,
    global_pix::Int,
    halo_ux::Float64,
    halo_uy::Float64,
    halo_uz::Float64,
    frb_ux,
    frb_uy,
    frb_uz,
    theta_min::Float64,
    theta_max::Float64,
    mass::Float64,
    redshift::Float64,
    dm_model_interp
)
    frb_range = searchsorted(sorted_frb_pixels, global_pix)
    isempty(frb_range) && return nothing

    @inbounds for lookup_idx in frb_range
        frb_idx = sorted_frb_indices[lookup_idx]
        cosang = clamp(halo_ux * frb_ux[frb_idx] + halo_uy * frb_uy[frb_idx] + halo_uz * frb_uz[frb_idx], -1.0, 1.0)
        theta = acos(cosang)
        if theta <= theta_max
            local_dm[frb_idx] += Float64(dm_model_interp(max(theta, theta_min), mass, redshift))
            local_hits[] += 1
        end
    end
    return nothing
end

function accumulate_frb_dm_from_halo_batch!(
    frb_dm,
    workspace,
    dm_model_interp,
    sorted_frb_pixels,
    sorted_frb_indices,
    frb_ux,
    frb_uy,
    frb_uz,
    x,
    y,
    z,
    masses,
    redshifts
)
    isempty(masses) && return 0

    theta_min = compute_theta_min_local(dm_model_interp)
    nfrb = length(frb_dm)
    nthreads_capacity = thread_capacity()
    thread_dm = [zeros(Float64, nfrb) for _ in 1:nthreads_capacity]
    thread_hits = zeros(Int, nthreads_capacity)

    @threads for i in eachindex(masses)
        tid = Threads.threadid()
        local_dm = thread_dm[tid]
        local_hits = Ref(0)

        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        mass_i = Float64(masses[i])
        redshift_i = Float64(redshifts[i])
        radius = sqrt(xi * xi + yi * yi + zi * zi)
        halo_ux = xi / radius
        halo_uy = yi / radius
        halo_uz = zi / radius
        center_theta, center_phi = Healpix.vec2ang(halo_ux, halo_uy, halo_uz)

        theta_max = Float64(compute_theta_max_local(dm_model_interp, mass_i * XGPaint.M_sun, redshift_i))
        if !isfinite(theta_max) || theta_max <= 0.0
            continue
        end

        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, theta_max)
        for ring_idx in ring_start:ring_stop
            range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring_idx, center_theta, center_phi, theta_max)
            first_pixel = workspace.ring_first_pixels[ring_idx]
            for local_pix_idx in range1
                add_if_frb_pixel!(
                    local_dm,
                    local_hits,
                    sorted_frb_pixels,
                    sorted_frb_indices,
                    first_pixel + local_pix_idx - 1,
                    halo_ux,
                    halo_uy,
                    halo_uz,
                    frb_ux,
                    frb_uy,
                    frb_uz,
                    theta_min,
                    theta_max,
                    mass_i,
                    redshift_i,
                    dm_model_interp
                )
            end
            for local_pix_idx in range2
                add_if_frb_pixel!(
                    local_dm,
                    local_hits,
                    sorted_frb_pixels,
                    sorted_frb_indices,
                    first_pixel + local_pix_idx - 1,
                    halo_ux,
                    halo_uy,
                    halo_uz,
                    frb_ux,
                    frb_uy,
                    frb_uz,
                    theta_min,
                    theta_max,
                    mass_i,
                    redshift_i,
                    dm_model_interp
                )
            end
        end

        thread_hits[tid] += local_hits[]
    end

    for local_dm in thread_dm
        frb_dm .+= local_dm
    end
    return sum(thread_hits)
end

function build_random_frb_dm_map(nside, frb_pixels, frb_dm_values; overlap_mode::AbstractString="mean")
    mode = lowercase(strip(String(overlap_mode)))
    mode in ("mean", "sum", "last") || error("frb_overlap_mode must be mean, sum, or last.")

    frb_dm_hp = HealpixMap{Float64, RingOrder}(nside)
    fill!(frb_dm_hp.pixels, 0.0)

    if mode == "mean"
        pixel_counts = Dict{Int, Int}()
        @inbounds for i in eachindex(frb_pixels)
            pix = Int(frb_pixels[i])
            frb_dm_hp.pixels[pix] += frb_dm_values[i]
            pixel_counts[pix] = get(pixel_counts, pix, 0) + 1
        end
        for (pix, count) in pixel_counts
            frb_dm_hp.pixels[pix] /= count
        end
    elseif mode == "sum"
        @inbounds for i in eachindex(frb_pixels)
            frb_dm_hp.pixels[frb_pixels[i]] += frb_dm_values[i]
        end
    else
        @inbounds for i in eachindex(frb_pixels)
            frb_dm_hp.pixels[frb_pixels[i]] = frb_dm_values[i]
        end
    end

    return frb_dm_hp
end

function draw_random_frb_pixels(rng, npix::Int, frb_count::Int; unique_pixels::Bool=true)
    frb_count > 0 || error("frb_count must be positive.")
    npix > 0 || error("npix must be positive.")

    if !unique_pixels
        return rand(rng, 1:npix, frb_count)
    end

    frb_count <= npix || error("Cannot draw $(frb_count) unique FRB pixels from npix=$(npix).")

    pixels = Vector{Int}(undef, frb_count)
    seen = Set{Int}()
    i = 1
    while i <= frb_count
        pix = rand(rng, 1:npix)
        pix in seen && continue
        push!(seen, pix)
        pixels[i] = pix
        i += 1
    end

    return pixels
end

function save_frb_source_catalog(path, frb_pixels, frb_ra, frb_dec, frb_redshift::Float64, frb_dm)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "frb_index,pixel_ring,ra_rad,dec_rad,z_source,dm_pc_cm3")
        @inbounds for i in eachindex(frb_pixels)
            println(io, "$(i),$(frb_pixels[i]),$(frb_ra[i]),$(frb_dec[i]),$(frb_redshift),$(frb_dm[i])")
        end
    end

    return path
end

function save_run_summary(
    path;
    catalog_source,
    catalog_path,
    output_dir,
    dm_cache_file,
    nside,
    chunkN,
    frb_count,
    unique_frb_pixels,
    frb_seed,
    frb_redshift,
    z_max_halos,
    halo_mass_min,
    halo_mass_max,
    processed_halo_count,
    los_intersection_count,
    frb_pixels,
    frb_dm,
    map_pixels,
    save_foreground_map,
    foreground_map_path,
    foreground_map_pixels,
    foreground_paint_counters,
    save_power_spectrum,
    cl_lmax,
    cl_niter,
    subtract_cl_mean,
    cl_table_path,
    cl_plot_path,
    save_frb_corrected_estimator,
    frb_corrected_lmax,
    frb_corrected_subtract_sample_mean,
    frb_corrected_shot_noise,
    frb_corrected_n_shuffle,
    frb_corrected_seed,
    frb_corrected_result,
    frb_corrected_table_path,
    frb_corrected_plot_path,
    frb_corrected_map_path,
)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "Random fixed-redshift FRB DM map")
        println(io, "catalog_source=$(catalog_source)")
        println(io, "catalog_path=$(catalog_path)")
        println(io, "output_dir=$(output_dir)")
        println(io, "dm_cache_file=$(dm_cache_file)")
        println(io, "nside=$(nside)")
        println(io, "chunkN=$(chunkN)")
        println(io, "frb_count=$(frb_count)")
        println(io, "unique_frb_pixels=$(unique_frb_pixels)")
        println(io, "unique_pixel_count=$(length(unique(frb_pixels)))")
        println(io, "frb_seed=$(frb_seed)")
        println(io, "frb_redshift=$(frb_redshift)")
        println(io, "z_max_halos=$(z_max_halos)")
        println(io, "halo_mass_min=$(halo_mass_min)")
        println(io, "halo_mass_max=$(halo_mass_max)")
        println(io, "processed_halo_count=$(processed_halo_count)")
        println(io, "los_intersection_count=$(los_intersection_count)")
        println(io, "frb_dm_min=$(minimum(frb_dm))")
        println(io, "frb_dm_max=$(maximum(frb_dm))")
        println(io, "frb_dm_mean=$(sum(frb_dm) / length(frb_dm))")
        println(io, "map_nonzero_pixels=$(count(!=(0.0), map_pixels))")
        println(io, "map_min=$(minimum(map_pixels))")
        println(io, "map_max=$(maximum(map_pixels))")
        println(io)
        println(io, "save_foreground_map=$(save_foreground_map)")
        println(io, "foreground_map_path=$(foreground_map_path)")
        if foreground_map_pixels !== nothing
            println(io, "foreground_map_nonzero_pixels=$(count(!=(0.0), foreground_map_pixels))")
            println(io, "foreground_map_min=$(minimum(foreground_map_pixels))")
            println(io, "foreground_map_max=$(maximum(foreground_map_pixels))")
            println(io, "foreground_map_mean=$(mean(foreground_map_pixels))")
        end
        if foreground_paint_counters !== nothing
            println(io, "foreground_map_total_halo_count=$(foreground_paint_counters.total_halo_count)")
            println(io, "foreground_map_halos_passing_cuts=$(foreground_paint_counters.halos_passing_cuts)")
            println(io, "foreground_map_halos_painted=$(foreground_paint_counters.halos_painted)")
            println(io, "foreground_map_redshift_min=$(foreground_paint_counters.z_min)")
            println(io, "foreground_map_redshift_max=$(foreground_paint_counters.z_max)")
            println(io, "foreground_map_mass_min=$(foreground_paint_counters.mass_min)")
            println(io, "foreground_map_mass_max=$(foreground_paint_counters.mass_max)")
        end
        println(io)
        println(io, "save_power_spectrum=$(save_power_spectrum)")
        println(io, "power_spectrum_input=continuous_foreground_dm_map")
        println(io, "cl_lmax=$(cl_lmax)")
        println(io, "cl_niter=$(cl_niter)")
        println(io, "subtract_cl_mean=$(subtract_cl_mean)")
        println(io, "cl_table_path=$(cl_table_path)")
        println(io, "cl_plot_path=$(cl_plot_path)")
        println(io)
        println(io, "save_frb_corrected_estimator=$(save_frb_corrected_estimator)")
        println(io, "frb_corrected_input=sparse_frb_los_dm_samples")
        println(io, "frb_corrected_lmax=$(frb_corrected_lmax)")
        println(io, "frb_corrected_subtract_sample_mean=$(frb_corrected_subtract_sample_mean)")
        println(io, "frb_corrected_shot_noise=$(frb_corrected_shot_noise)")
        println(io, "frb_corrected_n_shuffle=$(frb_corrected_n_shuffle)")
        println(io, "frb_corrected_seed=$(frb_corrected_seed)")
        println(io, "frb_corrected_table_path=$(frb_corrected_table_path)")
        println(io, "frb_corrected_plot_path=$(frb_corrected_plot_path)")
        println(io, "frb_corrected_map_path=$(frb_corrected_map_path)")
        if frb_corrected_result !== nothing
            println(io, "frb_corrected_nfrb=$(frb_corrected_result.nfrb)")
            println(io, "frb_corrected_nbar_sr=$(frb_corrected_result.nbar_sr)")
            println(io, "frb_corrected_q_mean=$(frb_corrected_result.q_mean)")
            println(io, "frb_corrected_q_var=$(frb_corrected_result.q_var)")
        end
    end

    return path
end

function write_cl_table(path, cl_values)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "ell,C_ell,D_ell")
        @inbounds for i in eachindex(cl_values)
            ell = i - 1
            cl = Float64(cl_values[i])
            dl = ell * (ell + 1) * cl / (2.0 * pi)
            println(io, "$(ell),$(cl),$(dl)")
        end
    end

    return path
end

function save_power_spectrum_plot(path, cl_values)
    ell = collect(0:length(cl_values)-1)
    dl = ell .* (ell .+ 1) .* Float64.(cl_values) ./ (2.0 * pi)
    keep = (ell .>= 2) .& isfinite.(dl) .& (dl .> 0.0)
    count(keep) > 0 || error("No positive finite D_ell values available for a log-log power-spectrum plot.")

    p = plot(
        ell[keep],
        dl[keep],
        xscale=:log10,
        yscale=:log10,
        xlabel="ell",
        ylabel="D_ell = ell(ell+1)C_ell/2pi",
        label="continuous foreground DM",
        linewidth=2,
        size=(760, 520),
        title="Foreground DM power spectrum",
    )
    savefig(p, path)
    return path
end

function compute_and_save_power_spectrum(
    map::HealpixMap{Float64, RingOrder},
    nside::Int;
    table_path::AbstractString,
    plot_path::AbstractString,
    lmax::Int,
    niter::Int,
    subtract_mean::Bool,
)
    cl_map = HealpixMap{Float64, RingOrder}(nside)
    cl_map.pixels .= map.pixels
    if subtract_mean
        cl_map.pixels .-= mean(cl_map.pixels)
    end

    cl_values =
        lmax < 0 ?
        Healpix.anafast(cl_map; niter=niter) :
        Healpix.anafast(cl_map; lmax=lmax, niter=niter)

    write_cl_table(table_path, cl_values)
    save_power_spectrum_plot(plot_path, cl_values)
    return cl_values
end

function build_frb_sparse_estimator_map(
    frb_pixels,
    frb_dm,
    nside::Int;
    subtract_sample_mean::Bool,
)
    npix = 12 * nside^2
    valid = BitVector(undef, length(frb_dm))
    @inbounds for i in eachindex(frb_dm)
        pix = Int(frb_pixels[i])
        dm = Float64(frb_dm[i])
        valid[i] = isfinite(dm) && pix >= 1 && pix <= npix
    end

    nfrb = count(valid)
    nfrb > 0 || error("FRB sparse estimator has no finite valid FRB DM samples.")

    q = Vector{Float64}(undef, nfrb)
    valid_pixels = Vector{Int}(undef, nfrb)
    j = 1
    @inbounds for i in eachindex(frb_dm)
        if valid[i]
            valid_pixels[j] = Int(frb_pixels[i])
            q[j] = Float64(frb_dm[i])
            j += 1
        end
    end

    if subtract_sample_mean
        q .-= mean(q)
    end

    estimator_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(estimator_map.pixels, 0.0)
    scale = npix / nfrb
    @inbounds for i in eachindex(q)
        estimator_map.pixels[valid_pixels[i]] += q[i] * scale
    end

    return estimator_map, valid_pixels, q
end

function write_frb_corrected_cl_table(path, cl_obs, cl_shot, cl_corr)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "ell,cl_obs,cl_shot,cl_corr,dl_obs,dl_corr")
        @inbounds for i in eachindex(cl_obs)
            ell = i - 1
            obs = Float64(cl_obs[i])
            shot = Float64(cl_shot[i])
            corr = Float64(cl_corr[i])
            dl_obs = ell * (ell + 1) * obs / (2.0 * pi)
            dl_corr = ell * (ell + 1) * corr / (2.0 * pi)
            println(io, "$(ell),$(obs),$(shot),$(corr),$(dl_obs),$(dl_corr)")
        end
    end

    return path
end

function save_frb_corrected_cl_plot(path, cl_obs, cl_corr)
    ell = collect(0:length(cl_obs)-1)
    dl_obs = ell .* (ell .+ 1) .* Float64.(cl_obs) ./ (2.0 * pi)
    dl_corr = ell .* (ell .+ 1) .* Float64.(cl_corr) ./ (2.0 * pi)
    keep_obs = (ell .>= 2) .& isfinite.(dl_obs) .& (dl_obs .> 0.0)
    keep_corr = (ell .>= 2) .& isfinite.(dl_corr) .& (dl_corr .> 0.0)
    (count(keep_obs) > 0 || count(keep_corr) > 0) ||
        error("No positive finite D_ell values available for FRB corrected estimator plot.")

    p = plot(
        xlabel="ell",
        ylabel="D_ell = ell(ell+1)C_ell/2pi",
        xscale=:log10,
        yscale=:log10,
        size=(760, 520),
        title="FRB LOS DM sparse estimator",
    )
    count(keep_obs) > 0 && plot!(p, ell[keep_obs], dl_obs[keep_obs], label="observed", linewidth=2)
    count(keep_corr) > 0 && plot!(p, ell[keep_corr], dl_corr[keep_corr], label="shot-noise corrected", linewidth=2)
    savefig(p, path)
    return path
end

function compute_and_save_frb_corrected_estimator(
    frb_pixels,
    frb_dm,
    nside::Int;
    table_path::AbstractString,
    plot_path::AbstractString,
    map_path::AbstractString,
    lmax::Int,
    subtract_sample_mean::Bool,
    shot_noise::AbstractString,
    n_shuffle::Int,
    seed::Int,
)
    shot_mode = lowercase(strip(String(shot_noise)))
    shot_mode in ("analytic", "shuffle", "none") ||
        error("frb_corrected_shot_noise must be analytic, shuffle, or none.")
    n_shuffle >= 0 || error("frb_corrected_n_shuffle must be non-negative.")

    estimator_map, valid_pixels, q = build_frb_sparse_estimator_map(
        frb_pixels,
        frb_dm,
        nside;
        subtract_sample_mean=subtract_sample_mean,
    )
    Healpix.saveToFITS(estimator_map, "!" * map_path, typechar="D")

    cl_obs =
        lmax < 0 ?
        Healpix.anafast(estimator_map; niter=0) :
        Healpix.anafast(estimator_map; lmax=lmax, niter=0)

    nfrb = length(q)
    nbar = nfrb / (4.0 * pi)
    cl_shot = zeros(Float64, length(cl_obs))

    if shot_mode == "analytic"
        cl_shot .= mean(q .^ 2) / nbar
    elseif shot_mode == "shuffle"
        n_shuffle > 0 || error("frb_corrected_n_shuffle must be positive when shot noise is shuffle.")
        rng = MersenneTwister(seed)
        npix = 12 * nside^2
        scale = npix / nfrb
        cl_sum = zeros(Float64, length(cl_obs))
        for _ in 1:n_shuffle
            q_shuf = q[randperm(rng, nfrb)]
            shuffle_map = HealpixMap{Float64, RingOrder}(nside)
            fill!(shuffle_map.pixels, 0.0)
            @inbounds for i in eachindex(q_shuf)
                shuffle_map.pixels[valid_pixels[i]] += q_shuf[i] * scale
            end
            cl_tmp =
                lmax < 0 ?
                Healpix.anafast(shuffle_map; niter=0) :
                Healpix.anafast(shuffle_map; lmax=lmax, niter=0)
            cl_sum .+= Float64.(cl_tmp)
        end
        cl_shot .= cl_sum ./ n_shuffle
    end

    cl_corr = Float64.(cl_obs) .- cl_shot
    length(cl_corr) >= 2 && (cl_corr[1:2] .= NaN)

    write_frb_corrected_cl_table(table_path, cl_obs, cl_shot, cl_corr)
    save_frb_corrected_cl_plot(plot_path, cl_obs, cl_corr)

    return (
        nfrb=nfrb,
        nbar_sr=nbar,
        q_mean=mean(q),
        q_var=mean(q .^ 2),
        map_path=map_path,
        table_path=table_path,
        plot_path=plot_path,
    )
end

catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env=("FRB_CATALOG_SOURCE", "FRB_Z1_CATALOG_SOURCE")))
catalog_source in ("halfdome", "websky") || error("catalog_source must be \"halfdome\" or \"websky\".")

halfdome_path = resolve_halfdome_catalog_path(get_string_arg("halfdome_path", "lightcone_100.hdf5"; env=("FRB_HALFDOME_PATH", "FRB_Z1_HALFDOME_PATH")))
websky_path = resolve_repo_path(get_string_arg("websky_path", "other_sims/sims/halos.pksc"; env=("FRB_WEBSKY_PATH", "FRB_Z1_WEBSKY_PATH")))
output_dir = resolve_repo_path(get_string_arg("output_dir", joinpath("batched_data", "frb_random"); env=("FRB_OUTPUT_DIR", "FRB_Z1_OUTPUT_DIR")))
default_dm_cache_file = joinpath(output_dir, "random_frb_xgpaint_dm_cache.jld2")
dm_cache_file = resolve_repo_path(get_string_arg("dm_cache_file", default_dm_cache_file; env=("FRB_DM_CACHE_FILE", "FRB_Z1_DM_CACHE_FILE")))

nside = get_int_arg("nside", 4096; env=("FRB_NSIDE", "FRB_Z1_NSIDE"))
chunkN = get_int_arg("chunkN", 1_000_000; env=("FRB_CHUNKN", "FRB_Z1_CHUNKN"))
frb_count_default = get_int_arg("frb_count", 10_000; env=("FRB_COUNT", "FRB_Z1_COUNT"))
frb_count = get_int_arg("N", frb_count_default; env=("FRB_N", "FRB_Z1_N"))
frb_seed = get_int_arg("frb_seed", 42; env=("FRB_SEED", "FRB_Z1_SEED"))
frb_redshift_default = get_float_arg("frb_redshift", DEFAULT_FRB_REDSHIFT; env=("FRB_REDSHIFT", "FRB_Z1_REDSHIFT"))
frb_redshift_source_default = get_float_arg("source_redshift", frb_redshift_default; env=("FRB_SOURCE_REDSHIFT", "FRB_Z1_SOURCE_REDSHIFT"))
frb_redshift = get_float_arg("z_source", frb_redshift_source_default; env=("FRB_Z_SOURCE", "FRB_Z1_Z_SOURCE"))
z_max_halos = get_float_arg("z_max_halos", frb_redshift; env=("FRB_HALO_Z_MAX", "FRB_Z1_HALO_Z_MAX"))
halo_mass_min = get_float_arg("halo_mass_min", 0.0; env=("FRB_HALO_MASS_MIN", "FRB_Z1_HALO_MASS_MIN"))
halo_mass_max = get_float_arg("halo_mass_max", Inf; env=("FRB_HALO_MASS_MAX", "FRB_Z1_HALO_MASS_MAX"))
frb_overlap_mode = lowercase(get_string_arg("frb_overlap_mode", "mean"; env=("FRB_OVERLAP_MODE", "FRB_Z1_OVERLAP_MODE")))
unique_frb_pixels = get_bool_arg("unique_frb_pixels", true; env=("FRB_UNIQUE_PIXELS", "FRB_Z1_UNIQUE_PIXELS"))
save_frb_catalog = get_bool_arg("save_frb_catalog", true; env=("FRB_SAVE_CATALOG", "FRB_Z1_SAVE_CATALOG"))
dm_cache_overwrite = get_bool_arg("dm_cache_overwrite", false; env=("FRB_DM_CACHE_OVERWRITE", "FRB_Z1_DM_CACHE_OVERWRITE"))
dm_value_sanity_max = get_float_arg("dm_value_sanity_max", 1.0e8; env=("FRB_DM_VALUE_SANITY_MAX", "FRB_Z1_DM_VALUE_SANITY_MAX"))
dm_cleanup_nonpositive = get_bool_arg("dm_cleanup_nonpositive", true; env=("FRB_DM_CLEANUP_NONPOSITIVE", "FRB_Z1_DM_CLEANUP_NONPOSITIVE"))
save_foreground_map = get_bool_arg("save_foreground_map", false; env=("FRB_SAVE_FOREGROUND_MAP", "FRB_Z1_SAVE_FOREGROUND_MAP"))
foreground_progress_every = get_int_arg("foreground_progress_every", 5; env=("FRB_FOREGROUND_PROGRESS_EVERY", "FRB_Z1_FOREGROUND_PROGRESS_EVERY"))
save_power_spectrum = get_bool_arg("save_power_spectrum", false; env=("FRB_SAVE_POWER_SPECTRUM", "FRB_Z1_SAVE_POWER_SPECTRUM"))
cl_lmax = get_int_arg("cl_lmax", 3 * nside - 1; env=("FRB_CL_LMAX", "FRB_Z1_CL_LMAX"))
cl_niter = get_int_arg("cl_niter", 0; env=("FRB_CL_NITER", "FRB_Z1_CL_NITER"))
subtract_cl_mean = get_bool_arg("subtract_cl_mean", true; env=("FRB_SUBTRACT_CL_MEAN", "FRB_Z1_SUBTRACT_CL_MEAN"))
save_frb_corrected_estimator = get_bool_arg("save_frb_corrected_estimator", save_power_spectrum; env=("FRB_SAVE_CORRECTED_ESTIMATOR", "FRB_Z1_SAVE_CORRECTED_ESTIMATOR"))
frb_corrected_lmax = get_int_arg("frb_corrected_lmax", cl_lmax; env=("FRB_CORRECTED_LMAX", "FRB_Z1_CORRECTED_LMAX"))
frb_corrected_subtract_sample_mean = get_bool_arg("frb_corrected_subtract_sample_mean", true; env=("FRB_CORRECTED_SUBTRACT_SAMPLE_MEAN", "FRB_Z1_CORRECTED_SUBTRACT_SAMPLE_MEAN"))
frb_corrected_shot_noise = lowercase(get_string_arg("frb_corrected_shot_noise", "shuffle"; env=("FRB_CORRECTED_SHOT_NOISE", "FRB_Z1_CORRECTED_SHOT_NOISE")))
frb_corrected_n_shuffle = get_int_arg("frb_corrected_n_shuffle", 5; env=("FRB_CORRECTED_N_SHUFFLE", "FRB_Z1_CORRECTED_N_SHUFFLE"))
frb_corrected_seed = get_int_arg("frb_corrected_seed", frb_seed; env=("FRB_CORRECTED_SEED", "FRB_Z1_CORRECTED_SEED"))

nside > 0 || error("nside must be positive.")
chunkN > 0 || error("chunkN must be positive.")
frb_count > 0 || error("frb_count must be positive.")
frb_redshift > 0.0 || error("frb_redshift must be positive.")
z_max_halos > 0.0 || error("z_max_halos must be positive.")
z_max_halos <= frb_redshift || error("z_max_halos=$(z_max_halos) must not exceed frb_redshift=$(frb_redshift).")
halo_mass_min >= 0.0 || error("halo_mass_min must be nonnegative.")
halo_mass_max > halo_mass_min || error("halo_mass_max must be greater than halo_mass_min.")
frb_overlap_mode in ("mean", "sum", "last") || error("frb_overlap_mode must be mean, sum, or last.")
foreground_progress_every >= 0 || error("foreground_progress_every must be non-negative.")
cl_lmax == -1 || cl_lmax >= 2 || error("cl_lmax must be -1 for Healpix default or >= 2.")
cl_niter >= 0 || error("cl_niter must be non-negative.")
frb_corrected_lmax == -1 || frb_corrected_lmax >= 2 || error("frb_corrected_lmax must be -1 for Healpix default or >= 2.")
frb_corrected_shot_noise in ("analytic", "shuffle", "none") || error("frb_corrected_shot_noise must be analytic, shuffle, or none.")
frb_corrected_n_shuffle >= 0 || error("frb_corrected_n_shuffle must be non-negative.")
frb_corrected_shot_noise == "shuffle" && frb_corrected_n_shuffle == 0 &&
    error("frb_corrected_n_shuffle must be positive when frb_corrected_shot_noise=shuffle.")
isdir(output_dir) || mkpath(output_dir)

catalog_path = catalog_source == "halfdome" ? halfdome_path : websky_path
base_tag = "$(catalog_source)_zfrb$(fmt_param_value(frb_redshift))_zhalomax$(fmt_param_value(z_max_halos))_nside$(nside)_nfrb$(frb_count)_seed$(frb_seed)"
foreground_map_path = joinpath(output_dir, "$(base_tag)_foreground_dm_map.fits")
cl_table_path = joinpath(output_dir, "$(base_tag)_foreground_dm_power_spectrum.csv")
cl_plot_path = joinpath(output_dir, "$(base_tag)_foreground_dm_power_spectrum_loglog.png")
frb_corrected_table_path = joinpath(output_dir, "$(base_tag)_frb_corrected_estimator_power_spectrum.csv")
frb_corrected_plot_path = joinpath(output_dir, "$(base_tag)_frb_corrected_estimator_power_spectrum_loglog.png")
frb_corrected_map_path = joinpath(output_dir, "$(base_tag)_frb_corrected_estimator_map.fits")

println("Random fixed-redshift FRB DM configuration:")
println("  catalog_source=$(catalog_source)")
println("  catalog_path=$(catalog_path)")
println("  output_dir=$(output_dir)")
println("  dm_cache_file=$(dm_cache_file)")
println("  dm_cache_overwrite=$(dm_cache_overwrite)")
println("  dm_cleanup_nonpositive=$(dm_cleanup_nonpositive)")
println("  nside=$(nside), chunkN=$(chunkN)")
println("  frb_count=$(frb_count), frb_seed=$(frb_seed), frb_redshift=$(frb_redshift)")
println("  unique_frb_pixels=$(unique_frb_pixels), save_frb_catalog=$(save_frb_catalog)")
println("  frb_overlap_mode=$(frb_overlap_mode)")
println("  foreground halo cut: 0 <= z <= $(z_max_halos), mass in [$(halo_mass_min), $(halo_mass_max))")
println("  save_foreground_map=$(save_foreground_map), foreground_progress_every=$(foreground_progress_every)")
println("  save_power_spectrum=$(save_power_spectrum), input=continuous foreground map, cl_lmax=$(cl_lmax), cl_niter=$(cl_niter), subtract_cl_mean=$(subtract_cl_mean)")
println("  save_frb_corrected_estimator=$(save_frb_corrected_estimator), input=sparse FRB LOS DM, lmax=$(frb_corrected_lmax), shot_noise=$(frb_corrected_shot_noise), n_shuffle=$(frb_corrected_n_shuffle)")

chi_of_z_itp, z_of_chi_itp = make_chi_and_z_of_chi_itp()
rho_m = 2.775e11 * OMEGAM * H_VALUE^2
res = Healpix.Resolution(nside)

rng = MersenneTwister(frb_seed)
npix = 12 * nside^2
frb_pixels = draw_random_frb_pixels(rng, npix, frb_count; unique_pixels=unique_frb_pixels)
frb_ra, frb_dec = pixel_centers_to_ra_dec(res, frb_pixels)
frb_ux, frb_uy, frb_uz = ra_dec_to_unit_vectors(frb_ra, frb_dec)
sorted_frb_pixels, sorted_frb_indices = build_frb_pixel_lookup(frb_pixels)
frb_dm = zeros(Float64, frb_count)
println("Random FRB source pixels: $(length(unique(frb_pixels))) unique pixels for $(frb_count) sources.")

println("Scanning catalog for selected halo limits...")
halo_limits = collect_selected_halo_limits(
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    z_of_chi_itp,
    rho_m;
    z_max=z_max_halos,
    mass_min=halo_mass_min,
    mass_max=halo_mass_max
)

println("Total halos in catalog: $(halo_limits.total_halo_count)")
println("Selected foreground halos: $(halo_limits.selected_count)")
println("Selected redshift range: [$(halo_limits.z_min), $(halo_limits.z_max)]")
println("Selected mass range: [$(halo_limits.mass_min), $(halo_limits.mass_max)]")

dm_model = HaloDMProfile(BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE))
ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = dm_cleanup_nonpositive ? "true" : "false"
dm_model_interp = build_interpolator(dm_model, cache_file=dm_cache_file, overwrite=dm_cache_overwrite)
workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

processed_halo_count = Ref(0)
los_intersection_count = Ref(0)

println("Accumulating per-FRB DM from foreground halos with z <= $(z_max_halos)...")
stream_catalog_batches(
    catalog_source,
    halfdome_path,
    websky_path,
    chunkN,
    z_of_chi_itp,
    rho_m
) do _batch_start, x, y, z, halo_mass, redshift
    keep = selected_halo_mask(halo_mass, redshift; z_max=z_max_halos, mass_min=halo_mass_min, mass_max=halo_mass_max)
    any(keep) || return

    xs = Float64.(x[keep])
    ys = Float64.(y[keep])
    zs = Float64.(z[keep])
    masses = Float64.(halo_mass[keep])
    redshifts = Float64.(redshift[keep])

    los_intersection_count[] += accumulate_frb_dm_from_halo_batch!(
        frb_dm,
        workspace,
        dm_model_interp,
        sorted_frb_pixels,
        sorted_frb_indices,
        frb_ux,
        frb_uy,
        frb_uz,
        xs,
        ys,
        zs,
        masses,
        redshifts
    )
    processed_halo_count[] += length(masses)
end

max_sampled_dm = maximum(frb_dm)
if !isfinite(max_sampled_dm) || max_sampled_dm > dm_value_sanity_max
    error(
        "Sampled FRB DM maximum $(max_sampled_dm) is not physically plausible. " *
        "Check the XGPaint DM profile cache. Try a known-good cache or rebuild with a safe thread count."
    )
end

println("Processed $(processed_halo_count[]) foreground halos.")
println("Found $(los_intersection_count[]) halo/FRB line-of-sight intersections.")
println("FRB DM summary: min=$(minimum(frb_dm)), max=$(maximum(frb_dm)), mean=$(sum(frb_dm) / length(frb_dm))")

frb_dm_map_path = joinpath(output_dir, "$(base_tag)_frb_dm_map.fits")
frb_catalog_path = joinpath(output_dir, "$(base_tag)_frb_sources.csv")
summary_path = joinpath(output_dir, "$(base_tag)_summary.txt")
frb_dm_hp = build_random_frb_dm_map(nside, frb_pixels, frb_dm; overlap_mode=frb_overlap_mode)
Healpix.saveToFITS(frb_dm_hp, "!" * frb_dm_map_path, typechar="D")
println("Saved FRB DM FITS map:")
println("  $(frb_dm_map_path)")

if save_frb_catalog
    save_frb_source_catalog(frb_catalog_path, frb_pixels, frb_ra, frb_dec, frb_redshift, frb_dm)
    println("Saved FRB source catalog:")
    println("  $(frb_catalog_path)")
end

frb_corrected_result = nothing
if save_frb_corrected_estimator
    println("Computing corrected sparse FRB LOS DM estimator...")
    frb_corrected_result = compute_and_save_frb_corrected_estimator(
        frb_pixels,
        frb_dm,
        nside;
        table_path=frb_corrected_table_path,
        plot_path=frb_corrected_plot_path,
        map_path=frb_corrected_map_path,
        lmax=frb_corrected_lmax,
        subtract_sample_mean=frb_corrected_subtract_sample_mean,
        shot_noise=frb_corrected_shot_noise,
        n_shuffle=frb_corrected_n_shuffle,
        seed=frb_corrected_seed,
    )
    println("Saved corrected FRB estimator map:")
    println("  $(frb_corrected_map_path)")
    println("Saved corrected FRB estimator power-spectrum table:")
    println("  $(frb_corrected_table_path)")
    println("Saved corrected FRB estimator loglog plot:")
    println("  $(frb_corrected_plot_path)")
end

foreground_dm_map = nothing
foreground_paint_counters = nothing
if save_foreground_map || save_power_spectrum
    println("Painting continuous foreground DM map for full-sky products...")
    foreground_dm_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(foreground_dm_map.pixels, 0.0)
    foreground_paint_counters = paint_full_foreground_map!(
        foreground_dm_map,
        workspace,
        dm_model_interp,
        catalog_source,
        halfdome_path,
        websky_path,
        chunkN,
        z_of_chi_itp,
        rho_m;
        z_max=z_max_halos,
        mass_min=halo_mass_min,
        mass_max=halo_mass_max,
        progress_every=foreground_progress_every,
    )
    println(
        "Continuous foreground DM map summary: min=$(minimum(foreground_dm_map.pixels)), " *
        "max=$(maximum(foreground_dm_map.pixels)), mean=$(mean(foreground_dm_map.pixels)), " *
        "nonzero=$(count(!=(0.0), foreground_dm_map.pixels))"
    )

    if save_foreground_map
        println("Saving continuous foreground DM FITS map:")
        println("  $(foreground_map_path)")
        Healpix.saveToFITS(foreground_dm_map, "!" * foreground_map_path, typechar="D")
    end
end

if save_power_spectrum
    foreground_dm_map === nothing && error("save_power_spectrum=true requires a painted foreground DM map.")
    println("Computing power spectrum from continuous foreground DM map...")
    compute_and_save_power_spectrum(
        foreground_dm_map,
        nside;
        table_path=cl_table_path,
        plot_path=cl_plot_path,
        lmax=cl_lmax,
        niter=cl_niter,
        subtract_mean=subtract_cl_mean,
    )
    println("Saved foreground DM power-spectrum table:")
    println("  $(cl_table_path)")
    println("Saved foreground DM power-spectrum loglog plot:")
    println("  $(cl_plot_path)")
end

save_run_summary(
    summary_path;
    catalog_source=catalog_source,
    catalog_path=catalog_path,
    output_dir=output_dir,
    dm_cache_file=dm_cache_file,
    nside=nside,
    chunkN=chunkN,
    frb_count=frb_count,
    unique_frb_pixels=unique_frb_pixels,
    frb_seed=frb_seed,
    frb_redshift=frb_redshift,
    z_max_halos=z_max_halos,
    halo_mass_min=halo_mass_min,
    halo_mass_max=halo_mass_max,
    processed_halo_count=processed_halo_count[],
    los_intersection_count=los_intersection_count[],
    frb_pixels=frb_pixels,
    frb_dm=frb_dm,
    map_pixels=frb_dm_hp.pixels,
    save_foreground_map=save_foreground_map,
    foreground_map_path=foreground_map_path,
    foreground_map_pixels=foreground_dm_map === nothing ? nothing : foreground_dm_map.pixels,
    foreground_paint_counters=foreground_paint_counters,
    save_power_spectrum=save_power_spectrum,
    cl_lmax=cl_lmax,
    cl_niter=cl_niter,
    subtract_cl_mean=subtract_cl_mean,
    cl_table_path=cl_table_path,
    cl_plot_path=cl_plot_path,
    save_frb_corrected_estimator=save_frb_corrected_estimator,
    frb_corrected_lmax=frb_corrected_lmax,
    frb_corrected_subtract_sample_mean=frb_corrected_subtract_sample_mean,
    frb_corrected_shot_noise=frb_corrected_shot_noise,
    frb_corrected_n_shuffle=frb_corrected_n_shuffle,
    frb_corrected_seed=frb_corrected_seed,
    frb_corrected_result=frb_corrected_result,
    frb_corrected_table_path=frb_corrected_table_path,
    frb_corrected_plot_path=frb_corrected_plot_path,
    frb_corrected_map_path=frb_corrected_map_path,
)
println("Saved run summary:")
println("  $(summary_path)")
println("Done.")
