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
using Base.Threads

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

repo_root() = @__DIR__

function resolve_repo_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(repo_root(), path))
end

function resolve_halfdome_catalog_path(path::AbstractString)
    resolved = resolve_repo_path(path)
    isdir(resolved) || return resolved

    candidates = filter(readdir(resolved; join=true)) do entry
        isfile(entry) && lowercase(splitext(entry)[2]) in (".h5", ".hdf5")
    end
    isempty(candidates) && error("halfdome_path=$(resolved) is a directory, but it contains no HDF5 catalog.")

    for preferred in ("lightcone_100.hdf5", "lightcone_100.h5", "halos.hdf5", "halos.h5")
        matches = filter(entry -> lowercase(basename(entry)) == preferred, candidates)
        length(matches) == 1 && return only(matches)
    end

    length(candidates) == 1 && return only(candidates)
    error("halfdome_path=$(resolved) contains multiple HDF5 files. Pass the exact catalog file.")
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

catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env=("FRB_CATALOG_SOURCE", "FRB_Z1_CATALOG_SOURCE")))
catalog_source in ("halfdome", "websky") || error("catalog_source must be \"halfdome\" or \"websky\".")

halfdome_path = resolve_halfdome_catalog_path(get_string_arg("halfdome_path", "lightcone_100.hdf5"; env=("FRB_HALFDOME_PATH", "FRB_Z1_HALFDOME_PATH")))
websky_path = resolve_repo_path(get_string_arg("websky_path", "other_sims/sims/halos.pksc"; env=("FRB_WEBSKY_PATH", "FRB_Z1_WEBSKY_PATH")))
output_dir = resolve_repo_path(get_string_arg("output_dir", joinpath("batched_data", "frb_random"); env=("FRB_OUTPUT_DIR", "FRB_Z1_OUTPUT_DIR")))
default_dm_cache_file = joinpath(output_dir, "random_frb_xgpaint_dm_cache.jld2")
dm_cache_file = resolve_repo_path(get_string_arg("dm_cache_file", default_dm_cache_file; env=("FRB_DM_CACHE_FILE", "FRB_Z1_DM_CACHE_FILE")))

nside = get_int_arg("nside", 4096; env=("FRB_NSIDE", "FRB_Z1_NSIDE"))
chunkN = get_int_arg("chunkN", 1_000_000; env=("FRB_CHUNKN", "FRB_Z1_CHUNKN"))
frb_count = get_int_arg("frb_count", 10_000; env=("FRB_COUNT", "FRB_Z1_COUNT"))
frb_seed = get_int_arg("frb_seed", 42; env=("FRB_SEED", "FRB_Z1_SEED"))
frb_redshift = get_float_arg("frb_redshift", DEFAULT_FRB_REDSHIFT; env=("FRB_REDSHIFT", "FRB_Z1_REDSHIFT"))
z_max_halos = get_float_arg("z_max_halos", frb_redshift; env=("FRB_HALO_Z_MAX", "FRB_Z1_HALO_Z_MAX"))
halo_mass_min = get_float_arg("halo_mass_min", 0.0; env=("FRB_HALO_MASS_MIN", "FRB_Z1_HALO_MASS_MIN"))
halo_mass_max = get_float_arg("halo_mass_max", Inf; env=("FRB_HALO_MASS_MAX", "FRB_Z1_HALO_MASS_MAX"))
frb_overlap_mode = lowercase(get_string_arg("frb_overlap_mode", "mean"; env=("FRB_OVERLAP_MODE", "FRB_Z1_OVERLAP_MODE")))
dm_cache_overwrite = get_bool_arg("dm_cache_overwrite", false; env=("FRB_DM_CACHE_OVERWRITE", "FRB_Z1_DM_CACHE_OVERWRITE"))
dm_value_sanity_max = get_float_arg("dm_value_sanity_max", 1.0e8; env=("FRB_DM_VALUE_SANITY_MAX", "FRB_Z1_DM_VALUE_SANITY_MAX"))
dm_cleanup_nonpositive = get_bool_arg("dm_cleanup_nonpositive", true; env=("FRB_DM_CLEANUP_NONPOSITIVE", "FRB_Z1_DM_CLEANUP_NONPOSITIVE"))

nside > 0 || error("nside must be positive.")
chunkN > 0 || error("chunkN must be positive.")
frb_count > 0 || error("frb_count must be positive.")
frb_redshift > 0.0 || error("frb_redshift must be positive.")
z_max_halos > 0.0 || error("z_max_halos must be positive.")
z_max_halos <= frb_redshift || error("z_max_halos=$(z_max_halos) must not exceed frb_redshift=$(frb_redshift).")
halo_mass_min >= 0.0 || error("halo_mass_min must be nonnegative.")
halo_mass_max > halo_mass_min || error("halo_mass_max must be greater than halo_mass_min.")
frb_overlap_mode in ("mean", "sum", "last") || error("frb_overlap_mode must be mean, sum, or last.")
isdir(output_dir) || mkpath(output_dir)

catalog_path = catalog_source == "halfdome" ? halfdome_path : websky_path
base_tag = "$(catalog_source)_zfrb$(fmt_param_value(frb_redshift))_zhalomax$(fmt_param_value(z_max_halos))_nside$(nside)_nfrb$(frb_count)_seed$(frb_seed)"

println("Random fixed-redshift FRB DM configuration:")
println("  catalog_source=$(catalog_source)")
println("  catalog_path=$(catalog_path)")
println("  output_dir=$(output_dir)")
println("  dm_cache_file=$(dm_cache_file)")
println("  dm_cache_overwrite=$(dm_cache_overwrite)")
println("  dm_cleanup_nonpositive=$(dm_cleanup_nonpositive)")
println("  nside=$(nside), chunkN=$(chunkN)")
println("  frb_count=$(frb_count), frb_seed=$(frb_seed), frb_redshift=$(frb_redshift)")
println("  frb_overlap_mode=$(frb_overlap_mode)")
println("  foreground halo cut: 0 <= z <= $(z_max_halos), mass in [$(halo_mass_min), $(halo_mass_max))")

chi_of_z_itp, z_of_chi_itp = make_chi_and_z_of_chi_itp()
rho_m = 2.775e11 * OMEGAM * H_VALUE^2
res = Healpix.Resolution(nside)

rng = MersenneTwister(frb_seed)
npix = 12 * nside^2
frb_pixels = rand(rng, 1:npix, frb_count)
frb_ra, frb_dec = pixel_centers_to_ra_dec(res, frb_pixels)
frb_ux, frb_uy, frb_uz = ra_dec_to_unit_vectors(frb_ra, frb_dec)
sorted_frb_pixels, sorted_frb_indices = build_frb_pixel_lookup(frb_pixels)
frb_dm = zeros(Float64, frb_count)

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
frb_dm_hp = build_random_frb_dm_map(nside, frb_pixels, frb_dm; overlap_mode=frb_overlap_mode)
Healpix.saveToFITS(frb_dm_hp, "!" * frb_dm_map_path, typechar="D")
println("Saved FRB DM FITS map:")
println("  $(frb_dm_map_path)")
println("Done.")
