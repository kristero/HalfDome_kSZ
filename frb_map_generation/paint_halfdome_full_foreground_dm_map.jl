if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# Fully paint the HalfDome foreground halo DM field up to a source redshift.
#
# This is different from make_random_frb_dm_map_z1.jl:
#   - this script paints every foreground halo into every affected HEALPix pixel
#   - then it optionally samples N random FRB source pixels from that full map
#   - the full map is the object to inspect if you want to see low-z halos

using XGPaint
using HDF5
using Healpix
using Random
using Statistics

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const DEFAULT_SOURCE_REDSHIFT = 1.0

code_root() = @__DIR__
project_root() = basename(code_root()) == "frb_map_generation" ? dirname(code_root()) : code_root()

function resolve_project_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(project_root(), path))
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
        push!(trial_paths, normpath(joinpath(project_root(), path)))
        push!(trial_paths, normpath(joinpath(code_root(), path)))
        push!(trial_paths, normpath(joinpath(pwd(), path)))
        push!(trial_paths, normpath(joinpath(project_root(), "halfdome", path)))
        push!(trial_paths, normpath(joinpath(project_root(), "HalfDome", path)))
    end

    unique_trial_paths = unique(trial_paths)
    for candidate in unique_trial_paths
        isfile(candidate) && return candidate
        isdir(candidate) && return resolve_hdf5_catalog_from_directory(candidate)
    end

    error("Could not find HalfDome catalog $(repr(path)). Tried: $(join(unique_trial_paths, ", "))")
end

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

function halo_dm_constructor()
    if isdefined(Main, :HaloDMProfile)
        return getfield(Main, :HaloDMProfile)
    elseif isdefined(XGPaint, :HaloDMProfile)
        return getfield(XGPaint, :HaloDMProfile)
    end
    error("HaloDMProfile is not available in this Julia/XGPaint environment.")
end

function xgpaint_paint_function()
    if isdefined(Main, :paint!)
        return getfield(Main, :paint!)
    elseif isdefined(XGPaint, :paint!)
        return getfield(XGPaint, :paint!)
    end
    error("paint! is not available in this Julia/XGPaint environment.")
end

function xgpaint_build_interpolator_function()
    if isdefined(Main, :build_interpolator)
        return getfield(Main, :build_interpolator)
    elseif isdefined(XGPaint, :build_interpolator)
        return getfield(XGPaint, :build_interpolator)
    end
    error("build_interpolator is not available in this Julia/XGPaint environment.")
end

function make_dm_model()
    constructor = halo_dm_constructor()
    tau_model = XGPaint.BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE)
    return constructor(tau_model)
end

function positions_to_ra_dec(positions)
    n = size(positions, 2)
    ras = Vector{Float64}(undef, n)
    decs = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        x = positions[1, i]
        y = positions[2, i]
        z = positions[3, i]
        r = sqrt(x * x + y * y + z * z)
        r > 0.0 || error("Catalog halo in chunk has zero radius.")
        theta, phi = Healpix.vec2ang(x / r, y / r, z / r)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function foreground_mask(masses, redshifts; z_min, z_max, mass_min, mass_max)
    keep = isfinite.(masses) .& isfinite.(redshifts)
    keep .&= masses .> 0.0
    keep .&= redshifts .>= z_min
    keep .&= redshifts .<= z_max
    keep .&= masses .>= mass_min
    isfinite(mass_max) && (keep .&= masses .< mass_max)
    return keep
end

function draw_random_pixels(rng, npix::Int, count::Int; unique_pixels::Bool=true)
    count > 0 || error("N must be positive.")
    npix > 0 || error("npix must be positive.")

    if !unique_pixels
        return rand(rng, 1:npix, count)
    end

    count <= npix || error("Cannot draw $(count) unique pixels from npix=$(npix).")
    pixels = Vector{Int}(undef, count)
    seen = Set{Int}()
    i = 1
    while i <= count
        pix = rand(rng, 1:npix)
        pix in seen && continue
        push!(seen, pix)
        pixels[i] = pix
        i += 1
    end
    return pixels
end

function pixel_centers_to_ra_dec(res, pixels)
    ras = Vector{Float64}(undef, length(pixels))
    decs = Vector{Float64}(undef, length(pixels))

    @inbounds for i in eachindex(pixels)
        vx, vy, vz = Healpix.pix2vecRing(res, pixels[i])
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function build_sparse_source_map(nside::Int, source_pixels, source_dm)
    source_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(source_map.pixels, 0.0)

    pixel_counts = Dict{Int, Int}()
    @inbounds for i in eachindex(source_pixels)
        pix = Int(source_pixels[i])
        source_map.pixels[pix] += Float64(source_dm[i])
        pixel_counts[pix] = get(pixel_counts, pix, 0) + 1
    end
    for (pix, count) in pixel_counts
        source_map.pixels[pix] /= count
    end

    return source_map
end

function write_source_catalog(path, source_pixels, source_ra, source_dec, source_redshift::Float64, source_dm)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "frb_index,pixel_ring_1based,pixel_ring_0based_for_healpy,ra_rad,dec_rad,z_source,dm_pc_cm3")
        @inbounds for i in eachindex(source_pixels)
            pix = Int(source_pixels[i])
            println(io, "$(i),$(pix),$(pix - 1),$(source_ra[i]),$(source_dec[i]),$(source_redshift),$(source_dm[i])")
        end
    end

    return path
end

function save_summary(path; config_lines, counters, map_pixels, source_dm=nothing)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "HalfDome full foreground DM map")
        for line in config_lines
            println(io, line)
        end
        println(io)
        println(io, "total_halo_count=$(counters.total_halo_count)")
        println(io, "halos_passing_cuts=$(counters.halos_passing_cuts)")
        println(io, "halos_painted=$(counters.halos_painted)")
        println(io, "painted_redshift_min=$(counters.redshift_min)")
        println(io, "painted_redshift_max=$(counters.redshift_max)")
        println(io, "painted_mass_min=$(counters.mass_min)")
        println(io, "painted_mass_max=$(counters.mass_max)")
        println(io, "map_nonzero_pixels=$(count(!=(0.0), map_pixels))")
        println(io, "map_min=$(minimum(map_pixels))")
        println(io, "map_max=$(maximum(map_pixels))")
        println(io, "map_mean=$(mean(map_pixels))")
        if source_dm !== nothing
            println(io)
            println(io, "source_dm_min=$(minimum(source_dm))")
            println(io, "source_dm_max=$(maximum(source_dm))")
            println(io, "source_dm_mean=$(mean(source_dm))")
            println(io, "source_dm_std=$(std(source_dm))")
        end
    end

    return path
end

function paint_full_foreground_map!(
    dm_map,
    workspace,
    dm_model_interp,
    catalog_path::AbstractString;
    z_min::Float64,
    z_max::Float64,
    mass_min::Float64,
    mass_max::Float64,
    chunkN::Int,
    progress_every::Int,
)
    total_halo_count = 0
    halos_passing_cuts = 0
    halos_painted = 0
    z_min_seen = Inf
    z_max_seen = -Inf
    mass_min_seen = Inf
    mass_max_seen = -Inf

    paint_fn = xgpaint_paint_function()

    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halo_count = size(pos_ds, 2)

        chunk_index = 0
        for batch_start in 1:chunkN:total_halo_count
            chunk_index += 1
            batch_stop = min(batch_start + chunkN - 1, total_halo_count)
            idx = batch_start:batch_stop

            positions = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            keep = foreground_mask(
                masses,
                redshifts;
                z_min=z_min,
                z_max=z_max,
                mass_min=mass_min,
                mass_max=mass_max,
            )

            selected_count = count(keep)
            selected_count == 0 && continue
            halos_passing_cuts += selected_count

            selected_positions = positions[:, keep]
            selected_masses = masses[keep]
            selected_redshifts = redshifts[keep]

            z_min_seen = min(z_min_seen, minimum(selected_redshifts))
            z_max_seen = max(z_max_seen, maximum(selected_redshifts))
            mass_min_seen = min(mass_min_seen, minimum(selected_masses))
            mass_max_seen = max(mass_max_seen, maximum(selected_masses))

            ras, decs = positions_to_ra_dec(selected_positions)
            perm = sortperm(decs)

            paint_fn(
                dm_map,
                workspace,
                dm_model_interp,
                selected_masses[perm],
                selected_redshifts[perm],
                ras[perm],
                decs[perm],
            )

            halos_painted += selected_count
            if progress_every > 0 && (chunk_index % progress_every == 0 || batch_stop == total_halo_count)
                println(
                    "  chunk $(chunk_index): catalog rows $(batch_start)-$(batch_stop), " *
                    "painted so far $(halos_painted)"
                )
                flush(stdout)
            end
        end
    end

    halos_painted > 0 || error("No HalfDome halos passed the requested foreground cuts.")
    return (
        total_halo_count=total_halo_count,
        halos_passing_cuts=halos_passing_cuts,
        halos_painted=halos_painted,
        redshift_min=z_min_seen,
        redshift_max=z_max_seen,
        mass_min=mass_min_seen,
        mass_max=mass_max_seen,
    )
end

function main()
    catalog_path = resolve_halfdome_catalog_path(get_string_arg(
        "halfdome_path",
        "lightcone_100.hdf5";
        env=("HALFDOME_FULL_DM_PATH", "FRB_HALFDOME_PATH"),
    ))
    output_dir = resolve_project_path(get_string_arg(
        "output_dir",
        joinpath("frb_map_generation", "outputs", "halfdome_full_foreground_dm_z1");
        env="HALFDOME_FULL_DM_OUTPUT_DIR",
    ))
    isdir(output_dir) || mkpath(output_dir)

    nside = get_int_arg("nside", 4096; env="HALFDOME_FULL_DM_NSIDE")
    source_count = get_int_arg("N", 10_000; env=("HALFDOME_FULL_DM_N", "FRB_N"))
    seed = get_int_arg("seed", 42; env=("HALFDOME_FULL_DM_SEED", "FRB_SEED"))
    chunkN = get_int_arg("chunkN", 100_000; env="HALFDOME_FULL_DM_CHUNKN")
    source_redshift_default = get_float_arg("source_redshift", DEFAULT_SOURCE_REDSHIFT; env=("HALFDOME_FULL_DM_SOURCE_REDSHIFT", "FRB_SOURCE_REDSHIFT"))
    source_redshift = get_float_arg("z_source", source_redshift_default; env=("HALFDOME_FULL_DM_Z_SOURCE", "FRB_Z_SOURCE"))
    z_min = get_float_arg("z_min", 0.0; env="HALFDOME_FULL_DM_Z_MIN")
    z_max_halos = get_float_arg("z_max_halos", source_redshift; env=("HALFDOME_FULL_DM_Z_MAX", "FRB_HALO_Z_MAX"))
    halo_mass_min = get_float_arg("halo_mass_min", 0.0; env="HALFDOME_FULL_DM_MASS_MIN")
    halo_mass_max = get_float_arg("halo_mass_max", Inf; env="HALFDOME_FULL_DM_MASS_MAX")
    unique_source_pixels = get_bool_arg("unique_source_pixels", true; env="HALFDOME_FULL_DM_UNIQUE_SOURCE_PIXELS")
    save_full_map = get_bool_arg("save_full_map", true; env="HALFDOME_FULL_DM_SAVE_FULL_MAP")
    save_source_map = get_bool_arg("save_source_map", true; env="HALFDOME_FULL_DM_SAVE_SOURCE_MAP")
    should_save_source_catalog = get_bool_arg("save_source_catalog", true; env="HALFDOME_FULL_DM_SAVE_SOURCE_CATALOG")
    progress_every = get_int_arg("progress_every", 5; env="HALFDOME_FULL_DM_PROGRESS_EVERY")
    dm_cleanup_nonpositive = get_bool_arg("dm_cleanup_nonpositive", true; env="HALFDOME_FULL_DM_CLEANUP_NONPOSITIVE")
    dm_cache_file = resolve_project_path(get_string_arg(
        "dm_cache_file",
        joinpath(output_dir, "halfdome_full_foreground_dm_profile_cache.jld2");
        env="HALFDOME_FULL_DM_CACHE_FILE",
    ))
    dm_cache_overwrite = get_bool_arg("dm_cache_overwrite", false; env="HALFDOME_FULL_DM_CACHE_OVERWRITE")

    nside > 0 || error("nside must be positive.")
    source_count > 0 || error("N must be positive.")
    chunkN > 0 || error("chunkN must be positive.")
    source_redshift > 0.0 || error("z_source/source_redshift must be positive.")
    z_min >= 0.0 || error("z_min must be non-negative.")
    z_max_halos >= z_min || error("z_max_halos must be >= z_min.")
    z_max_halos <= source_redshift || error("z_max_halos=$(z_max_halos) must not exceed z_source=$(source_redshift).")
    halo_mass_min >= 0.0 || error("halo_mass_min must be non-negative.")
    halo_mass_max > halo_mass_min || error("halo_mass_max must be greater than halo_mass_min.")

    tag = "halfdome_full_foreground_dm_zsource$(fmt_param_value(source_redshift))_zhalomax$(fmt_param_value(z_max_halos))_nside$(nside)"
    full_map_path = joinpath(output_dir, "$(tag).fits")
    source_tag = "$(tag)_nfrb$(source_count)_seed$(seed)"
    source_map_path = joinpath(output_dir, "$(source_tag)_sparse_frb_dm_map.fits")
    source_catalog_path = joinpath(output_dir, "$(source_tag)_frb_sources.csv")
    summary_path = joinpath(output_dir, "$(source_tag)_summary.txt")

    println("HalfDome full foreground DM map configuration:")
    println("  catalog_path=$(catalog_path)")
    println("  output_dir=$(output_dir)")
    println("  nside=$(nside), N=$(source_count), seed=$(seed), chunkN=$(chunkN)")
    println("  source redshift=$(source_redshift)")
    println("  foreground halo cut: $(z_min) <= z <= $(z_max_halos), mass in [$(halo_mass_min), $(halo_mass_max))")
    println("  save_full_map=$(save_full_map), save_source_map=$(save_source_map), save_source_catalog=$(should_save_source_catalog)")
    println("  dm_cache_file=$(dm_cache_file), dm_cache_overwrite=$(dm_cache_overwrite)")

    res = Healpix.Resolution(nside)
    dm_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(dm_map.pixels, 0.0)
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

    dm_cache_dir = dirname(dm_cache_file)
    isempty(dm_cache_dir) || isdir(dm_cache_dir) || mkpath(dm_cache_dir)
    ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = dm_cleanup_nonpositive ? "true" : "false"
    dm_model = make_dm_model()
    dm_model_interp = xgpaint_build_interpolator_function()(
        dm_model;
        cache_file=dm_cache_file,
        overwrite=dm_cache_overwrite,
    )

    println("Painting the full foreground DM field...")
    counters = paint_full_foreground_map!(
        dm_map,
        workspace,
        dm_model_interp,
        catalog_path;
        z_min=z_min,
        z_max=z_max_halos,
        mass_min=halo_mass_min,
        mass_max=halo_mass_max,
        chunkN=chunkN,
        progress_every=progress_every,
    )
    println("Painted $(counters.halos_painted) foreground halos out of $(counters.total_halo_count) catalog rows.")
    println("Full map summary: min=$(minimum(dm_map.pixels)), max=$(maximum(dm_map.pixels)), mean=$(mean(dm_map.pixels)), nonzero=$(count(!=(0.0), dm_map.pixels))")

    if save_full_map
        println("Writing full foreground DM FITS map:")
        println("  $(full_map_path)")
        Healpix.saveToFITS(dm_map, "!" * full_map_path, typechar="D")
    end

    rng = MersenneTwister(seed)
    npix = length(dm_map.pixels)
    source_pixels = draw_random_pixels(rng, npix, source_count; unique_pixels=unique_source_pixels)
    source_ra, source_dec = pixel_centers_to_ra_dec(res, source_pixels)
    source_dm = Float64.(dm_map.pixels[source_pixels])
    println("Sampled $(source_count) FRB source pixels from the full map.")
    println("Source DM summary: min=$(minimum(source_dm)), max=$(maximum(source_dm)), mean=$(mean(source_dm)), std=$(std(source_dm))")

    if save_source_map
        source_map = build_sparse_source_map(nside, source_pixels, source_dm)
        println("Writing sparse sampled FRB DM FITS map:")
        println("  $(source_map_path)")
        Healpix.saveToFITS(source_map, "!" * source_map_path, typechar="D")
    end

    if should_save_source_catalog
        write_source_catalog(source_catalog_path, source_pixels, source_ra, source_dec, source_redshift, source_dm)
        println("Wrote sampled FRB source catalog:")
        println("  $(source_catalog_path)")
    end

    config_lines = [
        "catalog_path=$(catalog_path)",
        "output_dir=$(output_dir)",
        "full_map_path=$(full_map_path)",
        "source_map_path=$(source_map_path)",
        "source_catalog_path=$(source_catalog_path)",
        "nside=$(nside)",
        "N=$(source_count)",
        "seed=$(seed)",
        "unique_source_pixels=$(unique_source_pixels)",
        "z_source=$(source_redshift)",
        "z_min=$(z_min)",
        "z_max_halos=$(z_max_halos)",
        "halo_mass_min=$(halo_mass_min)",
        "halo_mass_max=$(halo_mass_max)",
        "chunkN=$(chunkN)",
        "dm_cache_file=$(dm_cache_file)",
        "dm_cache_overwrite=$(dm_cache_overwrite)",
        "dm_cleanup_nonpositive=$(dm_cleanup_nonpositive)",
    ]
    save_summary(summary_path; config_lines=config_lines, counters=counters, map_pixels=dm_map.pixels, source_dm=source_dm)
    println("Wrote summary:")
    println("  $(summary_path)")
    println("Done.")
end

main()
