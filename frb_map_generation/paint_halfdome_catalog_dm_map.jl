if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# Paint an XGPaint FRB-DM halo map from a random sample of the HalfDome catalog.
#
# Default behavior:
#   - read lightcone_100.hdf5
#   - randomly sample N=10000 halos from the full allowed catalog
#   - pass HalfDome masses to XGPaint as halo_mass_m200c / h
#   - paint HaloDMProfile(BattagliaTauProfile(...)) onto a HEALPix map
#   - write a FITS map and a small summary file

using XGPaint
using HDF5
using Healpix
using Random
using Statistics

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB

function code_root()
    return @__DIR__
end

function project_root()
    return basename(code_root()) == "frb_map_generation" ? dirname(code_root()) : code_root()
end

function resolve_project_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(project_root(), path))
end

function resolve_halfdome_catalog_path(path::AbstractString)
    resolved = resolve_project_path(path)
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

function passes_catalog_cuts(mass::Float64, redshift::Float64; z_min, z_max, mass_min, mass_max)
    isfinite(mass) || return false
    isfinite(redshift) || return false
    mass > 0.0 || return false
    redshift >= z_min || return false
    redshift <= z_max || return false
    mass >= mass_min || return false
    isfinite(mass_max) && mass >= mass_max && return false
    return true
end

function reservoir_sample_halfdome_halos(
    catalog_path::AbstractString;
    sample_count_target::Int,
    seed::Int,
    z_min::Float64,
    z_max::Float64,
    mass_min::Float64,
    mass_max::Float64,
    chunkN::Int,
)
    sample_count_target > 0 || error("N/sample_count must be positive.")
    rng = MersenneTwister(seed)

    sample_positions = Matrix{Float64}(undef, 3, sample_count_target)
    sample_masses = Vector{Float64}(undef, sample_count_target)
    sample_redshifts = Vector{Float64}(undef, sample_count_target)
    sample_indices = Vector{Int}(undef, sample_count_target)

    selected_count = 0
    stored_count = 0
    total_halo_count = 0

    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halo_count = size(pos_ds, 2)

        for batch_start in 1:chunkN:total_halo_count
            batch_stop = min(batch_start + chunkN - 1, total_halo_count)
            idx = batch_start:batch_stop

            pos = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])

            @inbounds for local_i in eachindex(masses)
                mass = masses[local_i]
                redshift = redshifts[local_i]
                passes_catalog_cuts(
                    mass,
                    redshift;
                    z_min=z_min,
                    z_max=z_max,
                    mass_min=mass_min,
                    mass_max=mass_max,
                ) || continue

                selected_count += 1
                store_i = 0
                if stored_count < sample_count_target
                    stored_count += 1
                    store_i = stored_count
                else
                    candidate = rand(rng, 1:selected_count)
                    if candidate <= sample_count_target
                        store_i = candidate
                    end
                end

                if store_i > 0
                    sample_positions[:, store_i] .= @view pos[:, local_i]
                    sample_masses[store_i] = mass
                    sample_redshifts[store_i] = redshift
                    sample_indices[store_i] = batch_start + local_i - 1
                end
            end
        end
    end

    stored_count > 0 || error("No HalfDome halos passed the requested cuts.")
    return (
        positions=sample_positions[:, 1:stored_count],
        masses=sample_masses[1:stored_count],
        redshifts=sample_redshifts[1:stored_count],
        indices=sample_indices[1:stored_count],
        selected_count=selected_count,
        total_halo_count=total_halo_count,
    )
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
        r > 0.0 || error("Catalog halo at sampled position $(i) has zero radius.")
        theta, phi = Healpix.vec2ang(x / r, y / r, z / r)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function save_sample_table(path, sample)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    open(path, "w") do io
        println(io, "sample_index,catalog_index,mass_msun,redshift")
        for i in eachindex(sample.masses)
            println(io, "$(i),$(sample.indices[i]),$(sample.masses[i]),$(sample.redshifts[i])")
        end
    end
    return path
end

function save_summary(path; config_lines, sample, map_pixels)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    nonzero_count = count(!=(0.0), map_pixels)
    open(path, "w") do io
        println(io, "HalfDome catalog DM painted map")
        for line in config_lines
            println(io, line)
        end
        println(io)
        println(io, "total_halo_count=$(sample.total_halo_count)")
        println(io, "halos_passing_cuts=$(sample.selected_count)")
        println(io, "halos_painted=$(length(sample.masses))")
        println(io, "painted_mass_min=$(minimum(sample.masses))")
        println(io, "painted_mass_max=$(maximum(sample.masses))")
        println(io, "painted_redshift_min=$(minimum(sample.redshifts))")
        println(io, "painted_redshift_max=$(maximum(sample.redshifts))")
        println(io, "map_nonzero_pixels=$(nonzero_count)")
        println(io, "map_min=$(minimum(map_pixels))")
        println(io, "map_max=$(maximum(map_pixels))")
        println(io, "map_mean=$(mean(map_pixels))")
    end
    return path
end

function main()
    catalog_path = resolve_halfdome_catalog_path(get_string_arg(
        "halfdome_path",
        "lightcone_100.hdf5";
        env=("HALFDOME_CATALOG_DM_PATH", "FRB_HALFDOME_PATH"),
    ))
    output_dir = resolve_project_path(get_string_arg(
        "output_dir",
        joinpath("frb_map_generation", "outputs", "halfdome_catalog_dm_maps");
        env="HALFDOME_CATALOG_DM_OUTPUT_DIR",
    ))
    isdir(output_dir) || mkpath(output_dir)

    nside = get_int_arg("nside", 4096; env="HALFDOME_CATALOG_DM_NSIDE")
    sample_count = get_int_arg("N", 10_000; env=("HALFDOME_CATALOG_DM_N", "HALFDOME_CATALOG_DM_SAMPLE_COUNT"))
    seed = get_int_arg("seed", 42; env="HALFDOME_CATALOG_DM_SEED")
    chunkN = get_int_arg("chunkN", 500_000; env="HALFDOME_CATALOG_DM_CHUNKN")
    z_min = get_float_arg("z_min", 0.0; env="HALFDOME_CATALOG_DM_Z_MIN")
    z_max = get_float_arg("z_max", 5.0; env="HALFDOME_CATALOG_DM_Z_MAX")
    halo_mass_min = get_float_arg("halo_mass_min", 0.0; env="HALFDOME_CATALOG_DM_MASS_MIN")
    halo_mass_max = get_float_arg("halo_mass_max", Inf; env="HALFDOME_CATALOG_DM_MASS_MAX")
    save_sample_catalog = get_bool_arg("save_sample_catalog", true; env="HALFDOME_CATALOG_DM_SAVE_SAMPLE")
    dm_cleanup_nonpositive = get_bool_arg("dm_cleanup_nonpositive", true; env="HALFDOME_CATALOG_DM_CLEANUP_NONPOSITIVE")
    dm_cache_file = resolve_project_path(get_string_arg(
        "dm_cache_file",
        joinpath(output_dir, "halfdome_catalog_dm_profile_cache.jld2");
        env="HALFDOME_CATALOG_DM_CACHE_FILE",
    ))
    dm_cache_overwrite = get_bool_arg("dm_cache_overwrite", false; env="HALFDOME_CATALOG_DM_CACHE_OVERWRITE")

    nside > 0 || error("nside must be positive.")
    sample_count > 0 || error("N/sample_count must be positive.")
    chunkN > 0 || error("chunkN must be positive.")
    z_max >= z_min || error("z_max must be >= z_min.")
    halo_mass_min >= 0.0 || error("halo_mass_min must be non-negative.")
    halo_mass_max > halo_mass_min || error("halo_mass_max must be greater than halo_mass_min.")

    tag = "halfdome_catalog_dm_nside$(nside)_N$(sample_count)_seed$(seed)_z$(fmt_param_value(z_min))to$(fmt_param_value(z_max))"
    map_path = joinpath(output_dir, "$(tag).fits")
    summary_path = joinpath(output_dir, "$(tag)_summary.txt")
    sample_path = joinpath(output_dir, "$(tag)_sample.csv")

    println("HalfDome catalog DM painted-map configuration:")
    println("  catalog_path=$(catalog_path)")
    println("  output_dir=$(output_dir)")
    println("  nside=$(nside), N=$(sample_count), seed=$(seed), chunkN=$(chunkN)")
    println("  z in [$(z_min), $(z_max)], mass in [$(halo_mass_min), $(halo_mass_max))")
    println("  dm_cache_file=$(dm_cache_file), dm_cache_overwrite=$(dm_cache_overwrite)")

    println("Sampling HalfDome catalog halos...")
    sample = reservoir_sample_halfdome_halos(
        catalog_path;
        sample_count_target=sample_count,
        seed=seed,
        z_min=z_min,
        z_max=z_max,
        mass_min=halo_mass_min,
        mass_max=halo_mass_max,
        chunkN=chunkN,
    )
    println("  total_halo_count=$(sample.total_halo_count)")
    println("  halos_passing_cuts=$(sample.selected_count)")
    println("  halos_painted=$(length(sample.masses))")

    ras, decs = positions_to_ra_dec(sample.positions)
    perm = sortperm(decs)

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

    println("Painting $(length(sample.masses)) HalfDome halos...")
    xgpaint_paint_function()(
        dm_map,
        workspace,
        dm_model_interp,
        sample.masses[perm],
        sample.redshifts[perm],
        ras[perm],
        decs[perm],
    )

    println("Writing FITS map:")
    println("  $(map_path)")
    Healpix.saveToFITS(dm_map, "!" * map_path, typechar="D")

    if save_sample_catalog
        save_sample_table(sample_path, sample)
        println("Wrote sampled halo table:")
        println("  $(sample_path)")
    end

    config_lines = [
        "catalog_path=$(catalog_path)",
        "output_dir=$(output_dir)",
        "map_path=$(map_path)",
        "nside=$(nside)",
        "N=$(sample_count)",
        "seed=$(seed)",
        "chunkN=$(chunkN)",
        "z_min=$(z_min)",
        "z_max=$(z_max)",
        "halo_mass_min=$(halo_mass_min)",
        "halo_mass_max=$(halo_mass_max)",
        "dm_cache_file=$(dm_cache_file)",
        "dm_cache_overwrite=$(dm_cache_overwrite)",
        "dm_cleanup_nonpositive=$(dm_cleanup_nonpositive)",
    ]
    save_summary(summary_path; config_lines=config_lines, sample=sample, map_pixels=dm_map.pixels)
    println("Wrote summary:")
    println("  $(summary_path)")
    println("Map summary: min=$(minimum(dm_map.pixels)), max=$(maximum(dm_map.pixels)), nonzero=$(count(!=(0.0), dm_map.pixels))")
    println("Done.")
end

main()
