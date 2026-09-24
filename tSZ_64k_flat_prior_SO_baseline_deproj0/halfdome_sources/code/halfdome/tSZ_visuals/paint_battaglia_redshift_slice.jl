using XGPaint, Healpix, HDF5, Interpolations
using Base.Threads

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "cosmology_helpers.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "output.jl"))
include(joinpath(@__DIR__, "painting.jl"))
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "catalog_halfdome.jl"))
include(joinpath(@__DIR__, "catalog_websky.jl"))

function arg_key_present(key::AbstractString; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return true
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    return any(a -> startswith(a, prefix1) || startswith(a, prefix2), ARGS)
end

function set_local_notebook_defaults!()
    if !arg_key_present("output_dir"; env="TSZ_VISUAL_OUTPUT_DIR")
        ENV["TSZ_VISUAL_OUTPUT_DIR"] = joinpath(repo_root(), "batched_data", "battaglia_redshift_slices")
    end
    if !arg_key_present("cache_dir"; env="TSZ_VISUAL_CACHE_DIR")
        ENV["TSZ_VISUAL_CACHE_DIR"] = repo_root()
    end
    if !arg_key_present("model_exists"; env="MODEL_EXISTS")
        ENV["MODEL_EXISTS"] = "true"
    end
    if !arg_key_present("apply_gaussian_beam"; env="APPLY_GAUSSIAN_BEAM")
        ENV["APPLY_GAUSSIAN_BEAM"] = "false"
    end
    if !arg_key_present("save_mass_map"; env="SAVE_MASS_MAP")
        ENV["SAVE_MASS_MAP"] = "false"
    end
    if !arg_key_present("halfdome_path"; env="HALFDOME_PATH")
        local_halfdome = joinpath(repo_root(), "lightcone_100.hdf5")
        if isfile(local_halfdome)
            ENV["HALFDOME_PATH"] = local_halfdome
        end
    end
    return nothing
end

function normalize_profile_kind(kind_raw::AbstractString)
    kind = lowercase(strip(String(kind_raw)))
    kind in ("pressure", "tsz", "y") && return "pressure"
    kind in ("density", "dm", "tau", "electron_density") && return "density"
    error("Unsupported profile_kind=$(repr(kind_raw)). Use pressure or density.")
end

function build_density_interpolator(cfg::VisualConfig, cache_file::AbstractString)
    model = HaloDMProfile(BattagliaTauProfile(
        Omega_c=cfg.cosmo_omegac,
        Omega_b=cfg.cosmo_omegab,
        h=cfg.cosmo_h
    ))
    parent = dirname(cache_file)
    isempty(parent) || isdir(parent) || mkpath(parent)

    println("Building/loading Battaglia density interpolator: $(cache_file)")
    return build_interpolator(
        model;
        cache_file=String(cache_file),
        overwrite=!cfg.model_exists,
        verbose=true
    )
end

function build_slice_interpolator(cfg::VisualConfig, profile_kind::AbstractString)
    if profile_kind == "pressure"
        return build_visual_interpolator(cfg)
    end

    default_cache_file = joinpath(
        cfg.cache_dir,
        "cached_density_$(cfg.simulation_tag)_battaglia_tau_$(cfg.cosmology_tag).jld2"
    )
    cache_file = get_string_arg("density_cache_file", default_cache_file; env="DENSITY_CACHE_FILE")
    return build_density_interpolator(cfg, resolve_repo_path(cache_file))
end

mutable struct SliceStats
    scanned_halos::Int
    in_slice_seen::Int
    painted_halos::Int
    z_min::Float64
    z_max::Float64
    mass_min::Float64
    mass_max::Float64
    stopped_by_max_halos::Bool
end

SliceStats() = SliceStats(0, 0, 0, Inf, -Inf, Inf, -Inf, false)

function update_stats!(stats::SliceStats, masses, redshifts)
    n = length(redshifts)
    n == 0 && return stats
    stats.painted_halos += n
    stats.z_min = min(stats.z_min, minimum(redshifts))
    stats.z_max = max(stats.z_max, maximum(redshifts))
    stats.mass_min = min(stats.mass_min, minimum(masses))
    stats.mass_max = max(stats.mass_max, maximum(masses))
    return stats
end

function slice_keep_mask(cfg::VisualConfig, masses, redshifts, z_min::Real, z_max::Real)
    keep = isfinite.(masses) .& isfinite.(redshifts) .&
           (redshifts .>= Float64(z_min)) .& (redshifts .< Float64(z_max))
    if cfg.apply_mass_cut
        keep .&= masses .>= cfg.mass_min
    end
    return keep
end

function remaining_quota(max_halos::Int, painted_halos::Int)
    max_halos <= 0 && return typemax(Int)
    return max(max_halos - painted_halos, 0)
end

function maybe_truncate_indices(local_idx::Vector{Int}, stats::SliceStats, max_halos::Int)
    quota = remaining_quota(max_halos, stats.painted_halos)
    if quota <= 0
        stats.stopped_by_max_halos = true
        return Int[]
    end
    if length(local_idx) > quota
        stats.stopped_by_max_halos = true
        return local_idx[1:quota]
    end
    return local_idx
end

function append_halo_rows!(
    halo_rows::Vector{NTuple{4, Float64}},
    x,
    y,
    z,
    masses,
    redshifts,
    max_rows::Int
)
    max_rows <= 0 && return nothing
    available = max_rows - length(halo_rows)
    available <= 0 && return nothing

    n = min(length(redshifts), available)
    ra, dec = xyz_to_ra_dec_threaded(Float64.(x[1:n]), Float64.(y[1:n]), Float64.(z[1:n]))
    for i in 1:n
        push!(halo_rows, (rad2deg(ra[i]), rad2deg(dec[i]), Float64(masses[i]), Float64(redshifts[i])))
    end
    return nothing
end

function paint_halfdome_slice!(
    cfg::VisualConfig,
    state,
    model_interp,
    z_min::Float64,
    z_max::Float64,
    max_halos::Int,
    halo_table_max_rows::Int
)
    stats = SliceStats()
    halo_rows = NTuple{4, Float64}[]

    h5open(cfg.halfdome_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halo_count = size(pos_ds, 2)
        println("HalfDome catalog halos: $(total_halo_count)")

        for chunk_start in 1:cfg.chunkN:total_halo_count
            chunk_stop = min(chunk_start + cfg.chunkN - 1, total_halo_count)
            idx_range = chunk_start:chunk_stop
            stats.scanned_halos += length(idx_range)

            masses = Float64.(mass_ds[idx_range]) ./ cfg.cosmo_h
            redshifts = Float64.(redshift_ds[idx_range])
            keep = slice_keep_mask(cfg, masses, redshifts, z_min, z_max)
            stats.in_slice_seen += count(keep)
            any(keep) || continue

            local_idx = maybe_truncate_indices(findall(keep), stats, max_halos)
            isempty(local_idx) && break

            idx_vec = collect(idx_range)[local_idx]
            mass_batch = masses[local_idx]
            redshift_batch = redshifts[local_idx]

            pos = read_hdf5_columns(h5["Position"], 1:3, idx_vec)
            radius_batch = nothing
            if state.batch_mass_hp !== nothing
                rdisp_spatial = read_hdf5_columns(h5["Rdisp"], 1:3, idx_vec)
                radius_batch = halfdome_radius_from_rdisp(rdisp_spatial)
            end

            paint_visual_batch!(
                state,
                model_interp,
                view(pos, 1, :),
                view(pos, 2, :),
                view(pos, 3, :),
                radius_batch,
                mass_batch,
                redshift_batch
            )
            append_halo_rows!(
                halo_rows,
                view(pos, 1, :),
                view(pos, 2, :),
                view(pos, 3, :),
                mass_batch,
                redshift_batch,
                halo_table_max_rows
            )
            update_stats!(stats, mass_batch, redshift_batch)
            stats.stopped_by_max_halos && break
        end
    end

    return stats, halo_rows
end

function paint_websky_slice!(
    cfg::VisualConfig,
    state,
    model_interp,
    z_min::Float64,
    z_max::Float64,
    max_halos::Int,
    halo_table_max_rows::Int
)
    stats = SliceStats()
    halo_rows = NTuple{4, Float64}[]
    itp_z_of_chi = make_z_of_chi_itp(omegam=cfg.cosmo_omegam, h_value=cfg.cosmo_h)

    open(cfg.websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        RTHmax = read(io, Float32)
        redshiftbox = read(io, Float32)
        @show total_halo_count RTHmax redshiftbox

        buf = Matrix{Float32}(undef, 10, cfg.chunkN)
        nleft = total_halo_count

        while nleft > 0 && !stats.stopped_by_max_halos
            nthis = min(cfg.chunkN, nleft)
            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            radius = @view cat[7, :]
            redshifts, masses = compute_redshift_and_mass(
                x,
                y,
                z,
                radius,
                itp_z_of_chi,
                cfg.cosmo_rho_m,
                cfg.cosmo_omegam
            )

            stats.scanned_halos += length(redshifts)
            keep = slice_keep_mask(cfg, masses, redshifts, z_min, z_max)
            stats.in_slice_seen += count(keep)
            if any(keep)
                local_idx = maybe_truncate_indices(findall(keep), stats, max_halos)
                if !isempty(local_idx)
                    x_batch = Float64.(x[local_idx])
                    y_batch = Float64.(y[local_idx])
                    z_batch = Float64.(z[local_idx])
                    radius_batch = state.batch_mass_hp === nothing ? nothing : Float64.(radius[local_idx])
                    mass_batch = masses[local_idx]
                    redshift_batch = redshifts[local_idx]

                    paint_visual_batch!(
                        state,
                        model_interp,
                        x_batch,
                        y_batch,
                        z_batch,
                        radius_batch,
                        mass_batch,
                        redshift_batch
                    )
                    append_halo_rows!(
                        halo_rows,
                        x_batch,
                        y_batch,
                        z_batch,
                        mass_batch,
                        redshift_batch,
                        halo_table_max_rows
                    )
                    update_stats!(stats, mass_batch, redshift_batch)
                end
            end

            nleft -= nthis
        end
    end

    return stats, halo_rows
end

function write_halo_table(path::AbstractString, halo_rows)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    open(path, "w") do io
        println(io, "ra_deg,dec_deg,mass_msun,redshift")
        for row in halo_rows
            println(io, join(row, ","))
        end
    end
    println("Saved halo table to $(abspath(path))")
    return path
end

function remove_existing_file(path::AbstractString; label::AbstractString="file")
    if isfile(path)
        println("Removing existing $(label) before this run: $(abspath(path))")
        rm(path; force=true)
    end
    return nothing
end

function save_healpix_map_overwrite(m, path::AbstractString; label::AbstractString="map")
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    target_path = String(path)
    tmp_path = tempname(isempty(parent) ? "." : parent) * ".fits"
    try
        rm(tmp_path; force=true)
        Healpix.saveToFITS(m, "!" * tmp_path, typechar="D")
        rm(target_path; force=true)
        mv(tmp_path, target_path; force=true)
    catch
        rm(tmp_path; force=true)
        rethrow()
    end
    println("Saved $(label) to $(abspath(target_path))")
    return target_path
end

function finite_or_nan(x::Real)
    return isfinite(Float64(x)) ? Float64(x) : NaN
end

function write_metadata(path::AbstractString, cfg::VisualConfig, stats::SliceStats, profile_kind, z_min, z_max, map_path, mass_map_path, halo_table_path)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    mass_map_display = isempty(mass_map_path) ? "" : abspath(mass_map_path)
    open(path, "w") do io
        println(io, "profile_kind=$(profile_kind)")
        println(io, "catalog_source=$(cfg.catalog_source)")
        println(io, "catalog_path=$(cfg.catalog_path)")
        println(io, "nside=$(cfg.nside)")
        println(io, "requested_z_min=$(z_min)")
        println(io, "requested_z_max=$(z_max)")
        println(io, "apply_mass_cut=$(cfg.apply_mass_cut)")
        println(io, "mass_min=$(cfg.mass_min)")
        println(io, "scanned_halos=$(stats.scanned_halos)")
        println(io, "in_slice_seen=$(stats.in_slice_seen)")
        println(io, "painted_halos=$(stats.painted_halos)")
        println(io, "actual_z_min=$(finite_or_nan(stats.z_min))")
        println(io, "actual_z_max=$(finite_or_nan(stats.z_max))")
        println(io, "actual_mass_min=$(finite_or_nan(stats.mass_min))")
        println(io, "actual_mass_max=$(finite_or_nan(stats.mass_max))")
        println(io, "stopped_by_max_halos=$(stats.stopped_by_max_halos)")
        println(io, "map_path=$(abspath(map_path))")
        println(io, "mass_map_path=$(mass_map_display)")
        println(io, "halo_table_path=$(abspath(halo_table_path))")
    end
    println("Saved metadata to $(abspath(path))")
    return path
end

function default_slice_tag(cfg::VisualConfig, profile_kind::AbstractString, z_min::Real, z_max::Real)
    return "$(cfg.simulation_tag)_battaglia_$(profile_kind)_z$(fmt_param_value(z_min))_$(fmt_param_value(z_max))_nside$(cfg.nside)_$(cfg.param_tag)_$(cfg.cosmology_tag)"
end

function run_battaglia_redshift_slice()
    t0 = time()
    println("Running $(abspath(@__FILE__)) with temp-file FITS writer.")
    set_local_notebook_defaults!()
    cfg = load_visual_config()

    profile_kind = normalize_profile_kind(
        get_string_arg("profile_kind", "pressure"; env="BARYON_SLICE_PROFILE_KIND")
    )
    z_min = get_float_arg("z_min", 0.5; env="BARYON_SLICE_Z_MIN")
    z_max = get_float_arg("z_max", 0.6; env="BARYON_SLICE_Z_MAX")
    max_halos = get_int_arg("max_halos", 100_000; env="BARYON_SLICE_MAX_HALOS")
    halo_table_max_rows = get_int_arg("halo_table_max_rows", 10_000; env="BARYON_SLICE_HALO_TABLE_MAX_ROWS")

    isfinite(z_min) && isfinite(z_max) && z_min >= 0.0 && z_max > z_min || error("Require 0 <= z_min < z_max.")
    max_halos >= 0 || error("max_halos must be non-negative. Use 0 for no cap.")
    halo_table_max_rows >= 0 || error("halo_table_max_rows must be non-negative.")

    ensure_output_dir(cfg)
    tag = default_slice_tag(cfg, profile_kind, z_min, z_max)
    map_path = get_string_arg("slice_map_path", joinpath(cfg.output_dir, "$(tag).fits"); env="BARYON_SLICE_MAP_PATH")
    mass_map_path = get_string_arg("slice_mass_map_path", joinpath(cfg.output_dir, "$(tag)_mass.fits"); env="BARYON_SLICE_MASS_MAP_PATH")
    halo_table_path = get_string_arg("slice_halo_table_path", joinpath(cfg.output_dir, "$(tag)_halos.csv"); env="BARYON_SLICE_HALO_TABLE_PATH")
    metadata_path = get_string_arg("slice_metadata_path", joinpath(cfg.output_dir, "$(tag)_metadata.txt"); env="BARYON_SLICE_METADATA_PATH")
    remove_existing_file(map_path; label="slice map")
    if cfg.save_mass_map
        remove_existing_file(mass_map_path; label="slice mass map")
    end

    print_visual_config(cfg)
    max_halos_display = max_halos == 0 ? "all" : string(max_halos)
    println("Redshift slice: z in [$(z_min), $(z_max)); profile_kind=$(profile_kind); max_halos=$(max_halos_display)")

    model_interp = build_slice_interpolator(cfg, profile_kind)
    trim_process_memory()
    state = init_visual_maps(cfg)

    paint_t0 = start_phase_timing()
    stats, halo_rows = if cfg.catalog_source == "halfdome"
        paint_halfdome_slice!(cfg, state, model_interp, z_min, z_max, max_halos, halo_table_max_rows)
    else
        paint_websky_slice!(cfg, state, model_interp, z_min, z_max, max_halos, halo_table_max_rows)
    end
    print_phase_usage("Slice painting", paint_t0)

    stats.painted_halos > 0 || error("No halos were painted for z in [$(z_min), $(z_max)).")

    output_map = prepare_tsz_map_for_output(cfg, state.m_hp)
    save_healpix_map_overwrite(output_map, map_path; label="slice map")

    if cfg.save_mass_map && state.mass_hp !== nothing
        save_healpix_map_overwrite(state.mass_hp, mass_map_path; label="slice mass map")
    else
        mass_map_path = ""
    end

    write_halo_table(halo_table_path, halo_rows)
    write_metadata(metadata_path, cfg, stats, profile_kind, z_min, z_max, map_path, mass_map_path, halo_table_path)

    println(
        "Painted $(stats.painted_halos) halos; actual z range " *
        "[$(finite_or_nan(stats.z_min)), $(finite_or_nan(stats.z_max))], " *
        "mass range [$(finite_or_nan(stats.mass_min)), $(finite_or_nan(stats.mass_max))]."
    )
    println("Finished Battaglia redshift-slice painting in $(round(time() - t0; digits=2)) s.")

    return (
        cfg=cfg,
        stats=stats,
        map_path=abspath(map_path),
        mass_map_path=isempty(mass_map_path) ? "" : abspath(mass_map_path),
        halo_table_path=abspath(halo_table_path),
        metadata_path=abspath(metadata_path)
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_battaglia_redshift_slice()
end
