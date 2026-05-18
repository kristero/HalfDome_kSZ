function build_tsz_model(cfg::VisualConfig)
    p = cfg.battaglia_params
    return Battaglia16ThermalSZProfile(
        Omega_c=cfg.cosmo_omegac,
        Omega_b=cfg.cosmo_omegab,
        h=cfg.cosmo_h,
        P0_amp=p.P0_amp,
        P0_alpha_m=p.P0_alpha_m,
        P0_alpha_z=p.P0_alpha_z,
        x_c_amp=p.x_c_amp,
        x_c_alpha_m=p.x_c_alpha_m,
        x_c_alpha_z=p.x_c_alpha_z,
        beta_amp=p.beta_amp,
        beta_alpha_m=p.beta_alpha_m,
        beta_alpha_z=p.beta_alpha_z,
        alpha_amp=p.alpha_amp,
        alpha_alpha_m=p.alpha_alpha_m,
        alpha_alpha_z=p.alpha_alpha_z,
        gamma_amp=p.gamma_amp,
        gamma_alpha_m=p.gamma_alpha_m,
        gamma_alpha_z=p.gamma_alpha_z
    )
end

function visual_interpolator_cache_filename(cfg::VisualConfig)
    cache_sim_tag = cfg.catalog_source == "halfdome" ? "HalfDome" : "Websky"
    return "cached_tSZ_$(cache_sim_tag)_cosmo_$(cfg.cache_param_tag)_$(cfg.cosmology_tag).jld2"
end

function legacy_visual_interpolator_cache_filename(cfg::VisualConfig)
    cache_sim_tag = cfg.catalog_source == "halfdome" ? "HalfDome" : "Websky"
    return "cached_tSZ_$(cache_sim_tag)_cosmo_$(cfg.param_tag).jld2"
end

function row_tag_visual_interpolator_cache_filename(cfg::VisualConfig)
    cache_sim_tag = cfg.catalog_source == "halfdome" ? "HalfDome" : "Websky"
    return "cached_tSZ_$(cache_sim_tag)_cosmo_$(cfg.param_tag)_$(cfg.cosmology_tag).jld2"
end

function visual_interpolator_filename_for_param_tag(cfg::VisualConfig, param_tag::AbstractString; include_cosmology::Bool=true)
    cache_sim_tag = cfg.catalog_source == "halfdome" ? "HalfDome" : "Websky"
    if include_cosmology
        return "cached_tSZ_$(cache_sim_tag)_cosmo_$(param_tag)_$(cfg.cosmology_tag).jld2"
    end
    return "cached_tSZ_$(cache_sim_tag)_cosmo_$(param_tag).jld2"
end

function split_sobol_cache_param_tag_aliases(cfg::VisualConfig)
    cfg.sobol_row > 0 || return String[]
    isempty(cfg.sobol_csv_path) && return String[]

    aliases = String[]
    csv_dir = dirname(cfg.sobol_csv_path)
    csv_stem = splitext(basename(cfg.sobol_csv_path))[1]

    split_match = match(r"^(.+)_([0-9]+)$", csv_stem)
    if split_match !== nothing
        full_stem = split_match.captures[1]
        split_index = parse(Int, split_match.captures[2])
        full_csv_path = joinpath(csv_dir, full_stem * ".csv")
        if split_index >= 1 && isfile(full_csv_path)
            split_rows = csv_data_row_count(cfg.sobol_csv_path)
            global_row = (split_index - 1) * split_rows + cfg.sobol_row
            if global_row >= 1
                push!(aliases, build_sobol_param_tag(full_csv_path, global_row))
            end
        end
    end

    for split_index in (1, 2)
        split_csv_path = joinpath(csv_dir, "$(csv_stem)_$(split_index).csv")
        isfile(split_csv_path) || continue
        split_rows = csv_data_row_count(split_csv_path)
        split_rows > 0 || continue
        split_start = (split_index - 1) * split_rows + 1
        split_stop = split_index * split_rows
        if split_start <= cfg.sobol_row <= split_stop
            split_row = cfg.sobol_row - split_start + 1
            push!(aliases, build_sobol_param_tag(split_csv_path, split_row))
        end
    end

    return [tag for tag in unique(aliases) if tag != cfg.param_tag]
end

function visual_interpolator_cache_paths(cfg::VisualConfig)
    filename = visual_interpolator_cache_filename(cfg)
    row_tag_filename = row_tag_visual_interpolator_cache_filename(cfg)
    legacy_filename = legacy_visual_interpolator_cache_filename(cfg)
    return (
        primary=joinpath(cfg.cache_dir, filename),
        row_tag=joinpath(cfg.cache_dir, row_tag_filename),
        primary_legacy_name=joinpath(cfg.cache_dir, legacy_filename),
        legacy=joinpath(repo_root(), filename),
        legacy_row_tag=joinpath(repo_root(), row_tag_filename),
        legacy_legacy_name=joinpath(repo_root(), legacy_filename)
    )
end

function same_cache_path(path_a::AbstractString, path_b::AbstractString)
    return normpath(String(path_a)) == normpath(String(path_b))
end

function unique_cache_paths(paths)
    unique_paths = String[]
    for path in paths
        any(existing -> same_cache_path(existing, path), unique_paths) && continue
        push!(unique_paths, String(path))
    end
    return unique_paths
end

function visual_interpolator_cache_candidates(cfg::VisualConfig)
    paths = visual_interpolator_cache_paths(cfg)
    candidates = String[
        paths.primary,
        paths.row_tag,
        paths.primary_legacy_name,
        paths.legacy,
        paths.legacy_row_tag,
        paths.legacy_legacy_name
    ]
    for alias_tag in split_sobol_cache_param_tag_aliases(cfg)
        alias_filename = visual_interpolator_filename_for_param_tag(cfg, alias_tag; include_cosmology=true)
        alias_legacy_filename = visual_interpolator_filename_for_param_tag(cfg, alias_tag; include_cosmology=false)
        push!(candidates, joinpath(cfg.cache_dir, alias_filename))
        push!(candidates, joinpath(cfg.cache_dir, alias_legacy_filename))
        push!(candidates, joinpath(repo_root(), alias_filename))
        push!(candidates, joinpath(repo_root(), alias_legacy_filename))
    end
    return unique_cache_paths(candidates)
end

function existing_visual_interpolator_cache(cfg::VisualConfig)
    paths = visual_interpolator_cache_paths(cfg)
    candidates = visual_interpolator_cache_candidates(cfg)
    for candidate in candidates
        if isfile(candidate)
            if !same_cache_path(candidate, paths.primary)
                println("Primary interpolator cache not found at $(paths.primary)")
                println("Using fallback interpolator cache: $(candidate)")
            end
            return candidate
        end
    end
    return nothing
end

function cache_file_settled(path::AbstractString; settle_seconds::Real=10.0)
    try
        info = stat(path)
        return info.size > 0 && (time() - info.mtime) >= Float64(settle_seconds)
    catch
        return false
    end
end

function wait_for_visual_interpolator_cache(cfg::VisualConfig)
    deadline = time() + cfg.cache_wait_seconds
    last_notice = 0.0

    while true
        existing_cache = existing_visual_interpolator_cache(cfg)
        if existing_cache !== nothing
            cache_file_settled(existing_cache) && return existing_cache
        end

        remaining = deadline - time()
        remaining <= 0.0 && return nothing

        now = time()
        if now - last_notice >= max(cfg.cache_poll_seconds, 1.0)
            if existing_cache === nothing
                println(
                    "Waiting for tSZ interpolator cache for $(cfg.cache_param_tag); " *
                    "$(round(remaining; digits=1)) s remaining."
                )
            else
                println(
                    "Waiting for tSZ interpolator cache write to settle: $(existing_cache); " *
                    "$(round(remaining; digits=1)) s remaining."
                )
            end
            last_notice = now
        end
        sleep(min(cfg.cache_poll_seconds, remaining))
    end
end

function missing_interpolator_cache_error(cfg::VisualConfig)
    checked_paths = join(visual_interpolator_cache_candidates(cfg), ", ")
    return (
        "model_exists=true, but no tSZ interpolator cache was found. Checked $(checked_paths). " *
        "The Battaglia interpolator build is expensive and can leave little memory for NSIDE=4096 map allocation. " *
        "Copy an existing cache to cache_dir=$(cfg.cache_dir), set cache_dir to the directory that already contains it, " *
        "or rerun deliberately with model_exists=false using enough walltime/memory."
    )
end

function build_visual_interpolator(cfg::VisualConfig)
    model = build_tsz_model(cfg)
    paths = visual_interpolator_cache_paths(cfg)
    load_existing_cache = false
    y_cache_file = if cfg.model_exists
        existing_cache = cfg.cache_wait_seconds > 0.0 ? wait_for_visual_interpolator_cache(cfg) : existing_visual_interpolator_cache(cfg)
        existing_cache === nothing && error(missing_interpolator_cache_error(cfg))
        println("Loading tSZ interpolator cache: $(existing_cache)")
        load_existing_cache = true
        existing_cache
    elseif cfg.reuse_existing_cache
        existing_cache = existing_visual_interpolator_cache(cfg)
        if existing_cache !== nothing
            println("Reusing existing tSZ interpolator cache: $(existing_cache)")
            load_existing_cache = true
            existing_cache
        else
            isdir(cfg.cache_dir) || mkpath(cfg.cache_dir)
            println(
                "Building tSZ interpolator cache: $(paths.primary) " *
                "(pad=$(cfg.interpolator_pad), logM_max=$(cfg.interpolator_logM_max))"
            )
            paths.primary
        end
    else
        isdir(cfg.cache_dir) || mkpath(cfg.cache_dir)
        println(
            "Building tSZ interpolator cache: $(paths.primary) " *
            "(pad=$(cfg.interpolator_pad), logM_max=$(cfg.interpolator_logM_max))"
        )
        paths.primary
    end

    if load_existing_cache
        println("Interpolator cache parameter tag: $(cfg.cache_param_tag)")
    end

    interp_t0 = start_phase_timing()
    cleanup_env_key = "XGPAINT_CLEANUP_NONPOSITIVE"
    previous_cleanup_env = get(ENV, cleanup_env_key, nothing)
    ENV[cleanup_env_key] = cfg.cleanup_nonpositive_profile_values ? "true" : "false"

    interpolator = try
        if load_existing_cache
            build_interpolator(
                model,
                cache_file=y_cache_file,
                overwrite=false
            )
        else
            build_interpolator(
                model;
                cache_file=y_cache_file,
                pad=cfg.interpolator_pad,
                logM_max=cfg.interpolator_logM_max,
                overwrite=true,
                verbose=true
            )
        end
    finally
        if previous_cleanup_env === nothing
            pop!(ENV, cleanup_env_key, nothing)
        else
            ENV[cleanup_env_key] = previous_cleanup_env
        end
    end

    print_phase_usage("Interpolator", interp_t0)
    return interpolator
end
