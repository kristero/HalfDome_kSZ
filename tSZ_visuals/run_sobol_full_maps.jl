function arg_matches_key(arg::AbstractString, key::AbstractString)
    return startswith(arg, key * "=") || startswith(arg, "--" * key * "=")
end

function replace_or_add_arg!(args::Vector{String}, key::AbstractString, value)
    new_arg = string(key, "=", value)
    for idx in eachindex(args)
        if arg_matches_key(args[idx], key)
            args[idx] = new_arg
            return args
        end
    end
    push!(args, new_arg)
    return args
end

function remove_arg_keys(args::Vector{String}, keys::Tuple)
    return [arg for arg in args if !any(key -> arg_matches_key(arg, key), keys)]
end

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))

function run_sobol_full_maps()
    original_args = copy(ARGS)

    sobol_csv_path_raw = get_string_arg("sobol_csv_path", CLUSTER_SOBOL_CSV_DEFAULT; env="BATTAGLIA_SOBOL_CSV")
    sobol_csv_path = resolve_repo_path(sobol_csv_path_raw)
    isfile(sobol_csv_path) || error("Sobol CSV file not found: $(sobol_csv_path)")

    raw_table, _ = readdlm(sobol_csv_path, ','; header=true)
    total_rows = size(raw_table, 1)

    slurm_array_task_id = strip(get(ENV, "SLURM_ARRAY_TASK_ID", ""))
    if !isempty(slurm_array_task_id)
        slurm_row = parse(Int, slurm_array_task_id)
        sobol_row_start = slurm_row
        sobol_row_stop = slurm_row
    else
        sobol_row_start = get_int_arg("sobol_row_start", 1; env="BATTAGLIA_SOBOL_ROW_START")
        sobol_row_stop = get_int_arg("sobol_row_stop", total_rows; env="BATTAGLIA_SOBOL_ROW_STOP")
    end
    sobol_row_start >= 1 || error("sobol_row_start must be >= 1.")
    sobol_row_stop >= sobol_row_start || error("sobol_row_stop must be >= sobol_row_start.")
    sobol_row_stop <= total_rows || error("sobol_row_stop=$(sobol_row_stop) exceeds the CSV row count $(total_rows).")

    shared_args = remove_arg_keys(copy(original_args), ("sobol_row", "sobol_row_start", "sobol_row_stop"))
    replace_or_add_arg!(shared_args, "batching_mode", "full")
    replace_or_add_arg!(shared_args, "save_bin_maps", "false")
    replace_or_add_arg!(shared_args, "save_cl", "true")
    replace_or_add_arg!(shared_args, "sobol_csv_path", sobol_csv_path)
    if !any(arg -> arg_matches_key(arg, "catalog_source"), shared_args)
        push!(shared_args, "catalog_source=halfdome")
    end

    try
        for sobol_row in sobol_row_start:sobol_row_stop
            run_args = copy(shared_args)
            replace_or_add_arg!(run_args, "sobol_row", sobol_row)
            empty!(ARGS)
            append!(ARGS, run_args)
            println("Starting full-map run for Sobol row $(sobol_row) / $(total_rows).")
            run_tsz_visual_fits()
        end
    finally
        empty!(ARGS)
        append!(ARGS, original_args)
    end
end

run_sobol_full_maps()
