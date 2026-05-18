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

function ensure_arg!(args::Vector{String}, key::AbstractString, value)
    any(arg -> arg_matches_key(arg, key), args) || push!(args, string(key, "=", value))
    return args
end

replace_or_add_arg!(ARGS, "batching_mode", "full")
replace_or_add_arg!(ARGS, "save_bin_maps", "false")
replace_or_add_arg!(ARGS, "save_cl", "true")
ensure_arg!(ARGS, "catalog_source", "halfdome")

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))
run_tsz_visual_fits()
