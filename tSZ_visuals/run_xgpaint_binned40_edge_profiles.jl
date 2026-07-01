using Printf

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))

const EDGE_PARAM_NAMES = (
    "P0",
    "xc",
    "beta",
    "alpha_m_P0",
    "alpha_m_xc",
    "alpha_m_beta",
    "alpha_z_P0",
    "alpha_z_xc",
    "alpha_z_beta"
)

const EDGE_BATTAGLIA_ARG_KEYS = (
    "battaglia_P0_amp",
    "battaglia_x_c_amp",
    "battaglia_beta_amp",
    "battaglia_P0_alpha_m",
    "battaglia_x_c_alpha_m",
    "battaglia_beta_alpha_m",
    "battaglia_P0_alpha_z",
    "battaglia_x_c_alpha_z",
    "battaglia_beta_alpha_z"
)

const EDGE_BATTAGLIA_ENV_KEYS = (
    "BATTAGLIA_P0_AMP",
    "BATTAGLIA_X_C_AMP",
    "BATTAGLIA_BETA_AMP",
    "BATTAGLIA_P0_ALPHA_M",
    "BATTAGLIA_X_C_ALPHA_M",
    "BATTAGLIA_BETA_ALPHA_M",
    "BATTAGLIA_P0_ALPHA_Z",
    "BATTAGLIA_X_C_ALPHA_Z",
    "BATTAGLIA_BETA_ALPHA_Z",
    "BATTAGLIA_ALPHA_AMP",
    "BATTAGLIA_ALPHA_ALPHA_M",
    "BATTAGLIA_ALPHA_ALPHA_Z",
    "BATTAGLIA_GAMMA_AMP",
    "BATTAGLIA_GAMMA_ALPHA_M",
    "BATTAGLIA_GAMMA_ALPHA_Z",
    "BATTAGLIA_SOBOL_CSV",
    "BATTAGLIA_SOBOL_ROW"
)

const EDGE_FIXED_BATTAGLIA_DEFAULT_ARGS = (
    "battaglia_alpha_amp" => 1.0,
    "battaglia_alpha_alpha_m" => 0.0,
    "battaglia_alpha_alpha_z" => 0.0,
    "battaglia_gamma_amp" => -0.3,
    "battaglia_gamma_alpha_m" => 0.0,
    "battaglia_gamma_alpha_z" => 0.0
)

const EDGE_BATTAGLIA12_THETA = [
    18.1,
    0.497,
    4.35,
    0.154,
    -0.00865,
    0.0393,
    -0.758,
    0.731,
    0.415
]

const EDGE_PRIOR_LOW = [
    1.8325239419937134,
    0.15001100301742554,
    3.4806270599365234,
    0.000311999989207834,
    -0.09971799701452255,
    -0.0199350006878376,
    -1.3634569644927979,
    0.14739300310611725,
    0.08380799740552902
]

const EDGE_PRIOR_HIGH = [
    34.34122085571289,
    0.8445029854774475,
    5.216610908508301,
    0.29225099086761475,
    0.09979499876499176,
    0.09976699948310852,
    -0.2288389950990677,
    1.3144739866256714,
    0.7458840012550354
]

const SO_B40_EDGES = vcat(collect(80:200:7880), [7979])

function arg_matches_key(arg::AbstractString, key::AbstractString)
    return startswith(arg, key * "=") || startswith(arg, "--" * key * "=")
end

function remove_arg_keys!(args::Vector{String}, keys)
    filter!(args) do arg
        !any(key -> arg_matches_key(arg, String(key)), keys)
    end
    return args
end

function set_arg!(args::Vector{String}, key::AbstractString, value)
    remove_arg_keys!(args, (key,))
    push!(args, "$(key)=$(value)")
    return args
end

function format_float_arg(x::Real)
    return @sprintf("%.17g", Float64(x))
end

function clear_inherited_battaglia_env!()
    for key in EDGE_BATTAGLIA_ENV_KEYS
        if haskey(ENV, key)
            println("Ignoring inherited $(key) for edge-profile generation.")
            delete!(ENV, key)
        end
    end
    return nothing
end

function default_edge_profile_index()
    for key in ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID")
        task_id = strip(get(ENV, key, ""))
        isempty(task_id) || return parse(Int, task_id)
    end
    return 0
end

function edge_profile_specs()
    specs = NamedTuple[]
    push!(
        specs,
        (
            index=1,
            label="battaglia12",
            varied_param="",
            edge="fiducial",
            edge_value=NaN,
            theta=copy(EDGE_BATTAGLIA12_THETA)
        )
    )

    profile_index = 2
    for (j, param_name) in enumerate(EDGE_PARAM_NAMES)
        theta_low = copy(EDGE_BATTAGLIA12_THETA)
        theta_low[j] = EDGE_PRIOR_LOW[j]
        push!(
            specs,
            (
                index=profile_index,
                label="$(param_name)_low",
                varied_param=param_name,
                edge="low",
                edge_value=EDGE_PRIOR_LOW[j],
                theta=theta_low
            )
        )
        profile_index += 1

        theta_high = copy(EDGE_BATTAGLIA12_THETA)
        theta_high[j] = EDGE_PRIOR_HIGH[j]
        push!(
            specs,
            (
                index=profile_index,
                label="$(param_name)_high",
                varied_param=param_name,
                edge="high",
                edge_value=EDGE_PRIOR_HIGH[j],
                theta=theta_high
            )
        )
        profile_index += 1
    end
    return specs
end

function build_profile_args(original_args::Vector{String}, spec)
    args = copy(original_args)
    fixed_arg_keys = [String(pair.first) for pair in EDGE_FIXED_BATTAGLIA_DEFAULT_ARGS]
    remove_arg_keys!(
        args,
        vcat(
            collect(EDGE_BATTAGLIA_ARG_KEYS),
            fixed_arg_keys,
            [
                "catalog_source",
                "batching_mode",
                "save_healpix_map",
                "save_mass_map",
                "save_bin_maps",
                "save_cl",
                "sobol_csv_path",
                "sobol_row",
                "run_instance_tag"
            ]
        )
    )

    set_arg!(args, "catalog_source", "halfdome")
    set_arg!(args, "batching_mode", "full")
    set_arg!(args, "save_healpix_map", "false")
    set_arg!(args, "save_mass_map", "false")
    set_arg!(args, "save_bin_maps", "false")
    set_arg!(args, "save_cl", "false")
    set_arg!(args, "sobol_csv_path", "")
    set_arg!(args, "sobol_row", "0")
    set_arg!(args, "run_instance_tag", spec.label)

    for (key, value) in EDGE_FIXED_BATTAGLIA_DEFAULT_ARGS
        set_arg!(args, key, format_float_arg(value))
    end
    for (key, value) in zip(EDGE_BATTAGLIA_ARG_KEYS, spec.theta)
        set_arg!(args, key, format_float_arg(value))
    end
    return args
end

function write_npy_float64_vector(path::AbstractString, values, label::AbstractString)
    abs_path = abspath(path)
    parent_dir = dirname(abs_path)
    isdir(parent_dir) || mkpath(parent_dir)

    data = vec(Float64.(collect(values)))
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': ($(length(data)),), }"
    preamble_nbytes = 10
    header_nbytes_without_padding = length(codeunits(header)) + 1
    padding_nbytes = mod(-(preamble_nbytes + header_nbytes_without_padding), 16)
    padded_header = header * repeat(" ", padding_nbytes) * "\n"
    padded_header_nbytes = length(codeunits(padded_header))
    padded_header_nbytes <= typemax(UInt16) || error(
        "NumPy v1.0 header too large for $(abs_path): $(padded_header_nbytes) bytes."
    )

    tmp_path = abs_path * ".tmp_" * string(getpid())
    open(tmp_path, "w") do io
        write(io, UInt8(0x93))
        write(io, codeunits("NUMPY"))
        write(io, UInt8(0x01))
        write(io, UInt8(0x00))
        write(io, UInt8(padded_header_nbytes & 0xff))
        write(io, UInt8((padded_header_nbytes >> 8) & 0xff))
        write(io, codeunits(padded_header))
        write(io, data)
    end
    mv(tmp_path, abs_path; force=true)

    println("Saved $(label) NumPy array to $(abs_path)")
    return abs_path
end

function weighted_mean(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    return sum(Float64.(values) .* Float64.(weights)) / sum(Float64.(weights))
end

function apply_gaussian_beam_to_cl(cfg::VisualConfig, cl)
    cl_values = Float64.(collect(cl))
    cfg.apply_gaussian_beam || return cl_values

    fwhm_rad = deg2rad(cfg.gaussian_beam_fwhm_arcmin / 60.0)
    sigma_rad = fwhm_rad / sqrt(8.0 * log(2.0))
    for ell in 0:(length(cl_values) - 1)
        cl_values[ell + 1] *= exp(-Float64(ell) * Float64(ell + 1) * sigma_rad^2)
    end
    return cl_values
end

function bin_cl_to_so_binned40_log10_dl(cl; floor_dl::Real=1.0e-40)
    lmax_needed = last(SO_B40_EDGES)
    length(cl) >= lmax_needed + 1 || error(
        "C_l vector has length $(length(cl)); need at least $(lmax_needed + 1) for ell=$(lmax_needed)."
    )

    ell_centers = Float64[]
    binned_log10_dl = Float64[]

    for bin_idx in 1:(length(SO_B40_EDGES) - 1)
        ell_min = SO_B40_EDGES[bin_idx]
        ell_max = SO_B40_EDGES[bin_idx + 1]
        ell_values = bin_idx == length(SO_B40_EDGES) - 1 ? collect(ell_min:ell_max) : collect(ell_min:(ell_max - 1))
        weights = 2.0 .* Float64.(ell_values) .+ 1.0

        dl_values = [
            max(Float64(cl[ell + 1]) * Float64(ell) * Float64(ell + 1) / (2.0 * pi), Float64(floor_dl))
            for ell in ell_values
        ]

        push!(ell_centers, weighted_mean(Float64.(ell_values), weights))
        push!(binned_log10_dl, weighted_mean(log10.(dl_values), weights))
    end

    return ell_centers, binned_log10_dl
end

function write_profile_metadata(path::AbstractString, spec, cfg::VisualConfig, profile_path::AbstractString, theta_path::AbstractString)
    abs_path = abspath(path)
    isdir(dirname(abs_path)) || mkpath(dirname(abs_path))
    theta_names = join(EDGE_PARAM_NAMES, ";")
    theta_values = join(map(format_float_arg, spec.theta), ";")
    open(abs_path, "w") do io
        println(io, "label,$(spec.label)")
        println(io, "profile_index,$(spec.index)")
        println(io, "varied_param,$(spec.varied_param)")
        println(io, "edge,$(spec.edge)")
        println(io, "edge_value,$(spec.edge_value)")
        println(io, "profile_path,$(abspath(profile_path))")
        println(io, "theta_path,$(abspath(theta_path))")
        println(io, "halfdome_path,$(abspath(cfg.halfdome_path))")
        println(io, "nside,$(cfg.nside)")
        println(io, "cl_lmax,$(cfg.cl_lmax)")
        println(io, "apply_gaussian_beam,$(cfg.apply_gaussian_beam)")
        println(io, "gaussian_beam_fwhm_arcmin,$(cfg.gaussian_beam_fwhm_arcmin)")
        println(io, "target_kind,binned_log10_D_ell")
        println(io, "theta_names,$(theta_names)")
        println(io, "theta_values,$(theta_values)")
    end
    println("Saved profile metadata to $(abs_path)")
    return abs_path
end

function run_one_edge_profile(original_args::Vector{String}, spec)
    t0 = time()
    run_args = build_profile_args(original_args, spec)
    empty!(ARGS)
    append!(ARGS, run_args)

    cfg = try
        load_visual_config()
    catch err
        if err isa SkipVisualRun
            println("Skipping edge profile $(spec.label): $(err.message)")
            return nothing
        end
        rethrow()
    end

    cfg.catalog_source == "halfdome" || error("This script is only for HalfDome.")
    cfg.batching_mode == "full" || error("This script requires batching_mode=full.")
    cfg.cl_lmax >= last(SO_B40_EDGES) || error(
        "cl_lmax=$(cfg.cl_lmax) is too small. Set cl_lmax=$(last(SO_B40_EDGES)) for SO binned-40 profiles."
    )

    output_dir = abspath(get_string_arg("edge_profile_output_dir", cfg.output_dir; env="EDGE_PROFILE_OUTPUT_DIR"))
    isdir(output_dir) || mkpath(output_dir)

    profile_path = joinpath(output_dir, "$(spec.label)_log10_dl.npy")
    theta_path = joinpath(output_dir, "$(spec.label)_theta.npy")
    metadata_path = joinpath(output_dir, "$(spec.label)_metadata.txt")
    ell_path = joinpath(output_dir, "ell.npy")
    overwrite = get_bool_arg("overwrite_edge_profile", true; env="OVERWRITE_EDGE_PROFILE")
    save_metadata = get_bool_arg("save_edge_metadata", false; env="SAVE_EDGE_METADATA")

    if isfile(profile_path) && !overwrite
        println("Skipping existing profile $(abspath(profile_path)); set overwrite_edge_profile=true to replace it.")
        return profile_path
    end

    println("=======================================")
    println("Running XGPaint binned-40 edge profile $(spec.index): $(spec.label)")
    println("Output directory: $(output_dir)")
    println("No emulator; no SO noise; no mask.")
    theta_display = join(map(format_float_arg, spec.theta), ", ")
    println("Battaglia theta: $(theta_display)")
    print_visual_config(cfg)
    if get_bool_arg("print_runtime_environment", false; env="PRINT_RUNTIME_ENVIRONMENT")
        safe_print_runtime_environment()
    end
    println("=======================================")

    y_model_interp = build_visual_interpolator(cfg)
    trim_process_memory()
    state = init_visual_maps(cfg)

    paint_t0 = start_phase_timing()
    run_halfdome_visuals!(cfg, state, y_model_interp)
    print_phase_usage("Painting", paint_t0)

    cl_t0 = start_phase_timing()
    cl = apply_gaussian_beam_to_cl(cfg, compute_cl(cfg, state.m_hp))
    ell_binned, profile_log10_dl = bin_cl_to_so_binned40_log10_dl(cl)
    print_phase_usage("C_l and binned profile", cl_t0)

    write_npy_float64_vector(ell_path, ell_binned, "SO binned-40 ell centers")
    write_npy_float64_vector(profile_path, profile_log10_dl, "$(spec.label) binned log10(D_l)")
    write_npy_float64_vector(theta_path, spec.theta, "$(spec.label) Battaglia theta")
    if save_metadata
        write_profile_metadata(metadata_path, spec, cfg, profile_path, theta_path)
    end

    state = nothing
    cl = nothing
    GC.gc()

    println("Finished $(spec.label) in $(round(time() - t0; digits=2)) s.")
    return profile_path
end

function run_xgpaint_binned40_edge_profiles()
    original_args = copy(ARGS)
    clear_inherited_battaglia_env!()

    specs = edge_profile_specs()
    selected_index = get_int_arg("edge_profile_index", default_edge_profile_index(); env="EDGE_PROFILE_INDEX")
    if selected_index == 0
        println("edge_profile_index=0: running all $(length(specs)) profiles sequentially.")
        return [run_one_edge_profile(original_args, spec) for spec in specs]
    end

    1 <= selected_index <= length(specs) || error(
        "edge_profile_index=$(selected_index) is invalid. Use 1:$(length(specs)), or 0 for all profiles."
    )
    return run_one_edge_profile(original_args, specs[selected_index])
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_xgpaint_binned40_edge_profiles()
end
