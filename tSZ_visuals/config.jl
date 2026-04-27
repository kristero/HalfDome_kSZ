using DelimitedFiles

const TSZ_H_VALUE = 0.68
const TSZ_C_KMS = 299_792.458
const TSZ_OMEGAB = 0.049
const TSZ_OMEGAC = 0.31 - TSZ_OMEGAB
const TSZ_OMEGAM = TSZ_OMEGAB + TSZ_OMEGAC
const TSZ_H0 = 100.0 * TSZ_H_VALUE
const TSZ_RHO_M = 2.775e11 * TSZ_OMEGAM * TSZ_H_VALUE^2

const BATTAGLIA_P0_AMP_DEFAULT = 18.1
const BATTAGLIA_P0_ALPHA_M_DEFAULT = 0.154
const BATTAGLIA_P0_ALPHA_Z_DEFAULT = -0.758
const BATTAGLIA_X_C_AMP_DEFAULT = 0.497
const BATTAGLIA_X_C_ALPHA_M_DEFAULT = -0.00865
const BATTAGLIA_X_C_ALPHA_Z_DEFAULT = 0.731
const BATTAGLIA_BETA_AMP_DEFAULT = 4.35
const BATTAGLIA_BETA_ALPHA_M_DEFAULT = 0.0393
const BATTAGLIA_BETA_ALPHA_Z_DEFAULT = 0.415
const BATTAGLIA_ALPHA_AMP_DEFAULT = 1.0
const BATTAGLIA_ALPHA_ALPHA_M_DEFAULT = 0.0
const BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT = 0.0
const BATTAGLIA_GAMMA_AMP_DEFAULT = -0.3
const BATTAGLIA_GAMMA_ALPHA_M_DEFAULT = 0.0
const BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT = 0.0

Base.@kwdef struct VisualConfig
    model_exists::Bool
    save_healpix_map::Bool
    save_mass_map::Bool
    save_cl::Bool
    save_bin_maps::Bool
    apply_mass_cut::Bool
    cumulative_bin_maps::Bool
    catalog_source::String
    halfdome_path::String
    websky_path::String
    catalog_path::String
    simulation_name::String
    simulation_tag::String
    nside::Int
    chunkN::Int
    add_str_end::String
    mass_min::Float64
    batching_mode::String
    redshift_binning_mode::String
    redshift_bin_width::Float64
    log_redshift_bin_width::Float64
    mass_bin_width_dex::Float64
    sobol_csv_path::String
    sobol_row::Int
    output_dir::String
    param_tag::String
    binning_tag::String
    bin_map_mode_tag::String
    run_tag::String
    fits_output_path::String
    mass_fits_output_path::String
    cl_output_path::String
    battaglia_params::NamedTuple
end

function repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function resolve_repo_path(path::AbstractString)
    return isabspath(path) ? String(path) : normpath(joinpath(repo_root(), path))
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
    if env !== nothing && haskey(ENV, env)
        return parse_bool_arg(ENV[env])
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse_bool_arg(split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse_bool_arg(split(a, "=", limit=2)[2])
        end
    end
    return Bool(default)
end

function get_int_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return parse(Int, ENV[env])
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse(Int, split(a, "=", limit=2)[2])
        end
    end
    return Int(default)
end

function get_float_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return parse(Float64, ENV[env])
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, prefix2)
            return parse(Float64, split(a, "=", limit=2)[2])
        end
    end
    return Float64(default)
end

function get_string_arg(key, default; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return ENV[env]
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1)
            return split(a, "=", limit=2)[2]
        elseif startswith(a, prefix2)
            return split(a, "=", limit=2)[2]
        end
    end
    return String(default)
end

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

function safe_filename_tag(s::AbstractString)
    tag = lowercase(strip(String(s)))
    tag = replace(tag, r"[^A-Za-z0-9_+\-.]+" => "_")
    tag = replace(tag, r"_+" => "_")
    return isempty(tag) ? "simulation" : tag
end

function normalize_batching_mode(mode_raw::AbstractString)
    mode = lowercase(strip(mode_raw))
    mode in ("redshift", "z", "zbin") && return "redshift"
    mode in ("mass", "m", "mbin") && return "mass"
    mode in ("initial", "chunk", "chunks", "catalog") && return "initial"
    mode in ("full", "all", "fullmap") && return "full"
    error("Unsupported batching_mode=$(repr(mode_raw)). Use redshift, mass, initial, or full.")
end

function normalize_redshift_binning_mode(mode_raw::AbstractString)
    mode = lowercase(strip(mode_raw))
    mode in ("linear", "lin", "z") && return "linear"
    mode in ("log", "logz", "log1p") && return "log1p"
    error("Unsupported redshift_binning_mode=$(repr(mode_raw)). Use linear or log1p.")
end

function battaglia_namedtuple(;
    P0_amp,
    P0_alpha_m,
    P0_alpha_z,
    x_c_amp,
    x_c_alpha_m,
    x_c_alpha_z,
    beta_amp,
    beta_alpha_m,
    beta_alpha_z,
    alpha_amp,
    alpha_alpha_m,
    alpha_alpha_z,
    gamma_amp,
    gamma_alpha_m,
    gamma_alpha_z
)
    return (
        P0_amp=Float64(P0_amp),
        P0_alpha_m=Float64(P0_alpha_m),
        P0_alpha_z=Float64(P0_alpha_z),
        x_c_amp=Float64(x_c_amp),
        x_c_alpha_m=Float64(x_c_alpha_m),
        x_c_alpha_z=Float64(x_c_alpha_z),
        beta_amp=Float64(beta_amp),
        beta_alpha_m=Float64(beta_alpha_m),
        beta_alpha_z=Float64(beta_alpha_z),
        alpha_amp=Float64(alpha_amp),
        alpha_alpha_m=Float64(alpha_alpha_m),
        alpha_alpha_z=Float64(alpha_alpha_z),
        gamma_amp=Float64(gamma_amp),
        gamma_alpha_m=Float64(gamma_alpha_m),
        gamma_alpha_z=Float64(gamma_alpha_z)
    )
end

function default_battaglia_params()
    return battaglia_namedtuple(
        P0_amp=BATTAGLIA_P0_AMP_DEFAULT,
        P0_alpha_m=BATTAGLIA_P0_ALPHA_M_DEFAULT,
        P0_alpha_z=BATTAGLIA_P0_ALPHA_Z_DEFAULT,
        x_c_amp=BATTAGLIA_X_C_AMP_DEFAULT,
        x_c_alpha_m=BATTAGLIA_X_C_ALPHA_M_DEFAULT,
        x_c_alpha_z=BATTAGLIA_X_C_ALPHA_Z_DEFAULT,
        beta_amp=BATTAGLIA_BETA_AMP_DEFAULT,
        beta_alpha_m=BATTAGLIA_BETA_ALPHA_M_DEFAULT,
        beta_alpha_z=BATTAGLIA_BETA_ALPHA_Z_DEFAULT,
        alpha_amp=BATTAGLIA_ALPHA_AMP_DEFAULT,
        alpha_alpha_m=BATTAGLIA_ALPHA_ALPHA_M_DEFAULT,
        alpha_alpha_z=BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT,
        gamma_amp=BATTAGLIA_GAMMA_AMP_DEFAULT,
        gamma_alpha_m=BATTAGLIA_GAMMA_ALPHA_M_DEFAULT,
        gamma_alpha_z=BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT
    )
end

function battaglia_params_from_args()
    return battaglia_namedtuple(
        P0_amp=get_float_arg("battaglia_P0_amp", BATTAGLIA_P0_AMP_DEFAULT; env="BATTAGLIA_P0_AMP"),
        P0_alpha_m=get_float_arg("battaglia_P0_alpha_m", BATTAGLIA_P0_ALPHA_M_DEFAULT; env="BATTAGLIA_P0_ALPHA_M"),
        P0_alpha_z=get_float_arg("battaglia_P0_alpha_z", BATTAGLIA_P0_ALPHA_Z_DEFAULT; env="BATTAGLIA_P0_ALPHA_Z"),
        x_c_amp=get_float_arg("battaglia_x_c_amp", BATTAGLIA_X_C_AMP_DEFAULT; env="BATTAGLIA_X_C_AMP"),
        x_c_alpha_m=get_float_arg("battaglia_x_c_alpha_m", BATTAGLIA_X_C_ALPHA_M_DEFAULT; env="BATTAGLIA_X_C_ALPHA_M"),
        x_c_alpha_z=get_float_arg("battaglia_x_c_alpha_z", BATTAGLIA_X_C_ALPHA_Z_DEFAULT; env="BATTAGLIA_X_C_ALPHA_Z"),
        beta_amp=get_float_arg("battaglia_beta_amp", BATTAGLIA_BETA_AMP_DEFAULT; env="BATTAGLIA_BETA_AMP"),
        beta_alpha_m=get_float_arg("battaglia_beta_alpha_m", BATTAGLIA_BETA_ALPHA_M_DEFAULT; env="BATTAGLIA_BETA_ALPHA_M"),
        beta_alpha_z=get_float_arg("battaglia_beta_alpha_z", BATTAGLIA_BETA_ALPHA_Z_DEFAULT; env="BATTAGLIA_BETA_ALPHA_Z"),
        alpha_amp=get_float_arg("battaglia_alpha_amp", BATTAGLIA_ALPHA_AMP_DEFAULT; env="BATTAGLIA_ALPHA_AMP"),
        alpha_alpha_m=get_float_arg("battaglia_alpha_alpha_m", BATTAGLIA_ALPHA_ALPHA_M_DEFAULT; env="BATTAGLIA_ALPHA_ALPHA_M"),
        alpha_alpha_z=get_float_arg("battaglia_alpha_alpha_z", BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT; env="BATTAGLIA_ALPHA_ALPHA_Z"),
        gamma_amp=get_float_arg("battaglia_gamma_amp", BATTAGLIA_GAMMA_AMP_DEFAULT; env="BATTAGLIA_GAMMA_AMP"),
        gamma_alpha_m=get_float_arg("battaglia_gamma_alpha_m", BATTAGLIA_GAMMA_ALPHA_M_DEFAULT; env="BATTAGLIA_GAMMA_ALPHA_M"),
        gamma_alpha_z=get_float_arg("battaglia_gamma_alpha_z", BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT; env="BATTAGLIA_GAMMA_ALPHA_Z")
    )
end

function normalize_csv_header_name(name::AbstractString)
    normalized = lowercase(strip(String(name)))
    normalized = replace(normalized, r"[^a-z0-9]+" => "_")
    normalized = replace(normalized, r"_+" => "_")
    return strip(normalized, '_')
end

function parse_csv_float(value)
    value isa Number && return Float64(value)
    return parse(Float64, strip(String(value)))
end

function battaglia_params_from_sobol_row(
    csv_path::AbstractString,
    sobol_row::Integer;
    base_params=default_battaglia_params()
)
    isfile(csv_path) || error("Sobol CSV file not found: $(csv_path)")
    sobol_row >= 1 || error("sobol_row must be >= 1.")

    raw_table, header = readdlm(csv_path, ','; header=true)
    nrows = size(raw_table, 1)
    sobol_row <= nrows || error("sobol_row=$(sobol_row) exceeds the CSV row count $(nrows).")

    header_names = vec(header)
    header_lookup = Dict{String, Int}()
    for (col_idx, header_name) in enumerate(header_names)
        header_lookup[normalize_csv_header_name(String(header_name))] = col_idx
    end

    defaults = Dict{Symbol, Float64}(k => v for (k, v) in pairs(base_params))
    column_specs = (
        (field=:P0_amp, names=("p0",)),
        (field=:x_c_amp, names=("xc", "x_c")),
        (field=:beta_amp, names=("beta",)),
        (field=:P0_alpha_m, names=("alpha_m_p0",)),
        (field=:x_c_alpha_m, names=("alpha_m_xc", "alpha_m_x_c")),
        (field=:beta_alpha_m, names=("alpha_m_beta",)),
        (field=:P0_alpha_z, names=("alpha_z_p0",)),
        (field=:x_c_alpha_z, names=("alpha_z_xc", "alpha_z_x_c")),
        (field=:beta_alpha_z, names=("alpha_z_beta",))
    )

    for spec in column_specs
        col_idx = 0
        for candidate_name in spec.names
            if haskey(header_lookup, candidate_name)
                col_idx = header_lookup[candidate_name]
                break
            end
        end
        col_idx > 0 || error("Missing required Sobol CSV column for $(spec.field). Expected one of $(collect(spec.names)).")
        defaults[spec.field] = parse_csv_float(raw_table[sobol_row, col_idx])
    end

    return battaglia_namedtuple(
        P0_amp=defaults[:P0_amp],
        P0_alpha_m=defaults[:P0_alpha_m],
        P0_alpha_z=defaults[:P0_alpha_z],
        x_c_amp=defaults[:x_c_amp],
        x_c_alpha_m=defaults[:x_c_alpha_m],
        x_c_alpha_z=defaults[:x_c_alpha_z],
        beta_amp=defaults[:beta_amp],
        beta_alpha_m=defaults[:beta_alpha_m],
        beta_alpha_z=defaults[:beta_alpha_z],
        alpha_amp=defaults[:alpha_amp],
        alpha_alpha_m=defaults[:alpha_alpha_m],
        alpha_alpha_z=defaults[:alpha_alpha_z],
        gamma_amp=defaults[:gamma_amp],
        gamma_alpha_m=defaults[:gamma_alpha_m],
        gamma_alpha_z=defaults[:gamma_alpha_z]
    )
end

function build_sobol_param_tag(csv_path::AbstractString, sobol_row::Integer)
    csv_tag = safe_filename_tag(splitext(basename(csv_path))[1])
    return "sobol_$(csv_tag)_row" * lpad(string(sobol_row), 4, '0')
end

function collect_param_tag_parts(p, reference_params=default_battaglia_params())
    parts = String[]
    p.P0_amp != reference_params.P0_amp && push!(parts, "battaglia_P0_amp_" * fmt_param_value(p.P0_amp))
    p.P0_alpha_m != reference_params.P0_alpha_m && push!(parts, "battaglia_P0_alpha_m_" * fmt_param_value(p.P0_alpha_m))
    p.P0_alpha_z != reference_params.P0_alpha_z && push!(parts, "battaglia_P0_alpha_z_" * fmt_param_value(p.P0_alpha_z))
    p.x_c_amp != reference_params.x_c_amp && push!(parts, "battaglia_x_c_amp_" * fmt_param_value(p.x_c_amp))
    p.x_c_alpha_m != reference_params.x_c_alpha_m && push!(parts, "battaglia_x_c_alpha_m_" * fmt_param_value(p.x_c_alpha_m))
    p.x_c_alpha_z != reference_params.x_c_alpha_z && push!(parts, "battaglia_x_c_alpha_z_" * fmt_param_value(p.x_c_alpha_z))
    p.beta_amp != reference_params.beta_amp && push!(parts, "battaglia_beta_amp_" * fmt_param_value(p.beta_amp))
    p.beta_alpha_m != reference_params.beta_alpha_m && push!(parts, "battaglia_beta_alpha_m_" * fmt_param_value(p.beta_alpha_m))
    p.beta_alpha_z != reference_params.beta_alpha_z && push!(parts, "battaglia_beta_alpha_z_" * fmt_param_value(p.beta_alpha_z))
    p.alpha_amp != reference_params.alpha_amp && push!(parts, "battaglia_alpha_amp_" * fmt_param_value(p.alpha_amp))
    p.alpha_alpha_m != reference_params.alpha_alpha_m && push!(parts, "battaglia_alpha_alpha_m_" * fmt_param_value(p.alpha_alpha_m))
    p.alpha_alpha_z != reference_params.alpha_alpha_z && push!(parts, "battaglia_alpha_alpha_z_" * fmt_param_value(p.alpha_alpha_z))
    p.gamma_amp != reference_params.gamma_amp && push!(parts, "battaglia_gamma_amp_" * fmt_param_value(p.gamma_amp))
    p.gamma_alpha_m != reference_params.gamma_alpha_m && push!(parts, "battaglia_gamma_alpha_m_" * fmt_param_value(p.gamma_alpha_m))
    p.gamma_alpha_z != reference_params.gamma_alpha_z && push!(parts, "battaglia_gamma_alpha_z_" * fmt_param_value(p.gamma_alpha_z))
    return parts
end

function build_param_tag(p)
    parts = collect_param_tag_parts(p)
    return isempty(parts) ? "base" : "base_plus_" * join(parts, "__")
end

function build_param_delta_tag(p, reference_params)
    parts = collect_param_tag_parts(p, reference_params)
    return isempty(parts) ? "" : "__plus_" * join(parts, "__")
end

function build_binning_tag(batching_mode::AbstractString, redshift_mode::AbstractString, dz::Real, dlogz::Real, dlogm::Real)
    batching_mode == "full" && return "fullmap"
    batching_mode == "initial" && return "initial_chunks"
    batching_mode == "mass" && return "masslog10_dlog$(fmt_param_value(Float64(dlogm)))"
    redshift_mode == "linear" && return "zlin_dz$(fmt_param_value(Float64(dz)))"
    return "zlog1p_dlog$(fmt_param_value(Float64(dlogz)))"
end

function build_bin_map_mode_tag(cumulative_bin_maps::Bool, save_bin_maps::Bool)
    !save_bin_maps && return "finalonly"
    return cumulative_bin_maps ? "cumulative" : "separate"
end

function load_visual_config()
    model_exists = get_bool_arg("model_exists", true; env="MODEL_EXISTS")
    save_healpix_map = get_bool_arg("save_healpix_map", true; env="SAVE_HEALPIX_MAP")
    save_mass_map = get_bool_arg("save_mass_map", true; env="SAVE_MASS_MAP")
    save_cl = get_bool_arg("save_cl", false; env="SAVE_CL")
    save_bin_maps = get_bool_arg("save_bin_maps", true; env="SAVE_BIN_MAPS")
    apply_mass_cut = get_bool_arg("apply_mass_cut", true; env="APPLY_MASS_CUT")
    cumulative_bin_maps = get_bool_arg("cumulative_bin_maps", true; env="CUMULATIVE_BIN_MAPS")
    catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env="TSZ_CATALOG_SOURCE"))
    catalog_source in ("halfdome", "websky") || error("Unsupported catalog_source=$(repr(catalog_source)).")

    halfdome_path = resolve_repo_path(get_string_arg("halfdome_path", "lightcone_100.hdf5"; env="HALFDOME_PATH"))
    websky_path = resolve_repo_path(get_string_arg("websky_path", "other_sims/sims/halos.pksc"; env="WEBSKY_PATH"))
    catalog_path = catalog_source == "halfdome" ? halfdome_path : websky_path
    simulation_name = get_string_arg("simulation_name", catalog_source; env="SIMULATION_NAME")
    simulation_tag = safe_filename_tag(simulation_name)

    nside = get_int_arg("nside", 1024; env="NSIDE")
    chunkN = get_int_arg("chunkN", 2_000_000; env="CHUNKN")
    add_str_end = get_string_arg("add_str_end", "13Msol_cutoff_HALO"; env="ADD_STR_END")
    mass_min = get_float_arg("mass_min", 1.0e12; env="MASS_MIN")
    batching_mode = normalize_batching_mode(get_string_arg("batching_mode", "redshift"; env="BATCHING_MODE"))
    redshift_binning_mode = normalize_redshift_binning_mode(get_string_arg("redshift_binning_mode", "linear"; env="REDSHIFT_BINNING_MODE"))
    redshift_bin_width = get_float_arg("redshift_bin_width", 1.0; env="REDSHIFT_BIN_WIDTH")
    log_redshift_bin_width = get_float_arg("log_redshift_bin_width", 0.2; env="LOG_REDSHIFT_BIN_WIDTH")
    mass_bin_width_dex = get_float_arg("mass_bin_width_dex", 0.5; env="MASS_BIN_WIDTH_DEX")
    sobol_csv_path_raw = strip(get_string_arg("sobol_csv_path", ""; env="BATTAGLIA_SOBOL_CSV"))
    sobol_row = get_int_arg("sobol_row", 0; env="BATTAGLIA_SOBOL_ROW")
    sobol_csv_path = isempty(sobol_csv_path_raw) ? "" : resolve_repo_path(sobol_csv_path_raw)
    redshift_bin_width > 0.0 || error("redshift_bin_width must be positive.")
    log_redshift_bin_width > 0.0 || error("log_redshift_bin_width must be positive.")
    mass_bin_width_dex > 0.0 || error("mass_bin_width_dex must be positive.")
    sobol_row >= 0 || error("sobol_row must be non-negative.")
    if sobol_row > 0
        isempty(sobol_csv_path) && error("sobol_row was set, but sobol_csv_path is empty.")
    elseif !isempty(sobol_csv_path)
        println("Sobol CSV path provided without sobol_row; using direct/manual Battaglia parameters.")
    end

    manual_battaglia_params = battaglia_params_from_args()
    if sobol_row > 0
        sobol_reference_params = battaglia_params_from_sobol_row(
            sobol_csv_path,
            sobol_row;
            base_params=default_battaglia_params()
        )
        battaglia_params = battaglia_params_from_sobol_row(
            sobol_csv_path,
            sobol_row;
            base_params=manual_battaglia_params
        )
        param_tag = build_sobol_param_tag(sobol_csv_path, sobol_row) * build_param_delta_tag(battaglia_params, sobol_reference_params)
    else
        battaglia_params = manual_battaglia_params
        param_tag = build_param_tag(battaglia_params)
    end
    binning_tag = build_binning_tag(
        batching_mode,
        redshift_binning_mode,
        redshift_bin_width,
        log_redshift_bin_width,
        mass_bin_width_dex
    )
    bin_map_mode_tag = build_bin_map_mode_tag(cumulative_bin_maps, save_bin_maps)
    run_tag = "$(add_str_end)_$(param_tag)_$(binning_tag)_$(bin_map_mode_tag)"

    output_dir = abspath(get_string_arg("output_dir", joinpath(homedir(), "HalfDome_outputs", "visuals"); env="TSZ_VISUAL_OUTPUT_DIR"))
    fits_output_path = joinpath(output_dir, "$(simulation_tag)_tSZ_nside$(nside)_$(run_tag)_m200c.fits")
    mass_fits_output_path = joinpath(output_dir, "$(simulation_tag)_mass_nside$(nside)_$(run_tag)_m200c.fits")
    cl_output_path = joinpath(output_dir, "$(simulation_tag)_tSZ_cl_m200c_$(param_tag)_nside$(nside)_$(binning_tag)_$(bin_map_mode_tag).fits")

    return VisualConfig(
        model_exists=model_exists,
        save_healpix_map=save_healpix_map,
        save_mass_map=save_mass_map,
        save_cl=save_cl,
        save_bin_maps=save_bin_maps,
        apply_mass_cut=apply_mass_cut,
        cumulative_bin_maps=cumulative_bin_maps,
        catalog_source=catalog_source,
        halfdome_path=halfdome_path,
        websky_path=websky_path,
        catalog_path=catalog_path,
        simulation_name=simulation_name,
        simulation_tag=simulation_tag,
        nside=nside,
        chunkN=chunkN,
        add_str_end=add_str_end,
        mass_min=mass_min,
        batching_mode=batching_mode,
        redshift_binning_mode=redshift_binning_mode,
        redshift_bin_width=redshift_bin_width,
        log_redshift_bin_width=log_redshift_bin_width,
        mass_bin_width_dex=mass_bin_width_dex,
        sobol_csv_path=sobol_csv_path,
        sobol_row=sobol_row,
        output_dir=output_dir,
        param_tag=param_tag,
        binning_tag=binning_tag,
        bin_map_mode_tag=bin_map_mode_tag,
        run_tag=run_tag,
        fits_output_path=fits_output_path,
        mass_fits_output_path=mass_fits_output_path,
        cl_output_path=cl_output_path,
        battaglia_params=battaglia_params
    )
end

function print_visual_config(cfg::VisualConfig)
    println("Using output directory: $(cfg.output_dir)")
    println("Using catalog source: $(cfg.catalog_source)")
    println("Using simulation tag: $(cfg.simulation_tag)")
    println("Using catalog path: $(cfg.catalog_path)")
    println("Batching mode: $(cfg.batching_mode)")
    println("Bin map save mode: $(cfg.bin_map_mode_tag)")
    println("Save per-bin maps: $(cfg.save_bin_maps)")
    println("Save mass maps: $(cfg.save_mass_map)")
    println("Redshift binning mode: $(cfg.redshift_binning_mode)")
    println("Linear redshift bin width dz: $(cfg.redshift_bin_width)")
    println("Log redshift bin width dlog10(1+z): $(cfg.log_redshift_bin_width)")
    println("Mass bin width dlog10(M): $(cfg.mass_bin_width_dex)")
    println("Mass map extent: HalfDome uses Rdisp spatial components; WebSky uses the catalog R column.")
    if cfg.sobol_row > 0
        println("Battaglia model source: Sobol CSV row $(cfg.sobol_row) from $(cfg.sobol_csv_path)")
    else
        println("Battaglia model source: direct/fiducial parameter inputs")
    end
    println("Battaglia16 physical parameters:")
    println("  P0_amp=$(cfg.battaglia_params.P0_amp), P0_alpha_m=$(cfg.battaglia_params.P0_alpha_m), P0_alpha_z=$(cfg.battaglia_params.P0_alpha_z)")
    println("  x_c_amp=$(cfg.battaglia_params.x_c_amp), x_c_alpha_m=$(cfg.battaglia_params.x_c_alpha_m), x_c_alpha_z=$(cfg.battaglia_params.x_c_alpha_z)")
    println("  beta_amp=$(cfg.battaglia_params.beta_amp), beta_alpha_m=$(cfg.battaglia_params.beta_alpha_m), beta_alpha_z=$(cfg.battaglia_params.beta_alpha_z)")
    println("  alpha_amp=$(cfg.battaglia_params.alpha_amp), alpha_alpha_m=$(cfg.battaglia_params.alpha_alpha_m), alpha_alpha_z=$(cfg.battaglia_params.alpha_alpha_z)")
    println("  gamma_amp=$(cfg.battaglia_params.gamma_amp), gamma_alpha_m=$(cfg.battaglia_params.gamma_alpha_m), gamma_alpha_z=$(cfg.battaglia_params.gamma_alpha_z)")
end
