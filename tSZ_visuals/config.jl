using DelimitedFiles
using SHA

const CLUSTER_HALFDOME_PATH_DEFAULT = "/lustre/work/Globus-lt/halfdome/full_res/halos"
const CLUSTER_OUTPUT_DIR_DEFAULT = "/lustre/work/kristero10/tSZ_data"
const CLUSTER_CACHE_DIR_DEFAULT = joinpath(CLUSTER_OUTPUT_DIR_DEFAULT, "cache")
const CLUSTER_SOBOL_CSV_DEFAULT = "/home/kristero10/tSZ_data/battaglia_sobol_32.csv"
const TSZ_GAUSSIAN_BEAM_FWHM_ARCMIN_DEFAULT = 1.6
const TSZ_CL_LMAX_DEFAULT = 4096

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

const BATTAGLIA_CACHE_TAG_DIGITS = 16
const BATTAGLIA_GUARD_LOGM_MIN_DEFAULT = 12.0
const BATTAGLIA_GUARD_Z_MAX_DEFAULT = 3.0
const BATTAGLIA_GUARD_DERIVED_X_C_MIN_DEFAULT = 0.08
const BATTAGLIA_GUARD_DERIVED_X_C_MAX_DEFAULT = 2.5
const BATTAGLIA_GUARD_BETA_OUTER_MIN_DEFAULT = 3.0
const BATTAGLIA_GUARD_BETA_OUTER_MAX_DEFAULT = 14.0

const BATTAGLIA_GUARD_PRIOR_BOUNDS = (
    P0_amp=(2.0, 25.0),
    x_c_amp=(0.12, 0.70),
    beta_amp=(3.8, 5.2),
    P0_alpha_m=(0.0, 0.30),
    x_c_alpha_m=(-0.08, 0.08),
    beta_alpha_m=(0.0, 0.08),
    P0_alpha_z=(-1.10, -0.40),
    x_c_alpha_z=(0.10, 0.90),
    beta_alpha_z=(0.25, 0.55),
    alpha_amp=(0.5, 2.0),
    alpha_alpha_m=(-0.25, 0.25),
    alpha_alpha_z=(-0.5, 0.5),
    gamma_amp=(-0.6, -0.05),
    gamma_alpha_m=(-0.25, 0.25),
    gamma_alpha_z=(-0.5, 0.5)
)

struct SkipVisualRun <: Exception
    message::String
end

Base.showerror(io::IO, err::SkipVisualRun) = print(io, err.message)

Base.@kwdef struct VisualConfig
    model_exists::Bool
    reuse_existing_cache::Bool
    cache_wait_seconds::Float64
    cache_poll_seconds::Float64
    save_healpix_map::Bool
    save_mass_map::Bool
    save_cl::Bool
    save_bin_maps::Bool
    skip_existing_outputs::Bool
    skip_existing_any_run_instance::Bool
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
    cosmo_h::Float64
    cosmo_omegab::Float64
    cosmo_omegac::Float64
    cosmo_omegam::Float64
    cosmo_h0::Float64
    cosmo_rho_m::Float64
    apply_gaussian_beam::Bool
    gaussian_beam_fwhm_arcmin::Float64
    cleanup_nonpositive_profile_values::Bool
    enforce_battaglia_guardrails::Bool
    skip_invalid_battaglia_rows::Bool
    interpolator_pad::Int
    interpolator_logM_max::Float64
    cl_lmax::Int
    cl_niter::Int
    batching_mode::String
    redshift_binning_mode::String
    redshift_bin_width::Float64
    log_redshift_bin_width::Float64
    mass_bin_width_dex::Float64
    sobol_csv_path::String
    sobol_row::Int
    output_dir::String
    cache_dir::String
    run_instance_tag::String
    param_tag::String
    cache_param_tag::String
    cosmology_tag::String
    beam_tag::String
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

function resolve_halfdome_catalog_path(path::AbstractString)
    resolved = resolve_repo_path(path)
    isdir(resolved) || return resolved

    entries = sort(readdir(resolved; join=true))
    hdf5_candidates = filter(entries) do entry
        isfile(entry) || return false
        ext = lowercase(splitext(entry)[2])
        return ext == ".h5" || ext == ".hdf5"
    end

    isempty(hdf5_candidates) && error(
        "halfdome_path=$(repr(resolved)) is a directory, but no .h5 or .hdf5 files were found inside it."
    )

    preferred_basenames = (
        "lightcone_100.hdf5",
        "lightcone_100.h5",
        "halos.hdf5",
        "halos.h5"
    )
    for preferred_name in preferred_basenames
        matches = filter(hdf5_candidates) do entry
            lowercase(basename(entry)) == preferred_name
        end
        length(matches) == 1 && return only(matches)
    end

    length(hdf5_candidates) == 1 && return only(hdf5_candidates)

    candidate_names = join(map(basename, hdf5_candidates), ", ")
    error(
        "halfdome_path=$(repr(resolved)) is a directory with multiple HDF5 files: $(candidate_names). " *
        "Pass halfdome_path as the exact HDF5 file path."
    )
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

function optional_filename_tag(s::AbstractString)
    raw = strip(String(s))
    isempty(raw) && return ""
    tag = lowercase(raw)
    tag = replace(tag, r"[^A-Za-z0-9_+\-.]+" => "_")
    tag = replace(tag, r"_+" => "_")
    return strip(tag, '_')
end

function default_slurm_run_instance_tag()
    job_id = strip(get(ENV, "SLURM_JOB_ID", ""))
    task_id = strip(get(ENV, "SLURM_ARRAY_TASK_ID", ""))

    if !isempty(job_id) && !isempty(task_id)
        return "slurm_job$(job_id)_task$(task_id)"
    elseif !isempty(job_id)
        return "slurm_job$(job_id)"
    elseif !isempty(task_id)
        return "slurm_task$(task_id)"
    end

    return ""
end

function default_sobol_row()
    task_id = strip(get(ENV, "SLURM_ARRAY_TASK_ID", ""))
    isempty(task_id) && return 0
    return parse(Int, task_id)
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

function csv_data_row_count(csv_path::AbstractString)
    isfile(csv_path) || return 0
    row_count = 0
    open(csv_path, "r") do io
        for _ in eachline(io)
            row_count += 1
        end
    end
    return max(row_count - 1, 0)
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

function battaglia_cache_tag(p)
    fields = (
        :P0_amp,
        :P0_alpha_m,
        :P0_alpha_z,
        :x_c_amp,
        :x_c_alpha_m,
        :x_c_alpha_z,
        :beta_amp,
        :beta_alpha_m,
        :beta_alpha_z,
        :alpha_amp,
        :alpha_alpha_m,
        :alpha_alpha_z,
        :gamma_amp,
        :gamma_alpha_m,
        :gamma_alpha_z
    )
    payload = join((string(field, "=", repr(getproperty(p, field))) for field in fields), ";")
    digest = bytes2hex(sha1(payload))[1:BATTAGLIA_CACHE_TAG_DIGITS]
    return "battaglia_phys_" * digest
end

function battaglia_powerlaw_param(amp::Real, alpha_m::Real, alpha_z::Real, m14::Real, z1::Real)
    return Float64(amp) * Float64(m14)^Float64(alpha_m) * Float64(z1)^Float64(alpha_z)
end

function battaglia_slice_params(p, mass_msun::Real, z::Real)
    m14 = Float64(mass_msun) / 1.0e14
    z1 = 1.0 + Float64(z)
    P0 = battaglia_powerlaw_param(p.P0_amp, p.P0_alpha_m, p.P0_alpha_z, m14, z1)
    x_c = battaglia_powerlaw_param(p.x_c_amp, p.x_c_alpha_m, p.x_c_alpha_z, m14, z1)
    alpha = battaglia_powerlaw_param(p.alpha_amp, p.alpha_alpha_m, p.alpha_alpha_z, m14, z1)
    beta_raw = battaglia_powerlaw_param(p.beta_amp, p.beta_alpha_m, p.beta_alpha_z, m14, z1)
    gamma = battaglia_powerlaw_param(p.gamma_amp, p.gamma_alpha_m, p.gamma_alpha_z, m14, z1)
    beta_outer = alpha * beta_raw - gamma
    return (P0=P0, x_c=x_c, alpha=alpha, beta_raw=beta_raw, gamma=gamma, beta_outer=beta_outer)
end

function validate_battaglia_params(
    p;
    enforce_prior_bounds::Bool=true,
    logM_min::Real=BATTAGLIA_GUARD_LOGM_MIN_DEFAULT,
    logM_max::Real=15.7,
    z_max::Real=BATTAGLIA_GUARD_Z_MAX_DEFAULT,
    derived_x_c_min::Real=BATTAGLIA_GUARD_DERIVED_X_C_MIN_DEFAULT,
    derived_x_c_max::Real=BATTAGLIA_GUARD_DERIVED_X_C_MAX_DEFAULT,
    beta_outer_min::Real=BATTAGLIA_GUARD_BETA_OUTER_MIN_DEFAULT,
    beta_outer_max::Real=BATTAGLIA_GUARD_BETA_OUTER_MAX_DEFAULT
)
    reasons = String[]

    for (field, value) in pairs(p)
        isfinite(value) || push!(reasons, "$(field) is not finite ($(value)).")
    end

    for field in (:P0_amp, :x_c_amp, :beta_amp, :alpha_amp)
        value = getproperty(p, field)
        value > 0.0 || push!(reasons, "$(field) must be positive, got $(value).")
    end

    if enforce_prior_bounds
        for (field, bounds) in pairs(BATTAGLIA_GUARD_PRIOR_BOUNDS)
            value = getproperty(p, field)
            low, high = bounds
            if !(low <= value <= high)
                push!(reasons, "$(field)=$(value) is outside the guarded prior range [$(low), $(high)].")
            end
        end
    end

    if isempty(reasons)
        logM_mid = 0.5 * (Float64(logM_min) + Float64(logM_max))
        z_mid = 0.5 * Float64(z_max)
        masses = 10.0 .^ unique(Float64[Float64(logM_min), logM_mid, Float64(logM_max)])
        redshifts = unique(Float64[0.0, z_mid, Float64(z_max)])

        for mass_msun in masses, z in redshifts
            s = battaglia_slice_params(p, mass_msun, z)
            if !(isfinite(s.P0) && s.P0 > 0.0)
                push!(reasons, "derived P0=$(s.P0) is invalid at M=$(mass_msun), z=$(z).")
            end
            if !(isfinite(s.x_c) && derived_x_c_min <= s.x_c <= derived_x_c_max)
                push!(
                    reasons,
                    "derived x_c=$(s.x_c) is outside [$(derived_x_c_min), $(derived_x_c_max)] at M=$(mass_msun), z=$(z)."
                )
            end
            if !(isfinite(s.alpha) && s.alpha > 0.0)
                push!(reasons, "derived alpha=$(s.alpha) is invalid at M=$(mass_msun), z=$(z).")
            end
            if !(isfinite(s.beta_raw) && s.beta_raw > 0.0)
                push!(reasons, "derived beta_raw=$(s.beta_raw) is invalid at M=$(mass_msun), z=$(z).")
            end
            if !(isfinite(s.gamma) && s.gamma < 0.0)
                push!(reasons, "derived gamma=$(s.gamma) must stay negative in this convention at M=$(mass_msun), z=$(z).")
            end
            if !(isfinite(s.beta_outer) && beta_outer_min <= s.beta_outer <= beta_outer_max)
                push!(
                    reasons,
                    "derived beta_outer=$(s.beta_outer) is outside [$(beta_outer_min), $(beta_outer_max)] at M=$(mass_msun), z=$(z)."
                )
            end
        end
    end

    return unique(reasons)
end

function battaglia_validation_message(reasons, sobol_csv_path::AbstractString, sobol_row::Integer)
    source = sobol_row > 0 ? "Sobol row $(sobol_row) from $(sobol_csv_path)" : "direct Battaglia parameters"
    return source * " failed Battaglia guardrails: " * join(reasons, " ")
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

function build_beam_tag(apply_gaussian_beam::Bool, gaussian_beam_fwhm_arcmin::Real)
    !apply_gaussian_beam && return "nobeam"
    return "gaussbeam_$(fmt_param_value(Float64(gaussian_beam_fwhm_arcmin)))arcmin"
end

function healpix_default_lmax(nside::Integer)
    nside > 0 || error("nside must be positive.")
    return 3 * Int(nside) - 1
end

function default_cl_lmax(nside::Integer)
    return min(TSZ_CL_LMAX_DEFAULT, healpix_default_lmax(nside))
end

function build_cl_lmax_tag(cl_lmax::Integer)
    return cl_lmax < 0 ? "lmaxdefault" : "lmax$(Int(cl_lmax))"
end

function build_cosmology_tag(cosmo_h::Real, cosmo_omegab::Real, cosmo_omegac::Real)
    if isapprox(Float64(cosmo_h), TSZ_H_VALUE; atol=0.0, rtol=1e-12) &&
       isapprox(Float64(cosmo_omegab), TSZ_OMEGAB; atol=0.0, rtol=1e-12) &&
       isapprox(Float64(cosmo_omegac), TSZ_OMEGAC; atol=0.0, rtol=1e-12)
        return "cosmo_fid"
    end

    return join(
        (
            "cosmo",
            "h$(fmt_param_value(Float64(cosmo_h)))",
            "ob$(fmt_param_value(Float64(cosmo_omegab)))",
            "oc$(fmt_param_value(Float64(cosmo_omegac)))"
        ),
        "_"
    )
end

function load_visual_config()
    model_exists = get_bool_arg("model_exists", true; env="MODEL_EXISTS")
    reuse_existing_cache = get_bool_arg("reuse_existing_cache", false; env="REUSE_EXISTING_CACHE")
    cache_wait_seconds = get_float_arg("cache_wait_seconds", 0.0; env="CACHE_WAIT_SECONDS")
    cache_poll_seconds = get_float_arg("cache_poll_seconds", 30.0; env="CACHE_POLL_SECONDS")
    save_healpix_map = get_bool_arg("save_healpix_map", true; env="SAVE_HEALPIX_MAP")
    save_mass_map = get_bool_arg("save_mass_map", true; env="SAVE_MASS_MAP")
    save_cl = get_bool_arg("save_cl", false; env="SAVE_CL")
    save_bin_maps = get_bool_arg("save_bin_maps", true; env="SAVE_BIN_MAPS")
    skip_existing_outputs = get_bool_arg("skip_existing_outputs", false; env="SKIP_EXISTING_OUTPUTS")
    skip_existing_any_run_instance = get_bool_arg("skip_existing_any_run_instance", true; env="SKIP_EXISTING_ANY_RUN_INSTANCE")
    apply_mass_cut = get_bool_arg("apply_mass_cut", true; env="APPLY_MASS_CUT")
    cumulative_bin_maps = get_bool_arg("cumulative_bin_maps", true; env="CUMULATIVE_BIN_MAPS")
    catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env="TSZ_CATALOG_SOURCE"))
    catalog_source in ("halfdome", "websky") || error("Unsupported catalog_source=$(repr(catalog_source)).")

    halfdome_path = resolve_halfdome_catalog_path(
        get_string_arg("halfdome_path", CLUSTER_HALFDOME_PATH_DEFAULT; env="HALFDOME_PATH")
    )
    websky_path = resolve_repo_path(get_string_arg("websky_path", "other_sims/sims/halos.pksc"; env="WEBSKY_PATH"))
    catalog_path = catalog_source == "halfdome" ? halfdome_path : websky_path
    simulation_name = get_string_arg("simulation_name", catalog_source; env="SIMULATION_NAME")
    simulation_tag = safe_filename_tag(simulation_name)

    nside = get_int_arg("nside", 1024; env="NSIDE")
    chunkN = get_int_arg("chunkN", 2_000_000; env="CHUNKN")
    add_str_end = get_string_arg("add_str_end", "13Msol_cutoff_HALO"; env="ADD_STR_END")
    mass_min = get_float_arg("mass_min", 1.0e12; env="MASS_MIN")
    cosmo_h = get_float_arg("cosmo_h", TSZ_H_VALUE; env="TSZ_COSMO_H")
    cosmo_omegab = get_float_arg("cosmo_omegab", TSZ_OMEGAB; env="TSZ_COSMO_OMEGAB")
    cosmo_omegac = get_float_arg("cosmo_omegac", TSZ_OMEGAC; env="TSZ_COSMO_OMEGAC")
    gaussian_beam_fwhm_arcmin = get_float_arg(
        "gaussian_beam_fwhm_arcmin",
        TSZ_GAUSSIAN_BEAM_FWHM_ARCMIN_DEFAULT;
        env="GAUSSIAN_BEAM_FWHM_ARCMIN"
    )
    apply_gaussian_beam = get_bool_arg(
        "apply_gaussian_beam",
        gaussian_beam_fwhm_arcmin > 0.0;
        env="APPLY_GAUSSIAN_BEAM"
    )
    cleanup_nonpositive_profile_values = get_bool_arg(
        "cleanup_nonpositive_profile_values",
        true;
        env="CLEANUP_NONPOSITIVE_PROFILE_VALUES"
    )
    enforce_battaglia_guardrails = get_bool_arg(
        "enforce_battaglia_guardrails",
        true;
        env="ENFORCE_BATTAGLIA_GUARDRAILS"
    )
    skip_invalid_battaglia_rows = get_bool_arg(
        "skip_invalid_battaglia_rows",
        true;
        env="SKIP_INVALID_BATTAGLIA_ROWS"
    )
    interpolator_pad = get_int_arg("interpolator_pad", 256; env="INTERPOLATOR_PAD")
    interpolator_logM_max = get_float_arg("interpolator_logM_max", 15.7; env="INTERPOLATOR_LOGM_MAX")
    cl_lmax = get_int_arg("cl_lmax", default_cl_lmax(nside); env="TSZ_CL_LMAX")
    cl_niter = get_int_arg("cl_niter", 0; env="TSZ_CL_NITER")
    batching_mode = normalize_batching_mode(get_string_arg("batching_mode", "redshift"; env="BATCHING_MODE"))
    redshift_binning_mode = normalize_redshift_binning_mode(get_string_arg("redshift_binning_mode", "linear"; env="REDSHIFT_BINNING_MODE"))
    redshift_bin_width = get_float_arg("redshift_bin_width", 1.0; env="REDSHIFT_BIN_WIDTH")
    log_redshift_bin_width = get_float_arg("log_redshift_bin_width", 0.2; env="LOG_REDSHIFT_BIN_WIDTH")
    mass_bin_width_dex = get_float_arg("mass_bin_width_dex", 0.5; env="MASS_BIN_WIDTH_DEX")
    sobol_csv_path_raw = strip(get_string_arg("sobol_csv_path", CLUSTER_SOBOL_CSV_DEFAULT; env="BATTAGLIA_SOBOL_CSV"))
    sobol_row = get_int_arg("sobol_row", default_sobol_row(); env="BATTAGLIA_SOBOL_ROW")
    sobol_csv_path = isempty(sobol_csv_path_raw) ? "" : resolve_repo_path(sobol_csv_path_raw)
    run_instance_tag_raw = get_string_arg("run_instance_tag", default_slurm_run_instance_tag(); env="TSZ_RUN_INSTANCE_TAG")
    run_instance_tag = optional_filename_tag(run_instance_tag_raw)
    cosmo_h > 0.0 || error("cosmo_h must be positive.")
    cosmo_omegab > 0.0 || error("cosmo_omegab must be positive.")
    cosmo_omegac > 0.0 || error("cosmo_omegac must be positive.")
    cosmo_omegam = cosmo_omegab + cosmo_omegac
    cosmo_omegam > 0.0 || error("cosmo_omegam must be positive.")
    cosmo_omegam < 1.0 || error("cosmo_omegam must be smaller than 1.")
    cosmo_h0 = 100.0 * cosmo_h
    cosmo_rho_m = 2.775e11 * cosmo_omegam * cosmo_h^2
    gaussian_beam_fwhm_arcmin >= 0.0 || error("gaussian_beam_fwhm_arcmin must be nonnegative. Set it to 0 to disable the beam.")
    if apply_gaussian_beam
        gaussian_beam_fwhm_arcmin > 0.0 || error("apply_gaussian_beam=true requires gaussian_beam_fwhm_arcmin > 0.")
    end
    cache_wait_seconds >= 0.0 || error("cache_wait_seconds must be nonnegative.")
    cache_poll_seconds > 0.0 || error("cache_poll_seconds must be positive.")
    interpolator_pad >= 0 || error("interpolator_pad must be nonnegative.")
    interpolator_logM_max > 0.0 || error("interpolator_logM_max must be positive.")
    cl_lmax == -1 || cl_lmax > 0 || error("cl_lmax must be positive, or -1 to use the Healpix default.")
    if cl_lmax > healpix_default_lmax(nside)
        error("cl_lmax=$(cl_lmax) exceeds the Healpix default maximum $(healpix_default_lmax(nside)) for nside=$(nside).")
    end
    cl_niter >= 0 || error("cl_niter must be nonnegative.")
    redshift_bin_width > 0.0 || error("redshift_bin_width must be positive.")
    log_redshift_bin_width > 0.0 || error("log_redshift_bin_width must be positive.")
    mass_bin_width_dex > 0.0 || error("mass_bin_width_dex must be positive.")
    sobol_row >= 0 || error("sobol_row must be non-negative.")
    if sobol_row > 0
        isempty(sobol_csv_path) && error("sobol_row was set, but sobol_csv_path is empty.")
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
    validation_reasons = validate_battaglia_params(
        battaglia_params;
        enforce_prior_bounds=enforce_battaglia_guardrails,
        logM_max=interpolator_logM_max
    )
    if !isempty(validation_reasons)
        validation_message = battaglia_validation_message(validation_reasons, sobol_csv_path, sobol_row)
        if sobol_row > 0 && skip_invalid_battaglia_rows
            throw(SkipVisualRun(validation_message * " Skipping this row without writing outputs."))
        end
        error(validation_message)
    end
    cache_param_tag = battaglia_cache_tag(battaglia_params)
    cosmology_tag = build_cosmology_tag(cosmo_h, cosmo_omegab, cosmo_omegac)
    beam_tag = build_beam_tag(apply_gaussian_beam, gaussian_beam_fwhm_arcmin)
    binning_tag = build_binning_tag(
        batching_mode,
        redshift_binning_mode,
        redshift_bin_width,
        log_redshift_bin_width,
        mass_bin_width_dex
    )
    bin_map_mode_tag = build_bin_map_mode_tag(cumulative_bin_maps, save_bin_maps)
    run_tag = "$(add_str_end)_$(param_tag)_$(cosmology_tag)_$(beam_tag)_$(binning_tag)_$(bin_map_mode_tag)"

    output_dir = abspath(get_string_arg("output_dir", CLUSTER_OUTPUT_DIR_DEFAULT; env="TSZ_VISUAL_OUTPUT_DIR"))
    cache_dir = abspath(get_string_arg("cache_dir", CLUSTER_CACHE_DIR_DEFAULT; env="TSZ_VISUAL_CACHE_DIR"))
    output_run_tag = isempty(run_instance_tag) ? run_tag : "$(run_tag)__$(run_instance_tag)"
    cl_lmax_tag = build_cl_lmax_tag(cl_lmax)
    cl_tag_base = "$(beam_tag)_$(binning_tag)_$(bin_map_mode_tag)_$(cl_lmax_tag)"
    cl_run_tag = isempty(run_instance_tag) ? cl_tag_base : "$(cl_tag_base)__$(run_instance_tag)"
    fits_output_path = joinpath(output_dir, "$(simulation_tag)_tSZ_nside$(nside)_$(output_run_tag)_m200c.fits")
    mass_fits_output_path = joinpath(output_dir, "$(simulation_tag)_mass_nside$(nside)_$(output_run_tag)_m200c.fits")
    cl_output_path = joinpath(output_dir, "$(simulation_tag)_tSZ_cl_m200c_$(param_tag)_$(cosmology_tag)_nside$(nside)_$(cl_run_tag).fits")

    return VisualConfig(
        model_exists=model_exists,
        reuse_existing_cache=reuse_existing_cache,
        cache_wait_seconds=cache_wait_seconds,
        cache_poll_seconds=cache_poll_seconds,
        save_healpix_map=save_healpix_map,
        save_mass_map=save_mass_map,
        save_cl=save_cl,
        save_bin_maps=save_bin_maps,
        skip_existing_outputs=skip_existing_outputs,
        skip_existing_any_run_instance=skip_existing_any_run_instance,
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
        cosmo_h=cosmo_h,
        cosmo_omegab=cosmo_omegab,
        cosmo_omegac=cosmo_omegac,
        cosmo_omegam=cosmo_omegam,
        cosmo_h0=cosmo_h0,
        cosmo_rho_m=cosmo_rho_m,
        apply_gaussian_beam=apply_gaussian_beam,
        gaussian_beam_fwhm_arcmin=gaussian_beam_fwhm_arcmin,
        cleanup_nonpositive_profile_values=cleanup_nonpositive_profile_values,
        enforce_battaglia_guardrails=enforce_battaglia_guardrails,
        skip_invalid_battaglia_rows=skip_invalid_battaglia_rows,
        interpolator_pad=interpolator_pad,
        interpolator_logM_max=interpolator_logM_max,
        cl_lmax=cl_lmax,
        cl_niter=cl_niter,
        batching_mode=batching_mode,
        redshift_binning_mode=redshift_binning_mode,
        redshift_bin_width=redshift_bin_width,
        log_redshift_bin_width=log_redshift_bin_width,
        mass_bin_width_dex=mass_bin_width_dex,
        sobol_csv_path=sobol_csv_path,
        sobol_row=sobol_row,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_instance_tag=run_instance_tag,
        param_tag=param_tag,
        cache_param_tag=cache_param_tag,
        cosmology_tag=cosmology_tag,
        beam_tag=beam_tag,
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
    run_instance_tag_display = isempty(cfg.run_instance_tag) ? "<none>" : cfg.run_instance_tag
    println("Using output directory: $(cfg.output_dir)")
    println("Using catalog source: $(cfg.catalog_source)")
    println("Using simulation tag: $(cfg.simulation_tag)")
    println("Using catalog path: $(cfg.catalog_path)")
    println("Using cache directory: $(cfg.cache_dir)")
    println("Run instance tag: $(run_instance_tag_display)")
    println("Cosmology tag: $(cfg.cosmology_tag)")
    println("Cosmology: h=$(cfg.cosmo_h), Omega_b=$(cfg.cosmo_omegab), Omega_c=$(cfg.cosmo_omegac), Omega_m=$(cfg.cosmo_omegam)")
    println("Gaussian beam: apply=$(cfg.apply_gaussian_beam), fwhm_arcmin=$(cfg.gaussian_beam_fwhm_arcmin)")
    println("Interpolator nonpositive cleanup: $(cfg.cleanup_nonpositive_profile_values)")
    println("Battaglia guardrails: enforce=$(cfg.enforce_battaglia_guardrails), skip_invalid_sobol_rows=$(cfg.skip_invalid_battaglia_rows)")
    println("Interpolator cache mode: model_exists=$(cfg.model_exists), reuse_existing_cache=$(cfg.reuse_existing_cache), wait_seconds=$(cfg.cache_wait_seconds), poll_seconds=$(cfg.cache_poll_seconds), pad=$(cfg.interpolator_pad), logM_max=$(cfg.interpolator_logM_max)")
    println("Interpolator cache parameter tag: $(cfg.cache_param_tag)")
    cl_lmax_display = cfg.cl_lmax < 0 ? "Healpix default ($(healpix_default_lmax(cfg.nside)))" : string(cfg.cl_lmax)
    println("C_l transform: save=$(cfg.save_cl), lmax=$(cl_lmax_display), niter=$(cfg.cl_niter)")
    println("Skip existing outputs: $(cfg.skip_existing_outputs), any_run_instance=$(cfg.skip_existing_any_run_instance)")
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
