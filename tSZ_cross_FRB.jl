if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

using XGPaint, Healpix, Interpolations, HDF5, Plots
using LinearAlgebra
using Base.Threads
using Random
using Statistics
using SHA

include("utils.jl")  # xyz_to_ra_dec
include("SOConvertNFW.jl")

using .M200Convert

const h_value = 0.68
const c_kms = 299_792.458
const ARCMIN2RAD = pi / (180.0 * 60.0)

# -------------------------
# options
# -------------------------
model_exists = true          # set to false to (re)build the model interpolator
save_healpix_maps = true    # save Healpix map FITS files
save_cl = true               # compute and save cross-spectrum
apply_mass_cut = true        # apply mass cut for the tSZ map
subtract_map_means = false    # remove monopole before anafast

t0 = time()

nside =4096
chunkN = 1_000_000

add_str_end = "13Msol_cutoff_HALO"
mass_min = 1.0e+13

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
    return default
end

function parse_bool_arg(value)
    value_norm = lowercase(strip(String(value)))
    if value_norm in ("1", "true", "t", "yes", "y", "on")
        return true
    elseif value_norm in ("0", "false", "f", "no", "n", "off")
        return false
    end
    error("Could not parse boolean value $(repr(value)). Use true/false, yes/no, on/off, or 1/0.")
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

function has_arg_value(key; env=nothing)
    if env !== nothing && haskey(ENV, env)
        return true
    end
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for a in ARGS
        if startswith(a, prefix1) || startswith(a, prefix2)
            return true
        end
    end
    return false
end

function save_plot_accessible(plot_obj, preferred_path::AbstractString)
    preferred_dir = dirname(preferred_path)
    isdir(preferred_dir) || mkpath(preferred_dir)

    preferred_error = nothing
    try
        savefig(plot_obj, preferred_path)
    catch err
        preferred_error = err
    end

    if isfile(preferred_path) && filesize(preferred_path) > 0
        return preferred_path
    end

    fallback_dir = joinpath(homedir(), "HalfDome_plot_outputs")
    isdir(fallback_dir) || mkpath(fallback_dir)
    fallback_path = joinpath(fallback_dir, basename(preferred_path))

    try
        savefig(plot_obj, fallback_path)
    catch err
        error(
            "Could not save plot to either $(preferred_path) or $(fallback_path)." *
            (preferred_error === nothing ? "" : " First save error: $(preferred_error).") *
            " Fallback save error: $(err)."
        )
    end

    if isfile(fallback_path) && filesize(fallback_path) > 0
        return fallback_path
    end

    error(
        "Plot save completed without producing a readable file at $(preferred_path) or $(fallback_path)." *
        (preferred_error === nothing ? "" : " First save error: $(preferred_error).")
    )
end

catalog_source = lowercase(get_string_arg("catalog_source", "halfdome"; env="FRB_DM_CATALOG_SOURCE"))
catalog_source in ("halfdome", "websky") || error("Unsupported catalog_source=$(repr(catalog_source)). Use \"halfdome\" or \"websky\".")

catalog_tag = catalog_source
halfdome_path = get_string_arg("halfdome_path", "lightcone_100.hdf5"; env="FRB_DM_HALFDOME_PATH")
websky_path = get_string_arg("websky_path", "other_sims/sims/halos.pksc"; env="FRB_DM_WEBSKY_PATH")

frb_count = get_int_arg("frb_count", 10_000; env="FRB_COUNT")
frb_seed = get_int_arg("frb_seed", 12345; env="FRB_SEED")
frb_selection_mode = lowercase(get_string_arg("frb_selection_mode", "redshift"; env="FRB_SELECTION_MODE"))
frb_z_cut = get_float_arg("frb_z_cut", 0.5; env="FRB_Z_CUT")
frb_z_max = get_float_arg("frb_z_max", 2.5; env="FRB_Z_MAX")
frb_redshift_weight_bin_width = get_float_arg("frb_redshift_weight_bin_width", 0.05; env="FRB_REDSHIFT_WEIGHT_BIN_WIDTH")
dm_residual_bin_width = get_float_arg("dm_residual_bin_width", 0.05; env="DM_RESIDUAL_BIN_WIDTH")
use_dm_residual_for_cross = get_bool_arg("use_dm_residual_for_cross", true; env="USE_DM_RESIDUAL_FOR_CROSS")
normalize_weighted_frb_map_by_density = get_bool_arg(
    "normalize_weighted_frb_map_by_density",
    false;
    env="NORMALIZE_WEIGHTED_FRB_MAP_BY_DENSITY"
)
use_normalized_dm_residual_estimator = get_bool_arg(
    "use_normalized_dm_residual_estimator",
    false;
    env="USE_NORMALIZED_DM_RESIDUAL_ESTIMATOR"
)
save_frb_catalog = get_bool_arg("save_frb_catalog", true; env="SAVE_FRB_CATALOG")
save_dm_residual_diagnostic = get_bool_arg("save_dm_residual_diagnostic", true; env="SAVE_DM_RESIDUAL_DIAGNOSTIC")
truncate_beam_support = get_bool_arg("truncate_beam_support", false; env="TRUNCATE_BEAM_SUPPORT")

if use_normalized_dm_residual_estimator
    use_dm_residual_for_cross = true
    normalize_weighted_frb_map_by_density = true
end

legacy_apply_gaussian_beam_set = has_arg_value("apply_gaussian_beam"; env="APPLY_GAUSSIAN_BEAM")
legacy_gaussian_beam_fwhm_set = has_arg_value("gaussian_beam_fwhm_arcmin"; env="GAUSSIAN_BEAM_FWHM_ARCMIN")
legacy_beam_support_radius_set = has_arg_value("beam_support_radius_arcmin"; env="BEAM_SUPPORT_RADIUS_ARCMIN")
legacy_beam_config_set = legacy_apply_gaussian_beam_set || legacy_gaussian_beam_fwhm_set || legacy_beam_support_radius_set

legacy_apply_gaussian_beam = get_bool_arg("apply_gaussian_beam", true; env="APPLY_GAUSSIAN_BEAM")
legacy_gaussian_beam_fwhm_arcmin = get_float_arg("gaussian_beam_fwhm_arcmin", 0.5; env="GAUSSIAN_BEAM_FWHM_ARCMIN")
legacy_beam_support_radius_arcmin = get_float_arg(
    "beam_support_radius_arcmin",
    3.0 * legacy_gaussian_beam_fwhm_arcmin;
    env="BEAM_SUPPORT_RADIUS_ARCMIN"
)

default_tsz_gaussian_beam_fwhm_arcmin = legacy_beam_config_set ?
    (legacy_apply_gaussian_beam ? legacy_gaussian_beam_fwhm_arcmin : 0.0) :
    1.6
default_dm_gaussian_beam_fwhm_arcmin = legacy_beam_config_set ?
    (legacy_apply_gaussian_beam ? legacy_gaussian_beam_fwhm_arcmin : 0.0) :
    0.0

tsz_gaussian_beam_fwhm_arcmin = get_float_arg(
    "tsz_gaussian_beam_fwhm_arcmin",
    default_tsz_gaussian_beam_fwhm_arcmin;
    env="TSZ_GAUSSIAN_BEAM_FWHM_ARCMIN"
)
dm_gaussian_beam_fwhm_arcmin = get_float_arg(
    "dm_gaussian_beam_fwhm_arcmin",
    default_dm_gaussian_beam_fwhm_arcmin;
    env="DM_GAUSSIAN_BEAM_FWHM_ARCMIN"
)

apply_tsz_gaussian_beam = get_bool_arg(
    "apply_tsz_gaussian_beam",
    tsz_gaussian_beam_fwhm_arcmin > 0.0;
    env="APPLY_TSZ_GAUSSIAN_BEAM"
)
apply_dm_gaussian_beam = get_bool_arg(
    "apply_dm_gaussian_beam",
    dm_gaussian_beam_fwhm_arcmin > 0.0;
    env="APPLY_DM_GAUSSIAN_BEAM"
)

tsz_beam_support_radius_arcmin = get_float_arg(
    "tsz_beam_support_radius_arcmin",
    legacy_beam_support_radius_set ? legacy_beam_support_radius_arcmin : 3.0 * tsz_gaussian_beam_fwhm_arcmin;
    env="TSZ_BEAM_SUPPORT_RADIUS_ARCMIN"
)
dm_beam_support_radius_arcmin = get_float_arg(
    "dm_beam_support_radius_arcmin",
    legacy_beam_support_radius_set ? legacy_beam_support_radius_arcmin : 3.0 * dm_gaussian_beam_fwhm_arcmin;
    env="DM_BEAM_SUPPORT_RADIUS_ARCMIN"
)

frb_count > 0 || error("frb_count must be positive.")
frb_selection_mode in ("random", "redshift") || error("Unsupported frb_selection_mode=$(repr(frb_selection_mode)). Use \"random\" or \"redshift\".")
frb_z_cut > 0.0 || error("frb_z_cut must be positive.")
frb_z_max > 0.0 || error("frb_z_max must be positive.")
frb_z_max >= frb_z_cut || error("frb_z_max must be greater than or equal to frb_z_cut.")
frb_redshift_weight_bin_width > 0.0 || error("frb_redshift_weight_bin_width must be positive.")
dm_residual_bin_width > 0.0 || error("dm_residual_bin_width must be positive.")
tsz_gaussian_beam_fwhm_arcmin >= 0.0 || error("tsz_gaussian_beam_fwhm_arcmin must be nonnegative. Set it to 0 to disable the tSZ beam.")
dm_gaussian_beam_fwhm_arcmin >= 0.0 || error("dm_gaussian_beam_fwhm_arcmin must be nonnegative. Set it to 0 to disable the DM beam.")
tsz_beam_support_radius_arcmin >= 0.0 || error("tsz_beam_support_radius_arcmin must be nonnegative.")
dm_beam_support_radius_arcmin >= 0.0 || error("dm_beam_support_radius_arcmin must be nonnegative.")
if apply_tsz_gaussian_beam
    tsz_gaussian_beam_fwhm_arcmin > 0.0 || error("apply_tsz_gaussian_beam=true requires tsz_gaussian_beam_fwhm_arcmin > 0.")
end
if apply_dm_gaussian_beam
    dm_gaussian_beam_fwhm_arcmin > 0.0 || error("apply_dm_gaussian_beam=true requires dm_gaussian_beam_fwhm_arcmin > 0.")
end

# -------------------------
# Battaglia16 model parameters (editable)
# -------------------------
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

battaglia_P0_amp = get_float_arg("battaglia_P0_amp", BATTAGLIA_P0_AMP_DEFAULT; env="BATTAGLIA_P0_AMP")
battaglia_P0_alpha_m = get_float_arg("battaglia_P0_alpha_m", BATTAGLIA_P0_ALPHA_M_DEFAULT; env="BATTAGLIA_P0_ALPHA_M")
battaglia_P0_alpha_z = get_float_arg("battaglia_P0_alpha_z", BATTAGLIA_P0_ALPHA_Z_DEFAULT; env="BATTAGLIA_P0_ALPHA_Z")

battaglia_x_c_amp = get_float_arg("battaglia_x_c_amp", BATTAGLIA_X_C_AMP_DEFAULT; env="BATTAGLIA_X_C_AMP")
battaglia_x_c_alpha_m = get_float_arg("battaglia_x_c_alpha_m", BATTAGLIA_X_C_ALPHA_M_DEFAULT; env="BATTAGLIA_X_C_ALPHA_M")
battaglia_x_c_alpha_z = get_float_arg("battaglia_x_c_alpha_z", BATTAGLIA_X_C_ALPHA_Z_DEFAULT; env="BATTAGLIA_X_C_ALPHA_Z")

battaglia_beta_amp = get_float_arg("battaglia_beta_amp", BATTAGLIA_BETA_AMP_DEFAULT; env="BATTAGLIA_BETA_AMP")
battaglia_beta_alpha_m = get_float_arg("battaglia_beta_alpha_m", BATTAGLIA_BETA_ALPHA_M_DEFAULT; env="BATTAGLIA_BETA_ALPHA_M")
battaglia_beta_alpha_z = get_float_arg("battaglia_beta_alpha_z", BATTAGLIA_BETA_ALPHA_Z_DEFAULT; env="BATTAGLIA_BETA_ALPHA_Z")

battaglia_alpha_amp = get_float_arg("battaglia_alpha_amp", BATTAGLIA_ALPHA_AMP_DEFAULT; env="BATTAGLIA_ALPHA_AMP")
battaglia_alpha_alpha_m = get_float_arg("battaglia_alpha_alpha_m", BATTAGLIA_ALPHA_ALPHA_M_DEFAULT; env="BATTAGLIA_ALPHA_ALPHA_M")
battaglia_alpha_alpha_z = get_float_arg("battaglia_alpha_alpha_z", BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT; env="BATTAGLIA_ALPHA_ALPHA_Z")

battaglia_gamma_amp = get_float_arg("battaglia_gamma_amp", BATTAGLIA_GAMMA_AMP_DEFAULT; env="BATTAGLIA_GAMMA_AMP")
battaglia_gamma_alpha_m = get_float_arg("battaglia_gamma_alpha_m", BATTAGLIA_GAMMA_ALPHA_M_DEFAULT; env="BATTAGLIA_GAMMA_ALPHA_M")
battaglia_gamma_alpha_z = get_float_arg("battaglia_gamma_alpha_z", BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT; env="BATTAGLIA_GAMMA_ALPHA_Z")

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

function shorten_filename_stem(stem::AbstractString; max_length::Int=220)
    stem_length = ncodeunits(stem)
    if stem_length <= max_length
        return String(stem)
    end

    digest = bytes2hex(sha1(stem))[1:12]
    head_length = max_length - 14
    head_length > 0 || error("max_length=$(max_length) is too small to shorten filename stem safely.")
    return String(stem[1:head_length]) * "_h" * digest
end

function make_output_filename(stem::AbstractString, extension::AbstractString; max_stem_length::Int=220)
    return shorten_filename_stem(stem; max_length=max_stem_length) * extension
end

function resolve_catalog_input_path(catalog_source::AbstractString, halfdome_path::AbstractString, websky_path::AbstractString)
    raw_path = catalog_source == "halfdome" ? halfdome_path : websky_path
    return abspath(raw_path)
end

function build_catalog_input_tag(catalog_source::AbstractString, catalog_input_path::AbstractString)
    if !isfile(catalog_input_path)
        signature = "$(catalog_source)|missing|$(catalog_input_path)"
        return "$(catalog_source)src_h" * bytes2hex(sha1(signature))[1:12]
    end

    canonical_path = realpath(catalog_input_path)
    path_stat = stat(canonical_path)
    signature = string(
        catalog_source,
        "|",
        canonical_path,
        "|",
        path_stat.size,
        "|",
        round(Int, path_stat.mtime)
    )
    return "$(catalog_source)src_h" * bytes2hex(sha1(signature))[1:12]
end

function build_param_tag()
    parts = String[]
    if battaglia_P0_amp != BATTAGLIA_P0_AMP_DEFAULT
        push!(parts, "battaglia_P0_amp_" * fmt_param_value(battaglia_P0_amp))
    end
    if battaglia_P0_alpha_m != BATTAGLIA_P0_ALPHA_M_DEFAULT
        push!(parts, "battaglia_P0_alpha_m_" * fmt_param_value(battaglia_P0_alpha_m))
    end
    if battaglia_P0_alpha_z != BATTAGLIA_P0_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_P0_alpha_z_" * fmt_param_value(battaglia_P0_alpha_z))
    end
    if battaglia_x_c_amp != BATTAGLIA_X_C_AMP_DEFAULT
        push!(parts, "battaglia_x_c_amp_" * fmt_param_value(battaglia_x_c_amp))
    end
    if battaglia_x_c_alpha_m != BATTAGLIA_X_C_ALPHA_M_DEFAULT
        push!(parts, "battaglia_x_c_alpha_m_" * fmt_param_value(battaglia_x_c_alpha_m))
    end
    if battaglia_x_c_alpha_z != BATTAGLIA_X_C_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_x_c_alpha_z_" * fmt_param_value(battaglia_x_c_alpha_z))
    end
    if battaglia_beta_amp != BATTAGLIA_BETA_AMP_DEFAULT
        push!(parts, "battaglia_beta_amp_" * fmt_param_value(battaglia_beta_amp))
    end
    if battaglia_beta_alpha_m != BATTAGLIA_BETA_ALPHA_M_DEFAULT
        push!(parts, "battaglia_beta_alpha_m_" * fmt_param_value(battaglia_beta_alpha_m))
    end
    if battaglia_beta_alpha_z != BATTAGLIA_BETA_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_beta_alpha_z_" * fmt_param_value(battaglia_beta_alpha_z))
    end
    if battaglia_alpha_amp != BATTAGLIA_ALPHA_AMP_DEFAULT
        push!(parts, "battaglia_alpha_amp_" * fmt_param_value(battaglia_alpha_amp))
    end
    if battaglia_alpha_alpha_m != BATTAGLIA_ALPHA_ALPHA_M_DEFAULT
        push!(parts, "battaglia_alpha_alpha_m_" * fmt_param_value(battaglia_alpha_alpha_m))
    end
    if battaglia_alpha_alpha_z != BATTAGLIA_ALPHA_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_alpha_alpha_z_" * fmt_param_value(battaglia_alpha_alpha_z))
    end
    if battaglia_gamma_amp != BATTAGLIA_GAMMA_AMP_DEFAULT
        push!(parts, "battaglia_gamma_amp_" * fmt_param_value(battaglia_gamma_amp))
    end
    if battaglia_gamma_alpha_m != BATTAGLIA_GAMMA_ALPHA_M_DEFAULT
        push!(parts, "battaglia_gamma_alpha_m_" * fmt_param_value(battaglia_gamma_alpha_m))
    end
    if battaglia_gamma_alpha_z != BATTAGLIA_GAMMA_ALPHA_Z_DEFAULT
        push!(parts, "battaglia_gamma_alpha_z_" * fmt_param_value(battaglia_gamma_alpha_z))
    end
    if isempty(parts)
        return "base"
    end
    return "base_plus_" * join(parts, "__")
end

function build_beam_tag(prefix::AbstractString, apply_beam::Bool, fwhm_arcmin::Real; truncate_support::Bool=false, support_radius_arcmin::Real=0.0)
    if !apply_beam
        return "$(prefix)beam_none"
    end

    return "$(prefix)beam_$(fmt_param_value(fwhm_arcmin))arcmin"
end

function build_path_config_tag(;
    add_str_end::AbstractString,
    apply_mass_cut::Bool,
    mass_min::Real,
    truncate_beam_support::Bool,
    tsz_beam_support_radius_arcmin::Real,
    dm_beam_support_radius_arcmin::Real
)
    signature = join(
        (
            "add_str_end=$(add_str_end)",
            "apply_mass_cut=$(apply_mass_cut)",
            "mass_min=$(Float64(mass_min))",
            "truncate_beam_support=$(truncate_beam_support)",
            "tsz_beam_support_radius_arcmin=$(Float64(tsz_beam_support_radius_arcmin))",
            "dm_beam_support_radius_arcmin=$(Float64(dm_beam_support_radius_arcmin))"
        ),
        "|"
    )
    return "cfg_h" * bytes2hex(sha1(signature))[1:10]
end

function build_tsz_path_config_tag(;
    add_str_end::AbstractString,
    apply_mass_cut::Bool,
    mass_min::Real,
    truncate_beam_support::Bool,
    tsz_beam_support_radius_arcmin::Real
)
    signature = join(
        (
            "add_str_end=$(add_str_end)",
            "apply_mass_cut=$(apply_mass_cut)",
            "mass_min=$(Float64(mass_min))",
            "truncate_beam_support=$(truncate_beam_support)",
            "tsz_beam_support_radius_arcmin=$(Float64(tsz_beam_support_radius_arcmin))"
        ),
        "|"
    )
    return "tszcfg_h" * bytes2hex(sha1(signature))[1:10]
end

param_tag = build_param_tag()
catalog_input_path = resolve_catalog_input_path(catalog_source, halfdome_path, websky_path)
catalog_input_tag = build_catalog_input_tag(catalog_source, catalog_input_path)
frb_redshift_tag = "zcut$(fmt_param_value(frb_z_cut))_zmax$(fmt_param_value(frb_z_max))"
path_config_tag = build_path_config_tag(
    add_str_end=add_str_end,
    apply_mass_cut=apply_mass_cut,
    mass_min=mass_min,
    truncate_beam_support=truncate_beam_support,
    tsz_beam_support_radius_arcmin=tsz_beam_support_radius_arcmin,
    dm_beam_support_radius_arcmin=dm_beam_support_radius_arcmin
)
tsz_path_config_tag = build_tsz_path_config_tag(
    add_str_end=add_str_end,
    apply_mass_cut=apply_mass_cut,
    mass_min=mass_min,
    truncate_beam_support=truncate_beam_support,
    tsz_beam_support_radius_arcmin=tsz_beam_support_radius_arcmin
)
mean_tag = subtract_map_means ? "monopole_removed" : "monopole_kept"
run_config_tag = "$(catalog_input_tag)_$(path_config_tag)"
tsz_beam_tag = build_beam_tag(
    "tsz",
    apply_tsz_gaussian_beam,
    tsz_gaussian_beam_fwhm_arcmin;
    truncate_support=truncate_beam_support && apply_tsz_gaussian_beam,
    support_radius_arcmin=tsz_beam_support_radius_arcmin
)
dm_beam_tag = build_beam_tag(
    "dm",
    apply_dm_gaussian_beam,
    dm_gaussian_beam_fwhm_arcmin;
    truncate_support=truncate_beam_support && apply_dm_gaussian_beam,
    support_radius_arcmin=dm_beam_support_radius_arcmin
)
frb_weight_tag =
    frb_selection_mode == "redshift" ?
    "redshiftcorr_dz$(fmt_param_value(frb_redshift_weight_bin_width))" :
    "random_uniformhalo"
frb_tag = "nfrb$(frb_count)_seed$(frb_seed)_$(frb_weight_tag)_$(frb_redshift_tag)_$(run_config_tag)"
tsz_map_tag = "$(catalog_input_tag)_$(param_tag)_$(tsz_beam_tag)_$(tsz_path_config_tag)_$(mean_tag)"
frb_map_tag = "$(frb_tag)_$(mean_tag)"
dm_map_tag = "$(frb_tag)_$(param_tag)_$(dm_beam_tag)_$(mean_tag)"
dm_residual_tag = "dmresid_dz$(fmt_param_value(dm_residual_bin_width))"
dm_norm_tag = normalize_weighted_frb_map_by_density ? "normnbar" : "unnorm"
dm_cross_tag = "$(use_dm_residual_for_cross ? "cross_dmresidual" : "cross_dmraw")_$(dm_norm_tag)"
cl_tag_base = "$(frb_tag)_$(param_tag)_$(tsz_beam_tag)_$(dm_beam_tag)_$(dm_residual_tag)_$(dm_cross_tag)"

output_dir = joinpath(@__DIR__, "batched_data")
plot_output_dir = joinpath(output_dir, "plots")
fits_output_path_tsz = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_tSZ_nside$(nside)_$(tsz_map_tag)_m200c", ".fits")
)
fits_output_path_frb = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_FRB_nside$(nside)_$(frb_map_tag)", ".fits")
)
frb_hosts_cache_path = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_FRB_hosts_nside$(nside)_$(frb_tag)", ".h5")
)
fits_output_path_dm = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_FRB_DM_nside$(nside)_$(dm_map_tag)", ".fits")
)
fits_output_path_dm_residual = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_FRB_DM_residual_nside$(nside)_$(dm_map_tag)_$(dm_residual_tag)_$(dm_norm_tag)", ".fits")
)
frb_catalog_output_path = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_FRB_catalog_nside$(nside)_$(frb_tag)_$(param_tag)_$(dm_residual_tag)", ".h5")
)
cl_output_path = joinpath(
    output_dir,
    make_output_filename("$(catalog_tag)_tSZ_x_FRB_selected1_DM_cl_$(cl_tag_base)_nside$(nside)", ".fits")
)
linear_hist_output_path = joinpath(plot_output_dir, "redshift_distribution_FRB_$(frb_selection_mode)_linear.png")
log_hist_output_path = joinpath(plot_output_dir, "redshift_distribution_FRB_$(frb_selection_mode)_logx.png")
selected_halo_hist_output_path = joinpath(
    plot_output_dir,
    make_output_filename("selected_FRB_host_halo_histograms_$(catalog_tag)_$(frb_tag)", ".png")
)
dm_residual_diagnostic_output_path = joinpath(
    plot_output_dir,
    make_output_filename("FRB_DM_residual_diagnostic_$(catalog_tag)_$(frb_tag)_$(dm_residual_tag)", ".png")
)

isdir(output_dir) || mkpath(output_dir)
isdir(plot_output_dir) || mkpath(plot_output_dir)

println("FRB configuration:")
println("  catalog_input_path=$(catalog_input_path)")
println("  catalog_input_tag=$(catalog_input_tag)")
println("  path_config_tag=$(path_config_tag)  # hidden mass-cut / beam-support config")
println("  tsz_path_config_tag=$(tsz_path_config_tag)  # hidden tSZ-only cache config")
println("  add_str_end=$(add_str_end), apply_mass_cut=$(apply_mass_cut), mass_min=$(mass_min)")
println("  frb_count=$(frb_count), frb_seed=$(frb_seed)")
println("  frb_selection_mode=$(frb_selection_mode)")
if frb_selection_mode == "redshift"
    println("  FRB redshift PDF: paper-inspired with z_cut=$(frb_z_cut), truncated at z<=$(frb_z_max)")
    println("  FRB host sampling: corrected halo weighting with dz=$(frb_redshift_weight_bin_width)")
end
println("  tSZ Gaussian beam: apply=$(apply_tsz_gaussian_beam), fwhm_arcmin=$(tsz_gaussian_beam_fwhm_arcmin)")
println("  DM Gaussian beam: apply=$(apply_dm_gaussian_beam), fwhm_arcmin=$(dm_gaussian_beam_fwhm_arcmin)")
println("  use_dm_residual_for_cross=$(use_dm_residual_for_cross)")
println("  normalize_weighted_frb_map_by_density=$(normalize_weighted_frb_map_by_density)")
println("  use_normalized_dm_residual_estimator=$(use_normalized_dm_residual_estimator)")
println("  dm_residual_bin_width=$(dm_residual_bin_width)")
println("  dm_cross_tag=$(dm_cross_tag)")
println("  truncate_beam_support=$(truncate_beam_support)")
println("  tSZ beam support radius arcmin=$(tsz_beam_support_radius_arcmin)")
println("  DM beam support radius arcmin=$(dm_beam_support_radius_arcmin)")
if legacy_beam_config_set
    println("  note: legacy shared beam arguments were detected; map-specific beam settings inherit from them unless explicitly overridden.")
end
if truncate_beam_support && !apply_tsz_gaussian_beam && !apply_dm_gaussian_beam
    println("  note: truncate_beam_support is enabled, but it only has an effect for maps with Gaussian beam smoothing enabled.")
end
println("Battaglia16 physical parameters:")
println("  P0_amp=$(battaglia_P0_amp), P0_alpha_m=$(battaglia_P0_alpha_m), P0_alpha_z=$(battaglia_P0_alpha_z)")
println("  x_c_amp=$(battaglia_x_c_amp), x_c_alpha_m=$(battaglia_x_c_alpha_m), x_c_alpha_z=$(battaglia_x_c_alpha_z)")
println("  beta_amp=$(battaglia_beta_amp), beta_alpha_m=$(battaglia_beta_alpha_m), beta_alpha_z=$(battaglia_beta_alpha_z)")
println("  alpha_amp=$(battaglia_alpha_amp), alpha_alpha_m=$(battaglia_alpha_alpha_m), alpha_alpha_z=$(battaglia_alpha_alpha_z)")
println("  gamma_amp=$(battaglia_gamma_amp), gamma_alpha_m=$(battaglia_gamma_alpha_m), gamma_alpha_z=$(battaglia_gamma_alpha_z)")

# -------------------------
# cosmology: chi(z) and z(chi)
# -------------------------
omegab = 0.049
omegac = 0.31 - omegab
omegam = omegab + omegac
omegal = 1 - omegam
h = 0.68
H0 = 100 * h

function make_chi_and_z_of_chi_itp(; omegam, h_value, z1=0.0, z2=6.0, nz=100_000)
    H0 = 100 * h_value
    H(z) = H0 * sqrt(omegam * (1 + z)^3 + 1 - omegam)
    dchidz(z) = c_kms / H(z)

    za = collect(range(z1, z2; length=nz))
    dz = za[2] - za[1]
    chia = similar(za)

    chia[1] = 0.0
    s = 0.0
    @inbounds for i in 2:length(za)
        s += 0.5 * (dchidz(za[i - 1]) + dchidz(za[i])) * dz
        chia[i] = s
    end

    chi_of_z_itp = linear_interpolation(za, chia; extrapolation_bc=Line())
    z_of_chi_itp = linear_interpolation(chia, za; extrapolation_bc=Line())

    return chi_of_z_itp, z_of_chi_itp
end

chi_of_z_itp, itp_z_of_chi = make_chi_and_z_of_chi_itp(omegam=omegam, h_value=h_value)

@inline H_of_z(z::Float64) = H0 * sqrt(omegam * (1.0 + z)^3 + 1.0 - omegam)
@inline luminosity_distance(z::Float64, chi_of_z_itp) = (1.0 + z) * chi_of_z_itp(z)

function frb_redshift_pdf_weight(z::Float64, chi_of_z_itp, d_l_cut_sq::Float64)
    chi = chi_of_z_itp(z)
    d_l = luminosity_distance(z, chi_of_z_itp)
    return chi^2 / ((1.0 + z) * H_of_z(z)) * exp(-(d_l * d_l) / d_l_cut_sq)
end

function normalize_pdf_grid!(z_grid, pdf_values)
    integral = 0.0
    @inbounds for i in 1:length(z_grid)-1
        integral += 0.5 * (pdf_values[i] + pdf_values[i + 1]) * (z_grid[i + 1] - z_grid[i])
    end
    integral > 0.0 || error("FRB redshift PDF normalization must be positive.")
    pdf_values ./= integral
    return pdf_values
end

function evaluate_normalized_frb_redshift_pdf(z_grid, chi_of_z_itp, d_l_cut_sq, frb_z_max)
    pdf_values = Vector{Float64}(undef, length(z_grid))

    @inbounds for i in eachindex(z_grid)
        z = Float64(z_grid[i])
        pdf_values[i] = (0.0 <= z <= frb_z_max) ? frb_redshift_pdf_weight(z, chi_of_z_itp, d_l_cut_sq) : 0.0
    end

    return normalize_pdf_grid!(z_grid, pdf_values)
end

function expected_histogram_counts(z_grid, pdf_values, bin_edges, sample_count::Int)
    length(bin_edges) >= 2 || error("Histogram bin_edges must contain at least two entries.")
    all(diff(bin_edges) .> 0) || error("Histogram bin_edges must be strictly increasing.")

    expected_counts = Vector{Float64}(undef, length(z_grid))

    @inbounds for i in eachindex(z_grid)
        z = z_grid[i]
        if z < first(bin_edges) || z > last(bin_edges)
            expected_counts[i] = 0.0
            continue
        end

        bin_idx = searchsortedlast(bin_edges, z)
        if bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end
        bin_width = bin_edges[bin_idx + 1] - bin_edges[bin_idx]
        expected_counts[i] = sample_count * pdf_values[i] * bin_width
    end

    return expected_counts
end

function histogram_counts(values, bin_edges)
    counts = zeros(Int, length(bin_edges) - 1)

    @inbounds for value in values
        if value < first(bin_edges) || value > last(bin_edges)
            continue
        end

        bin_idx = searchsortedlast(bin_edges, value)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end
        counts[bin_idx] += 1
    end

    return counts
end

function pdf_from_histogram_counts(counts, bin_edges)
    total_count = sum(counts)
    total_count > 0 || error("Histogram counts must sum to a positive value.")

    pdf = Vector{Float64}(undef, length(counts))
    @inbounds for i in eachindex(counts)
        bin_width = bin_edges[i + 1] - bin_edges[i]
        pdf[i] = counts[i] / (total_count * bin_width)
    end

    return pdf
end

function evaluate_histogram_pdf(z_grid, bin_edges, bin_pdf)
    pdf_values = zeros(Float64, length(z_grid))

    @inbounds for i in eachindex(z_grid)
        z = z_grid[i]
        if z < first(bin_edges) || z > last(bin_edges)
            continue
        end

        bin_idx = searchsortedlast(bin_edges, z)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end
        pdf_values[i] = bin_pdf[bin_idx]
    end

    return pdf_values
end

function lookup_histogram_pdf_value(z::Real, bin_edges, bin_pdf)
    if z < first(bin_edges) || z > last(bin_edges)
        return 0.0
    end

    bin_idx = searchsortedlast(bin_edges, z)
    if bin_idx == 0
        return 0.0
    elseif bin_idx >= length(bin_edges)
        bin_idx = length(bin_edges) - 1
    end

    return Float64(bin_pdf[bin_idx])
end

function compute_corrected_frb_selection_weights(
    eligible_redshifts,
    chi_of_z_itp,
    d_l_cut_sq,
    frb_z_max;
    dz=0.05
)
    isempty(eligible_redshifts) && error("Need at least one eligible FRB redshift to compute corrected selection weights.")
    dz > 0.0 || error("Corrected FRB selection weight bin width must be positive.")

    z_min = 0.0
    z_max = max(Float64(frb_z_max), maximum(eligible_redshifts))
    nbins = max(1, ceil(Int, (z_max - z_min) / dz))
    bin_edges = collect(range(z_min, z_min + nbins * dz; length=nbins + 1))

    halo_counts = histogram_counts(eligible_redshifts, bin_edges)
    halo_pdf = pdf_from_histogram_counts(halo_counts, bin_edges)
    corrected_weights = Vector{Float64}(undef, length(eligible_redshifts))

    @inbounds for i in eachindex(eligible_redshifts)
        z = Float64(eligible_redshifts[i])
        target_weight = z <= frb_z_max ? frb_redshift_pdf_weight(z, chi_of_z_itp, d_l_cut_sq) : 0.0
        halo_density = lookup_histogram_pdf_value(z, bin_edges, halo_pdf)
        corrected_weights[i] = (halo_density > 0.0 && isfinite(target_weight) && target_weight > 0.0) ? target_weight / halo_density : 0.0
    end

    count(>(0.0), corrected_weights) >= 1 || error("Corrected FRB selection weights produced no positive entries.")
    return corrected_weights, bin_edges, halo_pdf
end

function expanded_linear_limits(xmin::Float64, xmax::Float64; min_width::Float64=1e-3)
    xmin <= xmax || error("expanded_linear_limits requires xmin <= xmax, got $(xmin) > $(xmax).")
    if xmax > xmin
        return (xmin, xmax)
    end

    width = max(abs(xmax) * 1e-3, min_width)
    return (xmin, xmax + width)
end

function expanded_positive_limits(xmin::Float64, xmax::Float64; min_ratio::Float64=1.2, min_width::Float64=1e-6)
    xmin > 0.0 || error("expanded_positive_limits requires xmin > 0, got $(xmin).")
    xmax >= xmin || error("expanded_positive_limits requires xmax >= xmin, got $(xmax) < $(xmin).")
    if xmax > xmin
        return (xmin, xmax)
    end

    expanded_xmax = max(xmin * min_ratio, xmin + min_width)
    return (xmin, expanded_xmax)
end

function positive_finite_xy(x_values, y_values)
    keep = isfinite.(x_values) .& isfinite.(y_values) .& (y_values .> 0.0)
    return x_values[keep], y_values[keep]
end

function m200m_to_m200c(m200m, z)
    omegamz = omegam .* (1 .+ z) .^ 3 ./ (omegam .* (1 .+ z) .^ 3 .+ 1 .- omegam)
    return m200m .* omegamz .^ 0.35
end

@inline function m200m_to_m200c_scalar(m200m::Float64, z::Float64)
    one_plus_z = 1.0 + z
    ez_num = omegam * one_plus_z^3
    omegamz = ez_num / (ez_num + 1.0 - omegam)
    return m200m * omegamz^0.35
end

function compute_redshift_and_mass(x, y, z, R, itp_z_of_chi, rho_m)
    n = length(x)
    redshift = Vector{Float64}(undef, n)
    halo_mass = Vector{Float64}(undef, n)
    mass_prefactor = (4.0 * pi / 3.0) * rho_m

    @threads for i in 1:n
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        ri = Float64(R[i])

        chi = sqrt(xi * xi + yi * yi + zi * zi)
        zi_redshift = itp_z_of_chi(chi)

        redshift[i] = zi_redshift
        halo_mass[i] = m200m_to_m200c_scalar(mass_prefactor * ri^3, zi_redshift)
    end

    return redshift, halo_mass
end

function xyz_to_ra_dec_threaded(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where {T}
    @assert length(x) == length(y) == length(z)

    n = length(x)
    ra = Vector{T}(undef, n)
    dec = Vector{T}(undef, n)

    @threads for i in 1:n
        r = sqrt(x[i]^2 + y[i]^2 + z[i]^2)
        vx = x[i] / r
        vy = y[i] / r
        vz = z[i] / r

        theta, phi = Healpix.vec2ang(vx, vy, vz)
        dec[i] = T(pi) / 2 - theta
        ra[i] = phi
    end

    return ra, dec
end

function draw_weighted_sample_positions(rng, weights::AbstractVector{<:Real}, sample_count::Int)
    sample_count <= length(weights) || error("sample_count=$(sample_count) exceeds candidate count=$(length(weights)).")

    if sample_count == length(weights)
        return collect(eachindex(weights))
    end

    selection_keys = Vector{Float64}(undef, length(weights))
    @inbounds for i in eachindex(weights)
        weight = Float64(weights[i])
        weight > 0.0 || error("All FRB selection weights must be positive.")
        selection_keys[i] = randexp(rng) / weight
    end

    return partialsortperm(selection_keys, 1:sample_count)
end

function build_frb_overdensity_map(sample_x, sample_y, sample_z, nside)
    frb_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(frb_map.pixels, 0.0)
    res = Healpix.Resolution(nside)

    @inbounds for i in eachindex(sample_x)
        r = sqrt(sample_x[i]^2 + sample_y[i]^2 + sample_z[i]^2)
        vx = sample_x[i] / r
        vy = sample_y[i] / r
        vz = sample_z[i] / r
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        pix = Healpix.ang2pixRing(res, theta, phi)
        frb_map.pixels[pix] += 1.0
    end

    mean_counts = length(sample_x) / length(frb_map.pixels)
    frb_map.pixels ./= mean_counts
    frb_map.pixels .-= 1.0

    return frb_map
end

function build_frb_weighted_map(sample_x, sample_y, sample_z, weights, nside; normalize_by_mean_density::Bool=false)
    length(sample_x) == length(sample_y) == length(sample_z) == length(weights) || error(
        "FRB weighted map inputs must have the same length."
    )

    weighted_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(weighted_map.pixels, 0.0)
    res = Healpix.Resolution(nside)

    @inbounds for i in eachindex(sample_x)
        r = sqrt(sample_x[i]^2 + sample_y[i]^2 + sample_z[i]^2)
        vx = sample_x[i] / r
        vy = sample_y[i] / r
        vz = sample_z[i] / r
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        pix = Healpix.ang2pixRing(res, theta, phi)
        weighted_map.pixels[pix] += weights[i]
    end

    if normalize_by_mean_density
        mean_counts = length(sample_x) / length(weighted_map.pixels)
        mean_counts > 0.0 || error("Cannot normalize the FRB weighted map with zero selected FRBs.")
        weighted_map.pixels ./= mean_counts
    end

    return weighted_map
end

function sample_healpix_map_at_points(m::HealpixMap{<:Real, RingOrder}, ras, decs)
    length(ras) == length(decs) || error("ras and decs must have the same length.")
    sampled_values = Vector{Float64}(undef, length(ras))
    res = m.resolution

    @inbounds for i in eachindex(ras)
        theta = pi / 2 - decs[i]
        phi = ras[i]
        pix = Healpix.ang2pixRing(res, theta, phi)
        sampled_values[i] = Float64(m.pixels[pix])
    end

    return sampled_values
end

function compute_binned_mean_dm(redshifts, dm_values; z_min=0.0, z_max=maximum(redshifts), dz=0.05)
    length(redshifts) == length(dm_values) || error("redshifts and dm_values must have the same length.")
    dz > 0.0 || error("dz must be positive.")
    z_max > z_min || error("z_max must be greater than z_min.")

    nbins = max(1, ceil(Int, (z_max - z_min) / dz))
    bin_edges = collect(range(z_min, z_min + nbins * dz; length=nbins + 1))
    bin_sums = zeros(Float64, nbins)
    bin_counts = zeros(Int, nbins)

    @inbounds for i in eachindex(redshifts)
        z = Float64(redshifts[i])
        dm = Float64(dm_values[i])
        if !isfinite(z) || !isfinite(dm) || z < z_min || z > z_max
            continue
        end

        bin_idx = searchsortedlast(bin_edges, z)
        if bin_idx == 0
            continue
        elseif bin_idx >= length(bin_edges)
            bin_idx = length(bin_edges) - 1
        end

        bin_sums[bin_idx] += dm
        bin_counts[bin_idx] += 1
    end

    bin_centers = 0.5 .* (bin_edges[1:end-1] .+ bin_edges[2:end])
    bin_means = fill(NaN, nbins)
    @inbounds for i in eachindex(bin_means)
        if bin_counts[i] > 0
            bin_means[i] = bin_sums[i] / bin_counts[i]
        end
    end

    keep = isfinite.(bin_means)
    sum(keep) >= 2 || error("Need at least two populated redshift bins to build the DM mean interpolation.")

    kept_bin_centers = bin_centers[keep]
    kept_bin_means = bin_means[keep]
    kept_bin_counts = bin_counts[keep]
    mean_dm_itp = linear_interpolation(kept_bin_centers, kept_bin_means; extrapolation_bc=Line())

    mu_dm_at_z(z::Float64) =
        z <= kept_bin_centers[1] ? kept_bin_means[1] :
        z >= kept_bin_centers[end] ? kept_bin_means[end] :
        mean_dm_itp(z)

    return kept_bin_centers, kept_bin_means, kept_bin_counts, mu_dm_at_z
end

function save_frb_dm_catalog(
    path::AbstractString;
    sample_x,
    sample_y,
    sample_z,
    sample_ra,
    sample_dec,
    sample_mass,
    sample_redshift,
    sample_dm_raw,
    sample_dm_mean,
    sample_dm_residual,
    dm_mean_bin_centers,
    dm_mean_bin_values,
    dm_mean_bin_counts,
    dm_residual_bin_width
)
    sample_count = length(sample_x)
    length(sample_y) == sample_count || error("sample_y length does not match sample_x length.")
    length(sample_z) == sample_count || error("sample_z length does not match sample_x length.")
    length(sample_ra) == sample_count || error("sample_ra length does not match sample_x length.")
    length(sample_dec) == sample_count || error("sample_dec length does not match sample_x length.")
    length(sample_mass) == sample_count || error("sample_mass length does not match sample_x length.")
    length(sample_redshift) == sample_count || error("sample_redshift length does not match sample_x length.")
    length(sample_dm_raw) == sample_count || error("sample_dm_raw length does not match sample_x length.")
    length(sample_dm_mean) == sample_count || error("sample_dm_mean length does not match sample_x length.")
    length(sample_dm_residual) == sample_count || error("sample_dm_residual length does not match sample_x length.")

    catalog_dir = dirname(path)
    isdir(catalog_dir) || mkpath(catalog_dir)

    h5open(path, "w") do h5
        write(h5, "sample_x", Float64.(sample_x))
        write(h5, "sample_y", Float64.(sample_y))
        write(h5, "sample_z", Float64.(sample_z))
        write(h5, "sample_ra", Float64.(sample_ra))
        write(h5, "sample_dec", Float64.(sample_dec))
        write(h5, "sample_mass", Float64.(sample_mass))
        write(h5, "sample_redshift", Float64.(sample_redshift))
        write(h5, "sample_dm_raw", Float64.(sample_dm_raw))
        write(h5, "sample_dm_mean", Float64.(sample_dm_mean))
        write(h5, "sample_dm_residual", Float64.(sample_dm_residual))
        write(h5, "dm_mean_bin_centers", Float64.(dm_mean_bin_centers))
        write(h5, "dm_mean_bin_values", Float64.(dm_mean_bin_values))
        write(h5, "dm_mean_bin_counts", Int.(dm_mean_bin_counts))
        write(h5, "dm_residual_bin_width", [Float64(dm_residual_bin_width)])
    end

    return path
end

function save_dm_residual_diagnostic_plot(
    sample_redshift,
    sample_dm_raw,
    sample_dm_residual,
    dm_mean_bin_centers,
    dm_mean_bin_values,
    output_path::AbstractString
)
    p_raw = scatter(
        sample_redshift,
        sample_dm_raw;
        markersize=2,
        alpha=0.35,
        color=:grey,
        label="",
        xlabel="FRB redshift",
        ylabel="raw DM"
    )
    plot!(p_raw, dm_mean_bin_centers, dm_mean_bin_values; color=:red, linewidth=2, label="")

    p_residual = scatter(
        sample_redshift,
        sample_dm_residual;
        markersize=2,
        alpha=0.35,
        color=:grey,
        label="",
        xlabel="FRB redshift",
        ylabel="DM residual"
    )
    hline!(p_residual, [0.0]; color=:red, linewidth=2, label="")

    diagnostic_plot = plot(
        p_raw,
        p_residual;
        layout=(2, 1),
        size=(900, 900),
        plot_title="FRB DM residual diagnostics"
    )

    return save_plot_accessible(diagnostic_plot, output_path)
end

function save_selected_frb_host_histogram_plot(
    sample_redshift,
    sample_mass,
    output_path::AbstractString
)
    sample_count = length(sample_redshift)
    length(sample_mass) == sample_count || error("sample_mass length does not match sample_redshift length.")

    finite_sample_redshift = sample_redshift[isfinite.(sample_redshift)]
    positive_sample_mass = sample_mass[isfinite.(sample_mass) .& (sample_mass .> 0.0)]

    isempty(finite_sample_redshift) && error("Need at least one finite FRB host redshift to make the selected-halo histogram.")
    isempty(positive_sample_mass) && error("Need at least one positive FRB host mass to make the selected-halo histogram.")

    redshift_limits = expanded_linear_limits(0.0, maximum(finite_sample_redshift))
    mass_limits = expanded_positive_limits(minimum(positive_sample_mass), maximum(positive_sample_mass))

    redshift_bins = collect(range(redshift_limits[1], redshift_limits[2]; length=26))
    mass_bins = 10 .^ range(log10(mass_limits[1]), log10(mass_limits[2]); length=26)

    p_redshift = histogram(
        finite_sample_redshift;
        bins=redshift_bins,
        xlims=redshift_limits,
        color=:grey,
        linecolor=:grey,
        alpha=0.9,
        label="",
        xlabel="Chosen FRB host redshift",
        ylabel="counts",
        title="Chosen FRB host redshift distribution"
    )

    p_mass = histogram(
        positive_sample_mass;
        bins=mass_bins,
        xlims=mass_limits,
        xscale=:log10,
        color=:grey,
        linecolor=:grey,
        alpha=0.9,
        label="",
        xlabel="Chosen FRB host halo mass [Msun]",
        ylabel="counts",
        title="Chosen FRB host halo mass distribution"
    )

    combined_plot = plot(
        p_redshift,
        p_mass;
        layout=(2, 1),
        size=(900, 900),
        plot_title="Chosen FRB host halo histograms (N=$(sample_count))"
    )

    return save_plot_accessible(combined_plot, output_path)
end

function capture_selected_halos!(
    sample_x,
    sample_y,
    sample_z,
    sample_mass,
    sample_redshift,
    sampled_halo_indices,
    next_sampled_index_pos::Int,
    captured_host_count::Int,
    batch_start_halo_index::Int,
    x,
    y,
    z,
    halo_mass,
    redshift
)
    batch_end_halo_index = batch_start_halo_index + length(x) - 1

    while next_sampled_index_pos <= length(sampled_halo_indices) &&
          sampled_halo_indices[next_sampled_index_pos] <= batch_end_halo_index
        selected_halo_index = sampled_halo_indices[next_sampled_index_pos]
        local_halo_index = selected_halo_index - batch_start_halo_index + 1
        captured_host_count += 1

        sample_x[captured_host_count] = Float64(x[local_halo_index])
        sample_y[captured_host_count] = Float64(y[local_halo_index])
        sample_z[captured_host_count] = Float64(z[local_halo_index])
        sample_mass[captured_host_count] = Float64(halo_mass[local_halo_index])
        sample_redshift[captured_host_count] = Float64(redshift[local_halo_index])

        next_sampled_index_pos += 1
    end

    return next_sampled_index_pos, captured_host_count
end

function append_frb_candidate_halos!(
    eligible_frb_halo_indices,
    eligible_frb_halo_weights,
    eligible_frb_redshifts,
    batch_start_halo_index::Int,
    halo_mass,
    redshift;
    selection,
    mass_min,
    frb_selection_mode,
    frb_z_max,
    chi_of_z_itp,
    d_l_cut_sq
)
    @inbounds for local_halo_index in eachindex(redshift)
        if selection && halo_mass[local_halo_index] < mass_min
            continue
        end

        halo_redshift = Float64(redshift[local_halo_index])
        if halo_redshift < 0.0
            continue
        end

        selection_weight =
            if frb_selection_mode == "random"
                1.0
            else
                halo_redshift > frb_z_max ? 0.0 : frb_redshift_pdf_weight(halo_redshift, chi_of_z_itp, d_l_cut_sq)
            end

        if !isfinite(selection_weight) || selection_weight <= 0.0
            continue
        end

        push!(eligible_frb_halo_indices, batch_start_halo_index + local_halo_index - 1)
        push!(eligible_frb_halo_weights, selection_weight)
        if eligible_frb_redshifts !== nothing
            push!(eligible_frb_redshifts, halo_redshift)
        end
    end

    return nothing
end

function paint_tsz_batch!(
    m_hp,
    tmp_hp,
    workspace,
    y_model_interp,
    x,
    y,
    z,
    halo_mass,
    redshift;
    selection,
    mass_min,
    support_mask=nothing,
    support_extra_radius_rad=0.0
)
    if selection
        sel = halo_mass .>= mass_min
        if !any(sel)
            return nothing
        end

        xs = Float64.(x[sel])
        ys = Float64.(y[sel])
        zs = Float64.(z[sel])
        ms = halo_mass[sel]
        zsft = redshift[sel]
    else
        xs = Float64.(x)
        ys = Float64.(y)
        zs = Float64.(z)
        ms = halo_mass
        zsft = redshift
    end

    ra, dec = xyz_to_ra_dec_threaded(xs, ys, zs)
    perm = sortperm(dec)
    ra = ra[perm]
    dec = dec[perm]
    zsft = zsft[perm]
    ms = ms[perm]

    if !isnothing(support_mask)
        mark_healpix_profile_support!(
            support_mask,
            workspace,
            y_model_interp,
            ms,
            zsft,
            ra,
            dec,
            support_extra_radius_rad
        )
    end

    fill!(tmp_hp.pixels, 0.0)
    paint!(tmp_hp, workspace, y_model_interp, ms, zsft, ra, dec)
    m_hp.pixels .+= tmp_hp.pixels

    return nothing
end

function stream_catalog_batches(
    process_batch!::F,
    catalog_source::AbstractString,
    halfdome_path::AbstractString,
    websky_path::AbstractString,
    chunkN::Int,
    itp_z_of_chi,
    rho_m
) where {F}
    if catalog_source == "halfdome"
        return h5open(halfdome_path, "r") do h5
            pos_ds = h5["Position"]
            mass_ds = h5["halo_mass_m200c"]
            redshift_ds = h5["redshift"]

            total_halo_count = size(pos_ds, 2)
            @show total_halo_count

            for batch_start_halo_index in 1:chunkN:total_halo_count
                batch_end_halo_index = min(batch_start_halo_index + chunkN - 1, total_halo_count)
                idx = batch_start_halo_index:batch_end_halo_index

                pos = pos_ds[:, idx]
                x = @view pos[1, :]
                y = @view pos[2, :]
                z = @view pos[3, :]
                halo_mass = Float64.(mass_ds[idx]) ./ h_value
                redshift = Float64.(redshift_ds[idx])

                process_batch!(batch_start_halo_index, x, y, z, halo_mass, redshift)
            end

            total_halo_count
        end
    end

    return open(websky_path, "r") do io
        total_halo_count = Int(read(io, Int32))
        RTHmax = read(io, Float32)
        redshiftbox = read(io, Float32)
        @show total_halo_count RTHmax redshiftbox

        buf = Matrix{Float32}(undef, 10, chunkN)
        batch_start_halo_index = 1
        nleft = total_halo_count

        while nleft > 0
            nthis = min(chunkN, nleft)

            rawview = @view reinterpret(Float32, vec(buf))[1:10 * nthis]
            read!(io, rawview)

            cat = @view buf[:, 1:nthis]
            x = @view cat[1, :]
            y = @view cat[2, :]
            z = @view cat[3, :]
            R = @view cat[7, :]

            redshift, halo_mass = compute_redshift_and_mass(x, y, z, R, itp_z_of_chi, rho_m)

            process_batch!(batch_start_halo_index, x, y, z, halo_mass, redshift)

            batch_start_halo_index += nthis
            nleft -= nthis
        end

        total_halo_count
    end
end

function remove_monopole!(m)
    m.pixels .-= mean(m.pixels)
    return m
end

function copy_healpix_map(m::HealpixMap{<:Real, RingOrder})
    copied_map = HealpixMap{Float64, RingOrder}(m.resolution.nside)
    copied_map.pixels .= Float64.(m.pixels)
    return copied_map
end

function write_cl_fits_overwrite(path::AbstractString, cl_values)
    if isfile(path)
        rm(path; force=true)
    end
    writeClToFITS(path, collect(cl_values); overwrite=true)
    return path
end

function validate_healpix_map(m::HealpixMap{<:Real, RingOrder}, map_name::AbstractString)
    if !all(isfinite, m.pixels)
        error("$(map_name) contains non-finite pixel values.")
    end
    return nothing
end

function load_healpix_map_from_fits(path::AbstractString, nside::Int, map_name::AbstractString)
    println("Loading cached $(map_name) from $(path)")
    m = Healpix.readMapFromFITS(path, 1, Float64)
    expected_npix = 12 * nside^2
    length(m.pixels) == expected_npix || error(
        "Cached $(map_name) at $(path) has $(length(m.pixels)) pixels, expected $(expected_npix) for nside=$(nside)."
    )
    validate_healpix_map(m, "$(map_name) cache")
    return m
end

function save_frb_host_cache(
    path::AbstractString,
    sample_x,
    sample_y,
    sample_z,
    sample_mass,
    sample_redshift
)
    sample_count = length(sample_x)
    length(sample_y) == sample_count || error("sample_y length does not match sample_x length.")
    length(sample_z) == sample_count || error("sample_z length does not match sample_x length.")
    length(sample_mass) == sample_count || error("sample_mass length does not match sample_x length.")
    length(sample_redshift) == sample_count || error("sample_redshift length does not match sample_x length.")

    cache_dir = dirname(path)
    isdir(cache_dir) || mkpath(cache_dir)

    h5open(path, "w") do h5
        write(h5, "sample_x", Float64.(sample_x))
        write(h5, "sample_y", Float64.(sample_y))
        write(h5, "sample_z", Float64.(sample_z))
        write(h5, "sample_mass", Float64.(sample_mass))
        write(h5, "sample_redshift", Float64.(sample_redshift))
    end

    return path
end

function load_frb_host_cache(path::AbstractString, expected_sample_count::Int)
    println("Loading cached FRB host sample from $(path)")
    sample_x, sample_y, sample_z, sample_mass, sample_redshift = h5open(path, "r") do h5
        return (
            read(h5["sample_x"]),
            read(h5["sample_y"]),
            read(h5["sample_z"]),
            read(h5["sample_mass"]),
            read(h5["sample_redshift"])
        )
    end

    length(sample_x) == expected_sample_count || error(
        "Cached FRB host sample at $(path) has $(length(sample_x)) entries, expected $(expected_sample_count)."
    )
    length(sample_y) == expected_sample_count || error("Cached FRB host sample has inconsistent sample_y length.")
    length(sample_z) == expected_sample_count || error("Cached FRB host sample has inconsistent sample_z length.")
    length(sample_mass) == expected_sample_count || error("Cached FRB host sample has inconsistent sample_mass length.")
    length(sample_redshift) == expected_sample_count || error("Cached FRB host sample has inconsistent sample_redshift length.")

    return sample_x, sample_y, sample_z, sample_mass, sample_redshift
end

function smooth_healpix_map_gaussian!(m::HealpixMap{<:Real, RingOrder}, fwhm_arcmin::Real; niter::Integer=0)
    fwhm_arcmin > 0 || error("fwhm_arcmin must be positive.")
    validate_healpix_map(m, "Map passed to smooth_healpix_map_gaussian!")

    any(!iszero, m.pixels) || return m

    fwhm_rad = deg2rad(Float64(fwhm_arcmin) / 60.0)
    alm = Healpix.map2alm(m; niter=niter)
    beam = Healpix.gaussbeam(fwhm_rad, alm.lmax)
    Healpix.almxfl!(alm, beam)
    smoothed_map = Healpix.alm2map(alm, m.resolution.nside)
    m.pixels .= smoothed_map.pixels
    validate_healpix_map(m, "Smoothed Healpix map")
    return m
end

function mark_healpix_disc_support!(
    support_mask::BitVector,
    workspace::XGPaint.HealpixRingProfileWorkspace{T},
    alpha::Real,
    delta::Real,
    radius_rad::Real
) where {T}
    radius_rad > 0 || return support_mask

    center_theta = T(pi / 2 - delta)
    center_phi = mod(T(alpha), T(2pi))
    search_radius = T(radius_rad)

    ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, search_radius)
    @inbounds for ring_idx in ring_start:ring_stop
        range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring_idx, center_theta, center_phi, search_radius)
        first_pixel = workspace.ring_first_pixels[ring_idx]
        for local_pix_idx in range1
            support_mask[first_pixel + local_pix_idx - 1] = true
        end
        for local_pix_idx in range2
            support_mask[first_pixel + local_pix_idx - 1] = true
        end
    end

    return support_mask
end

function mark_healpix_profile_support!(
    support_mask::BitVector,
    workspace::XGPaint.HealpixRingProfileWorkspace{Float64},
    profile_model,
    masses,
    redshifts,
    ras,
    decs,
    extra_radius_rad::Float64
)
    @assert length(masses) == length(redshifts) == length(ras) == length(decs)

    @inbounds for i in eachindex(masses)
        theta_max = Float64(XGPaint.compute_θmax(profile_model, masses[i] * XGPaint.M_sun, redshifts[i]))
        mark_healpix_disc_support!(
            support_mask,
            workspace,
            ras[i],
            decs[i],
            theta_max + extra_radius_rad
        )
    end

    return support_mask
end

function apply_support_mask!(m::HealpixMap{<:Real, RingOrder}, support_mask::BitVector)
    length(m.pixels) == length(support_mask) || error("Support mask length does not match Healpix map length.")
    @inbounds for i in eachindex(m.pixels)
        support_mask[i] || (m.pixels[i] = 0.0)
    end
    return m
end

# -------------------------
# density + selection
# -------------------------
rho_m = 2.775e11 * omegam * h^2
selection = apply_mass_cut

# -------------------------
# model + map init
# -------------------------
reuse_cached_tsz_map = isfile(fits_output_path_tsz)
reuse_cached_frb_hosts = isfile(frb_hosts_cache_path)
reuse_cached_frb_map = reuse_cached_frb_hosts && isfile(fits_output_path_frb)

println("Cache status:")
println("  tSZ map cache: $(reuse_cached_tsz_map ? "found" : "missing")")
println("  FRB host cache: $(reuse_cached_frb_hosts ? "found" : "missing")")
if reuse_cached_frb_map
    println("  FRB map cache: found")
elseif isfile(fits_output_path_frb)
    println("  FRB map cache: map-only cache found without FRB host sidecar")
    println("  note: regenerating the FRB sample so the DM map stays consistent with the FRB map configuration.")
else
    println("  FRB map cache: missing")
end

y_cache_file = "cached_tSZ_Websky_cosmo_$(param_tag).jld2"
dm_cache_file = "cached_FRB_true_DM_Websky_cosmo.jld2"
y_model_interp = nothing

if !reuse_cached_tsz_map
    model = Battaglia16ThermalSZProfile(
        Omega_c=omegac,
        Omega_b=omegab,
        h=h_value,
        P0_amp=battaglia_P0_amp,
        P0_alpha_m=battaglia_P0_alpha_m,
        P0_alpha_z=battaglia_P0_alpha_z,
        x_c_amp=battaglia_x_c_amp,
        x_c_alpha_m=battaglia_x_c_alpha_m,
        x_c_alpha_z=battaglia_x_c_alpha_z,
        beta_amp=battaglia_beta_amp,
        beta_alpha_m=battaglia_beta_alpha_m,
        beta_alpha_z=battaglia_beta_alpha_z,
        alpha_amp=battaglia_alpha_amp,
        alpha_alpha_m=battaglia_alpha_alpha_m,
        alpha_alpha_z=battaglia_alpha_alpha_z,
        gamma_amp=battaglia_gamma_amp,
        gamma_alpha_m=battaglia_gamma_alpha_m,
        gamma_alpha_z=battaglia_gamma_alpha_z
    )

    if model_exists
        y_model_interp = build_interpolator(
            model,
            cache_file=y_cache_file,
            overwrite=false
        )
    else
        y_model_interp = build_interpolator(
            model;
            cache_file=y_cache_file,
            pad=256,
            logM_max=15.7,
            overwrite=true,
            verbose=true
        )
    end
else
    println("Reusing cached tSZ Healpix map and skipping tSZ halo painting.")
end

# Paint the true observed FRB DM halo contribution in pc/cm^3.
# HaloDMProfile reuses the Battaglia electron profile but evaluates it as DM
# instead of the dimensionless tau field.
dm_model = HaloDMProfile(BattagliaTauProfile(Omega_c=omegac, Omega_b=omegab, h=h_value))
if model_exists
    dm_model_interp = build_interpolator(
        dm_model,
        cache_file=dm_cache_file,
        overwrite=false
    )
else
    dm_model_interp = build_interpolator(
        dm_model;
        cache_file=dm_cache_file,
        pad=256,
        logM_max=15.7,
        overwrite=true,
        verbose=true
    )
end

if reuse_cached_tsz_map
    m_hp = load_healpix_map_from_fits(fits_output_path_tsz, nside, "tSZ map")
else
    m_hp = HealpixMap{Float64, RingOrder}(nside)
    fill!(m_hp.pixels, 0.0)
end
dm_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(dm_hp.pixels, 0.0)
dm_raw_frb_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(dm_raw_frb_hp.pixels, 0.0)
dm_residual_hp = HealpixMap{Float64, RingOrder}(nside)
fill!(dm_residual_hp.pixels, 0.0)
res = Healpix.Resolution(nside)
w = XGPaint.HealpixRingProfileWorkspace{Float64}(res)
tmp_hp = reuse_cached_tsz_map ? nothing : HealpixMap{Float64, RingOrder}(nside)
if !isnothing(tmp_hp)
    fill!(tmp_hp.pixels, 0.0)
end
tsz_beam_support_extra_radius_rad = tsz_beam_support_radius_arcmin * ARCMIN2RAD
dm_beam_support_extra_radius_rad = dm_beam_support_radius_arcmin * ARCMIN2RAD
tsz_support_mask = (!reuse_cached_tsz_map && apply_tsz_gaussian_beam && truncate_beam_support) ? falses(length(m_hp.pixels)) : nothing
dm_support_mask = (apply_dm_gaussian_beam && truncate_beam_support) ? falses(length(dm_hp.pixels)) : nothing
dm_point_support_mask = (apply_dm_gaussian_beam && truncate_beam_support) ? falses(length(dm_raw_frb_hp.pixels)) : nothing
tsz_map_loaded_from_cache = reuse_cached_tsz_map
frb_map_loaded_from_cache = false
frb_hosts_loaded_from_cache = false

rng = MersenneTwister(frb_seed)
d_l_cut = luminosity_distance(frb_z_cut, chi_of_z_itp)
d_l_cut_sq = d_l_cut^2

println("Initiating HealPix with NSide: $nside")
println("Julia threads available: $(nthreads())")
println("Using halo catalog source: $catalog_source")

sample_x = Float64[]
sample_y = Float64[]
sample_z = Float64[]
sample_mass = Float64[]
sample_redshift = Float64[]
eligible_frb_redshifts = nothing

if reuse_cached_frb_hosts
    sample_x, sample_y, sample_z, sample_mass, sample_redshift = load_frb_host_cache(frb_hosts_cache_path, frb_count)
    frb_hosts_loaded_from_cache = true
else
    # -------------------------
    # first pass: build y map and collect FRB host candidates after the mass cut
    # -------------------------
    eligible_frb_halo_indices = Int[]
    eligible_frb_halo_weights = Float64[]
    eligible_frb_redshifts = Float64[]

    total_halo_count = stream_catalog_batches(
        catalog_source,
        halfdome_path,
        websky_path,
        chunkN,
        itp_z_of_chi,
        rho_m
    ) do batch_start_halo_index, x, y, z, halo_mass, redshift
        append_frb_candidate_halos!(
            eligible_frb_halo_indices,
            eligible_frb_halo_weights,
            eligible_frb_redshifts,
            batch_start_halo_index,
            halo_mass,
            redshift;
            selection=selection,
            mass_min=mass_min,
            frb_selection_mode=frb_selection_mode,
            frb_z_max=frb_z_max,
            chi_of_z_itp=chi_of_z_itp,
            d_l_cut_sq=d_l_cut_sq
        )

        if !reuse_cached_tsz_map
            paint_tsz_batch!(
                m_hp,
                tmp_hp,
                w,
                y_model_interp,
                x,
                y,
                z,
                halo_mass,
                redshift;
                selection=selection,
                mass_min=mass_min,
                support_mask=tsz_support_mask,
                support_extra_radius_rad=tsz_beam_support_extra_radius_rad
            )
        end
    end

    eligible_frb_count = length(eligible_frb_halo_indices)
    println("Total halos in catalog: $(total_halo_count)")
    if frb_selection_mode == "redshift"
        println("Mass-cut FRB host candidates with z <= $(frb_z_max): $(eligible_frb_count)")
    else
        println("Mass-cut FRB host candidates: $(eligible_frb_count)")
    end

    frb_count <= eligible_frb_count || error(
        frb_selection_mode == "redshift" ?
            "frb_count=$(frb_count) exceeds the number of mass-cut FRB host candidates " *
            "with z <= $(frb_z_max), which is $(eligible_frb_count)." :
            "frb_count=$(frb_count) exceeds the number of mass-cut FRB host candidates, which is $(eligible_frb_count)."
    )

    if frb_selection_mode == "redshift"
        eligible_frb_halo_weights, corrected_weight_bin_edges, corrected_weight_halo_pdf = compute_corrected_frb_selection_weights(
            eligible_frb_redshifts,
            chi_of_z_itp,
            d_l_cut_sq,
            frb_z_max;
            dz=frb_redshift_weight_bin_width
        )
        println(
            "Computed corrected FRB halo weights using the eligible-halo redshift density with " *
            "$(length(corrected_weight_halo_pdf)) bins of width $(frb_redshift_weight_bin_width)."
        )
    end

    selected_candidate_positions = draw_weighted_sample_positions(rng, eligible_frb_halo_weights, frb_count)
    sampled_halo_indices = sort!(eligible_frb_halo_indices[selected_candidate_positions])

    sample_x = Vector{Float64}(undef, frb_count)
    sample_y = Vector{Float64}(undef, frb_count)
    sample_z = Vector{Float64}(undef, frb_count)
    sample_mass = Vector{Float64}(undef, frb_count)
    sample_redshift = Vector{Float64}(undef, frb_count)

    # -------------------------
    # second pass: capture the selected FRB host halos
    # -------------------------
    next_sampled_index_pos_ref = Ref(1)
    captured_host_count_ref = Ref(0)

    stream_catalog_batches(
        catalog_source,
        halfdome_path,
        websky_path,
        chunkN,
        itp_z_of_chi,
        rho_m
    ) do batch_start_halo_index, x, y, z, halo_mass, redshift
        next_sampled_index_pos_ref[], captured_host_count_ref[] = capture_selected_halos!(
            sample_x,
            sample_y,
            sample_z,
            sample_mass,
            sample_redshift,
            sampled_halo_indices,
            next_sampled_index_pos_ref[],
            captured_host_count_ref[],
            batch_start_halo_index,
            x,
            y,
            z,
            halo_mass,
            redshift
        )
    end

    captured_frb_host_count = captured_host_count_ref[]
    captured_frb_host_count == frb_count || error(
        "Expected to capture $(frb_count) FRB host halos, but only found $(captured_frb_host_count)."
    )

    if save_healpix_maps
        saved_frb_host_cache_path = save_frb_host_cache(
            frb_hosts_cache_path,
            sample_x,
            sample_y,
            sample_z,
            sample_mass,
            sample_redshift
        )
        println("Saved FRB host cache to $(saved_frb_host_cache_path)")
    end
end

if reuse_cached_frb_map
    frb_map = load_healpix_map_from_fits(fits_output_path_frb, nside, "FRB overdensity map")
    frb_map_loaded_from_cache = true
else
    if frb_hosts_loaded_from_cache
        println("Rebuilding FRB overdensity map from cached FRB host sample.")
    end
    frb_map = build_frb_overdensity_map(sample_x, sample_y, sample_z, nside)
end

# Paint the DM map only from the sampled FRB host halos.
sample_ra_unsorted, sample_dec_unsorted = xyz_to_ra_dec_threaded(sample_x, sample_y, sample_z)
sample_redshift_unsorted = copy(sample_redshift)
sample_mass_unsorted = copy(sample_mass)
selected_halo_hist_saved_path = save_selected_frb_host_histogram_plot(
    sample_redshift_unsorted,
    sample_mass_unsorted,
    selected_halo_hist_output_path
)
println("Saved selected FRB host halo histogram plot to $(selected_halo_hist_saved_path)")
sample_perm = sortperm(sample_dec_unsorted)
sample_ra = sample_ra_unsorted[sample_perm]
sample_dec = sample_dec_unsorted[sample_perm]
sample_redshift = sample_redshift_unsorted[sample_perm]
sample_mass = sample_mass_unsorted[sample_perm]

if !isnothing(dm_support_mask)
    mark_healpix_profile_support!(
        dm_support_mask,
        w,
        dm_model_interp,
        sample_mass,
        sample_redshift,
        sample_ra,
        sample_dec,
        dm_beam_support_extra_radius_rad
    )
end
if !isnothing(dm_point_support_mask)
    @inbounds for i in eachindex(sample_ra_unsorted)
        mark_healpix_disc_support!(
            dm_point_support_mask,
            w,
            sample_ra_unsorted[i],
            sample_dec_unsorted[i],
            dm_beam_support_extra_radius_rad
        )
    end
end

positive_sample_redshift = sample_redshift[sample_redshift .> 0.0]
isempty(positive_sample_redshift) && error("Need at least one positive sampled FRB redshift to make the histograms.")

plot_z_linear_limits = expanded_linear_limits(0.0, maximum(sample_redshift))
plot_z_log_limits = expanded_positive_limits(minimum(positive_sample_redshift), maximum(positive_sample_redshift))

linear_bins = collect(range(plot_z_linear_limits[1], plot_z_linear_limits[2]; length=26))
pdf_z_grid_linear = collect(range(plot_z_linear_limits[1], plot_z_linear_limits[2]; length=1000))

if frb_selection_mode == "redshift"
    pdf_values_linear = evaluate_normalized_frb_redshift_pdf(pdf_z_grid_linear, chi_of_z_itp, d_l_cut_sq, frb_z_max)
    linear_expected_counts = expected_histogram_counts(pdf_z_grid_linear, pdf_values_linear, linear_bins, length(sample_redshift))
else
    reference_redshifts_linear = isnothing(eligible_frb_redshifts) ? sample_redshift : eligible_frb_redshifts
    if isnothing(eligible_frb_redshifts)
        println("  note: random FRB redshift reference curve is using the sampled FRB redshifts because the full eligible-halo redshift list was not available from cache.")
    end
    reference_bins_linear = collect(range(plot_z_linear_limits[1], plot_z_linear_limits[2]; length=201))
    reference_counts_linear = histogram_counts(reference_redshifts_linear, reference_bins_linear)
    reference_pdf_linear = pdf_from_histogram_counts(reference_counts_linear, reference_bins_linear)
    pdf_values_linear = evaluate_histogram_pdf(pdf_z_grid_linear, reference_bins_linear, reference_pdf_linear)
    linear_expected_counts = expected_histogram_counts(pdf_z_grid_linear, pdf_values_linear, linear_bins, length(sample_redshift))
end

linear_curve_x, linear_curve_y = positive_finite_xy(pdf_z_grid_linear, linear_expected_counts)

p_linear = histogram(
    sample_redshift;
    bins=linear_bins,
    xlims=plot_z_linear_limits,
    yscale=:log10,
    color=:grey,
    linecolor=:grey,
    alpha=0.9,
    label="",
    xlabel="FRB redshift",
    ylabel="counts"
)
if !isempty(linear_curve_x)
    plot!(p_linear, linear_curve_x, linear_curve_y; color=:red, linewidth=2, label="")
end
linear_saved_path = save_plot_accessible(p_linear, linear_hist_output_path)
println("Saved linear redshift histogram to $(linear_saved_path)")

log_bins = 10 .^ range(log10(plot_z_log_limits[1]), log10(plot_z_log_limits[2]); length=26)
pdf_z_grid_log = collect(range(plot_z_log_limits[1], plot_z_log_limits[2]; length=1000))

if frb_selection_mode == "redshift"
    pdf_values_log = evaluate_normalized_frb_redshift_pdf(pdf_z_grid_log, chi_of_z_itp, d_l_cut_sq, frb_z_max)
    log_expected_counts = expected_histogram_counts(pdf_z_grid_log, pdf_values_log, log_bins, length(sample_redshift))
else
    reference_redshifts_log_source = isnothing(eligible_frb_redshifts) ? sample_redshift : eligible_frb_redshifts
    reference_redshifts_log = reference_redshifts_log_source[reference_redshifts_log_source .> 0.0]
    reference_bins_log = 10 .^ range(log10(plot_z_log_limits[1]), log10(plot_z_log_limits[2]); length=201)
    reference_counts_log = histogram_counts(reference_redshifts_log, reference_bins_log)
    reference_pdf_log = pdf_from_histogram_counts(reference_counts_log, reference_bins_log)
    pdf_values_log = evaluate_histogram_pdf(pdf_z_grid_log, reference_bins_log, reference_pdf_log)
    log_expected_counts = expected_histogram_counts(pdf_z_grid_log, pdf_values_log, log_bins, length(sample_redshift))
end

log_curve_x, log_curve_y = positive_finite_xy(pdf_z_grid_log, log_expected_counts)

p_log = histogram(
    sample_redshift;
    bins=log_bins,
    xlims=plot_z_log_limits,
    xscale=:log10,
    yscale=:log10,
    color=:grey,
    linecolor=:grey,
    alpha=0.9,
    label="",
    xlabel="FRB redshift",
    ylabel="counts"
)
if !isempty(log_curve_x)
    plot!(p_log, log_curve_x, log_curve_y; color=:red, linewidth=2, label="")
end
log_saved_path = save_plot_accessible(p_log, log_hist_output_path)
println("Saved log-x redshift histogram to $(log_saved_path)")

fill!(dm_hp.pixels, 0.0)
paint!(dm_hp, w, dm_model_interp, sample_mass, sample_redshift, sample_ra, sample_dec)

sample_dm_raw = sample_healpix_map_at_points(dm_hp, sample_ra_unsorted, sample_dec_unsorted)
dm_mean_bin_centers, dm_mean_bin_values, dm_mean_bin_counts, mu_dm_at_z = compute_binned_mean_dm(
    sample_redshift_unsorted,
    sample_dm_raw;
    z_min=0.0,
    z_max=maximum(sample_redshift_unsorted),
    dz=dm_residual_bin_width
)
sample_dm_mean = mu_dm_at_z.(sample_redshift_unsorted)
sample_dm_residual = sample_dm_raw .- sample_dm_mean

dm_raw_frb_hp = build_frb_weighted_map(
    sample_x,
    sample_y,
    sample_z,
    sample_dm_raw,
    nside;
    normalize_by_mean_density=normalize_weighted_frb_map_by_density
)
dm_residual_hp = build_frb_weighted_map(
    sample_x,
    sample_y,
    sample_z,
    sample_dm_residual,
    nside;
    normalize_by_mean_density=normalize_weighted_frb_map_by_density
)
dm_analysis_hp = use_dm_residual_for_cross ? dm_residual_hp : dm_raw_frb_hp

println(
    "Using $(use_dm_residual_for_cross ? "DM residuals" : "raw DM") " *
    "$(normalize_weighted_frb_map_by_density ? "with mean surface-density normalization" : "without mean surface-density normalization") " *
    "for downstream FRB x tSZ cross-correlations."
)

if save_frb_catalog
    saved_frb_catalog_path = save_frb_dm_catalog(
        frb_catalog_output_path;
        sample_x=sample_x,
        sample_y=sample_y,
        sample_z=sample_z,
        sample_ra=sample_ra_unsorted,
        sample_dec=sample_dec_unsorted,
        sample_mass=sample_mass_unsorted,
        sample_redshift=sample_redshift_unsorted,
        sample_dm_raw=sample_dm_raw,
        sample_dm_mean=sample_dm_mean,
        sample_dm_residual=sample_dm_residual,
        dm_mean_bin_centers=dm_mean_bin_centers,
        dm_mean_bin_values=dm_mean_bin_values,
        dm_mean_bin_counts=dm_mean_bin_counts,
        dm_residual_bin_width=dm_residual_bin_width
    )
    println("Saved FRB DM catalog to $(saved_frb_catalog_path)")
end

if save_dm_residual_diagnostic
    saved_dm_residual_diagnostic_path = save_dm_residual_diagnostic_plot(
        sample_redshift_unsorted,
        sample_dm_raw,
        sample_dm_residual,
        dm_mean_bin_centers,
        dm_mean_bin_values,
        dm_residual_diagnostic_output_path
    )
    println("Saved DM residual diagnostic plot to $(saved_dm_residual_diagnostic_path)")
end

if !tsz_map_loaded_from_cache && apply_tsz_gaussian_beam
    println("Applying Gaussian beam to tSZ map in map space with FWHM=$(tsz_gaussian_beam_fwhm_arcmin) arcmin.")
    smooth_healpix_map_gaussian!(m_hp, tsz_gaussian_beam_fwhm_arcmin)
    if truncate_beam_support
        println("Truncating beam-smoothed tSZ support to the original painted halo support plus $(tsz_beam_support_radius_arcmin) arcmin.")
        apply_support_mask!(m_hp, tsz_support_mask)
    end
end
if apply_dm_gaussian_beam
    println("Applying Gaussian beam to DM map in map space with FWHM=$(dm_gaussian_beam_fwhm_arcmin) arcmin.")
    smooth_healpix_map_gaussian!(dm_hp, dm_gaussian_beam_fwhm_arcmin)
    smooth_healpix_map_gaussian!(dm_raw_frb_hp, dm_gaussian_beam_fwhm_arcmin)
    smooth_healpix_map_gaussian!(dm_residual_hp, dm_gaussian_beam_fwhm_arcmin)
    if truncate_beam_support
        println("Truncating beam-smoothed DM support to the original painted halo support plus $(dm_beam_support_radius_arcmin) arcmin.")
        apply_support_mask!(dm_hp, dm_support_mask)
        apply_support_mask!(dm_raw_frb_hp, dm_point_support_mask)
        apply_support_mask!(dm_residual_hp, dm_point_support_mask)
    end
end

validate_healpix_map(m_hp, "tSZ map")
validate_healpix_map(frb_map, "FRB overdensity map")
validate_healpix_map(dm_hp, "DM profile map")
validate_healpix_map(dm_raw_frb_hp, "Raw FRB DM map")
validate_healpix_map(dm_residual_hp, "DM residual map")

cl_cross = nothing
if save_cl
    m_hp_cl_monopole_removed = copy_healpix_map(m_hp)
    dm_analysis_hp_cl_monopole_removed = copy_healpix_map(dm_analysis_hp)
    remove_monopole!(m_hp_cl_monopole_removed)
    remove_monopole!(dm_analysis_hp_cl_monopole_removed)
    cl_cross = anafast(m_hp_cl_monopole_removed, dm_analysis_hp_cl_monopole_removed, niter=0)
end

if subtract_map_means
    if !tsz_map_loaded_from_cache
        remove_monopole!(m_hp)
    end
    if !frb_map_loaded_from_cache
        remove_monopole!(frb_map)
    end
    remove_monopole!(dm_hp)
    remove_monopole!(dm_raw_frb_hp)
    remove_monopole!(dm_residual_hp)
end

validate_healpix_map(m_hp, "tSZ map after monopole subtraction")
validate_healpix_map(frb_map, "FRB overdensity map after monopole subtraction")
validate_healpix_map(dm_hp, "DM profile map after monopole subtraction")
validate_healpix_map(dm_raw_frb_hp, "Raw FRB DM map after monopole subtraction")
validate_healpix_map(dm_residual_hp, "DM residual map after monopole subtraction")

if save_healpix_maps
    if !tsz_map_loaded_from_cache
        Healpix.saveToFITS(m_hp, "!" * fits_output_path_tsz, typechar="D")
    end
    if !frb_map_loaded_from_cache
        Healpix.saveToFITS(frb_map, "!" * fits_output_path_frb, typechar="D")
    end
    Healpix.saveToFITS(dm_hp, "!" * fits_output_path_dm, typechar="D")
    Healpix.saveToFITS(dm_residual_hp, "!" * fits_output_path_dm_residual, typechar="D")
end

if save_cl
    write_cl_fits_overwrite(cl_output_path, cl_cross)
end

println("Finished Healpix y x true FRB DM cross-correlation")
elapsed = time() - t0
println("Elapsed time: $(round(elapsed; digits=2)) s")
