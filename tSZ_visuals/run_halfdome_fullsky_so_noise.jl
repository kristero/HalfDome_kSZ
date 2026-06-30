using DelimitedFiles
using Healpix
using Random
using Statistics

const SO_NOISE_ELL_MAX_DEFAULT = 7979
const LOCAL_HALFDOME_PATH_DEFAULT = normpath(joinpath(@__DIR__, "..", "lightcone_100.hdf5"))
const LOCAL_SO_NOISE_OUTPUT_DIR_DEFAULT = normpath(joinpath(@__DIR__, "outputs", "so_noise_fullsky"))
const LOCAL_SO_NOISE_CACHE_DIR_DEFAULT = normpath(joinpath(@__DIR__, "cache", "so_noise_fullsky"))

function arg_matches_key(arg::AbstractString, key::AbstractString)
    return startswith(arg, key * "=") || startswith(arg, "--" * key * "=")
end

function has_arg(args::Vector{String}, key::AbstractString)
    return any(arg -> arg_matches_key(arg, key), args)
end

function add_arg_if_missing!(args::Vector{String}, key::AbstractString, value)
    has_arg(args, key) && return args
    push!(args, string(key, "=", value))
    return args
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

replace_or_add_arg!(ARGS, "catalog_source", "halfdome")
replace_or_add_arg!(ARGS, "batching_mode", "full")
replace_or_add_arg!(ARGS, "save_bin_maps", "false")
replace_or_add_arg!(ARGS, "save_mass_map", "false")
add_arg_if_missing!(ARGS, "apply_gaussian_beam", "true")
add_arg_if_missing!(ARGS, "gaussian_beam_fwhm_arcmin", "2.0")
add_arg_if_missing!(ARGS, "cl_lmax", SO_NOISE_ELL_MAX_DEFAULT)
add_arg_if_missing!(ARGS, "so_noise_lmax", SO_NOISE_ELL_MAX_DEFAULT)
if !has_arg(ARGS, "halfdome_path") && !haskey(ENV, "HALFDOME_PATH") && isfile(LOCAL_HALFDOME_PATH_DEFAULT)
    add_arg_if_missing!(ARGS, "halfdome_path", LOCAL_HALFDOME_PATH_DEFAULT)
end
if !has_arg(ARGS, "output_dir") && !haskey(ENV, "TSZ_VISUAL_OUTPUT_DIR")
    add_arg_if_missing!(ARGS, "output_dir", LOCAL_SO_NOISE_OUTPUT_DIR_DEFAULT)
end
if !has_arg(ARGS, "cache_dir") && !haskey(ENV, "TSZ_VISUAL_CACHE_DIR")
    add_arg_if_missing!(ARGS, "cache_dir", LOCAL_SO_NOISE_CACHE_DIR_DEFAULT)
end

if !haskey(ENV, "APPLY_GAUSSIAN_BEAM")
    ENV["APPLY_GAUSSIAN_BEAM"] = "true"
end
if !haskey(ENV, "GAUSSIAN_BEAM_FWHM_ARCMIN")
    ENV["GAUSSIAN_BEAM_FWHM_ARCMIN"] = "2.0"
end

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))

const SO_NOISE_BASELINE_FILENAME = "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
const SO_NOISE_GOAL_FILENAME = "SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"
const SO_NOISE_DEPROJECTION_DEFAULT = 3

function default_so_noise_path(filename::AbstractString)
    return joinpath(repo_root(), "other_sims", "SO", String(filename))
end

Base.@kwdef struct HalfDomeFullSkySONoiseConfig
    base_cfg::VisualConfig
    baseline_noise_path::String
    goal_noise_path::String
    noise_deprojections::Vector{Int}
    noise_deprojection::Int
    noise_column::Int
    noise_is_dl::Bool
    noise_lmax::Int
    fsky::Float64
    mask_apodization_arcmin::Float64
    seed::Int
    save_unmasked_no_noise_cl::Bool
    save_no_noise_cl::Bool
    save_baseline_noise_cross_cl::Bool
    save_goal_noise_cross_cl::Bool
    save_noise_maps::Bool
    save_mask_map::Bool
    save_signal_map::Bool
    save_masked_signal_map::Bool
    save_noisy_maps::Bool
end

function so_noise_analysis_lmax(base_cfg::VisualConfig)
    return base_cfg.cl_lmax < 0 ? healpix_default_lmax(base_cfg.nside) : base_cfg.cl_lmax
end

function has_arg_or_env(key::AbstractString, env::AbstractString)
    haskey(ENV, env) && return true
    prefix1 = "--" * key * "="
    prefix2 = key * "="
    return any(arg -> startswith(arg, prefix1) || startswith(arg, prefix2), ARGS)
end

function so_noise_column_from_deprojection(deprojection::Integer)
    0 <= deprojection <= 3 || error("so_noise_deprojection must be one of 0, 1, 2, or 3.")
    return Int(deprojection) + 2
end

function parse_int_list_arg(value)
    values = Int[]
    for part in split(replace(String(value), ";" => ","), ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(values, parse(Int, stripped))
    end
    isempty(values) && error("Expected at least one integer value, got $(repr(value)).")
    return unique(values)
end

function load_halfdome_fullsky_so_noise_config()
    base_cfg = load_visual_config()
    base_cfg.catalog_source == "halfdome" || error("This script is only for HalfDome. Set catalog_source=halfdome.")
    base_cfg.batching_mode == "full" || error("This script is only for full-sky HalfDome maps. Set batching_mode=full.")

    analysis_lmax = so_noise_analysis_lmax(base_cfg)
    baseline_noise_path = resolve_repo_path(get_string_arg(
        "baseline_noise_path",
        default_so_noise_path(SO_NOISE_BASELINE_FILENAME);
        env="TSZ_SO_BASELINE_NOISE_PATH"
    ))
    goal_noise_path = resolve_repo_path(get_string_arg(
        "goal_noise_path",
        default_so_noise_path(SO_NOISE_GOAL_FILENAME);
        env="TSZ_SO_GOAL_NOISE_PATH"
    ))
    use_deprojection_list = has_arg_or_env("so_noise_deprojections", "TSZ_SO_NOISE_DEPROJECTIONS")
    use_explicit_column = has_arg_or_env("so_noise_column", "TSZ_SO_NOISE_COLUMN")
    use_deprojection_list && use_explicit_column && error(
        "Use either so_noise_deprojections/TSZ_SO_NOISE_DEPROJECTIONS or " *
        "so_noise_column/TSZ_SO_NOISE_COLUMN, not both."
    )

    noise_deprojections = if use_deprojection_list
        parse_int_list_arg(get_string_arg(
            "so_noise_deprojections",
            string(SO_NOISE_DEPROJECTION_DEFAULT);
            env="TSZ_SO_NOISE_DEPROJECTIONS"
        ))
    elseif use_explicit_column
        noise_column = get_int_arg(
            "so_noise_column",
            so_noise_column_from_deprojection(SO_NOISE_DEPROJECTION_DEFAULT);
            env="TSZ_SO_NOISE_COLUMN"
        )
        [noise_column - 2]
    else
        [get_int_arg(
            "so_noise_deprojection",
            SO_NOISE_DEPROJECTION_DEFAULT;
            env="TSZ_SO_NOISE_DEPROJECTION"
        )]
    end
    for deprojection in noise_deprojections
        0 <= deprojection <= 3 || error("SO noise deprojections must be in 0, 1, 2, 3; got $(noise_deprojections).")
    end
    noise_deprojection = first(noise_deprojections)
    noise_column = so_noise_column_from_deprojection(noise_deprojection)
    noise_is_dl = get_bool_arg("so_noise_is_dl", false; env="TSZ_SO_NOISE_IS_DL")
    noise_lmax = get_int_arg("so_noise_lmax", analysis_lmax; env="TSZ_SO_NOISE_LMAX")
    fsky = get_float_arg("mask_fsky", 0.4; env="TSZ_MASK_FSKY")
    mask_apodization_arcmin = get_float_arg(
        "mask_apodization_arcmin",
        60.0;
        env="TSZ_MASK_APODIZATION_ARCMIN"
    )
    seed = get_int_arg("seed", 12345; env="TSZ_FULLSKY_NOISE_SEED")
    save_unmasked_no_noise_cl = get_bool_arg(
        "save_unmasked_no_noise_cl",
        false;
        env="TSZ_SAVE_UNMASKED_NO_NOISE_CL"
    )
    save_no_noise_cl = get_bool_arg("save_no_noise_cl", true; env="TSZ_SAVE_NO_NOISE_CL")
    save_baseline_noise_cross_cl = get_bool_arg(
        "save_baseline_noise_cross_cl",
        true;
        env="TSZ_SAVE_BASELINE_NOISE_CROSS_CL"
    )
    save_goal_noise_cross_cl = get_bool_arg(
        "save_goal_noise_cross_cl",
        true;
        env="TSZ_SAVE_GOAL_NOISE_CROSS_CL"
    )
    save_noise_maps = get_bool_arg("save_noise_maps", false; env="TSZ_SAVE_SO_NOISE_MAPS")
    save_mask_map = get_bool_arg("save_mask_map", false; env="TSZ_SAVE_SO_MASK_MAP")
    save_signal_map = get_bool_arg("save_signal_map", false; env="TSZ_SAVE_SIGNAL_MAP")
    save_masked_signal_map = get_bool_arg("save_masked_signal_map", false; env="TSZ_SAVE_MASKED_SIGNAL_MAP")
    save_noisy_maps = get_bool_arg("save_noisy_maps", false; env="TSZ_SAVE_NOISY_SPLIT_MAPS")

    isfile(baseline_noise_path) || error("Baseline SO noise spectrum not found: $(baseline_noise_path)")
    isfile(goal_noise_path) || error("Goal SO noise spectrum not found: $(goal_noise_path)")
    0 <= noise_deprojection <= 3 || error(
        "so_noise_column=$(noise_column) is invalid for the SO file. " *
        "Use columns 2:5 for Deproj-0:3, or set so_noise_deprojection=0,1,2,3."
    )
    noise_lmax >= 0 || error("so_noise_lmax must be nonnegative.")
    noise_lmax <= healpix_default_lmax(base_cfg.nside) || error(
        "so_noise_lmax=$(noise_lmax) exceeds Healpix default maximum " *
        "$(healpix_default_lmax(base_cfg.nside)) for nside=$(base_cfg.nside)."
    )
    0.0 < fsky < 1.0 || error("mask_fsky must lie in (0, 1).")
    mask_apodization_arcmin >= 0.0 || error("mask_apodization_arcmin must be nonnegative.")

    return HalfDomeFullSkySONoiseConfig(
        base_cfg=base_cfg,
        baseline_noise_path=baseline_noise_path,
        goal_noise_path=goal_noise_path,
        noise_deprojections=noise_deprojections,
        noise_deprojection=noise_deprojection,
        noise_column=noise_column,
        noise_is_dl=noise_is_dl,
        noise_lmax=noise_lmax,
        fsky=fsky,
        mask_apodization_arcmin=mask_apodization_arcmin,
        seed=seed,
        save_unmasked_no_noise_cl=save_unmasked_no_noise_cl,
        save_no_noise_cl=save_no_noise_cl,
        save_baseline_noise_cross_cl=save_baseline_noise_cross_cl,
        save_goal_noise_cross_cl=save_goal_noise_cross_cl,
        save_noise_maps=save_noise_maps,
        save_mask_map=save_mask_map,
        save_signal_map=save_signal_map,
        save_masked_signal_map=save_masked_signal_map,
        save_noisy_maps=save_noisy_maps
    )
end

function so_noise_tag(cfg::HalfDomeFullSkySONoiseConfig)
    return join(
        (
            "so",
            "fsky$(fmt_param_value(cfg.fsky))",
            "apo$(fmt_param_value(cfg.mask_apodization_arcmin))arcmin",
            "seed$(cfg.seed)",
            "deproj$(cfg.noise_deprojection)"
        ),
        "_"
    )
end

function so_output_path(cfg::HalfDomeFullSkySONoiseConfig, label::AbstractString, extension::AbstractString)
    base_cfg = cfg.base_cfg
    lmax_tag = build_cl_lmax_tag(base_cfg.cl_lmax)
    stem = join(
        (
            "halfdome_fullsky",
            String(label),
            "m200c",
            "nside$(base_cfg.nside)",
            base_cfg.param_tag,
            base_cfg.cosmology_tag,
            base_cfg.beam_tag,
            so_noise_tag(cfg),
            lmax_tag
        ),
        "_"
    )
    return joinpath(base_cfg.output_dir, make_output_filename(stem, String(extension)))
end

function with_noise_deprojection(cfg::HalfDomeFullSkySONoiseConfig, deprojection::Integer)
    return HalfDomeFullSkySONoiseConfig(
        base_cfg=cfg.base_cfg,
        baseline_noise_path=cfg.baseline_noise_path,
        goal_noise_path=cfg.goal_noise_path,
        noise_deprojections=cfg.noise_deprojections,
        noise_deprojection=Int(deprojection),
        noise_column=so_noise_column_from_deprojection(deprojection),
        noise_is_dl=cfg.noise_is_dl,
        noise_lmax=cfg.noise_lmax,
        fsky=cfg.fsky,
        mask_apodization_arcmin=cfg.mask_apodization_arcmin,
        seed=cfg.seed,
        save_unmasked_no_noise_cl=cfg.save_unmasked_no_noise_cl,
        save_no_noise_cl=cfg.save_no_noise_cl,
        save_baseline_noise_cross_cl=cfg.save_baseline_noise_cross_cl,
        save_goal_noise_cross_cl=cfg.save_goal_noise_cross_cl,
        save_noise_maps=cfg.save_noise_maps,
        save_mask_map=cfg.save_mask_map,
        save_signal_map=cfg.save_signal_map,
        save_masked_signal_map=cfg.save_masked_signal_map,
        save_noisy_maps=cfg.save_noisy_maps
    )
end

function print_halfdome_fullsky_so_noise_config(cfg::HalfDomeFullSkySONoiseConfig)
    base_cfg = cfg.base_cfg
    println("Running full-sky HalfDome tSZ SO-noise test for one profile/parameter set.")
    println("Output directory: $(abspath(base_cfg.output_dir))")
    println("HalfDome catalog: $(abspath(base_cfg.halfdome_path))")
    println("NSIDE: $(base_cfg.nside)")
    println("C_l lmax: $(so_noise_analysis_lmax(base_cfg))")
    println("Noise map lmax: $(cfg.noise_lmax)")
    println("Gaussian beam on tSZ signal: apply=$(base_cfg.apply_gaussian_beam), fwhm_arcmin=$(base_cfg.gaussian_beam_fwhm_arcmin)")
    println("Battaglia parameter tag: $(base_cfg.param_tag)")
    println("Battaglia pressure/profile parameters:")
    for field in propertynames(base_cfg.battaglia_params)
        println("  $(field)=$(getproperty(base_cfg.battaglia_params, field))")
    end
    println("Cosmology tag: $(base_cfg.cosmology_tag)")
    println("Mass cut: apply=$(base_cfg.apply_mass_cut), mass_min=$(base_cfg.mass_min)")
    println("SO baseline noise spectrum: $(abspath(cfg.baseline_noise_path))")
    println("SO goal noise spectrum: $(abspath(cfg.goal_noise_path))")
    println("SO noise deprojections to run: $(join(cfg.noise_deprojections, ", "))")
    println("Current SO noise deprojection: Deproj-$(cfg.noise_deprojection)")
    println("Current SO noise column: $(cfg.noise_column)")
    println("SO noise input treated as D_l: $(cfg.noise_is_dl)")
    println("Mask support fsky: $(cfg.fsky)")
    println("Mask apodization: $(cfg.mask_apodization_arcmin) arcmin")
    println("Random seed: $(cfg.seed)")
    println("Save unmasked no-noise C_l: $(cfg.save_unmasked_no_noise_cl)")
    println("Save masked no-noise C_l: $(cfg.save_no_noise_cl)")
    println("Save baseline split-noise cross C_l: $(cfg.save_baseline_noise_cross_cl)")
    println("Save goal split-noise cross C_l: $(cfg.save_goal_noise_cross_cl)")
    println("Save raw split noise maps: $(cfg.save_noise_maps)")
    println("Save mask map: $(cfg.save_mask_map)")
    println("Save clean signal map: $(cfg.save_signal_map)")
    println("Save masked clean signal map: $(cfg.save_masked_signal_map)")
    println("Save masked noisy split maps: $(cfg.save_noisy_maps)")
    println("No-noise C_l NumPy array: $(abspath(so_output_path(cfg, "masked_no_noise_cl", ".npy")))")
    println("Baseline split-noise cross C_l NumPy array: $(abspath(so_output_path(cfg, "masked_baseline_noise_cross_cl", ".npy")))")
    println("Goal split-noise cross C_l NumPy array: $(abspath(so_output_path(cfg, "masked_goal_noise_cross_cl", ".npy")))")
end

function read_so_noise_native_cl(
    path::AbstractString,
    lmax::Integer;
    column::Integer=2,
    input_is_dl::Bool=false
)
    raw = readdlm(path, Float64; comments=true, comment_char='#')
    ndims(raw) == 2 || error("Noise spectrum $(path) did not read as a 2D table.")
    size(raw, 2) >= column || error(
        "Noise spectrum $(path) has $(size(raw, 2)) columns, but so_noise_column=$(column)."
    )

    ell_values = Int[]
    cl_values = Float64[]

    for row in axes(raw, 1)
        ell_float = raw[row, 1]
        isfinite(ell_float) || continue
        ell = Int(round(ell_float))
        0 <= ell <= lmax || continue

        value = raw[row, column]
        isfinite(value) || continue
        cl_value = if input_is_dl
            ell > 0 ? Float64(value) * 2.0 * pi / (ell * (ell + 1.0)) : 0.0
        else
            Float64(value)
        end
        isfinite(cl_value) || continue
        cl_value >= 0.0 || continue

        push!(ell_values, ell)
        push!(cl_values, cl_value)
    end

    isempty(ell_values) && error("No usable multipoles were read from $(path) up to lmax=$(lmax).")
    order = sortperm(ell_values)
    return ell_values[order], cl_values[order]
end

function so_noise_cl_vector_for_synalm(
    ell_values::AbstractVector{<:Integer},
    cl_values::AbstractVector{<:Real},
    lmax::Integer
)
    length(ell_values) == length(cl_values) || error("ell_values and cl_values must have the same length.")

    cl = zeros(Float64, Int(lmax) + 1)
    @inbounds for i in eachindex(ell_values)
        ell = Int(ell_values[i])
        0 <= ell <= lmax || continue
        value = Float64(cl_values[i])
        isfinite(value) && value >= 0.0 || continue
        cl[ell + 1] = value
    end
    return cl
end

function generate_gaussian_noise_map(
    noise_cl::Vector{Float64},
    nside::Integer,
    lmax::Integer,
    rng::AbstractRNG
)
    alm = Healpix.synalm(noise_cl, Int(lmax), Int(lmax), rng)
    return Healpix.alm2map(alm, Int(nside))
end

function random_unit_vector(rng::AbstractRNG)
    z = 2.0 * rand(rng) - 1.0
    phi = 2.0 * pi * rand(rng)
    rxy = sqrt(max(0.0, 1.0 - z^2))
    return rxy * cos(phi), rxy * sin(phi), z
end

function random_apodized_cap_mask(
    nside::Integer,
    fsky::Real,
    apodization_arcmin::Real,
    rng::AbstractRNG
)
    res = Healpix.Resolution(Int(nside))
    mask = HealpixMap{Float64, RingOrder}(Int(nside))
    fill!(mask.pixels, 0.0)

    cx, cy, cz = random_unit_vector(rng)
    radius = acos(1.0 - 2.0 * Float64(fsky))
    apo = deg2rad(Float64(apodization_arcmin) / 60.0)
    apo = min(max(apo, 0.0), 0.99 * radius)
    inner_radius = radius - apo
    support_count = 0

    @inbounds for pix in eachindex(mask.pixels)
        vx, vy, vz = Healpix.pix2vecRing(res, pix)
        cosang = clamp(cx * vx + cy * vy + cz * vz, -1.0, 1.0)
        angle = acos(cosang)

        if angle <= radius
            support_count += 1
            if apo == 0.0 || angle <= inner_radius
                mask.pixels[pix] = 1.0
            else
                x = (angle - inner_radius) / apo
                mask.pixels[pix] = 0.5 * (1.0 + cos(pi * x))
            end
        end
    end

    npix = length(mask.pixels)
    return (
        mask=mask,
        center=(x=cx, y=cy, z=cz),
        radius_rad=radius,
        support_fsky=support_count / npix,
        mean_weight=mean(mask.pixels),
        mean_weight2=mean(abs2, mask.pixels)
    )
end

function print_map_summary(label::AbstractString, m)
    pixels = m.pixels
    min_value, max_value = extrema(pixels)
    println(
        "$(label): min=$(min_value), max=$(max_value), mean=$(mean(pixels)), " *
        "std=$(std(pixels)), nonzero_frac=$(count(!iszero, pixels) / length(pixels))"
    )
    return nothing
end

function paint_halfdome_fullsky_signal_map(cfg::VisualConfig)
    y_model_interp = build_visual_interpolator(cfg)
    state = init_visual_maps(cfg)
    println("Painting full-sky HalfDome tSZ map; nside=$(cfg.nside).")
    run_halfdome_visuals!(cfg, state, y_model_interp)
    print_map_summary("Raw full-sky tSZ signal map", state.m_hp)

    if cfg.apply_gaussian_beam
        println("Applying Gaussian beam to full-sky tSZ map with FWHM=$(cfg.gaussian_beam_fwhm_arcmin) arcmin.")
    else
        println("Gaussian beam disabled for the full-sky tSZ signal map.")
    end
    signal_map = prepare_tsz_map_for_output(cfg, state.m_hp)
    signal_map === state.m_hp || print_map_summary("Output full-sky tSZ signal map after Gaussian beam", signal_map)
    return signal_map
end

function masked_signal_map(signal_map, mask_map)
    output_map = duplicate_healpix_map(signal_map)
    output_map.pixels .*= mask_map.pixels
    return output_map
end

function masked_signal_plus_noise_map(signal_map, noise_map, mask_map)
    output_map = HealpixMap{Float64, RingOrder}(signal_map.resolution.nside)
    length(signal_map.pixels) == length(noise_map.pixels) == length(mask_map.pixels) ||
        error("Signal, noise, and mask maps must have the same number of pixels.")

    @inbounds @simd for i in eachindex(output_map.pixels)
        output_map.pixels[i] = (signal_map.pixels[i] + noise_map.pixels[i]) * mask_map.pixels[i]
    end
    return output_map
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

    open(abs_path, "w") do io
        write(io, UInt8(0x93))
        write(io, codeunits("NUMPY"))
        write(io, UInt8(0x01))
        write(io, UInt8(0x00))
        write(io, UInt8(padded_header_nbytes & 0xff))
        write(io, UInt8((padded_header_nbytes >> 8) & 0xff))
        write(io, codeunits(padded_header))
        write(io, data)
    end

    println("Saved $(label) NumPy array to $(abs_path)")
    return abs_path
end

function save_map_overwrite(path::AbstractString, map, label::AbstractString)
    abs_path = abspath(path)
    parent_dir = dirname(abs_path)
    isdir(parent_dir) || mkpath(parent_dir)
    Healpix.saveToFITS(map, "!" * abs_path, typechar="D")
    println("Saved $(label) to $(abs_path)")
    return abs_path
end

function save_cl_values(cfg::HalfDomeFullSkySONoiseConfig, cl_values, label::AbstractString)
    npy_path = so_output_path(cfg, "$(label)_cl", ".npy")
    write_npy_float64_vector(npy_path, cl_values, "$(label) C_l")
    return cl_values
end

function compute_and_save_cl(cfg::HalfDomeFullSkySONoiseConfig, map, label::AbstractString)
    cl = compute_cl(cfg.base_cfg, map)
    save_cl_values(cfg, cl, label)
    return cl
end

function compute_cross_cl(cfg::VisualConfig, map1, map2)
    if cfg.cl_lmax < 0
        println("Computing cross C_l with Healpix default lmax=$(healpix_default_lmax(cfg.nside)), niter=$(cfg.cl_niter).")
        println("This mode is memory-heavy at high NSIDE; set cl_lmax to cap memory if needed.")
        return Healpix.anafast(map1, map2; niter=cfg.cl_niter)
    end

    println("Computing cross C_l with lmax=$(cfg.cl_lmax), niter=$(cfg.cl_niter).")
    return Healpix.anafast(map1, map2; lmax=cfg.cl_lmax, niter=cfg.cl_niter)
end

function compute_and_save_cross_cl(
    cfg::HalfDomeFullSkySONoiseConfig,
    map1,
    map2,
    label::AbstractString
)
    cl = compute_cross_cl(cfg.base_cfg, map1, map2)
    save_cl_values(cfg, cl, label)
    return cl
end

function noise_split_seed(cfg::HalfDomeFullSkySONoiseConfig, case_label::AbstractString, split_index::Integer)
    case_offset = case_label == "baseline" ? 100 : case_label == "goal" ? 200 : 300
    return cfg.seed + 10_000 * (cfg.noise_deprojection + 1) + case_offset + Int(split_index)
end

function process_noise_case_cross!(
    cfg::HalfDomeFullSkySONoiseConfig,
    signal_map,
    mask_map,
    case_label::AbstractString,
    noise_path::AbstractString
)
    println("Generating $(case_label) Gaussian noise split maps for Deproj-$(cfg.noise_deprojection).")
    noise_ell, noise_cl_native = read_so_noise_native_cl(
        noise_path,
        cfg.noise_lmax;
        column=cfg.noise_column,
        input_is_dl=cfg.noise_is_dl
    )
    println(
        "Read $(length(noise_ell)) $(case_label) SO noise multipoles " *
        "from ell=$(first(noise_ell)) to ell=$(last(noise_ell)); " *
        "padding only the internal synalm vector outside that native range with zero."
    )
    noise_cl = so_noise_cl_vector_for_synalm(noise_ell, noise_cl_native, cfg.noise_lmax)
    noise_map_1 = generate_gaussian_noise_map(
        noise_cl,
        cfg.base_cfg.nside,
        cfg.noise_lmax,
        MersenneTwister(noise_split_seed(cfg, case_label, 1))
    )
    noise_map_2 = generate_gaussian_noise_map(
        noise_cl,
        cfg.base_cfg.nside,
        cfg.noise_lmax,
        MersenneTwister(noise_split_seed(cfg, case_label, 2))
    )

    if cfg.save_noise_maps
        save_map_overwrite(
            so_output_path(cfg, "$(case_label)_raw_noise_split1_map", ".fits"),
            noise_map_1,
            "$(case_label) raw noise split 1 map"
        )
        save_map_overwrite(
            so_output_path(cfg, "$(case_label)_raw_noise_split2_map", ".fits"),
            noise_map_2,
            "$(case_label) raw noise split 2 map"
        )
    end

    masked_noisy_map_1 = masked_signal_plus_noise_map(signal_map, noise_map_1, mask_map)
    masked_noisy_map_2 = masked_signal_plus_noise_map(signal_map, noise_map_2, mask_map)

    if cfg.save_noisy_maps
        save_map_overwrite(
            so_output_path(cfg, "masked_$(case_label)_noisy_split1_map", ".fits"),
            masked_noisy_map_1,
            "$(case_label) masked signal plus noise split 1 map"
        )
        save_map_overwrite(
            so_output_path(cfg, "masked_$(case_label)_noisy_split2_map", ".fits"),
            masked_noisy_map_2,
            "$(case_label) masked signal plus noise split 2 map"
        )
    end

    cl = compute_and_save_cross_cl(
        cfg,
        masked_noisy_map_1,
        masked_noisy_map_2,
        "masked_$(case_label)_noise_cross"
    )

    noise_map_1 = nothing
    noise_map_2 = nothing
    masked_noisy_map_1 = nothing
    masked_noisy_map_2 = nothing
    GC.gc()
    return cl
end

function run_halfdome_fullsky_so_noise()
    t0 = time()
    cfg = try
        load_halfdome_fullsky_so_noise_config()
    catch err
        if err isa SkipVisualRun
            println("Skipping HalfDome full-sky SO-noise run: $(err.message)")
            return nothing
        end
        rethrow()
    end

    ensure_output_dir(cfg.base_cfg)
    print_halfdome_fullsky_so_noise_config(cfg)
    safe_print_runtime_environment()

    signal_map = paint_halfdome_fullsky_signal_map(cfg.base_cfg)

    println("Generating random apodized fsky=$(cfg.fsky) mask.")
    mask_info = random_apodized_cap_mask(
        cfg.base_cfg.nside,
        cfg.fsky,
        cfg.mask_apodization_arcmin,
        MersenneTwister(cfg.seed)
    )
    mask_map = mask_info.mask
    println("Mask support fsky after pixelization: $(mask_info.support_fsky)")
    println("Mean mask weight: $(mask_info.mean_weight)")
    println("Mean mask weight squared: $(mask_info.mean_weight2)")

    active_cfgs = [with_noise_deprojection(cfg, deprojection) for deprojection in cfg.noise_deprojections]

    for active_cfg in active_cfgs
        if active_cfg.save_signal_map
            save_map_overwrite(
                so_output_path(active_cfg, "clean_signal_map", ".fits"),
                signal_map,
                "clean full-sky tSZ signal map"
            )
        end

        if active_cfg.save_mask_map
            save_map_overwrite(
                so_output_path(active_cfg, "apodized_mask_map", ".fits"),
                mask_map,
                "apodized mask map"
            )
        end
    end

    cl_unmasked_no_noise = nothing
    if cfg.save_unmasked_no_noise_cl
        println("Computing unmasked no-noise full-sky C_l once.")
        cl_unmasked_no_noise = compute_cl(cfg.base_cfg, signal_map)
        for active_cfg in active_cfgs
            save_cl_values(active_cfg, cl_unmasked_no_noise, "unmasked_no_noise")
        end
    end

    cl_no_noise = nothing
    clean_masked_map = nothing
    if cfg.save_no_noise_cl || cfg.save_masked_signal_map
        println("Preparing masked no-noise map.")
        clean_masked_map = masked_signal_map(signal_map, mask_map)

        if cfg.save_masked_signal_map
            for active_cfg in active_cfgs
                save_map_overwrite(
                    so_output_path(active_cfg, "masked_clean_signal_map", ".fits"),
                    clean_masked_map,
                    "masked clean full-sky tSZ signal map"
                )
            end
        end

        if cfg.save_no_noise_cl
            println("Computing masked no-noise C_l once.")
            cl_no_noise = compute_cl(cfg.base_cfg, clean_masked_map)
            for active_cfg in active_cfgs
                save_cl_values(active_cfg, cl_no_noise, "masked_no_noise")
            end
        end
    end

    cl_baseline = Dict{Int, Any}()
    cl_goal = Dict{Int, Any}()
    for active_cfg in active_cfgs
        println("Starting SO noise cross-spectrum outputs for Deproj-$(active_cfg.noise_deprojection).")
        if active_cfg.save_baseline_noise_cross_cl
            cl_baseline[active_cfg.noise_deprojection] = process_noise_case_cross!(
                active_cfg,
                signal_map,
                mask_map,
                "baseline",
                active_cfg.baseline_noise_path
            )
        end
        if active_cfg.save_goal_noise_cross_cl
            cl_goal[active_cfg.noise_deprojection] = process_noise_case_cross!(
                active_cfg,
                signal_map,
                mask_map,
                "goal",
                active_cfg.goal_noise_path
            )
        end
    end

    clean_masked_map = nothing
    GC.gc()

    elapsed = time() - t0
    println("Finished full-sky HalfDome SO-noise run in $(round(elapsed; digits=2)) s.")
    return (
        cfg=cfg,
        mask_info=mask_info,
        cl_unmasked_no_noise=cl_unmasked_no_noise,
        cl_no_noise=cl_no_noise,
        cl_baseline=cl_baseline,
        cl_goal=cl_goal
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_halfdome_fullsky_so_noise()
end
