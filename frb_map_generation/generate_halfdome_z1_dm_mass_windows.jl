#!/usr/bin/env julia

# One-pass fixed-source-redshift HalfDome halo-DM mass-window generator.
#
# Every requested mass window is evaluated on the same random HEALPix
# sightlines.  The HalfDome HDF5 lightcone is streamed exactly once, and each
# halo/FRB profile value is evaluated once before it is routed to all matching
# windows.  The output is an analysis-ready HDF5 file plus summary/provenance
# sidecars.

using Dates
using Random
using SHA
using Statistics
using UUIDs

function canonical_option_name(name::AbstractString)
    return replace(lowercase(strip(String(name))), "-" => "_")
end

function parse_cli(args)
    options = Dict{String, String}()
    positionals = String[]
    i = 1
    while i <= length(args)
        arg = String(args[i])
        if startswith(arg, "--")
            body = arg[3:end]
            isempty(body) && error("Empty -- option.")
            if occursin('=', body)
                key, value = split(body, "="; limit=2)
                options[canonical_option_name(key)] = String(value)
            elseif i < length(args) && !startswith(String(args[i + 1]), "--") &&
                   !occursin('=', String(args[i + 1]))
                options[canonical_option_name(body)] = String(args[i + 1])
                i += 1
            else
                options[canonical_option_name(body)] = "true"
            end
        elseif occursin('=', arg)
            key, value = split(arg, "="; limit=2)
            options[canonical_option_name(key)] = String(value)
        else
            push!(positionals, arg)
        end
        i += 1
    end
    isempty(positionals) || error("Unexpected positional arguments: $(join(positionals, ", ")). Use --key=value.")
    return options
end

function parse_bool(value)
    text = lowercase(strip(String(value)))
    text in ("1", "true", "t", "yes", "y", "on") && return true
    text in ("0", "false", "f", "no", "n", "off") && return false
    error("Could not parse boolean value $(repr(value)).")
end

const CLI_OPTIONS = parse_cli(ARGS)
const HELP_MODE = get(CLI_OPTIONS, "help", "false") |> parse_bool
const SELF_TEST_MODE = get(CLI_OPTIONS, "self_test", "false") |> parse_bool
const DRY_RUN_MODE = get(CLI_OPTIONS, "dry_run", "false") |> parse_bool
const EARLY_MODE = HELP_MODE || SELF_TEST_MODE || DRY_RUN_MODE

# Keep --help, --dry-run, and --self-test usable on login nodes that do not
# have the production Julia depot mounted.
if !EARLY_MODE
    using HDF5
    using Healpix
    using Interpolations
    using XGPaint

    if isdefined(XGPaint, :HaloDMProfile)
        const HaloDMProfile = getfield(XGPaint, :HaloDMProfile)
        const HALO_DM_PROFILE_SOURCE = "XGPaint.HaloDMProfile"
        const GENERATED_DM_MODEL_FAMILY = "xgpaint_native_halo_dm_profile"
    else
        # Official XGPaint v0.4.0 provides BattagliaTauProfile but not the
        # small FRB wrapper used by the original local checkout.  The cached
        # interpolator contains the production DM grid; this wrapper also
        # supplies the exact observed tau -> pc cm^-3 unit/redshift conversion
        # for an explicitly requested public-v0.4 rebuild. Its profile family
        # is distinct from the authoritative legacy private-XGPaint DM cache.
        const THOMSON_CROSS_SECTION_CM2 = 6.6524587321e-25
        const PARSEC_CM = 3.0856775814913673e18
        const HALO_DM_PROFILE_SOURCE = "local wrapper around XGPaint.BattagliaTauProfile"
        const GENERATED_DM_MODEL_FAMILY = "public_v0.4_tau_conversion"

        struct HaloDMProfile{T,P,C} <: XGPaint.AbstractGNFW{T}
            tau_model::P
            cosmo::C
            tau_per_pc_cm3::T
        end

        function HaloDMProfile(
            tau_model::P,
        ) where {T,P<:XGPaint.AbstractBattagliaTauProfile{T}}
            conversion = T(THOMSON_CROSS_SECTION_CM2 * PARSEC_CM)
            return HaloDMProfile{T,P,typeof(tau_model.cosmo)}(
                tau_model, tau_model.cosmo, conversion
            )
        end

        @inline function (model::HaloDMProfile)(radius, mass_msun, redshift)
            return model.tau_model(radius, mass_msun, redshift) /
                   (model.tau_per_pc_cm3 * (one(redshift) + redshift))
        end
    end
end

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const OMEGAM = OMEGAB + OMEGAC
const DEFAULT_SOURCE_REDSHIFT = 1.0
const DEFAULT_HALFDOME_MASS_FLOOR_MSUN = 7.327e12

struct MassWindow
    label::String
    requested_min_msun::Float64
    requested_max_msun::Float64
    effective_min_msun::Float64
    effective_max_msun::Float64
end

window_is_empty(window::MassWindow) = window.effective_min_msun >= window.effective_max_msun

function mass_in_window(mass_msun::Real, window::MassWindow)
    window_is_empty(window) && return false
    mass = Float64(mass_msun)
    return isfinite(mass) && mass >= window.effective_min_msun && mass < window.effective_max_msun
end

function default_mass_windows(; catalog_mass_floor_msun=DEFAULT_HALFDOME_MASS_FLOOR_MSUN, apply_catalog_mass_floor=true)
    requested = [
        ("all", 0.0, Inf),
        ("m1e10_to_1e12", 1.0e10, 1.0e12),
        ("m1e10_to_1e13", 1.0e10, 1.0e13),
        ("m1e10_to_1e14", 1.0e10, 1.0e14),
        ("m1e10_to_1e15", 1.0e10, 1.0e15),
        ("m1e11_to_1e14", 1.0e11, 1.0e14),
        ("m1e12_to_1e14", 1.0e12, 1.0e14),
        ("m1e13_to_1e14", 1.0e13, 1.0e14),
        ("m1e9_to_1e13", 1.0e9, 1.0e13),
        ("m1e9_to_1e14", 1.0e9, 1.0e14),
    ]
    floor_msun = apply_catalog_mass_floor ? Float64(catalog_mass_floor_msun) : 0.0
    return [
        MassWindow(label, requested_min, requested_max, max(requested_min, floor_msun), requested_max)
        for (label, requested_min, requested_max) in requested
    ]
end

function parse_mass_bound(value::AbstractString)
    text = lowercase(strip(String(value)))
    text in ("inf", "+inf", "infinity", "+infinity") && return Inf
    bound = parse(Float64, text)
    isfinite(bound) || error("Mass-window bounds must be finite or Inf; received $(repr(value)).")
    return bound
end

"""Parse `label:min_msun:max_msun` entries separated by commas or semicolons."""
function parse_mass_windows(
    specification::AbstractString;
    catalog_mass_floor_msun=DEFAULT_HALFDOME_MASS_FLOOR_MSUN,
    apply_catalog_mass_floor=true,
)
    normalized = replace(strip(String(specification)), ';' => ',')
    isempty(normalized) && return default_mass_windows(
        catalog_mass_floor_msun=catalog_mass_floor_msun,
        apply_catalog_mass_floor=apply_catalog_mass_floor,
    )
    floor_msun = apply_catalog_mass_floor ? Float64(catalog_mass_floor_msun) : 0.0
    windows = MassWindow[]
    labels = Set{String}()
    for entry in split(normalized, ','; keepempty=false)
        fields = strip.(split(entry, ':'; keepempty=true))
        length(fields) == 3 || error(
            "Invalid mass-window entry $(repr(entry)); expected label:min_msun:max_msun.",
        )
        label, lower_text, upper_text = fields
        isempty(label) && error("Mass-window labels cannot be empty.")
        label in labels && error("Duplicate mass-window label: $(label)")
        lower = parse_mass_bound(lower_text)
        upper = parse_mass_bound(upper_text)
        isfinite(lower) && lower >= 0.0 || error("Mass-window lower bound must be finite and nonnegative: $(entry)")
        upper > lower || error("Mass-window upper bound must exceed its lower bound: $(entry)")
        push!(labels, label)
        push!(windows, MassWindow(label, lower, upper, max(lower, floor_msun), upper))
    end
    isempty(windows) && error("At least one mass window is required.")
    length(windows) <= 16 || error("At most 16 mass windows are supported in one pass.")
    return windows
end

function window_membership_mask(mass_msun::Real, windows)
    length(windows) <= 8 * sizeof(UInt16) || error("UInt16 membership masks support at most 16 windows.")
    mask = zero(UInt16)
    @inbounds for iw in eachindex(windows)
        mass_in_window(mass_msun, windows[iw]) && (mask |= UInt16(1) << (iw - 1))
    end
    return mask
end

function option_value(options, names, default)
    for name in names
        key = canonical_option_name(name)
        haskey(options, key) && return options[key]
    end
    return string(default)
end

get_string_option(options, names, default) = String(option_value(options, names, default))
get_int_option(options, names, default) = parse(Int, option_value(options, names, default))
get_float_option(options, names, default) = parse(Float64, option_value(options, names, default))
get_bool_option(options, names, default) = parse_bool(option_value(options, names, default))

function validate_known_options(options)
    known = Set([
        "help", "self_test", "dry_run", "catalog", "halfdome_path", "output", "output_path",
        "summary", "provenance", "source_redshift", "z_source", "nside", "nfrb", "n", "seed",
        "frb_seed", "unique_pixels", "chunk_size", "chunkn", "max_catalog_halos", "catalog_mass_floor",
        "sightline_mode", "sightline_catalog", "sightline_ra_column", "sightline_dec_column",
        "sightline_redshift_column", "sightline_redshift_width", "sightline_max_rows",
        "sightline_progress_every_rows",
        "apply_catalog_mass_floor", "catalog_masses_are_msun_h", "dm_cache", "dm_cache_file",
        "dm_cache_overwrite", "dm_cleanup_nonpositive", "dm_value_sanity_max",
        "dm_aperture_r200_multiplier", "pdf_bins",
        "pdf_edge_count", "pdf_spacing", "pdf_dm_min", "pdf_dm_max", "progress_every_batches", "overwrite",
        "mass_windows", "save_ray_dm",
    ])
    unknown = sort!(collect(setdiff(Set(keys(options)), known)))
    isempty(unknown) || error("Unknown option(s): $(join(unknown, ", ")). Run with --help.")
end

project_root() = dirname(@__DIR__)

function resolve_project_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? normpath(String(path)) : normpath(joinpath(project_root(), path))
end

function resolve_halfdome_catalog(path::AbstractString; require_exists=true)
    resolved = resolve_project_path(path)
    if isdir(resolved)
        candidates = filter(readdir(resolved; join=true)) do entry
            isfile(entry) && lowercase(splitext(entry)[2]) in (".h5", ".hdf5")
        end
        for preferred in ("lightcone_100.hdf5", "lightcone_100.h5", "halos.hdf5", "halos.h5")
            matches = filter(candidate -> lowercase(basename(candidate)) == preferred, candidates)
            length(matches) == 1 && return only(matches)
        end
        length(candidates) == 1 && return only(candidates)
        isempty(candidates) && error("Catalog directory $(resolved) contains no HDF5 files.")
        error("Catalog directory $(resolved) contains multiple HDF5 files; pass the exact file.")
    end
    require_exists && !isfile(resolved) && error("HalfDome catalog not found: $(resolved)")
    return resolved
end

function split_output_sidecars(output_path, summary_option, provenance_option)
    stem, _ = splitext(output_path)
    summary_path = isempty(summary_option) ? stem * "_summary.csv" : resolve_project_path(summary_option)
    provenance_path = isempty(provenance_option) ? stem * "_provenance.txt" : resolve_project_path(provenance_option)
    return summary_path, provenance_path
end

function show_help()
    println("""
One-pass HalfDome fixed-z foreground-halo DM mass-window generator

Usage:
  julia +1.8.5 --project=julia_env frb_map_generation/generate_halfdome_z1_dm_mass_windows.jl [options]

Core options (both --key=value and key=value are accepted):
  --catalog=lightcone_100.hdf5       HalfDome HDF5 lightcone (or containing directory)
  --output=batched_data/frb_z1_mass_windows/halfdome_z1_dm_mass_windows.h5
  --source-redshift=1.0              Common source redshift; foreground uses 0 <= z_halo <= z_source
  --nfrb=120000 --seed=42            Shared random HEALPix sightlines; matches the reference ray count
  --nside=4096 --unique-pixels=true
  --chunk-size=1000000               Catalog rows per HDF5 read
  --dm-cache=frb_map_generation/outputs/shared_xgpaint_dm_cache.jld2
  --dm-cache-overwrite=false          Existing DM cache is authoritative; a missing cache is an error
                                      Public-v0.4 rebuilds require a NEW, non-existing --dm-cache path
                                      Example: --dm-cache=.../public_v04_dm_cache.jld2 --dm-cache-overwrite=true
  --dm-aperture-r200-multiplier=1.0  Truncate at R200, matching DMhalo_r200 reference names
  --mass-windows=SPEC                 Comma/semicolon-separated label:min_msun:max_msun entries
                                       Example: m1e10_to_1e13:1e10:1e13,m1e10_to_1e16:1e10:1e16

Sightline selection (uniform is the default):
  --sightline-mode=uniform            Draw unique random HEALPix pixel centers
  --sightline-mode=catalog            Stream and reservoir-sample a catalogue CSV
  --sightline-catalog=PATH            Also selects catalog mode when mode is omitted
  --sightline-ra-column=ra_rad --sightline-dec-column=dec_rad
  --sightline-redshift-column=host_redshift --sightline-redshift-width=0.05
                                       Keep host_redshift in z_source +/- width
  --sightline-max-rows=0              0 scans the full CSV; positive values bound tests
  --sightline-progress-every-rows=1000000

HalfDome resolution options:
  --catalog-mass-floor=7.327e12      Observed physical-Msun floor after the catalog's /h conversion
  --apply-catalog-mass-floor=true    Enforce and record effective lower bounds
  --catalog-masses-are-msun-h=true   Divide halo_mass_m200c by h=0.68, matching existing scripts

Output/control options:
  --summary=PATH --provenance=PATH   Defaults are sidecars beside the HDF5 output
  --save-ray-dm=false                 Histogram-only HDF5: omit per-ray DM, pixels, and coordinates
  --pdf-edge-count=300 --pdf-spacing=log --pdf-dm-min=0.1 --pdf-dm-max=30000
                                      Exact target-notebook histogram convention (299 intervals)
  --max-catalog-halos=0              0 means the complete catalog; positive values enable test subsets
  --progress-every-batches=10 --overwrite=false
  --dry-run                          Resolve configuration and print effective windows only
  --self-test                        Dependency-free routing/histogram/sidecar smoke test

Production example:
  JULIA_NUM_THREADS=16 julia +1.8.5 --project=julia_env \\
    frb_map_generation/generate_halfdome_z1_dm_mass_windows.jl \\
    --catalog=lightcone_100.hdf5 --source-redshift=1 --nfrb=120000 \\
    --seed=42 --nside=4096 \\
    --output=batched_data/frb_z1_mass_windows/halfdome_z1_dm_mass_windows.h5

Catalogue-sightline example (the CSV is streamed once, not loaded in memory):
  JULIA_NUM_THREADS=16 julia +1.8.5 --project=julia_env \\
    frb_map_generation/generate_halfdome_z1_dm_mass_windows.jl \\
    --sightline-catalog=path/to/COSMOS2020_hosts.csv \\
    --sightline-redshift-width=0.05 --nfrb=120000 \\
    --output=batched_data/frb_z1_mass_windows/cosmos2020_z1_dm_mass_windows.h5

Python/h5py extraction (multidimensional datasets are Python-oriented on disk):
  labels = [x.decode() if isinstance(x, bytes) else str(x) for x in f["window_label"][:]]
  # dm_pc_cm3 exists only when --save-ray-dm=true
  edges = f["pdf_bin_edges_pc_cm3"][:]    # 300 common log edges
  counts = f["pdf_count"][:]               # shape (N_windows, 299)
  pdf = f["pdf_density_per_pc_cm3"][:]    # shape (N_windows, 299)

Without --mass-windows, the legacy notebook windows are used. Custom windows are
lower-inclusive/upper-exclusive and expressed in physical Msun.
""")
end

function format_bound(value::Real)
    isfinite(value) || return "Inf"
    return string(Float64(value))
end

function print_window_table(windows)
    println("Mass windows (physical Msun, lower-inclusive/upper-exclusive):")
    println("  label                     requested                  effective                  status")
    for window in windows
        requested = "[$(format_bound(window.requested_min_msun)), $(format_bound(window.requested_max_msun)))"
        effective = "[$(format_bound(window.effective_min_msun)), $(format_bound(window.effective_max_msun)))"
        status = window_is_empty(window) ? "EMPTY at HalfDome resolution" : "active"
        println("  ", rpad(window.label, 25), rpad(requested, 27), rpad(effective, 27), status)
    end
end

function equivalent_effective_window_groups(windows)
    groups = Dict{Tuple{Float64, Float64}, Vector{String}}()
    for window in windows
        key = (window.effective_min_msun, window.effective_max_msun)
        push!(get!(groups, key, String[]), window.label)
    end
    return filter(group -> length(last(group)) > 1, collect(groups))
end

function pixel_centers_to_ra_dec(res, pixels)
    ras = Vector{Float64}(undef, length(pixels))
    decs = Vector{Float64}(undef, length(pixels))
    Threads.@threads :static for i in eachindex(pixels)
        vx, vy, vz = Healpix.pix2vecRing(res, pixels[i])
        theta, phi = Healpix.vec2ang(vx, vy, vz)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end
    return ras, decs
end

function ra_dec_to_unit_vectors(ras, decs)
    ux = Vector{Float64}(undef, length(ras))
    uy = similar(ux)
    uz = similar(ux)
    Threads.@threads :static for i in eachindex(ras)
        cosdec = cos(Float64(decs[i]))
        ux[i] = cosdec * cos(Float64(ras[i]))
        uy[i] = cosdec * sin(Float64(ras[i]))
        uz[i] = sin(Float64(decs[i]))
    end
    return ux, uy, uz
end

function draw_random_frb_pixels(rng, npix::Int, frb_count::Int; unique_pixels=true)
    frb_count > 0 || error("nfrb must be positive.")
    if !unique_pixels
        return rand(rng, 1:npix, frb_count)
    end
    frb_count <= npix || error("Cannot draw $(frb_count) unique pixels from npix=$(npix).")
    pixels = Vector{Int}(undef, frb_count)
    seen = Set{Int}()
    i = 1
    while i <= frb_count
        pixel = rand(rng, 1:npix)
        pixel in seen && continue
        push!(seen, pixel)
        pixels[i] = pixel
        i += 1
    end
    return pixels
end

function ra_dec_to_ring_pixels(res, ras, decs)
    length(ras) == length(decs) || error("RA and Dec lengths differ.")
    pixels = Vector{Int}(undef, length(ras))
    @inbounds for i in eachindex(ras)
        pixels[i] = Healpix.ang2pixRing(res, pi / 2 - Float64(decs[i]), Float64(ras[i]))
    end
    return pixels
end

normalize_csv_header(value) = lowercase(strip(replace(String(value), "\"" => "")))

function parse_numeric_csv_field(value)
    cleaned = strip(replace(String(value), "\"" => ""))
    return tryparse(Float64, cleaned)
end

function select_catalog_sightlines(
    path::AbstractString,
    sample_limit::Int,
    seed::Int;
    ra_column="ra_rad",
    dec_column="dec_rad",
    redshift_column="host_redshift",
    target_redshift=DEFAULT_SOURCE_REDSHIFT,
    redshift_width=0.05,
    max_rows=0,
    progress_every_rows=1_000_000,
)
    isfile(path) || error("Sightline catalogue not found: $(path)")
    sample_limit > 0 || error("Catalogue sightline sample limit must be positive.")
    redshift_width >= 0.0 || error("sightline_redshift_width must be nonnegative.")
    max_rows >= 0 || error("sightline_max_rows must be nonnegative.")
    progress_every_rows >= 0 || error("sightline_progress_every_rows must be nonnegative.")

    rng = MersenneTwister(seed)
    selected_ra = Float64[]
    selected_dec = Float64[]
    selected_redshift = Float64[]
    selected_rows = Int64[]
    sizehint!(selected_ra, sample_limit)
    sizehint!(selected_dec, sample_limit)
    sizehint!(selected_redshift, sample_limit)
    sizehint!(selected_rows, sample_limit)
    rows_scanned = Int64(0)
    eligible_rows = Int64(0)
    malformed_rows = Int64(0)

    open(path, "r") do io
        eof(io) && error("Sightline catalogue is empty: $(path)")
        headers = normalize_csv_header.(split(readline(io), ','; keepempty=true))
        requested_headers = normalize_csv_header.([ra_column, dec_column, redshift_column])
        indices = map(requested_headers) do requested
            index = findfirst(==(requested), headers)
            index === nothing && error(
                "Sightline catalogue lacks column $(requested). Available columns: $(join(headers, ", ")).",
            )
            return index
        end
        ra_index, dec_index, redshift_index = indices
        required_field_count = max(ra_index, dec_index, redshift_index)

        for line in eachline(io)
            max_rows > 0 && rows_scanned >= max_rows && break
            rows_scanned += 1
            if progress_every_rows > 0 && rows_scanned % progress_every_rows == 0
                println("  sightline CSV rows=$(rows_scanned), eligible=$(eligible_rows), reservoir=$(length(selected_ra))")
            end
            fields = split(line, ','; keepempty=true)
            if length(fields) < required_field_count
                malformed_rows += 1
                continue
            end
            ra = parse_numeric_csv_field(fields[ra_index])
            dec = parse_numeric_csv_field(fields[dec_index])
            host_redshift = parse_numeric_csv_field(fields[redshift_index])
            if ra === nothing || dec === nothing || host_redshift === nothing ||
               !isfinite(ra) || !isfinite(dec) || !isfinite(host_redshift) ||
               dec < -pi / 2 || dec > pi / 2
                malformed_rows += 1
                continue
            end
            abs(host_redshift - target_redshift) <= redshift_width || continue
            eligible_rows += 1
            normalized_ra = mod(ra, 2pi)
            if length(selected_ra) < sample_limit
                push!(selected_ra, normalized_ra)
                push!(selected_dec, dec)
                push!(selected_redshift, host_redshift)
                push!(selected_rows, rows_scanned)
            else
                replacement_index = rand(rng, 1:eligible_rows)
                if replacement_index <= sample_limit
                    selected_ra[replacement_index] = normalized_ra
                    selected_dec[replacement_index] = dec
                    selected_redshift[replacement_index] = host_redshift
                    selected_rows[replacement_index] = rows_scanned
                end
            end
        end
    end

    isempty(selected_ra) && error(
        "No finite catalogue sightlines matched host redshift $(target_redshift) +/- $(redshift_width).",
    )
    return (;
        ras=selected_ra,
        decs=selected_dec,
        host_redshifts=selected_redshift,
        catalog_rows=selected_rows,
        rows_scanned,
        eligible_rows,
        malformed_rows,
    )
end

function build_frb_pixel_lookup(frb_pixels)
    order = sortperm(frb_pixels)
    return Int.(frb_pixels[order]), Int.(order)
end

thread_capacity() = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()

function compute_theta_min_local(model)
    if hasproperty(model, :itp)
        interpolation = getproperty(model, :itp)
        if hasproperty(interpolation, :ranges)
            return exp(Float64(first(first(getproperty(interpolation, :ranges)))))
        end
    end
    return eps(Float64)
end

function compute_theta_max_local(model, mass, redshift, aperture_r200_multiplier::Float64)
    function_name = Symbol("compute_", Char(0x03b8), "max")
    isdefined(XGPaint, function_name) || error("XGPaint does not define compute_theta_max.")
    return getfield(XGPaint, function_name)(model, mass, redshift; mult=aperture_r200_multiplier)
end

function cache_grid_value(model_grid, candidate_keys)
    for key in candidate_keys
        haskey(model_grid, key) && return model_grid[key], key
        symbol_key = Symbol(key)
        haskey(model_grid, symbol_key) && return model_grid[symbol_key], string(symbol_key)
    end
    error("Interpolator cache is missing all of $(candidate_keys); found keys $(collect(keys(model_grid))).")
end

"""Load both the private-fork ASCII cache and official v0.4 Unicode cache."""
function enforce_generated_cache_target_policy(cache_file, overwrite, generated_model_family)
    if overwrite && generated_model_family == "public_v0.4_tau_conversion" && isfile(cache_file)
        error(
            "Refusing to overwrite existing DM cache $(cache_file) with the distinct " *
            "public-v0.4 model family. Preserve the authoritative cache and choose a NEW, " *
            "non-existing path, for example --dm-cache=.../public_v04_dm_cache.jld2 " *
            "--dm-cache-overwrite=true.",
        )
    end
    return nothing
end

function build_dm_interpolator_compatible(
    model;
    cache_file::String,
    overwrite::Bool,
    cleanup_nonpositive::Bool,
)
    enforce_generated_cache_target_policy(cache_file, overwrite, GENERATED_DM_MODEL_FAMILY)
    if !isfile(cache_file) && !overwrite
        error(
            "Authoritative DM interpolator cache not found: $(cache_file). " *
            "Refusing an implicit rebuild because the vendored public-v0.4 tau conversion is " *
            "not numerically identical to the legacy private-XGPaint DM cache. Supply the cache, " *
            "or explicitly opt into the distinct model using a NEW, non-existing cache path, " *
            "for example --dm-cache=.../public_v04_dm_cache.jld2 --dm-cache-overwrite=true.",
        )
    end
    if overwrite
        profile = build_interpolator(model; cache_file=cache_file, overwrite=overwrite)
        return (
            profile=profile,
            loader="XGPaint.build_interpolator",
            logtheta_key="generated_overwrite",
            nonpositive_replaced=0,
            model_family=GENERATED_DM_MODEL_FAMILY,
        )
    end

    isdefined(XGPaint, :load) || error(
        "XGPaint does not expose its FileIO loader; cannot read cache $(cache_file). " *
        "To build the public-v0.4 model, use a NEW, non-existing --dm-cache path together " *
        "with --dm-cache-overwrite=true.",
    )
    model_grid = getfield(XGPaint, :load)(cache_file)
    unicode_logtheta_key = string("prof_log", Char(0x03b8), "s")
    prof_logthetas, logtheta_key = cache_grid_value(
        model_grid, ("prof_logthetas", unicode_logtheta_key),
    )
    prof_redshift, _ = cache_grid_value(model_grid, ("prof_redshift",))
    prof_logMs, _ = cache_grid_value(model_grid, ("prof_logMs",))
    prof_y, _ = cache_grid_value(model_grid, ("prof_y",))

    all(isfinite, prof_y) || error("DM interpolator cache contains NaN or Inf values: $(cache_file)")
    nonpositive_count = count(value -> value <= zero(value), prof_y)
    if nonpositive_count > 0
        cleanup_nonpositive || error(
            "DM interpolator cache contains $(nonpositive_count) nonpositive values and " *
            "--dm-cleanup-nonpositive=false.",
        )
        positive_min = minimum(value for value in prof_y if value > zero(value))
        floor_value = positive_min * convert(eltype(prof_y), 1e-6)
        @inbounds for index in eachindex(prof_y)
            prof_y[index] <= zero(eltype(prof_y)) && (prof_y[index] = floor_value)
        end
    end

    interpolation = Interpolations.interpolate(
        log.(prof_y),
        Interpolations.BSpline(
            Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())),
        ),
    )
    scaled = Interpolations.scale(interpolation, prof_logthetas, prof_redshift, prof_logMs)
    profile_type = getfield(XGPaint, :LogInterpolatorProfile)
    return (
        profile=profile_type(model, scaled),
        loader="generator ASCII/Unicode cache compatibility loader",
        logtheta_key=logtheta_key,
        nonpositive_replaced=nonpositive_count,
        model_family=logtheta_key == "prof_logthetas" ?
            "legacy_private_xgpaint_precomputed_dm_cache" :
            "precomputed_dm_cache_model_family_unencoded",
    )
end

function validate_dm_interpolator_scale(interpolated_model, direct_model)
    interpolation = getproperty(interpolated_model, :itp)
    ranges = getproperty(interpolation, :ranges)
    logtheta = clamp(log(1e-4), Float64(first(ranges[1])), Float64(last(ranges[1])))
    theta = exp(logtheta)
    redshift = clamp(1.0, Float64(first(ranges[2])), Float64(last(ranges[2])))
    logmass = clamp(14.0, Float64(first(ranges[3])), Float64(last(ranges[3])))
    mass_msun = 10.0^logmass
    cached_value = Float64(interpolated_model(theta, mass_msun, redshift))
    direct_value = Float64(direct_model(theta, mass_msun, redshift))
    isfinite(cached_value) && cached_value > 0.0 || error("DM cache scale check returned $(cached_value).")
    isfinite(direct_value) && direct_value > 0.0 || error("Direct DM scale check returned $(direct_value).")
    ratio = cached_value / direct_value
    0.01 <= ratio <= 100.0 || error(
        "Cache is $(ratio)x the direct HaloDM profile at the scale-check point. " *
        "This is probably a tau/tSZ cache rather than a DM cache; pass a DM cache or rebuild explicitly.",
    )
    return ratio
end

struct ReusableDiscQueryWorkspace{R,I}
    res::R
    thread_buffers::Vector{Vector{Int}}
    thread_ringinfo::Vector{I}
end

function ReusableDiscQueryWorkspace(resolution)
    isdefined(XGPaint, :queryDiscRing!) || error("XGPaint provides no sparse HEALPix disc query.")
    ringinfo_type = if isdefined(Healpix, :RingInfo)
        getfield(Healpix, :RingInfo)
    elseif isdefined(XGPaint, :RingInfo)
        getfield(XGPaint, :RingInfo)
    else
        error("Neither Healpix nor XGPaint exposes RingInfo.")
    end
    capacity = thread_capacity()
    ringinfo = [ringinfo_type(0, 0, 0, 0.0, true) for _ in 1:capacity]
    return ReusableDiscQueryWorkspace(
        resolution, [Int[] for _ in 1:capacity], ringinfo,
    )
end

function make_sparse_disc_workspace(resolution)
    if isdefined(XGPaint, :HealpixRingProfileWorkspace)
        workspace_type = getfield(XGPaint, :HealpixRingProfileWorkspace)
        return workspace_type{Float64}(resolution), "XGPaint.HealpixRingProfileWorkspace"
    end
    return ReusableDiscQueryWorkspace(resolution),
           "thread-local XGPaint.queryDiscRing! compatibility workspace"
end

mutable struct WindowDMAccumulator
    thread_dm::Vector{Matrix{Float64}}
    thread_unique_hits::Vector{Int64}
    thread_window_hits::Vector{Vector{Int64}}
end

function WindowDMAccumulator(nfrb::Int, nwindow::Int)
    capacity = thread_capacity()
    return WindowDMAccumulator(
        [zeros(Float64, nfrb, nwindow) for _ in 1:capacity],
        zeros(Int64, capacity),
        [zeros(Int64, nwindow) for _ in 1:capacity],
    )
end

function add_if_frb_pixel_windows!(
    local_dm,
    local_window_hits,
    sorted_frb_pixels,
    sorted_frb_indices,
    global_pixel::Int,
    halo_ux::Float64,
    halo_uy::Float64,
    halo_uz::Float64,
    frb_ux,
    frb_uy,
    frb_uz,
    theta_min::Float64,
    theta_max::Float64,
    mass_msun::Float64,
    redshift::Float64,
    membership_mask::UInt16,
    nwindow::Int,
    dm_model_interp,
)
    frb_range = searchsorted(sorted_frb_pixels, global_pixel)
    isempty(frb_range) && return 0
    unique_hits = 0
    @inbounds for lookup_index in frb_range
        frb_index = sorted_frb_indices[lookup_index]
        cosine = clamp(
            halo_ux * frb_ux[frb_index] + halo_uy * frb_uy[frb_index] + halo_uz * frb_uz[frb_index],
            -1.0,
            1.0,
        )
        theta = acos(cosine)
        theta <= theta_max || continue
        contribution = Float64(dm_model_interp(max(theta, theta_min), mass_msun, redshift))
        isfinite(contribution) || error("Non-finite XGPaint DM at mass=$(mass_msun), z=$(redshift), theta=$(theta).")
        unique_hits += 1
        for iw in 1:nwindow
            if (membership_mask & (UInt16(1) << (iw - 1))) != 0
                local_dm[frb_index, iw] += contribution
                local_window_hits[iw] += 1
            end
        end
    end
    return unique_hits
end

function accumulate_batch_windows!(
    accumulator::WindowDMAccumulator,
    workspace,
    dm_model_interp,
    theta_min::Float64,
    aperture_r200_multiplier::Float64,
    candidate_pixel_margin::Float64,
    sorted_frb_pixels,
    sorted_frb_indices,
    frb_ux,
    frb_uy,
    frb_uz,
    x,
    y,
    z,
    masses,
    redshifts,
    membership_masks,
    nwindow::Int,
)
    Threads.@threads :static for i in eachindex(masses)
        tid = Threads.threadid()
        local_dm = accumulator.thread_dm[tid]
        local_window_hits = accumulator.thread_window_hits[tid]
        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        radius = sqrt(xi * xi + yi * yi + zi * zi)
        isfinite(radius) && radius > 0.0 || continue
        halo_ux = xi / radius
        halo_uy = yi / radius
        halo_uz = zi / radius
        center_theta, center_phi = Healpix.vec2ang(halo_ux, halo_uy, halo_uz)
        mass_msun = Float64(masses[i])
        redshift = Float64(redshifts[i])
        theta_max = Float64(compute_theta_max_local(
            dm_model_interp, mass_msun * XGPaint.M_sun, redshift, aperture_r200_multiplier,
        ))
        isfinite(theta_max) && theta_max > 0.0 || continue
        candidate_theta_max = min(pi, theta_max + candidate_pixel_margin)

        halo_hits = 0
        if workspace isa ReusableDiscQueryWorkspace
            candidate_pixels = getfield(XGPaint, :queryDiscRing!)(
                workspace.thread_buffers[tid], workspace.thread_ringinfo[tid], workspace.res,
                center_theta, center_phi, candidate_theta_max,
            )
            for global_pixel in candidate_pixels
                halo_hits += add_if_frb_pixel_windows!(
                    local_dm, local_window_hits, sorted_frb_pixels, sorted_frb_indices,
                    global_pixel, halo_ux, halo_uy, halo_uz,
                    frb_ux, frb_uy, frb_uz, theta_min, theta_max, mass_msun, redshift,
                    membership_masks[i], nwindow, dm_model_interp,
                )
            end
        else
            ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, candidate_theta_max)
            for ring_index in ring_start:ring_stop
                range1, range2 = XGPaint.get_ring_disc_ranges(
                    workspace, ring_index, center_theta, center_phi, candidate_theta_max,
                )
                first_pixel = workspace.ring_first_pixels[ring_index]
                for local_pixel_index in Iterators.flatten((range1, range2))
                    halo_hits += add_if_frb_pixel_windows!(
                        local_dm, local_window_hits, sorted_frb_pixels, sorted_frb_indices,
                        first_pixel + local_pixel_index - 1, halo_ux, halo_uy, halo_uz,
                        frb_ux, frb_uy, frb_uz, theta_min, theta_max, mass_msun, redshift,
                        membership_masks[i], nwindow, dm_model_interp,
                    )
                end
            end
        end
        accumulator.thread_unique_hits[tid] += halo_hits
    end
    return nothing
end

function reduce_accumulator(accumulator::WindowDMAccumulator)
    dm = zeros(Float64, size(first(accumulator.thread_dm)))
    window_hits = zeros(Int64, size(dm, 2))
    for local_dm in accumulator.thread_dm
        dm .+= local_dm
    end
    for local_hits in accumulator.thread_window_hits
        window_hits .+= local_hits
    end
    return dm, sum(accumulator.thread_unique_hits), window_hits
end

function stream_halfdome_batches(process_batch!::F, catalog_path::AbstractString, chunk_size::Int;
                                 masses_are_msun_h=true, max_catalog_halos=0) where {F}
    return h5open(catalog_path, "r") do h5
        for dataset in ("Position", "halo_mass_m200c", "redshift")
            haskey(h5, dataset) || error("HalfDome catalog is missing dataset $(dataset).")
        end
        position_dataset = h5["Position"]
        mass_dataset = h5["halo_mass_m200c"]
        redshift_dataset = h5["redshift"]
        size(position_dataset, 1) == 3 || error("Position must have shape (3, N).")
        total_halos = size(position_dataset, 2)
        length(mass_dataset) == total_halos || error("halo_mass_m200c length does not match Position.")
        length(redshift_dataset) == total_halos || error("redshift length does not match Position.")
        streamed_halos = max_catalog_halos > 0 ? min(total_halos, max_catalog_halos) : total_halos

        batch_number = 0
        for batch_start in 1:chunk_size:streamed_halos
            batch_number += 1
            batch_stop = min(batch_start + chunk_size - 1, streamed_halos)
            indices = batch_start:batch_stop
            positions = position_dataset[:, indices]
            masses = Float64.(mass_dataset[indices])
            masses_are_msun_h && (masses ./= H_VALUE)
            redshifts = Float64.(redshift_dataset[indices])
            process_batch!(
                batch_number,
                batch_start,
                @view(positions[1, :]),
                @view(positions[2, :]),
                @view(positions[3, :]),
                masses,
                redshifts,
            )
        end
        return (catalog_halos=total_halos, streamed_halos=streamed_halos)
    end
end

function histogram_columns(dm, edge_count::Int, lower::Float64, upper_option; spacing="log")
    finite_values = dm[isfinite.(dm)]
    isempty(finite_values) && error("No finite DM values available for histograms.")
    upper = upper_option === nothing ? maximum(finite_values) : Float64(upper_option)
    upper > lower || (upper = lower + max(abs(lower), 1.0))
    spacing_normalized = lowercase(strip(String(spacing)))
    if spacing_normalized == "log"
        lower > 0.0 || error("A logarithmic histogram requires pdf_dm_min > 0.")
        edges = 10.0 .^ collect(range(log10(lower), log10(upper); length=edge_count))
        centers = sqrt.(edges[1:end-1] .* edges[2:end])
    elseif spacing_normalized == "linear"
        edges = collect(range(lower, upper; length=edge_count))
        centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    else
        error("pdf_spacing must be log or linear.")
    end
    interval_count = edge_count - 1
    counts = zeros(Int64, interval_count, size(dm, 2))
    density = zeros(Float64, interval_count, size(dm, 2))
    for iw in axes(dm, 2)
        @inbounds for value in @view(dm[:, iw])
            isfinite(value) && value >= lower && value <= upper || continue
            ibin = searchsortedlast(edges, value)
            ibin == 0 && continue
            ibin >= length(edges) && (ibin = length(edges) - 1)
            counts[ibin, iw] += 1
        end
        total = sum(@view counts[:, iw])
        if total > 0
            @inbounds for ibin in 1:interval_count
                density[ibin, iw] = counts[ibin, iw] / (total * (edges[ibin + 1] - edges[ibin]))
            end
        end
    end
    return edges, centers, counts, density
end

function atomic_write(writer::F, target::AbstractString; overwrite=false) where {F}
    parent = dirname(target)
    isdir(parent) || mkpath(parent)
    ispath(target) && !overwrite && error("Output exists: $(target). Pass --overwrite=true to replace it.")
    temporary = joinpath(parent, ".$(basename(target)).tmp-$(uuid4())")
    try
        writer(temporary)
        mv(temporary, target; force=overwrite)
    catch
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return target
end

csv_cell(value) = begin
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_summary_csv(path, run_id, windows, halo_counts, window_hits, dm; overwrite=false)
    atomic_write(path; overwrite=overwrite) do temporary
        open(temporary, "w") do io
            println(io, "run_id,window_index,label,requested_min_msun,requested_max_msun,effective_min_msun,effective_max_msun,effective_empty,halo_count,los_intersection_count,dm_mean_pc_cm3,dm_std_pc_cm3,dm_min_pc_cm3,dm_max_pc_cm3,zero_fraction")
            for iw in eachindex(windows)
                values = @view dm[:, iw]
                row = Any[
                    run_id, iw, windows[iw].label, windows[iw].requested_min_msun,
                    windows[iw].requested_max_msun, windows[iw].effective_min_msun,
                    windows[iw].effective_max_msun, window_is_empty(windows[iw]),
                    halo_counts[iw], window_hits[iw], mean(values), std(values; corrected=false),
                    minimum(values), maximum(values), count(iszero, values) / length(values),
                ]
                println(io, join(csv_cell.(row), ','))
            end
        end
    end
end

function write_provenance(path, entries; overwrite=false)
    atomic_write(path; overwrite=overwrite) do temporary
        open(temporary, "w") do io
            for key in sort!(collect(keys(entries)))
                value = replace(string(entries[key]), '\n' => "\\n")
                println(io, "$(key)=$(value)")
            end
        end
    end
end

function script_sha256()
    return bytes2hex(sha256(read(@__FILE__)))
end

function file_sha256(path)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function git_head(root)
    head_path = joinpath(root, ".git", "HEAD")
    isfile(head_path) || return "unavailable"
    head = strip(read(head_path, String))
    if startswith(head, "ref: ")
        reference = strip(head[6:end])
        reference_path = joinpath(root, ".git", split(reference, '/')...)
        return isfile(reference_path) ? strip(read(reference_path, String)) : reference
    end
    return head
end

function write_hdf5_output(
    path,
    run_id,
    created_utc,
    windows,
    frb_pixels,
    frb_ra,
    frb_dec,
    frb_host_redshifts,
    frb_catalog_rows,
    source_redshift,
    dm,
    halo_counts,
    unique_hits,
    window_hits,
    pdf_edges,
    pdf_centers,
    pdf_counts,
    pdf_density,
    provenance;
    overwrite=false,
    save_ray_dm=true,
)
    atomic_write(path; overwrite=overwrite) do temporary
        h5open(temporary, "w") do h5
            # HDF5.jl reverses multidimensional dimensions on disk. Writing
            # the Julia (ray, window) matrix directly therefore gives h5py a
            # notebook-friendly (window, ray) dataset when per-ray output is enabled.
            if save_ray_dm
                h5["dm_pc_cm3"] = dm
                h5["frb_index"] = Int64.(1:length(frb_pixels))
                h5["frb_pixel_ring_1based"] = Int64.(frb_pixels)
                h5["frb_ra_rad"] = frb_ra
                h5["frb_dec_rad"] = frb_dec
                h5["frb_input_host_redshift"] = frb_host_redshifts
                h5["frb_input_catalog_data_row_1based"] = Int64.(frb_catalog_rows)
                h5["frb_source_redshift"] = fill(Float64(source_redshift), length(frb_pixels))
            end
            h5["source_redshift_grid"] = [Float64(source_redshift)]
            h5["window_index"] = Int64.(1:length(windows))
            h5["window_label"] = getfield.(windows, :label)
            h5["window_requested_min_msun"] = getfield.(windows, :requested_min_msun)
            h5["window_requested_max_msun"] = getfield.(windows, :requested_max_msun)
            h5["window_effective_min_msun"] = getfield.(windows, :effective_min_msun)
            h5["window_effective_max_msun"] = getfield.(windows, :effective_max_msun)
            h5["window_effective_empty"] = Int8.(window_is_empty.(windows))
            h5["window_halo_count"] = Int64.(halo_counts)
            h5["window_los_intersection_count"] = Int64.(window_hits)
            h5["pdf_bin_edges_pc_cm3"] = pdf_edges
            h5["pdf_bin_centers_pc_cm3"] = pdf_centers
            h5["pdf_count"] = pdf_counts
            h5["pdf_density_per_pc_cm3"] = pdf_density

            metadata = attrs(h5)
            metadata["schema_name"] = save_ray_dm ?
                "halfdome_fixed_z_dm_mass_windows" : "halfdome_fixed_z_dm_mass_window_histograms"
            metadata["schema_version"] = "1.1.0"
            metadata["run_id"] = run_id
            metadata["created_utc"] = created_utc
            metadata["per_ray_dm_saved"] = save_ray_dm
            metadata["n_rays"] = Int64(length(frb_pixels))
            if save_ray_dm
                metadata["dm_matrix_axes"] = "window,ray"
                metadata["dm_matrix_axes_hdf5_jl"] = "ray,window"
                metadata["notebook_single_window_shape"] = "reshape dm_pc_cm3[window,:] to (1,N_rays)"
            end
            metadata["pdf_matrix_axes"] = "window,bin"
            metadata["pdf_matrix_axes_hdf5_jl"] = "bin,window"
            metadata["mass_unit"] = "Msun"
            metadata["dm_unit"] = "pc cm^-3"
            metadata["mass_interval_convention"] = "lower-inclusive, upper-exclusive"
            metadata["healpix_ordering"] = "RING"
            metadata["healpix_pixel_indexing"] = "Julia 1-based; subtract 1 for healpy"
            metadata["shared_sightlines_for_all_windows"] = true
            metadata["catalog_passes"] = Int64(1)
            metadata["unique_halo_frb_intersection_count"] = Int64(unique_hits)
            metadata["window_labels_csv"] = join(getfield.(windows, :label), ',')
            for (key, value) in provenance
                value isa Union{Bool, Int, Int64, Float64, String} || continue
                metadata["provenance_$(key)"] = value
            end
        end
    end
end

function configuration(options; require_catalog=true)
    validate_known_options(options)
    source_redshift = get_float_option(options, ("source_redshift", "z_source"), DEFAULT_SOURCE_REDSHIFT)
    nside = get_int_option(options, ("nside",), 4096)
    nfrb = get_int_option(options, ("nfrb", "n"), 120_000)
    seed = get_int_option(options, ("seed", "frb_seed"), 42)
    unique_pixels = get_bool_option(options, ("unique_pixels",), true)
    sightline_catalog_option = get_string_option(options, ("sightline_catalog",), "")
    sightline_mode_default = isempty(sightline_catalog_option) ? "uniform" : "catalog"
    sightline_mode = lowercase(strip(get_string_option(options, ("sightline_mode",), sightline_mode_default)))
    sightline_catalog = isempty(sightline_catalog_option) ? "" : resolve_project_path(sightline_catalog_option)
    sightline_ra_column = get_string_option(options, ("sightline_ra_column",), "ra_rad")
    sightline_dec_column = get_string_option(options, ("sightline_dec_column",), "dec_rad")
    sightline_redshift_column = get_string_option(options, ("sightline_redshift_column",), "host_redshift")
    sightline_redshift_width = get_float_option(options, ("sightline_redshift_width",), 0.05)
    sightline_max_rows = get_int_option(options, ("sightline_max_rows",), 0)
    sightline_progress_every_rows = get_int_option(options, ("sightline_progress_every_rows",), 1_000_000)
    chunk_size = get_int_option(options, ("chunk_size", "chunkn"), 1_000_000)
    max_catalog_halos = get_int_option(options, ("max_catalog_halos",), 0)
    catalog_mass_floor = get_float_option(options, ("catalog_mass_floor",), DEFAULT_HALFDOME_MASS_FLOOR_MSUN)
    apply_catalog_mass_floor = get_bool_option(options, ("apply_catalog_mass_floor",), true)
    catalog_masses_are_msun_h = get_bool_option(options, ("catalog_masses_are_msun_h",), true)
    catalog = resolve_halfdome_catalog(
        get_string_option(options, ("catalog", "halfdome_path"), "lightcone_100.hdf5");
        require_exists=require_catalog,
    )
    output = resolve_project_path(get_string_option(
        options,
        ("output", "output_path"),
        joinpath("batched_data", "frb_z1_mass_windows", "halfdome_z1_dm_mass_windows.h5"),
    ))
    summary, provenance = split_output_sidecars(
        output,
        get_string_option(options, ("summary",), ""),
        get_string_option(options, ("provenance",), ""),
    )
    dm_cache = resolve_project_path(get_string_option(
        options,
        ("dm_cache", "dm_cache_file"),
        joinpath("frb_map_generation", "outputs", "shared_xgpaint_dm_cache.jld2"),
    ))
    dm_cache_overwrite = get_bool_option(options, ("dm_cache_overwrite",), false)
    dm_cleanup_nonpositive = get_bool_option(options, ("dm_cleanup_nonpositive",), true)
    dm_value_sanity_max = get_float_option(options, ("dm_value_sanity_max",), 1.0e8)
    dm_aperture_r200_multiplier = get_float_option(options, ("dm_aperture_r200_multiplier",), 1.0)
    pdf_edge_count = get_int_option(options, ("pdf_edge_count", "pdf_bins"), 300)
    pdf_spacing = lowercase(strip(get_string_option(options, ("pdf_spacing",), "log")))
    pdf_dm_min = get_float_option(options, ("pdf_dm_min",), 0.1)
    pdf_dm_max_text = lowercase(strip(get_string_option(options, ("pdf_dm_max",), "30000")))
    pdf_dm_max = pdf_dm_max_text == "auto" ? nothing : parse(Float64, pdf_dm_max_text)
    progress_every_batches = get_int_option(options, ("progress_every_batches",), 10)
    overwrite = get_bool_option(options, ("overwrite",), false)
    mass_windows_specification = get_string_option(options, ("mass_windows",), "")
    save_ray_dm = get_bool_option(options, ("save_ray_dm",), true)

    source_redshift > 0.0 || error("source_redshift must be positive.")
    nside > 0 || error("nside must be positive.")
    nfrb > 0 || error("nfrb must be positive.")
    sightline_mode in ("uniform", "catalog") || error("sightline_mode must be uniform or catalog.")
    sightline_mode == "catalog" && isempty(sightline_catalog) && error(
        "sightline_mode=catalog requires --sightline-catalog=PATH.",
    )
    sightline_mode == "uniform" && !isempty(sightline_catalog) && error(
        "A sightline catalogue was supplied with sightline_mode=uniform; remove one of those options.",
    )
    sightline_mode == "catalog" && !isfile(sightline_catalog) && error(
        "Sightline catalogue not found: $(sightline_catalog)",
    )
    sightline_redshift_width >= 0.0 || error("sightline_redshift_width must be nonnegative.")
    sightline_max_rows >= 0 || error("sightline_max_rows must be nonnegative.")
    sightline_progress_every_rows >= 0 || error("sightline_progress_every_rows must be nonnegative.")
    chunk_size > 0 || error("chunk_size must be positive.")
    max_catalog_halos >= 0 || error("max_catalog_halos must be nonnegative.")
    catalog_mass_floor >= 0.0 || error("catalog_mass_floor must be nonnegative.")
    isfinite(dm_aperture_r200_multiplier) && dm_aperture_r200_multiplier > 0.0 || error(
        "dm_aperture_r200_multiplier must be finite and positive.",
    )
    pdf_edge_count >= 2 || error("pdf_edge_count must be at least 2.")
    pdf_spacing in ("log", "linear") || error("pdf_spacing must be log or linear.")
    pdf_spacing == "log" && pdf_dm_min <= 0.0 && error("pdf_dm_min must be positive for log spacing.")
    pdf_dm_max !== nothing && pdf_dm_max <= pdf_dm_min && error("pdf_dm_max must exceed pdf_dm_min.")
    progress_every_batches >= 0 || error("progress_every_batches must be nonnegative.")

    windows = parse_mass_windows(
        mass_windows_specification;
        catalog_mass_floor_msun=catalog_mass_floor,
        apply_catalog_mass_floor=apply_catalog_mass_floor,
    )
    return (;
        source_redshift, nside, nfrb, seed, unique_pixels, sightline_mode, sightline_catalog,
        sightline_ra_column, sightline_dec_column, sightline_redshift_column,
        sightline_redshift_width, sightline_max_rows, sightline_progress_every_rows,
        chunk_size, max_catalog_halos,
        catalog_mass_floor, apply_catalog_mass_floor, catalog_masses_are_msun_h, catalog,
        output, summary, provenance, dm_cache, dm_cache_overwrite, dm_cleanup_nonpositive,
        dm_value_sanity_max, dm_aperture_r200_multiplier, pdf_edge_count, pdf_spacing,
        pdf_dm_min, pdf_dm_max, progress_every_batches,
        overwrite, mass_windows_specification, save_ray_dm, windows,
    )
end

function print_configuration(config)
    println("HalfDome fixed-z mass-window DM configuration:")
    println("  catalog=$(config.catalog)")
    println("  output=$(config.output)")
    println("  summary=$(config.summary)")
    println("  provenance=$(config.provenance)")
    println("  source_redshift=$(config.source_redshift)")
    println("  nside=$(config.nside), nfrb=$(config.nfrb), seed=$(config.seed), unique_pixels=$(config.unique_pixels)")
    println("  sightline_mode=$(config.sightline_mode)")
    if config.sightline_mode == "catalog"
        println("  sightline_catalog=$(config.sightline_catalog)")
        println(
            "  sightline columns=$(config.sightline_ra_column),$(config.sightline_dec_column),$(config.sightline_redshift_column); " *
            "host-z shell=$(config.source_redshift) +/- $(config.sightline_redshift_width)",
        )
        println("  sightline_max_rows=$(config.sightline_max_rows)")
    end
    println("  chunk_size=$(config.chunk_size), max_catalog_halos=$(config.max_catalog_halos)")
    println("  catalog_mass_floor_msun=$(config.catalog_mass_floor), apply=$(config.apply_catalog_mass_floor)")
    println("  catalog_masses_are_msun_h=$(config.catalog_masses_are_msun_h), h=$(H_VALUE)")
    println("  dm_cache=$(config.dm_cache), aperture=$(config.dm_aperture_r200_multiplier) R200, JULIA_NUM_THREADS=$(Threads.nthreads())")
    println("  save_ray_dm=$(config.save_ray_dm), custom_mass_windows=$(!isempty(config.mass_windows_specification))")
    print_window_table(config.windows)
    for (bounds, labels) in equivalent_effective_window_groups(config.windows)
        println("  NOTE: identical effective bounds $(bounds): $(join(labels, ", "))")
    end
end

function run_self_test()
    windows = default_mass_windows()
    @assert length(windows) == 10
    @assert windows[1].effective_min_msun == DEFAULT_HALFDOME_MASS_FLOOR_MSUN
    @assert window_is_empty(windows[2])
    @assert !mass_in_window(5.0e11, windows[1])
    mask = window_membership_mask(8.0e12, windows)
    @assert count(iw -> (mask & (UInt16(1) << (iw - 1))) != 0, eachindex(windows)) == 8
    @assert window_membership_mask(5.0e11, windows) == 0

    requested_spec = "m1e10_to_1e13:1e10:1e13,m1e10_to_1e14:1e10:1e14," *
                     "m1e10_to_1e15:1e10:1e15,m1e10_to_1e16:1e10:1e16," *
                     "m1e12_to_1e16:1e12:1e16,m1e13_to_1e16:1e13:1e16," *
                     "m1e14_to_1e16:1e14:1e16,m1e15_to_1e16:1e15:1e16"
    requested_windows = parse_mass_windows(requested_spec)
    @assert length(requested_windows) == 8
    @assert getfield.(requested_windows, :label) == [
        "m1e10_to_1e13", "m1e10_to_1e14", "m1e10_to_1e15", "m1e10_to_1e16",
        "m1e12_to_1e16", "m1e13_to_1e16", "m1e14_to_1e16", "m1e15_to_1e16",
    ]
    @assert requested_windows[1].effective_min_msun == DEFAULT_HALFDOME_MASS_FLOOR_MSUN
    @assert requested_windows[4].effective_min_msun == requested_windows[5].effective_min_msun
    @assert requested_windows[end].requested_max_msun == 1.0e16

    synthetic_dm = [0.0 0.0; 1.0 2.0; 2.0 4.0; 3.0 6.0]
    edges, centers, counts, density = histogram_columns(synthetic_dm, 4, 0.0, 6.0; spacing="linear")
    @assert length(edges) == 4
    @assert length(centers) == 3
    @assert sum(counts[:, 1]) == 4
    @assert all(isfinite, density)

    target_values = reshape([0.0, 0.05, 0.1, 1.0, 30_000.0, 40_000.0], :, 1)
    target_edges, target_centers, target_counts, target_density = histogram_columns(
        target_values, 300, 0.1, 30_000.0; spacing="log",
    )
    @assert length(target_edges) == 300
    @assert length(target_centers) == 299
    @assert target_edges[1] == 0.1
    @assert isapprox(target_edges[end], 30_000.0; rtol=2eps(Float64))
    @assert sum(target_counts) == 3
    @assert isapprox(sum(target_density[:, 1] .* diff(target_edges)), 1.0; rtol=1e-14)

    mktempdir() do directory
        summary = joinpath(directory, "summary.csv")
        provenance = joinpath(directory, "provenance.txt")
        sightline_catalog = joinpath(directory, "sightlines.csv")
        open(sightline_catalog, "w") do io
            println(io, "ra_rad,dec_rad,host_redshift,unused")
            println(io, "-0.1,0.0,0.995,1")
            println(io, "1.0,0.1,1.0,2")
            println(io, "2.0,-0.1,1.005,3")
            println(io, "3.0,0.0,1.02,4")
            println(io, "bad,row")
            println(io, "4.0,2.0,1.0,6")
        end
        selected1 = select_catalog_sightlines(
            sightline_catalog, 2, 17; target_redshift=1.0, redshift_width=0.01,
            progress_every_rows=0,
        )
        selected2 = select_catalog_sightlines(
            sightline_catalog, 2, 17; target_redshift=1.0, redshift_width=0.01,
            progress_every_rows=0,
        )
        @assert selected1.rows_scanned == 6
        @assert selected1.eligible_rows == 3
        @assert selected1.malformed_rows == 2
        @assert length(selected1.ras) == 2
        @assert selected1.catalog_rows == selected2.catalog_rows
        @assert all(0.0 .<= selected1.ras .< 2pi)
        write_summary_csv(summary, "self-test", windows[1:2], [4, 0], [4, 0], synthetic_dm; overwrite=false)
        write_provenance(provenance, Dict("run_id" => "self-test", "catalog_passes" => 1); overwrite=false)
        @assert startswith(read(summary, String), "run_id,window_index")
        @assert occursin("catalog_passes=1", read(provenance, String))

        sentinel_cache = joinpath(directory, "authoritative_legacy_cache.jld2")
        sentinel_payload = "authoritative-cache-sentinel\n"
        open(sentinel_cache, "w") do io
            write(io, sentinel_payload)
        end
        sentinel_sha_before = bytes2hex(sha256(read(sentinel_cache)))
        overwrite_blocked = false
        try
            enforce_generated_cache_target_policy(
                sentinel_cache, true, "public_v0.4_tau_conversion",
            )
        catch exception
            overwrite_blocked = occursin("Refusing to overwrite existing DM cache", sprint(showerror, exception))
        end
        @assert overwrite_blocked
        @assert read(sentinel_cache, String) == sentinel_payload
        @assert bytes2hex(sha256(read(sentinel_cache))) == sentinel_sha_before
        new_public_cache = joinpath(directory, "public_v04_dm_cache.jld2")
        @assert !isfile(new_public_cache)
        @assert isnothing(enforce_generated_cache_target_policy(
            new_public_cache, true, "public_v0.4_tau_conversion",
        ))
    end
    println("PASS: routing, custom mass windows, CSV reservoir, histogram, sidecar, and cache-overwrite policy tests.")
end

function main(options)
    config = configuration(options)
    print_configuration(config)
    for path in (config.output, config.summary, config.provenance)
        ispath(path) && !config.overwrite && error("Output exists: $(path). Pass --overwrite=true to replace it.")
    end

    run_id = string(uuid4())
    created_utc = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
    resolution = Healpix.Resolution(config.nside)
    npix = 12 * config.nside^2
    if config.sightline_mode == "uniform"
        rng = MersenneTwister(config.seed)
        frb_pixels = draw_random_frb_pixels(rng, npix, config.nfrb; unique_pixels=config.unique_pixels)
        frb_ra, frb_dec = pixel_centers_to_ra_dec(resolution, frb_pixels)
        frb_host_redshifts = fill(NaN, length(frb_pixels))
        frb_catalog_rows = zeros(Int64, length(frb_pixels))
        sightline_rows_scanned = Int64(0)
        sightline_eligible_rows = Int64(length(frb_pixels))
        sightline_malformed_rows = Int64(0)
    else
        println("Streaming catalogue sightlines from $(config.sightline_catalog)...")
        selection = select_catalog_sightlines(
            config.sightline_catalog,
            config.nfrb,
            config.seed;
            ra_column=config.sightline_ra_column,
            dec_column=config.sightline_dec_column,
            redshift_column=config.sightline_redshift_column,
            target_redshift=config.source_redshift,
            redshift_width=config.sightline_redshift_width,
            max_rows=config.sightline_max_rows,
            progress_every_rows=config.sightline_progress_every_rows,
        )
        frb_ra = selection.ras
        frb_dec = selection.decs
        frb_host_redshifts = selection.host_redshifts
        frb_catalog_rows = selection.catalog_rows
        frb_pixels = ra_dec_to_ring_pixels(resolution, frb_ra, frb_dec)
        sightline_rows_scanned = selection.rows_scanned
        sightline_eligible_rows = selection.eligible_rows
        sightline_malformed_rows = selection.malformed_rows
        println(
            "Selected $(length(frb_pixels)) catalogue sightlines from $(sightline_eligible_rows) eligible " *
            "rows ($(sightline_rows_scanned) scanned).",
        )
        if length(frb_pixels) < config.nfrb
            println(
                "WARNING: requested $(config.nfrb) sightlines but only $(length(frb_pixels)) matched; " *
                "the actual count is recorded in HDF5/provenance.",
            )
        end
    end
    actual_nfrb = length(frb_pixels)
    frb_ux, frb_uy, frb_uz = ra_dec_to_unit_vectors(frb_ra, frb_dec)
    sorted_frb_pixels, sorted_frb_indices = build_frb_pixel_lookup(frb_pixels)

    ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = config.dm_cleanup_nonpositive ? "true" : "false"
    cache_parent = dirname(config.dm_cache)
    isdir(cache_parent) || mkpath(cache_parent)
    dm_model = HaloDMProfile(BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE))
    interpolator_build = build_dm_interpolator_compatible(
        dm_model;
        cache_file=config.dm_cache,
        overwrite=config.dm_cache_overwrite,
        cleanup_nonpositive=config.dm_cleanup_nonpositive,
    )
    dm_model_interp = interpolator_build.profile
    dm_cache_public_profile_spot_ratio = validate_dm_interpolator_scale(dm_model_interp, dm_model)
    theta_min = compute_theta_min_local(dm_model_interp)
    workspace, sparse_disc_backend = make_sparse_disc_workspace(resolution)
    candidate_pixel_margin = config.sightline_mode == "catalog" ? Float64(Healpix.max_pixrad(resolution)) : 0.0
    accumulator = WindowDMAccumulator(actual_nfrb, length(config.windows))

    halo_counts = zeros(Int64, length(config.windows))
    valid_foreground_count = Ref(Int64(0))
    selected_union_count = Ref(Int64(0))
    below_floor_count = Ref(Int64(0))
    observed_foreground_min = Ref(Inf)
    observed_foreground_max = Ref(-Inf)
    streamed = stream_halfdome_batches(
        config.catalog,
        config.chunk_size;
        masses_are_msun_h=config.catalog_masses_are_msun_h,
        max_catalog_halos=config.max_catalog_halos,
    ) do batch_number, batch_start, x, y, z, masses, redshifts
        selected_indices = Int[]
        membership_masks = UInt16[]
        sizehint!(selected_indices, length(masses))
        sizehint!(membership_masks, length(masses))
        @inbounds for i in eachindex(masses)
            mass = Float64(masses[i])
            redshift = Float64(redshifts[i])
            if !isfinite(mass) || mass <= 0.0 || !isfinite(redshift) || redshift < 0.0 || redshift > config.source_redshift
                continue
            end
            valid_foreground_count[] += 1
            observed_foreground_min[] = min(observed_foreground_min[], mass)
            observed_foreground_max[] = max(observed_foreground_max[], mass)
            mass < config.catalog_mass_floor && (below_floor_count[] += 1)
            mask = window_membership_mask(mass, config.windows)
            mask == 0 && continue
            selected_union_count[] += 1
            push!(selected_indices, i)
            push!(membership_masks, mask)
            for iw in eachindex(config.windows)
                (mask & (UInt16(1) << (iw - 1))) != 0 && (halo_counts[iw] += 1)
            end
        end

        if !isempty(selected_indices)
            accumulate_batch_windows!(
                accumulator, workspace, dm_model_interp, theta_min, config.dm_aperture_r200_multiplier,
                candidate_pixel_margin,
                sorted_frb_pixels, sorted_frb_indices, frb_ux, frb_uy, frb_uz,
                Float64.(x[selected_indices]), Float64.(y[selected_indices]), Float64.(z[selected_indices]),
                masses[selected_indices], redshifts[selected_indices], membership_masks,
                length(config.windows),
            )
        end
        if config.progress_every_batches > 0 && batch_number % config.progress_every_batches == 0
            println("  streamed through catalog row $(batch_start + length(masses) - 1); selected=$(selected_union_count[])")
        end
    end

    dm, unique_hits, window_hits = reduce_accumulator(accumulator)
    all(isfinite, dm) || error("Non-finite values found in the final DM matrix.")
    maximum(dm) <= config.dm_value_sanity_max || error(
        "DM maximum $(maximum(dm)) exceeds dm_value_sanity_max=$(config.dm_value_sanity_max). Check the XGPaint cache.",
    )
    pdf_edges, pdf_centers, pdf_counts, pdf_density = histogram_columns(
        dm, config.pdf_edge_count, config.pdf_dm_min, config.pdf_dm_max; spacing=config.pdf_spacing,
    )

    catalog_stat = stat(config.catalog)
    sightline_catalog_stat = config.sightline_mode == "catalog" ? stat(config.sightline_catalog) : nothing
    dm_cache_exists = isfile(config.dm_cache)
    dm_cache_stat = dm_cache_exists ? stat(config.dm_cache) : nothing
    dm_cache_hash = dm_cache_exists ? file_sha256(config.dm_cache) : "missing"
    xgpaint_version = try
        string(Base.pkgversion(XGPaint))
    catch
        "unknown"
    end
    provenance_entries = Dict{String, Any}(
        "run_id" => run_id,
        "created_utc" => created_utc,
        "script" => abspath(@__FILE__),
        "script_sha256" => script_sha256(),
        "git_head" => git_head(project_root()),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "command_args" => join(ARGS, " "),
        "catalog_path" => abspath(config.catalog),
        "catalog_size_bytes" => catalog_stat.size,
        "catalog_mtime_unix" => catalog_stat.mtime,
        "catalog_total_halos" => streamed.catalog_halos,
        "catalog_streamed_halos" => streamed.streamed_halos,
        "catalog_truncated" => streamed.streamed_halos < streamed.catalog_halos,
        "catalog_passes" => 1,
        "catalog_mass_dataset" => "halo_mass_m200c",
        "catalog_mass_input_unit" => config.catalog_masses_are_msun_h ? "Msun/h" : "Msun",
        "catalog_mass_conversion" => config.catalog_masses_are_msun_h ? "divide by h=0.68" : "none",
        "catalog_resolution_floor_msun" => config.catalog_mass_floor,
        "catalog_resolution_floor_is_approximate" => true,
        "apply_catalog_mass_floor" => config.apply_catalog_mass_floor,
        "foreground_valid_halo_count" => valid_foreground_count[],
        "foreground_below_declared_floor_count" => below_floor_count[],
        "foreground_observed_min_msun" => observed_foreground_min[],
        "foreground_observed_max_msun" => observed_foreground_max[],
        "selected_union_halo_count" => selected_union_count[],
        "source_redshift" => config.source_redshift,
        "nside" => config.nside,
        "nfrb_requested" => config.nfrb,
        "nfrb_actual" => actual_nfrb,
        "frb_seed" => config.seed,
        "unique_pixels" => config.unique_pixels,
        "sightline_mode" => config.sightline_mode,
        "sightline_sampling" => config.sightline_mode == "uniform" ?
            "uniform HEALPix pixel centers" : "one-pass uniform reservoir sample of matching CSV rows",
        "sightline_catalog_path" => config.sightline_mode == "catalog" ? abspath(config.sightline_catalog) : "",
        "sightline_catalog_size_bytes" => config.sightline_mode == "catalog" ? sightline_catalog_stat.size : -1,
        "sightline_catalog_mtime_unix" => config.sightline_mode == "catalog" ? sightline_catalog_stat.mtime : NaN,
        "sightline_ra_column" => config.sightline_ra_column,
        "sightline_dec_column" => config.sightline_dec_column,
        "sightline_redshift_column" => config.sightline_redshift_column,
        "sightline_redshift_width" => config.sightline_redshift_width,
        "sightline_rows_scanned" => sightline_rows_scanned,
        "sightline_eligible_rows" => sightline_eligible_rows,
        "sightline_malformed_rows" => sightline_malformed_rows,
        "sightline_max_rows" => config.sightline_max_rows,
        "candidate_pixel_search_margin_rad" => candidate_pixel_margin,
        "candidate_pixel_search_margin_policy" => config.sightline_mode == "catalog" ?
            "Healpix.max_pixrad added only to candidate search; exact RA/Dec still filtered at aperture" : "none",
        "shared_sightlines_for_all_windows" => true,
        "save_ray_dm" => config.save_ray_dm,
        "mass_windows_specification" => isempty(config.mass_windows_specification) ? "legacy defaults" : config.mass_windows_specification,
        "profile" => "HaloDMProfile(BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68))",
        "halo_dm_profile_source" => HALO_DM_PROFILE_SOURCE,
        "dm_observer_frame_redshift_dilution" => "1/(1+z_halo)",
        "dm_aperture_r200_multiplier" => config.dm_aperture_r200_multiplier,
        "profile_angular_support" => "XGPaint compute_theta_max with explicit mult=$(config.dm_aperture_r200_multiplier)",
        "xgpaint_version" => xgpaint_version,
        "dm_cache_file" => abspath(config.dm_cache),
        "dm_cache_exists_after_interpolator_build" => dm_cache_exists,
        "dm_cache_size_bytes" => dm_cache_exists ? dm_cache_stat.size : -1,
        "dm_cache_mtime_unix" => dm_cache_exists ? dm_cache_stat.mtime : NaN,
        "dm_cache_sha256" => dm_cache_hash,
        "dm_cache_overwrite" => config.dm_cache_overwrite,
        "dm_cleanup_nonpositive" => config.dm_cleanup_nonpositive,
        "dm_cache_loader" => interpolator_build.loader,
        "dm_cache_logtheta_key" => interpolator_build.logtheta_key,
        "dm_cache_nonpositive_replaced" => interpolator_build.nonpositive_replaced,
        "dm_model_family" => interpolator_build.model_family,
        "dm_cache_to_public_v0p4_profile_spot_ratio" => dm_cache_public_profile_spot_ratio,
        "dm_cache_to_public_v0p4_profile_spot_ratio_point" => "theta=1e-4 rad, mass=1e14 Msun, z=1 (clamped to cache axes)",
        "sparse_disc_backend" => sparse_disc_backend,
        "pdf_edge_count" => config.pdf_edge_count,
        "pdf_spacing" => config.pdf_spacing,
        "pdf_dm_min" => config.pdf_dm_min,
        "pdf_dm_max" => config.pdf_dm_max === nothing ? "auto" : config.pdf_dm_max,
        "pdf_density_normalization" => "linear DM; in-range samples renormalized to integral 1",
        "unique_halo_frb_intersection_count" => unique_hits,
        "output_hdf5" => abspath(config.output),
        "summary_csv" => abspath(config.summary),
        "provenance_file" => abspath(config.provenance),
    )

    write_hdf5_output(
        config.output, run_id, created_utc, config.windows, frb_pixels, frb_ra, frb_dec,
        frb_host_redshifts, frb_catalog_rows,
        config.source_redshift, dm, halo_counts, unique_hits, window_hits,
        pdf_edges, pdf_centers, pdf_counts, pdf_density, provenance_entries;
        overwrite=config.overwrite,
        save_ray_dm=config.save_ray_dm,
    )
    write_summary_csv(
        config.summary, run_id, config.windows, halo_counts, window_hits, dm;
        overwrite=config.overwrite,
    )
    write_provenance(config.provenance, provenance_entries; overwrite=config.overwrite)

    println("Completed one catalog pass: $(streamed.streamed_halos) / $(streamed.catalog_halos) halos streamed.")
    println("Selected union halos: $(selected_union_count[]); unique halo/FRB intersections: $(unique_hits).")
    println("Saved HDF5: $(config.output)")
    println("Saved summary: $(config.summary)")
    println("Saved provenance: $(config.provenance)")
end

if HELP_MODE
    show_help()
elseif SELF_TEST_MODE
    run_self_test()
elseif DRY_RUN_MODE
    config = configuration(CLI_OPTIONS; require_catalog=false)
    print_configuration(config)
    println("Dry run only: no catalog was opened and no output was written.")
else
    main(CLI_OPTIONS)
end
