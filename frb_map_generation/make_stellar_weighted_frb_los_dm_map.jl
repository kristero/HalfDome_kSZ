if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# Stellar-mass-weighted FRB host selection plus LOS foreground DM.
#
# This script is meant to mimic an observed FRB sample:
#   1. Select host halos in a thin source-redshift shell with probability
#      proportional to Mstar^alpha_star.
#   2. Use those host halo sky positions as the FRB sightlines.
#   3. Integrate foreground DM only along those FRB sightlines from all
#      intersecting foreground halos with z_min <= z_halo < z_FRB.
#
# The stellar-mass weight is used only for source/host selection. It never
# weights the foreground gas profile.
#
# Computed stellar masses support:
#   stellar_mass_relation=moster2013   # default, old behavior
#   stellar_mass_relation=cosmos2020   # COSMOS2020 central SHMR with M200c -> Mvir conversion

using XGPaint
using HDF5
using Healpix
using Random
using Statistics
using Base.Threads
using Plots

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const OMEGAM = OMEGAB + OMEGAC
const OMEGAL = 1.0 - OMEGAM
const DEFAULT_SOURCE_REDSHIFT = 1.0
const COMPUTED_STELLAR_MASS_FIELD = "computed_smhm_moster_like"
const DEFAULT_STELLAR_MASS_RELATION = "moster2013"

const COSMOS2020_H0_REFERENCE = 70.0
const COSMOS2020_H0_TARGET = 100.0 * H_VALUE
const COSMOS2020_LOG_MHALO_SHIFT = log10(COSMOS2020_H0_REFERENCE / COSMOS2020_H0_TARGET)
const COSMOS2020_LOG_MSTAR_SHIFT = 2.0 * log10(COSMOS2020_H0_REFERENCE / COSMOS2020_H0_TARGET)
const COSMOS2020_PARAMS = [
    0.2 0.5 12.629 10.855 0.487 0.935 1.939;
    0.5 0.8 12.793 10.927 0.502 0.802 3.132;
    0.8 1.1 12.730 11.013 0.454 1.109 1.925;
    1.1 1.5 12.673 10.967 0.393 0.746 0.335;
    1.5 2.0 12.787 11.040 0.410 0.716 1.312;
    2.0 2.5 13.097 11.254 0.495 0.668 1.077;
    2.5 3.0 12.627 10.920 0.393 0.274 0.446;
    3.0 3.5 12.820 11.067 0.465 0.354 0.741;
    3.5 4.5 13.638 12.222 0.551 1.557 3.149;
    4.5 5.5 13.547 12.105 0.567 1.427 3.225;
]
const COSMOS2020_TABLE = Ref{Any}(nothing)

const compute_theta_max_local =
    isdefined(XGPaint, Symbol("compute_", Char(0x03b8), "max")) ?
    getfield(XGPaint, Symbol("compute_", Char(0x03b8), "max")) :
    error("XGPaint does not define compute_theta_max.")

thread_capacity() = isdefined(Base.Threads, :maxthreadid) ? Base.Threads.maxthreadid() : Base.Threads.nthreads()
code_root() = @__DIR__
project_root() = basename(code_root()) == "frb_map_generation" ? dirname(code_root()) : code_root()

function resolve_project_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(project_root(), path))
end

function resolve_hdf5_catalog_from_directory(dir::AbstractString)
    candidates = filter(readdir(dir; join=true)) do entry
        isfile(entry) && lowercase(splitext(entry)[2]) in (".h5", ".hdf5")
    end
    isempty(candidates) && error("halfdome_path=$(dir) is a directory, but it contains no HDF5 catalog.")

    for preferred in ("lightcone_100.hdf5", "lightcone_100.h5", "halos.hdf5", "halos.h5")
        matches = filter(entry -> lowercase(basename(entry)) == preferred, candidates)
        length(matches) == 1 && return only(matches)
    end

    length(candidates) == 1 && return only(candidates)
    error("halfdome_path=$(dir) contains multiple HDF5 files. Pass the exact catalog file.")
end

function resolve_halfdome_catalog_path(path::AbstractString)
    isempty(path) && error("halfdome_path cannot be empty.")

    trial_paths = String[]
    if isabspath(path)
        push!(trial_paths, String(path))
    else
        push!(trial_paths, normpath(joinpath(project_root(), path)))
        push!(trial_paths, normpath(joinpath(code_root(), path)))
        push!(trial_paths, normpath(joinpath(pwd(), path)))
        push!(trial_paths, normpath(joinpath(project_root(), "halfdome", path)))
        push!(trial_paths, normpath(joinpath(project_root(), "HalfDome", path)))
    end

    unique_trial_paths = unique(trial_paths)
    for candidate in unique_trial_paths
        isfile(candidate) && return candidate
        isdir(candidate) && return resolve_hdf5_catalog_from_directory(candidate)
    end

    error("Could not find HalfDome catalog $(repr(path)). Tried: $(join(unique_trial_paths, ", "))")
end

function get_string_arg(key, default; env=nothing)
    if env !== nothing
        env_names = env isa AbstractString ? (env,) : env
        for env_name in env_names
            if haskey(ENV, env_name)
                return String(ENV[env_name])
            end
        end
    end

    prefix1 = "--" * key * "="
    prefix2 = key * "="
    for arg in ARGS
        if startswith(arg, prefix1)
            return String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, prefix2)
            return String(split(arg, "=", limit=2)[2])
        end
    end
    return String(default)
end

function get_int_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Int, value)
    return Int(default)
end

function get_float_arg(key, default; env=nothing)
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse(Float64, value)
    return Float64(default)
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
    value = get_string_arg(key, ""; env=env)
    isempty(value) || return parse_bool_arg(value)
    return Bool(default)
end

function fmt_param_value(x)
    s = string(x)
    s = replace(s, "-" => "m")
    s = replace(s, "." => "p")
    s = replace(s, "+" => "")
    return s
end

function hdf5_has_path(h5, path::AbstractString)
    try
        h5[path]
        return true
    catch
        return false
    end
end

function available_hdf5_keys(h5)
    return sort!(collect(String.(keys(h5))))
end

function is_computed_stellar_mass_field(field::AbstractString)
    return lowercase(strip(String(field))) in (
        COMPUTED_STELLAR_MASS_FIELD,
        "computed",
        "compute",
        "smhm",
        "analytic",
        "relation",
        "computed_smhm",
        "moster",
        "moster2013",
        "cosmos",
        "cosmos2020",
        "shuntov",
        "shuntov2022",
    )
end

function normalize_stellar_mass_relation(relation::AbstractString)
    value = lowercase(strip(String(relation)))
    value in ("", "auto", "computed", "moster", "moster2013", "moster_like", "moster-like") &&
        return "moster2013"
    value in ("cosmos", "cosmos2020", "shuntov", "shuntov2022") &&
        return "cosmos2020"
    error("stellar_mass_relation must be moster2013 or cosmos2020, got $(repr(relation)).")
end

function moster2013_stellar_mass_from_m200c(m200c_msun::Real, z::Real)
    mhalo = Float64(m200c_msun)
    redshift = Float64(z)
    if !isfinite(mhalo) || mhalo <= 0.0 || !isfinite(redshift) || redshift <= -1.0
        return NaN
    end

    zfrac = redshift / (1.0 + redshift)
    log10_m1 = 11.590 + 1.195 * zfrac
    m1 = 10.0^log10_m1
    n = 0.0351 - 0.0247 * zfrac
    beta = 1.376 - 0.826 * zfrac
    gamma = 0.608 + 0.329 * zfrac

    ratio = mhalo / m1
    denom = ratio^(-beta) + ratio^gamma
    if !isfinite(denom) || denom <= 0.0 || !isfinite(n) || n <= 0.0
        return NaN
    end

    return 2.0 * n * mhalo / denom
end

stellar_mass_from_halo_mass(mhalo_msun::Real, z::Real) =
    moster2013_stellar_mass_from_m200c(mhalo_msun, z)

omega_m_z(z::Real) = OMEGAM * (1.0 + Float64(z))^3 / (OMEGAM * (1.0 + Float64(z))^3 + OMEGAL)

function bryan_norman_delta_vir_critical(z::Real)
    x = omega_m_z(z) - 1.0
    return 18.0 * pi^2 + 82.0 * x - 39.0 * x^2
end

nfw_mass_fraction(c::Real) = log1p(Float64(c)) - Float64(c) / (1.0 + Float64(c))

function duffy2008_c200c(m200c_msun::Real, z::Real)
    mass_hinv_msun = Float64(m200c_msun) * H_VALUE
    c = 5.71 * (mass_hinv_msun / 2.0e12)^(-0.084) * (1.0 + Float64(z))^(-0.47)
    return isfinite(c) && c > 0.0 ? c : NaN
end

function log_mvir_from_log_m200c_direct(log_m200c::Real, z::Real)
    redshift = Float64(z)
    m200c = 10.0^Float64(log_m200c)
    if !isfinite(m200c) || m200c <= 0.0 || !isfinite(redshift) || redshift <= -1.0
        return NaN
    end

    c200c = duffy2008_c200c(m200c, redshift)
    if !isfinite(c200c) || c200c <= 0.0
        return NaN
    end

    delta_target = bryan_norman_delta_vir_critical(redshift)
    if !isfinite(delta_target) || delta_target <= 0.0
        return NaN
    end

    f_c = nfw_mass_fraction(c200c)
    target = delta_target / 200.0
    g(x) = nfw_mass_fraction(c200c * x) / f_c / x^3 - target

    lo = 0.05
    hi = 20.0
    g_lo = g(lo)
    g_hi = g(hi)
    if !isfinite(g_lo) || !isfinite(g_hi)
        return NaN
    end
    while g_hi > 0.0 && hi < 1.0e4
        hi *= 2.0
        g_hi = g(hi)
    end
    if g_lo < 0.0 || g_hi > 0.0
        return NaN
    end

    for _ in 1:60
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0
            lo = mid
        else
            hi = mid
        end
    end

    x = 0.5 * (lo + hi)
    mass_ratio = nfw_mass_fraction(c200c * x) / f_c
    return Float64(log_m200c) + log10(mass_ratio)
end

function cosmos2020_log_mhalo_from_log_mstar(log_mstar::Real, row)
    log_m1 = Float64(row[3]) + COSMOS2020_LOG_MHALO_SHIFT
    log_mstar0 = Float64(row[4]) + COSMOS2020_LOG_MSTAR_SHIFT
    beta = Float64(row[5])
    delta = Float64(row[6])
    gamma = Float64(row[7])

    x = 10.0^(Float64(log_mstar) - log_mstar0)
    return log_m1 + beta * log10(x) + x^delta / (1.0 + x^(-gamma)) - 0.5
end

function interp1_sorted(xs, ys, x::Real)
    value = Float64(x)
    if !isfinite(value) || value < first(xs) || value > last(xs)
        return NaN
    end
    idx = searchsortedlast(xs, value)
    idx <= 0 && return first(ys)
    idx >= length(xs) && return last(ys)
    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]
    t = (value - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
end

function cosmos2020_row_index(z::Real)
    redshift = Float64(z)
    redshift < COSMOS2020_PARAMS[1, 2] && return 1
    for i in 2:size(COSMOS2020_PARAMS, 1)
        if redshift >= COSMOS2020_PARAMS[i, 1] && redshift < COSMOS2020_PARAMS[i, 2]
            return i
        end
    end
    redshift >= COSMOS2020_PARAMS[end, 1] && return size(COSMOS2020_PARAMS, 1)
    return 1
end

function build_cosmos2020_inverse_tables()
    log_mstar_grid = collect(range(3.0, 14.0; length=6000))
    tables = Vector{NamedTuple}(undef, size(COSMOS2020_PARAMS, 1))
    for row_idx in 1:size(COSMOS2020_PARAMS, 1)
        row = @view COSMOS2020_PARAMS[row_idx, :]
        log_mhalo_grid = [cosmos2020_log_mhalo_from_log_mstar(log_mstar, row) for log_mstar in log_mstar_grid]
        order = sortperm(log_mhalo_grid)
        tables[row_idx] = (
            z_min=Float64(row[1]),
            z_max=Float64(row[2]),
            log_mhalo=Float64.(log_mhalo_grid[order]),
            log_mstar=Float64.(log_mstar_grid[order]),
        )
    end
    return tables
end

function build_cosmos2020_m200c_to_mstar_table()
    inverse_tables = build_cosmos2020_inverse_tables()
    z_grid = collect(range(0.0, 6.0; length=481))
    log_m200c_grid = collect(range(9.0, 16.7; length=501))
    log_mstar_table = Matrix{Float64}(undef, length(log_m200c_grid), length(z_grid))

    for iz in eachindex(z_grid)
        redshift = z_grid[iz]
        inv_table = inverse_tables[cosmos2020_row_index(redshift)]
        for im in eachindex(log_m200c_grid)
            log_mvir = log_mvir_from_log_m200c_direct(log_m200c_grid[im], redshift)
            log_mstar_table[im, iz] = interp1_sorted(inv_table.log_mhalo, inv_table.log_mstar, log_mvir)
        end
    end

    return (
        z_grid=z_grid,
        log_m200c_grid=log_m200c_grid,
        log_mstar_table=log_mstar_table,
    )
end

function ensure_cosmos2020_table!()
    if COSMOS2020_TABLE[] === nothing
        println("Building COSMOS2020 M200c -> Mvir -> Mstar interpolation table...")
        flush(stdout)
        COSMOS2020_TABLE[] = build_cosmos2020_m200c_to_mstar_table()
    end
    return COSMOS2020_TABLE[]
end

function cosmos2020_log_mstar_from_m200c(m200c_msun::Real, z::Real)
    m200c = Float64(m200c_msun)
    redshift = Float64(z)
    if !isfinite(m200c) || m200c <= 0.0 || !isfinite(redshift)
        return NaN
    end

    table = ensure_cosmos2020_table!()
    log_m200c = log10(m200c)
    logm_grid = table.log_m200c_grid
    z_grid = table.z_grid
    values = table.log_mstar_table

    if log_m200c < first(logm_grid) || log_m200c > last(logm_grid) ||
       redshift < first(z_grid) || redshift > last(z_grid)
        return NaN
    end

    im = searchsortedlast(logm_grid, log_m200c)
    iz = searchsortedlast(z_grid, redshift)
    im = clamp(im, 1, length(logm_grid) - 1)
    iz = clamp(iz, 1, length(z_grid) - 1)

    x0 = logm_grid[im]
    x1 = logm_grid[im + 1]
    z0 = z_grid[iz]
    z1 = z_grid[iz + 1]
    tx = (log_m200c - x0) / (x1 - x0)
    tz = (redshift - z0) / (z1 - z0)

    v00 = values[im, iz]
    v10 = values[im + 1, iz]
    v01 = values[im, iz + 1]
    v11 = values[im + 1, iz + 1]
    any(!isfinite, (v00, v10, v01, v11)) && return NaN

    return (1.0 - tx) * (1.0 - tz) * v00 +
           tx * (1.0 - tz) * v10 +
           (1.0 - tx) * tz * v01 +
           tx * tz * v11
end

function cosmos2020_stellar_mass_from_m200c(m200c_msun::Real, z::Real)
    log_mstar = cosmos2020_log_mstar_from_m200c(m200c_msun, z)
    return isfinite(log_mstar) ? 10.0^log_mstar : NaN
end

function computed_stellar_mass_from_m200c(m200c_msun::Real, z::Real, relation::AbstractString)
    relation_name = normalize_stellar_mass_relation(relation)
    relation_name == "moster2013" && return moster2013_stellar_mass_from_m200c(m200c_msun, z)
    relation_name == "cosmos2020" && return cosmos2020_stellar_mass_from_m200c(m200c_msun, z)
    error("Unsupported stellar_mass_relation=$(relation_name).")
end

function resolve_stellar_mass_field(h5, requested::AbstractString)
    requested_clean = strip(String(requested))
    if is_computed_stellar_mass_field(requested_clean)
        return COMPUTED_STELLAR_MASS_FIELD
    end

    if !isempty(requested_clean) && lowercase(requested_clean) != "auto"
        hdf5_has_path(h5, requested_clean) && return requested_clean
        error(
            "stellar_mass_field=$(repr(requested_clean)) was not found in the catalog. " *
            "Available top-level keys: $(join(available_hdf5_keys(h5), ", "))"
        )
    end

    candidates = (
        "stellar_mass",
        "stellar_mass_msun",
        "Mstar",
        "M_star",
        "mstar",
        "m_star",
        "galaxy_stellar_mass",
        "host_stellar_mass",
        "SubhaloMassType_Stars",
        "SubhaloStellarMass",
    )
    for candidate in candidates
        hdf5_has_path(h5, candidate) && return candidate
    end

    println(
        "No stellar-mass dataset was found. Using analytic stellar-mass-halo-mass relation " *
        "$(COMPUTED_STELLAR_MASS_FIELD)."
    )
    println("Available top-level keys were: $(join(available_hdf5_keys(h5), ", "))")
    return COMPUTED_STELLAR_MASS_FIELD
end

function halo_dm_constructor()
    if isdefined(Main, :HaloDMProfile)
        return getfield(Main, :HaloDMProfile)
    elseif isdefined(XGPaint, :HaloDMProfile)
        return getfield(XGPaint, :HaloDMProfile)
    end
    error("HaloDMProfile is not available in this Julia/XGPaint environment.")
end

function xgpaint_build_interpolator_function()
    if isdefined(Main, :build_interpolator)
        return getfield(Main, :build_interpolator)
    elseif isdefined(XGPaint, :build_interpolator)
        return getfield(XGPaint, :build_interpolator)
    end
    error("build_interpolator is not available in this Julia/XGPaint environment.")
end

function xgpaint_paint_function()
    if isdefined(Main, :paint!)
        return getfield(Main, :paint!)
    elseif isdefined(XGPaint, :paint!)
        return getfield(XGPaint, :paint!)
    end
    error("paint! is not available in this Julia/XGPaint environment.")
end

function make_dm_model()
    constructor = halo_dm_constructor()
    tau_model = XGPaint.BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE)
    return constructor(tau_model)
end

function positions_to_ra_dec(positions)
    n = size(positions, 2)
    ras = Vector{Float64}(undef, n)
    decs = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        x = positions[1, i]
        y = positions[2, i]
        z = positions[3, i]
        r = sqrt(x * x + y * y + z * z)
        r > 0.0 || error("Catalog halo position has zero radius.")
        theta, phi = Healpix.vec2ang(x / r, y / r, z / r)
        ras[i] = Float64(phi)
        decs[i] = Float64(pi / 2 - theta)
    end

    return ras, decs
end

function ra_dec_to_unit_vectors(ras, decs)
    ux = Vector{Float64}(undef, length(ras))
    uy = Vector{Float64}(undef, length(ras))
    uz = Vector{Float64}(undef, length(ras))

    @threads for i in eachindex(ras)
        ra = Float64(ras[i])
        dec = Float64(decs[i])
        cosdec = cos(dec)
        ux[i] = cosdec * cos(ra)
        uy[i] = cosdec * sin(ra)
        uz[i] = sin(dec)
    end

    return ux, uy, uz
end

function ra_dec_to_ring_pixels(res, ras, decs)
    pixels = Vector{Int}(undef, length(ras))
    @inbounds for i in eachindex(ras)
        theta = pi / 2 - Float64(decs[i])
        phi = Float64(ras[i])
        pixels[i] = Healpix.ang2pixRing(res, theta, phi)
    end
    return pixels
end

function stellar_mass_unnormalized_weights(mstar; alpha_star::Float64=1.0, eps::Float64=1.0e-30)
    alpha_star >= 0.0 && isfinite(alpha_star) || error("alpha_star must be finite and >= 0.")
    eps >= 0.0 && isfinite(eps) || error("eps must be finite and >= 0.")

    weights = zeros(Float64, length(mstar))
    @inbounds for i in eachindex(mstar)
        value = Float64(mstar[i])
        if isfinite(value) && value > eps
            weights[i] = value^alpha_star
        end
    end
    weights[.!isfinite.(weights)] .= 0.0
    return weights
end

function stellar_mass_weights(mstar; alpha_star::Float64=1.0, eps::Float64=1.0e-30)
    weights = stellar_mass_unnormalized_weights(mstar; alpha_star=alpha_star, eps=eps)
    total = sum(weights)
    total > 0.0 || error("All stellar-mass weights are zero. Check stellar masses, alpha_star, and eps.")
    return weights ./ total
end

function print_host_weight_diagnostics(mstar_shell, p_shell)
    positive = Float64[v for v in mstar_shell if isfinite(v) && v > 0.0]
    isempty(positive) && error("No positive finite stellar masses in the source shell.")

    p_sorted = sort(Float64.(p_shell), rev=true)
    n_eff = 1.0 / sum(p_shell .^ 2)

    println("Host-shell stellar-mass diagnostics:")
    println("  number of halos in shell: $(length(mstar_shell))")
    println("  positive finite Mstar in shell: $(length(positive))")
    println("  Mstar min/median/max [Msun]: $(minimum(positive)), $(median(positive)), $(maximum(positive))")
    println("  N_eff = $(n_eff)")
    for frac in (0.01, 0.10, 0.50)
        top_n = max(1, ceil(Int, frac * length(p_sorted)))
        contribution = sum(@view p_sorted[1:top_n])
        println("  top $(round(100 * frac; digits=0))% halos: $(top_n) halos contribute $(100 * contribution)% of probability")
    end
end

function weighted_sample_local_indices(rng, p, n::Int)
    cdf = cumsum(Float64.(p))
    cdf[end] = 1.0
    choices = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        choices[i] = searchsortedfirst(cdf, rand(rng))
    end
    return choices
end

function source_host_mask(halo_mass, redshift; z_min::Float64, z_max::Float64, mass_min::Float64, mass_max::Float64)
    keep = isfinite.(redshift) .& isfinite.(halo_mass)
    keep .&= redshift .>= z_min
    isfinite(z_max) && (keep .&= redshift .< z_max)
    keep .&= halo_mass .> 0.0
    mass_min > 0.0 && (keep .&= halo_mass .>= mass_min)
    isfinite(mass_max) && (keep .&= halo_mass .< mass_max)
    return keep
end

function collect_shell_host_candidates(
    catalog_path::AbstractString;
    stellar_mass_field::AbstractString,
    stellar_mass_relation::AbstractString,
    z_source::Float64,
    dz::Float64,
    chunkN::Int,
    stellar_mass_divide_by_h::Bool,
)
    dz > 0.0 || error("dz must be positive.")

    candidate_x = Float64[]
    candidate_y = Float64[]
    candidate_z = Float64[]
    candidate_mass = Float64[]
    candidate_redshift = Float64[]
    candidate_mstar = Float64[]
    candidate_indices = Int[]

    selected_field = Ref("")
    total_halo_count = Ref(0)

    h5open(catalog_path, "r") do h5
        selected_field[] = resolve_stellar_mass_field(h5, stellar_mass_field)
        use_computed_mstar = selected_field[] == COMPUTED_STELLAR_MASS_FIELD
        if use_computed_mstar && stellar_mass_divide_by_h
            println("stellar_mass_divide_by_h=true is ignored for computed SMHM stellar masses.")
        end

        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        mstar_ds = use_computed_mstar ? nothing : h5[selected_field[]]
        total_halo_count[] = size(pos_ds, 2)

        length(redshift_ds) == total_halo_count[] ||
            error("redshift dataset length does not match Position halo count.")
        length(mass_ds) == total_halo_count[] ||
            error("halo_mass_m200c dataset length does not match Position halo count.")
        (use_computed_mstar || length(mstar_ds) == total_halo_count[]) ||
            error("stellar mass dataset $(selected_field[]) length does not match Position halo count.")

        for batch_start in 1:chunkN:total_halo_count[]
            batch_stop = min(batch_start + chunkN - 1, total_halo_count[])
            idx = batch_start:batch_stop

            positions = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            if use_computed_mstar
                mstar = computed_stellar_mass_from_m200c.(masses, redshifts, Ref(stellar_mass_relation))
            else
                mstar = Float64.(mstar_ds[idx])
                stellar_mass_divide_by_h && (mstar ./= H_VALUE)
            end

            keep = isfinite.(redshifts) .& (abs.(redshifts .- z_source) .< 0.5 * dz)
            any(keep) || continue

            local_indices = findall(keep)
            append!(candidate_x, positions[1, local_indices])
            append!(candidate_y, positions[2, local_indices])
            append!(candidate_z, positions[3, local_indices])
            append!(candidate_mass, masses[local_indices])
            append!(candidate_redshift, redshifts[local_indices])
            append!(candidate_mstar, mstar[local_indices])
            append!(candidate_indices, batch_start .+ local_indices .- 1)
        end
    end

    isempty(candidate_indices) && error("No host halos found in abs(z_halo - $(z_source)) < $(0.5 * dz).")
    positions = Matrix{Float64}(undef, 3, length(candidate_indices))
    positions[1, :] .= candidate_x
    positions[2, :] .= candidate_y
    positions[3, :] .= candidate_z

    return (
        positions=positions,
        masses=candidate_mass,
        redshifts=candidate_redshift,
        mstar=candidate_mstar,
        indices=candidate_indices,
        stellar_mass_field=selected_field[],
        total_halo_count=total_halo_count[],
    )
end

function sample_stellar_mass_weighted_hosts(
    shell;
    n_frb::Int,
    alpha_star::Float64,
    eps::Float64,
    seed::Int,
)
    p_shell = stellar_mass_weights(shell.mstar; alpha_star=alpha_star, eps=eps)
    print_host_weight_diagnostics(shell.mstar, p_shell)

    rng = MersenneTwister(seed)
    local_choices = weighted_sample_local_indices(rng, p_shell, n_frb)
    ras, decs = positions_to_ra_dec(shell.positions[:, local_choices])

    return (
        local_choices=local_choices,
        catalog_indices=shell.indices[local_choices],
        masses=shell.masses[local_choices],
        redshifts=shell.redshifts[local_choices],
        mstar=shell.mstar[local_choices],
        probabilities=p_shell[local_choices],
        ras=ras,
        decs=decs,
        p_shell=p_shell,
    )
end

function sample_stellar_mass_weighted_hosts_all_redshifts(
    catalog_path::AbstractString;
    stellar_mass_field::AbstractString,
    stellar_mass_relation::AbstractString,
    n_frb::Int,
    alpha_star::Float64,
    eps::Float64,
    seed::Int,
    chunkN::Int,
    stellar_mass_divide_by_h::Bool,
    source_z_min::Float64,
    source_z_max::Float64,
    source_halo_mass_min::Float64,
    source_halo_mass_max::Float64,
)
    n_frb > 0 || error("n_frb must be positive.")
    source_z_min >= 0.0 || error("source_z_min must be non-negative.")
    isfinite(source_z_max) && source_z_max > source_z_min || !isfinite(source_z_max) ||
        error("source_z_max must be greater than source_z_min or Inf.")
    source_halo_mass_min >= 0.0 || error("source_halo_mass_min must be non-negative.")
    source_halo_mass_max > source_halo_mass_min || error("source_halo_mass_max must be greater than source_halo_mass_min.")

    selected_field = Ref("")
    total_halo_count = Ref(0)
    candidate_count = Ref(0)
    positive_weight_count = Ref(0)
    total_weight = Ref(0.0)
    total_weight_sq = Ref(0.0)
    z_min_seen = Ref(Inf)
    z_max_seen = Ref(-Inf)
    mass_min_seen = Ref(Inf)
    mass_max_seen = Ref(-Inf)
    mstar_min_seen = Ref(Inf)
    mstar_max_seen = Ref(-Inf)

    println("Scanning all-redshift source candidates for total stellar-mass weight...")
    h5open(catalog_path, "r") do h5
        selected_field[] = resolve_stellar_mass_field(h5, stellar_mass_field)
        use_computed_mstar = selected_field[] == COMPUTED_STELLAR_MASS_FIELD
        if use_computed_mstar && stellar_mass_divide_by_h
            println("stellar_mass_divide_by_h=true is ignored for computed SMHM stellar masses.")
        end

        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        mstar_ds = use_computed_mstar ? nothing : h5[selected_field[]]
        total_halo_count[] = size(pos_ds, 2)

        for batch_start in 1:chunkN:total_halo_count[]
            batch_stop = min(batch_start + chunkN - 1, total_halo_count[])
            idx = batch_start:batch_stop
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            if use_computed_mstar
                mstar = computed_stellar_mass_from_m200c.(masses, redshifts, Ref(stellar_mass_relation))
            else
                mstar = Float64.(mstar_ds[idx])
                stellar_mass_divide_by_h && (mstar ./= H_VALUE)
            end

            keep = source_host_mask(
                masses,
                redshifts;
                z_min=source_z_min,
                z_max=source_z_max,
                mass_min=source_halo_mass_min,
                mass_max=source_halo_mass_max,
            )
            any(keep) || continue

            local_mstar = mstar[keep]
            local_weights = stellar_mass_unnormalized_weights(local_mstar; alpha_star=alpha_star, eps=eps)
            positive = local_weights .> 0.0
            candidate_count[] += count(keep)
            positive_count = count(positive)
            positive_count == 0 && continue

            selected_weights = local_weights[positive]
            selected_mstar = local_mstar[positive]
            selected_masses = masses[keep][positive]
            selected_redshifts = redshifts[keep][positive]

            positive_weight_count[] += positive_count
            total_weight[] += sum(selected_weights)
            total_weight_sq[] += sum(abs2, selected_weights)
            z_min_seen[] = min(z_min_seen[], minimum(selected_redshifts))
            z_max_seen[] = max(z_max_seen[], maximum(selected_redshifts))
            mass_min_seen[] = min(mass_min_seen[], minimum(selected_masses))
            mass_max_seen[] = max(mass_max_seen[], maximum(selected_masses))
            mstar_min_seen[] = min(mstar_min_seen[], minimum(selected_mstar))
            mstar_max_seen[] = max(mstar_max_seen[], maximum(selected_mstar))
        end
    end

    total_weight[] > 0.0 || error("All all-redshift source weights are zero. Check Mstar, alpha_star, eps, and source cuts.")
    n_eff = total_weight[]^2 / total_weight_sq[]

    println("All-redshift stellar-mass diagnostics:")
    println("  total halos in catalog: $(total_halo_count[])")
    println("  candidate host halos passing source cuts: $(candidate_count[])")
    println("  positive-weight host halos: $(positive_weight_count[])")
    println("  selected source redshift range: [$(z_min_seen[]), $(z_max_seen[])]")
    println("  selected host mass range [Msun]: [$(mass_min_seen[]), $(mass_max_seen[])]")
    println("  selected Mstar range [Msun]: [$(mstar_min_seen[]), $(mstar_max_seen[])]")
    println("  N_eff = $(n_eff)")

    rng = MersenneTwister(seed)
    thresholds = rand(rng, n_frb) .* total_weight[]
    threshold_order = sortperm(thresholds)
    sorted_thresholds = thresholds[threshold_order]

    selected_positions = Matrix{Float64}(undef, 3, n_frb)
    selected_masses = Vector{Float64}(undef, n_frb)
    selected_redshifts = Vector{Float64}(undef, n_frb)
    selected_mstar = Vector{Float64}(undef, n_frb)
    selected_probabilities = Vector{Float64}(undef, n_frb)
    selected_indices = Vector{Int}(undef, n_frb)

    cumulative_weight = 0.0
    draw_idx = 1
    last_position = zeros(Float64, 3)
    last_mass = 0.0
    last_redshift = 0.0
    last_mstar = 0.0
    last_probability = 0.0
    last_index = 0
    println("Sampling all-redshift FRB host halos with replacement...")
    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        use_computed_mstar = selected_field[] == COMPUTED_STELLAR_MASS_FIELD
        mstar_ds = use_computed_mstar ? nothing : h5[selected_field[]]

        for batch_start in 1:chunkN:total_halo_count[]
            draw_idx > n_frb && break

            batch_stop = min(batch_start + chunkN - 1, total_halo_count[])
            idx = batch_start:batch_stop
            positions = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            if use_computed_mstar
                mstar = computed_stellar_mass_from_m200c.(masses, redshifts, Ref(stellar_mass_relation))
            else
                mstar = Float64.(mstar_ds[idx])
                stellar_mass_divide_by_h && (mstar ./= H_VALUE)
            end

            keep = source_host_mask(
                masses,
                redshifts;
                z_min=source_z_min,
                z_max=source_z_max,
                mass_min=source_halo_mass_min,
                mass_max=source_halo_mass_max,
            )
            any(keep) || continue

            local_indices = findall(keep)
            local_mstar = mstar[local_indices]
            local_weights = stellar_mass_unnormalized_weights(local_mstar; alpha_star=alpha_star, eps=eps)

            @inbounds for local_pos in eachindex(local_indices)
                weight = local_weights[local_pos]
                weight > 0.0 || continue
                cumulative_weight += weight
                local_i = local_indices[local_pos]
                last_position .= positions[:, local_i]
                last_mass = masses[local_i]
                last_redshift = redshifts[local_i]
                last_mstar = mstar[local_i]
                last_probability = weight / total_weight[]
                last_index = batch_start + local_i - 1

                while draw_idx <= n_frb && sorted_thresholds[draw_idx] <= cumulative_weight
                    output_idx = threshold_order[draw_idx]
                    selected_positions[:, output_idx] .= positions[:, local_i]
                    selected_masses[output_idx] = last_mass
                    selected_redshifts[output_idx] = last_redshift
                    selected_mstar[output_idx] = last_mstar
                    selected_probabilities[output_idx] = last_probability
                    selected_indices[output_idx] = last_index
                    draw_idx += 1
                end
            end
        end
    end
    if draw_idx <= n_frb
        last_index > 0 || error("Internal sampling error: no positive-weight host was available on the second pass.")
        println("  warning: filling $(n_frb - draw_idx + 1) final all-redshift host draws from the last positive-weight halo due to floating-point boundary roundoff.")
        while draw_idx <= n_frb
            output_idx = threshold_order[draw_idx]
            selected_positions[:, output_idx] .= last_position
            selected_masses[output_idx] = last_mass
            selected_redshifts[output_idx] = last_redshift
            selected_mstar[output_idx] = last_mstar
            selected_probabilities[output_idx] = last_probability
            selected_indices[output_idx] = last_index
            draw_idx += 1
        end
    end

    ras, decs = positions_to_ra_dec(selected_positions)
    source_pool = (
        stellar_mass_field=selected_field[],
        total_halo_count=total_halo_count[],
        candidate_count=candidate_count[],
        positive_weight_count=positive_weight_count[],
        redshift_min=z_min_seen[],
        redshift_max=z_max_seen[],
        mass_min=mass_min_seen[],
        mass_max=mass_max_seen[],
        mstar_min=mstar_min_seen[],
        mstar_max=mstar_max_seen[],
        n_eff=n_eff,
    )
    hosts = (
        local_choices=Int[],
        catalog_indices=selected_indices,
        masses=selected_masses,
        redshifts=selected_redshifts,
        mstar=selected_mstar,
        probabilities=selected_probabilities,
        ras=ras,
        decs=decs,
        p_shell=nothing,
    )
    return hosts, source_pool
end

function foreground_halo_mask(halo_mass, redshift; z_min::Float64, z_max::Float64, mass_min::Float64, mass_max::Float64)
    keep = isfinite.(redshift) .& isfinite.(halo_mass)
    keep .&= redshift .>= z_min
    isfinite(z_max) && (keep .&= redshift .< z_max)
    keep .&= halo_mass .> 0.0
    mass_min > 0.0 && (keep .&= halo_mass .>= mass_min)
    isfinite(mass_max) && (keep .&= halo_mass .< mass_max)
    return keep
end

function collect_foreground_halo_limits(
    catalog_path::AbstractString;
    z_min::Float64,
    z_max::Float64,
    mass_min::Float64,
    mass_max::Float64,
    chunkN::Int,
)
    total_halo_count = 0
    selected_count = 0
    z_min_seen = Inf
    z_max_seen = -Inf
    mass_min_seen = Inf
    mass_max_seen = -Inf

    h5open(catalog_path, "r") do h5
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halo_count = length(redshift_ds)

        for batch_start in 1:chunkN:total_halo_count
            batch_stop = min(batch_start + chunkN - 1, total_halo_count)
            idx = batch_start:batch_stop
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            keep = foreground_halo_mask(masses, redshifts; z_min=z_min, z_max=z_max, mass_min=mass_min, mass_max=mass_max)
            any(keep) || continue
            selected_mass = masses[keep]
            selected_redshift = redshifts[keep]
            selected_count += length(selected_mass)
            z_min_seen = min(z_min_seen, minimum(selected_redshift))
            z_max_seen = max(z_max_seen, maximum(selected_redshift))
            mass_min_seen = min(mass_min_seen, minimum(selected_mass))
            mass_max_seen = max(mass_max_seen, maximum(selected_mass))
        end
    end

    selected_count > 0 || error("No foreground halos passed cuts $(z_min) <= z < $(z_max).")
    return (
        total_halo_count=total_halo_count,
        selected_count=selected_count,
        z_min=z_min_seen,
        z_max=z_max_seen,
        mass_min=mass_min_seen,
        mass_max=mass_max_seen,
    )
end

function compute_theta_min_local(model)
    if hasproperty(model, :itp)
        itp = getproperty(model, :itp)
        if hasproperty(itp, :ranges)
            return exp(Float64(first(first(getproperty(itp, :ranges)))))
        end
    end
    return eps(Float64)
end

function build_frb_pixel_lookup(frb_pixels)
    order = sortperm(frb_pixels)
    return Int.(frb_pixels[order]), Int.(order)
end

function add_if_frb_pixel!(
    local_dm,
    local_hits::Base.RefValue{Int},
    sorted_frb_pixels,
    sorted_frb_indices,
    global_pix::Int,
    halo_ux::Float64,
    halo_uy::Float64,
    halo_uz::Float64,
    frb_ux,
    frb_uy,
    frb_uz,
    theta_min::Float64,
    theta_max::Float64,
    mass::Float64,
    redshift::Float64,
    dm_model_interp,
    ;
    frb_source_redshifts=nothing,
)
    frb_range = searchsorted(sorted_frb_pixels, global_pix)
    isempty(frb_range) && return nothing

    @inbounds for lookup_idx in frb_range
        frb_idx = sorted_frb_indices[lookup_idx]
        cosang = clamp(halo_ux * frb_ux[frb_idx] + halo_uy * frb_uy[frb_idx] + halo_uz * frb_uz[frb_idx], -1.0, 1.0)
        theta = acos(cosang)
        source_is_behind_halo = frb_source_redshifts === nothing || redshift < Float64(frb_source_redshifts[frb_idx])
        if source_is_behind_halo && theta <= theta_max
            local_dm[frb_idx] += Float64(dm_model_interp(max(theta, theta_min), mass, redshift))
            local_hits[] += 1
        end
    end
    return nothing
end

function accumulate_frb_dm_from_halo_batch!(
    frb_dm,
    workspace,
    dm_model_interp,
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
    ;
    frb_source_redshifts=nothing,
)
    isempty(masses) && return 0

    theta_min = compute_theta_min_local(dm_model_interp)
    nfrb = length(frb_dm)
    nthreads_capacity = thread_capacity()
    thread_dm = [zeros(Float64, nfrb) for _ in 1:nthreads_capacity]
    thread_hits = zeros(Int, nthreads_capacity)

    @threads for i in eachindex(masses)
        tid = Threads.threadid()
        local_dm = thread_dm[tid]
        local_hits = Ref(0)

        xi = Float64(x[i])
        yi = Float64(y[i])
        zi = Float64(z[i])
        mass_i = Float64(masses[i])
        redshift_i = Float64(redshifts[i])
        radius = sqrt(xi * xi + yi * yi + zi * zi)
        radius > 0.0 || continue

        halo_ux = xi / radius
        halo_uy = yi / radius
        halo_uz = zi / radius
        center_theta, center_phi = Healpix.vec2ang(halo_ux, halo_uy, halo_uz)

        theta_max = Float64(compute_theta_max_local(dm_model_interp, mass_i * XGPaint.M_sun, redshift_i))
        if !isfinite(theta_max) || theta_max <= 0.0
            continue
        end

        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, theta_max)
        for ring_idx in ring_start:ring_stop
            range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring_idx, center_theta, center_phi, theta_max)
            first_pixel = workspace.ring_first_pixels[ring_idx]
            for local_pix_idx in range1
                add_if_frb_pixel!(
                    local_dm,
                    local_hits,
                    sorted_frb_pixels,
                    sorted_frb_indices,
                    first_pixel + local_pix_idx - 1,
                    halo_ux,
                    halo_uy,
                    halo_uz,
                    frb_ux,
                    frb_uy,
                    frb_uz,
                    theta_min,
                    theta_max,
                    mass_i,
                    redshift_i,
                    dm_model_interp,
                    frb_source_redshifts=frb_source_redshifts,
                )
            end
            for local_pix_idx in range2
                add_if_frb_pixel!(
                    local_dm,
                    local_hits,
                    sorted_frb_pixels,
                    sorted_frb_indices,
                    first_pixel + local_pix_idx - 1,
                    halo_ux,
                    halo_uy,
                    halo_uz,
                    frb_ux,
                    frb_uy,
                    frb_uz,
                    theta_min,
                    theta_max,
                    mass_i,
                    redshift_i,
                    dm_model_interp,
                    frb_source_redshifts=frb_source_redshifts,
                )
            end
        end

        thread_hits[tid] += local_hits[]
    end

    for local_dm in thread_dm
        frb_dm .+= local_dm
    end
    return sum(thread_hits)
end

function accumulate_foreground_los_dm!(
    frb_dm,
    catalog_path::AbstractString,
    workspace,
    dm_model_interp,
    sorted_frb_pixels,
    sorted_frb_indices,
    frb_ux,
    frb_uy,
    frb_uz;
    z_min::Float64,
    z_max::Float64,
    mass_min::Float64,
    mass_max::Float64,
    chunkN::Int,
    frb_source_redshifts=nothing,
)
    processed_halo_count = 0
    los_intersection_count = 0

    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halo_count = size(pos_ds, 2)

        for batch_start in 1:chunkN:total_halo_count
            batch_stop = min(batch_start + chunkN - 1, total_halo_count)
            idx = batch_start:batch_stop
            positions = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            keep = foreground_halo_mask(masses, redshifts; z_min=z_min, z_max=z_max, mass_min=mass_min, mass_max=mass_max)
            any(keep) || continue

            xs = positions[1, keep]
            ys = positions[2, keep]
            zs = positions[3, keep]
            selected_masses = masses[keep]
            selected_redshifts = redshifts[keep]

            los_intersection_count += accumulate_frb_dm_from_halo_batch!(
                frb_dm,
                workspace,
                dm_model_interp,
                sorted_frb_pixels,
                sorted_frb_indices,
                frb_ux,
                frb_uy,
                frb_uz,
                xs,
                ys,
                zs,
                selected_masses,
                selected_redshifts,
                frb_source_redshifts=frb_source_redshifts,
            )
            processed_halo_count += length(selected_masses)
        end
    end

    return processed_halo_count, los_intersection_count
end

function paint_full_foreground_map!(
    dm_map,
    workspace,
    dm_model_interp,
    catalog_path::AbstractString;
    z_min::Float64,
    z_max::Float64,
    mass_min::Float64,
    mass_max::Float64,
    chunkN::Int,
    progress_every::Int,
)
    total_halo_count = 0
    halos_passing_cuts = 0
    halos_painted = 0
    z_min_seen = Inf
    z_max_seen = -Inf
    mass_min_seen = Inf
    mass_max_seen = -Inf

    paint_fn = xgpaint_paint_function()

    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halo_count = size(pos_ds, 2)

        chunk_index = 0
        for batch_start in 1:chunkN:total_halo_count
            chunk_index += 1
            batch_stop = min(batch_start + chunkN - 1, total_halo_count)
            idx = batch_start:batch_stop

            positions = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])
            keep = foreground_halo_mask(
                masses,
                redshifts;
                z_min=z_min,
                z_max=z_max,
                mass_min=mass_min,
                mass_max=mass_max,
            )

            selected_count = count(keep)
            selected_count == 0 && continue
            halos_passing_cuts += selected_count

            selected_positions = positions[:, keep]
            selected_masses = masses[keep]
            selected_redshifts = redshifts[keep]

            z_min_seen = min(z_min_seen, minimum(selected_redshifts))
            z_max_seen = max(z_max_seen, maximum(selected_redshifts))
            mass_min_seen = min(mass_min_seen, minimum(selected_masses))
            mass_max_seen = max(mass_max_seen, maximum(selected_masses))

            ras, decs = positions_to_ra_dec(selected_positions)
            perm = sortperm(decs)

            paint_fn(
                dm_map,
                workspace,
                dm_model_interp,
                selected_masses[perm],
                selected_redshifts[perm],
                ras[perm],
                decs[perm],
                zerobeforepainting=false,
            )

            halos_painted += selected_count
            if progress_every > 0 && (chunk_index % progress_every == 0 || batch_stop == total_halo_count)
                println(
                    "  foreground chunk $(chunk_index): catalog rows $(batch_start)-$(batch_stop), " *
                    "painted so far $(halos_painted)"
                )
                flush(stdout)
            end
        end
    end

    halos_painted > 0 || error("No foreground halos passed the requested cuts for the full foreground map.")
    return (
        total_halo_count=total_halo_count,
        halos_passing_cuts=halos_passing_cuts,
        halos_painted=halos_painted,
        redshift_min=z_min_seen,
        redshift_max=z_max_seen,
        mass_min=mass_min_seen,
        mass_max=mass_max_seen,
    )
end

function build_sparse_map_and_count(nside::Int, pixels, values; overlap_mode::AbstractString="mean")
    mode = lowercase(strip(String(overlap_mode)))
    mode in ("mean", "sum", "last") || error("frb_overlap_mode must be mean, sum, or last.")

    value_map = HealpixMap{Float64, RingOrder}(nside)
    count_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(value_map.pixels, 0.0)
    fill!(count_map.pixels, 0.0)

    @inbounds for i in eachindex(pixels)
        pix = Int(pixels[i])
        count_map.pixels[pix] += 1.0
        if mode == "last"
            value_map.pixels[pix] = Float64(values[i])
        else
            value_map.pixels[pix] += Float64(values[i])
        end
    end

    if mode == "mean"
        @inbounds for i in eachindex(value_map.pixels)
            count_map.pixels[i] > 0.0 && (value_map.pixels[i] /= count_map.pixels[i])
        end
    end

    return value_map, count_map
end

function write_host_catalog(path, hosts, frb_pixels, frb_dm, z_source)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(
            io,
            "frb_index,host_catalog_index,host_pixel_ring_1based,host_pixel_ring_0based_for_healpy,ra_rad,dec_rad,z_source,host_redshift,host_halo_mass_msun,host_stellar_mass_msun,selection_probability,dm_pc_cm3",
        )
        @inbounds for i in eachindex(frb_dm)
            pix = Int(frb_pixels[i])
            source_z = z_source isa AbstractVector ? Float64(z_source[i]) : Float64(z_source)
            println(
                io,
                "$(i),$(hosts.catalog_indices[i]),$(pix),$(pix - 1),$(hosts.ras[i]),$(hosts.decs[i]),$(source_z),$(hosts.redshifts[i]),$(hosts.masses[i]),$(hosts.mstar[i]),$(hosts.probabilities[i]),$(frb_dm[i])",
            )
        end
    end
    return path
end

function write_shell_probability_table(path, shell, p_shell)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "shell_index,catalog_index,redshift,halo_mass_msun,stellar_mass_msun,selection_probability")
        @inbounds for i in eachindex(p_shell)
            println(io, "$(i),$(shell.indices[i]),$(shell.redshifts[i]),$(shell.masses[i]),$(shell.mstar[i]),$(p_shell[i])")
        end
    end
    return path
end

function log_edges(values, nbins::Int)
    positive = Float64[v for v in values if isfinite(v) && v > 0.0]
    isempty(positive) && error("Need at least one positive finite value for a log histogram.")
    lo = minimum(positive)
    hi = maximum(positive)
    if lo == hi
        lo /= 10.0
        hi *= 10.0
    end
    return 10 .^ range(log10(lo), log10(hi); length=nbins + 1)
end

function histogram_pdf(values, edges)
    counts = zeros(Int, length(edges) - 1)
    @inbounds for value in values
        v = Float64(value)
        if isfinite(v) && v > 0.0 && v >= first(edges) && v <= last(edges)
            idx = searchsortedlast(edges, v)
            idx = min(max(idx, 1), length(edges) - 1)
            counts[idx] += 1
        end
    end

    total = sum(counts)
    total > 0 || error("Histogram has no positive finite values in the requested range.")
    widths = diff(edges)
    density = counts ./ (total .* widths)
    centers = sqrt.(edges[1:end-1] .* edges[2:end])
    return centers, density, counts
end

function save_dm_loglog_histogram(path, dm_values; nbins::Int=60)
    edges = log_edges(dm_values, nbins)
    centers, density, counts = histogram_pdf(dm_values, edges)
    keep = density .> 0.0

    p = plot(
        centers[keep],
        density[keep],
        xscale=:log10,
        yscale=:log10,
        xlabel="DM [pc cm^-3]",
        ylabel="p(DM)",
        label="FRB LOS DM",
        marker=:circle,
        linewidth=2,
        size=(760, 520),
        title="Stellar-mass-weighted FRB LOS DM PDF",
    )
    savefig(p, path)
    return path, counts
end

function save_stellar_mass_loglog_histogram(path, shell_mstar, selected_mstar; nbins::Int=60)
    edges = log_edges(vcat(shell_mstar, selected_mstar), nbins)
    centers_shell, density_shell, _ = histogram_pdf(shell_mstar, edges)
    centers_sel, density_sel, _ = histogram_pdf(selected_mstar, edges)
    keep_shell = density_shell .> 0.0
    keep_sel = density_sel .> 0.0

    p = plot(
        centers_shell[keep_shell],
        density_shell[keep_shell],
        xscale=:log10,
        yscale=:log10,
        xlabel="Mstar [Msun]",
        ylabel="PDF",
        label="All shell halos",
        linewidth=2,
        size=(760, 520),
        title="Stellar-mass-weighted host selection",
    )
    plot!(p, centers_sel[keep_sel], density_sel[keep_sel], label="Selected FRB hosts", linewidth=2)
    savefig(p, path)
    return path
end

function save_selected_stellar_mass_loglog_histogram(path, selected_mstar; nbins::Int=60)
    edges = log_edges(selected_mstar, nbins)
    centers, density, _ = histogram_pdf(selected_mstar, edges)
    keep = density .> 0.0

    p = plot(
        centers[keep],
        density[keep],
        xscale=:log10,
        yscale=:log10,
        xlabel="Mstar [Msun]",
        ylabel="PDF",
        label="Selected FRB hosts",
        linewidth=2,
        size=(760, 520),
        title="All-redshift stellar-mass-weighted hosts",
    )
    savefig(p, path)
    return path
end

function write_cl_table(path, cl_values)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "ell,C_ell,D_ell")
        @inbounds for i in eachindex(cl_values)
            ell = i - 1
            cl = Float64(cl_values[i])
            dl = ell * (ell + 1) * cl / (2.0 * pi)
            println(io, "$(ell),$(cl),$(dl)")
        end
    end

    return path
end

function save_power_spectrum_plot(path, cl_values)
    ell = collect(0:length(cl_values)-1)
    dl = ell .* (ell .+ 1) .* Float64.(cl_values) ./ (2.0 * pi)
    keep = (ell .>= 2) .& isfinite.(dl) .& (dl .> 0.0)
    count(keep) > 0 || error("No positive finite D_ell values available for a log-log power-spectrum plot.")

    p = plot(
        ell[keep],
        dl[keep],
        xscale=:log10,
        yscale=:log10,
        xlabel="ell",
        ylabel="D_ell = ell(ell+1)C_ell/2pi",
        label="observed sparse FRB LOS DM",
        linewidth=2,
        size=(760, 520),
        title="Observed FRB LOS DM power spectrum",
    )
    savefig(p, path)
    return path
end

function compute_and_save_power_spectrum(
    map::HealpixMap{Float64, RingOrder},
    nside::Int;
    table_path::AbstractString,
    plot_path::AbstractString,
    lmax::Int,
    niter::Int,
    subtract_mean::Bool,
)
    cl_map = HealpixMap{Float64, RingOrder}(nside)
    cl_map.pixels .= map.pixels
    if subtract_mean
        cl_map.pixels .-= mean(cl_map.pixels)
    end

    cl_values =
        lmax < 0 ?
        Healpix.anafast(cl_map; niter=niter) :
        Healpix.anafast(cl_map; lmax=lmax, niter=niter)

    write_cl_table(table_path, cl_values)
    save_power_spectrum_plot(plot_path, cl_values)
    return cl_values
end

function build_frb_sparse_estimator_map(
    frb_pixels,
    frb_dm,
    nside::Int;
    subtract_sample_mean::Bool,
)
    npix = 12 * nside^2
    valid = BitVector(undef, length(frb_dm))
    @inbounds for i in eachindex(frb_dm)
        pix = Int(frb_pixels[i])
        dm = Float64(frb_dm[i])
        valid[i] = isfinite(dm) && pix >= 1 && pix <= npix
    end

    nfrb = count(valid)
    nfrb > 0 || error("FRB sparse estimator has no finite valid FRB DM samples.")

    q = Vector{Float64}(undef, nfrb)
    valid_pixels = Vector{Int}(undef, nfrb)
    j = 1
    @inbounds for i in eachindex(frb_dm)
        if valid[i]
            valid_pixels[j] = Int(frb_pixels[i])
            q[j] = Float64(frb_dm[i])
            j += 1
        end
    end

    if subtract_sample_mean
        q .-= mean(q)
    end

    estimator_map = HealpixMap{Float64, RingOrder}(nside)
    fill!(estimator_map.pixels, 0.0)
    scale = npix / nfrb
    @inbounds for i in eachindex(q)
        estimator_map.pixels[valid_pixels[i]] += q[i] * scale
    end

    return estimator_map, valid_pixels, q
end

function write_frb_corrected_cl_table(path, cl_obs, cl_shot, cl_corr)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "ell,cl_obs,cl_shot,cl_corr,dl_obs,dl_corr")
        @inbounds for i in eachindex(cl_obs)
            ell = i - 1
            obs = Float64(cl_obs[i])
            shot = Float64(cl_shot[i])
            corr = Float64(cl_corr[i])
            dl_obs = ell * (ell + 1) * obs / (2.0 * pi)
            dl_corr = ell * (ell + 1) * corr / (2.0 * pi)
            println(io, "$(ell),$(obs),$(shot),$(corr),$(dl_obs),$(dl_corr)")
        end
    end

    return path
end

function save_frb_corrected_cl_plot(path, cl_obs, cl_corr)
    ell = collect(0:length(cl_obs)-1)
    dl_obs = ell .* (ell .+ 1) .* Float64.(cl_obs) ./ (2.0 * pi)
    dl_corr = ell .* (ell .+ 1) .* Float64.(cl_corr) ./ (2.0 * pi)
    keep_obs = (ell .>= 2) .& isfinite.(dl_obs) .& (dl_obs .> 0.0)
    keep_corr = (ell .>= 2) .& isfinite.(dl_corr) .& (dl_corr .> 0.0)
    (count(keep_obs) > 0 || count(keep_corr) > 0) ||
        error("No positive finite D_ell values available for FRB corrected estimator plot.")

    p = plot(
        xlabel="ell",
        ylabel="D_ell = ell(ell+1)C_ell/2pi",
        xscale=:log10,
        yscale=:log10,
        size=(760, 520),
        title="FRB LOS DM sparse estimator",
    )
    count(keep_obs) > 0 && plot!(p, ell[keep_obs], dl_obs[keep_obs], label="observed", linewidth=2)
    count(keep_corr) > 0 && plot!(p, ell[keep_corr], dl_corr[keep_corr], label="shot-noise corrected", linewidth=2)
    savefig(p, path)
    return path
end

function compute_and_save_frb_corrected_estimator(
    frb_pixels,
    frb_dm,
    nside::Int;
    table_path::AbstractString,
    plot_path::AbstractString,
    map_path::AbstractString,
    lmax::Int,
    subtract_sample_mean::Bool,
    shot_noise::AbstractString,
    n_shuffle::Int,
    seed::Int,
)
    shot_mode = lowercase(strip(String(shot_noise)))
    shot_mode in ("analytic", "shuffle", "none") ||
        error("frb_corrected_shot_noise must be analytic, shuffle, or none.")
    n_shuffle >= 0 || error("frb_corrected_n_shuffle must be non-negative.")

    estimator_map, valid_pixels, q = build_frb_sparse_estimator_map(
        frb_pixels,
        frb_dm,
        nside;
        subtract_sample_mean=subtract_sample_mean,
    )
    Healpix.saveToFITS(estimator_map, "!" * map_path, typechar="D")

    cl_obs =
        lmax < 0 ?
        Healpix.anafast(estimator_map; niter=0) :
        Healpix.anafast(estimator_map; lmax=lmax, niter=0)

    nfrb = length(q)
    nbar = nfrb / (4.0 * pi)
    cl_shot = zeros(Float64, length(cl_obs))

    if shot_mode == "analytic"
        cl_shot .= mean(q .^ 2) / nbar
    elseif shot_mode == "shuffle"
        n_shuffle > 0 || error("frb_corrected_n_shuffle must be positive when shot noise is shuffle.")
        rng = MersenneTwister(seed)
        npix = 12 * nside^2
        scale = npix / nfrb
        cl_sum = zeros(Float64, length(cl_obs))
        for _ in 1:n_shuffle
            q_shuf = q[randperm(rng, nfrb)]
            shuffle_map = HealpixMap{Float64, RingOrder}(nside)
            fill!(shuffle_map.pixels, 0.0)
            @inbounds for i in eachindex(q_shuf)
                shuffle_map.pixels[valid_pixels[i]] += q_shuf[i] * scale
            end
            cl_tmp =
                lmax < 0 ?
                Healpix.anafast(shuffle_map; niter=0) :
                Healpix.anafast(shuffle_map; lmax=lmax, niter=0)
            cl_sum .+= Float64.(cl_tmp)
        end
        cl_shot .= cl_sum ./ n_shuffle
    end

    cl_corr = Float64.(cl_obs) .- cl_shot
    length(cl_corr) >= 2 && (cl_corr[1:2] .= NaN)

    write_frb_corrected_cl_table(table_path, cl_obs, cl_shot, cl_corr)
    save_frb_corrected_cl_plot(plot_path, cl_obs, cl_corr)

    return (
        nfrb=nfrb,
        nbar_sr=nbar,
        q_mean=mean(q),
        q_var=mean(q .^ 2),
        map_path=map_path,
        table_path=table_path,
        plot_path=plot_path,
    )
end

function write_summary(
    path;
    catalog_path,
    output_dir,
    dm_cache_file,
    stellar_mass_field,
    stellar_mass_relation,
    stellar_mass_divide_by_h,
    source_selection_mode,
    source_z_min,
    source_z_max,
    source_halo_mass_min,
    source_halo_mass_max,
    nside,
    n_frb,
    seed,
    z_source,
    dz,
    alpha_star,
    eps,
    z_min_foreground,
    z_max_foreground,
    halo_mass_min,
    halo_mass_max,
    shell,
    foreground_limits,
    processed_halo_count,
    los_intersection_count,
    host_catalog_indices,
    frb_pixels,
    frb_dm,
    map_pixels,
    save_foreground_map,
    foreground_map_path,
    foreground_map_pixels,
    foreground_paint_counters,
    save_power_spectrum,
    cl_lmax,
    cl_niter,
    subtract_cl_mean,
    cl_table_path,
    cl_plot_path,
    save_frb_corrected_estimator,
    frb_corrected_lmax,
    frb_corrected_subtract_sample_mean,
    frb_corrected_shot_noise,
    frb_corrected_n_shuffle,
    frb_corrected_seed,
    frb_corrected_result,
    frb_corrected_table_path,
    frb_corrected_plot_path,
    frb_corrected_map_path,
)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)

    open(path, "w") do io
        println(io, "Stellar-mass-weighted FRB LOS DM map")
        println(io, "catalog_path=$(catalog_path)")
        println(io, "output_dir=$(output_dir)")
        println(io, "dm_cache_file=$(dm_cache_file)")
        println(io, "stellar_mass_field=$(stellar_mass_field)")
        println(io, "stellar_mass_relation=$(stellar_mass_relation)")
        println(io, "stellar_mass_divide_by_h=$(stellar_mass_divide_by_h)")
        println(io, "source_selection_mode=$(source_selection_mode)")
        println(io, "source_z_min=$(source_z_min)")
        println(io, "source_z_max=$(source_z_max)")
        println(io, "source_halo_mass_min=$(source_halo_mass_min)")
        println(io, "source_halo_mass_max=$(source_halo_mass_max)")
        if stellar_mass_field == COMPUTED_STELLAR_MASS_FIELD && stellar_mass_relation == "moster2013"
            println(io, "stellar_mass_relation=Mstar=2*N(z)*Mh/((Mh/M1(z))^(-beta(z))+(Mh/M1(z))^gamma(z))")
            println(io, "stellar_mass_relation_M10=11.590")
            println(io, "stellar_mass_relation_M11=1.195")
            println(io, "stellar_mass_relation_N10=0.0351")
            println(io, "stellar_mass_relation_N11=-0.0247")
            println(io, "stellar_mass_relation_beta10=1.376")
            println(io, "stellar_mass_relation_beta11=-0.826")
            println(io, "stellar_mass_relation_gamma10=0.608")
            println(io, "stellar_mass_relation_gamma11=0.329")
        elseif stellar_mass_field == COMPUTED_STELLAR_MASS_FIELD && stellar_mass_relation == "cosmos2020"
            println(io, "stellar_mass_relation_formula=COSMOS2020 central SHMR, inverted from Mvir to Mstar")
            println(io, "stellar_mass_relation_reference=Shuntov et al. 2022 / COSMOS2020")
            println(io, "stellar_mass_relation_input_mass=HalfDome halo_mass_m200c / h")
            println(io, "stellar_mass_relation_mass_conversion=M200c_to_Mvir_NFW")
            println(io, "stellar_mass_relation_delta_vir=BryanNorman1998_flat_LCDM_relative_to_critical")
            println(io, "stellar_mass_relation_concentration=Duffy2008_c200c_full_sample")
            println(io, "stellar_mass_relation_H0_reference=$(COSMOS2020_H0_REFERENCE)")
            println(io, "stellar_mass_relation_H0_target=$(COSMOS2020_H0_TARGET)")
            println(io, "stellar_mass_relation_log_mhalo_shift=$(COSMOS2020_LOG_MHALO_SHIFT)")
            println(io, "stellar_mass_relation_log_mstar_shift=$(COSMOS2020_LOG_MSTAR_SHIFT)")
        end
        println(io, "nside=$(nside)")
        println(io, "N=$(n_frb)")
        println(io, "seed=$(seed)")
        println(io, "z_source=$(z_source)")
        println(io, "dz=$(dz)")
        println(io, "alpha_star=$(alpha_star)")
        println(io, "eps=$(eps)")
        println(io, "foreground_z_min=$(z_min_foreground)")
        println(io, "foreground_z_max=$(z_max_foreground)")
        println(io, "halo_mass_min=$(halo_mass_min)")
        println(io, "halo_mass_max=$(halo_mass_max)")
        println(io)
        println(io, "total_halo_count=$(shell.total_halo_count)")
        host_candidate_count = hasproperty(shell, :candidate_count) ? shell.candidate_count : length(shell.indices)
        host_positive_weight_count = hasproperty(shell, :positive_weight_count) ? shell.positive_weight_count : host_candidate_count
        println(io, "host_candidate_count=$(host_candidate_count)")
        println(io, "host_positive_weight_count=$(host_positive_weight_count)")
        if hasproperty(shell, :redshift_min)
            println(io, "host_candidate_redshift_min=$(shell.redshift_min)")
            println(io, "host_candidate_redshift_max=$(shell.redshift_max)")
            println(io, "host_candidate_mass_min=$(shell.mass_min)")
            println(io, "host_candidate_mass_max=$(shell.mass_max)")
        end
        if hasproperty(shell, :mstar_min)
            println(io, "host_candidate_mstar_min=$(shell.mstar_min)")
            println(io, "host_candidate_mstar_max=$(shell.mstar_max)")
            println(io, "host_candidate_n_eff=$(shell.n_eff)")
        end
        println(io, "unique_selected_host_count=$(length(unique(host_catalog_indices)))")
        println(io, "unique_selected_pixel_count=$(length(unique(frb_pixels)))")
        println(io, "foreground_halos_passing_cuts=$(foreground_limits.selected_count)")
        println(io, "foreground_redshift_min=$(foreground_limits.z_min)")
        println(io, "foreground_redshift_max=$(foreground_limits.z_max)")
        println(io, "foreground_mass_min=$(foreground_limits.mass_min)")
        println(io, "foreground_mass_max=$(foreground_limits.mass_max)")
        println(io, "processed_foreground_halo_count=$(processed_halo_count)")
        println(io, "los_intersection_count=$(los_intersection_count)")
        println(io)
        println(io, "frb_dm_min=$(minimum(frb_dm))")
        println(io, "frb_dm_max=$(maximum(frb_dm))")
        println(io, "frb_dm_mean=$(mean(frb_dm))")
        println(io, "frb_dm_std=$(std(frb_dm))")
        println(io, "map_nonzero_pixels=$(count(!=(0.0), map_pixels))")
        println(io, "map_min=$(minimum(map_pixels))")
        println(io, "map_max=$(maximum(map_pixels))")
        println(io)
        println(io, "save_foreground_map=$(save_foreground_map)")
        println(io, "foreground_map_path=$(foreground_map_path)")
        if foreground_map_pixels !== nothing
            println(io, "foreground_map_nonzero_pixels=$(count(!=(0.0), foreground_map_pixels))")
            println(io, "foreground_map_min=$(minimum(foreground_map_pixels))")
            println(io, "foreground_map_max=$(maximum(foreground_map_pixels))")
            println(io, "foreground_map_mean=$(mean(foreground_map_pixels))")
        end
        if foreground_paint_counters !== nothing
            println(io, "foreground_map_total_halo_count=$(foreground_paint_counters.total_halo_count)")
            println(io, "foreground_map_halos_passing_cuts=$(foreground_paint_counters.halos_passing_cuts)")
            println(io, "foreground_map_halos_painted=$(foreground_paint_counters.halos_painted)")
            println(io, "foreground_map_redshift_min=$(foreground_paint_counters.redshift_min)")
            println(io, "foreground_map_redshift_max=$(foreground_paint_counters.redshift_max)")
            println(io, "foreground_map_mass_min=$(foreground_paint_counters.mass_min)")
            println(io, "foreground_map_mass_max=$(foreground_paint_counters.mass_max)")
        end
        println(io)
        println(io, "save_power_spectrum=$(save_power_spectrum)")
        println(io, "power_spectrum_input=continuous_foreground_dm_map")
        println(io, "cl_lmax=$(cl_lmax)")
        println(io, "cl_niter=$(cl_niter)")
        println(io, "subtract_cl_mean=$(subtract_cl_mean)")
        println(io, "cl_table_path=$(cl_table_path)")
        println(io, "cl_plot_path=$(cl_plot_path)")
        println(io)
        println(io, "save_frb_corrected_estimator=$(save_frb_corrected_estimator)")
        println(io, "frb_corrected_lmax=$(frb_corrected_lmax)")
        println(io, "frb_corrected_subtract_sample_mean=$(frb_corrected_subtract_sample_mean)")
        println(io, "frb_corrected_shot_noise=$(frb_corrected_shot_noise)")
        println(io, "frb_corrected_n_shuffle=$(frb_corrected_n_shuffle)")
        println(io, "frb_corrected_seed=$(frb_corrected_seed)")
        println(io, "frb_corrected_table_path=$(frb_corrected_table_path)")
        println(io, "frb_corrected_plot_path=$(frb_corrected_plot_path)")
        println(io, "frb_corrected_map_path=$(frb_corrected_map_path)")
        if frb_corrected_result !== nothing
            println(io, "frb_corrected_nfrb=$(frb_corrected_result.nfrb)")
            println(io, "frb_corrected_nbar_sr=$(frb_corrected_result.nbar_sr)")
            println(io, "frb_corrected_q_mean=$(frb_corrected_result.q_mean)")
            println(io, "frb_corrected_q_var=$(frb_corrected_result.q_var)")
        end
    end

    return path
end

function main()
    catalog_path = resolve_halfdome_catalog_path(get_string_arg(
        "halfdome_path",
        "lightcone_100.hdf5";
        env=("STELLAR_FRB_HALFDOME_PATH", "FRB_HALFDOME_PATH"),
    ))
    output_dir = resolve_project_path(get_string_arg(
        "output_dir",
        joinpath("frb_map_generation", "outputs", "stellar_weighted_frb_los_dm_z1");
        env="STELLAR_FRB_OUTPUT_DIR",
    ))
    isdir(output_dir) || mkpath(output_dir)

    nside = get_int_arg("nside", 4096; env="STELLAR_FRB_NSIDE")
    n_frb = get_int_arg("N", 10_000; env=("STELLAR_FRB_N", "FRB_N"))
    seed = get_int_arg("seed", 42; env=("STELLAR_FRB_SEED", "FRB_SEED"))
    chunkN = get_int_arg("chunkN", 1_000_000; env="STELLAR_FRB_CHUNKN")
    z_source_default = get_float_arg("source_redshift", DEFAULT_SOURCE_REDSHIFT; env=("STELLAR_FRB_SOURCE_REDSHIFT", "FRB_SOURCE_REDSHIFT"))
    z_source = get_float_arg("z_source", z_source_default; env=("STELLAR_FRB_Z_SOURCE", "FRB_Z_SOURCE"))
    dz = get_float_arg("dz", 0.02; env="STELLAR_FRB_DZ")
    source_selection_mode = lowercase(get_string_arg("source_selection_mode", "shell"; env="STELLAR_FRB_SOURCE_SELECTION_MODE"))
    source_mode_is_all = source_selection_mode in ("all", "all_redshifts", "allredshifts", "full")
    source_selection_mode = source_mode_is_all ? "all" : source_selection_mode
    source_z_min = get_float_arg("source_z_min", source_mode_is_all ? 0.0 : max(0.0, z_source - 0.5 * dz); env="STELLAR_FRB_SOURCE_Z_MIN")
    source_z_max = get_float_arg("source_z_max", source_mode_is_all ? Inf : z_source + 0.5 * dz; env="STELLAR_FRB_SOURCE_Z_MAX")
    source_halo_mass_min = get_float_arg("source_halo_mass_min", 0.0; env="STELLAR_FRB_SOURCE_MASS_MIN")
    source_halo_mass_max = get_float_arg("source_halo_mass_max", Inf; env="STELLAR_FRB_SOURCE_MASS_MAX")
    alpha_star = get_float_arg("alpha_star", 1.0; env="STELLAR_FRB_ALPHA_STAR")
    eps = get_float_arg("eps", 1.0e-30; env="STELLAR_FRB_EPS")
    stellar_mass_field = get_string_arg("stellar_mass_field", "auto"; env="STELLAR_FRB_MSTAR_FIELD")
    stellar_mass_relation = normalize_stellar_mass_relation(get_string_arg(
        "stellar_mass_relation",
        DEFAULT_STELLAR_MASS_RELATION;
        env="STELLAR_FRB_MSTAR_RELATION",
    ))
    if lowercase(strip(stellar_mass_field)) in ("cosmos", "cosmos2020", "shuntov", "shuntov2022")
        stellar_mass_relation = "cosmos2020"
    end
    stellar_mass_divide_by_h = get_bool_arg("stellar_mass_divide_by_h", false; env="STELLAR_FRB_MSTAR_DIVIDE_BY_H")
    z_min_foreground = get_float_arg("z_min_foreground", 0.0; env="STELLAR_FRB_FOREGROUND_Z_MIN")
    z_max_foreground = get_float_arg("z_max_foreground", source_mode_is_all ? Inf : z_source; env="STELLAR_FRB_FOREGROUND_Z_MAX")
    halo_mass_min = get_float_arg("halo_mass_min", 0.0; env="STELLAR_FRB_FOREGROUND_MASS_MIN")
    halo_mass_max = get_float_arg("halo_mass_max", Inf; env="STELLAR_FRB_FOREGROUND_MASS_MAX")
    frb_overlap_mode = lowercase(get_string_arg("frb_overlap_mode", "mean"; env="STELLAR_FRB_OVERLAP_MODE"))
    save_shell_probabilities = get_bool_arg("save_shell_probabilities", false; env="STELLAR_FRB_SAVE_SHELL_PROBABILITIES")
    dm_value_sanity_max = get_float_arg("dm_value_sanity_max", 1.0e8; env="STELLAR_FRB_DM_VALUE_SANITY_MAX")
    dm_cleanup_nonpositive = get_bool_arg("dm_cleanup_nonpositive", true; env="STELLAR_FRB_DM_CLEANUP_NONPOSITIVE")
    dm_cache_file = resolve_project_path(get_string_arg(
        "dm_cache_file",
        joinpath(output_dir, "stellar_weighted_frb_los_dm_profile_cache.jld2");
        env="STELLAR_FRB_DM_CACHE_FILE",
    ))
    dm_cache_overwrite = get_bool_arg("dm_cache_overwrite", false; env="STELLAR_FRB_DM_CACHE_OVERWRITE")
    dm_hist_bins = get_int_arg("dm_hist_bins", 60; env="STELLAR_FRB_DM_HIST_BINS")
    save_foreground_map = get_bool_arg("save_foreground_map", true; env="STELLAR_FRB_SAVE_FOREGROUND_MAP")
    foreground_progress_every = get_int_arg("foreground_progress_every", 5; env="STELLAR_FRB_FOREGROUND_PROGRESS_EVERY")
    save_power_spectrum = get_bool_arg("save_power_spectrum", true; env="STELLAR_FRB_SAVE_POWER_SPECTRUM")
    cl_lmax = get_int_arg("cl_lmax", 3 * nside - 1; env="STELLAR_FRB_CL_LMAX")
    cl_niter = get_int_arg("cl_niter", 0; env="STELLAR_FRB_CL_NITER")
    subtract_cl_mean = get_bool_arg("subtract_cl_mean", true; env="STELLAR_FRB_SUBTRACT_CL_MEAN")
    save_frb_corrected_estimator = get_bool_arg("save_frb_corrected_estimator", true; env="STELLAR_FRB_SAVE_CORRECTED_ESTIMATOR")
    frb_corrected_lmax = get_int_arg("frb_corrected_lmax", cl_lmax; env="STELLAR_FRB_CORRECTED_LMAX")
    frb_corrected_subtract_sample_mean = get_bool_arg("frb_corrected_subtract_sample_mean", true; env="STELLAR_FRB_CORRECTED_SUBTRACT_SAMPLE_MEAN")
    frb_corrected_shot_noise = lowercase(get_string_arg("frb_corrected_shot_noise", "shuffle"; env="STELLAR_FRB_CORRECTED_SHOT_NOISE"))
    frb_corrected_n_shuffle = get_int_arg("frb_corrected_n_shuffle", 5; env="STELLAR_FRB_CORRECTED_N_SHUFFLE")
    frb_corrected_seed = get_int_arg("frb_corrected_seed", seed; env="STELLAR_FRB_CORRECTED_SEED")

    nside > 0 || error("nside must be positive.")
    n_frb > 0 || error("N must be positive.")
    chunkN > 0 || error("chunkN must be positive.")
    z_source > 0.0 || error("z_source must be positive.")
    dz > 0.0 || error("dz must be positive.")
    source_selection_mode in ("shell", "all") || error("source_selection_mode must be shell or all.")
    source_z_min >= 0.0 || error("source_z_min must be non-negative.")
    (isfinite(source_z_max) && source_z_max > source_z_min) || !isfinite(source_z_max) ||
        error("source_z_max must be greater than source_z_min or Inf.")
    source_halo_mass_min >= 0.0 || error("source_halo_mass_min must be non-negative.")
    source_halo_mass_max > source_halo_mass_min || error("source_halo_mass_max must be greater than source_halo_mass_min.")
    alpha_star >= 0.0 && isfinite(alpha_star) || error("alpha_star must be finite and >= 0.")
    eps >= 0.0 && isfinite(eps) || error("eps must be finite and >= 0.")
    z_min_foreground >= 0.0 || error("z_min_foreground must be non-negative.")
    source_mode_is_all || z_max_foreground <= z_source || error("z_max_foreground must not exceed z_source in shell mode.")
    z_max_foreground > z_min_foreground || error("z_max_foreground must be > z_min_foreground.")
    halo_mass_min >= 0.0 || error("halo_mass_min must be non-negative.")
    halo_mass_max > halo_mass_min || error("halo_mass_max must be greater than halo_mass_min.")
    frb_overlap_mode in ("mean", "sum", "last") || error("frb_overlap_mode must be mean, sum, or last.")
    dm_hist_bins >= 2 || error("dm_hist_bins must be >= 2.")
    foreground_progress_every >= 0 || error("foreground_progress_every must be non-negative.")
    cl_lmax == -1 || cl_lmax >= 2 || error("cl_lmax must be -1 for Healpix default or >= 2.")
    cl_niter >= 0 || error("cl_niter must be non-negative.")
    frb_corrected_lmax == -1 || frb_corrected_lmax >= 2 || error("frb_corrected_lmax must be -1 for Healpix default or >= 2.")
    frb_corrected_shot_noise in ("analytic", "shuffle", "none") || error("frb_corrected_shot_noise must be analytic, shuffle, or none.")
    frb_corrected_n_shuffle >= 0 || error("frb_corrected_n_shuffle must be non-negative.")
    frb_corrected_shot_noise == "shuffle" && frb_corrected_n_shuffle == 0 &&
        error("frb_corrected_n_shuffle must be positive when frb_corrected_shot_noise=shuffle.")

    stellar_mass_relation == "cosmos2020" && ensure_cosmos2020_table!()

    source_tag =
        source_mode_is_all ?
        "allredshifts_zsrcmin$(fmt_param_value(source_z_min))_zsrcmax$(fmt_param_value(source_z_max))" :
        "zsource$(fmt_param_value(z_source))_dz$(fmt_param_value(dz))"
    relation_tag = stellar_mass_relation == DEFAULT_STELLAR_MASS_RELATION ? "" : "_$(stellar_mass_relation)"
    tag = "stellar_weighted_frb_los_$(source_tag)$(relation_tag)_alpha$(fmt_param_value(alpha_star))_nside$(nside)_nfrb$(n_frb)_seed$(seed)"
    foreground_map_path = joinpath(output_dir, "$(tag)_foreground_dm_map.fits")
    los_map_path = joinpath(output_dir, "$(tag)_dm_map.fits")
    count_map_path = joinpath(output_dir, "$(tag)_count_map.fits")
    host_catalog_path = joinpath(output_dir, "$(tag)_hosts.csv")
    shell_prob_path = joinpath(output_dir, "$(tag)_shell_probabilities.csv")
    summary_path = joinpath(output_dir, "$(tag)_summary.txt")
    dm_hist_path = joinpath(output_dir, "$(tag)_dm_pdf_loglog.png")
    mstar_hist_path = joinpath(output_dir, "$(tag)_stellar_mass_host_histogram_loglog.png")
    cl_table_path = joinpath(output_dir, "$(tag)_foreground_dm_power_spectrum.csv")
    cl_plot_path = joinpath(output_dir, "$(tag)_foreground_dm_power_spectrum_loglog.png")
    frb_corrected_table_path = joinpath(output_dir, "$(tag)_frb_corrected_estimator_power_spectrum.csv")
    frb_corrected_plot_path = joinpath(output_dir, "$(tag)_frb_corrected_estimator_power_spectrum_loglog.png")
    frb_corrected_map_path = joinpath(output_dir, "$(tag)_frb_corrected_estimator_map.fits")

    println("Stellar-mass-weighted FRB LOS DM configuration:")
    println("  catalog_path=$(catalog_path)")
    println("  output_dir=$(output_dir)")
    println("  nside=$(nside), N=$(n_frb), seed=$(seed), chunkN=$(chunkN)")
    println("  source_selection_mode=$(source_selection_mode)")
    if source_mode_is_all
        println("  source host cuts: $(source_z_min) <= z_halo < $(source_z_max), mass in [$(source_halo_mass_min), $(source_halo_mass_max))")
    else
        println("  host shell: abs(z_halo - $(z_source)) < $(0.5 * dz)")
    end
    println("  stellar_mass_field=$(stellar_mass_field), stellar_mass_relation=$(stellar_mass_relation), stellar_mass_divide_by_h=$(stellar_mass_divide_by_h)")
    println("  alpha_star=$(alpha_star), eps=$(eps)")
    println("  foreground halo cut: $(z_min_foreground) <= z_halo < $(z_max_foreground), mass in [$(halo_mass_min), $(halo_mass_max))")
    println("  dm_cache_file=$(dm_cache_file), dm_cache_overwrite=$(dm_cache_overwrite)")
    println("  save_foreground_map=$(save_foreground_map), foreground_progress_every=$(foreground_progress_every)")
    println("  save_power_spectrum=$(save_power_spectrum), input=continuous foreground map, cl_lmax=$(cl_lmax), cl_niter=$(cl_niter), subtract_cl_mean=$(subtract_cl_mean)")
    println("  save_frb_corrected_estimator=$(save_frb_corrected_estimator), lmax=$(frb_corrected_lmax), shot_noise=$(frb_corrected_shot_noise), n_shuffle=$(frb_corrected_n_shuffle)")

    shell = nothing
    hosts = nothing
    if source_mode_is_all
        println("Sampling FRB hosts over all selected HalfDome redshifts with p proportional to Mstar^alpha_star...")
        hosts, shell = sample_stellar_mass_weighted_hosts_all_redshifts(
            catalog_path;
            stellar_mass_field=stellar_mass_field,
            stellar_mass_relation=stellar_mass_relation,
            n_frb=n_frb,
            alpha_star=alpha_star,
            eps=eps,
            seed=seed,
            chunkN=chunkN,
            stellar_mass_divide_by_h=stellar_mass_divide_by_h,
            source_z_min=source_z_min,
            source_z_max=source_z_max,
            source_halo_mass_min=source_halo_mass_min,
            source_halo_mass_max=source_halo_mass_max,
        )
        println("  detected stellar_mass_field=$(shell.stellar_mass_field)")
        println("  all-redshift host candidates=$(shell.candidate_count)")
    else
        println("Collecting source-shell host candidates...")
        shell = collect_shell_host_candidates(
            catalog_path;
            stellar_mass_field=stellar_mass_field,
            stellar_mass_relation=stellar_mass_relation,
            z_source=z_source,
            dz=dz,
            chunkN=chunkN,
            stellar_mass_divide_by_h=stellar_mass_divide_by_h,
        )
        println("  detected stellar_mass_field=$(shell.stellar_mass_field)")
        println("  host shell halos=$(length(shell.indices))")

        println("Sampling FRB hosts with p proportional to Mstar^alpha_star...")
        hosts = sample_stellar_mass_weighted_hosts(
            shell;
            n_frb=n_frb,
            alpha_star=alpha_star,
            eps=eps,
            seed=seed,
        )
    end

    res = Healpix.Resolution(nside)
    frb_pixels = ra_dec_to_ring_pixels(res, hosts.ras, hosts.decs)
    frb_ux, frb_uy, frb_uz = ra_dec_to_unit_vectors(hosts.ras, hosts.decs)
    sorted_frb_pixels, sorted_frb_indices = build_frb_pixel_lookup(frb_pixels)
    frb_dm = zeros(Float64, n_frb)
    println("Selected FRB host pixels: $(length(unique(frb_pixels))) unique pixels for $(n_frb) FRBs.")

    println("Scanning foreground halo limits...")
    foreground_limits = collect_foreground_halo_limits(
        catalog_path;
        z_min=z_min_foreground,
        z_max=z_max_foreground,
        mass_min=halo_mass_min,
        mass_max=halo_mass_max,
        chunkN=chunkN,
    )
    println("  foreground halos passing cuts=$(foreground_limits.selected_count)")
    println("  foreground redshift range=[$(foreground_limits.z_min), $(foreground_limits.z_max)]")

    dm_cache_dir = dirname(dm_cache_file)
    isempty(dm_cache_dir) || isdir(dm_cache_dir) || mkpath(dm_cache_dir)
    ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = dm_cleanup_nonpositive ? "true" : "false"
    dm_model = make_dm_model()
    dm_model_interp = xgpaint_build_interpolator_function()(
        dm_model;
        cache_file=dm_cache_file,
        overwrite=dm_cache_overwrite,
    )
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)

    println("Accumulating LOS DM only at selected FRB host sightlines...")
    processed_halo_count, los_intersection_count = accumulate_foreground_los_dm!(
        frb_dm,
        catalog_path,
        workspace,
        dm_model_interp,
        sorted_frb_pixels,
        sorted_frb_indices,
        frb_ux,
        frb_uy,
        frb_uz;
        z_min=z_min_foreground,
        z_max=z_max_foreground,
        mass_min=halo_mass_min,
        mass_max=halo_mass_max,
        chunkN=chunkN,
        frb_source_redshifts=source_mode_is_all ? hosts.redshifts : nothing,
    )

    max_dm = maximum(frb_dm)
    if !isfinite(max_dm) || max_dm > dm_value_sanity_max
        error(
            "Sampled FRB DM maximum $(max_dm) is not physically plausible. " *
            "Check the XGPaint DM profile cache and unit conventions."
        )
    end

    println("Processed $(processed_halo_count) foreground halos.")
    println("Found $(los_intersection_count) foreground halo / FRB LOS intersections.")
    println("FRB LOS DM summary: min=$(minimum(frb_dm)), max=$(maximum(frb_dm)), mean=$(mean(frb_dm)), std=$(std(frb_dm))")

    los_dm_map, count_map = build_sparse_map_and_count(nside, frb_pixels, frb_dm; overlap_mode=frb_overlap_mode)
    println("Writing observed sparse LOS DM FITS map:")
    println("  $(los_map_path)")
    Healpix.saveToFITS(los_dm_map, "!" * los_map_path, typechar="D")
    println("Writing FRB count FITS map:")
    println("  $(count_map_path)")
    Healpix.saveToFITS(count_map, "!" * count_map_path, typechar="D")

    write_host_catalog(host_catalog_path, hosts, frb_pixels, frb_dm, source_mode_is_all ? hosts.redshifts : z_source)
    println("Wrote FRB host catalog:")
    println("  $(host_catalog_path)")

    if save_shell_probabilities && !source_mode_is_all
        write_shell_probability_table(shell_prob_path, shell, hosts.p_shell)
        println("Wrote source-shell probability table:")
        println("  $(shell_prob_path)")
    elseif save_shell_probabilities && source_mode_is_all
        println("save_shell_probabilities=true ignored in source_selection_mode=all; full-catalog probabilities are not materialized.")
    end

    save_dm_loglog_histogram(dm_hist_path, frb_dm; nbins=dm_hist_bins)
    println("Saved DM loglog PDF histogram:")
    println("  $(dm_hist_path)")
    if source_mode_is_all
        save_selected_stellar_mass_loglog_histogram(mstar_hist_path, hosts.mstar; nbins=dm_hist_bins)
    else
        save_stellar_mass_loglog_histogram(mstar_hist_path, shell.mstar, hosts.mstar; nbins=dm_hist_bins)
    end
    println("Saved stellar-mass host-selection histogram:")
    println("  $(mstar_hist_path)")

    frb_corrected_result = nothing
    if save_frb_corrected_estimator
        println("Computing corrected sparse FRB LOS DM estimator...")
        frb_corrected_result = compute_and_save_frb_corrected_estimator(
            frb_pixels,
            frb_dm,
            nside;
            table_path=frb_corrected_table_path,
            plot_path=frb_corrected_plot_path,
            map_path=frb_corrected_map_path,
            lmax=frb_corrected_lmax,
            subtract_sample_mean=frb_corrected_subtract_sample_mean,
            shot_noise=frb_corrected_shot_noise,
            n_shuffle=frb_corrected_n_shuffle,
            seed=frb_corrected_seed,
        )
        println("Saved corrected FRB estimator map:")
        println("  $(frb_corrected_map_path)")
        println("Saved corrected FRB estimator power-spectrum table:")
        println("  $(frb_corrected_table_path)")
        println("Saved corrected FRB estimator loglog plot:")
        println("  $(frb_corrected_plot_path)")
    end

    foreground_dm_map = nothing
    foreground_paint_counters = nothing
    if save_foreground_map || save_power_spectrum
        println("Painting continuous foreground DM map for full-sky products...")
        foreground_dm_map = HealpixMap{Float64, RingOrder}(nside)
        fill!(foreground_dm_map.pixels, 0.0)
        foreground_paint_counters = paint_full_foreground_map!(
            foreground_dm_map,
            workspace,
            dm_model_interp,
            catalog_path;
            z_min=z_min_foreground,
            z_max=z_max_foreground,
            mass_min=halo_mass_min,
            mass_max=halo_mass_max,
            chunkN=chunkN,
            progress_every=foreground_progress_every,
        )
        println(
            "Continuous foreground DM map summary: min=$(minimum(foreground_dm_map.pixels)), " *
            "max=$(maximum(foreground_dm_map.pixels)), mean=$(mean(foreground_dm_map.pixels)), " *
            "nonzero=$(count(!=(0.0), foreground_dm_map.pixels))"
        )

        if save_foreground_map
            println("Writing continuous foreground DM FITS map:")
            println("  $(foreground_map_path)")
            Healpix.saveToFITS(foreground_dm_map, "!" * foreground_map_path, typechar="D")
        end
    end

    if save_power_spectrum
        foreground_dm_map === nothing && error("save_power_spectrum=true requires a painted foreground DM map.")
        println("Computing power spectrum from continuous foreground DM map...")
        compute_and_save_power_spectrum(
            foreground_dm_map,
            nside;
            table_path=cl_table_path,
            plot_path=cl_plot_path,
            lmax=cl_lmax,
            niter=cl_niter,
            subtract_mean=subtract_cl_mean,
        )
        println("Saved foreground DM power-spectrum table:")
        println("  $(cl_table_path)")
        println("Saved foreground DM power-spectrum loglog plot:")
        println("  $(cl_plot_path)")
    end

    write_summary(
        summary_path;
        catalog_path=catalog_path,
        output_dir=output_dir,
        dm_cache_file=dm_cache_file,
        stellar_mass_field=shell.stellar_mass_field,
        stellar_mass_relation=stellar_mass_relation,
        stellar_mass_divide_by_h=stellar_mass_divide_by_h,
        source_selection_mode=source_selection_mode,
        source_z_min=source_z_min,
        source_z_max=source_z_max,
        source_halo_mass_min=source_halo_mass_min,
        source_halo_mass_max=source_halo_mass_max,
        nside=nside,
        n_frb=n_frb,
        seed=seed,
        z_source=z_source,
        dz=dz,
        alpha_star=alpha_star,
        eps=eps,
        z_min_foreground=z_min_foreground,
        z_max_foreground=z_max_foreground,
        halo_mass_min=halo_mass_min,
        halo_mass_max=halo_mass_max,
        shell=shell,
        foreground_limits=foreground_limits,
        processed_halo_count=processed_halo_count,
        los_intersection_count=los_intersection_count,
        host_catalog_indices=hosts.catalog_indices,
        frb_pixels=frb_pixels,
        frb_dm=frb_dm,
        map_pixels=los_dm_map.pixels,
        save_foreground_map=save_foreground_map,
        foreground_map_path=foreground_map_path,
        foreground_map_pixels=foreground_dm_map === nothing ? nothing : foreground_dm_map.pixels,
        foreground_paint_counters=foreground_paint_counters,
        save_power_spectrum=save_power_spectrum,
        cl_lmax=cl_lmax,
        cl_niter=cl_niter,
        subtract_cl_mean=subtract_cl_mean,
        cl_table_path=cl_table_path,
        cl_plot_path=cl_plot_path,
        save_frb_corrected_estimator=save_frb_corrected_estimator,
        frb_corrected_lmax=frb_corrected_lmax,
        frb_corrected_subtract_sample_mean=frb_corrected_subtract_sample_mean,
        frb_corrected_shot_noise=frb_corrected_shot_noise,
        frb_corrected_n_shuffle=frb_corrected_n_shuffle,
        frb_corrected_seed=frb_corrected_seed,
        frb_corrected_result=frb_corrected_result,
        frb_corrected_table_path=frb_corrected_table_path,
        frb_corrected_plot_path=frb_corrected_plot_path,
        frb_corrected_map_path=frb_corrected_map_path,
    )
    println("Wrote summary:")
    println("  $(summary_path)")
    println("Done.")
end

main()
