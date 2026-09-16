if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "png"
end
if !haskey(ENV, "GKS_WSTYPE")
    ENV["GKS_WSTYPE"] = "png"
end

# HalfDome catalog-halo FRB DM diagnostic.
#
# This checks whether a real HalfDome catalog halo gives the same DM(R_perp)
# as the isolated XGPaint profile when the catalog mass/redshift are used.
# It also sums all foreground catalog halos along the same sightlines so any
# excess above the selected-halo curve is visible.

using XGPaint
using HDF5
using Healpix
using Unitful
using UnitfulAstro
using Plots
using LinearAlgebra

const H_VALUE = 0.68
const OMEGAB = 0.049
const OMEGAC = 0.31 - OMEGAB
const OMEGAM = OMEGAB + OMEGAC
const ELECTRON_MASS = 9.1093837015e-28u"g"
const PROTON_MASS = 1.67262192369e-24u"g"
const GRAVITATIONAL_CONSTANT = 6.67430e-11u"m^3/kg/s^2"

const compute_theta_max_local =
    isdefined(XGPaint, Symbol("compute_", Char(0x03b8), "max")) ?
    getfield(XGPaint, Symbol("compute_", Char(0x03b8), "max")) :
    error("XGPaint does not define compute_theta_max.")

const xg_R_delta =
    isdefined(XGPaint, Symbol("R_", Char(0x0394))) ?
    getfield(XGPaint, Symbol("R_", Char(0x0394))) :
    error("XGPaint does not define R_delta.")

const SYM_ALPHA = Symbol(Char(0x03b1))
const SYM_BETA = Symbol(Char(0x03b2))
const SYM_GAMMA = Symbol(Char(0x03b3))
const SYM_P0 = Symbol("P", Char(0x2080))

struct CatalogHalo
    index::Int
    position::Vector{Float64}
    raw_mass_msun_h::Float64
    mass_msun::Float64
    redshift::Float64
    target_mass_msun::Float64
end

function code_root()
    return @__DIR__
end

function project_root()
    return basename(code_root()) == "frb_map_generation" ? dirname(code_root()) : code_root()
end

function resolve_project_path(path::AbstractString)
    isempty(path) && return String(path)
    return isabspath(path) ? String(path) : normpath(joinpath(project_root(), path))
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

function parse_float_list(text::AbstractString)
    values = Float64[]
    for token in split(text, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    isempty(values) && error("Need at least one value in $(repr(text)).")
    return values
end

function halo_dm_constructor()
    if isdefined(Main, :HaloDMProfile)
        return getfield(Main, :HaloDMProfile)
    elseif isdefined(XGPaint, :HaloDMProfile)
        return getfield(XGPaint, :HaloDMProfile)
    end
    error("HaloDMProfile is not available in this Julia/XGPaint environment.")
end

function xgpaint_paint_function()
    if isdefined(Main, :paint!)
        return getfield(Main, :paint!)
    elseif isdefined(XGPaint, :paint!)
        return getfield(XGPaint, :paint!)
    end
    error("paint! is not available in this Julia/XGPaint environment.")
end

function xgpaint_build_interpolator_function()
    if isdefined(Main, :build_interpolator)
        return getfield(Main, :build_interpolator)
    elseif isdefined(XGPaint, :build_interpolator)
        return getfield(XGPaint, :build_interpolator)
    end
    error("build_interpolator is not available in this Julia/XGPaint environment.")
end

function make_dm_model()
    constructor = halo_dm_constructor()
    tau_model = XGPaint.BattagliaTauProfile(Omega_c=OMEGAC, Omega_b=OMEGAB, h=H_VALUE)
    return constructor(tau_model)
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

function r200_kpc(profile_model, mass_msun::Real, z::Real)
    r200 = xg_R_delta(profile_model, Float64(mass_msun) * XGPaint.M_sun, Float64(z), 200)
    return Float64(ustrip(u"kpc", r200))
end

function rho_crit_hminus2_g_cm3(z::Real)
    h100 = 100.0u"km/s/Mpc"
    e2 = OMEGAM * (1.0 + Float64(z))^3 + (1.0 - OMEGAM)
    rho = 3.0 * h100^2 * e2 / (8.0 * pi * GRAVITATIONAL_CONSTANT)
    return uconvert(u"g/cm^3", rho)
end

function xgpaint_ne3d_cm3(profile_model, r_kpc::Real, mass_msun::Real, z::Real)
    mass_unitful = Float64(mass_msun) * XGPaint.M_sun
    r200 = xg_R_delta(profile_model, mass_unitful, Float64(z), 200)
    r200_kpc_value = Float64(ustrip(u"kpc", r200))
    x = max(Float64(r_kpc) / r200_kpc_value, eps(Float64))

    par = XGPaint.get_params(profile_model, mass_unitful, Float64(z))
    rho = getproperty(par, SYM_P0) * XGPaint.generalized_nfw(
        x,
        par.xc,
        getproperty(par, SYM_ALPHA),
        getproperty(par, SYM_BETA),
        getproperty(par, SYM_GAMMA),
    ) *
        rho_crit_hminus2_g_cm3(Float64(z))

    xH = 0.76
    nH_ne = 2 * xH / (xH + 1)
    nHe_ne = (1 - xH) / (2 * (1 + xH))
    factor = (ELECTRON_MASS + nH_ne * PROTON_MASS +
        nHe_ne * 4 * PROTON_MASS) / profile_model.cosmo.h^2

    ne = rho / factor
    return Float64(ustrip(u"cm^-3", ne))
end

function theta200_rad(profile_model, mass_msun::Real, z::Real)
    return Float64(compute_theta_max_local(
        profile_model,
        Float64(mass_msun) * XGPaint.M_sun,
        Float64(z);
        mult=1,
    ))
end

function dm_from_theta(profile_model, theta::Real, mass_msun::Real, z::Real)
    theta_eval = max(Float64(theta), compute_theta_min_local(profile_model))
    return Float64(profile_model(theta_eval, Float64(mass_msun), Float64(z)))
end

function unit_vector_from_position(position)
    norm_position = norm(position)
    norm_position > 0.0 || error("Catalog halo position has zero norm.")
    return Float64.(position) ./ norm_position
end

function perpendicular_basis(center_unit)
    reference = abs(dot(center_unit, [0.0, 0.0, 1.0])) < 0.9 ? [0.0, 0.0, 1.0] : [0.0, 1.0, 0.0]
    e1 = cross(reference, center_unit)
    e1 ./= norm(e1)
    e2 = cross(center_unit, e1)
    e2 ./= norm(e2)
    return e1, e2
end

function offset_sightline_vectors(center_unit, theta_values)
    e1, _ = perpendicular_basis(center_unit)
    vectors = Matrix{Float64}(undef, 3, length(theta_values))
    @inbounds for i in eachindex(theta_values)
        theta = Float64(theta_values[i])
        vectors[:, i] = cos(theta) .* center_unit .+ sin(theta) .* e1
        vectors[:, i] ./= norm(vectors[:, i])
    end
    return vectors
end

function angular_separation(u, v)
    return acos(clamp(dot(u, v), -1.0, 1.0))
end

function unit_vector_to_ra_dec(unit_vector)
    theta, phi = Healpix.vec2ang(unit_vector[1], unit_vector[2], unit_vector[3])
    return Float64(phi), Float64(pi / 2 - theta)
end

function sample_healpix_map_at_vectors!(out_values, out_pixels, map, sightline_vectors)
    res = map.resolution
    @inbounds for i in axes(sightline_vectors, 2)
        sightline_unit = @view sightline_vectors[:, i]
        theta, phi = Healpix.vec2ang(sightline_unit[1], sightline_unit[2], sightline_unit[3])
        pix = Healpix.ang2pixRing(res, theta, phi)
        out_pixels[i] = Int(pix)
        out_values[i] = Float64(map.pixels[pix])
    end
    return out_values
end

function pixel_radius_rad(nside::Integer)
    return 1.0 / (sqrt(3.0) * Float64(nside))
end

function make_paint_halo_sets(n::Integer)
    return [
        (masses=Float64[], redshifts=Float64[], ras=Float64[], decs=Float64[])
        for _ in 1:n
    ]
end

function push_paint_halo!(halo_set, halo_unit, mass::Float64, redshift::Float64)
    ra, dec = unit_vector_to_ra_dec(halo_unit)
    push!(halo_set.masses, mass)
    push!(halo_set.redshifts, redshift)
    push!(halo_set.ras, ra)
    push!(halo_set.decs, dec)
    return nothing
end

function choose_catalog_halo(
    catalog_path::AbstractString;
    halo_index::Int,
    z_pick_max::Float64,
    pick_mass_min::Float64,
    pick_mass_max::Float64,
    chunkN::Int,
)
    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halos = size(pos_ds, 2)

        if halo_index > 0
            1 <= halo_index <= total_halos || error("halo_index=$(halo_index) is outside 1:$(total_halos).")
            raw_mass = Float64(mass_ds[halo_index])
            mass = raw_mass / H_VALUE
            redshift = Float64(redshift_ds[halo_index])
            position = Float64.(pos_ds[:, halo_index])
            return CatalogHalo(halo_index, position, raw_mass, mass, redshift, NaN)
        end

        best_index = 0
        best_raw_mass = NaN
        best_mass = -Inf
        best_redshift = NaN

        for batch_start in 1:chunkN:total_halos
            batch_stop = min(batch_start + chunkN - 1, total_halos)
            idx = batch_start:batch_stop
            raw_masses = Float64.(mass_ds[idx])
            masses = raw_masses ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])

            @inbounds for local_i in eachindex(masses)
                mass = masses[local_i]
                redshift = redshifts[local_i]
                if !isfinite(mass) || !isfinite(redshift)
                    continue
                end
                if redshift < 0.0 || redshift > z_pick_max
                    continue
                end
                if mass < pick_mass_min || (isfinite(pick_mass_max) && mass >= pick_mass_max)
                    continue
                end
                if mass > best_mass
                    best_index = batch_start + local_i - 1
                    best_raw_mass = raw_masses[local_i]
                    best_mass = mass
                    best_redshift = redshift
                end
            end
        end

        best_index > 0 || error(
            "No HalfDome halo found with 0 <= z <= $(z_pick_max) and " *
            "mass in [$(pick_mass_min), $(pick_mass_max))."
        )
        position = Float64.(pos_ds[:, best_index])
        return CatalogHalo(best_index, position, best_raw_mass, best_mass, best_redshift, NaN)
    end
end

function choose_catalog_halos_near_targets(
    catalog_path::AbstractString;
    target_masses::Vector{Float64},
    z_pick_max::Float64,
    pick_mass_min::Float64,
    pick_mass_max::Float64,
    chunkN::Int,
)
    all(target -> isfinite(target) && target > 0.0, target_masses) ||
        error("target_masses must all be positive finite values.")

    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halos = size(pos_ds, 2)

        ntarget = length(target_masses)
        best_indices = zeros(Int, ntarget)
        best_raw_masses = fill(NaN, ntarget)
        best_masses = fill(NaN, ntarget)
        best_redshifts = fill(NaN, ntarget)
        best_scores = fill(Inf, ntarget)

        for batch_start in 1:chunkN:total_halos
            batch_stop = min(batch_start + chunkN - 1, total_halos)
            idx = batch_start:batch_stop
            raw_masses = Float64.(mass_ds[idx])
            masses = raw_masses ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])

            @inbounds for local_i in eachindex(masses)
                mass = masses[local_i]
                redshift = redshifts[local_i]
                if !isfinite(mass) || !isfinite(redshift)
                    continue
                end
                if redshift < 0.0 || redshift > z_pick_max
                    continue
                end
                if mass < pick_mass_min || (isfinite(pick_mass_max) && mass >= pick_mass_max)
                    continue
                end

                for target_i in eachindex(target_masses)
                    score = abs(log10(mass) - log10(target_masses[target_i]))
                    better_score = score < best_scores[target_i]
                    same_score_lower_z = score == best_scores[target_i] && redshift < best_redshifts[target_i]
                    if better_score || same_score_lower_z
                        best_indices[target_i] = batch_start + local_i - 1
                        best_raw_masses[target_i] = raw_masses[local_i]
                        best_masses[target_i] = mass
                        best_redshifts[target_i] = redshift
                        best_scores[target_i] = score
                    end
                end
            end
        end

        any(best_indices .== 0) && error(
            "Could not find a low-redshift HalfDome halo for every target mass. " *
            "Try increasing z_pick_max or widening the pick mass range."
        )

        halos = CatalogHalo[]
        for target_i in eachindex(target_masses)
            index = best_indices[target_i]
            position = Float64.(pos_ds[:, index])
            push!(
                halos,
                CatalogHalo(
                    index,
                    position,
                    best_raw_masses[target_i],
                    best_masses[target_i],
                    best_redshifts[target_i],
                    target_masses[target_i],
                ),
            )
        end
        return halos
    end
end

function build_halo_diagnostic(
    profile_model,
    halo::CatalogHalo;
    source_redshift::Float64,
    impact_points::Int,
    rmin_fraction::Float64,
    rmax_fraction::Float64,
)
    source_redshift > halo.redshift || error(
        "source_redshift=$(source_redshift) must be larger than selected halo redshift=$(halo.redshift)."
    )

    r200 = r200_kpc(profile_model, halo.mass_msun, halo.redshift)
    theta200 = theta200_rad(profile_model, halo.mass_msun, halo.redshift)
    rperp_kpc = 10 .^ range(log10(rmin_fraction * r200), log10(rmax_fraction * r200); length=impact_points)
    theta_values = rperp_kpc ./ r200 .* theta200

    halo_center_unit = unit_vector_from_position(halo.position)
    sightline_vectors = offset_sightline_vectors(halo_center_unit, theta_values)
    actual_theta_values = [
        angular_separation(halo_center_unit, @view sightline_vectors[:, i])
        for i in axes(sightline_vectors, 2)
    ]
    selected_halo_dm = [
        dm_from_theta(profile_model, actual_theta_values[i], halo.mass_msun, halo.redshift)
        for i in eachindex(actual_theta_values)
    ]

    return (;
        halo=halo,
        source_redshift=source_redshift,
        r200=r200,
        theta200=theta200,
        rperp_kpc=rperp_kpc,
        actual_theta_values=actual_theta_values,
        sightline_vectors=sightline_vectors,
        selected_halo_dm=selected_halo_dm,
        full_catalog_dm=zeros(Float64, impact_points),
        painted_map_dm=zeros(Float64, impact_points),
        painted_map_pixels=zeros(Int, impact_points),
        foreground_hits=zeros(Int, impact_points),
    )
end

function accumulate_foreground_catalog_dm_multi!(
    diagnostics,
    foreground_halo_counts,
    paint_halo_sets,
    catalog_path::AbstractString,
    profile_model;
    foreground_mass_min::Float64,
    foreground_mass_max::Float64,
    chunkN::Int,
    paint_relevance_margin_rad::Float64,
)
    theta_min = compute_theta_min_local(profile_model)
    max_source_redshift = maximum(d.source_redshift for d in diagnostics)

    h5open(catalog_path, "r") do h5
        pos_ds = h5["Position"]
        mass_ds = h5["halo_mass_m200c"]
        redshift_ds = h5["redshift"]
        total_halos = size(pos_ds, 2)

        for batch_start in 1:chunkN:total_halos
            batch_stop = min(batch_start + chunkN - 1, total_halos)
            idx = batch_start:batch_stop
            positions = Float64.(pos_ds[:, idx])
            masses = Float64.(mass_ds[idx]) ./ H_VALUE
            redshifts = Float64.(redshift_ds[idx])

            @inbounds for local_i in eachindex(masses)
                mass = masses[local_i]
                redshift = redshifts[local_i]
                if !isfinite(mass) || !isfinite(redshift)
                    continue
                end
                if redshift < 0.0 || redshift > max_source_redshift
                    continue
                end
                if mass < foreground_mass_min || (isfinite(foreground_mass_max) && mass >= foreground_mass_max)
                    continue
                end

                position = @view positions[:, local_i]
                radius = norm(position)
                radius > 0.0 || continue
                halo_unit = Float64.(position) ./ radius

                theta_max = Float64(compute_theta_max_local(
                    profile_model,
                    mass * XGPaint.M_sun,
                    redshift,
                ))
                if !isfinite(theta_max) || theta_max <= 0.0
                    continue
                end

                for diag_i in eachindex(diagnostics)
                    diag = diagnostics[diag_i]
                    redshift <= diag.source_redshift || continue
                    foreground_halo_counts[diag_i] += 1
                    overlaps_for_paint = false
                    for sightline_i in axes(diag.sightline_vectors, 2)
                        sightline_unit = @view diag.sightline_vectors[:, sightline_i]
                        theta = angular_separation(halo_unit, sightline_unit)
                        if theta <= theta_max
                            diag.full_catalog_dm[sightline_i] +=
                                Float64(profile_model(max(theta, theta_min), mass, redshift))
                            diag.foreground_hits[sightline_i] += 1
                        end
                        if theta <= theta_max + paint_relevance_margin_rad
                            overlaps_for_paint = true
                        end
                    end
                    if overlaps_for_paint
                        push_paint_halo!(paint_halo_sets[diag_i], halo_unit, mass, redshift)
                    end
                end
            end
        end
    end

    return foreground_halo_counts
end

function map_tag_for_diagnostic(diag_i::Int, diag)
    halo = diag.halo
    target_text = isfinite(halo.target_mass_msun) ? mass_label(halo.target_mass_msun) : "forced"
    target_text = replace(target_text, "." => "p")
    return "target_$(target_text)_halo$(halo.index)_$(diag_i)"
end

function paint_relevant_halo_maps!(
    diagnostics,
    paint_halo_sets,
    dm_model_interp,
    nside::Int,
    output_dir::AbstractString;
    save_painted_maps::Bool,
)
    res = Healpix.Resolution(nside)
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(res)
    paint_function = xgpaint_paint_function()
    painted_map_paths = fill("", length(diagnostics))

    for diag_i in eachindex(diagnostics)
        diag = diagnostics[diag_i]
        halo_set = paint_halo_sets[diag_i]
        dm_hp = HealpixMap{Float64, RingOrder}(nside)
        fill!(dm_hp.pixels, 0.0)

        if !isempty(halo_set.masses)
            paint_function(
                dm_hp,
                workspace,
                dm_model_interp,
                halo_set.masses,
                halo_set.redshifts,
                halo_set.ras,
                halo_set.decs,
            )
        end

        sample_healpix_map_at_vectors!(
            diag.painted_map_dm,
            diag.painted_map_pixels,
            dm_hp,
            diag.sightline_vectors,
        )

        if save_painted_maps
            map_path = joinpath(
                output_dir,
                "painted_dm_map_$(map_tag_for_diagnostic(diag_i, diag))_nside$(nside).fits",
            )
            Healpix.saveToFITS(dm_hp, "!" * map_path, typechar="D")
            painted_map_paths[diag_i] = map_path
        end
    end

    return painted_map_paths
end

function save_table(path::AbstractString, header::Vector{String}, rows)
    parent = dirname(path)
    isempty(parent) || isdir(parent) || mkpath(parent)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    return path
end

function mass_label(mass_msun::Real)
    logm = log10(Float64(mass_msun))
    if abs(logm - round(logm)) < 1.0e-8
        return "1e$(Int(round(logm)))"
    end
    coeff = Float64(mass_msun) / 10.0^floor(logm)
    return "$(round(coeff; digits=2))e$(Int(floor(logm)))"
end

function save_summary_multi(
    path::AbstractString;
    diagnostics,
    foreground_halo_counts,
    paint_halo_sets,
    nside,
    dm_cache_file,
    painted_map_paths,
    output_dir,
)
    open(path, "w") do io
        println(io, "HalfDome catalog halo DM diagnostic")
        println(io, "output_dir=$(abspath(output_dir))")
        println(io, "painted_map_nside=$(nside)")
        println(io, "dm_cache_file=$(dm_cache_file)")
        println(io)
        println(io, "target_mass_msun,catalog_halo_index,catalog_raw_mass_msun_h,mass_passed_to_xgpaint_msun,redshift,source_redshift,r200_kpc,theta200_rad,theta200_arcmin,foreground_catalog_halos_with_z_le_source,painted_relevant_halo_count,painted_map_path")
        for (i, diag) in enumerate(diagnostics)
            halo = diag.halo
            println(
                io,
                join(
                    [
                        halo.target_mass_msun,
                        halo.index,
                        halo.raw_mass_msun_h,
                        halo.mass_msun,
                        halo.redshift,
                        diag.source_redshift,
                        diag.r200,
                        diag.theta200,
                        diag.theta200 * 180.0 / pi * 60.0,
                        foreground_halo_counts[i],
                        length(paint_halo_sets[i].masses),
                        painted_map_paths[i],
                    ],
                    ",",
                ),
            )
        end
        println(io)
        println(io, "Interpretation:")
        println(io, "selected_halo_dm is the isolated XGPaint HaloDMProfile evaluated at the selected catalog halo mass and redshift.")
        println(io, "full_catalog_dm is the sum over all HalfDome foreground halos with z <= source_redshift along the same sightlines.")
        println(io, "painted_map_dm is sampled from a HEALPix map painted only with catalog halos relevant to those sightlines.")
        println(io, "extra_foreground_dm = full_catalog_dm - selected_halo_dm.")
    end
    return path
end

function make_plots()
    catalog_path = resolve_project_path(get_string_arg(
        "halfdome_path",
        "lightcone_100.hdf5";
        env=("HALFDOME_HALO_TEST_PATH", "FRB_HALFDOME_PATH"),
    ))
    output_dir = resolve_project_path(get_string_arg(
        "output_dir",
        joinpath("frb_map_generation", "outputs", "xgpaint_halfdome_catalog_halo_dm_test");
        env="HALFDOME_HALO_TEST_OUTPUT_DIR",
    ))
    isdir(output_dir) || mkpath(output_dir)

    target_masses = parse_float_list(get_string_arg(
        "target_masses",
        "5e12,1e13,1e14";
        env="HALFDOME_HALO_TEST_TARGET_MASSES",
    ))
    halo_index = get_int_arg("halo_index", 0; env="HALFDOME_HALO_TEST_INDEX")
    z_pick_max = get_float_arg("z_pick_max", 0.05; env="HALFDOME_HALO_TEST_Z_PICK_MAX")
    pick_mass_min = get_float_arg("pick_mass_min", 1.0e12; env="HALFDOME_HALO_TEST_PICK_MASS_MIN")
    pick_mass_max = get_float_arg("pick_mass_max", Inf; env="HALFDOME_HALO_TEST_PICK_MASS_MAX")
    foreground_mass_min = get_float_arg("foreground_mass_min", 0.0; env="HALFDOME_HALO_TEST_FOREGROUND_MASS_MIN")
    foreground_mass_max = get_float_arg("foreground_mass_max", Inf; env="HALFDOME_HALO_TEST_FOREGROUND_MASS_MAX")
    source_redshift_buffer = get_float_arg("source_redshift_buffer", 0.02; env="HALFDOME_HALO_TEST_SOURCE_DZ")
    source_redshift_override = get_float_arg("source_redshift", NaN; env="HALFDOME_HALO_TEST_SOURCE_REDSHIFT")
    impact_points = get_int_arg("impact_points", 10; env="HALFDOME_HALO_TEST_IMPACT_POINTS")
    chunkN = get_int_arg("chunkN", 100_000; env="HALFDOME_HALO_TEST_CHUNKN")
    nside = get_int_arg("nside", 4096; env="HALFDOME_HALO_TEST_NSIDE")
    rmin_fraction = get_float_arg("rmin_fraction", 0.01; env="HALFDOME_HALO_TEST_RMIN_FRACTION")
    rmax_fraction = get_float_arg("rmax_fraction", 0.98; env="HALFDOME_HALO_TEST_RMAX_FRACTION")
    paint_relevance_margin_pixels = get_float_arg(
        "paint_relevance_margin_pixels",
        2.0;
        env="HALFDOME_HALO_TEST_PAINT_MARGIN_PIXELS",
    )
    save_painted_maps = get_bool_arg(
        "save_painted_maps",
        false;
        env="HALFDOME_HALO_TEST_SAVE_PAINTED_MAPS",
    )
    dm_cache_file = resolve_project_path(get_string_arg(
        "dm_cache_file",
        joinpath("frb_map_generation", "outputs", "xgpaint_halfdome_catalog_halo_dm_test", "painted_dm_profile_cache.jld2");
        env="HALFDOME_HALO_TEST_DM_CACHE_FILE",
    ))
    dm_cache_overwrite = get_bool_arg(
        "dm_cache_overwrite",
        false;
        env="HALFDOME_HALO_TEST_DM_CACHE_OVERWRITE",
    )

    impact_points >= 2 || error("impact_points must be at least 2.")
    nside > 0 || error("nside must be positive.")
    rmin_fraction > 0.0 || error("rmin_fraction must be positive.")
    rmax_fraction > rmin_fraction || error("rmax_fraction must be larger than rmin_fraction.")
    paint_relevance_margin_pixels >= 0.0 || error("paint_relevance_margin_pixels must be non-negative.")

    profile_model = make_dm_model()
    halos = if halo_index > 0
        [
            choose_catalog_halo(
                catalog_path;
                halo_index=halo_index,
                z_pick_max=z_pick_max,
                pick_mass_min=pick_mass_min,
                pick_mass_max=pick_mass_max,
                chunkN=chunkN,
            ),
        ]
    else
        choose_catalog_halos_near_targets(
            catalog_path;
            target_masses=target_masses,
            z_pick_max=z_pick_max,
            pick_mass_min=pick_mass_min,
            pick_mass_max=pick_mass_max,
            chunkN=chunkN,
        )
    end

    diagnostics = [
        build_halo_diagnostic(
            profile_model,
            halo;
            source_redshift=isfinite(source_redshift_override) ?
                source_redshift_override :
                halo.redshift + source_redshift_buffer,
            impact_points=impact_points,
            rmin_fraction=rmin_fraction,
            rmax_fraction=rmax_fraction,
        )
        for halo in halos
    ]

    foreground_halo_counts = zeros(Int, length(diagnostics))
    paint_halo_sets = make_paint_halo_sets(length(diagnostics))
    paint_relevance_margin_rad = paint_relevance_margin_pixels * pixel_radius_rad(nside)
    accumulate_foreground_catalog_dm_multi!(
        diagnostics,
        foreground_halo_counts,
        paint_halo_sets,
        catalog_path,
        profile_model;
        foreground_mass_min=foreground_mass_min,
        foreground_mass_max=foreground_mass_max,
        chunkN=chunkN,
        paint_relevance_margin_rad=paint_relevance_margin_rad,
    )

    dm_cache_dir = dirname(dm_cache_file)
    isempty(dm_cache_dir) || isdir(dm_cache_dir) || mkpath(dm_cache_dir)
    ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = get(ENV, "XGPAINT_CLEANUP_NONPOSITIVE", "true")
    build_interpolator_function = xgpaint_build_interpolator_function()
    dm_model_interp = build_interpolator_function(
        profile_model;
        cache_file=dm_cache_file,
        overwrite=dm_cache_overwrite,
    )
    painted_map_paths = paint_relevant_halo_maps!(
        diagnostics,
        paint_halo_sets,
        dm_model_interp,
        nside,
        output_dir;
        save_painted_maps=save_painted_maps,
    )

    default(fontfamily="Computer Modern", linewidth=2, framestyle=:box)

    p = plot(
        xscale=:log10,
        yscale=:log10,
        xlabel="R_perp [kpc]",
        ylabel="DM [pc cm^-3]",
        title="HalfDome catalog halos: selected profile vs full foreground",
        grid=true,
        size=(960, 680),
    )
    p_extra = plot(
        xscale=:log10,
        yscale=:log10,
        xlabel="R_perp [kpc]",
        ylabel="extra DM [pc cm^-3]",
        title="Extra foreground DM beyond the selected halo",
        grid=true,
        size=(860, 620),
    )
    p_ne = plot(
        xscale=:log10,
        yscale=:log10,
        xlabel="r [kpc]",
        ylabel="n_e(r) [cm^-3]",
        title="HalfDome selected catalog halo n_e profiles",
        grid=true,
        size=(860, 620),
    )
    p_map_delta = plot(
        xscale=:log10,
        xlabel="R_perp [kpc]",
        ylabel="painted map - direct full [pc cm^-3]",
        title="Painted-map sample minus direct catalog sum",
        grid=true,
        size=(860, 620),
    )

    rows_dm = Vector{Any}[]
    rows_ne = Vector{Any}[]
    colors = palette(:viridis, length(diagnostics))
    for (diag_i, diag) in enumerate(diagnostics)
        halo = diag.halo
        target_text = isfinite(halo.target_mass_msun) ?
            "target $(mass_label(halo.target_mass_msun))" :
            "halo index $(halo.index)"
        actual_text = "actual $(mass_label(halo.mass_msun)), z=$(round(halo.redshift; digits=4))"
        curve_label = "$(target_text): $(actual_text)"

        plot!(
            p,
            diag.rperp_kpc,
            diag.selected_halo_dm;
            label="selected only, $(curve_label)",
            marker=:circle,
            color=colors[diag_i],
        )
        plot!(
            p,
            diag.rperp_kpc,
            diag.full_catalog_dm;
            label="full foreground, $(target_text)",
            marker=:diamond,
            linestyle=:dash,
            color=colors[diag_i],
        )
        plot!(
            p,
            diag.rperp_kpc,
            diag.painted_map_dm;
            label="painted map, $(target_text)",
            marker=:star5,
            linestyle=:dot,
            color=colors[diag_i],
        )

        extra_dm = diag.full_catalog_dm .- diag.selected_halo_dm
        positive_extra_dm = [value > 0.0 ? value : NaN for value in extra_dm]
        if any(value -> isfinite(value), positive_extra_dm)
            plot!(
                p_extra,
                diag.rperp_kpc,
                positive_extra_dm;
                label=curve_label,
                marker=:utriangle,
                color=colors[diag_i],
            )
        end

        map_minus_direct = diag.painted_map_dm .- diag.full_catalog_dm
        plot!(
            p_map_delta,
            diag.rperp_kpc,
            map_minus_direct;
            label=curve_label,
            marker=:circle,
            color=colors[diag_i],
        )

        r_ne = 10 .^ range(log10(0.005 * diag.r200), log10(diag.r200); length=300)
        ne = [xgpaint_ne3d_cm3(profile_model, r, halo.mass_msun, halo.redshift) for r in r_ne]
        plot!(
            p_ne,
            r_ne,
            ne;
            label=curve_label,
            color=colors[diag_i],
        )

        append!(
            rows_dm,
            (
                [
                    isfinite(halo.target_mass_msun) ? halo.target_mass_msun : NaN,
                    diag.rperp_kpc[i],
                    diag.actual_theta_values[i],
                    diag.actual_theta_values[i] * 180.0 / pi * 60.0,
                    diag.selected_halo_dm[i],
                    diag.full_catalog_dm[i],
                    diag.painted_map_dm[i],
                    diag.painted_map_dm[i] - diag.full_catalog_dm[i],
                    extra_dm[i],
                    diag.foreground_hits[i],
                    length(paint_halo_sets[diag_i].masses),
                    diag.painted_map_pixels[i],
                    halo.index,
                    halo.raw_mass_msun_h,
                    halo.mass_msun,
                    halo.redshift,
                    diag.source_redshift,
                    diag.r200,
                ]
                for i in eachindex(diag.rperp_kpc)
            ),
        )
        append!(
            rows_ne,
            (
                [
                    isfinite(halo.target_mass_msun) ? halo.target_mass_msun : NaN,
                    r_ne[i],
                    ne[i],
                    halo.index,
                    halo.mass_msun,
                    halo.redshift,
                    diag.r200,
                ]
                for i in eachindex(r_ne)
            ),
        )
    end
    savefig(p, joinpath(output_dir, "halfdome_catalog_halo_dm_comparison.png"))
    savefig(p_extra, joinpath(output_dir, "halfdome_catalog_halo_extra_foreground_dm.png"))
    savefig(p_map_delta, joinpath(output_dir, "halfdome_catalog_halo_painted_minus_direct.png"))
    savefig(p_ne, joinpath(output_dir, "halfdome_catalog_halo_ne_profile.png"))

    save_table(
        joinpath(output_dir, "halfdome_catalog_halo_dm_comparison.csv"),
        [
            "target_mass_msun",
            "Rperp_kpc",
            "theta_rad",
            "theta_arcmin",
            "selected_halo_dm_pc_cm3",
            "full_catalog_dm_pc_cm3",
            "painted_map_dm_pc_cm3",
            "painted_minus_direct_full_pc_cm3",
            "extra_foreground_dm_pc_cm3",
            "foreground_hit_count",
            "painted_relevant_halo_count",
            "painted_map_pixel",
            "selected_halo_index",
            "selected_raw_mass_msun_h",
            "selected_mass_msun",
            "selected_redshift",
            "source_redshift",
            "r200_kpc",
        ],
        rows_dm,
    )
    save_table(
        joinpath(output_dir, "halfdome_catalog_halo_ne_profile.csv"),
        ["target_mass_msun", "r_kpc", "ne_cm3", "selected_halo_index", "selected_mass_msun", "selected_redshift", "r200_kpc"],
        rows_ne,
    )
    save_summary_multi(
        joinpath(output_dir, "halfdome_catalog_halo_summary.txt");
        diagnostics=diagnostics,
        foreground_halo_counts=foreground_halo_counts,
        paint_halo_sets=paint_halo_sets,
        nside=nside,
        dm_cache_file=dm_cache_file,
        painted_map_paths=painted_map_paths,
        output_dir=output_dir,
    )

    println("Selected HalfDome catalog halos:")
    for (diag_i, diag) in enumerate(diagnostics)
        halo = diag.halo
        target_text = isfinite(halo.target_mass_msun) ? mass_label(halo.target_mass_msun) : "forced index"
        println("  target=$(target_text), index=$(halo.index)")
        println("    raw catalog mass=$(halo.raw_mass_msun_h), mass passed to XGPaint=$(halo.mass_msun)")
        println("    redshift=$(halo.redshift), source_redshift=$(diag.source_redshift)")
        println("    r200=$(diag.r200) kpc, theta200=$(diag.theta200 * 180.0 / pi * 60.0) arcmin")
        println("    foreground halos with z <= source_redshift=$(foreground_halo_counts[diag_i])")
        println("    relevant halos painted=$(length(paint_halo_sets[diag_i].masses))")
    end
    println("Painted map nside=$(nside), cache=$(dm_cache_file)")
    println("Wrote HalfDome catalog-halo diagnostic to $(abspath(output_dir))")
    println("  halfdome_catalog_halo_dm_comparison.png")
    println("  halfdome_catalog_halo_extra_foreground_dm.png")
    println("  halfdome_catalog_halo_painted_minus_direct.png")
    println("  halfdome_catalog_halo_ne_profile.png")
    println("  halfdome_catalog_halo_dm_comparison.csv")
    println("  halfdome_catalog_halo_summary.txt")
end

make_plots()
