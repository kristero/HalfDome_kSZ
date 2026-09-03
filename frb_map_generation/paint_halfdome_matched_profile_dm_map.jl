#!/usr/bin/env julia

# Paint one complete HalfDome foreground halo-DM map for a selected electron
# density profile. This workflow is for angular power spectra, not one-point
# PDFs: every resolved foreground halo is painted into a complete HEALPix map.

using Dates
using HDF5
using Healpix
using Interpolations
using Statistics
using XGPaint
using Base.Threads

# Reuse the profile constructors, cache compatibility loader, Lee22 safeguards,
# and external M200c/R200c geometry that are validated by the PDF generator.
module ValidatedDMProfileSupport
include(joinpath(@__DIR__, "generate_halfdome_z1_dm_mass_windows.jl"))
end
const ProfileSupport = ValidatedDMProfileSupport

const H_VALUE = 0.68
const DEFAULT_CATALOG =
    "/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5"

function parse_options(args)
    options = Dict{String,String}()
    i = 1
    while i <= length(args)
        argument = String(args[i])
        startswith(argument, "--") || error("Unexpected positional argument: $(argument)")
        body = argument[3:end]
        if occursin('=', body)
            key, value = split(body, "="; limit=2)
        elseif i < length(args) && !startswith(String(args[i + 1]), "--")
            key, value = body, String(args[i + 1])
            i += 1
        else
            key, value = body, "true"
        end
        options[replace(lowercase(key), "-" => "_")] = value
        i += 1
    end
    return options
end

function option(options, key, default)
    return get(options, replace(lowercase(String(key)), "-" => "_"), string(default))
end

function bool_option(options, key, default)
    value = lowercase(strip(option(options, key, default)))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    error("Cannot parse --$(key)=$(repr(value)) as a Boolean.")
end

int_option(options, key, default) = parse(Int, option(options, key, default))
float_option(options, key, default) = parse(Float64, option(options, key, default))

cluster_username() =
    get(ENV, "CLUSTER_USER", get(ENV, "USER", "kristero10"))

function default_cache(profile)
    if profile == "lee2022"
        return joinpath(
            "/lustre/work",
            cluster_username(),
            "frb_data",
            "halfdome_frb_inputs",
            "lee2022_tablea2_noconcentration_m200c_profile_owned_los_v2_dm_cache.jld2",
        )
    end
    return joinpath(
        dirname(@__FILE__), "outputs", "shared_xgpaint_dm_cache.jld2",
    )
end

function default_output_map(profile, nside, source_redshift, aperture)
    output_root = joinpath(
        "/lustre/work",
        cluster_username(),
        "frb_data",
        "lee22_vs_battaglia16_m200c_3r200c_z1_power_spectra",
        "maps",
    )
    tag = profile == "lee2022" ? "lee22_noconcentration" : "battaglia16"
    redshift_tag = replace(string(source_redshift), "." => "p")
    aperture_tag = replace(string(aperture), "." => "p")
    return joinpath(
        output_root,
        "$(tag)_halfdome_full_halo_dm_zsrc$(redshift_tag)_" *
        "nside$(nside)_m200c_r200cx$(aperture_tag).fits",
    )
end

thread_capacity() =
    isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()

"""
Paint a batch using an externally supplied R200c aperture.

XGPaint's interpolation supplies DM(theta, M200c, z), but its internal
compute_theta_max/paint! route is intentionally bypassed. Ring locks make
overlapping halo writes deterministic and race-free while retaining halo-level
threading.
"""
function paint_batch_external_r200c!(
    dm_map::HealpixMap{Float64,RingOrder},
    workspace,
    ring_locks,
    interpolated_profile,
    theta_min::Float64,
    aperture_r200c::Float64,
    masses_m200c,
    redshifts,
    positions;
    contribution_sanity_max::Float64,
    quantity_label::AbstractString="single-halo DM",
    quantity_units::AbstractString="pc cm^-3",
)
    update_counts = zeros(Int64, thread_capacity())
    halo_hit_counts = zeros(Int64, thread_capacity())

    Threads.@threads :static for halo_index in eachindex(masses_m200c)
        thread_index = Threads.threadid()
        x = Float64(positions[1, halo_index])
        y = Float64(positions[2, halo_index])
        z_position = Float64(positions[3, halo_index])
        distance = sqrt(x^2 + y^2 + z_position^2)
        isfinite(distance) && distance > 0.0 ||
            error("Halo $(halo_index) has an invalid Cartesian position.")

        ux, uy, uz = x / distance, y / distance, z_position / distance
        center_theta, center_phi = Healpix.vec2ang(ux, uy, uz)
        center_theta = Float64(center_theta)
        center_phi = mod(Float64(center_phi), 2pi)
        mass = Float64(masses_m200c[halo_index])
        redshift = Float64(redshifts[halo_index])
        theta_max = ProfileSupport.compute_theta_max_r200c_external(
            interpolated_profile, mass, redshift, aperture_r200c,
        )
        isfinite(theta_max) && theta_max > 0.0 ||
            error("Invalid 3R200c angular radius for M200c=$(mass), z=$(redshift).")
        theta_max = min(theta_max, pi)

        ring_start, ring_stop = XGPaint.get_relevant_rings(
            workspace.res, center_theta, theta_max,
        )
        halo_updates = 0
        for ring_index in ring_start:ring_stop
            range1, range2 = XGPaint.get_ring_disc_ranges(
                workspace, ring_index, center_theta, center_phi, theta_max,
            )
            first_pixel = workspace.ring_first_pixels[ring_index]

            # Different halos can intersect the same map ring. The lock covers
            # the read-modify-write operation and prevents lost additions.
            lock(ring_locks[ring_index]) do
                for local_pixel in Iterators.flatten((range1, range2))
                    global_pixel = first_pixel + local_pixel - 1
                    px, py, pz = Healpix.pix2vecRing(workspace.res, global_pixel)
                    cosine = clamp(ux * px + uy * py + uz * pz, -1.0, 1.0)
                    theta = acos(cosine)
                    theta < theta_max || continue
                    contribution = Float64(interpolated_profile(
                        max(theta, theta_min), mass, redshift,
                    ))
                    units_suffix = isempty(quantity_units) ? "" : " $(quantity_units)"
                    isfinite(contribution) && contribution >= 0.0 || error(
                        "Invalid $(quantity_label)=$(contribution)$(units_suffix) at " *
                        "M200c=$(mass), z=$(redshift), theta=$(theta).",
                    )
                    contribution <= contribution_sanity_max || error(
                        "$(quantity_label)=$(contribution)$(units_suffix) exceeds " *
                        "sanity maximum=$(contribution_sanity_max)$(units_suffix). " *
                        "The value is not clipped.",
                    )
                    dm_map.pixels[global_pixel] += contribution
                    halo_updates += 1
                end
            end
        end
        update_counts[thread_index] += halo_updates
        halo_hit_counts[thread_index] += halo_updates > 0
    end
    return sum(update_counts), sum(halo_hit_counts)
end

function write_provenance(path, entries)
    mkpath(dirname(path))
    open(path, "w") do io
        for key in sort!(collect(keys(entries)))
            println(io, "$(key)=$(entries[key])")
        end
    end
end

function main()
    options = parse_options(ARGS)
    profile = ProfileSupport.normalize_dm_profile_name(
        option(options, "dm_profile", "battaglia16"),
    )
    concentration_mode = ProfileSupport.normalize_lee2022_concentration_mode(
        option(options, "lee2022_concentration_mode", "none"),
    )
    profile == "lee2022" && concentration_mode != "none" && error(
        "Only Lee22 no-concentration is implemented; duffy2008 remains a future option.",
    )

    catalog = abspath(option(options, "halfdome_path", DEFAULT_CATALOG))
    nside = int_option(options, "nside", 4096)
    source_redshift = float_option(options, "source_redshift", 1.0)
    aperture_r200c = float_option(options, "halo_extension_r200_multiplier", 3.0)
    chunk_size = int_option(options, "chunk_size", 100_000)
    maximum_catalog_rows = int_option(options, "max_catalog_halos", 0)
    progress_every = int_option(options, "progress_every", 5)
    sanity_max = float_option(options, "dm_value_sanity_max", 1.0e8)
    cleanup_nonpositive = bool_option(options, "dm_cleanup_nonpositive", true)
    cache_overwrite = bool_option(options, "dm_cache_overwrite", false)
    overwrite = bool_option(options, "overwrite", false)

    dm_cache = abspath(option(options, "dm_cache", default_cache(profile)))
    output_map = abspath(option(
        options,
        "output_map",
        default_output_map(profile, nside, source_redshift, aperture_r200c),
    ))
    provenance_path = abspath(option(
        options, "provenance", splitext(output_map)[1] * "_provenance.txt",
    ))

    isfile(catalog) || error("HalfDome catalogue not found: $(catalog)")
    nside > 0 || error("nside must be positive.")
    source_redshift > 0.0 || error("source_redshift must be positive.")
    aperture_r200c > 0.0 || error("halo_extension_r200_multiplier must be positive.")
    chunk_size > 0 || error("chunk_size must be positive.")
    maximum_catalog_rows >= 0 || error("max_catalog_halos cannot be negative.")
    sanity_max > 0.0 || error("dm_value_sanity_max must be positive.")
    if isfile(output_map) && !overwrite
        error("Output map exists: $(output_map). Pass --overwrite=true to replace it.")
    end
    if isfile(provenance_path) && !overwrite
        error("Provenance exists: $(provenance_path). Pass --overwrite=true to replace it.")
    end
    mkpath(dirname(output_map))

    runtime = ProfileSupport.dm_profile_runtime_configuration((
        dm_profile=profile,
        lee2022_concentration_mode=concentration_mode,
    ))
    ENV["XGPAINT_CLEANUP_NONPOSITIVE"] = cleanup_nonpositive ? "true" : "false"
    cache_build = ProfileSupport.build_dm_interpolator_compatible(
        runtime.model;
        cache_file=dm_cache,
        overwrite=cache_overwrite,
        cleanup_nonpositive=cleanup_nonpositive,
        generated_model_family=runtime.generated_model_family,
        cache_signature=runtime.cache_signature,
    )
    interpolated_profile = cache_build.profile
    spot_check = ProfileSupport.validate_dm_interpolator_spot_value(
        interpolated_profile;
        direct_model=profile == "lee2022" ? runtime.model : nothing,
        sanity_max=sanity_max,
    )
    theta_min = ProfileSupport.compute_theta_min_local(interpolated_profile)
    logmass_bounds = ProfileSupport.interpolator_logmass_bounds(interpolated_profile)

    println("Matched HalfDome halo-DM full-map configuration")
    println("  profile=$(profile): $(runtime.description)")
    println("  catalogue=$(catalog)")
    println("  output_map=$(output_map)")
    println("  M200c profile/cache bounds log10(M/Msun)=$(logmass_bounds)")
    println("  foreground interval: 0 < z_halo <= $(source_redshift)")
    println("  aperture=$(aperture_r200c) R200c, externally enforced")
    println("  NSIDE=$(nside), threads=$(Threads.nthreads())")
    println("  cache=$(dm_cache)")
    println("  cache interpolation=$(cache_build.interpolation_scheme)")

    resolution = Healpix.Resolution(nside)
    dm_map = HealpixMap{Float64,RingOrder}(nside)
    fill!(dm_map.pixels, 0.0)
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(resolution)
    ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]

    total_catalog_rows = 0
    rows_scanned = 0
    halos_selected = 0
    halos_with_pixels = 0
    pixel_updates = 0
    selected_mass_min = Inf
    selected_mass_max = -Inf
    selected_redshift_min = Inf
    selected_redshift_max = -Inf
    batch_number = 0
    start_time = time()

    h5open(catalog, "r") do handle
        positions_dataset = handle["Position"]
        masses_dataset = handle["halo_mass_m200c"]
        redshift_dataset = handle["redshift"]
        total_catalog_rows = size(positions_dataset, 2)
        size(positions_dataset, 1) == 3 ||
            error("Position must have shape (3, N).")
        length(masses_dataset) == total_catalog_rows ||
            error("halo_mass_m200c length does not match Position.")
        length(redshift_dataset) == total_catalog_rows ||
            error("redshift length does not match Position.")

        final_row = maximum_catalog_rows == 0 ?
            total_catalog_rows : min(maximum_catalog_rows, total_catalog_rows)
        for batch_start in 1:chunk_size:final_row
            batch_number += 1
            batch_stop = min(batch_start + chunk_size - 1, final_row)
            indices = batch_start:batch_stop
            positions = Float64.(positions_dataset[:, indices])
            masses = Float64.(masses_dataset[indices]) ./ H_VALUE
            redshifts = Float64.(redshift_dataset[indices])
            rows_scanned += length(indices)

            keep = isfinite.(masses) .& isfinite.(redshifts)
            keep .&= masses .> 0.0
            keep .&= redshifts .> 0.0
            keep .&= redshifts .<= source_redshift
            selected_count = count(keep)
            selected_count == 0 && continue

            selected_positions = positions[:, keep]
            selected_masses = masses[keep]
            selected_redshifts = redshifts[keep]
            ProfileSupport.validate_profile_masses_in_cache(
                selected_masses, logmass_bounds,
            )

            updates, hit_halos = paint_batch_external_r200c!(
                dm_map,
                workspace,
                ring_locks,
                interpolated_profile,
                theta_min,
                aperture_r200c,
                selected_masses,
                selected_redshifts,
                selected_positions;
                contribution_sanity_max=sanity_max,
            )
            halos_selected += selected_count
            halos_with_pixels += hit_halos
            pixel_updates += updates
            selected_mass_min = min(selected_mass_min, minimum(selected_masses))
            selected_mass_max = max(selected_mass_max, maximum(selected_masses))
            selected_redshift_min =
                min(selected_redshift_min, minimum(selected_redshifts))
            selected_redshift_max =
                max(selected_redshift_max, maximum(selected_redshifts))

            if progress_every > 0 &&
               (batch_number % progress_every == 0 || batch_stop == final_row)
                println(
                    "  batch=$(batch_number), rows=$(batch_start):$(batch_stop), " *
                    "halos=$(halos_selected), pixel updates=$(pixel_updates)",
                )
                flush(stdout)
            end
        end
    end

    halos_selected > 0 || error("No foreground halos passed the selection.")
    all(isfinite, dm_map.pixels) || error("Painted map contains NaN or Inf.")
    minimum(dm_map.pixels) >= 0.0 || error("Painted halo-DM map contains negative pixels.")
    maximum(dm_map.pixels) <= sanity_max * max(1, halos_selected) ||
        error("Accumulated map has a numerically impossible maximum.")

    Healpix.saveToFITS(dm_map, "!" * output_map, typechar="D")
    elapsed_seconds = time() - start_time
    entries = Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "profile" => profile,
        "profile_description" => runtime.description,
        "dm_model_family" => cache_build.model_family,
        "lee2022_concentration_mode" => concentration_mode,
        "catalogue" => catalog,
        "catalog_mass_dataset" => "halo_mass_m200c",
        "catalog_mass_native_units" => "Msun/h",
        "profile_mass_definition" => "M200c",
        "profile_mass_units" => "physical Msun",
        "aperture_radius_definition" => "R200c",
        "aperture_r200c_multiplier" => aperture_r200c,
        "aperture_owner" => "this generator; XGPaint compute_theta_max bypassed",
        "source_redshift" => source_redshift,
        "foreground_redshift_interval" => "(0, source_redshift]",
        "mass_selection" => "complete resolved catalogue range; no science mass cut",
        "nside" => nside,
        "ordering" => "RING",
        "map_units" => "observer-frame pc cm^-3",
        "observer_redshift_dilution" => "1/(1+z_halo), implemented by profile",
        "thread_safety" => "one lock per HEALPix ring for overlapping halo writes",
        "julia_threads" => Threads.nthreads(),
        "catalog_total_rows" => total_catalog_rows,
        "catalog_rows_scanned" => rows_scanned,
        "halos_selected" => halos_selected,
        "halos_with_at_least_one_pixel_center" => halos_with_pixels,
        "pixel_profile_updates" => pixel_updates,
        "selected_mass_min_msun" => selected_mass_min,
        "selected_mass_max_msun" => selected_mass_max,
        "selected_redshift_min" => selected_redshift_min,
        "selected_redshift_max" => selected_redshift_max,
        "profile_logmass_cache_min" => logmass_bounds[1],
        "profile_logmass_cache_max" => logmass_bounds[2],
        "dm_cache" => dm_cache,
        "dm_cache_interpolation" => cache_build.interpolation_scheme,
        "dm_cache_spot_value_pc_cm3" => spot_check.value,
        "dm_value_sanity_max_pc_cm3" => sanity_max,
        "map_min_pc_cm3" => minimum(dm_map.pixels),
        "map_max_pc_cm3" => maximum(dm_map.pixels),
        "map_mean_pc_cm3" => mean(dm_map.pixels),
        "map_nonzero_pixels" => count(!=(0.0), dm_map.pixels),
        "elapsed_seconds" => elapsed_seconds,
        "output_map" => output_map,
    )
    for (key, value) in runtime.provenance
        entries[String(key)] = value
    end
    write_provenance(provenance_path, entries)

    println("Saved complete halo-DM map: $(output_map)")
    println("Saved provenance: $(provenance_path)")
    println(
        "Map summary: min=$(minimum(dm_map.pixels)), " *
        "max=$(maximum(dm_map.pixels)), mean=$(mean(dm_map.pixels)) pc cm^-3",
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
