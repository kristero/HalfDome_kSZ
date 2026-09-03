#!/usr/bin/env julia

# Paint a full-sky HalfDome Compton-y map for the fiducial Battaglia thermal
# pressure profile.  The halo selection and external R200c aperture deliberately
# match the full-halo FRB-DM maps used by the Lee22/Battaglia16 comparison.

using Dates
using HDF5
using Healpix
using Interpolations
using XGPaint

module MatchedMapSupport
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))
end
const Support = MatchedMapSupport
const ProfileSupport = Support.ProfileSupport

const H_VALUE = 0.68
const OMEGA_B = 0.049
const OMEGA_C = 0.261
const DEFAULT_CATALOG =
    "/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5"

function cluster_username()
    return get(ENV, "CLUSTER_USER", get(ENV, "USER", "kristero10"))
end

function default_output_map(nside, source_redshift, aperture)
    root = joinpath(
        "/lustre/work",
        cluster_username(),
        "frb_data",
        "battaglia12_tsz_x_lee22_battaglia16_dm_m200c_3r200c_z1",
        "maps",
    )
    redshift_tag = replace(string(source_redshift), "." => "p")
    aperture_tag = replace(string(aperture), "." => "p")
    return joinpath(
        root,
        "battaglia12_fiducial_halfdome_compton_y_zmax$(redshift_tag)_" *
        "nside$(nside)_m200c_r200cx$(aperture_tag).fits",
    )
end

function default_cache()
    return joinpath(
        "/lustre/work",
        cluster_username(),
        "tSZ_data",
        "cache",
        "battaglia12_fiducial_m200c_logm15p7_tsz_interpolator.jld2",
    )
end

function build_battaglia12_fiducial_model()
    # These are the repository's fiducial Battaglia thermal-pressure
    # coefficients. XGPaint retains the historical Julia type name
    # Battaglia16ThermalSZProfile, while this pressure fit is referred to as
    # Battaglia12 throughout the HalfDome tSZ workflow.
    return Battaglia16ThermalSZProfile(
        Omega_c=OMEGA_C,
        Omega_b=OMEGA_B,
        h=H_VALUE,
        P0_amp=18.1,
        P0_alpha_m=0.154,
        P0_alpha_z=-0.758,
        x_c_amp=0.497,
        x_c_alpha_m=-0.00865,
        x_c_alpha_z=0.731,
        alpha_amp=1.0,
        alpha_alpha_m=0.0,
        alpha_alpha_z=0.0,
        beta_amp=4.35,
        beta_alpha_m=0.0393,
        beta_alpha_z=0.415,
        gamma_amp=-0.3,
        gamma_alpha_m=0.0,
        gamma_alpha_z=0.0,
    )
end

function build_or_load_interpolator(
    model,
    cache_path;
    overwrite_cache::Bool,
    pad::Int,
    logmass_max::Float64,
)
    if isfile(cache_path) && !overwrite_cache
        println("Loading existing Battaglia12 tSZ interpolator: $(cache_path)")
        return build_interpolator(
            model,
            cache_file=cache_path,
            overwrite=false,
        ), "loaded"
    end

    mkpath(dirname(cache_path))
    println("Building Battaglia12 tSZ interpolator: $(cache_path)")
    interpolated = build_interpolator(
        model;
        cache_file=cache_path,
        pad=pad,
        logM_max=logmass_max,
        overwrite=true,
        verbose=false,
    )
    return interpolated, "built"
end

function main()
    options = Support.parse_options(ARGS)
    catalog = abspath(Support.option(options, "halfdome_path", DEFAULT_CATALOG))
    nside = Support.int_option(options, "nside", 4096)
    source_redshift = Support.float_option(options, "source_redshift", 1.0)
    aperture_r200c =
        Support.float_option(options, "halo_extension_r200_multiplier", 3.0)
    chunk_size = Support.int_option(options, "chunk_size", 100_000)
    maximum_catalog_rows = Support.int_option(options, "max_catalog_halos", 0)
    progress_every = Support.int_option(options, "progress_every", 5)
    sanity_max = Support.float_option(options, "tsz_value_sanity_max", 1.0)
    interpolator_pad = Support.int_option(options, "interpolator_pad", 256)
    interpolator_logmass_max =
        Support.float_option(options, "interpolator_logmass_max", 15.7)
    cache_overwrite = Support.bool_option(options, "tsz_cache_overwrite", false)
    overwrite = Support.bool_option(options, "overwrite", false)

    cache_path = abspath(Support.option(options, "tsz_cache", default_cache()))
    output_map = abspath(Support.option(
        options,
        "output_map",
        default_output_map(nside, source_redshift, aperture_r200c),
    ))
    provenance_path = abspath(Support.option(
        options,
        "provenance",
        splitext(output_map)[1] * "_provenance.txt",
    ))

    isfile(catalog) || error("HalfDome catalogue not found: $(catalog)")
    nside > 0 || error("nside must be positive.")
    source_redshift > 0.0 || error("source_redshift must be positive.")
    aperture_r200c > 0.0 ||
        error("halo_extension_r200_multiplier must be positive.")
    chunk_size > 0 || error("chunk_size must be positive.")
    maximum_catalog_rows >= 0 || error("max_catalog_halos cannot be negative.")
    sanity_max > 0.0 || error("tsz_value_sanity_max must be positive.")
    interpolator_pad >= 0 || error("interpolator_pad cannot be negative.")
    interpolator_logmass_max > 12.0 ||
        error("interpolator_logmass_max must exceed 12.")
    if isfile(output_map) && !overwrite
        error("Output map exists: $(output_map). Pass --overwrite=true to replace it.")
    end
    if isfile(provenance_path) && !overwrite
        error("Provenance exists: $(provenance_path). Pass --overwrite=true to replace it.")
    end
    mkpath(dirname(output_map))

    model = build_battaglia12_fiducial_model()
    interpolated_profile, cache_action = build_or_load_interpolator(
        model,
        cache_path;
        overwrite_cache=cache_overwrite,
        pad=interpolator_pad,
        logmass_max=interpolator_logmass_max,
    )
    theta_min = ProfileSupport.compute_theta_min_local(interpolated_profile)
    logmass_bounds = ProfileSupport.interpolator_logmass_bounds(interpolated_profile)
    spot_theta = max(theta_min, 1.0e-5)
    spot_value = Float64(interpolated_profile(spot_theta, 1.0e14, 0.5))
    isfinite(spot_value) && 0.0 <= spot_value <= sanity_max || error(
        "Invalid Battaglia12 tSZ cache spot value $(spot_value) at " *
        "theta=$(spot_theta), M200c=1e14 Msun, z=0.5.",
    )

    println("Matched HalfDome Battaglia12 Compton-y map")
    println("  catalogue=$(catalog)")
    println("  output_map=$(output_map)")
    println("  foreground interval: 0 < z_halo <= $(source_redshift)")
    println("  mass selection: complete resolved catalogue range")
    println("  profile mass: physical M200c (catalogue halo_mass_m200c / h)")
    println("  aperture=$(aperture_r200c) R200c, externally enforced")
    println("  NSIDE=$(nside), threads=$(Threads.nthreads())")
    println("  cache=$(cache_path) ($(cache_action))")
    println("  cache log10(M200c/Msun) bounds=$(logmass_bounds)")
    println("  cache spot Compton-y=$(spot_value)")

    resolution = Healpix.Resolution(nside)
    y_map = HealpixMap{Float64,RingOrder}(nside)
    fill!(y_map.pixels, 0.0)
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
        size(positions_dataset, 1) == 3 || error("Position must have shape (3, N).")
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
                selected_masses,
                logmass_bounds,
            )

            updates, hit_halos = Support.paint_batch_external_r200c!(
                y_map,
                workspace,
                ring_locks,
                interpolated_profile,
                theta_min,
                aperture_r200c,
                selected_masses,
                selected_redshifts,
                selected_positions;
                contribution_sanity_max=sanity_max,
                quantity_label="single-halo Compton-y",
                quantity_units="",
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
    all(isfinite, y_map.pixels) || error("Painted Compton-y map contains NaN or Inf.")
    minimum(y_map.pixels) >= 0.0 || error("Painted Compton-y map contains negatives.")

    Healpix.saveToFITS(y_map, "!" * output_map, typechar="D")
    elapsed_seconds = time() - start_time
    entries = Dict{String,Any}(
        "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "observable" => "thermal SZ Compton-y",
        "profile_label" => "Battaglia12 fiducial thermal pressure",
        "xgpaint_profile_type" => "Battaglia16ThermalSZProfile",
        "catalogue" => catalog,
        "catalog_mass_dataset" => "halo_mass_m200c",
        "catalog_mass_native_units" => "Msun/h",
        "profile_mass_definition" => "M200c",
        "profile_mass_units" => "physical Msun",
        "catalog_mass_conversion" => "halo_mass_m200c / h",
        "h" => H_VALUE,
        "Omega_b" => OMEGA_B,
        "Omega_c" => OMEGA_C,
        "aperture_radius_definition" => "R200c",
        "aperture_r200c_multiplier" => aperture_r200c,
        "aperture_owner" => "this generator; XGPaint default 4R200 bypassed",
        "foreground_redshift_interval" => "(0, source_redshift]",
        "source_redshift" => source_redshift,
        "mass_selection" => "complete resolved catalogue range; no science mass cut",
        "nside" => nside,
        "ordering" => "RING",
        "map_units" => "dimensionless Compton-y",
        "beam" => "none",
        "interpolator_cache" => cache_path,
        "interpolator_cache_action" => cache_action,
        "profile_logmass_cache_min" => logmass_bounds[1],
        "profile_logmass_cache_max" => logmass_bounds[2],
        "interpolator_spot_theta_rad" => spot_theta,
        "interpolator_spot_y" => spot_value,
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
        "map_min_y" => minimum(y_map.pixels),
        "map_max_y" => maximum(y_map.pixels),
        "map_mean_y" => sum(y_map.pixels) / length(y_map.pixels),
        "elapsed_seconds" => elapsed_seconds,
        "P0_amp" => 18.1,
        "P0_alpha_m" => 0.154,
        "P0_alpha_z" => -0.758,
        "x_c_amp" => 0.497,
        "x_c_alpha_m" => -0.00865,
        "x_c_alpha_z" => 0.731,
        "alpha_amp" => 1.0,
        "beta_amp" => 4.35,
        "beta_alpha_m" => 0.0393,
        "beta_alpha_z" => 0.415,
        "gamma_amp" => -0.3,
    )
    Support.write_provenance(provenance_path, entries)

    println("Saved Battaglia12 Compton-y map: $(output_map)")
    println("Saved provenance: $(provenance_path)")
    println("Selected $(halos_selected) halos; elapsed $(round(elapsed_seconds; digits=1)) s")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
