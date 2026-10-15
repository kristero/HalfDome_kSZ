#!/usr/bin/env julia
# The density, XGPaint wrapper, chord factor, pixel query,
# and multithreaded accumulator are reused from the validated HalfDome code.
# The chord is evaluated directly to avoid low-redshift angular-cache errors.
# --simulation=halfdome also reruns the HalfDome control with direct chords.

module ValidatedPainter
include(joinpath(@__DIR__, "..", "generate_halfdome_z1_dm_mass_windows.jl"))
end
const P = ValidatedPainter
using HDF5, XGPaint, SHA, Statistics, Dates

function main()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    output = get(ENV, "TNG_COMPARISON_OUTPUT",
                 joinpath(root, "frb_map_generation", "outputs", "tng300_lee22_1r200c_20260921"))
    is_halfdome = get(P.CLI_OPTIONS, "simulation", "tng") == "halfdome"
    directory = joinpath(output, is_halfdome ? "halfdome_direct" : "tng")
    mkpath(directory)
    target = joinpath(directory, "halo_dm.h5")
    isfile(target) && error("Preserving existing result: $target")
    profile_file = joinpath(root, "frb_map_generation", "lee2022_frb_dm_profile.jl")
    expected_profile_sha = "1fe10765ba8e38ae42c0cf1d20630babff0b8b10aa2a41e1a617bdbc50ae8806"
    bytes2hex(sha256(read(profile_file))) == expected_profile_sha || error("Validated Lee22 source has changed")

    h = is_halfdome ? 0.68 : 0.6774
    omega_m = is_halfdome ? 0.31 : 0.3089
    omega_b = is_halfdome ? 0.049 : 0.0486
    lee = P.Lee2022NoConcentrationDMProfile(
        Omega_c=omega_m - omega_b, Omega_b=omega_b, h=h,
        normalization=:xgpaint_ne2d, n0_pivot=:mcut,
        shape_clip_mass_msun=10.0^14.8 / h, redshift_scaling=:physical,
    )
    model = P.SphericalChordDMProfile(P.Battaglia16DensityDMProfile(P.Lee2022XGPaintDMProfile(lee)), 1.0)
    println("Evaluating unchanged Lee22 no-c chords directly; h=$h Omega_m=$omega_m Omega_b=$omega_b")
    flush(stdout)
    # Verify tighter quadrature tolerances and the native XGPaint density mapping
    # over the selected support, including central and grazing chords.
    errors = Float64[]
    for mass in (1e13, 1.4762326542663123e13, 10.0^13.61 / h, 1e14, 8e14), z in (0.03, 0.3, 0.7, 1.0), x in (0.001, 0.04, 0.2, 0.8, 0.99)
        theta_max = P.compute_theta_max_r200c_external(model, mass, z, 1.0)
        theta = x * theta_max
        exact = P.chord_dm_pc_cm3(model.inner, x, sqrt(1 - x^2), mass, z; rtol=1e-11)
        value = model(theta, mass, z) * P.spherical_chord_factor(theta, theta_max, 1.0)
        push!(errors, abs(value / exact - 1))
        @assert isapprox(P.halo_electron_density_m3(model.inner, x, mass, z),
                         P.halo_electron_density_m3(lee, x, mass, z); rtol=2e-12)
    end
    maximum(errors) < 1e-6 || error("Direct chord quadrature accuracy check failed")
    println("Direct chord maximum relative error against tighter quadrature: $(maximum(errors))")
    flush(stdout)
    open(joinpath(directory, "direct_validation.txt"), "w") do io
        println(io, "direct_integration_checks=$(length(errors))")
        println(io, "maximum_relative_error=$(maximum(errors))")
        println(io, "profile_sha256=$expected_profile_sha")
    end
    get(P.CLI_OPTIONS, "validate_only", "false") == "true" && return

    hd = get(ENV, "TNG_REFERENCE_RAYS", joinpath(output, "halfdome", "halo_dm.h5"))
    pixels, ra, dec, labels, mass_low, mass_high = h5open(hd, "r") do handle
        (read(handle["frb_pixel_ring_1based"]), read(handle["frb_ra_rad"]), read(handle["frb_dec_rad"]),
         read(handle["window_label"]), read(handle["window_effective_min_msun"]), read(handle["window_effective_max_msun"]))
    end
    n = length(pixels)
    n == 120000 || error("Expected exactly 120000 reference sightlines")
    n_windows = length(labels)
    1 <= n_windows <= 16 || error("Expected 1--16 mass windows")
    all(mass_low .< mass_high) || error("Invalid mass windows")
    selection_low, selection_high = minimum(mass_low), maximum(mass_high)
    ux, uy, uz = cos.(dec) .* cos.(ra), cos.(dec) .* sin.(ra), sin.(dec)
    order = sortperm(pixels)
    resolution = P.Healpix.Resolution(4096)
    workspace, backend = P.make_sparse_disc_workspace(resolution)
    accumulator = P.WindowDMAccumulator(n, n_windows)
    theta_min = eps(Float64)
    halo_counts = zeros(Int64, n_windows)
    lightcone = is_halfdome ? joinpath(root, "lightcone_100.hdf5") : joinpath(output, "tng_lightcone.h5")
    h5open(lightcone, "r") do handle
        if !is_halfdome
            attrs(handle)["complete"] == 1 || error("Incomplete TNG lightcone")
        end
        mass_dataset = is_halfdome ? "halo_mass_m200c" : "mass_msun"
        position_dataset = is_halfdome ? "Position" : "position_comoving_mpc"
        count = length(handle[mass_dataset])
        for first in 1:250000:count
            last = min(first + 249999, count)
            positions = handle[position_dataset][:, first:last]
            masses = Float64.(handle[mass_dataset][first:last])
            is_halfdome && (masses ./= h)
            redshifts = handle["redshift"][first:last]
            if is_halfdome
                selected = (selection_low .<= masses .< selection_high) .& (0 .< redshifts .<= 1)
                positions = positions[:, selected]
                masses = masses[selected]
                redshifts = redshifts[selected]
            end
            all(selection_low .<= masses .< selection_high) || error("Mass outside configured interval")
            all(0 .< redshifts .<= 1) || error("Redshift outside lightcone")
            masks = zeros(UInt16, length(masses))
            for window in 1:n_windows
                in_window = mass_low[window] .<= masses .< mass_high[window]
                masks[in_window] .|= UInt16(1) << (window - 1)
                halo_counts[window] += sum(in_window)
            end
            P.accumulate_batch_windows!(accumulator, workspace, model, theta_min, 1.0, 0.0,
                pixels[order], order, ux, uy, uz,
                view(positions, 1, :), view(positions, 2, :), view(positions, 3, :),
                masses, redshifts, masks, n_windows, 1.0e8, 1.0)
            if first == 1 || last == count || first % 2500000 == 1
                println("Catalogue rows processed: $last / $count; window counts=$halo_counts")
                flush(stdout)
            end
        end
    end
    dm, hits, window_hits = P.reduce_accumulator(accumulator)
    all(isfinite, dm) && minimum(dm) >= 0 || error("Invalid DM result")
    if n_windows == 3 && mass_low[1] == mass_low[3] && mass_high[1] == mass_low[2] && mass_high[2] == mass_high[3]
        maximum(abs.(dm[:, 3] .- dm[:, 1] .- dm[:, 2])) < 1e-8 || error("Disjoint mass-window additivity failed")
    end
    edges, centers, counts, density = P.histogram_columns(dm, 300, 0.1, 30000.; spacing="log")
    P.atomic_write(target; overwrite=false) do temporary
        h5open(temporary, "w") do handle
            handle["dm_pc_cm3"] = dm
            handle["frb_pixel_ring_1based"] = pixels
            handle["frb_ra_rad"] = ra
            handle["frb_dec_rad"] = dec
            handle["window_label"] = labels
            handle["window_effective_min_msun"] = mass_low
            handle["window_effective_max_msun"] = mass_high
            handle["window_halo_count"] = halo_counts
            handle["window_los_intersection_count"] = window_hits
            handle["pdf_bin_edges_pc_cm3"] = edges
            handle["pdf_bin_centers_pc_cm3"] = centers
            handle["pdf_count"] = counts
            handle["pdf_density_per_pc_cm3"] = density
            metadata = attrs(handle)
            for (key, value) in Dict(
                "simulation"=>(is_halfdome ? "HalfDome" : "TNG300-1-Dark"),
                "halo_finder"=>(is_halfdome ? "HalfDome native catalogue" : "FoF/SUBFIND"),
                "source_redshift"=>1.0, "n_rays"=>n, "nside"=>4096, "seed"=>42,
                "little_h"=>h, "omega_m"=>omega_m, "omega_b"=>omega_b,
                "halo_boundary"=>"spherical", "sphere_r200c"=>1.0, "mass_definition"=>"M200c",
                "concentration"=>"none", "normalization"=>"xgpaint_ne2d", "n0_pivot"=>"mcut",
                "profile_sha256"=>expected_profile_sha, "profile_evaluation"=>"direct chord quadrature; no interpolation cache",
                "mass_extrapolation_below_fit"=>(selection_low < 1e13 / h),
                "calibrated_mass_min_msun"=>1e13 / h,
                "ray_configuration_sha256"=>P.file_sha256(hd),
                "direct_quadrature_max_relative_error"=>maximum(errors),
                "lightcone_sha256"=>P.file_sha256(lightcone), "painter_sha256"=>P.file_sha256(@__FILE__),
                "reused_generator_sha256"=>P.file_sha256(joinpath(@__DIR__, "..", "generate_halfdome_z1_dm_mass_windows.jl")),
                "sparse_disc_backend"=>backend, "unique_halo_frb_intersection_count"=>hits,
                "created_utc"=>string(now(UTC)), "complete"=>true,
            )
                metadata[key] = value
            end
        end
    end
    println("Completed direct-chord painting: $target")
    println("All-ray mean DM by window: $(vec(mean(dm; dims=1)))")
end

main()
