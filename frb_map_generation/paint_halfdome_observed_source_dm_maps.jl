#!/usr/bin/env julia
# Observation-redshift-averaged DM maps. The full-map average is the expected
# uniform-sightline cross signal, NOT a realization of 71 or 31 noisy FRBs.
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))

struct SourceKernel
    redshift::Vector{Float64}
    tail_weight::Vector{Float64}
end

function source_kernel(path)
    lines = readlines(path)
    first(lines) == "frb,redshift,weight" || error("Unexpected kernel CSV header")
    rows = [split(line, ',') for line in lines[2:end] if !isempty(strip(line))]
    zs = [parse(Float64, row[2]) for row in rows]
    weights = [parse(Float64, row[3]) for row in rows]
    !isempty(zs) && all(isfinite, zs) && all(zs .> 0) || error("Invalid source redshifts")
    all(isfinite, weights) && all(weights .> 0) || error("Invalid source weights")
    order = sortperm(zs)
    zs, weights = zs[order], weights[order]
    tail = reverse(cumsum(reverse(weights))) ./ sum(weights)
    return SourceKernel(zs, vcat(tail, 0.0))
end

# A halo contributes only to sources behind it. Equality matches the existing
# thin-screen convention 0 < z_halo <= z_source; there is no source clustering.
source_fraction(kernel::SourceKernel, z) =
    kernel.tail_weight[searchsortedfirst(kernel.redshift, z)]

struct SourceWeightedProfile{P}
    profile::P
    kernel::SourceKernel
end

function prepare_painted_halo(profile::SourceWeightedProfile, mass, z, aperture)
    prepared = prepare_painted_halo(profile.profile, mass, z, aperture)
    weight = source_fraction(profile.kernel, z)
    return (theta_max=prepared.theta_max, value=theta -> weight * prepared.value(theta))
end

function test_source_kernel()
    kernel = SourceKernel([0.2, 0.5, 1.0], [1.0, 5/6, 0.5, 0.0])
    @assert source_fraction(kernel, 0.0) == 1
    @assert source_fraction(kernel, 0.2) == 1
    @assert source_fraction(kernel, 0.3) == 5/6
    @assert source_fraction(kernel, 1.0) == 0.5
    @assert source_fraction(kernel, 1.01) == 0
    halo_z = [0.1, 0.4, 0.8, 1.2]
    columns = [2.0, 3.0, 5.0, 7.0]
    direct = sum(w * sum(columns[halo_z .<= z]) for
                 (z, w) in zip(kernel.redshift, [1/6, 2/6, 3/6]))
    averaged = sum(columns .* [source_fraction(kernel, z) for z in halo_z])
    @assert isapprox(direct, averaged; rtol=1e-14)
    println("PASS: source-kernel endpoints and equality to averaging individual source screens")
end

function observed_source_main()
    test_source_kernel()
    options = parse_options(ARGS)
    bool_option(options, "self_test_only", false) && return
    output = abspath(option(options, "output_dir", "outputs/takahashi_comparison"))
    kernel_dir = abspath(option(options, "kernel_dir", joinpath(output, "kernels")))
    catalog = abspath(option(options, "halfdome_path", DEFAULT_CATALOG))
    label = option(options, "label", "battaglia16")
    label in ("battaglia16", "lee22_legacy", "lee22_preferred") || error("Unknown model")
    profile = label == "battaglia16" ? "battaglia16" : "lee2022"
    mode = label == "lee22_preferred" ? "duffy2008" : "none"
    nside = int_option(options, "nside", 4096)
    chunk_size = int_option(options, "chunk_size", 100_000)
    nside > 0 && chunk_size > 0 || error("Invalid resolution or chunk size")
    surveys = ["planck", "act"]
    kernels = [source_kernel(joinpath(kernel_dir, survey * "_sources.csv")) for survey in surveys]
    length(kernels[1].redshift) == 71 && length(kernels[2].redshift) == 31 ||
        error("Expected the selected 71 Planck / 31 ACT sources")
    zmax = maximum(last(kernel.redshift) for kernel in kernels)
    paths = [joinpath(output, "maps", label * "_" * survey * ".fits") for survey in surveys]
    any(isfile, paths) && error("An output map exists; use a fresh output directory")
    for folder in ("maps", "cache", "analysis")
        mkpath(joinpath(output, folder))
    end
    runtime = ProfileSupport.dm_profile_runtime_configuration((
        dm_profile=profile, lee2022_concentration_mode=mode))
    cache_path = joinpath(output, "cache", label * "_observed_z_spherical3.h5")
    cache = build_radius_scaled_cache(runtime, cache_path;
                                     refinement=2, zmax=zmax, spherical_cut=3.0)
    validate_radius_cache(cache; output_path=joinpath(output, "analysis", label * "_cache_validation.csv"))
    weighted_profiles = [SourceWeightedProfile(cache, kernel) for kernel in kernels]
    maps = [HealpixMap{Float64,RingOrder}(nside) for _ in surveys]
    for map in maps
        fill!(map.pixels, 0.0)
    end
    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(Healpix.Resolution(nside))
    locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
    bounds = (first(cache.logmasses), last(cache.logmasses))
    scanned, selected, total = 0, 0, 0
    selected_min_mass, selected_max_mass = Inf, -Inf
    selected_min_z, selected_max_z = Inf, -Inf
    start = time()
    h5open(catalog, "r") do h5
        positions = h5["Position"]
        masses = h5["halo_mass_m200c"]
        redshifts = h5["redshift"]
        size(positions, 1) == 3 || error("Position must have Julia shape (3,N)")
        total = size(positions, 2)
        length(masses) == length(redshifts) == total || error("Catalogue dimensions disagree")
        for first_row in 1:chunk_size:total
            rows = first_row:min(first_row + chunk_size - 1, total)
            z = Float64.(redshifts[rows])
            m = Float64.(masses[rows]) ./ H_VALUE
            pos = Float64.(positions[:, rows])
            all(isfinite, z) && all(isfinite, m) || error("Non-finite catalogue row")
            all(m .> 0) || error("Non-positive halo mass")
            keep = (z .> 0) .& (z .<= zmax)
            scanned += length(rows)
            any(keep) || continue
            z, m, pos = z[keep], m[keep], pos[:, keep]
            ProfileSupport.validate_profile_masses_in_cache(m, bounds)
            for index in eachindex(surveys)
                paint_batch_external_r200c!(maps[index], workspace, locks,
                    weighted_profiles[index], 0.0, 3.0, m, z, pos;
                    contribution_sanity_max=1e8)
            end
            selected += length(z)
            selected_min_mass = min(selected_min_mass, minimum(m))
            selected_max_mass = max(selected_max_mass, maximum(m))
            selected_min_z = min(selected_min_z, minimum(z))
            selected_max_z = max(selected_max_z, maximum(z))
            if div(first_row-1, chunk_size) % 10 == 0 || last(rows) == total
                println(label, " rows=", scanned, "/", total, " foreground halos=", selected)
                flush(stdout)
            end
        end
    end
    selected > 0 && scanned == total || error("Incomplete or empty catalogue pass")
    for (index, survey) in enumerate(surveys)
        map = maps[index]
        all(isfinite, map.pixels) && minimum(map.pixels) >= 0 || error("Invalid map values")
        Healpix.saveToFITS(map, "!" * paths[index], typechar="D")
        entries = Dict{String,Any}(
            "profile_label" => label, "dm_profile" => profile,
            "lee2022_concentration_mode" => mode,
            "survey" => survey, "source_count" => length(kernels[index].redshift),
            "source_kernel_file" => joinpath(kernel_dir, survey * "_sources.csv"),
            "source_kernel_sha256" => bytes2hex(sha256(read(joinpath(kernel_dir, survey * "_sources.csv")))),
            "source_max_redshift" => zmax,
            "source_selection" => "uniform independent directions with observed redshifts; ensemble mean, not sparse mocks",
            "map_definition" => "sum_halo DM_halo(n) * sum_source[w_source I(z_source>=z_halo)] / sum_source[w_source]",
            "map_units" => "observer-frame pc cm^-3", "ordering" => "RING", "nside" => nside,
            "catalogue" => catalog, "catalog_total_rows" => total, "catalog_rows_scanned" => scanned,
            "halos_selected" => selected, "mass_selection" => "all resolved foreground halos; no mass cut",
            "profile_mass_definition" => "M200c", "catalog_mass_native_units" => "Msun/h",
            "catalog_mass_conversion" => "halo_mass_m200c / 0.68",
            "selected_mass_min_msun" => selected_min_mass, "selected_mass_max_msun" => selected_max_mass,
            "selected_redshift_min" => selected_min_z, "selected_redshift_max" => selected_max_z,
            "spherical_cut_r200c" => 3.0, "aperture_r200c_multiplier" => 3.0,
            "observer_redshift_dilution" => "1/(1+z_halo) already included in profile",
            "beam" => "none", "noise" => "none", "mask" => "none",
            "dm_scope" => "halo-only partial prediction; no diffuse IGM or host-galaxy DM",
            "preferred_model_warning" => label == "lee22_preferred" ?
                "diagnostic only: high-mass extrapolation fails gas-mass consistency checks" : "not applicable",
            "map_mean_pc_cm3" => mean(map.pixels), "map_max_pc_cm3" => maximum(map.pixels),
            "elapsed_seconds" => time() - start, "output_map" => paths[index])
        write_provenance(splitext(paths[index])[1] * "_provenance.txt", entries)
        println("Saved ", paths[index], " mean DM=", mean(map.pixels))
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    observed_source_main()
end
