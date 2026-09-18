#!/usr/bin/env julia
# Full-sky halo-DM maps weighted by the observed FRB redshift distributions (the method of the
# 2026-09-13 "cluster" products, paint_halfdome_observed_source_dm_maps.jl) for the UPDATED density
# implementations of sample_halfdome_updated_sightlines.jl:
#   b16_sphere1                 Battaglia16 (XGPaint parameters), gas inside the R200c sphere
#   lee22_noconc_sphere1        Lee22 Table A2 no-c fit, XGPaint-native reading, R200c sphere
#   lee22_noconc_sphere1_calib  the same, only halos inside the Lee22 calibration ranges
#                               (1e13-10^14.8 h^-1 Msun, z <= 2)
# One catalogue pass paints every model for every survey kernel.
#
# Map definition for one survey: DM(n) = sum_halo w(z_halo) DM_halo(n) with
#   w(z_halo) = sum_source[weight I(z_source >= z_halo)] / sum_source[weight],
# the fraction of the observed sources behind the halo (equality matches the sightline convention
# 0 < z_halo <= z_source). The full-sky <DM y> of such a map is the ensemble mean of the stratified
# sightline estimator (one stratum per observed redshift): the expected uniform-sightline cross
# signal, NOT a realization of 71 or 31 FRBs.
include(joinpath(@__DIR__, "sample_halfdome_updated_sightlines.jl"))
using Dates

const DEFAULT_MAP_MODELS = ("b16_sphere1", "lee22_noconc_sphere1", "lee22_noconc_sphere1_calib")
const DM_SANITY_MAX = 1.0e6  # pc cm^-3, single halo, single pixel; values are never clipped

struct SourceKernel
    redshift::Vector{Float64}
    tail_weight::Vector{Float64}
end

function source_kernel(path)
    lines = readlines(path)
    first(lines) == "frb,redshift,weight" || error("Unexpected kernel CSV header in $(path)")
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

# A halo contributes only to sources behind it: fraction of the source weight with z_source >= z_halo.
source_fraction(kernel::SourceKernel, z) = kernel.tail_weight[searchsortedfirst(kernel.redshift, z)]

function test_source_kernel()
    kernel = SourceKernel([0.2, 0.5, 1.0], [1.0, 5/6, 0.5, 0.0])
    @assert source_fraction(kernel, 0.0) == 1
    @assert source_fraction(kernel, 0.2) == 1
    @assert source_fraction(kernel, 0.3) == 5/6
    @assert source_fraction(kernel, 1.0) == 0.5
    @assert source_fraction(kernel, 1.01) == 0
    halo_z = [0.1, 0.4, 0.8, 1.2]
    columns = [2.0, 3.0, 5.0, 7.0]
    direct = sum(w * sum(columns[halo_z .<= z]) for (z, w) in zip(kernel.redshift, [1/6, 2/6, 3/6]))
    averaged = sum(columns .* [source_fraction(kernel, z) for z in halo_z])
    @assert isapprox(direct, averaged; rtol=1e-14)
    println("PASS: source-kernel endpoints and equality to averaging individual source screens")
end

parse_rows(text, total) = begin
    isempty(text) && return 1:total
    parts = split(text, ':')
    length(parts) == 2 || error("--rows must be start:stop (one-based, inclusive)")
    lo, hi = parse(Int, parts[1]), parse(Int, parts[2])
    1 <= lo <= hi <= total || error("--rows outside 1:$(total)")
    lo:hi
end

"""Angle between a halo direction and a pixel centre, exactly as in the painting loop."""
pixel_angle(res, pixel, ux, uy, uz) = begin
    px, py, pz = Healpix.pix2vecRing(res, pixel)
    acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0))
end

"""Paint one catalogue chunk into every (model, survey) map. Returns (pixel updates, halos painted)."""
function paint_kernel_chunk!(pix, workspace, ring_locks, caches, specs, cache_index, kernels, map_model, map_survey,
                             m, z, pos, maxaperture, model0, buffers)
    nmodel, nmaps = length(specs), length(pix)
    updates = zeros(Int64, thread_capacity())
    painted = zeros(Int64, thread_capacity())
    Threads.@threads :static for h in eachindex(z)
        tid = Threads.threadid()
        weights = [source_fraction(k, z[h]) for k in kernels]
        all(iszero, weights) && continue  # no observed source behind this halo
        distance = sqrt(pos[1, h]^2 + pos[2, h]^2 + pos[3, h]^2)
        isfinite(distance) && distance > 0 || error("Invalid halo position")
        ux, uy, uz = pos[1, h] / distance, pos[2, h] / distance, pos[3, h] / distance
        center_theta, center_phi = Healpix.vec2ang(ux, uy, uz)
        center_theta = Float64(center_theta)
        center_phi = mod(Float64(center_phi), 2pi)
        theta_max = min(ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], maxaperture), pi)
        prepared = [prepare_painted_halo(caches[cache_index[p]], m[h], z[h], specs[p].aperture) for p in 1:nmodel]
        theta_r200c = ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], 1.0)
        # spheres are already zero outside their edge; projected (cylinder) models are cut here like XGPaint
        theta_aperture = [specs[p].aperture == 1.0 ? theta_r200c :
                          ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], specs[p].aperture) for p in 1:nmodel]
        selected = [in_selection(specs[p], m[h], z[h]) for p in 1:nmodel]
        vals = buffers[tid]
        ring_start, ring_stop = XGPaint.get_relevant_rings(workspace.res, center_theta, theta_max)
        n = 0
        for ring in ring_start:ring_stop
            range1, range2 = XGPaint.get_ring_disc_ranges(workspace, ring, center_theta, center_phi, theta_max)
            first_pixel = workspace.ring_first_pixels[ring]
            lock(ring_locks[ring]) do
                for local_pixel in Iterators.flatten((range1, range2))
                    gp = first_pixel + local_pixel - 1
                    theta = pixel_angle(workspace.res, gp, ux, uy, uz)
                    theta < theta_max || continue
                    for p in 1:nmodel
                        v = (theta < theta_aperture[p] && selected[p]) ? Float64(prepared[p].value(theta)) : 0.0
                        isfinite(v) && 0.0 <= v <= DM_SANITY_MAX ||
                            error("Invalid single-halo DM=$(v) at M200c=$(m[h]), z=$(z[h]), theta=$(theta)")
                        vals[p] = v
                    end
                    for k in 1:nmaps
                        pix[k][gp] += weights[map_survey[k]] * vals[map_model[k]]
                    end
                    n += 1
                end
            end
        end
        updates[tid] += n
        painted[tid] += n > 0
    end
    return sum(updates), sum(painted)
end

"""Brute-force check of the painted subset: for random pixels, sum every halo's weighted, selected
column directly (no ring/disc enumeration). Same pixel vectors and angle formula as the painter."""
function brute_force_check(pix, workspace, caches, specs, cache_index, kernels, map_model, map_survey,
                           m, z, pos, maxaperture, model0, n_pixels; output_path)
    rng = MersenneTwister(20260918)
    touched = findall(!iszero, pix[1])
    isempty(touched) && error("Self-test subset painted no pixel")
    test_pixels = vcat(touched[rand(rng, 1:length(touched), min(n_pixels, length(touched)))],
                       rand(rng, 1:length(pix[1]), max(n_pixels ÷ 4, 1)))
    expected = zeros(length(test_pixels), length(pix))
    prepared_all = [[prepare_painted_halo(caches[cache_index[p]], m[h], z[h], specs[p].aperture) for p in eachindex(specs)]
                    for h in eachindex(z)]
    for h in eachindex(z)
        weights = [source_fraction(k, z[h]) for k in kernels]
        distance = sqrt(pos[1, h]^2 + pos[2, h]^2 + pos[3, h]^2)
        ux, uy, uz = pos[1, h] / distance, pos[2, h] / distance, pos[3, h] / distance
        theta_max = min(ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], maxaperture), pi)
        theta_r200c = ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], 1.0)
        theta_aperture = [specs[p].aperture == 1.0 ? theta_r200c :
                          ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], specs[p].aperture) for p in eachindex(specs)]
        for (i, gp) in enumerate(test_pixels)
            theta = pixel_angle(workspace.res, gp, ux, uy, uz)
            theta < theta_max || continue
            for k in eachindex(pix)
                p = map_model[k]
                (theta < theta_aperture[p] && in_selection(specs[p], m[h], z[h])) || continue
                expected[i, k] += weights[map_survey[k]] * Float64(prepared_all[h][p].value(theta))
            end
        end
    end
    worst = 0.0
    open(output_path, "w") do io
        println(io, "pixel_ring_one_based,map_index,painted,brute_force,relative_difference")
        for (i, gp) in enumerate(test_pixels), k in eachindex(pix)
            painted, direct = pix[k][gp], expected[i, k]
            scale = max(abs(direct), 1e-12)
            rel = abs(painted - direct) / scale
            worst = max(worst, rel)
            println(io, "$(gp),$(k),$(painted),$(direct),$(rel)")
        end
    end
    nonzero = count(!iszero, expected)
    worst <= 1e-9 || error("Painted subset disagrees with the brute-force sum: worst relative difference $(worst)")
    println("PASS: brute-force check on $(length(test_pixels)) pixels x $(length(pix)) maps ($(nonzero) non-zero entries); " *
            "worst relative difference $(worst)")
    return worst, length(test_pixels), nonzero
end

function kernel_weighted_main()
    options = parse_options(ARGS)
    test_source_kernel()
    output = abspath(option(options, "output_dir", joinpath(@__DIR__, "outputs", "tsz_dm_fullsky_20260918")))
    kernel_dir = abspath(option(options, "kernel_dir", joinpath(output, "kernels")))
    cache_dir = abspath(option(options, "cache_dir", joinpath(output, "cache")))
    catalog = abspath(option(options, "halfdome_path", DEFAULT_CATALOG))
    nside = int_option(options, "nside", 4096)
    chunk_size = int_option(options, "chunk_size", 100_000)
    model_labels = String.(split(option(options, "models", join(DEFAULT_MAP_MODELS, ",")), ","))
    survey_names = String.(split(option(options, "surveys", "planck,act"), ","))
    rows_text = option(options, "rows", "")
    self_test_pixels = int_option(options, "self_test_pixels", 0)
    map_tag = option(options, "map_tag", "")
    nside > 0 && chunk_size > 0 || error("Invalid resolution or chunk size")
    isfile(catalog) || error("Missing HalfDome catalogue: $(catalog)")
    for folder in ("maps", "analysis", "logs")
        mkpath(joinpath(output, folder))
    end
    mkpath(cache_dir)

    all_specs = updated_model_specs()
    specs = [only(filter(s -> s.label == label, all_specs)) for label in model_labels]
    kernels = [source_kernel(joinpath(kernel_dir, survey * "_sources.csv")) for survey in survey_names]
    zmax = maximum(last(k.redshift) for k in kernels)
    map_model = Int[]
    map_survey = Int[]
    map_paths = String[]
    for (p, spec) in enumerate(specs), (s, survey) in enumerate(survey_names)
        push!(map_model, p)
        push!(map_survey, s)
        push!(map_paths, joinpath(output, "maps", spec.label * "_" * survey * map_tag * ".fits"))
    end
    any(isfile, map_paths) && error("An output map exists; use a fresh output directory or --map-tag")
    println("Kernel-weighted full-sky DM maps: models=$(join(model_labels, ",")); surveys=$(join(survey_names, ",")) " *
            "(sources: $(join(string.(length.(getfield.(kernels, :redshift))), "/"))); zmax=$(zmax); NSIDE=$(nside); " *
            "threads=$(Threads.nthreads())")
    flush(stdout)

    # one radius cache per distinct cache_label (models that differ only by a halo selection share it)
    cache_labels = unique(spec.cache_label for spec in specs)
    caches = Any[]
    provenance = Dict{String,Any}()
    for cache_label in cache_labels
        spec = first(filter(s -> s.cache_label == cache_label, all_specs))
        runtime = ProfileSupport.dm_profile_runtime_configuration(spec.config)
        path = joinpath(cache_dir, cache_label * "_radius_cache.h5")
        println("== cache $(cache_label): $(runtime.description); sphere=$(spec.cut) R200c")
        flush(stdout)
        t0 = time()
        cache = build_radius_scaled_cache(runtime, path; refinement=2, zmax=zmax, spherical_cut=spec.cut)
        worst_grid = validate_radius_cache(cache; sample_count=100,
            output_path=joinpath(output, "analysis", cache_label * "_cache_check.csv"))
        worst_chord = validate_cache_against_profile_owned_chord(cache, runtime, spec.cut, spec.aperture;
            output_path=joinpath(output, "analysis", cache_label * "_profile_owned_chord_check.csv"))
        if spec.config.dm_profile == "lee2022" && spec.config.lee2022_normalization == "xgpaint_ne2d"
            check_xgpaint_native_routes(runtime)
        end
        push!(caches, cache)
        provenance[cache_label * ".description"] = runtime.description
        provenance[cache_label * ".model_family"] = runtime.generated_model_family
        provenance[cache_label * ".cache_signature"] = runtime.cache_signature
        provenance[cache_label * ".cache_file"] = path
        provenance[cache_label * ".cache_check_max_relative_error"] = worst_grid
        provenance[cache_label * ".profile_owned_chord_check_max_relative_error"] = worst_chord
        if runtime.model isa ProfileSupport.AbstractLee2022DMProfile
            provenance[cache_label * ".lee2022_normalization_factor_on_printed_eq9"] =
                Float64(ProfileSupport.lee2022_normalization_factor(runtime.model))
        end
        for (key, value) in runtime.provenance
            provenance[cache_label * "." * String(key)] = value
        end
        println("   cache ready in $(round(time() - t0)) s")
        flush(stdout)
    end
    cache_index = [findfirst(==(spec.cache_label), cache_labels) for spec in specs]
    model0 = caches[1].model
    logmass_bounds = (first(caches[1].logmasses), last(caches[1].logmasses))
    maxaperture = maximum(spec.aperture for spec in specs)

    workspace = XGPaint.HealpixRingProfileWorkspace{Float64}(Healpix.Resolution(nside))
    ring_locks = [ReentrantLock() for _ in eachindex(workspace.ring_thetas)]
    maps = [HealpixMap{Float64,RingOrder}(nside) for _ in map_paths]
    for map in maps
        fill!(map.pixels, 0.0)
    end
    pix = Tuple(map.pixels for map in maps)
    buffers = [zeros(length(specs)) for _ in 1:thread_capacity()]

    scanned, selected, total, updates_total, painted_total = 0, 0, 0, 0, 0
    sel_mass = [Inf, -Inf]
    sel_z = [Inf, -Inf]
    subset = (Float64[], Float64[], zeros(3, 0))
    rows_range = 1:0
    start = time()
    h5open(catalog, "r") do h5
        positions, masses, redshifts = h5["Position"], h5["halo_mass_m200c"], h5["redshift"]
        total = length(redshifts)
        size(positions) == (3, total) && length(masses) == total || error("Invalid catalogue shape")
        rows_range = parse_rows(rows_text, total)
        for first_row in first(rows_range):chunk_size:last(rows_range)
            rows = first_row:min(first_row + chunk_size - 1, last(rows_range))
            z = Float64.(redshifts[rows])
            m = Float64.(masses[rows]) ./ H_VALUE
            pos = Float64.(positions[:, rows])
            all(isfinite, z) && all(isfinite, m) && all(m .> 0) && all(isfinite, pos) || error("Invalid catalogue values")
            scanned += length(rows)
            keep = (z .> 0) .& (z .<= zmax)
            any(keep) || continue
            z, m, pos = z[keep], m[keep], pos[:, keep]
            ProfileSupport.validate_profile_masses_in_cache(m, logmass_bounds)
            selected += length(z)
            sel_mass .= (min(sel_mass[1], minimum(m)), max(sel_mass[2], maximum(m)))
            sel_z .= (min(sel_z[1], minimum(z)), max(sel_z[2], maximum(z)))
            n_updates, n_painted = paint_kernel_chunk!(pix, workspace, ring_locks, caches, specs, cache_index, kernels,
                                                       map_model, map_survey, m, z, pos, maxaperture, model0, buffers)
            updates_total += n_updates
            painted_total += n_painted
            if self_test_pixels > 0
                subset = (vcat(subset[1], m), vcat(subset[2], z), hcat(subset[3], pos))
            end
            if div(first_row - first(rows_range), chunk_size) % 20 == 0 || last(rows) == last(rows_range)
                println("Scanned ", scanned, "/", length(rows_range), " rows; foreground halos=", selected,
                        "; pixel updates=", updates_total, "; elapsed seconds=", round(time() - start))
                flush(stdout)
            end
        end
    end
    scanned == length(rows_range) && selected > 0 || error("Incomplete catalogue scan or no foreground halo")
    elapsed_paint = time() - start

    check = nothing
    if self_test_pixels > 0
        check = brute_force_check(pix, workspace, caches, specs, cache_index, kernels, map_model, map_survey,
                                  subset..., maxaperture, model0, self_test_pixels;
                                  output_path=joinpath(output, "analysis", "brute_force_check" * map_tag * ".csv"))
    end

    kernel_digests = [bytes2hex(sha256(read(joinpath(kernel_dir, survey * "_sources.csv")))) for survey in survey_names]
    for (k, path) in enumerate(map_paths)
        map = maps[k]
        all(isfinite, map.pixels) && minimum(map.pixels) >= 0 || error("Invalid map values in $(path)")
        Healpix.saveToFITS(map, "!" * path, typechar="D")
        spec, s = specs[map_model[k]], map_survey[k]
        entries = Dict{String,Any}(
            "profile_label" => spec.label, "dm_profile" => spec.config.dm_profile,
            "model_description" => spec.note, "cache_label" => spec.cache_label,
            "survey" => survey_names[s], "source_count" => length(kernels[s].redshift),
            "source_kernel_file" => joinpath(kernel_dir, survey_names[s] * "_sources.csv"),
            "source_kernel_sha256" => kernel_digests[s], "source_max_redshift" => zmax,
            "source_selection" => "uniform independent directions with observed redshifts; ensemble mean, not sparse mocks",
            "map_definition" => "sum_halo DM_halo(n) * sum_source[w_source I(z_source>=z_halo)] / sum_source[w_source]",
            "map_units" => "observer-frame pc cm^-3", "ordering" => "RING", "nside" => nside,
            "catalogue" => catalog, "catalog_total_rows" => total, "catalog_rows_scanned" => scanned,
            "catalog_rows_range" => "$(first(rows_range)):$(last(rows_range))",
            "halos_selected" => selected, "halos_with_at_least_one_pixel_center" => painted_total,
            "pixel_profile_updates_all_maps" => updates_total,
            "halo_selection" => isfinite(spec.mass_max) || isfinite(spec.z_max) ?
                "$(spec.mass_min) <= M200c/Msun < $(spec.mass_max), z <= $(spec.z_max)" : "all resolved foreground halos; no mass cut",
            "profile_mass_definition" => "M200c", "catalog_mass_native_units" => "Msun/h",
            "catalog_mass_conversion" => "halo_mass_m200c / $(H_VALUE)",
            "selected_mass_min_msun" => sel_mass[1], "selected_mass_max_msun" => sel_mass[2],
            "selected_redshift_min" => sel_z[1], "selected_redshift_max" => sel_z[2],
            "spherical_cut_r200c" => spec.cut, "aperture_r200c_multiplier" => spec.aperture,
            "geometry" => spec.cut > 0 ? "sphere; chord-limited LOS; b/R200c = tan(theta)/tan(theta200c); DM -> 0 at the edge" :
                "cylinder; full LOS to 1e5 R200c; angular aperture cut (XGPaint convention)",
            "observer_redshift_dilution" => "1/(1+z_halo) already included in profile",
            "beam" => "none", "noise" => "none", "mask" => "none",
            "dm_scope" => "halo-only partial prediction; no diffuse IGM or host-galaxy DM",
            "map_mean_pc_cm3" => mean(map.pixels), "map_max_pc_cm3" => maximum(map.pixels),
            "elapsed_seconds_painting" => elapsed_paint, "julia_threads" => Threads.nthreads(),
            "created_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "output_map" => path)
        for key in ("description", "model_family", "cache_signature", "cache_check_max_relative_error",
                    "profile_owned_chord_check_max_relative_error", "lee2022_normalization_factor_on_printed_eq9")
            haskey(provenance, spec.cache_label * "." * key) && (entries["cache." * key] = provenance[spec.cache_label * "." * key])
        end
        write_provenance(splitext(path)[1] * "_provenance.txt", entries)
        println("Saved ", path, " mean DM=", mean(map.pixels), " max=", maximum(map.pixels))
    end
    provenance["run.catalog"] = catalog
    provenance["run.catalog_rows_range"] = "$(first(rows_range)):$(last(rows_range))"
    provenance["run.catalog_rows_scanned"] = scanned
    provenance["run.catalog_total_rows"] = total
    provenance["run.foreground_halos_selected"] = selected
    provenance["run.models"] = join(model_labels, ",")
    provenance["run.surveys"] = join(survey_names, ",")
    provenance["run.kernel_sha256"] = join(kernel_digests, ",")
    provenance["run.zmax"] = zmax
    provenance["run.nside"] = nside
    provenance["run.elapsed_seconds_painting"] = elapsed_paint
    provenance["run.julia_threads"] = Threads.nthreads()
    provenance["run.created_utc"] = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
    if check !== nothing
        provenance["run.brute_force_check_worst_relative_difference"] = check[1]
        provenance["run.brute_force_check_pixels"] = check[2]
        provenance["run.brute_force_check_nonzero_entries"] = check[3]
    end
    write_provenance(joinpath(output, "maps", "run_provenance" * map_tag * ".txt"), provenance)
    println("Saved kernel-weighted DM maps (", round(time() - start), " s)")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    kernel_weighted_main()
end
