#!/usr/bin/env julia
# Individual halo-only DMs for the UPDATED density implementations, at the same 100,000
# random source positions and observed-redshift assignments as the 2026-09-14 run
# (outputs/takahashi_100k_20260914/rays/source_positions.h5), plus one extra source plane
# with every source at a common redshift (default z = 2, the Medlock & Nagai model kernel).
# The annular Compton-y samples at these positions (rays/annular_y_samples.npz) are reused.
#
# All models use a finite spherical boundary with a chord-limited line of sight (DM -> 0 at
# the edge), evaluated through the validated radius-coordinate cache:
#   b16_sphere1           Battaglia16 (XGPaint parameters, ne2d electrons), gas inside 1 R200c
#   b16_sphere3           same, gas inside 3 R200c (the previous cross-correlation convention)
#   lee22_noconc_sphere1  Lee22 Table A2 no-c fit, XGPaint-native normalization (P0 = 200 n0 with
#                         XGPaint's ne2d electrons), M_cut n0 pivot, fit-range shape clip; 1 R200c
#   lee22_noconc_sphere3  the same fit extrapolated to 3 R200c (outside its 0.04-1.34 R200c fit range)
#   lee22_pref_sphere1    Lee22 Table 3 fit + TNG-mean concentration, same reading; 1 R200c (diagnostic)
#   lee22_legacy_sphere3  the previous Lee22 reading (literal eq. 9, 1e14 pivot, no clip); 3 R200c
#                         regression against the 2026-09-14 product
include(joinpath(@__DIR__, "sample_halfdome_observed_sightlines.jl"))

struct UpdatedModelSpec
    label::String
    config::NamedTuple
    cut::Float64
    note::String
end

# halo_boundary="projected" here only means "hand the plain density model to the radius cache";
# the spherical chord integration is done by build_radius_scaled_cache(spherical_cut=...).
function updated_model_specs()
    clip_fit = 10.0^14.8 / H_VALUE
    base = (dm_profile="battaglia16", lee2022_concentration_mode="none", lee2022_normalization="literal",
            lee2022_n0_pivot="legacy_1e14", lee2022_concentration_source="duffy2008",
            lee2022_shape_mass_clip_msun=Inf, lee2022_redshift_scaling="physical",
            halo_boundary="projected", dm_aperture_r200_multiplier=1.0)
    lee_new = merge(base, (dm_profile="lee2022", lee2022_normalization="xgpaint_ne2d",
                           lee2022_n0_pivot="mcut", lee2022_shape_mass_clip_msun=clip_fit))
    lee_pref = merge(lee_new, (lee2022_concentration_mode="duffy2008", lee2022_concentration_source="tng_mean"))
    lee_legacy = merge(base, (dm_profile="lee2022",))
    return UpdatedModelSpec[
        UpdatedModelSpec("b16_sphere1", base, 1.0,
            "Battaglia16; gas inside the R200c sphere (TNG like-for-like implementation)"),
        UpdatedModelSpec("b16_sphere3", base, 3.0,
            "Battaglia16; gas inside 3 R200c (previous cross-correlation convention)"),
        UpdatedModelSpec("lee22_noconc_sphere1", lee_new, 1.0,
            "Lee22 no-c fit; XGPaint-native normalization, M_cut pivot, fit-range shape clip; R200c sphere"),
        UpdatedModelSpec("lee22_noconc_sphere3", lee_new, 3.0,
            "same Lee22 no-c fit extrapolated to 3 R200c (outside the 0.04-1.34 R200c fit range)"),
        UpdatedModelSpec("lee22_pref_sphere1", lee_pref, 1.0,
            "Lee22 Table-3 fit + TNG-mean concentration, same reading; R200c sphere (diagnostic)"),
        UpdatedModelSpec("lee22_legacy_sphere3", lee_legacy, 3.0,
            "previous Lee22 reading (literal eq. 9, 1e14 pivot, no clip); 3 R200c regression"),
    ]
end

"""Independent check of the radius cache against the profile-owned chord integral
(`chord_dm_pc_cm3` in lee2022_frb_dm_profile.jl, the implementation compared with TNG). It shares
no code with the cache's dimensionless-shape / amplitude split."""
function validate_cache_against_profile_owned_chord(cache, runtime, cut; sample_count=60, tolerance=0.01, output_path)
    model = runtime.model
    inner = model isa ProfileSupport.AbstractLee2022DMProfile ? model :
            ProfileSupport.Battaglia16DensityDMProfile(model)
    rng = MersenneTwister(20260917)
    worst = 0.0
    open(output_path, "w") do io
        println(io, "mass_msun,redshift,impact_r200c,profile_owned_chord_dm,cached_dm,relative_error")
        for i in 1:sample_count
            mass = 10.0^(12.9 + rand(rng) * (15.5 - 12.9))
            z = 0.01 + rand(rng) * (last(cache.redshifts) - 0.02)
            x = cut * (i % 4 == 0 ? 1 - 10.0^(-1 - 3rand(rng)) : max(rand(rng)^2, 1.0e-5))
            prepared = prepare_painted_halo(cache, mass, z, cut)
            r200 = XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200)
            theta200 = Float64(XGPaint.angular_size(model, r200, z))
            theta = atan(x * tan(theta200))
            cached = prepared.value(theta)
            direct = ProfileSupport.chord_dm_pc_cm3(inner, x, sqrt((cut - x) * (cut + x)), mass, z)
            err = cached / direct - 1
            isfinite(err) || error("Non-finite profile-owned chord check")
            worst = max(worst, abs(err))
            println(io, join((mass, z, x, direct, cached, err), ','))
        end
    end
    worst <= tolerance || error("Radius cache disagrees with the profile-owned chord integral: max relative error $(worst)")
    println("PASS profile-owned chord check: $(sample_count) points, maximum relative error=$(worst)")
    return worst
end

"""The XGPaint-native reading two ways: the plain Lee22 model with normalization=:xgpaint_ne2d (what
this run caches) versus the same fit wrapped as an XGPaint profile (P0 = 200 n0, XGPaint's own
ne2d constants through the cache's Battaglia branch)."""
function check_xgpaint_native_routes(runtime)
    lee = runtime.model
    wrapped = ProfileSupport.Lee2022XGPaintDMProfile(lee)
    for (mass, z, x) in ((1.0e13, 0.3, 0.2), (1.0e14, 0.8, 0.9), (5.0e14, 1.5, 0.05), (2.0e15, 0.1, 2.5))
        a = radius_column_amplitude(lee, mass, z) * radius_density_shape(lee, mass, z)(x)
        b = radius_column_amplitude(wrapped, mass, z) * radius_density_shape(wrapped, mass, z)(x)
        isapprox(a, b; rtol=1.0e-9) || error("XGPaint-native Lee22 routes differ: $(a) vs $(b) at M=$(mass) z=$(z) x=$(x)")
    end
    println("PASS: Lee22 XGPaint-native normalization identical via the plain model and via XGPaint get_params (P0 = 200 n0)")
end

function updated_sightline_main()
    options = parse_options(ARGS)
    test_sparse_ring_rays()
    bool_option(options, "self_test_only", false) && return
    output = abspath(option(options, "output_dir", joinpath(@__DIR__, "outputs", "tsz_dm_cross_updated_20260917")))
    cache_dir = abspath(option(options, "cache_dir", joinpath(output, "cache")))
    positions_path = abspath(option(options, "positions",
        joinpath(@__DIR__, "outputs", "takahashi_100k_20260914", "rays", "source_positions.h5")))
    catalog = abspath(option(options, "halfdome_path", DEFAULT_CATALOG))
    z_plane = float_option(options, "common_source_redshift", 2.0)
    max_rows = int_option(options, "max_catalog_halos", 0)
    chunk_size = int_option(options, "chunk_size", 100_000)
    chunk_size > 0 || error("Invalid chunk size")
    isfinite(z_plane) && z_plane > 0 || error("Invalid common source redshift")
    result = joinpath(output, "rays", "individual_dm_updated.h5")
    isfile(result) && error("Output exists; use a fresh run directory: $(result)")
    for name in ("rays", "analysis", "logs")
        mkpath(joinpath(output, name))
    end
    mkpath(cache_dir)
    isfile(positions_path) || error("Missing source positions: $(positions_path)")
    isfile(catalog) || error("Missing HalfDome catalogue: $(catalog)")

    pixels, observed_z, nside = h5open(positions_path, "r") do h5
        (Int.(read(h5["pixel_ring_zero_based"])) .+ 1,
         hcat([read(h5[s * "/redshift"]) for s in RAY_SURVEYS]...),
         Int(read(attributes(h5)["nside"])))
    end
    nrays = length(pixels)
    plane_label = "z" * replace(string(z_plane), "." => "p")
    surveys = (RAY_SURVEYS..., plane_label)
    source_z = hcat(observed_z, fill(z_plane, nrays))
    nsurvey = size(source_z, 2)
    zmax = maximum(source_z)
    lookup = SparseRingRays(nside, pixels)
    specs = updated_model_specs()
    nmodel = length(specs)
    cuts = [spec.cut for spec in specs]
    maxcut = maximum(cuts)
    println("Updated-model sightlines: $(nrays) rays x $(nsurvey) source planes ($(join(surveys, ", "))); " *
            "zmax=$(zmax); $(nmodel) models; threads=$(Threads.nthreads())")
    flush(stdout)

    caches = Any[]
    provenance = Dict{String,Any}()
    for spec in specs
        runtime = ProfileSupport.dm_profile_runtime_configuration(spec.config)
        path = joinpath(cache_dir, spec.label * "_radius_cache.h5")
        println("== $(spec.label): $(runtime.description); sphere=$(spec.cut) R200c")
        flush(stdout)
        t0 = time()
        cache = build_radius_scaled_cache(runtime, path; refinement=2, zmax=zmax, spherical_cut=spec.cut)
        worst_grid = validate_radius_cache(cache; sample_count=100,
            output_path=joinpath(output, "analysis", spec.label * "_cache_check.csv"))
        worst_chord = validate_cache_against_profile_owned_chord(cache, runtime, spec.cut;
            output_path=joinpath(output, "analysis", spec.label * "_profile_owned_chord_check.csv"))
        if spec.config.dm_profile == "lee2022" && spec.config.lee2022_normalization == "xgpaint_ne2d"
            check_xgpaint_native_routes(runtime)
        end
        push!(caches, cache)
        provenance[spec.label * ".description"] = runtime.description
        provenance[spec.label * ".model_family"] = runtime.generated_model_family
        provenance[spec.label * ".cache_signature"] = runtime.cache_signature
        provenance[spec.label * ".sphere_r200c"] = spec.cut
        provenance[spec.label * ".note"] = spec.note
        provenance[spec.label * ".cache_file"] = path
        provenance[spec.label * ".cache_check_max_relative_error"] = worst_grid
        provenance[spec.label * ".profile_owned_chord_check_max_relative_error"] = worst_chord
        if runtime.model isa ProfileSupport.AbstractLee2022DMProfile
            provenance[spec.label * ".lee2022_normalization_factor_on_printed_eq9"] =
                Float64(ProfileSupport.lee2022_normalization_factor(runtime.model))
        end
        for (key, value) in runtime.provenance
            provenance[spec.label * "." * String(key)] = value
        end
        println("   cache ready in $(round(time() - t0)) s")
        flush(stdout)
    end
    model0 = caches[1].model
    logmass_bounds = (first(caches[1].logmasses), last(caches[1].logmasses))

    accumulators = [zeros(nrays, nsurvey, nmodel) for _ in 1:thread_capacity()]
    hits_outer = [zeros(Int64, nrays, nsurvey) for _ in 1:thread_capacity()]
    hits_inner = [zeros(Int64, nrays, nsurvey) for _ in 1:thread_capacity()]
    scanned, selected, total, final_row = 0, 0, 0, 0
    mass_min, mass_max = Inf, -Inf
    start = time()
    h5open(catalog, "r") do h5
        positions, masses, redshifts = h5["Position"], h5["halo_mass_m200c"], h5["redshift"]
        total = length(redshifts)
        size(positions) == (3, total) && length(masses) == total || error("Invalid catalogue shape")
        final_row = max_rows == 0 ? total : min(max_rows, total)
        for first_row in 1:chunk_size:final_row
            rows = first_row:min(first_row + chunk_size - 1, final_row)
            z = Float64.(redshifts[rows])
            m = Float64.(masses[rows]) ./ H_VALUE
            pos = Float64.(positions[:, rows])
            all(isfinite, z) && all(isfinite, m) && all(m .> 0) || error("Invalid catalogue values")
            keep = (z .> 0) .& (z .<= zmax)
            scanned += length(rows)
            any(keep) || continue
            z, m, pos = z[keep], m[keep], pos[:, keep]
            ProfileSupport.validate_profile_masses_in_cache(m, logmass_bounds)
            selected += length(z)
            mass_min, mass_max = min(mass_min, minimum(m)), max(mass_max, maximum(m))
            Threads.@threads :static for h in eachindex(z)
                tid = Threads.threadid()
                distance = sqrt(sum(abs2, view(pos, :, h)))
                isfinite(distance) && distance > 0 || error("Invalid halo position")
                direction = (pos[1, h] / distance, pos[2, h] / distance, pos[3, h] / distance)
                # The disc is the largest sphere's footprint; smaller spheres return 0 outside
                # their own edge. Amplitudes are prepared only when a ray intersects the halo.
                theta_max = ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], maxcut)
                prepared = nothing
                theta_r200c = 0.0
                visit_ray_disc(lookup, direction, theta_max) do ray, theta
                    any(source_z[ray, s] >= z[h] for s in 1:nsurvey) || return
                    if prepared === nothing
                        prepared = [prepare_painted_halo(caches[p], m[h], z[h], cuts[p]) for p in 1:nmodel]
                        theta_r200c = ProfileSupport.compute_theta_max_r200c_external(model0, m[h], z[h], 1.0)
                    end
                    values = [Float64(prepared[p].value(theta)) for p in 1:nmodel]
                    all(v -> isfinite(v) && 0 <= v <= 1.0e8, values) ||
                        error("Invalid single-halo DM; values are never clipped")
                    inside = theta < theta_r200c
                    for s in 1:nsurvey
                        source_z[ray, s] >= z[h] || continue
                        for p in 1:nmodel
                            accumulators[tid][ray, s, p] += values[p]
                        end
                        hits_outer[tid][ray, s] += 1
                        inside && (hits_inner[tid][ray, s] += 1)
                    end
                end
            end
            if div(first_row - 1, chunk_size) % 20 == 0 || last(rows) == final_row
                println("Scanned ", scanned, "/", total, "; foreground halos=", selected,
                        "; elapsed seconds=", round(time() - start))
                flush(stdout)
            end
        end
    end
    scanned == final_row && selected > 0 || error("Incomplete catalogue scan")
    dm = reduce(+, accumulators)
    outer, inner = reduce(+, hits_outer), reduce(+, hits_inner)
    all(isfinite, dm) && minimum(dm) >= 0 || error("Invalid accumulated DM")
    all(inner .<= outer) || error("Inner-sphere hit counts exceed outer counts")
    positions_digest = bytes2hex(sha256(read(positions_path)))
    elapsed = time() - start
    h5open(result, "w") do h5
        for (s, survey) in enumerate(surveys)
            group = create_group(h5, survey)
            group["source_redshift"] = source_z[:, s]
            group["foreground_halo_hits_outer_sphere"] = outer[:, s]
            group["foreground_halo_hits_r200c"] = inner[:, s]
            for (p, spec) in enumerate(specs)
                group[spec.label] = dm[:, s, p]
            end
        end
        entries = Dict{String,Any}(
            "catalog_rows_scanned" => scanned, "catalog_total_rows" => total,
            "foreground_halos_considered" => selected, "rays_per_survey" => nrays,
            "mass_min_msun" => mass_min, "mass_max_msun" => mass_max,
            "mass_definition" => "M200c; catalogue Msun/h divided by 0.68",
            "source_positions_sha256" => positions_digest, "source_positions_file" => positions_path,
            "nside" => nside, "model_labels" => join([spec.label for spec in specs], ","),
            "sphere_r200c_by_model" => join(string.(cuts), ","),
            "outer_sphere_r200c" => maxcut, "source_planes" => join(surveys, ","),
            "common_source_redshift" => z_plane,
            "observed_planes" => "planck, act: 2026-09-14 stratified observed-redshift assignments (unchanged)",
            "selection" => "all resolved halos with 0 < z_halo <= individual z_source",
            "impact_parameter" => "screen-plane b/R200c = tan(theta)/tan(theta200c); chord-limited LOS inside the sphere; DM -> 0 at the edge",
            "dm_scope" => "halo-only; no host or diffuse IGM DM; observer-frame pc cm^-3",
            "direction_sampling" => "iid uniform NSIDE pixel centres; with replacement (from source_positions.h5)",
            "elapsed_seconds" => elapsed, "julia_threads" => Threads.nthreads())
        for (key, value) in entries
            attributes(h5)[key] = value
        end
    end
    provenance["run.catalog"] = catalog
    provenance["run.catalog_rows_scanned"] = scanned
    provenance["run.catalog_total_rows"] = total
    provenance["run.foreground_halos_considered"] = selected
    provenance["run.source_positions_sha256"] = positions_digest
    provenance["run.source_planes"] = join(surveys, ",")
    provenance["run.common_source_redshift"] = z_plane
    provenance["run.elapsed_seconds"] = elapsed
    provenance["run.created_utc"] = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
    write_provenance(joinpath(output, "rays", "individual_dm_updated_provenance.txt"), provenance)
    println("Saved individual DMs to ", result, " (", round(elapsed), " s)")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    updated_sightline_main()
end
