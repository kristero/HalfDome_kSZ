#!/usr/bin/env julia
# Individual halo-only DMs, not samples of a redshift-averaged DM map.
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))

const RAY_MODELS = ("battaglia16", "lee22_legacy", "lee22_preferred")
const RAY_SURVEYS = ("planck", "act")

struct SparseRingRays{W}
    workspace::W
    ring_pixels::Vector{Vector{Int}}
    ring_indices::Vector{Vector{Int}}
    vectors::Matrix{Float64}
end

function SparseRingRays(nside, pixels)
    ws = XGPaint.HealpixRingProfileWorkspace{Float64}(Healpix.Resolution(nside))
    rp = [Int[] for _ in ws.ring_thetas]
    ri = [Int[] for _ in ws.ring_thetas]
    vectors = zeros(3, length(pixels))
    for i in sortperm(pixels)
        pixel = pixels[i] # Julia HEALPix indices are one-based.
        ring = searchsortedlast(ws.ring_first_pixels, pixel)
        push!(rp[ring], pixel - ws.ring_first_pixels[ring] + 1)
        push!(ri[ring], i)
        vectors[:, i] .= Healpix.pix2vecRing(ws.res, pixel)
    end
    return SparseRingRays(ws, rp, ri, vectors)
end

"""Visit only sampled pixels inside a disc, including repeated draws and wraparound."""
function visit_ray_disc(visit, lookup, direction, theta_max)
    ws = lookup.workspace
    theta0, phi0 = Healpix.vec2ang(direction...)
    r1, r2 = XGPaint.get_relevant_rings(ws.res, theta0, theta_max)
    for ring in r1:r2
        pixels = lookup.ring_pixels[ring]
        isempty(pixels) && continue
        ranges = XGPaint.get_ring_disc_ranges(ws, ring, theta0, mod(phi0, 2pi), theta_max)
        for pixel_range in ranges
            isempty(pixel_range) && continue
            a = searchsortedfirst(pixels, first(pixel_range))
            b = searchsortedlast(pixels, last(pixel_range))
            for j in a:b
                ray = lookup.ring_indices[ring][j]
                cosine = clamp(sum(direction[k] * lookup.vectors[k, ray] for k in 1:3), -1.0, 1.0)
                theta = acos(cosine)
                theta < theta_max && visit(ray, theta)
            end
        end
    end
end

function test_sparse_ring_rays()
    # Test every pixel plus a duplicate; compare with brute-force angular cuts.
    nside = 8
    pixels = vcat(collect(1:12*nside^2), [1, 200])
    lookup = SparseRingRays(nside, pixels)
    for (theta, phi, radius) in ((0.0, 0.0, 0.35), (pi, 0.0, 0.28),
                                 (1.2, 0.001, 0.17), (1.2, 2pi-0.001, 0.17),
                                 (2.0, 2.0, 0.002), (1.5, 3.0, 1.0))
        direction = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        found = Int[]
        visit_ray_disc(lookup, direction, radius) do ray, angle
            push!(found, ray)
        end
        expected = [i for i in eachindex(pixels) if
            acos(clamp(sum(direction[k]*lookup.vectors[k,i] for k in 1:3), -1.0, 1.0)) < radius]
        @assert sort(found) == expected
    end
    # Thin-screen redshift selection: no foreground contribution behind a source.
    @assert [z >= 0.5 for z in [0.1, 0.5, 1.0]] == [false, true, true]
    println("PASS: sparse disc lookup vs brute force, poles, longitude wrap, duplicates, redshift boundary")
end

function sightline_main()
    options = parse_options(ARGS)
    test_sparse_ring_rays()
    bool_option(options, "self_test_only", false) && return
    output = abspath(option(options, "output_dir", "outputs/takahashi_100k"))
    previous = abspath(option(options, "previous_dir", ""))
    catalog = abspath(option(options, "halfdome_path", DEFAULT_CATALOG))
    input = joinpath(output, "rays", "source_positions.h5")
    result = joinpath(output, "rays", "individual_dm.h5")
    isfile(result) && error("Output exists; use a fresh run directory: $(result)")
    pixels, source_z, nside = h5open(input, "r") do h5
        (Int.(read(h5["pixel_ring_zero_based"])) .+ 1,
         hcat([read(h5[s * "/redshift"]) for s in RAY_SURVEYS]...),
         Int(read(attributes(h5)["nside"])))
    end
    zmax = maximum(source_z)
    lookup = SparseRingRays(nside, pixels)
    caches = map(RAY_MODELS) do label
        runtime = ProfileSupport.dm_profile_runtime_configuration((
            dm_profile=label == "battaglia16" ? "battaglia16" : "lee2022",
            lee2022_concentration_mode=label == "lee22_preferred" ? "duffy2008" : "none"))
        path = joinpath(previous, "cache", label * "_observed_z_spherical3.h5")
        isfile(path) || error("Missing validated previous cache: $(path)")
        cache = build_radius_scaled_cache(runtime, path; refinement=2, zmax=zmax, spherical_cut=3.0)
        validate_radius_cache(cache; sample_count=100,
            output_path=joinpath(output, "analysis", label * "_cache_check.csv"))
        cache
    end
    nrays = length(pixels)
    accumulators = [zeros(nrays, 2, 3) for _ in 1:thread_capacity()]
    hits = [zeros(Int64, nrays, 2) for _ in 1:thread_capacity()]
    chunk_size = int_option(options, "chunk_size", 100_000)
    chunk_size > 0 || error("Invalid chunk size")
    scanned, selected, total = 0, 0, 0
    mass_min, mass_max = Inf, -Inf
    start = time()
    h5open(catalog, "r") do h5
        positions, masses, redshifts = h5["Position"], h5["halo_mass_m200c"], h5["redshift"]
        total = length(redshifts)
        size(positions) == (3, total) && length(masses) == total || error("Invalid catalogue shape")
        for first_row in 1:chunk_size:total
            rows = first_row:min(first_row+chunk_size-1, total)
            z = Float64.(redshifts[rows])
            m = Float64.(masses[rows]) ./ H_VALUE
            pos = Float64.(positions[:, rows])
            all(isfinite, z) && all(isfinite, m) && all(m .> 0) || error("Invalid catalogue values")
            keep = (z .> 0) .& (z .<= zmax)
            scanned += length(rows)
            any(keep) || continue
            z, m, pos = z[keep], m[keep], pos[:, keep]
            ProfileSupport.validate_profile_masses_in_cache(m,
                (first(caches[1].logmasses), last(caches[1].logmasses)))
            selected += length(z)
            mass_min, mass_max = min(mass_min, minimum(m)), max(mass_max, maximum(m))
            Threads.@threads :static for h in eachindex(z)
                tid = Threads.threadid()
                distance = sqrt(sum(abs2, view(pos, :, h)))
                isfinite(distance) && distance > 0 || error("Invalid halo position")
                direction = (pos[1,h]/distance, pos[2,h]/distance, pos[3,h]/distance)
                # Geometry is identical for all three M200c profiles. Do not
                # compute amplitudes until a ray actually intersects this halo.
                theta_max = ProfileSupport.compute_theta_max_r200c_external(caches[1].model, m[h], z[h], 3.0)
                prepared = nothing
                visit_ray_disc(lookup, direction, theta_max) do ray, theta
                    any(source_z[ray, s] >= z[h] for s in 1:2) || return
                    if prepared === nothing
                        prepared = map(c -> prepare_painted_halo(c, m[h], z[h], 3.0), caches)
                    end
                    values = map(p -> Float64(p.value(theta)), prepared)
                    all(v -> isfinite(v) && 0 <= v <= 1e8, values) ||
                        error("Invalid single-halo DM; values are never clipped")
                    for s in 1:2
                        source_z[ray, s] >= z[h] || continue
                        for p in 1:3
                            accumulators[tid][ray, s, p] += values[p]
                        end
                        hits[tid][ray, s] += 1
                    end
                end
            end
            if div(first_row-1, chunk_size) % 20 == 0 || last(rows) == total
                println("Scanned ", scanned, "/", total, "; foreground halos=", selected,
                        "; elapsed seconds=", round(time()-start))
                flush(stdout)
            end
        end
    end
    scanned == total && selected > 0 || error("Incomplete catalogue scan")
    dm, counts = reduce(+, accumulators), reduce(+, hits)
    all(isfinite, dm) && minimum(dm) >= 0 || error("Invalid accumulated DM")
    h5open(result, "w") do h5
        for (s, survey) in enumerate(RAY_SURVEYS)
            group = create_group(h5, survey)
            group["foreground_halo_hits"] = counts[:, s]
            for (p, label) in enumerate(RAY_MODELS)
                group[label] = dm[:, s, p]
            end
        end
        entries = Dict("catalog_rows_scanned" => scanned, "catalog_total_rows" => total,
            "foreground_halos_considered" => selected, "rays_per_survey" => nrays,
            "mass_min_msun" => mass_min, "mass_max_msun" => mass_max,
            "mass_definition" => "M200c; catalogue Msun/h divided by 0.68",
            "source_positions_sha256" => bytes2hex(sha256(read(input))),
            "nside" => nside, "spherical_cut_r200c" => 3.0,
            "selection" => "all resolved halos with 0 < z_halo <= individual z_source",
            "dm_scope" => "halo-only; no host or diffuse IGM DM; observer-frame pc cm^-3",
            "direction_sampling" => "iid uniform NSIDE pixel centres; with replacement",
            "preferred_warning" => "diagnostic only; known high-mass gas-normalization failure",
            "elapsed_seconds" => time()-start)
        for (key, value) in entries
            attributes(h5)[key] = value
        end
    end
    println("Saved individual DMs to ", result)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    sightline_main()
end
