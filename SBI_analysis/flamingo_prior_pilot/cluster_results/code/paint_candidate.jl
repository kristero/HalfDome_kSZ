# Reuse the frozen FLAMINGO/HalfDome campaign operators and exact noise cache.
# The original analysis files and trained bundle are never modified.
using TOML

const PILOT_OUTPUT = ENV["PILOT_OUTPUT"]
mkpath(PILOT_OUTPUT)
# Permit a requested runner restart between maps, without interrupting an
# active transform or leaving a detached Julia calculation behind.
const RESTART_CONTROL = joinpath(dirname(dirname(PILOT_OUTPUT)), "restart_before_next_map_for_job.txt")
if isfile(RESTART_CONTROL) && strip(read(RESTART_CONTROL, String)) == get(ENV, "PBS_JOBID", "")
    error("Requested runner restart before this map to load the revised numerical-search guard")
end
ENV["FLAMINGO_MODE"] = "probe"
const VALIDATE_ONLY = get(ENV, "PILOT_VALIDATE_ONLY", "0") == "1"
const AUDIT_ONLY = get(ENV, "PILOT_AUDIT_ONLY", "0") == "1"
ENV["FLAMINGO_OUTPUT"] = joinpath(PILOT_OUTPUT, VALIDATE_ONLY ? "parameters_probe.toml" : "operator_probe.toml")
include(joinpath(ENV["FLAMINGO_CAMPAIGN"], "code", "process_maps.jl"))

function candidate_main()
    cfg = load_halfdome_fullsky_so_noise_config()
    save_toml(joinpath(PILOT_OUTPUT, "candidate_parameters.toml"),
        Dict(string(key) => Float64(value) for (key, value) in pairs(cfg.base_cfg.battaglia_params)))
    @assert cfg.base_cfg.nside == 4096 && cfg.base_cfg.cl_lmax == 7979
    @assert cfg.base_cfg.cl_niter == 0 && cfg.noise_lmax == 7979
    @assert cfg.noise_seed == cfg.mask_seed == 12345 && cfg.noise_deprojection == 0
    @assert cfg.base_cfg.gaussian_beam_fwhm_arcmin == 2.0
    @assert cfg.fsky == 0.4 && cfg.mask_apodization_arcmin == 60.0
    # The original hard bounds describe the old training design. The new
    # driver checks the authorized proposed bounds, while the derived physical
    # checks remain active over the entire grid used by interpolation.
    reasons = validate_battaglia_params(cfg.base_cfg.battaglia_params;
        enforce_prior_bounds=false, enforce_derived_bounds=true,
        logM_min=12.0, logM_max=15.7, z_max=5.0, beta_outer_min=1.05)
    isempty(reasons) || error(join(reasons, "\n"))
    VALIDATE_ONLY && return
    if !AUDIT_ONLY && !isfile(joinpath(PILOT_OUTPUT, "complete.toml"))
        process_signal(ENV["FLAMINGO_CAMPAIGN"], PILOT_OUTPUT, "halfdome", cfg)
    end
    GC.gc()

    # Check the actual production interpolation against direct LOS quadrature.
    # No new interpolation scheme or radial cutoff is substituted here.
    model = build_tsz_model(cfg.base_cfg)
    cache = visual_interpolator_cache_paths(cfg.base_cfg).primary
    grid = XGPaint.JLD2.load(cache)
    raw = grid["prof_y"]
    grid_nonpositive = count(x -> x <= 0, raw)
    grid_nonfinite = count(x -> !isfinite(x), raw)
    interpolator = build_interpolator(model; cache_file=cache, overwrite=false, verbose=false)
    relative_errors = Float64[]
    significant_errors = Float64[]
    point_rows = Vector{Float64}[]
    for mass in (1e12, 1e13, 1e14, 1e15, 10.0^15.7), z in (.01, .1, .5, 1.0, 3.0, 5.0)
        theta200 = XGPaint.compute_θmax(model, mass * XGPaint.M_sun, z) / 4
        radii = [.003, .01, .03, .1, .3, 1.0, 2.0, 4.0]
        direct = [Float64(model(x*theta200, mass, z)) for x in radii]
        predicted = [Float64(interpolator(x*theta200, mass, z)) for x in radii]
        scale = maximum(direct)
        for (x, actual, estimate) in zip(radii, direct, predicted)
            err = actual > 0 ? abs(estimate/actual-1) : NaN
            if isfinite(err)
                push!(relative_errors, err)
                actual > 1e-8*scale && push!(significant_errors, err)
            end
            push!(point_rows, [mass, z, x, actual, estimate, err])
        end
    end
    open(joinpath(PILOT_OUTPUT, "profile_interpolation_check.csv"), "w") do stream
        println(stream, "mass_Msun,z,r_over_R200,direct_y,interpolated_y,relative_error")
        for row in point_rows
            println(stream, join(row, ","))
        end
    end
    sort!(significant_errors)
    p95 = significant_errors[ceil(Int, .95*length(significant_errors))]
    save_toml(joinpath(PILOT_OUTPUT, "profile_validation.toml"), Dict(
        "grid_nonpositive" => grid_nonpositive, "grid_nonfinite" => grid_nonfinite,
        "grid_entries" => length(raw), "checked_points" => length(point_rows),
        "interpolator_theta_min_rad" => XGPaint.compute_θmin(interpolator),
        "max_relative_error" => maximum(relative_errors),
        "max_significant_relative_error" => maximum(significant_errors),
        "p95_significant_relative_error" => p95,
        "status" => p95 < .01 ? "p95_below_one_percent" : "needs_interpolation_refinement",
        "note" => "Significant means direct y exceeds 1e-8 of the central sampled value for its mass/redshift"))
    println("Candidate completed with interpolation p95 = ", p95)
end

candidate_main()
