VERSION == v"1.12.2" || error("This dataset requires Julia 1.12.2 for reproducible RNG and dependencies.")
include(joinpath(@__DIR__, "simulator", "tSZ_visuals", "run_halfdome_fullsky_so_noise.jl"))
include(joinpath(@__DIR__, "safe_paint.jl"))
# The included driver sets a default beam ENV even when a CLI value was supplied.
# This wrapper's explicit arguments are authoritative.
for (key, env_key) in (("apply_gaussian_beam", "APPLY_GAUSSIAN_BEAM"),
                        ("gaussian_beam_fwhm_arcmin", "GAUSSIAN_BEAM_FWHM_ARCMIN"))
    argument = only(filter(a -> arg_matches_key(a, key), ARGS))
    ENV[env_key] = split(argument, '='; limit=2)[2]
end
realpath(pathof(XGPaint)) == realpath(joinpath(@__DIR__, "vendor", "XGPaint", "src", "XGPaint.jl")) ||
    error("Wrong XGPaint source; use the bundled --project path.")
# Bounds come from the validated config/CSV, not the old hard-coded Julia prior.
# Retain the independent physical positivity and LOS-convergence safeguards.
preflight = load_halfdome_fullsky_so_noise_config()
reasons = validate_battaglia_params(preflight.base_cfg.battaglia_params;
    enforce_prior_bounds=false, enforce_derived_bounds=true,
    logM_max=preflight.base_cfg.interpolator_logM_max)
isempty(reasons) || error(join(reasons, "\n"))
if "--validate-only" in ARGS
    println("Validated beam: ", preflight.base_cfg.gaussian_beam_fwhm_arcmin)
    println("Validated deprojections: ", join(preflight.noise_deprojections, ","))
    println("Validated mask seed: ", preflight.mask_seed)
    println("Validated noise seed: ", preflight.noise_seed)
    println("PASSED: configuration and physical pressure guardrails; no maps generated.")
    exit(0)
end
result = run_halfdome_fullsky_so_noise()
result === nothing && error("Simulation was skipped; no completed row can be recorded.")
println("Actual beam FWHM: ", result.cfg.base_cfg.gaussian_beam_fwhm_arcmin)
println("Painter: ring_locked")
for d in result.cfg.noise_deprojections
    cfg = with_noise_deprojection(result.cfg, d)
    for (case, enabled) in (("baseline", cfg.save_baseline_noise_cross_cl),
                            ("goal", cfg.save_goal_noise_cross_cl))
        enabled || continue
        println("Actual split seeds ", case, "_deproj", d, ": ",
                noise_split_seed(cfg, case, 1), ",", noise_split_seed(cfg, case, 2))
    end
end
p = result.cfg.base_cfg.battaglia_params
theta = (p.P0_amp, p.x_c_amp, p.beta_amp, p.P0_alpha_m, p.x_c_alpha_m,
         p.beta_alpha_m, p.P0_alpha_z, p.x_c_alpha_z, p.beta_alpha_z)
println("Actual theta: [", join(theta, ","), "]")
