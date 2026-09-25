using TOML
const ROW_OUTPUT = ENV["EXTENDED_ROW_OUTPUT"]
mkpath(ROW_OUTPUT)
ENV["FLAMINGO_MODE"] = "probe"
ENV["FLAMINGO_OUTPUT"] = joinpath(ROW_OUTPUT, "operator_probe.toml")
include(joinpath(ENV["FLAMINGO_CAMPAIGN"], "code", "process_maps.jl"))
include(joinpath(@__DIR__, "stable_los.jl"))
include(joinpath(@__DIR__, "independent_noise.jl"))

function paint_extended_row()
    cfg = load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.nside == 4096 && cfg.base_cfg.cl_lmax == 7979
    @assert cfg.base_cfg.cl_niter == 0 && cfg.noise_lmax == 7979
    @assert cfg.noise_seed == cfg.mask_seed == 12345 && cfg.noise_deprojection == 0
    @assert cfg.base_cfg.gaussian_beam_fwhm_arcmin == 2.0
    @assert cfg.fsky == 0.4 && cfg.mask_apodization_arcmin == 60.0
    parsed = Dict(string(k) => Float64(v) for (k,v) in pairs(cfg.base_cfg.battaglia_params))
    save_toml(joinpath(ROW_OUTPUT, "parameters.toml"), parsed)
    reasons = validate_battaglia_params(cfg.base_cfg.battaglia_params;
        enforce_prior_bounds=false, enforce_derived_bounds=true,
        logM_min=12.0, logM_max=15.7, z_max=5.0, beta_outer_min=3.09)
    isempty(reasons) || error(join(reasons, "\n"))
    process_signal(ENV["FLAMINGO_CAMPAIGN"], ROW_OUTPUT, "halfdome", cfg)
    model = build_tsz_model(cfg.base_cfg)
    # Bounded native-vs-scaled checks at radii actually used for painting.
    maximum_error = 0.0
    for mass in (1e12, 1e14, 10.0^15.7), z in (.01, .5, 5.), x in (.003, .1, 1., 4.)
        slice = XGPaint.prepare_profile_slice(model, mass, z)
        native, err = XGPaint.QuadGK.quadgk(l -> 1e9*XGPaint.generalized_nfw(
            hypot(l,x), slice.xc, slice.alpha, slice.beta, slice.gamma),
            0., 1e5; rtol=1e-10, order=9, maxevals=4096)
        native > 0 && err <= 5e-10*native || error("Native validation failed")
        actual = XGPaint._nfw_profile_los_quadrature(x, slice.xc, slice.alpha, slice.beta, slice.gamma)
        maximum_error = max(maximum_error, abs(actual/(2native/1e9)-1))
    end
    @assert maximum_error < 1e-7
    save_toml(joinpath(ROW_OUTPUT, "numerics.toml"), Dict(
        "stable_los_sha256" => file_sha(joinpath(@__DIR__, "stable_los.jl")),
        "maximum_column_relative_error" => maximum_error, "column_checks" => 108,
        "cache_nonpositive_replaced" => XGPaint.extended_cleanup_stats[].count,
        "cache_floor" => XGPaint.extended_cleanup_stats[].floor,
        "map_definition" => "unchanged historical HalfDome; scaled finite LOS quadrature"))
end
paint_extended_row()
