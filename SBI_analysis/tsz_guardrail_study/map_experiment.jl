# Standalone study process. No archived or production source is modified.
using TOML
const STUDY_OUTPUT = ENV["STUDY_OUTPUT"]
mkpath(STUDY_OUTPUT)
ENV["FLAMINGO_MODE"] = "probe"
ENV["FLAMINGO_OUTPUT"] = joinpath(STUDY_OUTPUT, "operator_probe.toml")
include(joinpath(ENV["FLAMINGO_CAMPAIGN"], "code", "process_maps.jl"))
include(joinpath(@__DIR__, "baseline", "stable_los.jl"))

# The wrapper preserves the angular lower bound used by the historical painter.
struct RedshiftCoordinate{I,R}
    interpolator::I
    ranges::R
end
@inline (a::RedshiftCoordinate)(logtheta, z, logmass) = a.interpolator(logtheta, log(z), logmass)
Base.size(a::RedshiftCoordinate) = size(a.interpolator)

const GRID_MODE = get(ENV, "STUDY_GRID", "historical")
if GRID_MODE == "pixel"
    function prepare_tsz_map_for_output(cfg::VisualConfig, raw; niter::Integer=0)
        # Same harmonic beam and band limit as the original NSIDE4096 operator.
        # Only the raw integration/sampling resolution is increased; synthesize
        # back to 4096 before applying the exact historical mask and binning.
        alm = Healpix.map2alm(raw; lmax=healpix_default_lmax(4096), niter=niter)
        beam = Healpix.gaussbeam(deg2rad(cfg.gaussian_beam_fwhm_arcmin/60), alm.lmax)
        Healpix.almxfl!(alm, beam)
        return Healpix.alm2map(alm, 4096)
    end
end
if GRID_MODE != "historical"
    function build_visual_interpolator(cfg::VisualConfig)
        model = build_tsz_model(cfg)
        factor = parse(Int, ENV["STUDY_REFINEMENT"])
        # Retain exactly the same angular interpolation support and floor.
        rft = XGPaint.RadialFourierTransform(n=512, pad=cfg.interpolator_pad)
        lt = LinRange(log(minimum(rft.r)), log(maximum(rft.r)), 512*factor)
        lz = LinRange(log(.001), log(5.), 256*factor)
        lm = LinRange(12., cfg.interpolator_logM_max, 128*factor)
        _, _, _, values = XGPaint.profile_grid(model, lt, exp.(lz), lm)
        @assert all(isfinite, values)
        XGPaint.replace_nonpositive_with_floor!(values)
        itp = XGPaint.Interpolations.interpolate(log.(values),
            XGPaint.BSpline(XGPaint.Cubic(XGPaint.Line(XGPaint.OnGrid()))))
        scaled = XGPaint.scale(itp, lt, lz, lm)
        return XGPaint.LogInterpolatorProfile(model,
            RedshiftCoordinate(scaled, (lt, lz, lm)))
    end
end

function experiment()
    cfg = load_halfdome_fullsky_so_noise_config()
    @assert cfg.base_cfg.cl_lmax == 7979 && cfg.base_cfg.cl_niter == 0
    @assert cfg.base_cfg.gaussian_beam_fwhm_arcmin == 2.0
    @assert cfg.fsky == .4 && cfg.mask_seed == 12345
    save_toml(joinpath(STUDY_OUTPUT, "parameters.toml"),
        Dict(string(k) => Float64(v) for (k,v) in pairs(cfg.base_cfg.battaglia_params)))
    model = build_tsz_model(cfg.base_cfg)
    open(joinpath(STUDY_OUTPUT, "columns.csv"), "w") do stream
        println(stream, "mass,z,radius,stable,native,relative_difference,native_evaluations,native_error_ratio")
        for mass in (1e12, 1e14, 10.0^15.7), z in (.001, .5, 5.), radius in (.003, .1, 1., 4., 1000.)
            slice = XGPaint.prepare_profile_slice(model, mass, z)
            stable = XGPaint._nfw_profile_los_quadrature(radius, slice.xc, slice.alpha, slice.beta, slice.gamma)
            evaluations = Ref(0)
            integrand(l) = begin
                evaluations[] += 1
                1e9*XGPaint.generalized_nfw(hypot(l, radius), slice.xc, slice.alpha, slice.beta, slice.gamma)
            end
            native, err = XGPaint.QuadGK.quadgk(integrand, 0., 1e5; rtol=1e-10, order=9, maxevals=4096)
            difference = native > 0 ? abs(stable/(2native/1e9)-1) : NaN
            println(stream, join((mass, z, radius, stable, 2native/1e9, difference,
                evaluations[], native > 0 ? err/native : NaN), ","))
        end
    end
    started = time()
    signal = paint_halfdome_fullsky_signal_map(cfg.base_cfg)
    unmasked = compute_cl(cfg.base_cfg, signal)
    write_npy_float64_vector(joinpath(STUDY_OUTPUT, "unmasked_clean_cl.npy"), unmasked, "study clean")
    # Recreate the original deterministic mask, avoiding two unused noise maps.
    maskinfo = random_apodized_cap_mask(4096, cfg.fsky,
        cfg.mask_apodization_arcmin, MersenneTwister(cfg.mask_seed))
    signal.pixels .*= maskinfo.mask.pixels
    masked = compute_cl(cfg.base_cfg, signal)
    write_npy_float64_vector(joinpath(STUDY_OUTPUT, "masked_clean_cl.npy"), masked, "study masked clean")
    save_toml(joinpath(STUDY_OUTPUT, "numerics.toml"), Dict(
        "grid" => GRID_MODE, "refinement" => get(ENV, "STUDY_REFINEMENT", "1"),
        "nside" => cfg.base_cfg.nside, "seconds" => time()-started,
        "output_nside" => 4096, "beam_lmax" => healpix_default_lmax(4096),
        "mask_pixel_sha256" => pixel_sha(maskinfo.mask),
        "cache_nonpositive_replaced" => XGPaint.extended_cleanup_stats[].count,
        "cache_floor" => XGPaint.extended_cleanup_stats[].floor))
end
experiment()
