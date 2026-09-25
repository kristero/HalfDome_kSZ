# Diagnose native quadrature cost with an explicit evaluation budget.
# This file never changes a production profile or interpolation cache.
using TOML
const OUT = ENV["PILOT_OUTPUT"]
mkpath(OUT)
ENV["FLAMINGO_MODE"] = "probe"
ENV["FLAMINGO_OUTPUT"] = joinpath(OUT, "operator_probe.toml")
include(joinpath(ENV["FLAMINGO_CAMPAIGN"], "code", "process_maps.jl"))

function scaled_log_column(x, xc, beta_raw; endpoint=1e5)
    # y=x*sinh(u), so radius=x*cosh(u). The peak of the transformed
    # integrand is at r/xc=0.7/(beta_raw-0.7), clipped to this LOS.
    logx = log(x)
    logxc = log(xc)
    upper = asinh(endpoint/x)
    peak = clamp(logxc+log(.7/(beta_raw-.7)), logx, log(hypot(x, endpoint)))
    logshape(logr) = .7*logr+.3*logxc-beta_raw*log1p(exp(logr-logxc))
    normalization = logshape(peak)
    evaluations = Ref(0)
    function integrand(u)
        evaluations[] += 1
        return exp(logshape(logx+log(cosh(u)))-normalization)
    end
    value, error = XGPaint.QuadGK.quadgk(integrand, 0., upper;
        rtol=1e-10, order=9, maxevals=4096)
    return log(2)+normalization+log(value), error/value, evaluations[]
end

function main_diagnostic()
    cfg = load_halfdome_fullsky_so_noise_config()
    proposed = build_tsz_model(cfg.base_cfg)
    reference = Battaglia16ThermalSZProfile(Omega_c=cfg.base_cfg.cosmo_omegac,
        Omega_b=cfg.base_cfg.cosmo_omegab, h=cfg.base_cfg.cosmo_h)
    open(joinpath(OUT, "quadrature_cost_dense.csv"), "w") do stream
        println(stream, "case,mass_Msun,z,r_over_R200,theta_rad,native_evaluations,native_relative_error,native_y,scaled_evaluations,scaled_relative_error,log_y_scaled,outside_painted_disc")
        for (name, model) in (("Battaglia12", reference), ("slow_fit_trial", proposed))
            for mass in (1e12, 1e14, 10.0^15.7), z in (.1, 1., 5.)
                prepared = XGPaint.prepare_profile_slice(model, mass, z)
                for x in vcat([.003, .1, 1., 4.], 10.0 .^ range(2, 8; length=49))
                    theta = x*prepared.theta_scale
                    theta <= exp(11.49493652) || continue
                    count = Ref(0)
                    function original(y)
                        count[] += 1
                        return 1e9*XGPaint.generalized_nfw(hypot(y, x), prepared.xc,
                            prepared.alpha, prepared.beta, prepared.gamma)
                    end
                    value, error = XGPaint.QuadGK.quadgk(original, 0., 1e5;
                        rtol=1e-12, order=9, maxevals=4096)
                    native_y = prepared.amplitude*2value/1e9
                    native_relative = value > 0 ? error/value : NaN
                    raw_beta = prepared.beta+prepared.gamma
                    logcolumn, scaled_error, scaled_count = scaled_log_column(x, prepared.xc, raw_beta)
                    logy = log(prepared.amplitude)+logcolumn
                    println(stream, join((name, mass, z, x, theta, count[], native_relative,
                        native_y, scaled_count, scaled_error, logy, x > 4), ","))
                end
            end
        end
    end
    println("Bounded native/scaled quadrature comparison completed")
end

main_diagnostic()
