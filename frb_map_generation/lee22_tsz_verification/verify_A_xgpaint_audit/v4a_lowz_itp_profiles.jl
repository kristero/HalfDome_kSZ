# Verifier A: production-interpolator y(x) for all catalogue halos with z < 0.1 (painter arithmetic), on
# x = theta/theta200 in 128 log points [1e-3, 3.999]; binary output for v4b_lowz_impact.py. Read-only on caches.
using XGPaint, Interpolations, Unitful, UnitfulAstro
const XG = XGPaint
include("/home/cbllover/HalfDome/frb_map_generation/lee2022_tsz_pressure_profile.jl")
const S = ARGS[1]   # scratch dir with lowz_mz.bin
const H = 0.68
b12_model() = Battaglia16ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758, x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0, beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)
function th_painter(model, m, z, mult)
    rho = XG.ρ_crit(model, z); r200 = ∛(m * XG.M_sun / (4pi / 3 * 200 * rho))
    return Float64(XG.angular_size(model, mult * r200, z))
end
raw = reinterpret(Float64, read(joinpath(S, "lowz_mz.bin")))
N = length(raw) ÷ 2; mz = reshape(raw, 2, N)
xs = exp.(range(log(1e-3), log(3.999), length=128))
for (name, model, cache) in (("b12", b12_model(), "/home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2"),
        ("lee22", Lee2022ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H),
         "/home/cbllover/HalfDome/frb_map_generation/outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2"))
    itp = build_interpolator(model; cache_file=cache, overwrite=false, verbose=false)
    tmin = exp(Float64(first(first(itp.itp.ranges))))
    out = zeros(Float64, 130, N)
    Threads.@threads for i in 1:N
        m = mz[1, i]; z = mz[2, i]
        t200 = th_painter(model, m, z, 1.0); tmax = th_painter(model, m, z, 4.0)
        out[1, i] = t200; out[2, i] = tmax
        for (k, x) in enumerate(xs)
            out[2 + k, i] = Float64(itp(max(x * t200, tmin), m, z))
        end
    end
    write(joinpath(S, "lowz_itp_$(name).bin"), out)
    println(name, " done N=", N)
end
