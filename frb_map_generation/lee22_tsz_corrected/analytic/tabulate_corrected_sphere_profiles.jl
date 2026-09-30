# Tabulate the per-halo DM "gas inside the R200c sphere" for the analytic (Torkamani et al. 2026) halo-model DM PDF,
# for the corrected Lee22 densities of 2026-09-30 plus the two reference models. Same nodes, quadrature and self-check
# as ../../analytic_pdf_torkamani2026/tabulate_halfdome_sphere_profiles.jl (copied functions below), every model through
# Battaglia16DensityDMProfile(Lee2022XGPaintDMProfile(...)) as in the ray generator (DM_PROFILE=lee2022_xgpaint):
#   b16                      Battaglia16 (XGPaint HaloDMProfile)
#   lee22_noconc_xgpnative   normalization=:xgpaint_ne2d (production, 0.6016 x literal)
#   lee22_noconc_tngcal      literal x C(z) = 0.2842 (1+min(z,2))^2.483
#   lee22_noconc_a3fn        literal x (1+z)^3 / F_n (F_n from ENV LEE2022_BOOKKEEPING_FN, default 5.4)
#   lee22_noconc_phys        literal x (0.6774 (1+z))^3 X_H^2 (derived bookkeeping correction)
# Usage: julia -t N tabulate_corrected_sphere_profiles.jl <output.h5>
using XGPaint, Unitful, UnitfulAstro, HDF5, Printf, SHA
include(joinpath(@__DIR__, "..", "..", "lee2022_frb_dm_profile.jl"))

const OUT = ARGS[1]
const H = 0.68
const LOG10M = collect(9.005:0.01:15.995)
const ZS = collect(0.005:0.01:0.995)
const XS = vcat(10 .^ range(-4, log10(0.3); length=90), collect(range(0.31, 0.9995; length=110)))
const quad = lee2022_quadgk_function()
lee(norm, zs) = Battaglia16DensityDMProfile(Lee2022XGPaintDMProfile(Lee2022NoConcentrationDMProfile(Omega_c=0.261,
    Omega_b=0.049, h=H, normalization=norm, n0_pivot=:mcut, shape_clip_mass_msun=10.0^14.8 / H, redshift_scaling=zs)))
const MODELS = (
    ("b16", Battaglia16DensityDMProfile(XGPaint.HaloDMProfile(XGPaint.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=H)))),
    ("lee22_noconc_xgpnative", lee(:xgpaint_ne2d, :physical)),
    ("lee22_noconc_tngcal", lee(:literal, :tng_calibrated)),
    ("lee22_noconc_a3fn", lee(:literal, :bookkeeping_a3)),
    ("lee22_noconc_phys", lee(:literal, :bookkeeping_physical)),
)

"""n_e(x) [m^-3] with the per-halo parameters hoisted (identical arithmetic to halo_electron_density_m3)."""
function density_closure(model::Battaglia16DensityDMProfile, mass_msun, z)
    inner = model.inner
    par = XGPaint.get_params(inner, mass_msun * XGPaint.M_sun, z)
    rho_crit = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(inner, z)
    me = XGPaint.constants.ElectronMass
    mH = XGPaint.constants.ProtonMass
    xH = 0.76
    factor = me + (2xH / (xH + 1)) * mH + ((1 - xH) / (2(1 + xH))) * 4mH
    scale = Float64(ustrip(uconvert(u"m^-3", 0.9 * par.P₀ * inner.f_b * rho_crit / factor)))
    xc, a, b, g = par.xc, par.α, par.β, par.γ
    return x -> scale * XGPaint.generalized_nfw(x, xc, a, b, g)
end

function tabulate(model, name)
    nm, nz, nx = length(LOG10M), length(ZS), length(XS)
    g = zeros(nx, nz, nm)
    r200 = zeros(nz, nm)
    theta200 = zeros(nz, nm)
    Threads.@threads for idx in 1:(nm * nz)
        j = (idx - 1) ÷ nz + 1
        i = (idx - 1) % nz + 1
        mass = 10.0^LOG10M[j]; z = ZS[i]
        r200q = getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, mass * XGPaint.M_sun, z, 200)
        r200m = Float64(ustrip(uconvert(u"m", r200q)))
        r200[i, j] = Float64(ustrip(uconvert(u"Mpc", r200q)))
        theta200[i, j] = Float64(XGPaint.angular_size(model, r200q, z))
        ne = density_closure(model, mass, z)
        fac = r200m * M2_TO_PC_CM3_LOCAL / (1 + z)
        for k in 1:nx
            x = XS[k]
            L = sqrt(1 - x^2)
            integral, _ = quad(l -> ne(sqrt(x^2 + l^2)), 0.0, L; rtol=1.0e-8, order=9)
            g[k, i, j] = 2 * integral * fac / (2 * L)
        end
    end
    # self-check against the production function at scattered nodes
    worst = 0.0
    for (k, i, j) in ((1, 1, 1), (45, 10, 60), (90, 25, 81), (150, 40, 300), (200, 100, 700), (120, 77, 450))
        mass = 10.0^LOG10M[j]; z = ZS[i]; x = XS[k]; L = sqrt(1 - x^2)
        ref = chord_dm_pc_cm3(model, x, L, mass, z) / (2L)
        worst = max(worst, abs(g[k, i, j] / ref - 1))
    end
    @printf("%s: max relative difference to chord_dm_pc_cm3 at check nodes = %.2e\n", name, worst)
    worst < 1e-7 || error("tabulation does not reproduce chord_dm_pc_cm3 for $(name)")
    return g, r200, theta200
end

function main()
    mkpath(dirname(OUT))
    h5open(OUT, "w") do f
        f["log10_m200c_msun"] = LOG10M
        f["redshift"] = ZS
        f["x_r200c"] = XS
        for (name, model) in MODELS
            t0 = time()
            g, r200, theta200 = tabulate(model, name)
            f["$(name)/chord_mean_dm_pc_cm3"] = g
            f["$(name)/r200c_mpc"] = r200
            f["$(name)/theta200c_rad"] = theta200
            @printf("%s done in %.1f s\n", name, time() - t0); flush(stdout)
        end
        attrs(f)["description"] = "g = DM_sphere/(2 sqrt(1-x^2)) [pc cm^-3], observer frame, gas inside the R200c sphere; arrays (x, z, logM) in Julia order"
        attrs(f)["profile_file_sha256"] = bytes2hex(open(sha256, joinpath(@__DIR__, "..", "..", "lee2022_frb_dm_profile.jl")))
        attrs(f)["cosmology"] = "Omega_c=0.261, Omega_b=0.049, h=0.68 (XGPaint profile cosmology of the generator)"
        attrs(f)["lee22_bookkeeping_fn"] = LEE2022_BOOKKEEPING_FN
        attrs(f)["lee22_signatures"] = join([lee2022_cache_signature(m.inner.lee) for (n, m) in MODELS[2:end]], " || ")
    end
    println("wrote ", OUT)
end

main()
