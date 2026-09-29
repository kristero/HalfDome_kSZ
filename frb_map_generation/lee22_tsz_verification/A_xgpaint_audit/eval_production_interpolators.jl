# Workstream A, goal 2 (Julia half): evaluate the PRODUCTION Compton-y interpolators exactly as the HalfDome
# tSZ painter (frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl) builds and calls them, plus
# XGPaint's direct (non-interpolated) LOS quadrature, on a (M, z, x) grid. The Python half
# (compare_painted_profiles.py) adds an independent line-of-sight integral of lee22_models.py.
#
# Read-only: loads the two caches with build_interpolator(...; overwrite=false) (no write), includes the
# repository's lee2022_tsz_pressure_profile.jl unchanged. Writes only into the workstream output directory.
#
# Also runs a "cache vintage" test: stored cache values on the native (logtheta, z, logM) grid nodes are
# recomputed with the CURRENT XGPaint code (fill_profile_slice!, the same call profile_grid uses), which
# tells whether a change of XGPaint since the cache was built would change the painted profile.

using XGPaint
using Interpolations
using Printf
using Unitful, UnitfulAstro
const XG = XGPaint

const REPO = "/home/cbllover/HalfDome/frb_map_generation"
include(joinpath(REPO, "lee2022_tsz_pressure_profile.jl"))

const OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit"
const B12_CACHE = "/home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2"
const LEE_CACHE = joinpath(REPO, "outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2")
const H = 0.68

# exactly paint_halfdome_battaglia12_tsz_map.jl: build_battaglia12_fiducial_model()
b12_model() = Battaglia16ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758,
    x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0,
    beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)

# exactly generate_halfdome_z1_dm_mass_windows.jl: compute_theta_max_r200c_external (multiplier 1)
function theta200_painter(model, m_msun, z)
    rho = XG.ρ_crit(model, z)
    r200 = ∛(m_msun * XG.M_sun / (4pi / 3 * 200 * rho))
    return Float64(XG.angular_size(model, r200, z)), r200
end

theta_min_local(itp) = exp(Float64(first(first(itp.itp.ranges))))

function main()
    mkpath(OUT)
    models = [("b12", b12_model(), B12_CACHE), ("lee22", Lee2022ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H), LEE_CACHE)]
    logms = [13.0, 13.5, 14.0, 14.5, 15.0]
    zs = [0.1, 0.5, 1.0, 2.0]
    xs = exp.(range(log(0.02), log(4.0), length=30))

    open(joinpath(OUT, "production_interpolator_eval.csv"), "w") do io
        println(io, "model,log10M200c_msun,z,x_r200c,theta_rad,theta_eval_rad,y_interpolator,y_xgpaint_direct,",
                    "theta200_rad,r200c_mpc,rho_crit_xgpaint_kg_m3,dA_xgpaint_mpc,inside_aperture")
        for (name, model, cache) in models
            itp = build_interpolator(model; cache_file=cache, overwrite=false, verbose=false)
            tmin = theta_min_local(itp)
            @printf("%s cache=%s theta_min=%.3e ranges: logtheta [%.3f, %.3f] z [%.4f, %.4f] logM [%.3f, %.3f] size=%s\n",
                name, cache, tmin, first(itp.itp.ranges[1]), last(itp.itp.ranges[1]), first(itp.itp.ranges[2]),
                last(itp.itp.ranges[2]), first(itp.itp.ranges[3]), last(itp.itp.ranges[3]), string(size(itp.itp)))
            for lm in logms, z in zs
                m = 10.0^lm
                th200, r200 = theta200_painter(model, m, z)
                thmax = XG.angular_size(model, 4.0 * r200, z)   # painter's 4 R200c aperture
                rho = Float64(ustrip(uconvert(u"kg/m^3", XG.ρ_crit(model, z))))
                dA = Float64(ustrip(uconvert(u"Mpc", XG.angular_diameter_dist(model.cosmo, z))))
                r200_mpc = Float64(ustrip(uconvert(u"Mpc", r200)))
                for x in xs
                    th = x * th200
                    th_eval = max(th, tmin)      # painter clamps at theta_min
                    yi = Float64(itp(th_eval, m, z))
                    yd = Float64(model(th, m, z))  # XGPaint direct LOS quadrature, current code
                    inside = th < thmax
                    @printf(io, "%s,%.4f,%.4f,%.10e,%.10e,%.10e,%.12e,%.12e,%.10e,%.10e,%.10e,%.10e,%d\n",
                        name, lm, z, x, th, th_eval, yi, yd, th200, r200_mpc, rho, dA, inside)
                end
            end
        end
    end

    # ---- cache vintage test on native grid nodes (no interpolation)
    open(joinpath(OUT, "cache_vintage_gridnode_test.csv"), "w") do io
        println(io, "model,iz,z,im,log10M,n_theta,max_abs_rel_diff,median_abs_rel_diff,stored_at_theta_near_r200,current_at_theta_near_r200")
        for (name, model, cache) in models
            g = XG.load(cache)
            lt = g[haskey(g, "prof_logthetas") ? "prof_logthetas" : "prof_logθs"]
            zg = g["prof_redshift"]; lmg = g["prof_logMs"]; A = g["prof_y"]
            @printf("%s: stored grid logtheta %d [%.4f, %.4f], z %d [%.4f, %.4f], logM %d [%.4f, %.4f]\n", name,
                length(lt), first(lt), last(lt), length(zg), first(zg), last(zg), length(lmg), first(lmg), last(lmg))
            izs = unique(round.(Int, range(1, length(zg), length=9)))
            ims = unique(round.(Int, range(1, length(lmg), length=9)))
            worst = 0.0
            for iz in izs, im in ims
                dest = zeros(Float64, length(lt))
                XG.fill_profile_slice!(dest, model, lt, 10.0^lmg[im], zg[iz])
                stored = A[:, iz, im]
                ok = (stored .> 0) .& (dest .> 0)
                rel = abs.(stored[ok] ./ dest[ok] .- 1)
                th200, _ = theta200_painter(model, 10.0^lmg[im], zg[iz])
                k = argmin(abs.(exp.(lt) .- th200))
                mx = isempty(rel) ? NaN : maximum(rel)
                worst = max(worst, mx)
                @printf(io, "%s,%d,%.6f,%d,%.6f,%d,%.6e,%.6e,%.10e,%.10e\n", name, iz, zg[iz], im, lmg[im], count(ok),
                    mx, isempty(rel) ? NaN : sort(rel)[cld(length(rel), 2)], stored[k], dest[k])
            end
            @printf("%s: cache-vs-current-code max |rel diff| over %d nodes x %d thetas = %.3e\n", name,
                length(izs) * length(ims), length(lt), worst)
        end
    end
    println("wrote ", OUT)
end

main()
