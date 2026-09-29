# Verifier A: decompose the v1 pixel-check residuals. For my discs (v1b_disc_*.csv), paint with the PRODUCTION caches using
# the painter's arithmetic (theta < atan(4 R200c/dA), itp(max(theta, theta_min), M, z)) and with XGPaint's direct LOS
# quadrature model(theta, M, z) (no interpolation). Also tabulates itp/direct at low z and near the Lee22 shape clip.
# Read-only on caches/maps; writes CSVs to the verifier output dir.
using XGPaint, Interpolations, Healpix, Printf, Unitful, UnitfulAstro
const XG = XGPaint
include("/home/cbllover/HalfDome/frb_map_generation/lee2022_tsz_pressure_profile.jl")
const O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
const REPO = "/home/cbllover/HalfDome/frb_map_generation"
const H = 0.68
b12_model() = Battaglia16ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758, x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0, beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)
function thmax_painter(model, m, z, mult)
    rho = XG.ρ_crit(model, z); r200 = ∛(m * XG.M_sun / (4pi / 3 * 200 * rho))
    return Float64(XG.angular_size(model, mult * r200, z))
end
tmin_local(itp) = exp(Float64(first(first(itp.itp.ranges))))
maps = [("b12_allz", b12_model(), "/home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2", 0.0, Inf, Inf),
        ("lee22_allz", Lee2022ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H),
         joinpath(REPO, "outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2"), 0.0, Inf, Inf),
        ("lee22_calib", Lee2022ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H),
         joinpath(REPO, "outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2"),
         1.4705882352941176e13, 9.278784477649915e14, 2.0)]
# read exports
pixrows = [split(l, ",") for l in readlines(joinpath(O, "v1b_disc_pixels.csv"))[2:end]]
halrows = [split(l, ",") for l in readlines(joinpath(O, "v1b_disc_halos.csv"))[2:end]]
res = Healpix.Resolution(4096)
itps = Dict{String,Any}()
open(joinpath(O, "v1c_disc_itp_direct.csv"), "w") do io
    println(io, "disc,map,pix_ring0,y_itp,y_direct,y_itp_lowz_part,y_direct_lowz_part")
    for (mname, model, cache, mmin, mmax, zmax) in maps
        itp = get!(itps, cache) do
            build_interpolator(model; cache_file=cache, overwrite=false, verbose=false)
        end
        tmin = tmin_local(itp)
        for disc in unique(r[1] for r in pixrows)
            pix = [parse(Int, r[2]) for r in pixrows if r[1] == disc]
            hs = [r for r in halrows if r[1] == disc]
            yi = zeros(length(pix)); yd = zeros(length(pix)); yil = zeros(length(pix)); ydl = zeros(length(pix))
            for r in hs
                m = parse(Float64, r[3]); z = parse(Float64, r[4])
                (m >= mmin && m < mmax && z <= zmax) || continue
                x, y, w = parse(Float64, r[5]), parse(Float64, r[6]), parse(Float64, r[7])
                d = sqrt(x^2 + y^2 + w^2); ux, uy, uz = x / d, y / d, w / d
                tmax = min(thmax_painter(model, m, z, 4.0), pi)
                for (k, p) in enumerate(pix)
                    px, py, pz = Healpix.pix2vecRing(res, p + 1)
                    th = acos(clamp(ux * px + uy * py + uz * pz, -1.0, 1.0))
                    th < tmax || continue
                    a = Float64(itp(max(th, tmin), m, z)); b = Float64(model(th, m, z))
                    yi[k] += a; yd[k] += b
                    if z < 0.05
                        yil[k] += a; ydl[k] += b
                    end
                end
            end
            for k in eachindex(pix)
                @printf(io, "%s,%s,%d,%.15e,%.15e,%.15e,%.15e\n", disc, mname, pix[k], yi[k], yd[k], yil[k], ydl[k])
            end
            println(mname, " ", disc, " done")
        end
    end
end
# grid: itp/direct - 1 at low z and near the shape clip, x = theta/theta200
open(joinpath(O, "v1c_itp_vs_direct_grid.csv"), "w") do io
    println(io, "map,log10M,z,x,y_itp,y_direct,rel")
    for (mname, model, cache, _, _, _) in maps[1:2]
        itp = itps[cache]; tmin = tmin_local(itp)
        for lm in (12.5, 13.0, 13.5, 14.0, 14.5, 14.9, 14.95, 14.9675, 14.98, 15.0, 15.3),
            z in (0.0022, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 2.0, 3.0), x in (0.02, 0.1, 0.3, 1.0, 2.0, 3.9)
            m = 10.0^lm
            th = x * thmax_painter(model, m, z, 1.0)
            a = Float64(itp(max(th, tmin), m, z)); b = Float64(model(th, m, z))
            @printf(io, "%s,%.4f,%.4f,%.3f,%.10e,%.10e,%.6e\n", mname, lm, z, x, a, b, a / b - 1)
        end
    end
end
itp = itps["/home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2"]
println("B12 cache ranges: ", [ (first(r), last(r), length(r)) for r in itp.itp.ranges])
itp = itps[joinpath(REPO, "outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2")]
println("Lee22 cache ranges: ", [ (first(r), last(r), length(r)) for r in itp.itp.ranges])
println("itp type: ", typeof(itp.itp).name)
