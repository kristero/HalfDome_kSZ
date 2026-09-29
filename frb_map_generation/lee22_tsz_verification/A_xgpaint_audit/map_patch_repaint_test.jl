# Workstream A, goal 1 (map level): repaint three small discs of the PRODUCTION y maps with today's code and compare
# with the stored map pixels. Every halo whose 4 R200c aperture reaches a disc is re-evaluated exactly as in
# paint_halfdome_matched_profile_dm_map.jl::paint_batch_external_r200c! (unit vector from Position, theta from
# acos(clamp(dot)), theta < theta_max strict, clamp at theta_min, interpolator(theta, M200c/h, z)).
# Read-only on all inputs; writes one CSV + one text summary into the workstream output directory.
using XGPaint, Interpolations, HDF5, Healpix, Printf, Random, Unitful, UnitfulAstro
const XG = XGPaint
include("/home/cbllover/HalfDome/frb_map_generation/lee2022_tsz_pressure_profile.jl")

const OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit"
const REPO = "/home/cbllover/HalfDome/frb_map_generation"
const CAT = "/home/cbllover/HalfDome/lightcone_100.hdf5"
const H = 0.68
const NSIDE = 4096
const APERTURE = 4.0
const DISC_RADIUS = deg2rad(0.25)

b12_model() = Battaglia16ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H,
    P0_amp=18.1, P0_alpha_m=0.154, P0_alpha_z=-0.758, x_c_amp=0.497, x_c_alpha_m=-0.00865, x_c_alpha_z=0.731,
    alpha_amp=1.0, alpha_alpha_m=0.0, alpha_alpha_z=0.0, beta_amp=4.35, beta_alpha_m=0.0393, beta_alpha_z=0.415,
    gamma_amp=-0.3, gamma_alpha_m=0.0, gamma_alpha_z=0.0)

function theta_max_painter(model, m, z)   # compute_theta_max_r200c_external(model, m, z, 4)
    rho = XG.ρ_crit(model, z)
    r200 = ∛(m * XG.M_sun / (4pi / 3 * 200 * rho))
    return Float64(XG.angular_size(model, APERTURE * r200, z))
end

maps = [
    (name="b12_allz", model=b12_model(), cache="/home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2",
     map=joinpath(REPO, "outputs/tsz_frb_1m_z1_r200c_apertures/maps/battaglia12_fiducial_halfdome_compton_y_allz_nside4096_m200c_r200cx4.fits"),
     mmin=0.0, mmax=Inf, zmax=Inf),
    (name="lee22_allz", model=Lee2022ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H),
     cache=joinpath(REPO, "outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2"),
     map=joinpath(REPO, "outputs/tsz_dm_cross_updated_20260917/inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits"),
     mmin=0.0, mmax=Inf, zmax=Inf),
    (name="lee22_calib", model=Lee2022ThermalSZProfile(Omega_c=0.261, Omega_b=0.049, h=H),
     cache=joinpath(REPO, "outputs/halfdome_frb_inputs/lee22_noconc_pressure_m200c_logm15p7_tsz_interpolator.jld2"),
     map=joinpath(REPO, "outputs/tsz_dm_fullsky_20260918/maps/lee22_noconc_pressure_compton_y_calib_zmax2p0_nside4096_m200c_r200cx4.fits"),
     mmin=1.4705882352941176e13, mmax=9.278784477649915e14, zmax=2.0),
]


@inline function unitvec(pos, i)
    x = Float64(pos[1, i]); y = Float64(pos[2, i]); w = Float64(pos[3, i]); d = sqrt(x^2 + y^2 + w^2)
    return (x / d, y / d, w / d)
end

# pre-selection bound: theta_max(m, z) = atan(r1(z) (m/1e14)^(1/3)), r1 = tan(theta_max(1e14, z)) tabulated
struct ThetaBound
    zt::Vector{Float64}
    r1::Vector{Float64}
end
function ThetaBound(model)
    zt = collect(range(1e-3, 4.0, length=8001))
    return ThetaBound(zt, [tan(theta_max_painter(model, 1.0e14, z)) for z in zt])
end
function (b::ThetaBound)(m, z)
    k = clamp(searchsortedfirst(b.zt, z), 2, length(b.zt))
    r = max(b.r1[k - 1], b.r1[k])
    return 1.02 * atan(r * (m / 1.0e14)^(1 / 3)) + 1e-6
end

function select_halos(pos, mass, zz, c, mp, bound)
    sel = Int[]
    for i in eachindex(mass)
        m = mass[i]; z = Float64(zz[i])
        (isfinite(m) && isfinite(z) && m > 0 && z > 0) || continue
        (m >= mp.mmin && m < mp.mmax && z <= mp.zmax) || continue
        ux, uy, uz = unitvec(pos, i)
        cosang = clamp(ux * c[1] + uy * c[2] + uz * c[3], -1.0, 1.0)
        acos(cosang) < DISC_RADIUS + min(bound(m, z), pi) || continue
        push!(sel, i)
    end
    return sel
end

function repaint(pos, mass, zz, sel, pv, mp, itp, tmin)
    yrep = zeros(length(pv)); ncontrib = 0; nedge = 0
    for i in sel
        m = mass[i]; z = Float64(zz[i])
        ux, uy, uz = unitvec(pos, i)
        thmax = min(theta_max_painter(mp.model, m, z), pi)
        hit = false
        for (k, p) in enumerate(pv)
            th = acos(clamp(ux * p[1] + uy * p[2] + uz * p[3], -1.0, 1.0))
            th < thmax || continue
            yrep[k] += Float64(itp(max(th, tmin), m, z))
            hit = true
        end
        ncontrib += hit
    end
    return yrep, ncontrib
end

function main()
    println("reading catalogue ..."); flush(stdout)
    pos, mass, zz = h5open(CAT, "r") do f
        read(f["Position"]), Float64.(read(f["halo_mass_m200c"])) ./ H, read(f["redshift"])
    end
    N = size(pos, 2)
    println("halos: ", N, "  Position eltype ", eltype(pos), " size ", size(pos)); flush(stdout)
    lowz = findall(z -> z < 0.3, zz)
    i_big = lowz[argmax(mass[lowz])]
    i_mid = findfirst(i -> 0.95 < zz[i] < 1.05 && 0.9e14 < mass[i] < 1.1e14, 1:N)
    rng = MersenneTwister(20260929)
    v = randn(rng, 3); v ./= sqrt(sum(abs2, v))
    centres = [("massive_lowz_halo", unitvec(pos, i_big)), ("1e14_z1_halo", unitvec(pos, i_mid)), ("random", (v[1], v[2], v[3]))]
    @printf("disc centres: massive halo M=%.3e z=%.4f ; 1e14 halo M=%.3e z=%.4f\n", mass[i_big], zz[i_big], mass[i_mid], zz[i_mid])
    res = Resolution(NSIDE)
    open(joinpath(OUT, "map_patch_repaint_test.csv"), "w") do io
        println(io, "map,disc,n_pixels,n_halos_preselected,n_halos_contributing,max_abs_rel_diff,rms_rel_diff,median_y_map,max_abs_diff")
        for mp in maps
            itp = build_interpolator(mp.model; cache_file=mp.cache, overwrite=false, verbose=false)
            tmin = exp(Float64(first(first(itp.itp.ranges))))
            bound = ThetaBound(mp.model)
            println("loading map ", mp.map); flush(stdout)
            ymap = Healpix.readMapFromFITS(mp.map, 1, Float64)
            for (dname, c) in centres
                cth, cph = Healpix.vec2ang(c...)
                pix = Healpix.queryDiscRing(res, cth, cph, DISC_RADIUS)
                pv = [Healpix.pix2vecRing(res, p) for p in pix]
                sel = select_halos(pos, mass, zz, c, mp, bound)
                yrep, ncontrib = repaint(pos, mass, zz, sel, pv, mp, itp, tmin)
                ystore = ymap.pixels[pix]
                rel = [ystore[k] > 0 ? yrep[k] / ystore[k] - 1 : (yrep[k] == 0 ? 0.0 : Inf) for k in eachindex(pix)]
                @printf(io, "%s,%s,%d,%d,%d,%.3e,%.3e,%.4e,%.3e\n", mp.name, dname, length(pix), length(sel), ncontrib,
                    maximum(abs.(rel)), sqrt(sum(abs2, rel) / length(rel)), sort(ystore)[cld(length(ystore), 2)],
                    maximum(abs.(yrep .- ystore)))
                @printf("%-12s %-18s pixels=%d preselected=%d contributing=%d max|rel|=%.3e median y=%.3e\n", mp.name, dname,
                    length(pix), length(sel), ncontrib, maximum(abs.(rel)), sort(ystore)[cld(length(ystore), 2)]); flush(stdout)
            end
            ymap = nothing; GC.gc()
        end
    end
    println("wrote map_patch_repaint_test.csv")
end

main()
