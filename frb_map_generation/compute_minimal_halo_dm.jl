# Minimal observer-frame DM a ray can pick up from a single halo of mass M200c at
# redshift z, i.e. the DM of a ray that grazes the aperture edge (b = N R200c).
# Projected apertures use the profile's own long line of sight (XGPaint convention);
# the spherical variant integrates the Battaglia16 density only inside the R200c
# sphere, where the chord length -> 0 at the edge, so DM -> 0 continuously.
using XGPaint, Unitful, UnitfulAstro, Printf
include(joinpath(@__DIR__, "lee2022_frb_dm_profile.jl"))

const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "outputs", "minimal_dm_20260916")
mkpath(OUT)
const b16 = XGPaint.HaloDMProfile(XGPaint.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68))
const lee_nc = Lee2022NoConcentrationDMProfile(normalization=:baryon_fraction, n0_pivot=:mcut,
    shape_clip_mass_msun=LEE2022_FIT_MASS_MAX_MSUN_AT_H068)
const lee_pref = Lee2022ConcentrationDMProfile(normalization=:baryon_fraction, concentration_source=:tng_mean,
    shape_clip_mass_msun=LEE2022_FIT_MASS_MAX_MSUN_AT_H068)
const quad = lee2022_quadgk_function()
const M2_TO_PC_CM3 = XGPaint.M2_TO_PC_CM3

r200c(model, M, z) = getfield(XGPaint, Symbol("R_", Char(0x0394)))(model, M * XGPaint.M_sun, z, 200)
theta200c(model, M, z) = XGPaint.angular_size(model, r200c(model, M, z), z)

# Battaglia16 physical 3-D electron density [m^-3] at x = r/R200c (XGPaint ne2d convention).
function b16_ne(x, M, z)
    par = XGPaint.get_params(b16, M * XGPaint.M_sun, z)
    rho_crit = getfield(XGPaint, Symbol(Char(0x03c1), "_crit"))(b16, z)
    rho_gas = par.P₀ * XGPaint.generalized_nfw(x, par.xc, par.α, par.β, par.γ) * b16.f_b * rho_crit
    me = XGPaint.constants.ElectronMass; mH = XGPaint.constants.ProtonMass; xH = 0.76
    factor = me + (2xH / (xH + 1)) * mH + ((1 - xH) / (2(1 + xH))) * 4mH
    return 0.9 * rho_gas / factor
end
# Observer-frame DM through the R200c sphere at impact parameter b = xb R200c.
function b16_spherical_dm(xb, M, z; sphere=1.0)
    xb >= sphere && return 0.0
    lmax = sqrt(sphere^2 - xb^2)
    I, _ = quad(l -> Float64(ustrip(uconvert(u"m^-3", b16_ne(sqrt(xb^2 + l^2), M, z)))), 0.0, lmax; rtol=1e-8)
    R = Float64(ustrip(uconvert(u"m", r200c(b16, M, z))))
    return 2 * I * R * M2_TO_PC_CM3 / (1 + z)
end
b16_projected_dm(xb, M, z) = b16(xb * theta200c(b16, M, z), M, z)
lee_projected_dm(model, xb, M, z) = model(xb * theta200c(model, M, z), M, z)

logMs = collect(log10(7.327e12):0.05:log10(3.8e15))
zs = (0.05, 0.2, 0.5, 0.8, 1.0)
open(joinpath(OUT, "minimal_halo_dm_grid.csv"), "w") do io
    println(io, "log10_m200c_msun,z,r200c_mpc,theta200c_arcmin,b16_edge_dm_1r200c,b16_edge_dm_3r200c,b16_sphere_dm_b0p99r200c,b16_sphere_dm_b0p9r200c,b16_sphere_dm_b0p5r200c,lee22_noc_edge_dm_1r200c,lee22_best_edge_dm_1r200c,b16_center_dm_b0p01r200c")
    for z in zs, lm in logMs
        M = 10.0^lm
        R = Float64(ustrip(uconvert(u"Mpc", r200c(b16, M, z))))
        th = theta200c(b16, M, z) * 180 / pi * 60
        @printf(io, "%.4f,%.2f,%.5f,%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", lm, z, R, th,
            b16_projected_dm(1.0, M, z), b16_projected_dm(3.0, M, z),
            b16_spherical_dm(0.99, M, z), b16_spherical_dm(0.9, M, z), b16_spherical_dm(0.5, M, z),
            lee_projected_dm(lee_nc, 1.0, M, z), lee_projected_dm(lee_pref, 1.0, M, z),
            b16_projected_dm(0.01, M, z))
    end
end
# Global minimum over the HalfDome foreground (floor mass, z in (0,1]) for each convention.
open(joinpath(OUT, "minimal_halo_dm_summary.txt"), "w") do io
    for (name, f) in (("B16 projected 1R200c", (M, z) -> b16_projected_dm(1.0, M, z)),
                      ("B16 projected 3R200c", (M, z) -> b16_projected_dm(3.0, M, z)),
                      ("Lee22 no-c corrected projected 1R200c", (M, z) -> lee_projected_dm(lee_nc, 1.0, M, z)),
                      ("Lee22 best corrected projected 1R200c", (M, z) -> lee_projected_dm(lee_pref, 1.0, M, z)))
        vals = [(f(7.327e12, z), z) for z in 0.02:0.02:1.0]
        vmin, zmin = minimum(vals)
        vmax, zmax = maximum(vals)
        @printf(io, "%-42s floor-mass edge DM: min %.3f pc cm^-3 at z=%.2f, max %.3f at z=%.2f\n", name, vmin, zmin, vmax, zmax)
    end
    for z in (0.2, 0.5, 1.0)
        @printf(io, "B16 spherical R200c, floor mass, z=%.1f: DM(b=0.99R)=%.3f  DM(b=0.999R)=%.3f  DM(b=0.9999R)=%.3f\n", z,
            b16_spherical_dm(0.99, 7.327e12, z), b16_spherical_dm(0.999, 7.327e12, z), b16_spherical_dm(0.9999, 7.327e12, z))
    end
end
println(read(joinpath(OUT, "minimal_halo_dm_summary.txt"), String))
println("Saved grid to ", joinpath(OUT, "minimal_halo_dm_grid.csv"))
