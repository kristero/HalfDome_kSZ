# Single-halo visual explanation: Battaglia16 electron density in the plane that contains
# the lines of sight, and the cumulative DM along rays at several impact parameters, for
#   (a) the previous projected convention: a ray is credited with the halo if b <= R200c and
#       then receives the full column integrated from -1e5 to +1e5 R200c (XGPaint LOS);
#   (b) the like-for-like spherical boundary used to compare with TNG's within-R200
#       catalogue: only the chord inside the R200c sphere counts, so DM -> 0 as b -> R200c.
# Default halo: HalfDome catalogue floor mass (M200c = 7.327e12 Msun) at z = 0.5.
# Usage: julia compute_single_halo_ray_dm.jl [mass_msun] [z] [output_dir]
using XGPaint, Unitful, UnitfulAstro, Printf, DelimitedFiles
include(joinpath(@__DIR__, "lee2022_frb_dm_profile.jl"))

const MASS_MSUN = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 7.327e12
const Z = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
const OUT = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "outputs", "single_halo_visual_20260917")
mkpath(OUT)

const b16 = XGPaint.HaloDMProfile(XGPaint.BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68))
const dens = Battaglia16DensityDMProfile(b16)
const quad = lee2022_quadgk_function()

const r200c_q = getfield(XGPaint, Symbol("R_", Char(0x0394)))(b16, MASS_MSUN * XGPaint.M_sun, Z, 200)
const R200C_M = Float64(ustrip(uconvert(u"m", r200c_q)))
const R200C_MPC = Float64(ustrip(uconvert(u"Mpc", r200c_q)))
const THETA200C_RAD = Float64(XGPaint.angular_size(b16, r200c_q, Z))
const DM_FACTOR = R200C_M * M2_TO_PC_CM3_LOCAL / (1 + Z)  # int n_e[m^-3] dl[R200c] -> observer-frame pc cm^-3

ne_m3(r) = halo_electron_density_m3(dens, max(r, 1e-3), MASS_MSUN, Z)
segment(b, a, c) = c > a ? quad(l -> ne_m3(sqrt(l^2 + b^2)), a, c; rtol=1e-8, order=9)[1] : 0.0

# 1. density in the (l, x) plane: l along the line of sight, x transverse (impact parameter axis)
const L_MAX = 3.0
const X_MAX = 1.5
ls = collect(range(-L_MAX, L_MAX; length=601))
xs = collect(range(-X_MAX, X_MAX; length=301))
grid = [ne_m3(sqrt(l^2 + x^2)) * 1e-6 for x in xs, l in ls]  # cm^-3; rows = x, columns = l
writedlm(joinpath(OUT, "density_grid_cm3.csv"), grid, ',')
writedlm(joinpath(OUT, "grid_l_r200c.csv"), ls)
writedlm(joinpath(OUT, "grid_x_r200c.csv"), xs)

# 2. rays: cumulative DM from the source side, both conventions
const IMPACTS = [0.05, 0.35, 0.65, 0.85, 1.0, 1.25]
lray = collect(range(-L_MAX, L_MAX; length=401))
open(joinpath(OUT, "rays.csv"), "w") do io
    println(io, "impact_r200c,l_r200c,dm_projected_would_be,dm_projected_credited,dm_sphere")
    for b in IMPACTS
        credited = b <= 1.0
        lmax = b < 1.0 ? sqrt(1 - b^2) : 0.0
        cum_proj = segment(b, -LEE2022_LOS_MAX_R200C, -L_MAX)  # gas on the source side of the panel
        cum_sph = 0.0
        prev = lray[1]
        for (i, l) in enumerate(lray)
            if i > 1
                cum_proj += segment(b, prev, l)
                cum_sph += segment(b, max(prev, -lmax), min(l, lmax))
            end
            prev = l
            @printf(io, "%.4f,%.6f,%.8g,%.8g,%.8g\n", b, l, cum_proj * DM_FACTOR,
                credited ? cum_proj * DM_FACTOR : 0.0, cum_sph * DM_FACTOR)
        end
    end
end

# 3. totals and cross-checks against the production code paths
open(joinpath(OUT, "ray_totals.csv"), "w") do io
    println(io, "impact_r200c,credited_projected,dm_projected_full_los,dm_projected_beyond_panel,xgpaint_projected,dm_sphere,chord_check,chord_half_length_r200c")
    for b in IMPACTS
        full = (segment(b, -LEE2022_LOS_MAX_R200C, L_MAX) + segment(b, L_MAX, LEE2022_LOS_MAX_R200C)) * DM_FACTOR
        beyond = (segment(b, -LEE2022_LOS_MAX_R200C, -L_MAX) + segment(b, L_MAX, LEE2022_LOS_MAX_R200C)) * DM_FACTOR
        xg = b16(b * THETA200C_RAD, MASS_MSUN, Z)
        lmax = b < 1.0 ? sqrt(1 - b^2) : 0.0
        sph = 2 * segment(b, 0.0, lmax) * DM_FACTOR
        chk = chord_dm_pc_cm3(dens, b, lmax, MASS_MSUN, Z)
        @printf(io, "%.4f,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n", b, b <= 1.0, full, beyond, xg, sph, chk, lmax)
        abs(full - xg) <= 1e-4 * max(xg, 1e-12) || @warn "projected LOS differs from XGPaint" b full xg
        abs(sph - chk) <= 1e-6 * max(chk, 1e-12) || @warn "sphere differs from chord_dm_pc_cm3" b sph chk
    end
end
open(joinpath(OUT, "halo.txt"), "w") do io
    @printf(io, "mass_msun=%.6g\nz=%.3f\nr200c_mpc=%.6f\ntheta200c_arcmin=%.5f\ndm_factor_pc_cm3_per_m3_r200c=%.8g\nl_max_r200c=%.2f\nx_max_r200c=%.2f\nlos_max_r200c=%.1f\nmodel=Battaglia16 (XGPaint BattagliaTauProfile Omega_c=0.261 Omega_b=0.049 h=0.68), rho_gas = P0 gNFW f_b rho_crit(z), ne2d composition\n",
        MASS_MSUN, Z, R200C_MPC, THETA200C_RAD * 180 / pi * 60, DM_FACTOR, L_MAX, X_MAX, LEE2022_LOS_MAX_R200C)
end
println(read(joinpath(OUT, "halo.txt"), String))
println(read(joinpath(OUT, "ray_totals.csv"), String))
