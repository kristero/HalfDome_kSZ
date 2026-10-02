# Independent (no XGPaint, no cache) tau profiles per (M, z) cell and their flat-sky Hankel amplitudes (2026-10-02, ksz_debug):
#  (1) "b16_indep": re-implementation of XGPaint BattagliaTauProfile + 4 R200c sphere from first principles:
#        tau = sigma_T * 0.9 / m_per_e * f_b rho_c(z) R200c * P0 * 2 int_0^sqrt(16-x^2) g(sqrt(s^2+x^2)) ds,
#        g(x) = (x/xc)^gamma (1 + (x/xc)^alpha)^(-(beta+gamma)/alpha), B16 AGN parameters, M = M200c [Msun]
#      -> checks XGPaint's tau amplitude/units and our cache/Hankel chain (compare with tau_ell_table_b16.h5).
#  (2) "pks2map": the WebSky Fortran profile as transcribed (externally) from pks2map/bbps_profile.f90 by Y. Guan
#      (github.com/guanyilun/PeakPatch.jl PR 1, compare_composite_ksz_battaglia.jl, NOT verified against the Fortran here):
#        tau = sigma_T/(1.136 m_p) f_b rhobar_m,phys(z) R200m,phys(mh) * 2 int_0^sqrt(16-x^2) rt(sqrt(s^2+x^2)) ds,
#        rt(x) = P0 (x/0.5)^-0.2 (1 + (x/0.5)^alpha)^-beta / Omega_m(z), x = r/R200c(mh)
#      evaluated at mh = the cell's M200c.
# Usage: julia -t 4 tau_ell_independent.jl
using HDF5, Base.Threads
import XGPaint                                   # only for SpecialFunctions.besselj0 (not in the default env); no XGPaint physics used
const besselj0 = XGPaint.SpecialFunctions.besselj0
const OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_catalogue_test"
const OM = 0.31; const OB = 0.049; const H = 0.68; const FB = OB / OM; const OL = 1 - OM
const SIGMA_T = 6.6524587321e-29; const MP = 1.67262192369e-27; const ME = 9.1093837015e-31; const MSUN = 1.98847e30
const MPC = 3.0856775814913673e22; const G = 6.67430e-11; const C_KMS = 299792.458
const XH = 0.76
const M_PER_E = ME + 2XH / (XH + 1) * MP + (1 - XH) / (2 * (1 + XH)) * 4MP      # XGPaint ne2d factor
E2(z) = OM * (1 + z)^3 + OL
omz(z) = OM * (1 + z)^3 / E2(z)
rhoc_phys(z) = 3 * (100H * 1e3 / MPC)^2 * E2(z) / (8π * G)                          # kg/m^3
const ZG = range(0.0, 6.0; length=600001)
const CHIG = let c = zeros(length(ZG)); for i in 2:length(ZG)
        c[i] = c[i-1] + 0.5 * (C_KMS / (100H * sqrt(E2(ZG[i]))) + C_KMS / (100H * sqrt(E2(ZG[i-1])))) * step(ZG)
    end; c end
chi_of_z(z) = (t = z / step(ZG) + 1; i = clamp(floor(Int, t), 1, length(ZG) - 1); f = t - i; CHIG[i] * (1 - f) + CHIG[i+1] * f)   # Mpc
r_delta(Mkg, Δ, rho) = cbrt(3Mkg / (4π * Δ * rho))                                   # m

tab = h5open(joinpath(OUT, "tau_ell_table_b16.h5"), "r")
ells = read(tab["ell"]); le = read(tab["logm_edges"]); ze = read(tab["z_edges"]); close(tab)
lmc = 0.5 .* (le[1:end-1] .+ le[2:end]); zc = 0.5 .* (ze[1:end-1] .+ ze[2:end])
NM, NZ, NL = length(lmc), length(zc), length(ells)
const NU = 160; const NT = 900; const XS = 4.0
# LOS chord integral 2 int_0^L f(sqrt(s^2+x^2)) ds with s = x sinh u (x > 0)
function chord(f, x, L)
    umax = asinh(L / x); du = umax / NU; acc = 0.0
    for k in 1:NU
        u = (k - 0.5) * du; c = cosh(u)
        acc += f(x * c) * x * c
    end
    return 2acc * du
end
A_b16 = zeros(NL, NM, NZ); A_pks = zeros(NL, NM, NZ); A0_b16 = zeros(NM, NZ); A0_pks = zeros(NM, NZ)
cells = [(i, j) for i in 1:NM, j in 1:NZ]
t0 = time()
@threads :dynamic for ci in eachindex(cells)
    im, iz = cells[ci]; M = exp10(lmc[im]); z = zc[iz]; zp = 1 + z; m14 = M / 1e14
    dA = chi_of_z(z) / zp * MPC                                                      # m
    r200c = r_delta(M * MSUN, 200, rhoc_phys(z)); th200 = atan(r200c / dA)
    # B16 (XGPaint parameterisation)
    P0 = 4.0e3 * m14^0.29 * zp^-0.66; xc = 0.5
    α = 0.88 * m14^-0.03 * zp^0.19; β = 3.83 * m14^0.04 * zp^-0.025; γ = -0.2
    g_b16(x) = (x / xc)^γ * (1 + (x / xc)^α)^(-(β + γ) / α)
    amp_b16 = SIGMA_T * 0.9 / M_PER_E * FB * rhoc_phys(z) * r200c * P0
    # pks2map (as transcribed): rhobar_m,phys * R200m,phys(mh) amplitude, rt in mean-matter units with 1/Omega_m(z)
    rhom_phys = OM * rhoc_phys(0.0) * zp^3
    r200m = r_delta(M * MSUN, 200, rhom_phys)
    g_pks(x) = P0 * (x / 0.5)^-0.2 * (1 + (x / 0.5)^α)^-β / omz(z)
    amp_pks = SIGMA_T / (1.136 * MP) * FB * rhom_phys * r200m
    lnx = range(log(1e-5), log(XS); length=NT); dln = step(lnx)
    w = zeros(NT); fb16 = zeros(NT); fpk = zeros(NT); th = zeros(NT)
    for j in 1:NT
        x = exp(lnx[j]); L = sqrt(max(XS^2 - x^2, 0.0))
        th[j] = x * th200
        wt = (j == 1 || j == NT) ? 0.5 : 1.0
        w[j] = wt * th[j]^2 * dln                                                   # theta dtheta = theta^2 dln theta
        fb16[j] = amp_b16 * chord(g_b16, x, L); fpk[j] = amp_pks * chord(g_pks, x, L)
    end
    inner = π * th[1]^2                                                              # inner disc, tau ~ tau(theta_min)
    A0_b16[im, iz] = 2π * sum(fb16 .* w) + fb16[1] * inner; A0_pks[im, iz] = 2π * sum(fpk .* w) + fpk[1] * inner
    for (il, ℓ) in enumerate(ells)
        sb = 0.0; sp = 0.0
        @inbounds for j in 1:NT
            jb = besselj0(ℓ * th[j]) * w[j]; sb += fb16[j] * jb; sp += fpk[j] * jb
        end
        A_b16[il, im, iz] = 2π * sb + fb16[1] * inner; A_pks[il, im, iz] = 2π * sp + fpk[1] * inner
    end
end
println("done in $(round(time() - t0; digits=1)) s")
h5open(joinpath(OUT, "tau_ell_table_independent.h5"), "w") do f
    f["ell"] = ells; f["logm_edges"] = le; f["z_edges"] = ze
    f["a_b16_indep_x4p0"] = A_b16; f["a0_b16_indep_x4p0"] = A0_b16
    f["a_pks2map_x4p0"] = A_pks; f["a0_pks2map_x4p0"] = A0_pks
    attrs(f)["note"] = "independent B16 (XGPaint convention) and pks2map-as-transcribed tau Hankel amplitudes, 4 R200c sphere, cell centres"
end
