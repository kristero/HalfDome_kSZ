# Unit test of the Lee22 redshift_scaling=:tng_calibrated option (2026-09-29).
using XGPaint, Unitful, UnitfulAstro, Printf
include(joinpath(@__DIR__, "..", "lee2022_frb_dm_profile.jl"))
const H = 0.68
mk(norm, zs) = Lee2022NoConcentrationDMProfile(Omega_c=0.261, Omega_b=0.049, h=H, normalization=norm, n0_pivot=:mcut,
                                              shape_clip_mass_msun=10.0^14.8 / H, redshift_scaling=zs)
lit, xgp, cal = mk(:literal, :physical), mk(:xgpaint_ne2d, :physical), mk(:literal, :tng_calibrated)
C(z) = 0.2841790493880425 * (1 + min(z, 2.0))^2.483190184608004
worst = 0.0
for m in (7.3e12, 3e13, 2e14, 1e15), z in (0.01, 0.35, 1.0, 1.9, 2.5), x in (0.05, 0.5, 0.99)
    r_lit = halo_electron_density_m3(cal, x, m, z) / halo_electron_density_m3(lit, x, m, z)
    r_xgp = halo_electron_density_m3(cal, x, m, z) / halo_electron_density_m3(xgp, x, m, z)
    global worst = max(worst, abs(r_lit / C(z) - 1), abs(r_xgp / (C(z) / 0.6016316602183194) - 1))
end
@printf("density ratio vs C(z): worst relative error %.2e\n", worst); worst < 1e-12 || error("density factor wrong")
# through the XGPaint wrapper (production PDF path): P0 ratio and n_e through XGPaint's own ne2d path
wc, wx = Lee2022XGPaintDMProfile(cal), Lee2022XGPaintDMProfile(xgp)
worst = 0.0
for m in (1e13, 5e13), z in (0.1, 0.8, 2.0)
    pc = XGPaint.get_params(wc, m * XGPaint.M_sun, z); px = XGPaint.get_params(wx, m * XGPaint.M_sun, z)
    global worst = max(worst, abs(pc.P₀ / px.P₀ / (C(z) / 0.6016316602183194) - 1))
    pc.β == px.β && pc.xc == px.xc || error("shape changed")
    dc = Battaglia16DensityDMProfile(wc); dx = Battaglia16DensityDMProfile(wx)
    global worst = max(worst, abs(halo_electron_density_m3(dc, 0.3, m, z) / halo_electron_density_m3(dx, 0.3, m, z) / (C(z) / 0.6016316602183194) - 1))
end
@printf("XGPaint wrapper P0 and ne2d ratio: worst relative error %.2e\n", worst); worst < 1e-12 || error("wrapper factor wrong")
# guard and unchanged signatures
try; mk(:xgpaint_ne2d, :tng_calibrated); error("guard missing"); catch e; occursin("normalization=:literal", sprint(showerror, e)) || rethrow(); end
println("guard: tng_calibrated with xgpaint_ne2d is refused")
sig_x = lee2022_cache_signature(xgp)
sig_x == "lee2022_table_a2_no_concentration_v2|M200c|R200c|alpha=1|gamma=-0.3|XH=0.76|profile_owned_los|observer=1/(1+z)|norm=xgpaintne2d|n0pivot=mcut|shapeclip=1e14.9675Msun" ||
    error("existing signature changed: $sig_x")
println("existing xgpnative cache signature unchanged")
println("new signature: ", lee2022_cache_signature(cal))
@printf("C(z): z=0 %.4f  0.35 %.4f  1 %.4f  2 %.4f  (C/0.6016: %.3f %.3f %.3f %.3f)\n", C(0), C(0.35), C(1), C(2),
        C(0)/0.60163, C(0.35)/0.60163, C(1)/0.60163, C(2)/0.60163)
println("PASS")
