# Unit test of the Lee22 redshift_scaling=:bookkeeping_a3 option, n_e = literal x (1+z)^3 / F_n (2026-09-30).
using XGPaint, Unitful, UnitfulAstro, Printf
include(joinpath(@__DIR__, "..", "lee2022_frb_dm_profile.jl"))
const H = 0.68
const FN = parse(Float64, get(ENV, "LEE2022_BOOKKEEPING_FN", "5.4"))
mk(norm, zs) = Lee2022NoConcentrationDMProfile(Omega_c=0.261, Omega_b=0.049, h=H, normalization=norm, n0_pivot=:mcut,
                                              shape_clip_mass_msun=10.0^14.8 / H, redshift_scaling=zs)
lit, xgp, a3, cal = mk(:literal, :physical), mk(:xgpaint_ne2d, :physical), mk(:literal, :bookkeeping_a3), mk(:literal, :tng_calibrated)
worst = 0.0
for m in (7.3e12, 3e13, 2e14, 1e15), z in (0.01, 0.35, 1.0, 1.9, 3.5), x in (0.05, 0.5, 0.99)
    r = halo_electron_density_m3(a3, x, m, z) / halo_electron_density_m3(lit, x, m, z)
    global worst = max(worst, abs(r / ((1 + z)^3 / FN) - 1))
end
@printf("density ratio vs (1+z)^3/F_n (F_n = %.3f): worst relative error %.2e\n", FN, worst); worst < 1e-12 || error("density factor wrong")
wa, wx = Lee2022XGPaintDMProfile(a3), Lee2022XGPaintDMProfile(xgp)
worst = 0.0
for m in (1e13, 5e13), z in (0.1, 0.8, 2.0)
    pa = XGPaint.get_params(wa, m * XGPaint.M_sun, z); px = XGPaint.get_params(wx, m * XGPaint.M_sun, z)
    global worst = max(worst, abs(pa.P₀ / px.P₀ / ((1 + z)^3 / FN / 0.6016316602183194) - 1))
    pa.β == px.β && pa.xc == px.xc || error("shape changed")
end
@printf("XGPaint wrapper P0 ratio: worst relative error %.2e\n", worst); worst < 1e-12 || error("wrapper factor wrong")
try; mk(:xgpaint_ne2d, :bookkeeping_a3); error("guard missing"); catch e; occursin("normalization=:literal", sprint(showerror, e)) || rethrow(); end
println("guard: bookkeeping_a3 with xgpaint_ne2d is refused")
lee2022_cache_signature(xgp) == "lee2022_table_a2_no_concentration_v2|M200c|R200c|alpha=1|gamma=-0.3|XH=0.76|profile_owned_los|observer=1/(1+z)|norm=xgpaintne2d|n0pivot=mcut|shapeclip=1e14.9675Msun" ||
    error("existing xgpnative signature changed")
occursin("zscale=tngcal_C00.284179_k2.48319_zmax2", lee2022_cache_signature(cal)) || error("tngcal signature changed")
println("existing signatures unchanged; new signature: ", lee2022_cache_signature(a3))
for z in (0.0, 0.35, 1.0, 2.0)
    @printf("z=%.2f  (1+z)^3/F_n = %.4f   C(z) = %.4f   ratio %.3f\n", z, (1 + z)^3 / FN,
            lee2022_redshift_scaling_factor(cal, z), (1 + z)^3 / FN / lee2022_redshift_scaling_factor(cal, z))
end
# derived option: literal x (0.6774 (1+z))^3 X_H^2, i.e. F_n = 1/(h_TNG^3 X_H^2) = 5.570
ph = mk(:literal, :bookkeeping_physical)
fn_phys = 1 / (0.6774^3 * 0.76^2)
worst = 0.0
for m in (7.3e12, 3e13, 2e14, 1e15), z in (0.01, 0.35, 1.0, 1.9, 3.5), x in (0.05, 0.5, 0.99)
    r = halo_electron_density_m3(ph, x, m, z) / halo_electron_density_m3(lit, x, m, z)
    global worst = max(worst, abs(r / ((1 + z)^3 / fn_phys) - 1))
end
@printf("bookkeeping_physical: F_n = 1/(h^3 X_H^2) = %.4f; density ratio worst relative error %.2e\n", fn_phys, worst)
worst < 1e-12 || error("physical factor wrong")
try; mk(:xgpaint_ne2d, :bookkeeping_physical); error("guard missing"); catch e; occursin("normalization=:literal", sprint(showerror, e)) || rethrow(); end
println("physical signature: ", lee2022_cache_signature(ph))
println("PASS")
