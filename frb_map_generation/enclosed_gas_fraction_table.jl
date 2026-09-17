# Enclosed gas fraction f_gas(<X R200c)/f_b for Battaglia16 and the Lee22 readings, at the low
# redshifts that dominate the Takahashi cross-correlation. Uses the profile-owned 3-D electron
# density (halo_electron_density_m3) and XGPaint's ne2d electron-per-mass convention for both.
include(joinpath(@__DIR__, "paint_halfdome_matched_profile_dm_map.jl"))
using Unitful, UnitfulAstro
const PS = ProfileSupport
clip = 10.0^14.8 / H_VALUE
base = (dm_profile="battaglia16", lee2022_concentration_mode="none", lee2022_normalization="literal",
        lee2022_n0_pivot="legacy_1e14", lee2022_concentration_source="duffy2008",
        lee2022_shape_mass_clip_msun=Inf, lee2022_redshift_scaling="physical",
        halo_boundary="projected", dm_aperture_r200_multiplier=1.0)
models = [
    ("B16", PS.Battaglia16DensityDMProfile(PS.dm_profile_runtime_configuration(base).model)),
    ("Lee22 no-c XGPaint-native", PS.dm_profile_runtime_configuration(merge(base, (dm_profile="lee2022", lee2022_normalization="xgpaint_ne2d", lee2022_n0_pivot="mcut", lee2022_shape_mass_clip_msun=clip))).model),
    ("Lee22 legacy", PS.dm_profile_runtime_configuration(merge(base, (dm_profile="lee2022",))).model),
]
mp = Float64(ustrip(uconvert(u"kg", XGPaint.constants.ProtonMass)))
m_per_e = PS.xgpaint_mass_per_electron_kg()
msun_kg = Float64(ustrip(uconvert(u"kg", 1.0 * XGPaint.M_sun)))
f_b = 0.049 / 0.31
quad = PS.lee2022_quadgk_function()
function enclosed_over_fb(model, mass, z, X)
    r200_m = Float64(ustrip(uconvert(u"m", XGPaint.R_Δ(model, mass * XGPaint.M_sun, z, 200))))
    ne_count = quad(x -> 4pi * x^2 * PS.halo_electron_density_m3(model, x, mass, z), 1e-6, X; rtol=1e-7)[1] * r200_m^3
    cosmic = f_b * mass * msun_kg * 0.9 / m_per_e     # electrons if the halo held its cosmic baryon share (XGPaint ne2d convention)
    return ne_count / cosmic
end
println("f_gas(<X R200c)/f_b  [columns: X=1 | X=3]")
for (label, model) in models
    println("== ", label)
    for z in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0), mass in (1e13, 1e14, 1e15)
        println(rpad("  z=$(z) M=$(mass)", 26), "  ", round(enclosed_over_fb(model, mass, z, 1.0), digits=3), "   ", round(enclosed_over_fb(model, mass, z, 3.0), digits=3))
    end
end
