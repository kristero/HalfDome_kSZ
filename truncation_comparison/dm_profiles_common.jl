# Shared construction of the two halo-DM profiles used in the tSZ x DM comparison:
# Battaglia16 (XGPaint HaloDMProfile) and Lee22 no-concentration in the XGPaint-native reading.
module FRB
include(joinpath(@__DIR__, "..", "frb_map_generation", "generate_halfdome_z1_dm_mass_windows.jl"))
end
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation
const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603; const SPHERE_R200C = 4.0
const LEE22_SHAPE_CLIP_MSUN = 10.0^14.8 / H
function dm_inner_profiles()
    lee = FRB.Lee2022XGPaintDMProfile(FRB.Lee2022NoConcentrationDMProfile(
        Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H, normalization=:xgpaint_ne2d, n0_pivot=:mcut,
        shape_clip_mass_msun=LEE22_SHAPE_CLIP_MSUN, redshift_scaling=:physical))
    b16 = FRB.HaloDMProfile(BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H))
    return (b16=b16, lee22=lee)
end
const DM_CACHE_NAMES = ("dm_b16_projected", "dm_b16_sphere4p0_chordmean", "dm_lee22noc_xgpnative_projected", "dm_lee22noc_xgpnative_sphere4p0_chordmean")
function dm_models()
    p = dm_inner_profiles()
    return (p.b16, ChordMeanProfile(p.b16, SPHERE_R200C), p.lee22, ChordMeanProfile(p.lee22, SPHERE_R200C))
end
