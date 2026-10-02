# kSZ model definitions in HalfDome's own cosmology (2026-10-02). The HalfDome lightcone follows Omega_m = 0.3089 exactly
# (|Position| / chi_h(z) = 1 to 4e-7), i.e. the TNG/Planck15 cosmology: h = 0.6774, Omega_b = 0.0486, Omega_c = 0.2603. Catalogue
# masses are Msun/h, converted with h = 0.6774. Used by paint_ksz_websky.jl (ksz_models.jl keeps the earlier h = 0.68 setup).
const FMG = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(FMG, "lee2022_frb_dm_profile.jl"))
include(joinpath(FMG, "..", "truncation_comparison", "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603; const OMEGA_M = OMEGA_B + OMEGA_C
const C_KMS = 299_792.458
const SHAPE_CLIP = LEE2022_FIT_MASS_MAX_HINV_MSUN / H        # Lee22 shape freeze (production 10^14.8 h^-1 Msun)

"""Lee22 electron density as an XGPaint tau profile (see ksz_models.jl)."""
struct Lee22TauProfile{T,C,P} <: XGPaint.AbstractBattagliaTauProfile{T}
    f_b::T
    cosmo::C
    lee::P
end
Lee22TauProfile(lee::Lee2022XGPaintDMProfile{T}) where {T} = Lee22TauProfile{T,typeof(lee.cosmo),typeof(lee)}(lee.f_b, lee.cosmo, lee)
XGPaint.get_params(m::Lee22TauProfile, M_200c, z) = XGPaint.get_params(m.lee, M_200c, z)
XGPaint.object_size(m::Lee22TauProfile, physical_size, z) = XGPaint.object_size(m.lee, physical_size, z)

lee_dm(norm, zs) = Lee2022XGPaintDMProfile(Lee2022NoConcentrationDMProfile(; Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H,
    normalization=norm, n0_pivot=:mcut, shape_clip_mass_msun=SHAPE_CLIP, redshift_scaling=zs))
const B16 = BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
const L22PIPE = Lee22TauProfile(lee_dm(:xgpaint_ne2d, :physical))
const L22CORR = Lee22TauProfile(lee_dm(:literal, :bookkeeping_physical))
const DM_REF = Dict("lee22pipe" => lee_dm(:xgpaint_ne2d, :physical), "lee22corr" => lee_dm(:literal, :bookkeeping_physical))
