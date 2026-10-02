# Shared kSZ model definitions for paint_ksz_variants.jl and poisson_mass_split.jl (2026-10-02).
const FMG = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(FMG, "lee2022_frb_dm_profile.jl"))
include(joinpath(FMG, "..", "truncation_comparison", "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.68; const OMEGA_B = 0.049; const OMEGA_C = 0.261
const X = 4.0
const C_KMS = 299_792.458
const SHAPE_CLIP = LEE2022_FIT_MASS_MAX_MSUN_AT_H068

"""Lee22 electron density as an XGPaint tau profile: XGPaint's tau machinery (rho_2d, ne2d, sigma_T) with the Lee22
parameters of `Lee2022XGPaintDMProfile` (get_params: beta = alpha beta' - gamma, P0 including normalization and redshift
factor). Calling it returns tau, not the FRB DM."""
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
const MODELS = (
    ("b16", BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)),
    ("lee22pipe", Lee22TauProfile(lee_dm(:xgpaint_ne2d, :physical))),
    ("lee22corr", Lee22TauProfile(lee_dm(:literal, :bookkeeping_physical))),
)
const DM_REF = Dict("lee22pipe" => lee_dm(:xgpaint_ne2d, :physical), "lee22corr" => lee_dm(:literal, :bookkeeping_physical))

