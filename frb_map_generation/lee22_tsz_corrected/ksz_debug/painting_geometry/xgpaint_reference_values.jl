# Reference values from XGPaint itself for validating the Python reimplementation (ksz_debug/painting_geometry).
using XGPaint, Unitful, UnitfulAstro
include(joinpath("/home/cbllover/HalfDome/truncation_comparison", "spherical_truncation_profiles.jl"))
using .SphericalTruncation
m = BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
cm = ChordMeanProfile(m, 4.0)
for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0), (3e13, 2.0))
    th = theta_r200c(m, M, z)
    for f in (0.1, 0.5, 1.0, 2.0, 3.5)
        θ = f * th
        tp = Float64(m(θ, M, z))
        ts = spherical_value_direct(cm, θ, M, z)
        println("M=$M z=$z theta200=$th f=$f tau_proj=$tp tau_sphere4=$ts")
    end
end
