# Evaluate the production XGPaint interpolator (loaded from a copy of the B16 cache) at the B1 test points.
using XGPaint
include("/home/cbllover/HalfDome/truncation_comparison/spherical_truncation_profiles.jl"); using .SphericalTruncation
m = BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
cm = ChordMeanProfile(m, 4.0)
itp = build_interpolator(cm; cache_file=ARGS[1], overwrite=false, verbose=false)
for z in (0.004, 0.0125, 0.03, 0.0375, 0.05), M in (1e13, 1e14, 1e15)
    t2 = theta_r200c(m, M, z)
    vals = [itp(x*t2, M, z) / cm(x*t2, M, z) - 1 for x in (0.05, 0.3, 1.0, 2.0, 3.5)]
    println("julia z=$z M=$M: ", join([string(round(v; digits=4)) for v in vals], " "))
end
