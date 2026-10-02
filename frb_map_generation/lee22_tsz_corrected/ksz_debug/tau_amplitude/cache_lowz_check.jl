# Accuracy of the b16 chord-mean tau cache (log-cubic in z on a 0.001..5 grid with dz ~ 0.0196) at low redshift, where theta200
# changes by large factors between z grid nodes. Uses the private cache copy; read only.
using XGPaint, Unitful, UnitfulAstro, Printf
include("/home/cbllover/HalfDome/truncation_comparison/spherical_truncation_profiles.jl")
using .SphericalTruncation
const OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/tau_amplitude"
model = BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68)
cm = ChordMeanProfile(model, 4.0)
itp = build_interpolator(cm; cache_file=joinpath(OUT, "tau_b16_sphere4p0_chordmean_interpolator_COPY.jld2"), overwrite=false, verbose=false)
for z in (0.004, 0.0125, 0.0375, 0.0625, 0.1125, 0.2), M in (1e13, 1e14, 1e15)
    th = theta_r200c(model, M, z)
    devs = [itp(f * th, M, z) / cm(f * th, M, z) - 1 for f in (0.05, 0.3, 1.0, 2.0, 3.5)]
    @printf("z=%.4f M=%.0e: cache/direct-1 at x=0.05,0.3,1,2,3.5: %s\n", z, M, join([@sprintf("%+.4f", d) for d in devs], " "))
end
