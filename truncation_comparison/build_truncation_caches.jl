# Build the four XGPaint interpolator caches used by the truncation comparison:
# Battaglia16ThermalSZProfile and BattagliaTauProfile, each with the original
# projected (infinite LOS) evaluation and with the chord-mean spherical wrapper.
using XGPaint
include(joinpath(@__DIR__, "spherical_truncation_profiles.jl"))
using .SphericalTruncation

const H = 0.6774; const OMEGA_B = 0.0486; const OMEGA_C = 0.2603
const SPHERE_R200C = 4.0
cache_dir = joinpath(@__DIR__, "caches")
overwrite = "--overwrite" in ARGS

tsz = Battaglia16ThermalSZProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
tau = BattagliaTauProfile(Omega_c=OMEGA_C, Omega_b=OMEGA_B, h=H)
models = (
    ("tsz_projected", tsz),
    ("tsz_sphere4p0_chordmean", ChordMeanProfile(tsz, SPHERE_R200C)),
    ("tau_projected", tau),
    ("tau_sphere4p0_chordmean", ChordMeanProfile(tau, SPHERE_R200C)),
)
for (name, model) in models
    file = joinpath(cache_dir, "$(name)_interpolator.jld2")
    if isfile(file) && !overwrite
        println("exists, skipping: $file"); continue
    end
    println("Building $name -> $file with $(Threads.nthreads()) threads"); flush(stdout)
    t0 = time()
    build_interpolator(model; cache_file=file, N_logtheta=512, pad=128, logM_max=15.7, overwrite=true, verbose=true)
    println("done $name in $(round(time() - t0; digits=1)) s"); flush(stdout)
end
