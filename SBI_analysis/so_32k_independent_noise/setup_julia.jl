get!(ENV, "JULIA_NUM_PRECOMPILE_TASKS", "2")
using Pkg
root = @__DIR__
VERSION == v"1.12.2" || error("Use Julia 1.12.2 for the supplied lockfile and RNG reproducibility.")
Pkg.activate(joinpath(root, "vendor", "XGPaint"))
Pkg.instantiate()
Pkg.precompile()
using XGPaint, Healpix, HDF5, Interpolations
println("Julia: ", VERSION)
println("XGPaint: ", pathof(XGPaint))
println("Dependency installation complete. Run check_runtime.jl next.")
