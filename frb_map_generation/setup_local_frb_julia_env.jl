#!/usr/bin/env julia

# Build an isolated local environment for the fixed-z FRB generator. This
# avoids the cluster Manifest's machine-specific XGPaint path and keeps
# compiled caches out of a shared or incorrectly owned Julia depot.

using Pkg

length(ARGS) == 2 || error(
    "Usage: julia setup_local_frb_julia_env.jl ENVIRONMENT_DIR XGPAINT_SOURCE_DIR"
)

environment_dir = abspath(ARGS[1])
xgpaint_source = abspath(ARGS[2])
isdir(xgpaint_source) || error("Local XGPaint source does not exist: $(xgpaint_source)")
isfile(joinpath(xgpaint_source, "Project.toml")) || error(
    "Local XGPaint source has no Project.toml: $(xgpaint_source)"
)

mkpath(environment_dir)
Pkg.activate(environment_dir)
Pkg.add([
    Pkg.PackageSpec(name="HDF5"),
    Pkg.PackageSpec(name="Healpix"),
    Pkg.PackageSpec(name="Interpolations"),
])
Pkg.develop(path=xgpaint_source)
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()

println("Local FRB Julia environment is ready: $(environment_dir)")
println("XGPaint source: $(xgpaint_source)")
