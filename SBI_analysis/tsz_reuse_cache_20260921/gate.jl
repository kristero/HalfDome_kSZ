# Numerical equivalence of the two painting kernels and child-pixel mapping.
ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__, "benchmark.jl"))
using LinearAlgebra

task = TOML.parsefile(ENV["BENCHMARK_TASK"])
profiles = [build_cache(configure(case["theta"]).base_cfg,[128,64,32]) for case in task["cases"]]
rng = MersenneTwister(917260921)
positions = randn(rng, 3, 256)
masses = exp10.(12.7 .+ 2.5rand(rng,256))
redshifts = exp.(log(.003) .+ log(3.8/.003)*rand(rng,256))
serial = [blank_state(128) for _ in profiles]
shared = [blank_state(128) for _ in profiles]
for j in eachindex(profiles)
    paint_visual_batch!(serial[j],profiles[j],positions[1,:],positions[2,:],positions[3,:],
                        nothing,masses,redshifts)
end
paint_shared!(shared,profiles,positions,masses,redshifts)
errors = [norm(a.m_hp.pixels-b.m_hp.pixels)/norm(a.m_hp.pixels) for (a,b) in zip(serial,shared)]
@assert maximum(errors) < 1e-12
@assert all(all(isfinite,s.m_hp.pixels) && maximum(s.m_hp.pixels)>0 for s in shared)

# Fixed cosmology makes geometry exactly parameter-independent, not just close.
for profile in profiles, (mass,z) in zip(masses,redshifts)
    @assert theta_r200c(profile.model,mass,z) == theta_r200c(profiles[1].model,mass,z)
end

# Test equal-area nesting, conservation of total flux and unity preservation.
fine = blank_state(32).m_hp
fine.pixels .= rand(rng,length(fine.pixels))
coarse = average_children(fine,8)
flux_error = abs(sum(coarse.pixels)*16-sum(fine.pixels))/sum(fine.pixels)
@assert flux_error < 1e-14
for pixel in eachindex(fine.pixels)
    p = (Healpix.ring2nest(fine.resolution,pixel)-1) ÷ 16 + 1
    parent = Healpix.nest2ring(coarse.resolution,p)
    # Independent parent assignment via direction agrees with hierarchical IDs.
    tc,pc = Healpix.pix2angRing(fine.resolution,pixel)
    @assert Healpix.ang2pixRing(coarse.resolution,tc,pc) == parent
end
fill!(fine.pixels,1.)
@assert all(average_children(fine,8).pixels .== 1.)

save_toml(joinpath(OUT,"gate.toml"),Dict("passed"=>true,"painting_relative_l2"=>errors,
    "flux_relative_error"=>flux_error,"geometry_identical_across_cases"=>true,
    "nested_parent_direction_check"=>true,"threads"=>Threads.nthreads()))
println("PASS shared geometry, pressure evaluation, flux conservation and HEALPix nesting")
