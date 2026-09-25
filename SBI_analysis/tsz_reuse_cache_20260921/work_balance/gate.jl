ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__,"..","benchmark.jl"))
using LinearAlgebra
task=TOML.parsefile(ENV["BENCHMARK_TASK"])
profiles=[build_cache(configure(c["theta"]).base_cfg,[128,64,32]) for c in task["cases"]]
rng=MersenneTwister(11260921)
positions=randn(rng,3,1024)
masses=exp10.(12.7 .+ 2.5rand(rng,1024))
redshifts=exp.(log(.003) .+ log(3.8/.003)*rand(rng,1024))
original=[blank_state(128) for _ in profiles]
for j in eachindex(profiles)
    paint_visual_batch!(original[j],profiles[j],positions[1,:],positions[2,:],positions[3,:],
                        nothing,masses,redshifts)
end
include(joinpath(@__DIR__,"balanced_painter.jl"))
errors=Dict[]
for block_size in (16,256)
    ENV["HALO_BLOCK_SIZE"]=string(block_size)
    states=[blank_state(128) for _ in profiles]
    paint_shared!(states,profiles,positions,masses,redshifts)
    relative=[norm(s.m_hp.pixels-o.m_hp.pixels)/norm(o.m_hp.pixels) for (s,o) in zip(states,original)]
    @assert maximum(relative)<1e-12
    push!(errors,Dict("block_size"=>block_size,"relative_map_l2"=>relative))
end
save_toml(joinpath(OUT,"gate.toml"),Dict("passed"=>true,"tests"=>errors,
    "halos"=>1024,"threads"=>Threads.nthreads()))
println("PASS greedy blocks reproduce original pressure maps")
