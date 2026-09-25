ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__,"..","benchmark.jl"))
include(joinpath(@__DIR__,"smooth_exterior.jl"))
main_benchmark()
