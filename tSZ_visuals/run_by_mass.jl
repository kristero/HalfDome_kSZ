ENV["BATCHING_MODE"] = "mass"

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))
run_tsz_visual_fits()
