ENV["BATCHING_MODE"] = "redshift"

include(joinpath(@__DIR__, "run_tSZ_visuals.jl"))
run_tsz_visual_fits()
