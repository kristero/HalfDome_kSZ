using Test
design_dir = isempty(ARGS) ? joinpath(@__DIR__, "designs") : abspath(first(ARGS))
include(joinpath(@__DIR__, "simulator", "tSZ_visuals", "run_halfdome_fullsky_so_noise.jl"))
@test VERSION == v"1.12.2"
@test realpath(pathof(XGPaint)) == realpath(joinpath(@__DIR__, "vendor", "XGPaint", "src", "XGPaint.jl"))
fields = (:P0_amp, :x_c_amp, :beta_amp, :P0_alpha_m, :x_c_alpha_m,
          :beta_alpha_m, :P0_alpha_z, :x_c_alpha_z, :beta_alpha_z)
for mode in ("two_param", "nine_param")
    csv_path = joinpath(design_dir, mode * ".csv")
    table, _ = readdlm(csv_path, ','; header=true)
    for row in axes(table, 1)
        p = merge(default_battaglia_params(), NamedTuple{fields}(Tuple(table[row, :])))
        reasons = validate_battaglia_params(p; enforce_prior_bounds=false, enforce_derived_bounds=true)
        isempty(reasons) || error("$(mode) row $(row): $(join(reasons, "; "))")
    end
    println("All ", size(table, 1), " pressure rows pass physical guardrails: ", mode)
    for row in (1, size(table, 1))
        p = battaglia_params_from_sobol_row(csv_path, row)
        @test [getproperty(p, key) for key in fields] == table[row, :]
        @test p.alpha_amp == 1.0
        @test p.gamma_amp == -0.3
    end
end
# Small map checks only: no catalogue or expensive NSIDE=4096 run is needed.
cl = ones(32) .* 1e-15
n1 = generate_gaussian_noise_map(cl, 16, 31, MersenneTwister(1010101))
n1_again = generate_gaussian_noise_map(cl, 16, 31, MersenneTwister(1010101))
n2 = generate_gaussian_noise_map(cl, 16, 31, MersenneTwister(1010102))
@test n1.pixels == n1_again.pixels
@test n1.pixels != n2.pixels
m1 = random_apodized_cap_mask(16, 0.4, 60.0, MersenneTwister(12345))
m2 = random_apodized_cap_mask(16, 0.4, 60.0, MersenneTwister(12345))
@test m1.mask.pixels == m2.mask.pixels
if haskey(ENV, "HALFDOME_PATH") && isfile(ENV["HALFDOME_PATH"])
    h5open(ENV["HALFDOME_PATH"], "r") do catalogue
        @test all(haskey(catalogue, key) for key in ("Position", "halo_mass_m200c", "redshift"))
        n = length(catalogue["halo_mass_m200c"])
        @test size(catalogue["Position"], 2) == n
        @test length(catalogue["redshift"]) == n
        println("Catalogue layout passed; N_halo=", n)
    end
end
println("PASSED: vendored runtime, CSV parameter mapping, fixed mask and distinct noise maps.")
