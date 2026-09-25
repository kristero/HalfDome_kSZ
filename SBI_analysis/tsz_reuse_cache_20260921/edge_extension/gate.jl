ENV["BENCHMARK_LOAD_ONLY"] = "1"
include(joinpath(@__DIR__,"..","benchmark.jl"))

cases=[(.497,4.35),(4.,2.8),(.025,16.),(1.,.2)]
xs=[0.,1e-8,.01,1.,3.5,3.999999]
before=[SphericalTruncation.chord_mean(x,xc,1.,beta+.3,-.3,4.) for (xc,beta) in cases for x in xs]
include(joinpath(@__DIR__,"smooth_exterior.jl"))
after=[SphericalTruncation.chord_mean(x,xc,1.,beta+.3,-.3,4.) for (xc,beta) in cases for x in xs]
@assert before == after
derivative_errors=Float64[]
curves=Vector{Float64}[]
for (id,(xc,beta)) in enumerate(cases)
    X=4.; epsilon=1e-5
    p=XGPaint.generalized_nfw(X,xc,1.,beta+.3,-.3)
    expected=(2/3)*p*(-.3-beta*X/(xc+X))/X
    right=(smooth_exterior_mean(X+epsilon,xc,1.,beta+.3,-.3,X)-
           smooth_exterior_mean(X,xc,1.,beta+.3,-.3,X))/epsilon
    left=(SphericalTruncation.chord_mean(X,xc,1.,beta+.3,-.3,X)-
          SphericalTruncation.chord_mean(X-epsilon,xc,1.,beta+.3,-.3,X))/epsilon
    push!(derivative_errors,max(abs(right/expected-1),abs(left/expected-1)))
    @assert derivative_errors[end] < 1e-4
    for x in range(3.,5.,length=201)
        smooth=SphericalTruncation.chord_mean(x,xc,1.,beta+.3,-.3,X)
        original=x<X ? smooth : XGPaint.generalized_nfw(x,xc,1.,beta+.3,-.3)
        push!(curves,[id,x,original/p,smooth/p])
    end
    @assert chord_factor(4.001,4.,4.) == 0.
end
for x in (4.,40.,4e6),xc in (.00001,.5,1e4),beta in (.1,2.8,16.,1000.)
    value=smooth_exterior_mean(x,xc,1.,beta+.3,-.3,4.)
    @assert isfinite(value) && value>=0
end
save_toml(joinpath(OUT,"gate.toml"),Dict("passed"=>true,
    "interior_bitwise_unchanged"=>true,"edge_derivative_relative_errors"=>derivative_errors,
    "physical_support_unchanged"=>true))
writedlm(joinpath(OUT,"edge_curves.csv"),reduce(hcat,curves)',',')
println("PASS unchanged interior and support, continuous edge derivative, finite unused tails")
