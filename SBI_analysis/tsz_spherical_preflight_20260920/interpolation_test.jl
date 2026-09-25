# Reproduce the clean-map log-z chord-mean cache and audit off-grid halo slices.
using XGPaint,Random,Statistics,TOML
include(joinpath(@__DIR__,"..","..","truncation_comparison","spherical_truncation_profiles.jl"))
const ST=SphericalTruncation
const ROOT=@__DIR__
const results=Dict[]
const PAD=parse(Int,get(ENV,"CACHE_TEST_PAD","128"))
const OMEGA_B=parse(Float64,get(ENV,"CACHE_TEST_OMEGA_B","0.0486"))
const OMEGA_C=parse(Float64,get(ENV,"CACHE_TEST_OMEGA_C","0.2603"))
const FACTORS=parse.(Int,split(get(ENV,"CACHE_TEST_FACTORS","1,2"),","))
const CASES=split(get(ENV,"CACHE_TEST_CASES","B12,compact,shallow"),",")
const OUTPUT_STEM=get(ENV,"CACHE_TEST_OUTPUT_STEM","interpolation")
for (label,xc,beta) in (("B12",.497,4.35),("compact",.025,16.),("shallow",4.,2.8))
    label in CASES || continue
    model=ST.ChordMeanProfile(Battaglia16ThermalSZProfile(x_c_amp=xc,beta_amp=beta,
        h=.68,Omega_b=OMEGA_B,Omega_c=OMEGA_C),4.)
    for factor in FACTORS
        rft=XGPaint.RadialFourierTransform(n=512,pad=PAD)
        lt=LinRange(log(minimum(rft.r)),log(maximum(rft.r)),512factor)
        lz=LinRange(log(.001),log(5.),256factor);lm=LinRange(12.,15.7,128factor)
        started=time()
        _,_,_,values=XGPaint.profile_grid(model,lt,exp.(lz),lm)
        @assert all(isfinite,values) && minimum(values)>=0
        floored=count(<=(0),values)
        values .= log.(max.(values,1e-300))
        interp=XGPaint.scale(XGPaint.Interpolations.interpolate(values,
            XGPaint.BSpline(XGPaint.Cubic(XGPaint.Line(XGPaint.OnGrid())))),lt,lz,lm)
        rng=MersenneTwister(20260920);relative=Float64[];central_scaled=Float64[]
        angular_floor_hits=0
        for i in 1:10000
            mass=10.0^(12+3.7rand(rng));z=exp(log(.001)+rand(rng)*log(5000))
            x=exp(log(1e-8)+rand(rng)*log(4/1e-8))
            slice=XGPaint.prepare_profile_slice(model,mass,z)
            theta=x*slice.θ200
            theta_eval=max(theta,exp(first(lt)))
            angular_floor_hits+=(theta_eval!=theta)
            estimate=exp(interp(log(theta_eval),log(z),log10(mass)))*2sqrt((4-x)*(4+x))
            exact=ST.spherical_value_direct(model,theta,mass,z)
            centre=ST.spherical_value_direct(model,0.,mass,z)
            @assert isfinite(estimate) && estimate>=0
            exact>1e-30 && push!(relative,abs(estimate/exact-1))
            push!(central_scaled,abs(estimate-exact)/centre)
        end
        result=Dict("case"=>label,"grid_factor"=>factor,"seconds"=>time()-started,
            "angular_pad"=>PAD,"theta_min_rad"=>exp(first(lt)),"theta_max_rad"=>exp(last(lt)),
            "Omega_b"=>OMEGA_B,"Omega_c"=>OMEGA_C,"h"=>.68,
            "cache_underflow_cells"=>floored,"angular_floor_hits"=>angular_floor_hits,
            "relative_error_p50_p95_p99_max"=>quantile(relative,[.5,.95,.99,1.]),
            "central_scaled_error_p50_p95_p99_max"=>quantile(central_scaled,[.5,.95,.99,1.]),
            "scope"=>"Pointwise cache + historical angular floor; not a bandpower error budget")
        push!(results,result)
        open(joinpath(ROOT,"results",OUTPUT_STEM*".toml"),"w") do io;TOML.print(io,Dict("cases"=>results));end
        println(result);flush(stdout)
        values=nothing;interp=nothing;GC.gc()
    end
end
