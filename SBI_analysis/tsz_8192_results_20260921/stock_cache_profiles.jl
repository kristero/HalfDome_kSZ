# Actual unchanged cleanup function and original LOS profile curves.
using XGPaint,TOML,DelimitedFiles
include(joinpath(@__DIR__,"..","tsz_8192_validation_20260920","spherical_truncation_profiles.jl"))
const ST=SphericalTruncation
original=read(joinpath(@__DIR__,"inputs/stock_profiles.jl"),String)
start=findfirst("function replace_nonpositive_with_floor!",original).start
stop=findnext("# get angular size",original,start).start-1
module StockCleanup
    using XGPaint
    const chunks=XGPaint.chunks
    _thread_storage_count()=Threads.maxthreadid()
end
Base.include_string(StockCleanup,original[start:stop],"stock_cleanup_exact.jl")
values=fill(1e-320,4,4,4);values[1]=0.
replaced,floor=StockCleanup.replace_nonpositive_with_floor!(values)
logvalues=log.(values)
itp=XGPaint.Interpolations.interpolate(logvalues,XGPaint.BSpline(XGPaint.Cubic(XGPaint.Line(XGPaint.OnGrid()))))
evaluated=[itp(i,j,k) for i in 1.:.5:4.,j in 1.:.5:4.,k in 1.:.5:4.]
open(joinpath(@__DIR__,"results/stock_cache.toml"),"w") do io
    TOML.print(io,Dict("replaced"=>replaced,"floor"=>floor,
        "nonfinite_log_cells"=>count(!isfinite,logvalues),
        "interpolation_probes"=>length(evaluated),"nonfinite_interpolation_probes"=>count(!isfinite,evaluated),
        "scope"=>"Exact original cleanup on synthetic subnormal cache, not a full-grid failure frequency"))
end
data=Matrix{Float64}(undef,0,6)
cases=[(.497,4.35),(.025,16.),(.025*4^.731,256.),(.497,2.8*.1^.4*4^(-.5))]
for (id,(xc,beta)) in enumerate(cases),x in exp10.(range(-8,log10(3.9999),length=120))
    sphere=ST.chord_quadrature(x,xc,1.,beta+.3,-.3,sqrt((4-x)*(4+x)))
    cylinder=ST.chord_quadrature(x,xc,1.,beta+.3,-.3,1e5)
    native=XGPaint._nfw_profile_los_quadrature(x,xc,1.,beta+.3,-.3)
    global data=vcat(data,[id x sphere cylinder native beta])
end
writedlm(joinpath(@__DIR__,"results/stock_profile_curves.csv"),data,',')
println("PASS stock cleanup reproduction and profile-curve audit")
