using XGPaint, DelimitedFiles, TOML
const ROOT=@__DIR__
include(get(ENV, "SPHERICAL_WRAPPER", joinpath(ROOT,"..","..","truncation_comparison","spherical_truncation_profiles.jl")))
const ST=SphericalTruncation
points=readdlm(joinpath(ROOT,"inputs","columns.csv"), ',', Float64; skipstart=1)
values=zeros(size(points,1));errors=Float64[]
started=time()
for i in axes(points,1)
    x,xc,b,reference=points[i,:]
    # Raw Battaglia exponent must be converted to XGPaint's internal slope.
    values[i]=x>=4 ? 0.0 : ST.chord_quadrature(x,xc,1.0,b+.3,-.3,sqrt((4-x)*(4+x)))
    if max(values[i],reference)>1e-280
        push!(errors,abs(values[i]-reference)/max(values[i],reference))
    end
end
@assert all(isfinite,values) && all(values .>= 0)
@assert maximum(errors)<1e-7
# Check that the spherical tSZ path never calls the old LOS routine.
@eval XGPaint function _nfw_profile_los_quadrature(x,xc,α,β,γ; kwargs...)
    error("old LOS intentionally disabled by normalization regression test")
end
normalization=Dict[]
for beta0 in (.1,.644,4.35,16.), xc0 in (.025,.497,4.)
    model=XGPaint.Battaglia16ThermalSZProfile(beta_amp=beta0,x_c_amp=xc0)
    sphere=ST.ChordMeanProfile(model,4.)
    p=XGPaint.prepare_profile_slice(sphere,1e14,.5)
    prepared=XGPaint.prepare_profile_slice(model,1e14,.5)
    @assert p.A==prepared.amplitude
    val=ST.spherical_value_direct(sphere,.1*p.θ200,1e14,.5)
    @assert isfinite(val) && val>0
    push!(normalization,Dict("beta0"=>beta0,"xc0"=>xc0,"amplitude"=>p.A,"value"=>val))
end
mkpath(joinpath(ROOT,"results"))
writedlm(joinpath(ROOT,"results","julia_columns.csv"),values,',')
open(joinpath(ROOT,"results","profile_julia.toml"),"w") do io
    TOML.print(io,Dict("columns"=>length(values),"maximum_relative_error"=>maximum(errors),
        "normalization_without_old_los_passed"=>true,"seconds"=>time()-started,
        "normalization_cases"=>normalization,"xgpaint_source"=>pathof(XGPaint)))
end
println("PASS ",length(values)," columns; maximum relative error=",maximum(errors))
