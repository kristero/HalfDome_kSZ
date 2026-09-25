using XGPaint,DelimitedFiles,TOML,Test
include(joinpath(@__DIR__,"spherical_truncation_profiles.jl"))
const ST=SphericalTruncation
points=readdlm(joinpath(@__DIR__,"inputs","columns.csv"),',',Float64;skipstart=1)
errors=Float64[]
for row in eachrow(points)
    x,xc,beta,reference=row
    value=x>=4 ? 0. : ST.chord_quadrature(x,xc,1.,beta+.3,-.3,sqrt((4-x)*(4+x)))
    @assert isfinite(value) && value>=0
    max(value,reference)>1e-280 && push!(errors,abs(value-reference)/max(value,reference))
end
@assert maximum(errors)<1e-7
@test_throws ErrorException ST.checked_quadrature(x->1+.5sin(10000x),0.,1.;rtol=1e-13,maxevals=20)
low=[1.,.025,2.8,-.6,-1.,-.2,-6.,-1.5,-.5]
high=[60.,4.,16.,1.5,.4,.4,.5,3.,2.]
count=0
for corner in 0:511
    v=[((corner>>(j-1))&1)==0 ? low[j] : high[j] for j in 1:9]
    model=Battaglia16ThermalSZProfile(P0_amp=v[1],x_c_amp=v[2],beta_amp=v[3],
        P0_alpha_m=v[4],x_c_alpha_m=v[5],beta_alpha_m=v[6],P0_alpha_z=v[7],
        x_c_alpha_z=v[8],beta_alpha_z=v[9],h=.68,Omega_b=.049,Omega_c=.261)
    sphere=ST.ChordMeanProfile(model,4.)
    for mass in (1e12,1e14,10.0^15.7),z in (.001,.5,5.)
        p=XGPaint.prepare_profile_slice(sphere,mass,z)
        for x in (0.,1e-10,1e-6,.001,.1,1.,3.99,3.9999999999)
            value=p.A*ST.chord_quadrature(x,p.xc,p.α,p.β,p.γ,sqrt((4-x)*(4+x)))
            @assert isfinite(value) && value>=0
            global count+=1
        end
    end
end
open(joinpath(@__DIR__,"results","quadrature.toml"),"w") do io
    TOML.print(io,Dict("reference_points"=>size(points,1),"max_relative_error"=>maximum(errors),
        "corner_columns"=>count,"maxevals"=>4096,"rtol"=>1e-10,"failure_path_passed"=>true))
end
println("PASS reference columns and ",count," bounded corner integrals")
