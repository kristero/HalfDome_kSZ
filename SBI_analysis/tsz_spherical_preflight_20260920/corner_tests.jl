# All 512 box corners: finite projected columns and physical amplitudes.
using XGPaint,TOML
include(joinpath(@__DIR__,"..","..","truncation_comparison","spherical_truncation_profiles.jl"))
const ST=SphericalTruncation
const LOW=[1.,.025,2.8,-.6,-1.,-.2,-6.,-1.5,-.5]
const HIGH=[60.,4.,16.,1.5,.4,.4,.5,3.,2.]
started=time();evaluated=0;zeros_count=0;largest=0.;max_amplitude=0.
for corner in 0:511
    p=[((corner>>(j-1))&1)==0 ? LOW[j] : HIGH[j] for j in 1:9]
    model=Battaglia16ThermalSZProfile(P0_amp=p[1],x_c_amp=p[2],beta_amp=p[3],
        P0_alpha_m=p[4],x_c_alpha_m=p[5],beta_alpha_m=p[6],
        P0_alpha_z=p[7],x_c_alpha_z=p[8],beta_alpha_z=p[9],h=.68,Omega_b=.0486,Omega_c=.2603)
    sphere=ST.ChordMeanProfile(model,4.)
    for mass in (1e12,1e14,10.0^15.7),z in (.001,.5,5.)
        slice=XGPaint.prepare_profile_slice(sphere,mass,z)
        global max_amplitude=max(max_amplitude,slice.A)
        for x in (0.,1e-10,1e-6,.001,.1,1.,3.99,3.9999999999)
            value=slice.A*ST.chord_quadrature(x,slice.xc,slice.α,slice.β,slice.γ,sqrt((4-x)*(4+x)))
            @assert isfinite(value) && value>=0
            global evaluated+=1
            global zeros_count+=(value==0)
            global largest=max(largest,value)
        end
    end
end
result=Dict("box_corners"=>512,"columns"=>evaluated,"underflow_count"=>zeros_count,
    "maximum_y"=>largest,"maximum_amplitude_on_grid"=>max_amplitude,
    "seconds"=>time()-started,"scope"=>"Scalar projection and amplitude; not interpolation or map fidelity")
open(joinpath(@__DIR__,"results","corner_tests.toml"),"w") do io;TOML.print(io,result);end
println(result)
