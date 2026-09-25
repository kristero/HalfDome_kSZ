# Numerical audit only. No methods in XGPaint are replaced.
using XGPaint,TOML
include(joinpath(@__DIR__,"..","tsz_8192_validation_20260920","spherical_truncation_profiles.jl"))
const ST=SphericalTruncation

function stable_cylinder(x,xc,beta,zmax)
    # Same finite cylinder as stock XGPaint, evaluated in stable coordinates.
    return ST.chord_quadrature(x,xc,1.,beta+.3,-.3,zmax;rtol=1e-10,maxevals=4096)
end

if ARGS[1]=="parse"
    source=read(joinpath(@__DIR__,"inputs","stock_profiles.jl"),String)
    ast=Meta.parseall(source);errors=String[]
    function visit(ex)
        ex isa Expr || return
        ex.head in (:error,:incomplete) && push!(errors,sprint(show,ex)[1:min(end,1800)])
        foreach(visit,ex.args)
    end
    visit(ast)
    # Demonstrate exactly the stock relative-floor arithmetic.
    minimum_positive=1e-320;floor=minimum_positive*1e-6
    open(joinpath(@__DIR__,"results/stock_parse.toml"),"w") do io
        TOML.print(io,Dict("parse_errors"=>errors,"relative_floor"=>floor,
            "minimum_positive"=>minimum_positive,"log_floor"=>log(floor)))
    end
else
    label,x,xc,beta=ARGS[1],parse.(Float64,ARGS[2:4])...
    # Compile on a benign point before the measured call.
    XGPaint._nfw_profile_los_quadrature(.1,.497,1.,4.65,-.3)
    reference=stable_cylinder(x,xc,beta,1e5)
    sphere=x<4 ? stable_cylinder(x,xc,beta,sqrt((4-x)*(4+x))) : 0.
    started=time_ns()
    value=XGPaint._nfw_profile_los_quadrature(x,xc,1.,beta+.3,-.3)
    seconds=(time_ns()-started)/1e9
    integrand(l)=1e9*XGPaint.generalized_nfw(hypot(l,x),xc,1.,beta+.3,-.3)
    initial,estimated=XGPaint.QuadGK.quadgk(integrand,0.,1e5;rtol=1e-12,order=9,maxevals=19)
    record=Dict("label"=>label,"x"=>x,"xc"=>xc,"beta_raw"=>beta,
        "stock_value"=>value,"stable_same_cylinder"=>reference,"sphere4"=>sphere,
        "stock_seconds_after_compile"=>seconds,"initial19_value"=>2initial/1e9,
        "initial19_error"=>2estimated/1e9,"stock_relative_error"=>abs(value-reference)/max(reference,1e-300),
        "cylinder_to_sphere"=>reference/max(sphere,1e-300))
    if beta<.7
        record["cutoff_cylinder_1e4_1e5_1e6"]=[stable_cylinder(x,xc,beta,l) for l in (1e4,1e5,1e6)]
    end
    open(joinpath(@__DIR__,"results","stock_"*label*".toml"),"w") do io;TOML.print(io,record);end
    println(record)
end
