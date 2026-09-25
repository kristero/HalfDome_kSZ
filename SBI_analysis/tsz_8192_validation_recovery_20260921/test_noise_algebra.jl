# Small, independent check of map-route versus masked-alm noise splitting.
using Healpix,Random,LinearAlgebra,TOML
nside=32;lmax=63;rng=MersenneTwister(20260920)
cl=[ell<2 ? 0. : 1e-10/(ell*(ell+1)) for ell in 0:lmax]
signal=Healpix.alm2map(Healpix.synalm(cl,lmax,lmax,rng),nside)
mask=[Healpix.pix2vecRing(signal.resolution,p)[3]>.2 ? 1. : 0. for p in eachindex(signal.pixels)]
signal.pixels .*=mask
first=Healpix.alm2map(Healpix.synalm(cl,lmax,lmax,rng),nside)
second=Healpix.alm2map(Healpix.synalm(cl,lmax,lmax,rng),nside)
first.pixels .*=mask;second.pixels .*=mask
s=Healpix.map2alm(signal;lmax=lmax,niter=0)
a=Healpix.map2alm(first;lmax=lmax,niter=0);b=Healpix.map2alm(second;lmax=lmax,niter=0)
fast=Healpix.alm2cl(s)+Healpix.alm2cl(s,a)+Healpix.alm2cl(s,b)+Healpix.alm2cl(a,b)
first.pixels .+=signal.pixels;second.pixels .+=signal.pixels
original=Healpix.anafast(first,second;lmax=lmax,niter=0)
error=norm(original-fast)/norm(original)
@assert error<1e-12
open(joinpath(@__DIR__,"results","noise_algebra.toml"),"w") do io
    TOML.print(io,Dict("relative_error"=>error,"nside"=>nside,"lmax"=>lmax,"full_row_validated"=>false))
end
println("PASS masked-noise algebra: ",error)
