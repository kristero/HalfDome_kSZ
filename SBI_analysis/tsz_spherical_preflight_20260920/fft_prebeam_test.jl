# Test XGPaint/Pixell's radial transform against independent real-space beam
# convolution. This is an experiment, not an adopted production renderer.
using XGPaint,DelimitedFiles,TOML
include(joinpath(@__DIR__,"..","..","truncation_comparison","spherical_truncation_profiles.jl"))
const ST=SphericalTruncation
const SIGMA=deg2rad(2/60)/sqrt(8log(2))
const CASES=[(.35,.497,4.35),(2.,.497,4.35),(1.,.025,4.35),(1.,.497,16.),(1.,1.13,.644)]
for n in (512,1024,2048)
    rft=XGPaint.RadialFourierTransform(n=n,pad=128,rrange=(1e-11,.1))
    matrix=zeros(length(rft.r),1+2length(CASES));matrix[:,1].=rft.r
    for (i,(t200,xc,beta)) in enumerate(CASES)
        t200=deg2rad(t200/60)
        raw=[r<4t200 ? ST.chord_quadrature(r/t200,xc,1.,beta+.3,-.3,
             sqrt((4-r/t200)*(4+r/t200))) : 0. for r in rft.r]
        harmonic=XGPaint.real2harm(rft,raw)
        harmonic .*= exp.(-.5 .* rft.l.^2 .* SIGMA^2)
        pre=XGPaint.harm2real(rft,reverse(harmonic))
        matrix[:,2i].=raw;matrix[:,2i+1].=pre
    end
    writedlm(joinpath(@__DIR__,"results","fft_prebeam_$(n).csv"),matrix,',')
end
println("FFT pre-beam profiles saved")
