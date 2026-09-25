# Smooth continuation of the CACHED chord mean beyond the physical sphere.
# No gas is added outside X: the painter still applies its exact chord/disc cut.
using DelimitedFiles
const EXTERIOR_RULE = readdlm(joinpath(@__DIR__,"legendre16.csv"),',',Float64)
const EXTERIOR_U = EXTERIOR_RULE[:,1]
const EXTERIOR_W = EXTERIOR_RULE[:,2]

function smooth_exterior_mean(x,xc,alpha,beta,gamma,X)
    # h(x)=int_0^1 p(sqrt(x^2+(X^2-x^2)u^2))du is the exact chord mean
    # inside X. For x>X it is a positive analytic continuation, never painted.
    # Log-sum-exp avoids underflow in steep, unused cache tails.
    logs = ntuple(16) do i
        u = EXTERIOR_U[i]
        radius = sqrt(x*x*(1-u*u) + X*X*u*u)
        logq = log(radius/xc)
        gamma*logq - ((beta+gamma)/alpha)*SphericalTruncation.log1pexp_stable(alpha*logq)
    end
    peak = maximum(logs)
    total = sum(EXTERIOR_W[i]*exp(logs[i]-peak) for i in 1:16)
    return exp(peak+log(total))
end

# Preserve the exact original interior branch and bounded LOS quadrature.
function SphericalTruncation.chord_mean(x,xc,alpha,beta,gamma,X)
    if x < X
        L = sqrt((X-x)*(X+x))
        return SphericalTruncation.chord_quadrature(x,xc,alpha,beta,gamma,L)/(2L)
    end
    return smooth_exterior_mean(x,xc,alpha,beta,gamma,X)
end
