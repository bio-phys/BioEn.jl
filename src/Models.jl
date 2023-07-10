
module DoubleWell

import StatsBase

function energy(x, n)
    a=(n+1.)/2.
    b=(n+2.)/4.
    return 3.0*((x.-a).^2.0.-b^2).^2.0./b^4
end

function distribution(x, n)
    u = energy(x,n)
    p = exp.(-u)
    p ./= sum(p)
    return p
end

function mean_from_distribution(x, p)
    return sum(x.*p)
end
    
function var_from_distribution(x, p, mean)    
    var=sum(x.^2.0.*p).-mean^2
    return var
end

function var_from_distribution(x, p)    
    var=sum(x.^2.0.*p).-mean_from_distribution(x,p)^2
    return var
end

function sample(N, x, p)
    return StatsBase.sample(x, StatsBase.Weights(p), N)
end

function hist(sample, n, density=true)
    counts = StatsBase.counts(sample, 1:n)
    if density
        return counts./size(sample)[1]
    else 
        return counts
    end
end 
    
end # end module DoubleWell