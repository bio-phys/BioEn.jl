module DoubleWell

import StatsBase

"""
    energy(x)

Reduced energies for states x = range(1,n).
"""
function energy(x)
    n = size(x, 1)
    a=(n+1.)/2.
    b=(n+2.)/4.
    return 3.0*((x.-a).^2.0.-b^2).^2.0./b^4
end

"""
    distribution(x)

Boltzmann distributions for states x = range(1,n).
"""
function distribution(x)
    u = energy(x)
    p = exp.(-u)
    p ./= sum(p)
    return p
end

"""
    mean_from_distribution(x, p)

States x = range(1,n) also serve as observables. 
"""
function mean_from_distribution(x, p)
    return sum(x.*p)
end

"""
    var_from_distribution(x, p, mean)    

Variance of states x = range(1,n), which also serve as observables, for given mean value.

    var_from_distribution(x, p)  

Calcualtion of mean value within function. 
"""
function var_from_distribution(x, p, mean)    
    var=sum(x.^2.0.*p).-mean^2
    return var
end

function var_from_distribution(x, p)    
    var=sum(x.^2.0.*p).-mean_from_distribution(x,p)^2
    return var
end

"""
    sample(N, x, p)

Sampling of N states (=observables) from x using probability density p. 
"""
function sample(N, x, p)
    return StatsBase.sample(x, StatsBase.Weights(p), N)
end

"""
    hist(sample, x, density=true)

Counting of states (integers) in 'sample'.

'x' serves to determine the range of states to count. 
"""
function hist(sample, x, density=true)
    counts = StatsBase.counts(sample, x)
    if density
        return counts./size(sample)[1]
    else 
        return counts
    end
end 
    
end # end module DoubleWell