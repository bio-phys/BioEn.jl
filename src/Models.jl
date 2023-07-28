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
    return 3*((x.-a).^2.0.-b^2).^2.0./b^4
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
        return counts./size(sample,1)
    else 
        return counts
    end
end 
    
end # end module DoubleWell

module Quartic

import StatsBase
import SpecialFunctions: besseli

function gen_x(dx, n)
    return [dx*i for i in range(-n, n)]
end



"""
    energy(x, a=3.0, x0=1.0)

V(x) = a(x^2-x0^2)^2. 
"""
function energy(x, a=3.0, x0=1.0)
    return a.*(x.^2.0.-x0^2).^2
end

function norm(a, x0)
    b = a*x0^2/2
    return norm = pi/2*exp(-b)*x0*(besseli(-1/4, b)+besseli(1/4, b))
end

"""
    distribution(x, a=3.0, x0=1.0)

Boltzmann distribution.
"""
function distribution(x, a=3.0, x0=1.0)
    return exp.(-energy(x, a, x0))./norm(a,x0)
end

"""
    distribution_max(a=3.0, x0=1.0)

Maximum value of Boltzmann distribution.

"""
function distribution_max(a=3.0, x0=1.0)
    return 1/norm(a,x0)
end

"""
    Grid{T<:Real}

A one-demensional grid (dx, n, x)

    Grid(dx, n)

Intialize grid x from (dx, n).
"""
struct Grid{T<:Real}
    dx::T
    n::Int64
    x::Vector{T}
end

function Grid(dx, n)
    return Grid(dx, n, gen_x(dx, n))
end

"""
    mean_from_distribution(grid, p; moment=1)

States grid.x = [grid.dx*i for i in range(-n, n)] also serve as observables. 

`moment` determines the moment. Default value `momement=1` give the arithmetic mean.
Use  `momement=0` for norm.
"""
function mean_from_distribution(grid, p; moment=1)
    return sum(grid.x.^moment.*p).*grid.dx
end

function sample(N, half_width=1.5, a=3.0, x0=1.0)
    x = (2.0*rand(N).-1).*half_width
    p = Quartic.distribution(x, a, x0)
    ran = rand(N).*Quartic.distribution_max(a, x0)
    sample = x[p.>ran]
    return sample
end

# """
#     var_from_distribution(x, p, mean)    

# Variance of states x = range(1,n), which also serve as observables, for given mean value.

#     var_from_distribution(x, p)  

# Calcualtion of mean value within function. 
# """
# function var_from_distribution(dx, n, p, mean)    
#     x = [dx*i for i in range(-n, n)]
#     var=sum(x.^2.0.*p).*dx.-mean^2
#     return var
# end

# function var_from_distribution(dx, n, p)    
#     x = [dx*i for i in range(-n, n)]
#     var = sum(x.^2.0.*p).*dx.-mean_from_distribution(dx, n, p)^2
#     return var
# end

# """
#     sample(N, x, p)

# Sampling of N states (=observables) from x using probability density p. 
# """
# function sample(N, width, a=3.0, x0=1.0)
#     x = (rand(N).-1/2).*width
#     p = distribution(x, a, b)
#     max = 
#     return

# """
#     hist(sample, x, density=true)

# Counting of states (integers) in 'sample'.

# 'x' serves to determine the range of states to count. 
# """
# function hist(sample, x, density=true)
#     counts = StatsBase.counts(sample, x)
#     if density
#         return counts./size(sample)[1]
#     else 
#         return counts
#     end
# end 
    
end # end module DoubleWell