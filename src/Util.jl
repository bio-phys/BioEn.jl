module Util

function normalize(Y_org::Array{<:Real,1}, sigmas::Array{<:Real,1})
    Y = copy(Y_org)
    for i in eachindex(Y)
        Y[i] /= sigmas[i]
    end
    return Y 
end

function normalize(y_org::Array{<:Real,2}, sigmas::Array{<:Real,1})
    y = copy(y_org)
    for j in 1:size(y)[2]
        for i in 1:size(y)[1]
            y[i, j] /= sigmas[j]
        end
    end
    return y 
end

"""
Series of theta values linear in log-space 
"""
function theta_series(min, max, n; base=2.)
    powers = range(min, max, n)
    thetas = base.^powers
    return collect(powers), thetas
end

"""
Discrete relative entropy (Kullback-Leibler divergence)
"""
function SKL(p, q)
    S = 0.
    for i in eachindex(p)
        tmp = p[i]/q[i]
        if tmp>eps(0.0)
            S += p[i]*log(tmp)
        end
    end
    return S
end

"""
Average of calculated observables for given weights 

dim(y) = NxM, where N is the number of samples and M the number of data points. 
"""
function averages!(aves, w, y)
    for i in 1:size(y)[2] # observables 
        aves[i] = 0.
        for j in 1:size(y)[1]
            aves[i] += w[j]*y[j,i]
        end
    end
    return
end

function averages(w, y)
    aves = zeros(size(y)[2])
    averages!(aves, w, y)
    return aves
end

"""
Chi-squared value for given weight
"""
function chi2(aves, w, y, Y, sigmas)
    chi2 = 0.
    for i in eachindex(Y)
        chi2 += ( (aves[i]-Y[i])/sigmas[i] )^2
    end
    return chi2
end

function chi2!(aves, w, y, Y, sigmas)
    averages!(aves, w, y)
    return chi2(aves, w, y, Y, sigmas)
end

"""
BioEn negative log-posterior
"""
function neg_log_posterior(theta, S, c2)
    return theta*S + c2/2.
end

function neg_log_posterior!(aves, theta, w, w0, y, Y, sigmas)
    S = SKL(w, w0)
    c2 = chi2!(aves, w, y, Y, sigmas)
    return neg_log_posterior(theta, S, c2)
end

function neg_log_posterior(aves, theta, w, w0, y, Y, sigmas)
    S = SKL(w, w0)
    c2 = chi2(aves, w, y, Y, sigmas)
    return neg_log_posterior(theta, S, c2)
end

"""
Relative entropy from 2d array of weights
"""
function get_S_opt(theta_series, ws, w0)
    n_theta = size(theta_series)[1]
    S_opt = zeros(n_theta)
    for i in 1:n_theta
        S_opt[i] = SKL(ws[:,i], w0)
    end
    return S_opt
end

"""
Chi-squared from 2d array of weights
"""
function get_chi2_opt(theta_series, ws, y, Y, sigmas)
    n_theta = size(theta_series)[1]
    chi2_opt = zeros(n_theta)
    aves = zeros(size(Y)[1])
    for i in 1:n_theta
        chi2_opt[i] = chi2!(aves, ws[:,i], y, Y, sigmas)
    end
    return chi2_opt
end

end # end module 