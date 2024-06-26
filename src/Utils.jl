"""
The BioEn posterior [Hummer and Köfinger, J. Chem. Phys.  143 (2015)] and helper functions. 
"""
module Utils

# todo: replace S by SKL

function xlogx(x::Number)
    result = x * log(x)
    return iszero(x) ? zero(result) : result
end

function xlogy(x::Number, y::Number)
    result = x * log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end
    

"""
    normalize(Y_org::Array{T,1}, sigmas::Array{T,1}) where T<:AbstractFloat

Normalize experimental observables by errors sigma.

    function normalize(y_org::Array{T,2}, sigmas::Array{T,1}) where T<:AbstractFloat

Normalize calcualted observables by errors sigma.

"""
function normalize(Y_org::Array{T,1}, sigmas::Array{T,1}) where T<:AbstractFloat
    Y = copy(Y_org)
    for i in eachindex(Y)
        Y[i] /= sigmas[i]
    end
    return Y 
end

function normalize(y_org::Array{T,2}, sigmas::Array{T,1}) where T<:AbstractFloat
    y = copy(y_org)
    for j in 1:size(y, 2)
        for i in 1:size(y, 1)
            y[i, j] /= sigmas[j]
        end
    end
    return y 
end

"""
    theta_series(min, max, n; base=2.)

Series of theta values linear in log-space.
"""
function theta_series(min, max, n; base=2.)
    powers = range(min, max, n)
    thetas = base.^powers
    return collect(powers), thetas
end

"""
    SKL(p::Vector{T}, q::Vector{T}) where T<:AbstractFloat

Discrete relative entropy (Kullback-Leibler divergence).

    SKL(p::Vector{T}, q::Vector{T}, logq::Vector{T}) where T<:AbstractFloat

Pre-calculated `logq' = log(w0) for increased efficiency. 
"""
function SKL(p::Vector{T}, q::Vector{T}) where T<:AbstractFloat
    S = zero(T)
    for i in eachindex(p) #1:size(p,1) #
        S += xlogy(p[i], p[i]/q[i]) 
    end
    return S
end

function SKL(p::Vector{T}, q::Vector{T}, logq::Vector{T}) where T<:AbstractFloat
    S = zero(T)
    for i in eachindex(p) #1:size(p,1) #
        S += xlogx(p[i])-logq[i] 
    end
    return S
end

"""
    averages!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat

Average of calculated observables for given weights 

dim(y) = NxM, where N is the number of samples and M the number of data points. 
"""
function averages!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat
    for i in axes(y,2) # observables 
        aves[i] = 0.
        for j in axes(y,1) #
            aves[i] += w[j]*y[j,i]
        end
    end
    return
end

"""
    averages(w::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat

Average of calculated observables for given weights 

dim(y) =  NxM, where N is the number of samples and M the number of data points. 
"""
function averages(w::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat
    aves = zeros(size(y, 2))
    averages!(aves, w, y)
    return aves
end


"""
    chi2(aves::Vector{T}, Y::Vector{T}) where T<:AbstractFloat

Chi-squared value for given averages using normalized observabels.

    chi2(w::Vector{T}, y::Array{T, 2}, Y::Vector{T}) where T<:AbstractFloat

Chi-squared value using normalized observabels.
"""
function chi2(aves::Vector{T}, Y::Vector{T}) where T<:AbstractFloat
    chi2 = zero(T)
    for i in eachindex(Y)
        chi2 += (aves[i]-Y[i])^2
    end
    return chi2
end

function chi2(w::Vector{T}, y::Array{T, 2}, Y::Vector{T}) where T<:AbstractFloat
    aves = averages(w, y)
    return chi2(aves, Y)
end

"""
    chi2!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}, Y::Vector{T}) where T<:AbstractFloat

Calculate averages in-place. 
"""
function chi2!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}, Y::Vector{T}) where T<:AbstractFloat
    averages!(aves, w, y)
    return chi2(aves, Y)
end

"""
    neg_log_posterior(theta, S, c2)

BioEn negative log-posterior.

    neg_log_posterior(theta_series::Vector{T}, S::Vector{T}, chi2::Vector{T}) where T<:AbstractFloat

Evaluation for vectors. 

    neg_log_posterior(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, Y::Vector{T}) where T<:AbstractFloat

Calculating chi2 and S. Use pre-calculated averages for efficiency. 

Useful for optimization, wher averages have to be pre-calculated for the objective function
and its gradient. 

"""
neg_log_posterior(theta, S, c2) = theta*S + c2/2.

function neg_log_posterior(theta_series::Vector{T}, S::Vector{T}, chi2::Vector{T}) where T<:AbstractFloat
    n_thetas = size(theta_series, 1)
    values = zeros(T, n_thetas)
    for i = range(1, n_thetas)
        values[i] = neg_log_posterior(theta_series[i], S[i], chi2[i])
    end
    return values
end

function neg_log_posterior(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, Y::Vector{T}) where T<:AbstractFloat
    S = SKL(w, w0)
    c2 = chi2(aves, Y)
    return neg_log_posterior(theta, S, c2)
end

function neg_log_posterior(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, logw0::Vector{T}, Y::Vector{T}) where T<:AbstractFloat
    S = SKL(w, w0, logw0)
    c2 = chi2(aves, Y)
    return neg_log_posterior(theta, S, c2)
end

"""
    neg_log_posterior!(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, y::Array{T,2}, Y::Vector{T}) where T<:AbstractFloat

Calculate averages in-place.
"""
function neg_log_posterior!(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, y::Array{T,2}, Y::Vector{T}) where T<:AbstractFloat
    S = SKL(w, w0)
    c2 = chi2!(aves, w, y, Y)
    return neg_log_posterior(theta, S, c2)
end

"""
    get_S_opt(ws::Array{T,2}, w0::Vector{T}) where T<:AbstractFloat

Relative entropy from 2d array of weights.
"""
function get_S_opt(ws::Array{T,2}, w0::Vector{T}) where T<:AbstractFloat
    n_theta = size(ws, 2)
    S_opt = zeros(T, n_theta)
    for i in 1:n_theta
        S_opt[i] = SKL(ws[:,i], w0)
    end
    return S_opt
end

"""
    get_chi2_opt(ws::Array{T,2}, y::Array{T,2}, Y::Vector{T}) where T<:AbstractFloat

Chi-squared from 2d array of weights for use with normalized observables.
"""
function get_chi2_opt(ws::Array{T,2}, y::Array{T,2}, Y::Vector{T}) where T<:AbstractFloat 
    n_theta = size(ws, 2)
    chi2_opt = zeros(T, n_theta)
    aves = zeros(T, size(Y, 1))
    for i in 1:n_theta
        chi2_opt[i] = chi2!(aves, ws[:,i], y, Y)
    end
    return chi2_opt
end

"""
    sizes(theta_series, y)

Auxiliary function to get sizes n_thetas, N, M.
"""
function sizes(theta_series, y)
    n_thetas = size(theta_series, 1)
    N = size(y,1)
    M = size(y,2)
    return n_thetas, N, M
end

end # end module 
