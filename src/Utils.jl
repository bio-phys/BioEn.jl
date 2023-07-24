module Utils

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
"""
function SKL(p::Vector{T}, q::Vector{T}) where T<:AbstractFloat
    S =  zero(eltype(p))
    for i in eachindex(p)
        tmp = p[i]/q[i]
        if tmp>eps(0.0) # larger than the numerical zero
            S += p[i]*log(tmp)
        end
    end
    return S
end

"""
    averages!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat

Average of calculated observables for given weights 

dim(y) = NxM, where N is the number of samples and M the number of data points. 
"""
function averages!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat
    for i in 1:size(y, 2) # observables 
        aves[i] = 0.
        for j in 1:size(y, 1)
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
    chi2(aves, Y)

Chi-squared value for given averages using normalized observabels.
"""
function chi2(aves::Vector{T}, Y::Vector{T}) where T<:AbstractFloat
    chi2 = zero(T)
    for i in eachindex(Y)
        chi2 += (aves[i]-Y[i])^2
    end
    return chi2
end

"""
    chi2!(aves, w, y, Y)

Calculate averages in-place. 
"""
function chi2!(aves::Vector{T}, w::Vector{T}, y::Array{T, 2}, Y::Vector{T}) where T<:AbstractFloat
    averages!(aves, w, y)
    return chi2(aves, Y)
end

"""
    neg_log_posterior(theta, S, c2)

BioEn negative log-posterior.

    neg_log_posterior(theta_series::Vector{<:Real}, S::Vector{<:Real}, chi2::Vector{<:Real})

Evaluation for vectors. 

    neg_log_posterior(aves, theta, w, w0, Y)

Calculating chi2 and S. Use pre-calculated averages for efficiency. 

Useful for optimization, wher averages have to be pre-calculated for the objective function
and its gradient. 

"""
neg_log_posterior(theta, S, c2) = theta*S + c2/2.

function neg_log_posterior(theta_series::Vector{T}, S::Vector{T}, chi2::Vector{T}) where T<:AbstractFloat
    n_thetas = size(theta_series, 1)
    values = zeros(T, n_thetas)
    for i in range(1, n_thetas)
        values[i] = Util.neg_log_posterior(theta_series[i], S[i], chi2[i])
    end
    return values
end

function neg_log_posterior(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, Y::Vector{T}) where T<:AbstractFloat
    S = SKL(w, w0)
    c2 = chi2(aves, Y)
    return neg_log_posterior(theta, S, c2)
end

"""
    neg_log_posterior!(aves, theta, w, w0, y, Y)

Calculate averages in-place.
"""
function neg_log_posterior!(aves::Vector{T}, theta::T, w::Vector{T}, w0::Vector{T}, y::Array{T,2}, Y::Vector{T}) where T<:AbstractFloat
    S = SKL(w, w0)
    c2 = chi2!(aves, w, y, Y)
    return neg_log_posterior(theta, S, c2)
end

"""
    get_S_opt(theta_series, ws, w0)

Relative entropy from 2d array of weights.
"""
function get_S_opt(theta_series::Vector{T}, ws::Vector{T}, w0::Vector{T}) where T<:AbstractFloat
    n_theta = size(theta_series, 1)
    S_opt = zeros(T, n_theta)
    for i in 1:n_theta
        S_opt[i] = SKL(ws[:,i], w0)
    end
    return S_opt
end

"""
Chi-squared from 2d array of weights for use with normalized observables.
"""
function get_chi2_opt(theta_series::Vector{T}, ws::Vector{T}, y::Array{T,2}, Y::Vector{T}) where T<:AbstractFloat
    n_theta = size(theta_series, 1)
    chi2_opt = zeros(T, n_theta)
    aves = zeros(T, size(Y,1))
    for i in 1:n_theta
        chi2_opt[i] = chi2!(aves, ws[:,i], y, Y)
    end
    return chi2_opt
end

end # end module 
