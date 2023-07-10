module Util

function theta_series(min, max, n; base=2.)
    powers = range(min, max, n)
    thetas = base.^powers
    return collect(powers), thetas
end
    
function SKL!(s, p, q)
    S = 0.
    for i in eachindex(p)
        tmp = p[i]/q[i]
        if tmp>eps(0.0)
            s[i] = p[i]*log(tmp)
            S += s[i]
        end
    end
    # dh = log.(p/q))
    return S
    #isapprox(1e-20, 0.0; atol=eps(Float64), rtol=0)
end

function SKL(p, q)
    s = zeros(size(p)[1])
    S = SKL!(s, p, q)
    return S, s
end
"""
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

function chi2(aves, w, y, Y, sigmas)
    chi2 = 0.
    for i in eachindex(Y)
        chi2 += ((aves[i]-Y[i])/sigmas[i])^2
    end
    return chi2
end

function chi2!(aves, w, y, Y, sigmas)
    chi2 = 0.
    averages!(aves, w, y)
    for i in eachindex(Y)
        chi2 += ((aves[i]-Y[i])/sigmas[i])^2
    end
    return chi2
end

function neg_log_posterior(theta, S, chi2)
    return theta*S + chi2/2.
end

end
