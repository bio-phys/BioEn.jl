module Forces

import ..Util, Optim
using Printf

function forces_from_averages!(f, aves, Y, sigmas, theta)
    for i in eachindex(f)
        f[i] = -(aves[i]-Y[i])/(theta*sigmas[i]^2)
    end
    return nothing
end

function weights_from_forces!(w, g0, f, y)
    max = 0. # to avoid overflow when calcualting exp()
    for alpha in eachindex(w)
        tmp = 0.
        for j in eachindex(f)
            tmp += y[alpha, j]*f[j]
        end    
        w[alpha] = tmp + g0[alpha]
        if w[alpha] > max 
            max = w[alpha]
        end
    end
    norm = 0. 
    for alpha in eachindex(w)
        w[alpha] = exp(w[alpha]-max)
        norm += w[alpha]
    end
    w ./= norm
    return nothing
end

function grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y, sigmas)
    for k in eachindex(grad)
        grad[k]=0.
        for alpha in eachindex(w)
            if w[alpha]>eps(0.0)
                tmp = theta*(log(w[alpha]/w0[alpha])+1)
                for i in eachindex(Y)
                    tmp += y[alpha, i]*(aves[i]-Y[i])/sigmas[i]^2
                end
                grad[k] += tmp*w[alpha]*(y[alpha, k]-aves[k])
            end
        end
    end
    return nothing
end

function weights_and_averages!(w, aves, g0, f, y)
    weights_from_forces!(w, g0, f, y)
    Util.averages!(aves, w, y)
end


"""
Numerical approximation of gradient.
"""
function optimize_grad_num!(aves, theta, f, w, w0, g0, y, Y, sigmas, method, options)
    # calculate forces
    # calculate weights from forces
    function func(f) 
        weights_and_averages!(w, aves, g0, f, y)
        return Util.neg_log_posterior(aves, theta, w, w0, y, Y, sigmas)
    end
   
    results = Optim.optimize(func, f, method, options)
    return results
end

function optimize!(grad, aves, theta, f, w, w0, g0, y, Y, sigmas, method, options)
    # calculate forces
    # calculate weights from forces
    # evaluate L and its gradient
    function func(f) 
        weights_and_averages!(w, aves, g0, f, y)
        return Util.neg_log_posterior(aves, theta, w, w0, y, Y, sigmas)
    end
    function g!(grad, f)
        weights_and_averages!(w, aves, g0, f, y)
        return grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y, sigmas)
    end
    results = Optim.optimize(func, g!, f, method, options)
    return results
end

function optimize_fg!(grad, aves, theta, f, w, w0, g0, y, Y, sigmas, method, options)

    function fg!(F, G, f)
        weights_and_averages!(w, aves, g0, f, y)
        if G != nothing
            grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y, sigmas)
            G .= grad
        end
        if F != nothing
            return Util.neg_log_posterior(aves, theta, w, w0, y, Y, sigmas)
        end
    end
    
    results = Optim.optimize(Optim.only_fg!(fg!), f, method, options)
    
    return results
end

function print_info(i, theta, M, f)
    @printf("theta[%05d] = %3.2e", i, theta)
    fmt = Printf.Format( " f = "*("%3.2e "^M) )
    Printf.format(stdout, fmt, f... )
    @printf("\n")
    return nothing
end

function optimize_series(theta_series, w0, y, Y, sigmas, method, options)
    w = copy(w0) # we start with a large theta value
    g0 = log.(w0) # for efficiency reasons

    n_thetas = size(theta_series)[1]
    N = size(y)[1]
    M = size(y)[2]
    
    aves = zeros(M)
    grad = zeros(M)
    f = zeros(M) # forces 
    ws = zeros((N, n_thetas)) # all optimal weights
    fs = zeros((M, n_thetas)) # all optimal forces
    
    results = []
    
    for (i, theta) in enumerate(theta_series)
        Util.averages!(aves, w, y)
        #Forces.forces_from_averages!(f, aves, Y, sigmas, theta)
        #res = optimize_grad_num!(aves, theta, f, w, w0, g0, y, Y, sigmas, method, options)
        res = optimize_fg!(grad, aves, theta, f, w, w0, g0, y, Y, sigmas, method, options)
        #res = optimize!(grad, aves, theta, f, w, w0, g0, y, Y, sigmas, method, options)

        f = Optim.minimizer(res)
        weights_from_forces!(w, g0, f, y)
        ws[:, i] .= w 
        fs[:, i] .= f
        print_info(i, theta, M, f)
        push!(results, res)
    end
    return ws, fs, results
end

end # module
