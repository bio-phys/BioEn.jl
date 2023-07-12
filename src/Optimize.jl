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
    max = 0. 
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
        for alpha in eachindex(w)
            dummy = w[alpha]/w0[alpha]
            if dummy>eps(0.0)
                tmp = theta*(log(dummy)+1)
            else
                tmp = theta
            end
            #println(tmp)
            for i in eachindex(Y)
                tmp += y[alpha, i]*(aves[i]-Y[i])/sigmas[i]^2
            end
            grad[k] = tmp*w[alpha]*(y[alpha, k]-aves[k])
        end
    end
    return nothing
end




function optimize!(grad, aves, s, theta, f, w, w0, g0, w_init, y, Y, sigmas, method, options)
    # calculate forces
    # calculate weights from forces
    # evaluate L and its gradient
    function func(f) 
        weights_from_forces!(w, g0, f, y)
        Util.averages!(aves, w, y)
        return Forces.Util.neg_log_posterior_2!(aves, s, theta, w, w0, y, Y, sigmas)
    end
    function g!(grad, f)
        weights_from_forces!(w, g0, f, y)
        Util.averages!(aves, w, y)
        return Forces.grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y, sigmas)
    end
    results = Optim.optimize(func, g!, f, method, options)
    return results
end

function optimize_fg!(grad, aves, s, theta, f, w, w0, g0, w_init, y, Y, sigmas, method, options)

    function fg!(F, G, x)
        weights_from_forces!(w, g0, f, y)
        Util.averages!(aves, w, y)
        if G != nothing
            Forces.grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y, sigmas)
            G = grad
        end
        if F != nothing
            return Forces.Util.neg_log_posterior_2!(aves, s, theta, w, w0, y, Y, sigmas)
        end
    end
    
    results = Optim.optimize(Optim.only_fg!(fg!), f, method, options)
    
    return results
end

function optimize_series(theta_series, N, M, w0, y, Y, sigmas, method, options)

    w_init = copy(w0)
    w = zeros(N)
    aves = zeros(M)
    grad = zeros(M)
    s = zeros(N)
    f = zeros(M)
    n_thetas = size(theta_series)[1]
    ws = zeros((N, n_thetas))
    fs = zeros((M, n_thetas))
    Util.averages!(aves, w_init, y)
    results = []
    g0 = log.(w0)
    
    for (i, theta) in enumerate(theta_series)
        
        Forces.forces_from_averages!(f, aves, Y, sigmas, theta)
        res = optimize_fg!(grad, aves, s, theta, f, w, w0, g0, w_init, y, Y, sigmas, method, options)
        #f_final = Optim.minimizer(res)
        ws[:, i] .= w 
        fs[:, i] .= f
        @printf("theta[%05d] = %3.2e", i, theta)
        fmt = Printf.Format( " f = "*("%3.2e "^M) )
        Printf.format(stdout, fmt, f... )
        @printf("\n")
        push!(results, res)
    end
    return ws, fs, results
end

end # module
