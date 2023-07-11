module Forces

import ..Util, Optim

function forces_from_averages!(f, aves, Y, sigmas, theta)
    for i in eachindex(f)
        f[i] = -(aves[i]-Y[i])/(theta*sigmas[i]^2)
    end
    return nothing
end

function weights_from_forces!(w, w0, f, y)
    max = 0. 
    for alpha in eachindex(w)
        tmp = 0.
        for j in eachindex(f)
            tmp += y[alpha, j]*f[j]
        end    
        w[alpha] = tmp + log(w0[alpha])
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

function grad_neg_log_posterior!(grad, aves, s, theta, w, w0, y, Y, sigmas)
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



function optimize!(grad, aves, s, theta, f_init, w, w0, w_init, y, Y, sigmas)
    # calculate forces
    # calculate weights from forces
    # evaluate L
    # evalutate gradient of L 
    function func(f) 
        weights_from_forces!(w, w0, f, y)
        Util.averages!(aves, w, y)
        return Forces.Util.neg_log_posterior!(aves, s, theta, w, w0, y, Y, sigmas)
    end
    function g!(grad, f)
        weights_from_forces!(w, w0, f, y)
        Util.averages!(aves, w, y)
        return Forces.grad_neg_log_posterior!(grad, aves, s, theta, w, w0, y, Y, sigmas)
    end
    results = Optim.optimize(func, g!, f_init, Optim.LBFGS())
    return results
end

function optimize_series(theta_series, N, M, w0, y, Y, sigmas, options)
    method = Optim.LBFGS()
    
    w_init = copy(w0)
    w = zeros(N)
    aves = zeros(M)
    grad = zeros(M)
    s = zeros(N)
    f = zeros(M)
    f_init = zeros(M)
    
    n_thetas = size(theta_series)[1]
    ws = zeros((N, n_thetas))
    fs = zeros((M, n_thetas))
    Util.averages!(aves, w_init, y)
    results = []
    for (i, theta) in enumerate(theta_series)
        println("theta = $theta" )
        Forces.forces_from_averages!(f_init, aves, Y, sigmas, theta)
        function func(f) 
            Forces.weights_from_forces!(w, w0, f, y)
            Util.averages!(aves, w, y)
            return Forces.Util.neg_log_posterior!(aves, s, theta, w, w0, y, Y, sigmas)
        end
    
        function g!(grad, f)
            Forces.weights_from_forces!(w, w0, f, y)
            Util.averages!(aves, w, y)
            #print(w)
            return Forces.grad_neg_log_posterior!(grad, aves, s, theta, w, w0, y, Y, sigmas)
        end
        res = Optim.optimize(func, g!, f_init, method, options)
        f_final = Optim.minimizer(res)
        ws[:, i] .= w 
        fs[:, i] .= f_final
        println(f_final)
        push!(results, res)
    end
    return ws, fs, results
end

# function fg!(F,G,x)
#   # do common computations here
#   # ...
#   if G != nothing
#     # code to compute gradient here
#     # writing the result to the vector G
#   end
#   if F != nothing
#     # value = ... code to compute objective function
#     return value
#   end
# end

# Optim.optimize(Optim.only_fg!(fg!), [0., 0.], Optim.LBFGS())



end
