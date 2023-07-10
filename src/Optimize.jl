module Forces

import ..Util, Optim

function forces!(f, aves, Y, sigmas, theta)
    for i in eachindex(f)
        f[i] = (aves[i]-Y[i])/(theta*sigmas[i])
    end
    return 
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
end

function weights_from_forces!(w, w0, f, y)
    for alpha in eachindex(w)
        tmp = 0.
        for j in eachindex(f)
            tmp += y[alpha, j]*f[j]
        end    
        w[alpha] = w0[alpha]*exp(tmp)
    end
    w ./= sum(w)
end

function optimize!(grad, aves, s, theta, w, w0, w_init, y, Y, sigmas)
    # calculate forces
    # calculate weights from forces
    # evaluate L
    # evalutate gradient of L 
    function func(f) 
        Forces.weights_from_forces!(w, w0, f, y)
        Util.averages!(aves, w, y)
        return Forces.Util.neg_log_posterior!(aves, s, theta, w, w0, y, Y, sigmas)
    end
    function g!(grad, f)
        Forces.weights_from_forces!(w, w0, f, y)
        Util.averages!(aves, w, y)
        return Forces.grad_neg_log_posterior!(grad, aves, s, theta, w, w0, y, Y, sigmas)
    end
    results = Optim.optimize(func, g!, f_init, Optim.LBFGS())
    return results
end


end
