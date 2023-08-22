"""
The BioEn log-weights method. 

The code should be as slim as possible:
o No calculation of quantities that can be recalculated afterwards. 
o No calculation of quantities for monitoring the progress. 
o No saving of results during theta-series. 
"""
module LogWeights

function weights_from_logweights!(w, g)
    for mu in eachindex(g)
        w[mu] = exp(g)
    end
    w[end] = 1.
    norm = zero(eltype(w))
    for mu in eachindex(w)
        norm += w[mu]
    end
    for mu in eachindex(w)
        w[mu] /= norm
    end 
end

function logweights_from_weights!(g, w)
    for mu in eachindex(g)
        g = log(w)
    end
end

function average_log_weights(w, g) # The name is too specific. The function might be overly complicated. 
    ave = zero(eltype(g))
    for mu in eachindex(g)
        ave += w[mu]*g[mu]
    end
    # Notet that g[end]=0 
    return ave
end

function grad_neg_log_posterior!(grad, g, G, G_ave, aves, theta, w, w0, y, Y)
    g_ave = average_log_weights(w, g)
    for mu in eachindex(grad)
        grad[mu]=theta*(g[mu]-g_ave-G[mu]+G_ave)
        for i in eachindex(Y)
            grad[mu] += (aves[i]-Y[i])*(y[mu, i]-aves[i])
        end
        grad[mu] *= w[mu]
    end
    return nothing
end

"""
    allocate(N, M)

Auxiliary function to pre-allocate arrays, which are frequently updated. 
"""
function allocate(N, M)
    w = zeros(N)
    g = zeros(N-1) #log-weights
    G = zeros(N-1) # log of reference weights
    aves = zeros(M)
    grad = zeros(N-1)
    return w, g, G, aves, grad
end 

"""
    initialize_outout(N, n_thetas)

Auxiliary function to allocate output arrays. 
"""
function allocate_output(N, M, n_thetas)
    ws = zeros((N, n_thetas)) # all optimal weights
    return ws
end 


"""
    optimize_fg!(grad, aves, theta, g, w, w0, G0, G0_ave, y, Y, method, options)

Optimization with analytical calculation of gradient. Single function fg!() for evaluation of
objective function and its gradient. 

"""
function optimize_fg!(grad, aves, theta, g, w, w0, G0, G0_ave, y, Y, method, options)

    function fg!(F, G, f)
        Utils.averages!(aves, w, y)
        if G != nothing
            grad_neg_log_posterior!(grad, g, G0, G0_ave, aves, theta, w, w0, y, Y)
            G .= grad
        end
        if F != nothing
            return Utils.neg_log_posterior(aves, theta, w, w0, Y)
        end
    end
    
    results = Optim.optimize(Optim.only_fg!(fg!), g, method, options)
    
    return results
end


"""
    optimize_series(theta_series, w0, y, Y, method, options)

Optimization for series of theta-values (from large to small). 

We start at reference weigths, which is accurate for large theta. 
Newly optimized log-weights are used as initial conditions for next smaller value of theta. 

"""
function optimize_series(theta_series, w0, y, Y, method, options)
    n_thetas, N, M = sizes(theta_series, y)
    w, g, G, aves, grad = allocate(N, M)
    ws = allocate_output(N, M, n_thetas)
    results = []

    w .= w0 # initial value
    logweights_from_weights!(g, w)
    G0 = log.(w0) # pre-computing for efficiency
    G0_ave = average_log_weights(w, G0)
    
    for (i, theta) in enumerate(theta_series)
        res = optimize_fg!(grad, aves, theta, g, w, w0, G0, G0_ave, y, Y, method, options)
        g .= Optim.minimizer(res) # added dot to refer to same array
        weights_from_logweights!(w, g) # weights should be updated and this line should be superfluous. Test!
        ws[:, i] .= w 
        #print_info(i, theta, M, f)
        push!(results, res)
    end
    return ws, results
end


# place holder
end # module