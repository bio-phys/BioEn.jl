"""
The BioEn log-weights method. 

The code should be as slim as possible:
o No calculation of quantities that can be recalculated afterwards. 
o No calculation of quantities for monitoring the progress. 
o No saving of results during theta-series. 
o N>>M
"""
module LogWeights

import ..Utils, Optim
using Printf

function weights_from_logweights!(w, g)
    norm = zero(eltype(w))
    for mu in eachindex(g)
        w[mu] = exp(g[mu])
        norm += w[mu]
    end
    w[end] = 1.
    norm += 1. # g[end]=log(w[end])=0.
    for mu in eachindex(w)
        w[mu] /= norm
    end 
end

function logweights_from_weights!(g, w) 
    tmp = -log(w[end])
    for mu in eachindex(g)
        g[mu] = log(w[mu])+tmp # necessary for g[N]==0; w[end]==1/norm
    end
end

function average_log_weights(w, g) # todo: The name is too specific. The function might be overly complicated. 
    ave = zero(eltype(g))
    for mu in eachindex(g)
        ave += w[mu]*g[mu]
    end
    # Note that g[end]=0 
    return ave
end

function grad_neg_log_posterior!(grad, g, G, aves, theta, w, w0, y, Y)
    # todo: averages of log-weights should be done in single loop
    g_ave = average_log_weights(w, g) 
    G_ave = average_log_weights(w, G)
    d_ave = G_ave-g_ave
    for mu in eachindex(grad)
        grad[mu]=theta*(g[mu]-G[mu]+d_ave)
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
    G0 = zeros(N-1) # log of reference weights
    aves = zeros(M)
    return w, g, G0, aves, grad
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
# function optimize_fg!(grad, aves, theta, g, w, w0, G0, y, Y, method, options)

#     function fg!(F, G, g)
#         weights_from_logweights!(w, g)
#         Utils.averages!(aves, w, y) 
#         if G != nothing
#             grad_neg_log_posterior!(grad, g, G0, aves, theta, w, w0, y, Y) 
#             G .= grad
#         end
#         if F != nothing
#             return Utils.neg_log_posterior(aves, theta, w, w0, Y)
#         end
#     end
    
#     results = Optim.optimize(Optim.only_fg!(fg!), g, method, options)
    
#     return results
# end

function optimize_fg!(aves, theta, g, w, w0, G0, y, Y, method, options)

    function fg!(F, G, g)
        weights_from_logweights!(w, g)
        Utils.averages!(aves, w, y) 
        if G != nothing
            grad_neg_log_posterior!(G, g, G0, aves, theta, w, w0, y, Y) 
        end
        if F != nothing
            return Utils.neg_log_posterior(aves, theta, w, w0, Y)
        end
    end
    
    results = Optim.optimize(Optim.only_fg!(fg!), g, method, options)
    
    return results
end


"""
    print_info(i, theta)

Auxiliary function for printing information on optimization.
"""
function print_info(i, theta)
    @printf("theta[%05d] = %3.2e", i, theta)
    @printf("\n")
    return nothing
end


"""
    optimize_series(theta_series, w0, y, Y, method, options)

Optimization for series of theta-values (from large to small). 

We start at reference weigths, which is accurate for large theta. 
Newly optimized log-weights are used as initial conditions for next smaller value of theta. 

"""
function optimize_series(theta_series, w0, y, Y, method, options; verbose=false)
    # todo: test if theta_series is properly sorted
    n_thetas, N, M = Utils.sizes(theta_series, y)
    w, g, G0, aves = allocate(N, M)
    ws = allocate_output(N, M, n_thetas)
    results = []

    w .= w0 # initial value
    logweights_from_weights!(g, w)
    logweights_from_weights!(G0, w0) # pre-computing for efficiency
    
    for (i, theta) in enumerate(theta_series)
        res = optimize_fg!(aves, theta, g, w, w0, G0, y, Y, method, options)
        g .= Optim.minimizer(res) # added dot to refer to same array
        weights_from_logweights!(w, g) # weights should be updated and this line should be superfluous. Test!
        ws[:, i] .= w 
        if verbose
            print_info(i, theta)
        end
        push!(results, res)
    end
    return ws, results
end


# place holder
end # module