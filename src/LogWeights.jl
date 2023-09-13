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

function average(w::Vector{T}, g::Vector{T})  where T<:AbstractFloat
    ave = zero(T)
    for mu in eachindex(g) # Note that g[end]=0, where g are log-weights
        ave += w[mu]*g[mu]
    end
    return ave
end

function weights_from_logweights!(w, g)
    norm = zero(eltype(w))
    max = maximum(g)
    for mu in eachindex(g)
        w[mu] = exp(g[mu]-max)
        norm += w[mu]
    end
    w[end] = exp(-max) #1. # exp(g[end]==0)
    norm += w[end] # g[end]=log(w[end])=0.
    for mu in eachindex(w)
        w[mu] /= norm
    end 
end

function logweights_from_weights!(g, w) # for w=1/N, all log-weights are zero!
    tmp = -log(w[end])
    for mu in eachindex(g)
        g[mu] = log(w[mu])+tmp # necessary for g[N]==0; w[end]==1/norm
    end
end

function grad_neg_log_posterior!(grad, g, G, aves, theta, w, w0, y, Y)
    d_ave = average(w, G) - average(w, g) 
    for mu in eachindex(grad)
        grad[mu]=theta*(g[mu] - G[mu] + d_ave)
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
    G = zeros(N-1) # log-weights of reference
    aves = zeros(M)
    return w, g, G, aves
end 

"""
    initialize_outout(N, n_thetas)

Auxiliary function to allocate output arrays. 
"""
function allocate_output(N, n_thetas)
    gs = zeros((N-1, n_thetas)) # all optimal weights
    return ws
end 

"""
    optimize_fg!(grad, aves, theta, g, w, w0, G, y, Y, method, options)

Optimization with analytical calculation of gradient. Single function fg!() for evaluation of
objective function and its gradient. 

"""
function optimize_fg!(grad, aves, theta, g, w, w0, G0, y, Y, method, options)

    function fg!(F, G, g)
        weights_from_logweights!(w, g)
        Utils.averages!(aves, w, y) 
        if G != nothing
            grad_neg_log_posterior!(grad, g, G0, aves, theta, w, w0, y, Y) 
            G .= grad
        end
        if F != nothing
            return Utils.neg_log_posterior(aves, theta, w, w0, Y)
        end
    end
    
    results = Optim.optimize(Optim.only_fg!(fg!), g, method, options)
    
    return results
end

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
    w, g, G, aves = allocate(N, M)
    gs = allocate_output(N, n_thetas)
    results = []

    w .= w0 # initial value
    logweights_from_weights!(g, w)
    logweights_from_weights!(G, w0) # pre-computing for efficiency
    
    for (i, theta) in enumerate(theta_series)
        res = optimize_fg!(aves, theta, g, w, w0, G, y, Y, method, options)
        g .= Optim.minimizer(res) # added dot to refer to same array
        weights_from_logweights!(w, g) # weights should be updated and this line should be superfluous. Test!
        gs[:, i] .= g 
        if verbose
            print_info(i, theta)
        end
        push!(results, res)
    end
    return gs, results
end

function refined_weights(gs)
    ws = zeros(size(gs,1)+1, size(gs,2))
    for i in axes(gs, 2)
        weights_from_logweights!(@view(ws[:,i]), @view(gs[:,i])) 
    end
    return ws
end


# place holder
end # module