"""
The BioEn forces method. 

The code should be as slim as possible:
o No calculation of quantities that can be recalculated afterwards. 
o No calculation of quantities for monitoring the progress. 
o No saving of results during theta-series. 
"""
module Forces

import ..Utils, Optim, Random
using Printf, Base.Threads 

"""
    forces_from_averages!(f, aves, Y, theta)

Normalized forces from normalized observables and their averages [eq 18 of JCTC 2019]]
"""
function forces_from_averages!(f, aves, Y, theta)
    for i in eachindex(f)
        f[i] = ( Y[i]-aves[i] )/theta # NO division by sigma^2 for "normalized" forces
    end
    return nothing
end

"""
    weights_from_forces!(w, G, f, y)

Weights from forces [eq 9 of JCTC 2019] (normalized or original quantities).

G = log w0 are the reference log-weights. 

We iterate three times over the number of weights to calculate 
(1) log-weights and their maximum from forces,
(2) non-normalized weights and their norm, and 
(3) normalized weights. 
"""
function weights_from_forces!(w, G, f, y)
    max = 0. # to avoid overflow when calcualting exp()
    for alpha in eachindex(w)
        tmp = 0.
        for j in eachindex(f)
            tmp += y[alpha, j]*f[j]
        end    
        w[alpha] = tmp + G[alpha]
        if w[alpha] > max  # an issue for paralleliztion 
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

"""
    grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y)

Gradient of the negative log-posterior [eq 20 of JCTC 2019] for normalized forces and observables

The averages are pre-calculated for efficiency reasons. The gradient 'grad' is updated. 
"""
function grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y)
    for k in eachindex(grad)
        grad[k]=0.
        for alpha in eachindex(w)
            if w[alpha]>eps(0.0) # might be inefficient as we do different things for different entries
                tmp = theta*(log(w[alpha]/w0[alpha])+1) # we could use pre-calculated entropy contributions of each structure 
                for i in eachindex(Y)
                    tmp += y[alpha, i]*(aves[i]-Y[i])
                end
                grad[k] += tmp*w[alpha]*(y[alpha, k]-aves[k])
            end
        end
    end
    return nothing
end

"""
    weights_and_averages!(w, aves, G0, f, y)

Weights and averages from forces. 

Auxiliarz function for optimization, where we calculate weights and averages from forces
at each iteration.
"""
function weights_and_averages!(w, aves, G0, f, y)
    weights_from_forces!(w, G0, f, y)
    Utils.averages!(aves, w, y)
end

"""
    optimize_grad_num!(aves, theta, f, w, w0, G0, y, Y, method, options)

Optimization with numerical approximation of gradient for testing. 
"""
function optimize_grad_num!(aves, theta, f, w, w0, G0, y, Y, method, options)
    # calculate forces
    # calculate weights from forces
    function func(f) 
        weights_and_averages!(w, aves, G0, f, y)
        return Utils.neg_log_posterior(aves, theta, w, w0, Y)
    end
   
    results = Optim.optimize(func, f, method, options)
    return results
end

"""
    optimize_fg!(grad, aves, theta, f, w, w0, G0, y, Y, method, options)

Optimization with analytical calculation of gradient. Single function fg!() for evaluation of
objective function and its gradient. 

"""
function optimize_fg!(grad, aves, theta, f, w, w0, G0, y, Y, method, options)

    function fg!(F, G, f)
        weights_and_averages!(w, aves, G0, f, y)
        if G != nothing
            grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y)
            G .= grad
        end
        if F != nothing
            return Utils.neg_log_posterior(aves, theta, w, w0, Y)
        end
    end
    
    results = Optim.optimize(Optim.only_fg!(fg!), f, method, options)
    
    return results
end

"""
    print_info(i, theta, M, f)

Auxiliary function for printing information on optimization.
"""
function print_info(i, theta, M, f)
    @printf("theta[%05d] = %3.2e", i, theta)
    fmt = Printf.Format( " f = "*("%3.2e "^M) )
    Printf.format(stdout, fmt, f... )
    @printf("\n")
    return nothing
end

"""
    allocate(N, M)

Auxiliary function to pre-allocate arrays, which are frequently updated. 
"""
function allocate(N, M)
    w = zeros(N)
    aves = zeros(M)
    grad = zeros(M)
    f = zeros(M) # forces 
    return w, aves, grad, f
end 

"""
    initialize_outout(N, n_thetas)

Auxiliary function to allocate output arrays. 
"""
function allocate_output(N, M, n_thetas)
    ws = zeros((N, n_thetas)) # all optimal weights
    fs = zeros((M, n_thetas)) # all optimal forces
    return ws, fs 
end 

"""
    optimize_series(theta_series, w0, y, Y, method, options)

Optimization for series of theta-values (from large to small). 

We start at reference weigths and forces that are zero, which is accurate for large theta. 
Newly optimized forces are used as initial conditions for next smaller value of theta. 
"""
function optimize_series(theta_series, w0, y, Y, method, options)
    # todo: test if theta_series is properly sorted
    n_thetas, N, M = Utils.sizes(theta_series, y)
    w, aves, grad, f = allocate(N, M)
    ws, fs = allocate_output(N, M, n_thetas)
    results = []

    w .= w0 # initial value
    G0 = log.(w0) # pre-computing for efficiency
    
    for (i, theta) in enumerate(theta_series)
        #Utils.averages!(aves, w, y)
        #Forces.forces_from_averages!(f, aves, Y, theta)
        #res = optimize_grad_num!(aves, theta, f, w, w0, G0, y, Y, method, options)
        res = optimize_fg!(grad, aves, theta, f, w, w0, G0, y, Y, method, options)
        #res = optimize!(grad, aves, theta, f, w, w0, G0, y, Y, method, options)
        f .= Optim.minimizer(res) # added dot to refer to same array
        weights_from_forces!(w, G0, f, y) # weights should be updated and this line should be superfluous
        ws[:, i] .= w 
        fs[:, i] .= f
        print_info(i, theta, M, f)
        push!(results, res)
    end
    return ws, fs, results
end

"""
    optimize(theta_series, w0, y, Y, method, options, fs_init)

Reoptimize based on previous results 'fs_init' for forces of corresponding theta values.

We use @threads for speed-up of computation. 

"""
function optimize(theta_series, w0, y, Y, method, options, fs_init) # should be named differently; perhaps move loop outside?
    n_thetas, N, M = Utils.sizes(theta_series, y)
    ws, fs = allocate_output(N, M, n_thetas)

    results = Vector{Any}(undef, n_thetas)
    G0 = log.(w0)   # pre-computing for efficiency
    @threads for i in Random.shuffle(range(1, n_thetas)) # shuffling the indices is one way to better spread workload on threads
        theta = theta_series[i]
        w, aves, grad, f = allocate(N, M) # each thread is independent (aves, w, grad, f)
        f .= fs_init[i]
        res = optimize_fg!(grad, aves, theta, f, w, w0, G0, y, Y, method, options)
        f .= Optim.minimizer(res) 
        weights_from_forces!(w, G0, f, y) # Just to make sure. Weights should be updated and this line should be superfluous.
        ws[:, i] .= w 
        fs[:, i] .= f
        results[i] = deepcopy(res) 
    end
    return ws, fs, results # todo: force should not be stored and returned. They can be recalculated from weights.
end

end # module




# function optimize!(grad, aves, theta, f, w, w0, G0, y, Y, method, options)
#     # calculate forces
#     # calculate weights from forces
#     # evaluate L and its gradient
#     function func(f) 
#         weights_and_averages!(w, aves, G0, f, y)
#         return Utils.neg_log_posterior(aves, theta, w, w0, y, Y)
#     end
#     function g!(grad, f)
#         weights_and_averages!(w, aves, G0, f, y)
#         return grad_neg_log_posterior!(grad, aves, theta, w, w0, y, Y)
#     end
#     results = Optim.optimize(func, g!, f, method, options)
#     return results
# end
