module LogWeights

"""
    allocate(N, M)

Auxiliary function to pre-allocate arrays, which are frequently updated. 
"""
function allocate(N, M)
    w = zeros(N)
    g = zeros(N-1)
    G = zeros(N-1)
    aves = zeros(M)
    grad = zeros(N-1)
    return w, aves, grad, f
end 

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

function average_log_weights(w, g)
    ave = zero(eltype(g))
    for mu in eachindex(g)
        ave += w[mu]*g[mu]
    end
    # Notet that g[end]=0 
    return ave
end

function grad_neg_log_posterior!(grad, g, G, aves, theta, w, w0, y, Y)
    g_ave = average_log_weights(w, g)
    G_ave = average_log_weights(w, G)
    for mu in eachindex(grad)
        grad[mu]=theta*(g[mu]-g_ave-G[mu]+G_ave)
        for i in eachindex(Y)
            grad[mu] += (aves[i]-Y[i])*(y[mu, i]-aves[i])
        end
        grad[mu] *= w[mu]
    end
    return nothing
end


# place holder
end # module