using Test
include("../../BioEn.jl/src/BioEn.jl")

N = 3
M = 1
y = zeros(N, M);
y[:,1] =  [0.2, 0.3, 1.1]
w0 = ones(N)/N
g0 = log.(w0)
# If calculated and measured averages are equal, then first calculating the forces from the reference weights and calculating the weights from thees forces should give the original reference weights (independent of theta)
theta = 1.
aves = zeros(M)
f = zeros(M)
w = zeros(N)
BioEn.Utils.averages!(aves, w0, y)
BioEn.Forces.forces_from_averages!(f, aves, aves, theta)
BioEn.Forces.weights_from_forces!(w, g0, f, y)

@test all((isapprox.(w, w0)).==true)
S = BioEn.Utils.SKL(w0, w0)
@test isapprox(S, 0.) 
