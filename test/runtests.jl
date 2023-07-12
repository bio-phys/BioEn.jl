using Test
includet("../src/Util.jl")
includet("../src/Models.jl")
includet("../src/Optimize.jl")

# Input
n=50
Y=[40.]
M = size(Y)[1]
sigmas = [3.]
x = range(1,n)

# Model
u = DoubleWell.energy(x, n)
p = DoubleWell.distribution(x, n)

# Sampling
N = 5000
sample = DoubleWell.sample(N, x, p)
y = zeros(N, M);
y[:,1] =  sample;
w0 = ones(N)/N
g0 = log.(w0)
# If calculated and measured averages are equal, then first calculating the forces from the reference weights and calculating the weights from thees forces should give the original reference weights (independent of theta)
theta = 1.
aves = zeros(M)
f = zeros(M)
w = zeros(N)
Util.averages!(aves, w0, y)
Forces.forces_from_averages!(f, aves, aves, sigmas, theta)
Forces.weights_from_forces!(w, g0, f, y)

@test all((isapprox.(w, w0)).==true)
S, s = Util.SKL(w0, w0)
@test isapprox(S, 0.) 
S = Util.SKL!(s, w0, w0)
@test isapprox(S, 0.) 