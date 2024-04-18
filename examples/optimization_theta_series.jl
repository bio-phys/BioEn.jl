import Pkg

Pkg.activate("../../BioEn.jl")
import BioEn 

import Optim
using LineSearches

ipath = "./data/input/"
opath = "./data/output/"

Y, y, w0, N, M = BioEn.Input.read_txt_normalized(ipath; Y_name="exp.txt", y_name="obs.txt", w0_name="w0.txt")

power_min = -6
power_max = 5
base = 10
factor = 100
nr_theta_values  = (power_max-power_min)*factor+1

print("nr_theta_values=", nr_theta_values)

powers, theta_series = BioEn.Utils.theta_series(power_max, power_min, nr_theta_values, base=base);

options = Optim.Options(iterations=10000, g_reltol = 1e-8, store_trace = false, extended_trace = false)
method = Optim.LBFGS(alphaguess = LineSearches.InitialHagerZhang(α0=1.), linesearch = LineSearches.HagerZhang())

algorithm = "forces"
algorithm = "log-weights"

print("\nUsing algorithm \"$algorithm\".")

if algorithm == "forces"
    fs, res = BioEn.Forces.optimize_series(theta_series, w0, y, Y, method, options);
    ws = BioEn.Forces.refined_weights(fs, log.(w0), y);
elseif algorithm == "log-weights"
    gs, res = BioEn.LogWeights.optimize_series(theta_series, w0, y, Y, method, options; verbose=false)
    ws = BioEn.LogWeights.refined_weights(gs)
else
    print("\nAlgorithm \"$algorithm\" not available. No ouput written. Exiting.") 
end

filename = opath*algorithm*"_results.hdf5"

BioEn.Output.delete(filename)
BioEn.Output.write_HDF5(filename, ws=ws, thetas=theta_series)

