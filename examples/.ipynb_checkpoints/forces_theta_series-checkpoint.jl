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
nr_theta_values  = 1101
base = 10

powers, theta_series = BioEn.Utils.theta_series(power_max, power_min, nr_theta_values, base=base);

options = Optim.Options(iterations=10000, g_reltol = 1e-8, store_trace = false, extended_trace = false)
method = Optim.LBFGS(alphaguess = LineSearches.InitialHagerZhang(α0=1.), linesearch = LineSearches.HagerZhang())


fs, res = BioEn.Forces.optimize_series([theta_series[1]], w0, y, Y, method, options);


ws = BioEn.Forces.refined_weights(fs, log.(w0), y);

BioEn.Output.write_HDF5(opath*"forces_results.hdf5", ws=ws, thetas=theta_series)