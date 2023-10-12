BioEn.jl
========

# Overview

The module provides methods [2] to solve the optimization problem underlying BioEn [1] and related ensemble refinement methods. 

Input are the calculated observables from all structures in the ensemble, the experimentally measured observables, and the reference weights. 
Output are the refined weights and the corresponding series of values of the confidence parameter $`\theta`$.

For optimization, one can either choose the log-weights method (LogWeights.jl) or the forces method (Forces.jl) [2]. We use https://github.com/JuliaNLSolvers/Optim.jl [3] for optimization. 

Note that https://github.com/bio-phys/RefinementModels.jl provides synthetic data for simple models for refinement. Example scripts save output for use with BioEn.jl. 

Note that BioEn.jl is mainly intended for method development. It is an alternative to the efficient Python/C implementation at https://github.com/bio-phys/BioEen intended for applications. 

# Usage 

To make the functions available, e.g., in a Jupyter notebook, use 

import Pkg
Pkg.activate("PATH/BioEn.jl")
import BioEn

where PATH is your path to the module directory. 

# References

- [1] Köfinger and Hummer, J. Chem. Phys. 143 (2015) https://aip.scitation.org/doi/10.1063/1.4937786
- [2] Köfinger et al. J. Chem. Theory Comput. 15 (2019) https://doi.org/10.1021/acs.jctc.8b01231
- [3] Mogensen and Riseth, J. Open Source Soft. 3 (2018) https://doi.org/10.21105/joss.00615
