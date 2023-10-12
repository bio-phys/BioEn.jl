BioEn.jl
========

# Overview

The module provides methods [2] to solve the optimization problem underlying BioEn [1] and related ensemble refinement methods. 

Input are the calculated observables from all structures in the ensemble, the experimentally measured observables, and the reference weights. 
Output are the refined weights and the corresponding series of values of the confidence parameter $`\theta`$.

For optimization, one can either use the log-weights method or the forces method [2]. 

Note that https://github.com/bio-phys/RefinementModels.jl provides simple models for refinement with example scripts saving the synthetic data for use with BioEn.jl. 

# References

- [1] Köfinger and Hummer, J. Chem. Phys. 143 (2015) https://aip.scitation.org/doi/10.1063/1.4937786
- [2] Köfinger et al. J. Chem. Theory and Comput. 15 (2019) https://doi.org/10.1021/acs.jctc.8b01231 
