module SDFT

using Distributions
using DFTK
import DFTK: filled_occupation
import DFTK: transfer_density, transfer_mapping
using LinearAlgebra
using Arpack, KrylovKit
using FFTW
using IterTools
using Accessors
using Printf

include("basis.jl")

export SmearFunction
export Gaussian
export FermiDirac
include("smear.jl")

export chebyshev_info
include("Chebyshev.jl")

export SDFTMethod
export MC, CT
export MLMC
export PDegreeML, PDegreeCT
export ECutoffML, ECutoffCT
include("sdft_method.jl")
include("wavefun.jl")

export estimate_var
include("variance.jl")

export OptimalMLMC
export OptimalPD
export OptimalEC
export optimal_mlmc
include("mlmc.jl")

export compute_stoc_density
include("density.jl")

include("supercell.jl")

end # module SDFT
