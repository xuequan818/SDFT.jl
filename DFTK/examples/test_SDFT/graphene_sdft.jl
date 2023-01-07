"""For example singe layer Graphene 
Different three methods: 
solving eigenpaires(DFTK); 
Chebyshev polynomial method; 
Stochastic DFT method.
scf:
1. SimpleMixing + Anderson (m=10); (default stage in DFTK)
2. SimpleMixing
"""

using DFTK
using Unitful
using UnitfulAtomic
using LinearAlgebra

## Define the convergence parameters (these should be increased in production)
L = 20  # height of the simulation box
kgrid = [1, 1, 1]
# kgrid = [6, 6, 1]
Ecut = 15     
temperature = 1e-3 #1e-2

## Define the geometry and pseudopotential
a = 4.66  # lattice constant
a1 = a*[1/2,-sqrt(3)/2, 0]
a2 = a*[1/2, sqrt(3)/2, 0]
a3 = L*[0  , 0        , 1]
lattice = [a1 a2 a3]
C1 = [1/3,-1/3,0.0]  # in reduced coordinates
C2 = -C1
positions = [C1, C2]
# We can shoose lda or pbe as the psp data 
C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms = [C, C]

model = model_PBE(lattice, atoms, positions; temperature)

# run file "standard_models.jl" (remove the Entropy), i.e. the following function, since we don't solving eigenvalues, Entropy needs eigenvalues as input.
function model_atomic(lattice::AbstractMatrix,
	atoms::Vector{<:Element},
	positions::Vector{<:AbstractVector};
	extra_terms=[], kinetic_blowup=BlowupIdentity(), kwargs...)
	@assert !(:terms in keys(kwargs))
	terms = [Kinetic(; blowup = kinetic_blowup),
	AtomicLocal(),
	AtomicNonlocal(),
	Ewald(),
	PspCorrection(),
	extra_terms...]
	#=
	if :temperature in keys(kwargs) && kwargs[:temperature] != 0
	terms = [terms..., Entropy()]
	end
	=#    
	Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end
model = model_PBE(lattice, atoms, positions; temperature)

basis = PlaneWaveBasis(model; Ecut, kgrid)


"""scf: SimpleMixing + Anderson"""

"""solving eigenpaire"""
scfres = self_consistent_field(basis)

"""Chebyshev polynomial method"""
scfres_ChebyP = self_consistent_field_ChebyP(basis; M = 5000)

"""Stochastic DFT"""
scfres_sdft = self_consistent_field_sdft(basis; M = 5000, Ns = 2000)   

"""scf: SimpleMixing"""
scfres_simplemixing = self_consistent_field(basis;  solver = scf_damping_solver())

scfres_ChebyP_simplemixing = self_consistent_field_ChebyP(basis; M = 5000, solver = scf_damping_solver())

scfres_sdft_simplemixing = self_consistent_field_sdft(basis; M = 5000, Ns = 2000, solver = scf_damping_solver())  