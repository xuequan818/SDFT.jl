"""
Given a ham Single K point [0,0,0], we compare three different methods generate the density ρ: 
1. eigenpaire
2. Cheby polynomial method
3. Stochastic method
"""

using DFTK
using Unitful
using UnitfulAtomic
using LinearAlgebra

## Define the convergence parameters (these should be increased in production)
L = 20  # height of the simulation box
kgrid = [1, 1, 1]
#kgrid = [6, 6, 1]
Ecut = 15     
temperature = 1e-3 

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

# run file "standard_models.jl" (remove the Entropy),i.e. the following function, since we don't solving eigenvalues, Entropy needs eigenvalues as input.
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

# Given ρin, generat Ham, 
ρin = guess_density(basis)
ψ = nothing
occupation = nothing
eigenvalues = nothing
εF = nothing
n_iter = 0
energies = nothing
ham = nothing

energies, ham = energy_hamiltonian(basis, ψ, occupation;
ρ = ρin, eigenvalues, εF)

# solving eigenpaires, obtain ρout
eigensolver=lobpcg_hyper
eigres = diagonalize_all_kblocks(eigensolver, ham, 100;
ψguess=ψ) # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
ρout = compute_density(ham.basis, eigres.X, occupation)

""" ρout = rhoGenChebyP """
ψ, occupation, ρout_ChebP = rhoGenChebP(ham, model, εF, 2000)

""" ρout = rhoGenStoc """
ψ, occupation, ρout_sdft = rhoGenStoc(ham, model, εF, 2000, 500)

norm(ρout - ρout_ChebP) #/sqrt(basis.dvol) L2
norm(ρout_ChebP - ρout_sdft) #/sqrt(basis.dvol) 

using Plots
plot(ρout[:,16,1,1])
plot!(ρout_ChebP[:,16,1,1])
plot!(ρout_sdft[:,16,1,1])


