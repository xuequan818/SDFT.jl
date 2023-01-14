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
L = 5  # height of the simulation box
kgrid = [1, 1, 1]
# kgrid = [6, 6, 1]
Ecut = 10
temperature = 1e-3 #1e-2

## Define the geometry and pseudopotential
a = 4.66  # lattice constant
a1 = a * [1 / 2, -sqrt(3) / 2, 0]
a2 = a * [1 / 2, sqrt(3) / 2, 0]
a3 = L * [0, 0, 1]
lattice = [a1 a2 a3]
C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
C2 = -C1
positions = [C1, C2]
# We can shoose lda or pbe as the psp data 
C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms = [C, C]

model = model_PBE(lattice, atoms, positions; temperature)
# remove the Entropy, since we don't solving eigenvalues, Entropy needs eigenvalues as input.
filter!(x -> x != Entropy(), model.term_types)

basis = PlaneWaveBasis(model; Ecut, kgrid)


"""scf: SimpleMixing + Anderson"""

"""solving eigenpaire"""
scfres = self_consistent_field(basis);
ρ = scfres.ρ;

"""Chebyshev polynomial method"""
rc = rhoCheb(800)
scfres_ChebP = self_consistent_field_sdft(basis, rc);
ρc = scfres_ChebP.ρ;
norm(ρc - ρ)

"""Stochastic DFT"""
rs = rhoStoc(800, 400)
scfres_sdft = self_consistent_field_sdft(basis, rs; maxiter=20);
ρs = scfres_sdft.ρ;
norm(ρc - ρs)

"""scf: SimpleMixing"""
scfres_simplemixing = self_consistent_field(basis; solver=scf_damping_solver())
ρ = scfres.ρ;

rc = rhoCheb(800)
scfres_ChebyP_simplemixing = self_consistent_field_sdft(basis, rc; solver=scf_damping_solver());
ρc = scfres_ChebyP_simplemixing.ρ;
norm(ρc - ρ)


rs = rhoStoc(800, 200)
scfres_sdft_simplemixing = self_consistent_field_sdft(basis, rs; solver=scf_damping_solver(), maxiter=20);
ρc = scfres_sdft_simplemixing.ρ;
norm(ρc - ρs)
