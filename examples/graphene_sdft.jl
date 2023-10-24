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
kgrid = [6, 6, 1]
Ecut = 15
temperature = 1e-3

## Define the geometry and pseudopotential
a = 4.66  # lattice constant
a1 = a * [1 / 2, -sqrt(3) / 2, 0]
a2 = a * [1 / 2, sqrt(3) / 2, 0]
a3 = L * [0, 0, 1]
lattice = [a1 a2 a3]
C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
C2 = -C1
positions = [C1, C2]
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
dof = length(G_vectors(scfres.ham.basis, scfres.ham.basis.kpoints[1]))

"""Chebyshev polynomial method"""
rc = rhoCheb(2000)
@time scfres_ChebP, ρf = self_consistent_field_sdft(basis, rc; maxiter=20, damping=[0.8, 0.0, 1.0]);
ρc = scfres_ChebP.ρ;
norm(ρc - ρ)
@. norm(ρf - [ρ])

"""Stochastic DFT"""
mix = [0.8, 0.0, 1.0]
tM = [2000, 500]
tNs = [50, 400]
β = @. 1 / temperature * [1, 0.1]
rs = rhoTStoc(length(tM), β, tM, tNs)
@time scfres_sdft, ρf = self_consistent_field_sdft(basis, rs; maxiter=20, damping=mix);
ρs = scfres_sdft.ρ;
norm(ρc - ρs)
res = @. norm(ρf - [ρc])
cols = collect(palette(:tab10))
P = plot(collect(1:length(res)), res, xlabel="step", ylabel="error", label=L"\alpha= 0.2", yscale=:log10, size=(500, 400), grids=:off, box = :on, lw=2, color = cols[1])
plot!(P, collect(1:length(res)), res2, label=L"\alpha= %$(mix)", lw=2., color = cols[3])
plot!(P, collect(1:length(res)), res, label=L"{\rm diminishing} ~\alpha", lw=2, ls=:dash, color =cols[2])
savefig("fixed.pdf")

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
