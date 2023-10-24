"""
Given a ham Single K point [0,0,0], we compare three different methods generate the density ρ: 
1. eigenpaire
2. Chebyshev polynomial method
3. Stochastic DFT method
"""

using DFTK
using Unitful
using UnitfulAtomic
using LinearAlgebra
using Plots


## Define the convergence parameters (these should be increased in production)
L = 10  # height of the simulation box
kgrid = [1, 1, 1]
Ecut = 10
temperature = 5e-3

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

## Run SCF
model = model_PBE(lattice, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis)
ham = Hamiltonian(basis; ρ=scfres.ρ)
npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
println("dof : $(npw)")

# solving eigenpaires, obtain ρ
eigensolver = lobpcg_hyper
eigres = diagonalize_all_kblocks(eigensolver, ham, 30; ψguess=nothing) # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
ρout = compute_density(ham.basis, eigres.X, occupation);

M = 2000
""" ρout = rhoCheb """
rc = rhoCheb(M)
@time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc);
print("$(rc.M)-order Cheb error : $(norm(ρout - ρout_ChebP)) ")

#=
Profile.clear()
Profile.init(delay=1e-4)
@profile rhoGen(ham, model, εF, rc);
Profile.print()
=#

""" ρout = rhoStoc """
rs = rhoStoc(M, 200)
@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
print("$(rs.M)-order $(rs.Ns)-stoc error : $(norm(ρout_ChebP - ρout_sdft)) ")

""" ρout = rhoTStoc """
nlv = [1, 0.5, 0.1]
tM = M .* nlv
tNs = [100, 400, 300]
β = @. 1 / temperature * nlv
rs = rhoTStoc(length(tM), β, tM, tNs)
@time ψ, occupation, ρout_tsdft = rhoGen(ham, model, εF, rs);
print("$(rs.nL)-level tstoc error : $(norm(ρout_ChebP - ρout_tsdft)) ")

""" ρout = rhoCutStoc """
cut = [Ecut, Ecut - 2., Ecut - 5.]
Ns = [200, 100, 100]
rcut = rhoCutStoc(M, cut, Ns, model; kgrid);
@time ψ, occupation, ρout_cutsdft = rhoGen(ham, model, εF, rcut);
print("$(rcut.nL)-level cutstoc error : $(norm(ρout_ChebP - ρout_cutsdft)) ")

#norm(ρout - ρout_ChebP) #/sqrt(basis.dvol) L2 
#norm(ρout_ChebP - ρout_sdft) #/sqrt(basis.dvol) 
plot(ρout[:, 16, 1, 1])
plot!(ρout_ChebP[:, 16, 1, 1])
plot!(ρout_sdft[:, 16, 1, 1])
plot!(ρout_tsdft[:, 16, 1, 1])



