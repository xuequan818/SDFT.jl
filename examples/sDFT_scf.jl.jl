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
using Plots, Plots.PlotMeasures, LaTeXStrings


## Define the convergence parameters (these should be increased in production)
L = 5  # height of the simulation box
kgrid = [1, 1, 1]
# kgrid = [6, 6, 1]
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
@show dof = length(G_vectors(scfres.ham.basis, scfres.ham.basis.kpoints[1]))

"""Chebyshev polynomial method"""
M = 3000
rc = rhoCheb(M)
mix = [0.8, 0.0, 1.0] # damping = mix[1] / (mix[2]* niter + mix[3])
@time scfres_ChebP, ρf = self_consistent_field_sdft(basis, rc; maxiter=20, damping=mix);
ρc = scfres_ChebP.ρ;
@show norm(ρc - ρ)

"""Stochastic DFT"""
Ns = 300
rs = rhoStoc(M, Ns)
mix = [[0.8, 0.0, 1.0], [0.2, 0.0, 1.0], [1., 0.25, 1.0]]
cols = collect(palette(:tab10))
plname = [L"\alpha= %$(mix[1][1])", L"\alpha= %$(mix[2][1])", L"{\rm diminishing} ~\alpha"]
P = plot(xlabel="step", ylabel="error", grids=:off, box=:on, guidefontsize=22, tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, size=(740, 620), right_margin=3mm, top_margin=3mm)
for i = 1:length(mix)
	@time scfres_sdft, ρf = self_consistent_field_sdft(basis, rs; maxiter=20, damping=mix[i]);
	res = @. norm(ρf - [ρc])
	plot!(P, collect(1:length(res)), res, yscale =:log10, label=plname[i], lw=3.0, color=cols[i])
end
P
#savefig("mixing.pdf")