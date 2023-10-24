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
Ecut = 12
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
scfres = self_consistent_field(basis);
ham = Hamiltonian(basis; ρ=scfres.ρ)
npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
println("dof : $(npw)")

# solving eigenpaires, obtain ρ
eigensolver = lobpcg_hyper
@time eigres = diagonalize_all_kblocks(eigensolver, ham, 30; ψguess=nothing) # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
ρout = compute_density(ham.basis, eigres.X, occupation);
E = Etot(ρout, basis)

M = 2000
""" ρout = rhoCheb """
rc = rhoCheb(M)
@time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc);
ECheb = Etot(ρout_ChebP, basis)
print("$(rc.M)-order error : density  $(norm(ρout - ρout_ChebP)), energy  $(abs(ECheb-E))")

rs = rhoStoc(M, 300)
@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
Esdft = Etot(ρout_sdft, basis)
print("$(rs.M)-order $(rs.Ns)-stoc error : density  $(norm(ρout_ChebP - ρout_sdft)), energy  $(abs(ECheb-Esdft)) ")


""" ρout = rhoStoc """
sterrInf = []
sterrTwo = []
sterrE = []
Et = []
Ns = collect(50:50:501)
K = 8
for ns in Ns
	rs = rhoStoc(M, ns)
	veInf = 0.
	veTwo = 0.
	veE = 0.
	Ev = 0.
	for k = 1:K
		println("k-$(k),  Ns-$(ns)")
		@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
        Esdft = Etot(ρout_sdft, basis)
        veTwo += norm(ρout_sdft - ρout_ChebP)^2
        veInf += norm(ρout_sdft - ρout_ChebP, Inf)^2
		veE += (Esdft-ECheb)^2
		Ev += Esdft
	end
    push!(sterrTwo, sqrt(veTwo / K))
    push!(sterrInf, sqrt(veInf / K))
    push!(sterrE, sqrt(veE / K))
    push!(Et, (Ev / K))
end
P = plot(Ns, sterrInf, scale = :log10, ylabel=L"\sigma[\rho]", xlabel=L"N_{\rm s}", guidefontsize=22, st=:scatter, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(800, 600), titlefontsize=30, margin=3mm, marker=:circle, markersize=6, markercolor=:white, markerstrokecolor=:black)
c = 0.65
plot!(P, Ns,c.* Ns.^(-1/2),color=:red,lw=1.5,label=L"%$c N_s^{-0.5}")

ll = 1:10
P2 = plot(Ns[ll], Et[ll], yerr=sterrE[ll]./10, msc=1, ylabel=L"E_{tot}", xlabel=L"N_{\rm s}", guidefontsize=22, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(800, 600), titlefontsize=30, margin=3mm, lw = 1.5, st=:scatter)
