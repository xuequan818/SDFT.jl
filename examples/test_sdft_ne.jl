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
using SDFT
using Plots


model, basis = graphene_setup(20; Ecut=12.0, kgrid=[1, 1, 1])
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

M = 300
""" ρout = rhoCheb """
rc = rhoCheb(M)
@time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc);
print("$(rc.M)-order error : density  $(norm(ρout - ρout_ChebP))")

rs = rhoStoc(M, 300)
@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
print("$(rs.M)-order $(rs.Ns)-stoc error : density  $(norm(ρout_ChebP - ρout_sdft))")
#writedlm("4ne_49N_0.1a_0.5b_1d_error.txt", e, '\t');

""" ρout = rhoStoc """
sterrOne = []
sterrTwo = []
sterrInf = []
sterrE = []
Et = []
rep = collect(1:6)
K = 3
M = 3000
rc = rhoCheb(M)
rs = rhoStoc(M, 500)
for ne in rep
	veOne = 0.
	veTwo = 0.
    veInf = 0.0
	veE = 0.
	Ev = 0.

    model, basis = graphene_setup(ne; Ecut=12.0, kgrid=[1, 1, 1])
    scfres = self_consistent_field(basis)
    ham = Hamiltonian(basis; ρ=scfres.ρ)
    npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
    println("dof : $(npw)")
    # solving eigenpaires, obtain ρ
    @time eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, 30; ψguess=nothing) # 100 is the number of the bands (only for test)
    occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

    """ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
    ρout = compute_density(ham.basis, eigres.X, occupation)
    #E = Etot(ρout, basis)

    """ ρout = rhoCheb """
    @time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc)
    #ECheb = Etot(ρout_ChebP, basis)

	for k = 1:K
		println("k-$(k),  Ne-$(ne)")
		@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
        Esdft = Etot(ρout_sdft, basis)
        veTwo += norm(ρout_sdft - ρout_ChebP)
        veOne += norm(ρout_sdft - ρout_ChebP, 1)
        veInf += norm(ρout_sdft - ρout_ChebP, Inf)
		#veE += abs(Esdft-ECheb)
		#Ev += Esdft
	end
    push!(sterrTwo, veTwo/ K)
    push!(sterrOne, veOne / K)
    push!(sterrInf, veInf / K)
    #push!(sterrE, veE / K)
    #push!(Et, Ev / K)
end
#writedlm("error_ne.txt", hcat(sterrOne, sterrTwo, sterrInf), '\t')
using DelimitedFiles

Error = readdlm("error_sdft/ne_test/error_ne.txt", '\t')
Ne = 2*rep
P = plot(Ne, Error[:, 3], scale=:log10, ylabel=L"\Vert\Delta\rho\Vert_\infty", xlabel=L"N_{\rm C}", guidefontsize=22, st=:scatter, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(800, 650), titlefontsize=30, margin=4mm, marker=:circle, markersize=6, markercolor=:white, markerstrokecolor=:black, ylims=(0.01, 0.1), xticks=(Ne,Ne))
c = 0.033
s = 0
plot!(P, Ne, c .* Ne .^ s, ls = :dash, color=:red, lw=2., label="")
savefig(P, "error_sdft/ne_test/errorInf_ne.pdf")

#=
cols = collect(palette(:tab10))
using LaTeXStrings, Plots;
P = plot(scale=:log10, ylabel=L"\Vert\Delta\rho\Vert_k",
    xlabel=L"N_{\rm C}", guidefontsize=22, st=:scatter, tickfontsize=20, legend=(0.83, 0.86), grid=:off, box=:on, size=(800, 650), ylims=[0.01,130],titlefontsize=30, margin=4mm, fg_legend=:false, legendfontsize=18)
for (i, k, mst) in zip(1:3, [L"k=1", L"k=2", L"k=\infty"], [:circle, :rect, :utriangle])
    scatter!([minimum(2*rep)], [0], label=" ", ms=0, mc=:white, msc=:white)
    plot!(P, 2 * rep, Error[:, i], st=:scatter, markersize=8, markercolor=cols[i], markerstrokecolor = cols[i], label=k, marker=mst)
end
for (i, c, s) in zip(1:3,[8,0.3,0.03], [1.0,0.5,0.])
    plot!(P, 2 * rep, c * (2*rep) .^ (s), color=cols[i], lw=2., ls=:dash, label="")
end
P

savefig(P, "SDFT.jl/error_sdft/ne_test/error_ne.pdf")

ll = 1:10
P2 = plot(Ns[ll], Et[ll], yerr=sterrE[ll]./10, msc=1, ylabel=L"E_{tot}", xlabel=L"N_{\rm s}", guidefontsize=22, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(800, 600), titlefontsize=30, margin=3mm, lw = 1.5, st=:scatter)
=#
