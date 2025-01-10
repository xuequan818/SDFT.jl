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

model, basis = graphene_setup(1; Ecut=12.0, kgrid=[1, 1, 1])
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

M = 3000
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
sterrOne = []
sterrTwo = []
sterrInf = []
sterrE = []
Et = []
M = 3000
Ns = collect(50:50:1000)
K = 4
for ns in Ns
	rs = rhoStoc(M, ns)
    veOne = 0.0
    veTwo = 0.0
    veInf = 0.0
	for k = 1:K
		println("k-$(k),  Ns-$(ns)")
		@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
        #Esdft = Etot(ρout_sdft, basis)
        veTwo += norm(ρout_sdft - ρout_ChebP)
        veOne += norm(ρout_sdft - ρout_ChebP, 1)
        veInf += norm(ρout_sdft - ρout_ChebP, Inf)
	end
    push!(sterrTwo, veTwo / K)
    push!(sterrOne, veOne / K)
    push!(sterrInf, veInf / K)
    #push!(sterrE, veE / K)
    #push!(Et, (Ev / K))
end
#writedlm("error_sdft/ns_test/error_ns.txt", hcat(sterrOne, sterrTwo, sterrInf), '\t')

Error = readdlm("error_sdft/ns_test/error_ns.txt", '\t')

P = plot(Ns, Error[:, 3], scale=:log10, ylabel=L"\Vert\Delta\rho\Vert_\infty", xlabel=L"N_{\mathbb{S}}", guidefontsize=22, st=:scatter, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(800, 650), titlefontsize=30, margin=4mm, marker=:circle, markersize=6, markercolor=:white, markerstrokecolor=:black, ylims=(0.01, 0.1))
c = 0.6
plot!(P, Ns,c.* Ns.^(-1/2),color=:red,ls=:dash,lw=2,label="")
P
savefig(P, "error_sdft/ns_test/errorInf_ns.pdf")


ll = 1:10
P2 = plot(Ns[ll], Et[ll], yerr=sterrE[ll]./10, msc=1, ylabel=L"E_{tot}", xlabel=L"N_{\rm s}", guidefontsize=22, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(800, 600), titlefontsize=30, margin=3mm, lw = 1.5, st=:scatter)
