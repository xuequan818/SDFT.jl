cd("SDFT.jl")
using Pkg
Pkg.activate(".")

using SDFT
using DFTK
using LinearAlgebra
using DelimitedFiles

#-----------------------------------------------------------------------
#initial step
model, basis = graphene_setup(2; Ecut=12.0, kgrid=[1, 1, 1])
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

M = 100
""" ρout = rhoCheb """
rc = rhoCheb(M)
@time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc);
print("$(rc.M)-order error : density  $(norm(ρout - ρout_ChebP))")

rs = rhoStoc(M, 100)
@time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs);
print("$(rs.M)-order $(rs.Ns)-stoc error : density  $(norm(ρout_ChebP - ρout_sdft))")

#-----------------------------------------------------------------------
model, basis = graphene_setup(2; Ecut=12.0, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis);
ham = Hamiltonian(basis; ρ=scfres.ρ)
npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
println("dof : $(npw)")

# solving eigenpaires, obtain ρ
eigensolver = lobpcg_hyper
@time eigres = diagonalize_all_kblocks(eigensolver, ham, Int(round(npw / 10)); ψguess=nothing) # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
ρout = compute_density(ham.basis, eigres.X, occupation);

M = 3000
""" ρout = rhoCheb """
rc = rhoCheb(M)
@time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc);
ECheb = Etot(ρout_ChebP, basis)
print("$(rc.M)-order error : density  $(norm(ρout - ρout_ChebP))")

sterrOne = []
sterrTwo = []
sterrInf = []
Ns = collect(50:50:5000)
K = 3
for ns in Ns
    rs = rhoStoc(M, ns)
    veOne = 0.0
    veTwo = 0.0
    veInf = 0.0
    for k = 1:K
        println("k-$(k),  Ns-$(ns)")
        @time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs)
        veTwo += norm(ρout_sdft - ρout_ChebP)
        veOne += norm(ρout_sdft - ρout_ChebP, 1)
        veInf += norm(ρout_sdft - ρout_ChebP, Inf)
    end
    push!(sterrTwo, veTwo / K)
    push!(sterrOne, veOne / K)
    push!(sterrInf, veInf / K)
end

writedlm("error_ns_4.txt", hcat(Ns,sterrOne, sterrTwo, sterrInf), '\t')
