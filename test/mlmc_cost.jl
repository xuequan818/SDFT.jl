using SDFT
import SDFT: estimate_fermi, _optimal_mlmc, optimal_ns
using DFTK
using LinearAlgebra
using JLD2
include("testcase.jl")

function run_mc_cost(Ecut, temperature, repeat;
                     sys="graphene",Ns=200, 
                     Ecut_fermi=min(15, Ecut),
                     tot_tol=0.1, kws...)
    fun = eval(Symbol(sys, "_setup"))
    basis = fun(repeat; Ecut, temperature)
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    smearf = FermiDirac(εF, inv(temperature))
    ham = Hamiltonian(basis; ρ)
    Cheb = chebyshev_info(ham.blocks[1], smearf; ρ, kws...)

    t0 = time()

    ST, p_opt, vars, _, _ = optimal_mlmc(basis, Cheb, OptimalPD(Cheb.order, [Ns]); tot_tol, ρ, kws...)

    elapsed = round(time() - t0; digits=1)

    println("  Building Optimal MC in $(elapsed)s.\n")
    println("  Orbital numbers: $(ST.nsl[1])\n")
    flush(stdout)

    M = Cheb.order
    dof = take_dof(basis)
    nsl = ST.nsl[1]
    cost = nsl * M * xlog(dof)
    cost = BigFloat(cost)
    Ne = basis.model.n_electrons

    return (; var=vars[1], M, nsl, dof, cost, Ne)
end

function run_mlmcpd_cost(L, Ecut, temperature, repeat;
                         sys="graphene",
                         Ns=200, M=Int(1e5),
                         Ecut_fermi=min(15, Ecut),
                         tot_tol=0.1, kws...)
    fun = eval(Symbol(sys, "_setup"))
    basis = fun(repeat; Ecut, temperature)
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    smearf = FermiDirac(εF, inv(temperature))
    ham = Hamiltonian(basis; ρ)
    Cheb = chebyshev_info(ham.blocks[1], smearf; ρ, kws...)

    nsl0 = ceil.(Int, Ns ./ [2^i for i = 0:L])
    nsl0 = [i <= 20 ? 20 : i for i in nsl0]
    
    t0 = time()

    ST, p_opt, vars, _, _ = optimal_mlmc(basis, Cheb, OptimalPD(Cheb.order, nsl0); tot_tol, ρ, kws...)

    elapsed = round(time() - t0; digits=1)

    if L > 0
        println("  Building Optimal PDML for L=$L with q=$(p_opt) in $(elapsed)s.\n")
        println("  Level Information:\n")
        println("  Polynomial degrees: $(ST.Ml)")
    end
    println("  Orbital numbers:    $(ST.nsl)\n")
    flush(stdout)

    Ql = ST.Ml
    nsl = ST.nsl
    dof = take_dof(basis)
    cost = sum(nsl .* Ql) * xlog(dof)
    cost = BigFloat(cost)
    Ne = basis.model.n_electrons

    return (; p_opt, vars, Ql, nsl, dof, cost, Ne)
end

function run_mlmcec_cost(L, Ecut, temperature, repeat;
                         sys="graphene",
                         Ns=200, M=Int(1e5),
                         Ecut_fermi=min(15, Ecut),
                         tot_tol=0.1, kws...)
    fun = eval(Symbol(sys, "_setup"))
    basis = fun(repeat; Ecut, temperature)
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    smearf = FermiDirac(εF, inv(temperature))
    ham = Hamiltonian(basis; ρ)
    Cheb = chebyshev_info(ham.blocks[1], smearf; ρ, kws...)

    nsl0 = ceil.(Int, Ns ./ [2^i for i = 0:L])
    nsl0 = [i <= 20 ? 20 : i for i in nsl0]

    t0 = time()

    ST, p_opt, vars, _, _ = optimal_mlmc(basis, Cheb, OptimalEC(Ecut, nsl0); ρ, tot_tol, kws...)

    elapsed = round(time() - t0; digits=1)

    if L > 0
        println("  Building Optimal ECML for L=$L with p=$(p_opt) in $(elapsed)s.\n")
        println("  Level Information:\n")
        Ecl = tuple(round.(take_cut.(ST.basisl), digits=2)...)
        println("  Energy cutoffs:  $(Ecl)\n")
    end
    println("  Orbital numbers:    $(ST.nsl)\n")
    flush(stdout)

    nsl = ST.nsl
    Ql = take_cut.(ST.basisl)
    dofl = take_dof.(ST.basisl)
    M = Cheb.order
    fc2(l) = isone(l) ? xlog(dofl[l]) : (xlog(dofl[l]) + xlog(dofl[l-1]))
    cost = sum(fc2.(1:length(Ql)) .* nsl) * M
    cost = BigFloat(cost)
    Ne = basis.model.n_electrons

    return (; p_opt, vars, nsl, Ql, dofl, M, cost, Ne)
end

xlog(x) = x*log(x)
