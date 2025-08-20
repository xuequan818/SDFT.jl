using SDFT
import SDFT: _optimal_mlmc
using DFTK
using LinearAlgebra
using Dates
using JLD2

function run_mlmcpd_var(L; Ns=500, Q0=500, Qc=0, Nmax=1, 
                        Ecut=20, temperature=1e-3,
                        tol_cheb=1e-5, M=Int(1e5), 
                        save_file=false, 
                        cal_way=:cal_mat)
    basis = graphene_setup([Nmax, Nmax]; Ecut, temperature)
    scfres = self_consistent_field(basis)
    εF = scfres.εF
    ρ = scfres.ρ
    ham = Hamiltonian(basis; ρ)
    @show dof = size(ham.blocks[1], 1)
    
    smearf = FermiDirac(εF, inv(temperature))
    Cheb = chebyshev_info(ham.blocks[1], smearf; tol_cheb, M);

    Ns = ceil.(Int, Ns ./ [2^i for i = 0:L])
    f30(x) = x < 30 ? 30 : x
    Ns = f30.(Ns)
    @time Ql, var, ψ, hambl0 = _optimal_mlmc(basis, Cheb, OptimalPD(Cheb.order, Ns); ρ, cal_way, Q0, Qc);

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch 
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
        output_file = joinpath(outdir, "mlmcpd_var_graphene_$(L)L_$(date_str).jld2")
        ChebInfo = [Cheb.E1, Cheb.E2, Cheb.order, Cheb.coef]
        jldsave(output_file; Ecut, temperature, Nmax, Ql, var, ψ, ChebInfo, ρ)
    else
        return (; var, Ql, ψ, basis, Cheb, ρ)
    end
end

function run_mlmcec_var(L; Ns=100, Q0=6, Qc=0.1, Nmax=1, 
                        Ecut=20, temperature=1e-3,
                        tol_cheb=1e-5, M=Int(1e5), 
                        save_file=false, 
                        cal_way=:cal_mat, slope=nothing)
    basis = graphene_setup([Nmax, Nmax]; Ecut, temperature)
    scfres = self_consistent_field(basis)
    εF = scfres.εF
    ρ = scfres.ρ
    ham = Hamiltonian(basis; ρ)
    @show dof = size(ham.blocks[1], 1)
    
    smearf = FermiDirac(εF, inv(temperature))
    Cheb = chebyshev_info(ham.blocks[1], smearf; tol_cheb, M);

    Ns = ceil.(Int, Ns ./ [2^i for i = 0:L])
    f30(x) = x < 30 ? 30 : x
    Ns = f30.(Ns)
    @time Ql, var, ψ, hambl0 = _optimal_mlmc(basis, Cheb, OptimalEC(Ecut, Ns); ρ, cal_way, Q0, Qc, slope);

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch 
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
        output_file = joinpath(outdir, "mlmcec_var_graphene_$(L)L_$(date_str).jld2")
        ChebInfo = [Cheb.E1, Cheb.E2, Cheb.order, Cheb.coef]
        jldsave(output_file; Ecut, temperature, Nmax, Ql, var, ψ, ChebInfo, ρ)
    else
        return (; var, Ql, ψ, basis, Cheb, ρ)
    end
end
