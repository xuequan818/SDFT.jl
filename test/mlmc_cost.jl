using SDFT
using DFTK
using LinearAlgebra
using Dates
using JLD2
include("mlmc_var.jl")

function run_mlmc_costs(ml_way::Symbol; 
                        case_setup="graphene",
                        Ls=fill(2, 1, 1),
                        repeats=[fill(1, 2)], Ecuts=[10.0],
                        temperatures=[1e-3],
                        save_file=false, 
                        Q0_pd=85, Q0_ec=10.0, kws...)
    mlmc_time = zeros(length(repeats), length(Ecuts), length(temperatures))
    mc_time = copy(mlmc_time)
    Ne_mat = Int.(mlmc_time)
    ns_mat = copy(Ne_mat)
    Ms_mat = copy(Ne_mat)
    var_mat = Array{Any,3}(undef, size(mlmc_time))
    Ql_mat = Array{Any,3}(undef, size(mlmc_time))
    for (i, N12) in enumerate(repeats)
        for (j, Ecut) in enumerate(Ecuts)
            for (k, temperature) in enumerate(temperatures)
                if ml_way == :mlmcpd
                    t_mlmc, t_mc, basis, Cheb, var, Ql = run_mlmcpd_cost(Ls[i, j, k]; case_setup, N1=N12[1], N2=N12[2], Ecut, temperature, Q0=Q0_pd, kws...)
                elseif ml_way == :mlmcec
                    t_mlmc, t_mc, basis, Cheb, var, Ql = run_mlmcec_cost(Ls[i, j, k]; case_setup, N1=N12[1], N2=N12[2], Ecut, temperature, Q0=Q0_ec, kws...)
                    #Q0_ec*prod(N12)^(-2/3)
                end
                mlmc_time[i, j, k] = t_mlmc
                mc_time[i, j, k] = t_mc
                Ne_mat[i, j, k] = take_ne(basis)
                ns_mat[i, j, k] = take_dof(basis)
                Ms_mat[i, j, k] = Cheb.order
                var_mat[i, j, k] = var
                Ql_mat[i, j, k] = Ql
            end
        end
    end

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM")
        output_file = joinpath(outdir, "$(ml_way)_cost_$(case_setup)_$(date_str).jld2")
        jldsave(output_file; Ecuts, temperatures, mlmc_time, mc_time, Ne_mat, ns_mat, Ms_mat, var_mat, Ql_mat)
    else
        return (; mlmc_time, mc_time, Ne_mat, ns_mat, Ms_mat, var_mat, Ql_mat)
    end
end

for mlmcfun in [:run_mlmcpd, :run_mlmcec]
    fun_cost = Symbol(mlmcfun, "_cost")
    _fun_cost = Symbol("_", fun_cost)
    @eval function $fun_cost(L::Int; ϵ=0.5, cal_way=:cal_mat, kws...)
        mlmc_time, var, Ql, basis, Cheb, ρ = $_fun_cost(L; ϵ, cal_way, kws...)

        ns_mc = Int(cld(var[2][end], ϵ^2))
        @show ns_mc

        mc_time = ns_mc * Cheb.order * take_dof(basis) / basis.model.n_electrons
        #=
        mc_start_time = time()
        ρmc = compute_stoc_density(basis, 0, MC(ns_mc); Cheb, ρ, cal_way);
        mc_end_time = time()
        @show mc_time = mc_end_time - mc_start_time
        =#

        return mlmc_time, mc_time, basis, Cheb, var, Ql
    end
end

function _run_mlmcpd_cost(L; ϵ=0.5, cal_way=:cal_mat, kws...)
    var, Ql, ψ, basis, Cheb, ρ = run_mlmcpd_var(L; cal_way, kws...)

    pd_nsl = SDFT.optimal_ns(var[1], Ql, ϵ)
    @show pd_nsl

    pd_time = sum(pd_nsl .* Ql) * take_dof(basis) / basis.model.n_electrons
    #=
    PDML = PDegreeML(Ql, pd_nsl)
    pd_start_time = time()
    ρpd = compute_stoc_density(basis, 0, PDML; Cheb, ρ, ψin=ψ, cal_way);
    pd_end_time = time()
    @show pd_time = pd_end_time - pd_start_time
    =#

    return pd_time, var, Ql, basis, Cheb, ρ
end

function _run_mlmcec_cost(L; ϵ=0.5, cal_way=:cal_mat, kws...)
    var, Ql, ψ, basis, Cheb, ρ = run_mlmcec_var(L; cal_way, kws...)

    dim = basis.model.n_dim
    fc(l) = isone(l) ? Ql[l]^(dim / 2) : (Ql[l]^(dim / 2) + Ql[l-1]^(dim / 2))
    Cl = fc.(1:length(Ql))
    ec_nsl = SDFT.optimal_ns(var[1], Cl, ϵ)
    @show ec_nsl
    ECML = ECutoffML(basis, Ql, ec_nsl)

    fc2(l) = isone(l) ? take_dof(ECML.basisl[l]) : (take_dof(ECML.basisl[l]) + take_dof(ECML.basisl[l-1]))
    ec_time = sum(fc2.(1:length(Ql)) .* ec_nsl) * Cheb.order / basis.model.n_electrons
    #=
    ec_start_time = time()
    ρec = compute_stoc_density(basis, 0, ECML; Cheb, ρ, ψin=ψ, cal_way);    
    ec_end_time = time()
    @show ec_time = ec_end_time - ec_start_time
    =#

    return ec_time, var, Ql, basis, Cheb, ρ
end

take_dof(basis) = length(basis.kpoints[1].mapping)
take_ne(basis) = basis.model.n_electrons