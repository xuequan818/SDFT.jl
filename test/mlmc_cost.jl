using SDFT
using DFTK
using LinearAlgebra
using Dates
using JLD2
include("mlmc_var.jl")

function run_mlmc_costs(ml_way::Symbol; Ls=fill(2,1,1), 
                        N1s=[1], N2s=[1], 
                        Ecuts=[10.0], temperatures=[1e-3], 
                        save_file=false, kws...)
    mlmc_time = zeros(length(N1s)*length(N2s), length(Ecuts), length(temperatures))
    mc_time = copy(mlmc_time)
    Ne = Int.(mlmc_time)
    ns = copy(Ne)
    Ms = copy(Ne)
    i = 1
    for N1 in N1s, N2 in N2s
        for (j, Ecut) in enumerate(Ecuts)
            for (k, temperature) in enumerate(temperatures)
                if ml_way == :mlmcpd
                    t_mlmc, t_mc, basis, Cheb = run_mlmcpd_cost(Ls[i, j, k]; N1, N2, Ecut, temperature, kws...)
                elseif ml_way == :mlmcec
                    t_mlmc, t_mc, basis, Cheb = run_mlmcec_cost(Ls[i, j, k]; N1, N2, Ecut, temperature, kws...)
                end
                mlmc_time[i,j,k] = t_mlmc
                mc_time[i,j,k] = t_mc
                Ne[i,j,k] = take_ne(basis)
                ns[i,j,k] = take_dof(basis)
                Ms[i,j,k] = Cheb.order
            end
        end
        i += 1
    end

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch 
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
        output_file = joinpath(outdir, "$(ml_way)_cost_graphene_$(date_str).jld2")
        jldsave(output_file; mlmc_time, mc_time, Ne, ns, Ms)
    else
        return (; mlmc_time, mc_time, Ne, ns, Ms)
    end
end

for mlmcfun in [:run_mlmcpd, :run_mlmcec]
    fun_cost = Symbol(mlmcfun, "_cost")
    _fun_cost = Symbol("_", fun_cost)
    @eval function $fun_cost(L::Int; ϵ=0.5, cal_way=:cal_mat, kws...)
        mlmc_time, var, basis, Cheb, ρ = $_fun_cost(L; ϵ, cal_way, kws...)

        ns_mc = Int(cld(var[2][end], ϵ^2))
        @show ns_mc
        mc_start_time = time()
        ρmc = compute_stoc_density(basis, 0, MC(ns_mc); Cheb, ρ, cal_way);
        mc_end_time = time()
        @show mc_time = mc_end_time - mc_start_time

        return mlmc_time, mc_time, basis, Cheb
    end
end

function _run_mlmcpd_cost(L; ϵ=0.5, cal_way=:cal_mat, kws...)
    var, Ql, ψ, basis, Cheb, ρ = run_mlmcpd_var(L; cal_way, kws...);

    pd_nsl = SDFT.optimal_ns(var[1], Ql, ϵ)
    @show pd_nsl
    PDML = PDegreeML(Ql, pd_nsl)
    pd_start_time = time()
    ρpd = compute_stoc_density(basis, 0, PDML; Cheb, ρ, ψin=ψ, cal_way);
    pd_end_time = time()
    @show pd_time = pd_end_time - pd_start_time
    
    return pd_time, var, basis, Cheb, ρ
end

function _run_mlmcec_cost(L; ϵ=0.5, cal_way=:cal_mat, kws...)
    var, Ql, ψ, basis, Cheb, ρ = run_mlmcec_var(L; cal_way, kws...);

    dim = basis.model.n_dim
    fc(l) = isone(l) ? Ql[l]^(dim/2) : (Ql[l]^(dim/2) + Ql[l-1]^(dim/2))
    Cl = fc.(1:length(Ql))
    ec_nsl = SDFT.optimal_ns(var[1], Cl, ϵ)
    @show ec_nsl
    ECML = ECutoffML(basis, Ql, ec_nsl)
    ec_start_time = time()
    ρec = compute_stoc_density(basis, 0, ECML; Cheb, ρ, ψin=ψ, cal_way);    
    ec_end_time = time()
    @show ec_time = ec_end_time - ec_start_time

    return ec_time, var, basis, Cheb, ρ
end

take_dof(basis) = length(basis.kpoints[1].mapping)
take_ne(basis) = basis.model.n_electrons