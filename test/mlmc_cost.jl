using SDFT
using DFTK
using LinearAlgebra
using Dates
using JLD2
include("mlmc_var.jl")

function run_mlmcpd_cost(L::Int, ϵs::Vector{Float64}; save_file=false, cal_way=:cal_mat, kws...)
    var, Ql, ψ, basis, Cheb, ρ = run_mlmcpd_var(L; cal_way, kws...)
    _run_mlmcpd_cost(ϵs, var, Ql, ψ, basis, Cheb, ρ, save_file, cal_way)
end

function run_mlmcec_cost(L::Int, ϵs::Vector{Float64}; save_file=false, cal_way=:cal_mat, kws...)
    var, Ql, ψ, basis, Cheb, ρ = run_mlmcec_var(L; cal_way, kws...)
    _run_mlmcec_cost(ϵs, var, Ql, ψ, basis, Cheb, ρ, save_file, cal_way)
end

for fun in [:run_mlmcpd_cost, :run_mlmcec_cost]
    _fun = Symbol("_", fun)
    @eval function $fun(data::Dict, ϵs::Vector{Float64}; save_file=false, cal_way=:cal_mat, kws...)
        Ecut, temperature, Nmax = data["Ecut"], data["temperature"], data["Nmax"]
        Ql, var, ψ = data["Ql"], data["var"], data["ψ"]
        chebinfo, ρ = data["ChebInfo"], data["ρ"]
        basis = graphene_setup([Nmax, Nmax]; Ecut, temperature)
        Cheb = SDFT.ChebInfo(chebinfo[1], chebinfo[2], chebinfo[3], chebinfo[4])

        $_fun(ϵs, var, Ql, ψ, basis, Cheb, ρ, save_file, cal_way)
    end
end

function _run_mlmcpd_cost(ϵs::Vector{Float64}, var, Ql, ψ, basis, Cheb, ρ, save_file, cal_way) 
    err_pd = Float64[]
    err_mc = Float64[]
    cost = []
    for (i, ϵ) in enumerate(ϵs)
        println(" $i-th ϵ")
        pd_nsl = SDFT.optimal_ns(var[1], Ql, ϵ)
        PDML = PDegreeML(Ql, pd_nsl)
        @time ρpd = compute_stoc_density(basis, 0, PDML; Cheb, ρ, ψin=ψ, cal_way);
        push!(err_pd, norm(ρpd - ρ) * sqrt(basis.dvol))

        ns_mc = ceil(Int, sum(Ql .* pd_nsl) / Ql[end])
        @time ρmc = compute_stoc_density(basis, 0, MC(ns_mc); Cheb, ρ, cal_way);
        push!(err_mc, norm(ρmc - ρ) * sqrt(basis.dvol))

        push!(cost, sum(Ql .* pd_nsl))
    end

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch 
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
        output_file = joinpath(outdir, "mlmcpd_cost_graphene_$(date_str).jld2")
        jldsave(output_file; err_mc, err_pd, cost)
    else
        return (; err_mc, err_pd, cost)
    end
end

function _run_mlmcec_cost(ϵs::Vector{Float64}, var, Ql, ψ, basis, Cheb, ρ, save_file, cal_way) 
    dim = basis.model.n_dim
    fc(l) = isone(l) ? Ql[l]^(dim/2) : (Ql[l]^(dim/2) + Ql[l-1]^(dim/2))
    Cl = fc.(1:length(Ql))

    err_ec = Float64[]
    err_mc = Float64[]
    cost = []
    for (i, ϵ) in enumerate(ϵs)
        println(" $i-th ϵ")
        ec_nsl = SDFT.optimal_ns(var[1], Cl, ϵ)
        ECML = ECutoffML(basis, Ql, ec_nsl)
        @time ρec = compute_stoc_density(basis, 0, ECML; Cheb, ρ, ψin=ψ, cal_way);
        push!(err_ec, norm(ρec - ρ) * sqrt(basis.dvol))

        ic = sum(Cl .* ec_nsl)
        ns_mc = ceil(Int, ic / Ql[end]^(dim/2))
        @time ρmc = compute_stoc_density(basis, 0, MC(ns_mc); Cheb, ρ, cal_way);
        push!(err_mc, norm(ρmc - ρ) * sqrt(basis.dvol))

        push!(cost, ic)
    end

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch 
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
        output_file = joinpath(outdir, "mlmcec_cost_graphene_$(date_str).jld2")
        jldsave(output_file; err_mc, err_ec, cost)
    else
        return (; err_mc, err_ec, cost)
    end
end
