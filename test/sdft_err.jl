using SDFT
using DFTK
using LinearAlgebra
using Dates
using JLD2

function run_err_rho(Ns; Nmax=3, Ecut=15, 
                     temperature=1e-3, 
                     M=5000, tol_cheb=1e-5,
                     cal_way=:cal_mat,
                     save_file=false)
    Error = Vector{Float64}[]
    Ne = Int[]
    for ni in 1:Nmax
        basis = graphene_setup([ni,ni]; Ecut, temperature)
        dof = length(basis.kpoints[1].mapping)
        push!(Ne, basis.model.n_electrons)
        println(" SIZE = ($ni, $ni),  DOF = $(dof) \n")
 
        scfres = self_consistent_field(basis)
        εF = scfres.εF
        ρ = scfres.ρ

        smearf = FermiDirac(εF, inv(temperature))
        hambl = [iham.blocks[1] for iham in SDFT.sdft_hamiltonian(basis, CT(); ρ)];
        Cheb = chebyshev_info(hambl[end], smearf, M, cal_way; ρ, tol_cheb);

        err = Float64[]
        for ns in Ns
            println(" Ns = $ns")
            @time ρmc = compute_stoc_density(basis, hambl, Cheb, MC(ns); cal_way);
            push!(err, norm(ρmc - ρ) * sqrt(basis.dvol))
        end
        push!(Error, err)
    end

    if save_file
        outdir = try
            joinpath(@__DIR__, "..", "data")
        catch 
            @__DIR__
        end
        date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
        output_file = joinpath(outdir, "density_graphene_$(Nmax)n_$(date_str).jld2")
        jldsave(output_file; Ns, Ecut, temperature, Ne, Error)
    else
        return (; Ne, Ns, Error)
    end
end
