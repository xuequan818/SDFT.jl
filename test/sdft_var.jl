using SDFT
using DFTK
using LinearAlgebra
using Dates
using JLD2

function sdft_var_theory(basis::PlaneWaveBasis, εF::Real;
                         scfres_ref=nothing, Ecut_ref=30, kws...)
    smearf = FermiDirac(εF, inv(basis.model.temperature))

    if !isnothing(scfres_ref) && scfres_ref.basis == basis
        λs = scfres_ref.eigenvalues[1]
        ψs = scfres_ref.ψ[1]
        return sdft_var_eigs(smearf, λs, ψs)
    end

    if isnothing(scfres_ref)
        basis_ref = PlaneWaveBasis(basis, Ecut_ref)
        scfres_ref = self_consistent_field(basis_ref)
    else
        basis_ref = scfres_ref.basis
    end

    idcsk_in, idcsk_out = DFTK.transfer_mapping(basis_ref, basis_ref.kpoints[1],
        basis, basis.kpoints[1])
    ψguess = [scfres_ref.ψ[1][idcsk_in, :]]
    ρ = DFTK.transfer_density(scfres_ref.ρ, basis_ref, basis)
    ham = Hamiltonian(basis; ρ)

    n_bands = length(scfres_ref.eigenvalues[1])
    eigres = DFTK.diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands; ψguess)
    sdft_var_eigs(smearf, eigres.λ[1], eigres.X[1])
end

sdft_var_theory(scfres) = sdft_var_theory(scfres.basis, scfres.εF; scfres_ref=scfres)

function sdft_var_eigs(smearf::FermiDirac, λs::AbstractVector,
                       ψs::AbstractMatrix{T}) where T
    f(x) = SDFT.evalf(x, smearf)

    dof = size(ψs, 1)
    fH = zeros(T, dof, dof)
    @views for (i, iλ) in enumerate(λs)
        SDFT.outersum!(fH, ψs[:, i], f(iλ))
    end
    D = 2 * real(diag(fH))

    return sum(D)^2 - sum(D .^ 2)
end

function run_var(Nmax::Int; case_setup="graphene", 
                 Ns=500, Ecut=15, 
                 temperature=1e-3, 
                 M=5000, tol_cheb=1e-3,
                 cal_way=:cal_mat,
                 save_file=false)
    Var = Float64[]
    VarT = Float64[]
    Ne = Int[]
    count = 1
    for ni = 1:Nmax
        for nj in 1:ni
            fun = eval(Symbol(case_setup, "_setup"))
            basis = fun([ni, nj]; Ecut, temperature)
            push!(Ne, basis.model.n_electrons)
            dof = length(basis.kpoints[1].mapping)
            println(" SIZE = ($ni, $nj),  DOF = $(dof) \n")
            scfres = self_consistent_field(basis)
            ρ = scfres.ρ
            εF = scfres.εF

            @time var = estimate_var(basis, εF, MC(Ns); M, tol_cheb, ρ, cal_way)[1]
            @time varT = sdft_var_theory(scfres)

            push!(Var, var)
            push!(VarT, varT)
            count += 1
        end
    end
    
    function save_output(outdir)
        date_str = Dates.format(now(), "yyyymmdd_HH_MM")
        output_file = joinpath(outdir, "variance_$(case_setup)_$(date_str).jld2")
        jldsave(output_file; Ns, Ecut, temperature, Ne, Var, VarT)
    end

    if save_file
        try
            outdir = joinpath(@__DIR__, "..", "data")
            save_output(outdir)
        catch
            save_output(@__DIR__)
        end
    else
        return (; Ne, Var, VarT)
    end
end
