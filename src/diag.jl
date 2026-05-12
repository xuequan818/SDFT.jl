export compute_density_eigs, determine_n_bands_ks

function compute_density_eigs(basis, εF; n_bands=false, kws...)
    eigres = SDFT.diagonalize(basis; eigensolver=lobpcg_hyper, n_bands, kws...)
    occupation = DFTK.compute_occupation(basis, eigres.λ, εF).occupation
    
    DFTK.compute_density(basis, eigres.X, occupation)
end

function determine_n_bands_ks(basis::PlaneWaveBasis{T}, εF; 
                              eigensolver=diag_full, n_bands=false, 
                              occupation_threshold=DFTK.default_occupation_threshold(T), kws...) where {T}
    nbandsalg = AdaptiveBands(basis.model; occupation_threshold)
    eigenvalues, ψ = diagonalize(basis; eigensolver, n_bands, kws...)
    occupation = DFTK.compute_occupation(basis, eigenvalues, εF).occupation
    n_bands_converge, _ = DFTK.determine_n_bands(nbandsalg, occupation,
                                                 eigenvalues, ψ)

    return n_bands_converge                                
end

function diagonalize(basis; eigensolver=diag_full, n_bands=false, kws...)
    ham  = Hamiltonian(basis; kws...)
    data = if n_bands isa Int
        diagonalize_all_kblocks(eigensolver, ham, n_bands)
    else
        _diagonalize(ham)
    end
    (; λ=getfield(data, :λ), X=getfield(data, :X))
end

# Get all eigensolutions.
function _diagonalize(ham::Hamiltonian)
    kpoints = ham.basis.kpoints
    results = Vector{Any}(undef, length(kpoints))

    for ik in eachindex(kpoints)
        Afull = Hermitian(Array(ham[ik]))
        E = eigen(Afull)
        X = E.vectors
        λ = E.values
        results[ik] = (; λ, X)
    end
    (; λ=[real.(result.λ) for result in results],
       X=[result.X for result in results])
end
