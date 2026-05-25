export compute_density_eigs, determine_n_bands_ks, lapack_partial

function compute_density_eigs(basis, εF, eigensolver, n_bands; kws...)
    eigres = diagonalize(eigensolver, basis, n_bands; kws...)
    occupation = DFTK.compute_occupation(basis, eigres.λ, εF).occupation
    
    DFTK.compute_density(basis, eigres.X, occupation)
end

function determine_n_bands_ks(eigensolver, basis::PlaneWaveBasis{T}, εF;
                              n_bands::Union{Nothing,Int}=nothing,
                              occupation_threshold=DFTK.default_occupation_threshold(T), 
                              kws...) where {T}
    nbandsalg = AdaptiveBands(basis.model; occupation_threshold)
    max_nb = minimum(ik->take_dof(basis, ik), 1:length(basis.kpoints))
    if isnothing(n_bands) || n_bands ≥ max_nb
        n_bands = max_nb
    end
    eigenvalues, ψ = diagonalize(eigensolver, basis, n_bands; kws...)
    occupation = DFTK.compute_occupation(basis, eigenvalues, εF).occupation
    n_bands_converge, _ = DFTK.determine_n_bands(nbandsalg, occupation,
                                                 eigenvalues, ψ)

    return n_bands_converge                                
end

function diagonalize(eigensolver, basis, n_bands::Int; kws...)
    ham  = Hamiltonian(basis; kws...)
    data = diagonalize_all_kblocks(eigensolver, ham, n_bands; ψguess=nothing)
    (; λ=getfield(data, :λ), X=getfield(data, :X))
end

function lapack_partial(A, X0; kws...)
    Neig = size(X0, 2)
    Afull = Hermitian(Array(A))
    E = eigen(Afull, 1:Neig)

    (; λ=E.values, X=E.vectors)
end
