function estimate_var(basis::PlaneWaveBasis,
                      εF::Real, ST::SDFTMethod;
                      cal_way=:cal_mat, M=Int(5e4),
                      tol_cheb=1e-6, kws...)
    ham = Hamiltonian(basis; kws...).blocks[1]
    smearf = FermiDirac(εF, inv(basis.model.temperature))
    Cheb = chebyshev_info(ham, smearf, M, cal_way; tol_cheb, kws...)

    estimate_var(basis, Cheb, ST; cal_way, kws...)
end

function estimate_var(basis::PlaneWaveBasis{T},
                      Cheb::ChebInfo,
                      ST::SDFTMethod;
                      cal_way=:cal_mat,
                      batch_size=256, kws...) where {T}
    occ = filled_occupation(basis.model)
    hambls = all_level_ham_blocks(basis, ST; kws...)
    ψ = compute_stoc_wavefun(hambls, cal_way, Cheb, ST; batch_size)

    nl = count_nl(ST)
    function allocate_local_storage()
        (; vars=fill(zero(T), 2, nl))
    end

    nk = length(basis.kpoints)
    storages = parallel_loop_over_range(1:nk; allocate_local_storage) do ik, storage
        estimate_var_single_k!(storage, basis, ψ, ik, ST)
    end

    vars = occ^2 .* sum(storage -> storage.vars, storages)
    DFTK.mpi_sum!(vars, basis.comm_kpts)

    return vars, ψ, hambls
end

function estimate_var_single_k!(storage, basis::PlaneWaveBasis, ψ, ik, ST::MC)
    storage.vars[1, 1] += basis.kweights[ik]^2 * variance(ψ[ik][1])
    return nothing
end

function estimate_var_single_k!(storage, basis::PlaneWaveBasis,
                                ψ, ik, ST::PDegreeML{N}) where {N}
    weight = basis.kweights[ik]^2
    @views storage.vars[:, 1] .+= (weight * variance(ψ[ik][1]))
    for l = 2:N
        storage.vars[1, l] += weight * variance(ψ[ik][2l-2], ψ[ik][2l-1])
        storage.vars[2, l] += weight * variance(ψ[ik][2l-1])
    end
end

function estimate_var_single_k!(storage, basis::PlaneWaveBasis,
                                ψ, ik, ST::ECutoffML{N}) where {N}
    basisl = ST.basisl
    weight = basis.kweights[ik]^2
    @views storage.vars[:, 1] .+= (weight * variance(ψ[ik][1]))
    for l = 2:N
        ψ_coarse = transfer_blochwave_kpt(ψ[ik][2l-2], basisl[l-1],
                                          basisl[l-1].kpoints[ik],
                                          basisl[l], basisl[l].kpoints[ik])
        storage.vars[1, l] += weight * variance(ψ_coarse, ψ[ik][2l-1])
        storage.vars[2, l] += weight * variance(ψ[ik][2l-1])
    end
end

function variance(X::Matrix{T}) where {T}
    n, N = size(X)

    S = X' * X

    norms2 = [real(S[i, i]) for i in 1:N]

    sum_val = zero(real(T))
    for a in 1:N
        for b in 1:N
            sum_val += norms2[a]^2 + norms2[b]^2 - 2 * abs2(S[a, b])
        end
    end

    return sum_val / (2 * N^2)
end

function variance(X::Matrix{T}, Y::Matrix{T}) where {T}
    @assert size(X) == size(Y)
    n, N = size(X)

    Sxx = X' * X
    Syy = Y' * Y
    Sxy = X' * Y

    term1 = zero(real(T))
    for i in 1:N
        term1 += real(Sxx[i, i])^2 + real(Syy[i, i])^2 - 2 * abs2(Sxy[i, i])
    end
    E_norm_sq = term1 / N

    term2 = zero(real(T))
    @inbounds for b in 1:N
        for a in 1:N
            term2 += abs2(Sxx[a, b]) + abs2(Syy[a, b]) - 2 * abs2(Sxy[a, b])
        end
    end
    norm_E_sq = term2 / (N^2)

    return E_norm_sq - norm_E_sq
end
