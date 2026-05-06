function compute_stoc_density(basis::PlaneWaveBasis, 
                              εF, ST::SDFTMethod;
                              cal_way=:cal_mat, 
                              Cheb=nothing, M=Int(1e5), 
                              tol_cheb=1e-6, kws...)
    if isnothing(Cheb) 
        smearf = FermiDirac(εF, inv(basis.model.temperature))
        ham = Hamiltonian(basis; kws...)
        Cheb = chebyshev_info(ham.blocks[1], smearf, M, cal_way; tol_cheb, kws...)
    end

    compute_stoc_density(basis, Cheb, ST; cal_way, kws...)
end

function compute_stoc_density(basis::PlaneWaveBasis, 
                              Cheb::ChebInfo, 
                              ST::SDFTMethod; 
                              cal_way=:cal_mat,
                              batch_size=256,  
                              ψin=nothing, kws...)
    hambl = [iham.blocks[1] for iham in sdft_hamiltonian(basis, ST; kws...)]

    compute_stoc_density(basis, hambl, Cheb, ST; ψin, cal_way, batch_size)
end

function compute_stoc_density(basis::PlaneWaveBasis, 
                              Cheb::ChebInfo, 
                              ST::PDegreeML{N}; 
                              cal_way=:cal_mat,
                              batch_size=256, 
                              isoptML=!isnothing(ST.d),
                              kws...) where {N}
    if isoptML
        ST, var, ψin, hambl = optimal_mlmc(basis, Cheb, OptimalPD(Cheb.order, ST.nsl, ST.d); kws...)
    else
        hambl = [iham.blocks[1] for iham in sdft_hamiltonian(basis, ST; kws...)]
        ψin = nothing
    end

    compute_stoc_density(basis, hambl, Cheb, ST; ψin, cal_way, batch_size)
end

function compute_stoc_density(basis::PlaneWaveBasis, 
                              Cheb::ChebInfo, 
                              ST::ECutoffML{N}; 
                              cal_way=:cal_mat,
                              batch_size=256, 
                              isoptML=!isnothing(ST.d), 
                              kws...) where {N}
    if isoptML
        ST, var, ψin, hambl = optimal_mlmc(basis, Cheb, OptimalEC(basis.Ecut, ST.nsl, ST.d); kws...)
    else
        hambl = [iham.blocks[1] for iham in sdft_hamiltonian(basis, ST; kws...)]
        ψin = nothing
    end

    compute_stoc_density(basis, hambl, Cheb, ST; ψin, cal_way, batch_size)
end

function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              hambl, Cheb::ChebInfo, 
                              ST::SDFTMethod; 
                              ψin=nothing,
                              cal_way=:cal_mat, 
                              batch_size=256) where {T} 
    TT = complex(T)
    filled_occ = filled_occupation(basis.model)
    occfun(n::Integer) = fill(filled_occ, n)
    occfun(A::AbstractArray) = occfun(size(A, 2))

    ns_in = count_orbital_by_wf(ψin, ST)
    if !isnothing(ψin)
        new_ns = ST.nsl .- ns_in
        ST = reset_ns(ST, new_ns)
    end

    Hs = [S2_ham(iham, Val(cal_way), Cheb.E1, Cheb.E2) for iham in hambl]
    dofs, cols_list = get_total_cols_list(Hs, ST)
    num_levels = count_nl(ST)

    ρtot = zeros(T, basis.fft_size..., basis.model.n_spin_components)

    for l in 1:num_levels
        out_indices = (l == 1) ? (1:1) : (2l-2:2l-1)
        total_cols_l = cols_list[2l-1]

        Ncl = isnothing(ST.d) ? one(T) : inv(ST.nsl[l] + ns_in[l])

        if !isnothing(ψin)
            ψin_l = [ψin[i] for i in out_indices]
            occ_in_l = occfun.(ψin_l)
            accumulate_stoc_density!(ρtot, basis, ψin_l, occ_in_l, Ncl, l, ST)
        end

        max_batch = min(batch_size, total_cols_l)
        ψ_buf_full = [Matrix{TT}(undef, dofs[i], max_batch) for i in out_indices]
        occ_buf_full = [occfun(max_batch) for _ in out_indices]

        for start_col in 1:batch_size:total_cols_l
            end_col = min(start_col + batch_size - 1, total_cols_l)
            batch_range = start_col:end_col
            curr_len = length(batch_range)

            ψ_buf = [@view(buf[:, 1:curr_len]) for buf in ψ_buf_full]
            occ_buf = [@view(buf[1:curr_len]) for buf in occ_buf_full]

            compute_wavefun_batch!(ψ_buf, Hs, hambl, Cheb, l, batch_range, ST)

            accumulate_stoc_density!(ρtot, basis, ψ_buf, occ_buf, Ncl, l, ST)
        end
    end

    return ρtot
end

function accumulate_stoc_density!(ρtot, basis::PlaneWaveBasis, ψ, occ, Nc, l, ST::MC)
    ρ = compute_density_single_k(basis, ψ, occ)
    ρtot .+= Nc .* ρ

    return nothing
end

function accumulate_stoc_density!(ρtot, basis::PlaneWaveBasis, ψ, occ,
                                  Nc, l, ST::PDegreeML{N}) where {N}
    if l == 1
        ρ = compute_density_single_k(basis, ψ, occ)
        ρtot .+= Nc .* ρ
    else
        ρ_coarse = compute_density_single_k(basis, @view(ψ[1:1]), @view(occ[1:1]))
        ρ_fine = compute_density_single_k(basis, @view(ψ[2:2]), @view(occ[2:2]))
        ρtot .+= Nc .* ρ_fine .- Nc .* ρ_coarse
    end
end

function accumulate_stoc_density!(ρtot, basis::PlaneWaveBasis, ψ, occ,
                                  Nc, l, ST::ECutoffML{N}) where {N}
    basisl = ST.basisl

    if l == 1
        ρ = compute_density_single_k(basisl[1], ψ, occ)
        ρtot .+= Nc .* transfer_density(ρ, basisl[1], basis)
    else
        ρ_coarse = compute_density_single_k(basisl[l-1], @view(ψ[1:1]), @view(occ[1:1]))
        ρ_fine = compute_density_single_k(basisl[l], @view(ψ[2:2]), @view(occ[2:2]))
        ρtot .+= Nc .* transfer_density(ρ_fine, basisl[l], basis) .- 
                 Nc .* transfer_density(ρ_coarse, basisl[l-1], basis)
    end
end

function compute_density_single_k(basis::PlaneWaveBasis{T}, ψ, occ) where {T}
    # TODO: reset the basis.kweight for multiple k-points calculation
    @assert length(ψ) == 1
    if !isempty(ψ[1])
        return compute_density(basis, ψ, occ)
    else
        return zeros(T, basis.fft_size..., basis.model.n_spin_components)
    end
end
