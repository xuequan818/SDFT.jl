function compute_stoc_density(basis::PlaneWaveBasis, 
                              εF, ST::SDFTMethod;
                              cal_way=:cal_mat, 
                              Cheb=nothing, M=Int(1e5), 
                              tol_cheb=1e-6, kws...)
    if isnothing(Cheb) 
        smearf = FermiDirac(εF, inv(basis.model.temperature))
        if length(basis.kpoints) == 1
            ham = Hamiltonian(basis; kws...)
            Cheb = chebyshev_info(ham.blocks[1], smearf, M, cal_way; tol_cheb, kws...)
        else
            Cheb = chebyshev_info(basis, smearf, M, cal_way; tol_cheb, kws...)
        end
    end

    compute_stoc_density(basis, Cheb, ST; cal_way, kws...)
end

function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              Cheb::ChebInfo, 
                              ST::SDFTMethod; 
                              cal_way=:cal_mat,
                              batch_size=256,  
                              occupation_threshold=zero(T),
                              ψin=nothing, kws...) where {T}
    hambls = all_level_ham_blocks(basis, ST; kws...)
    compute_stoc_density(basis, hambls, Cheb, ST; ψin, cal_way, batch_size, occupation_threshold)
end

function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              Cheb::ChebInfo, 
                              ST::PDegreeML{N}; 
                              cal_way=:cal_mat,
                              batch_size=256, 
                              occupation_threshold=zero(T),
                              isoptML=!isnothing(ST.d),
                              kws...) where {T,N}
    if isoptML
        t0 = time()

        ST, p_opt, var, ψin, hambls = optimal_mlmc(basis, Cheb, OptimalPD(Cheb.order, ST.nsl, ST.d); cal_way, batch_size, kws...)
        elapsed = round(time() - t0; digits=1)

        if N > 1
            println("  Building Optimal PDML for L=$(N-1) with q=$(p_opt) in $(elapsed)s.\n")
            println("  Level Information:\n")
            println("  Polynomial degrees: $(ST.Ml)")
        end
        println("  Orbital numbers:    $(ST.nsl)\n")
        flush(stdout)
    else
        hambls = all_level_ham_blocks(basis, ST; kws...)
        ψin = nothing
    end

    compute_stoc_density(basis, hambls, Cheb, ST; ψin, cal_way, batch_size, occupation_threshold)
end

function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              Cheb::ChebInfo, 
                              ST::ECutoffML{N}; 
                              cal_way=:cal_mat,
                              batch_size=256, 
                              occupation_threshold=zero(T),
                              isoptML=!isnothing(ST.d), 
                              kws...) where {T,N}
    if isoptML
        t0 = time()

        ST, p_opt, var, ψin, hambls = optimal_mlmc(basis, Cheb, OptimalEC(basis.Ecut, ST.nsl, ST.d); cal_way, batch_size, kws...)
        elapsed = round(time() - t0; digits=1)

        if N > 1
            println("  Building Optimal ECML for L=$(N-1) with p=$(p_opt) in $(elapsed)s.\n")
            println("  Level Information:\n")
            Ecl = tuple(round.(take_cut.(ST.basisl),digits=2)...)
            println("  Energy cutoffs:  $(Ecl)\n")
        end
        println("  Orbital numbers:    $(ST.nsl)\n")
        flush(stdout)
    else
        hambls = all_level_ham_blocks(basis, ST; kws...)
        ψin = nothing
    end

    compute_stoc_density(basis, hambls, Cheb, ST; ψin, cal_way, batch_size, occupation_threshold)
end

function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              hambls, Cheb::ChebInfo, 
                              ST::SDFTMethod; ψin=nothing,
                              cal_way=:cal_mat, batch_size=256,
                              occupation_threshold=zero(T)) where {T}
    TT = complex(T)
    filled_occ = filled_occupation(basis.model)
    occfun(n::Integer) = fill(filled_occ, n)

    nl = count_nl(ST)
    ns_in = isnothing(ψin) ? fill(0, nl) : count_orbital_by_wf(ψin[1], ST)
    if !isnothing(ψin)
        new_ns = ST.nsl .- ns_in
        ST = reset_ns(ST, new_ns)
    end
    Nc = [isnothing(ST.d) ? one(T) : inv(ST.nsl[l] + ns_in[l]) for l = 1:nl]
    
    max_dof = maximum(x->length(x.mapping), basis.kpoints)
    is_ecut = ST isa ECutoffML
    basis_list = is_ecut ? ST.basisl : [basis]

    nk = length(basis.kpoints)

    storage = (;
           ρ_acc = [zeros_like(G_vectors(b), T, b.fft_size..., 
                               b.model.n_spin_components) for b in basis_list],
           ψ_buf_full = [Matrix{TT}(undef, max_dof, batch_size) for _ in 1:2],
           occ_buf_full = [occfun(batch_size) for _ in 1:2]
        )

    for ik = 1:nk
        ham = hambls[ik]
        Hs = [S2_ham(iham, Val(cal_way), Cheb.E1, Cheb.E2) for iham in ham]
        dofs, cols_list = get_total_cols_list(Hs, ST)
        for l = 1:nl
            Ncl = Nc[l]
            cols_list_l = cols_list[2l-1]
            out_indices = (l == 1) ? (1:1) : (2l-2:2l-1)
            n_out = length(out_indices)

            if !isnothing(ψin)
                ψin_l = [ψin[ik][i] for i in out_indices]
                occ_in_l = occfun.(size.(ψin_l, 2))
                accumulate_stoc_density!(storage, basis, ψin_l, occ_in_l, Ncl, ik, l, ST; occupation_threshold)
            end

            for start_col in 1:batch_size:cols_list[2l-1]
                end_col = min(start_col + batch_size - 1, cols_list_l)
                batch_range = start_col:end_col
                curr_len = length(batch_range)

                ψ_buf = ntuple(idx -> @view(storage.ψ_buf_full[idx][1:dofs[out_indices[idx]], 1:curr_len]), n_out)
                occ_buf = ntuple(idx -> @view(storage.occ_buf_full[idx][1:curr_len]), n_out)

                compute_wavefun_batch!(ψ_buf, Hs, ham, Cheb, l, batch_range, ST)

                accumulate_stoc_density!(storage, basis, ψ_buf, occ_buf, Ncl, ik, l, ST; occupation_threshold)
            end
        end
    end

    ρtot = zeros_like(G_vectors(basis), T, basis.fft_size..., basis.model.n_spin_components)
    for il in eachindex(basis_list)
        ρ_acc_il = storage.ρ_acc[il]

        if is_ecut && il < nl
            ρtot .+= transfer_density(ρ_acc_il, basis_list[il], basis)
        else
            ρtot .+= ρ_acc_il
        end
    end

    DFTK.mpi_sum!(ρtot, basis.comm_kpts)
    ρtot = DFTK.symmetrize_ρ(basis, ρtot; do_lowpass=false)

    # There can always be small negative densities, e.g. due to numerical fluctuations
    # in a vacuum region, so put some tolerance even if occupation_threshold == 0
    negtol = max(sqrt(eps(T)), 10occupation_threshold)
    minimum(ρtot) < -negtol && @warn("Negative ρ detected", min_ρ=minimum(ρtot))

    ρtot
end

function accumulate_stoc_density!(storage, basis::PlaneWaveBasis{T}, 
                                  ψ, occ, Nc, ik, l, ST::MC; 
                                  occupation_threshold=zero(T)) where {T}
    compute_density_single_k!(storage.ρ_acc[1], basis, ψ[1], occ[1],
                              ik, Nc; occupation_threshold)
    return nothing
end

function accumulate_stoc_density!(storage, basis::PlaneWaveBasis{T},
                                  ψ, occ, Nc, ik, l, ST::PDegreeML{N}; 
                                  occupation_threshold=zero(T)) where {T,N}
    if l == 1
        compute_density_single_k!(storage.ρ_acc[1], basis, ψ[1], occ[1],
                                  ik, Nc; occupation_threshold)
    else
        compute_density_single_k!(storage.ρ_acc[1], basis, ψ[1], occ[1],
                                  ik, -Nc; occupation_threshold)
        compute_density_single_k!(storage.ρ_acc[1], basis, ψ[2], occ[2],
                                  ik, Nc; occupation_threshold)
    end
end

function accumulate_stoc_density!(storage, basis::PlaneWaveBasis{T},
                                  ψ, occ, Nc, ik, l, ST::ECutoffML{N}; 
                                  occupation_threshold=zero(T)) where {T,N}
    basisl = ST.basisl

    if l == 1
        compute_density_single_k!(storage.ρ_acc[1], basisl[1], ψ[1], occ[1],
                                  ik, Nc; occupation_threshold)
    else
        compute_density_single_k!(storage.ρ_acc[l-1], basisl[l-1], ψ[1], occ[1],
                                 ik, -Nc; occupation_threshold)
        compute_density_single_k!(storage.ρ_acc[l], basisl[l], ψ[2], occ[2],
                                  ik, Nc; occupation_threshold)
    end
end

function compute_density_single_k!(ρ, basis::PlaneWaveBasis{T}, 
                                   ψ::AbstractMatrix, 
                                   occupation::AbstractVector{<:Integer}, ik, Nc;
                                   occupation_threshold=zero(T)) where {T}
    isempty(ψ) && return ρ
                                            
    # Occupation should be on the CPU as we are going to be doing scalar indexing.
    occ_k = DFTK.to_cpu(occupation)
    mask_occ = findall(occnk -> abs(occnk) ≥ occupation_threshold, occ_k)
    kpt = basis.kpoints[ik]

    weight = basis.kweights[ik] * (basis.fft_grid.ifft_normalization)^2 * Nc

    function allocate_local_storage()
        (; ρ_loc=zeros_like(ρ),
           ψnk_real=zeros_like(G_vectors(basis), complex(T), basis.fft_size...))
    end

    storages = parallel_loop_over_range(mask_occ; allocate_local_storage) do n, storage
        ifft!(storage.ψnk_real, basis, kpt, @view(ψ[:, n]); normalize=false)
        weight_n = occ_k[n] * weight
        storage.ρ_loc[:, :, :, kpt.spin] .+= weight_n .* abs2.(storage.ψnk_real)
    end

    if !isnothing(storages)
        for st in storages
            ρ .+= st.ρ_loc
        end
    end

    return ρ
end
