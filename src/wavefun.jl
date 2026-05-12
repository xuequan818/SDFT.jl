function compute_stoc_wavefun(hambls, cal_way, Cheb, ST::SDFTMethod; batch_size=256)
    nk = length(hambls)
    nl = count_nl(ST)
    T = eltype(hambls[1][1])
    ψml = Vector{Vector{Matrix{T}}}(undef, nk)

    for ik = 1:nk
        ham = hambls[ik]
        Hs = [S2_ham(iham, Val(cal_way), Cheb.E1, Cheb.E2) for iham in ham]
        dofs, cols_list = get_total_cols_list(Hs, ST)
        
        ψk = [zeros(T, dofs[i], cols_list[i]) for i = 1:length(dofs)]

        for l = 1:nl
            cols_list_l = cols_list[2l-1]
            out_indices = (l == 1) ? (1:1) : (2l-2:2l-1)
            n_out = length(out_indices)

            for start_col in 1:batch_size:cols_list[2l-1]
                end_col = min(start_col + batch_size - 1, cols_list_l)
                batch_range = start_col:end_col

                ψ_buf = ntuple(idx -> @view(ψk[out_indices[idx]][:, batch_range]), n_out)

                compute_wavefun_batch!(ψ_buf, Hs, ham, Cheb, l, batch_range, ST)
            end
        end
        ψml[ik] = ψk
    end

    return ψml
end

function count_orbital_by_wf(ψ::Vector{<:AbstractArray}, ::SDFTMethod)
    N = length(ψ)
    @assert isodd(N)
    size.(ψ, 2)[1:2:N]
end

function get_total_cols_list(H, ST::MC)
    dof = size(H[1], 1)
    cols = [orbital_size(dof, ST, 1)]

    return [dof], cols
end

function get_total_cols_list(H, PD::PDegreeML{N}) where {N}
    dof = size(H[1], 1)
    coltmp = [orbital_size(dof, PD, l) for l = 1:N]
    cols = fill(coltmp[1], 2N - 1)
    for l = 2:N
        cols[2l-2] = coltmp[l]
        cols[2l-1] = coltmp[l]
    end
    dofs = fill(dof, 2N - 1)

    return dofs, cols
end

function get_total_cols_list(H, EC::ECutoffML{N}) where {N}
    doftmp = size.(H, 1)
    coltmp = [orbital_size(doftmp[l], EC, l) for l = 1:N]
    cols = fill(coltmp[1], 2N - 1)
    dofs = fill(doftmp[1], 2N - 1)
    for l = 2:N
        cols[2l-2] = coltmp[l]
        cols[2l-1] = coltmp[l]
        dofs[2l-2] = doftmp[l-1]
        dofs[2l-1] = doftmp[l]
    end

    return dofs, cols
end

function compute_wavefun_batch!(ψ, Hs, ham, Cheb, l, rng, ST::MC)
    E1, E2, coef = Cheb.E1, Cheb.E2, Cheb.coef
    X = random_orbital(eltype(Hs[1]), size(Hs[1], 1), rng, ST)
    compute_cheb_recur!(ψ[1], Hs[1], X, coef, E1, E2)
    return nothing
end

function compute_wavefun_batch!(ψ, Hs, ham, Cheb, l, rng, PD::PDegreeML)
    E1, E2, coef = Cheb.E1, Cheb.E2, Cheb.coef
    H = Hs[1]
    X = random_orbital(eltype(H), size(H, 1), rng, PD)
    Ml = PD.Ml

    if l == 1
        compute_cheb_recur!(ψ[1], H, X, coef[1:Ml[1]+1], E1, E2)
    else
        _, U0, U1, U2 = compute_cheb_recur!(ψ[1], H, X, coef[1:Ml[l-1]+1],
            			                    E1, E2, true)
        copy!(ψ[2], ψ[1])
        _compute_cheb_recur!(ψ[2], H, U0, U1, U2,
							 coef[Ml[l-1]+2:Ml[l]+1], 
							 E1, E2, false)   
    end                                  
end

function compute_wavefun_batch!(ψ, Hs, ham, Cheb, l, rng, EC::ECutoffML)
    E1, E2, coef = Cheb.E1, Cheb.E2, Cheb.coef
    T = eltype(Hs[1])

    if l == 1
        X = random_orbital(T, size(Hs[1], 1), rng, EC)
        compute_cheb_recur!(ψ[1], Hs[1], X, coef, E1, E2)
    else
        X2 = random_orbital(T, size(Hs[l], 1), rng, EC)
        X1 = transfer_blochwave_kpt(X2, ham[l].basis, ham[l].kpoint, 
									ham[l-1].basis, ham[l-1].kpoint)
        compute_cheb_recur!(ψ[1], Hs[l-1], X1, coef, E1, E2)
        compute_cheb_recur!(ψ[2], Hs[l], X2, coef, E1, E2)
    end
end

function compute_cheb_recur(H, U0, coef, E1, E2, Ureturn=false)
    TH = similar(U0)
    compute_cheb_recur!(TH, H, U0, coef, E1, E2, Ureturn)
end

function compute_cheb_recur!(TH, H, U0, coef, E1, E2, Ureturn=false)
    @. TH = coef[1] * U0
    U1 = similar(U0)
    U2 = similar(U0)

    N = size(U0, 2)
	if !iszero(N) && length(coef) > 1
		inv_E2 = inv(E2)
		S2_mul!(U1, H, U0, E1, inv_E2)
        axpy!(coef[2], U1, TH)
	end

    coef_view = view(coef, 3:lastindex(coef))
    _compute_cheb_recur!(TH, H, U0, U1, U2, coef_view, E1, E2, Ureturn)
end
    					
function _compute_cheb_recur!(TH, H, U0, U1, U2,
    					     coef, E1, E2, 
							 Ureturn::Bool) 
    if !isempty(U0)                         
        @assert size(coef, 2) == 1
        SE2 = 2 * inv(E2)

        for ic in coef
            # compute U2 = 2 * H * U1 - U0
            S2_mul!(U2, H, U1, E1, SE2)
            axpy!(-1.0, U0, U2)
            axpy!(ic, U2, TH)

            U0, U1, U2 = U1, U2, U0
        end
    end

    if Ureturn
        return TH, U0, U1, U2
    else
        return TH
    end
end
