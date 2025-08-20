# TODO: profile
function compute_stoc_density(basis::PlaneWaveBasis, 
                              εF, ST::SDFTMethod;
                              cal_way=:cal_mat, 
                              Cheb=nothing,
                              M=Int(1e5), kws...)
    hambl = [iham.blocks[1] for iham in sdft_hamiltonian(basis, ST; kws...)]
    if isnothing(Cheb) 
        smearf = FermiDirac(εF, inv(basis.model.temperature))
        Cheb = chebyshev_info(hambl[end], smearf, M, cal_way; kws...)
    end

    compute_stoc_density(basis, hambl, Cheb, ST; cal_way, kws...)
end

function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              hambl, Cheb::ChebInfo, 
                              ST::SDFTMethod; ψin=nothing,
                              cal_way=:cal_mat, kws...) where {T} 
    @assert length(basis.kpoints) == 1

    TT = complex(T)
    filled_occ = filled_occupation(basis.model)
    occfun(A::AbstractArray) = fill(filled_occ, size(A, 2))

    function reset_ns(ST::SDFTMethod, new_ns)
        if ST isa MC
            ST = @set ST.ns = new_ns
        else
            ST = @set ST.nsl = tuple(new_ns...)
        end
        ST
    end

    if !isnothing(ψin)
        ns_eval = ST.nsl .- count_orbital_by_wf(ψin, ST)
        ST = reset_ns(ST, ns_eval)
    end
    
    nl = count_nl(ST)
    ψ = compute_wavefun(hambl, cal_way, Cheb, ST)
    if !isnothing(ψin)
        for i = 1:2nl-1
            ψ[i] = hcat(ψin[i],ψ[i])
        end
    end

    if !isnothing(ψin)
        ST = reset_ns(ST, count_orbital_by_wf(ψ, ST))
    end

    norb = orbital_normalize(ST)
    ψ = reshape(ψ, 1, 2nl-1)
    occ = occfun.(ψ)

    return compute_stoc_density(basis, ψ, occ, norb, ST)#, ψ, occ
end

function compute_stoc_density(basis::PlaneWaveBasis, ψ, occ, norb, ST::MC)
    ρ = compute_density(basis, ψ, occ)
    @. ρ * norb
end

function compute_stoc_density(basis::PlaneWaveBasis, ψ, occ, norb,
                              ST::PDegreeML{N}) where {N}
    ρtot = norb[1] .* compute_density(basis, ψ[:, 1], occ[:, 1]) 
    ρml = Vector{AbstractArray}(undef, N)
    dρml = Vector{AbstractArray}(undef, N-1)
    for l = 2:N
        ρ1 = norb[l] .* compute_density(basis, ψ[:, 2l-2], occ[:, 2l-2])
        ρ2 = norb[l] .* compute_density(basis, ψ[:, 2l-1], occ[:, 2l-1])
        Δρ = ρ2 - ρ1
        ρtot += Δρ
        if l == 2
            ρml[l-1] = ρ1
        end
        ρml[l] = ρ2
        dρml[l-1] = Δρ
    end
    ρtot#, ρml, dρml
end

function compute_stoc_density(basis::PlaneWaveBasis, ψ, occ, norb,
                              ST::ECutoffML{N}) where {N}
    basisl = ST.basisl
    ρtot = transfer_density(norb[1] .* compute_density(basisl[1], ψ[:, 1], occ[:, 1]), basisl[1], basis)
    ρml = Vector{AbstractArray}(undef, N)
    dρml = Vector{AbstractArray}(undef, N - 1)
    for l = 2:N
        ρ1 = norb[l] .* compute_density(basisl[l-1], ψ[:, 2l-2], occ[:, 2l-2])
        ρ2 = norb[l] .* compute_density(basisl[l], ψ[:, 2l-1], occ[:, 2l-1])
        Δρ = transfer_density(ρ2, basisl[l], basis) - transfer_density(ρ1, basisl[l-1], basis)
        ρtot += Δρ
        if l == 2
            ρml[l-1] = ρ1
        end
        ρml[l] = ρ2
        dρml[l-1] = Δρ
    end
    ρtot#, ρml, dρml
end
