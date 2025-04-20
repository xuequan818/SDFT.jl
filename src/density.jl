import DFTK: transfer_density

# TODO: profile
function compute_stoc_density(basis::PlaneWaveBasis{T}, 
                              εF, ST::SDFTMethod;
                              cal_way=:cal_mat,
						  	  M=Int(5e4), kws...) where {T}
    TT = complex(T)
    nk = length(basis.kpoints)
    filled_occ = DFTK.filled_occupation(basis.model)
    occfun(A::AbstractArray) = fill(filled_occ, size(A, 2))
    smearf = FermiDirac(εF, inv(basis.model.temperature))
    haml = sdft_hamiltonian(basis, ST; kws...)

    nl = count_nl(ST)
    ψ = Matrix{Matrix{TT}}(undef, nk, 2nl-1)
    occ = Matrix{Vector{T}}(undef, nk, 2nl-1)
    #Folds.foreach(1:nk, WorkStealingEx()) do ik
	for ik = 1:nk
        # compute the Chebyshev coefficients
        Cheb = chebyshev_info(haml[end].blocks[ik], smearf, M, cal_way; kws...)
        println(" Expansion order = $(Cheb.order)")

        hamk = [iham.blocks[ik] for iham in haml]        
        ψik = compute_wavefun(hamk, cal_way, Cheb, ST)
        if isone(nl)
            ψ[ik] = ψik
            occ[ik] = occfun(ψik)
        else
            ψ[ik, :] = ψik
            occ[ik, :] = occfun.(ψik)
        end
	end

    return compute_stoc_density(basis, ψ, occ, ST)#, ψ, occ
end

function compute_stoc_density(basis::PlaneWaveBasis, ψ, occ, ST::MC)
    norb = orbital_normalize(ST)
    ρ = compute_density(basis, ψ, occ)
    @. ρ * norb
end

function compute_stoc_density(basis::PlaneWaveBasis, ψ, occ, 
                              ST::PDegreeML{N}) where {N}
    norb = orbital_normalize(ST)
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
    ρtot, ρml, dρml
end

function compute_stoc_density(basis::PlaneWaveBasis, ψ, occ,
                              ST::ECutoffML{N}) where {N}
    norb = orbital_normalize(ST)

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
    ρtot, ρml, dρml
end
