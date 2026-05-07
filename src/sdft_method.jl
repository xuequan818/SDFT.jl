const DEFAULT_DISTR = Uniform(0, 2pi)

abstract type SDFTMethod end

function sdft_hamiltonian(basis::PlaneWaveBasis, ST::SDFTMethod; kws...)
    nl = count_nl(ST)
    if ST isa ECutoffML
        basisl = ST.basisl
        if :ρ in keys(kws)
            ρ = get(kws, :ρ, nothing)
            haml = map(1:nl) do i
                ρl = transfer_density(ρ, basis, basisl[i])
                Hamiltonian(basisl[i]; kws..., ρ=ρl)
            end
        else
            haml = [Hamiltonian(ibasis; kws...) for ibasis in basisl]
        end
    else
        haml = [Hamiltonian(basis; kws...)]
    end
    return haml
end

function all_level_ham_blocks(basis, ST; kws...)
    hams = sdft_hamiltonian(basis, ST; kws...)
    hambls = [Vector{HamiltonianBlock}(undef, length(hams)) for _ = 1:length(basis.kpoints)]
    for (il, haml) in enumerate(hams)
        for (ik, hambk) in enumerate(haml.blocks)
            hambls[ik][il] = hambk
        end
    end
    return hambls
end

pos_map(x) = x >= 0 ? x : zero(x)

function random_orbital(T, dof, rng::UnitRange{Int64}, d::Nothing)
    X = zeros(T, dof, length(rng))
    for (i, ir) in enumerate(rng)
        X[ir,i] = one(T)
    end
    return X
end
random_orbital(T, dof, rng::UnitRange{Int64}, d::Uniform) = T.(exp.(im .* rand(d, dof, length(rng))))
random_orbital(T, dof, rng::UnitRange{Int64}, ST::SDFTMethod) = random_orbital(T, dof, rng, ST.d)

#random_orbital(T, dof, M::MC) = isnothing(M.d) ? Matrix{T}(I, dof, dof) : T.(exp.(im .* rand(M.d, dof, M.ns)))

orbital_size(dof, ST::SDFTMethod, l::Integer) = isnothing(ST.d) ? dof : ST.nsl[l]

function reset_ns(ST::SDFTMethod, new_ns::Union{T,Vector{T}}) where {T<:Integer}
    ST_new = @set ST.nsl = tuple(new_ns...)
end

# Monte Carlo SDFT (One level)
struct MC <: SDFTMethod 
    nsl::NTuple{1,Integer}
    d::Union{Distribution,Nothing}
    function MC(nsl::NTuple{1,Integer}, d::Union{Distribution,Nothing})
		new(pos_map.(nsl), d)
	end
end
MC(nsl; d=DEFAULT_DISTR) = MC(tuple(nsl...), d)
CT() = MC(0; d=nothing)

count_nl(::MC) = 1

# Multilevel Monte Carlo SDFT 
abstract type MLMC{N} <: SDFTMethod end

@noinline function throw_cannot_mlmc()
    error("Hierarchy inconsistency.")
end

count_nl(::MLMC{N}) where {N} = N

# Polynomial degree multilevel
struct PDegreeML{N} <: MLMC{N}
    Ml::NTuple{N,Integer}
    nsl::NTuple{N,Integer}
    d::Union{Distribution,Nothing}
    function PDegreeML(Ml::NTuple{N1,Integer}, nsl::NTuple{N2,Integer},
        			   d::Union{Distribution,Nothing}) where {N1,N2}
        @assert issorted(Ml)
        N1 == N2 || throw_cannot_mlmc()
        new{N1}(Ml, pos_map.(nsl), d)
    end
end
PDegreeML(Ml, nsl, d::Union{Distribution,Nothing}) = PDegreeML(tuple(Ml...), tuple(nsl...), d)
PDegreeML(Ml, nsl; d=DEFAULT_DISTR) = PDegreeML(Ml, nsl, d)
PDegreeML(nsl) = PDegreeML(zero(nsl), nsl)
PDegreeCT(Ml) = PDegreeML(Ml, zero.(Ml); d=nothing)

# Energy cutoff multilevel
struct ECutoffML{N} <: MLMC{N}
    basisl::Vector{<:PlaneWaveBasis}
    nsl::NTuple{N,Integer}
    d::Union{Distribution,Nothing}
    function ECutoffML(basisl::Vector{<:PlaneWaveBasis}, 
                       nsl::NTuple{N,Integer},
                       d::Union{Distribution,Nothing}) where {N}
        length(basisl) == N || throw_cannot_mlmc()
        new{N}(basisl, pos_map.(nsl), d)
    end
end
function ECutoffML(basis::PlaneWaveBasis, Ecl, nsl,
                   d::Union{Distribution,Nothing})
    @assert issorted(Ecl)
	basisl = [PlaneWaveBasis(basis, ec) for ec in Ecl]
    ECutoffML(basisl, tuple(nsl...), d)
end
ECutoffML(basis, Ecl, nsl; d=DEFAULT_DISTR) = ECutoffML(basis, Ecl, nsl, d)
ECutoffML(basis, nsl) = ECutoffML(basis, zero(nsl), nsl)
ECutoffCT(basis, Ecl) = ECutoffML(basis, Ecl, Int.(zero.(Ecl)); d=nothing)
