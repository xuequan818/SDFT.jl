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

pos_map(x) = x >= 0 ? x : zero(x)

# Monte Carlo SDFT (One level)
struct MC <: SDFTMethod 
	ns::Integer
    d::Union{Distribution,Nothing}
	function MC(ns::Integer, d::Union{Distribution,Nothing})
		new(pos_map(ns), d)
	end
end
MC(ns::Integer) = MC(ns::Integer, DEFAULT_DISTR)
CT() = MC(0, nothing)

count_nl(::MC) = 1

random_orbital(T, dof, M::MC) = random_orbital(T, dof, M.ns, M.d)

orbital_normalize(M::MC) = isnothing(M.d) ? one(M.ns) : inv(M.ns)

# Multilevel Monte Carlo SDFT 
abstract type MLMC{N} <: SDFTMethod end

@noinline function throw_cannot_mlmc()
    error("Hierarchy inconsistency.")
end

count_nl(::MLMC{N}) where {N} = N

function random_orbital(T, dof, ML::MLMC{N}, l::Integer) where {N}
	@assert l <= N 
	random_orbital(T, dof, ML.nsl[l], ML.d)
end
random_orbital(T, dof, ns, d::Nothing) = Matrix{T}(I, dof, dof)
random_orbital(T, dof, ns, d::Uniform) = T.(exp.(im .* rand(d, dof, ns)))

orbital_normalize(ML::MLMC) = isnothing(ML.d) ? one.(ML.nsl) : inv.(ML.nsl)

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
