# Polynomial degree multilevel
struct PDegreeML{N} <: MLMC{N}
    Ml::NTuple{N,Integer}
    nsl::NTuple{N,Integer}
    d::Union{Distribution,Nothing}
    function PDegreeML(Ml::NTuple{N1,Integer}, nsl::NTuple{N2,Integer}, d::Union{Distribution,Nothing}) where {N1,N2}
        @assert issorted(Ml)
        N1 == N2 || throw_cannot_mlmc()
        new{N1}(Ml, nsl, d)
    end
end
PDegreeML(Ml::Vector{T}, nsl::Vector{T}, d::Union{Distribution,Nothing}) where {T<:Integer} = PDegreeML(tuple(Ml...), tuple(nsl...), d)
PDegreeML(Ml, nsl) = PDegreeML(Ml, nsl, Uniform(0, 2pi))
PDegreeCT(Ml) = PDegreeML(Ml, zero.(Ml), nothing)

# TODO: optimal PDegreeML

# Energy cutoff multilevel
struct ECutoffML{N} <: MLMC{N}
    basisl::Vector{<:PlaneWaveBasis}
    nsl::NTuple{N,Integer}
    d::Union{Distribution,Nothing}
    function ECutoffML(basisl::Vector{<:PlaneWaveBasis}, nsl::NTuple{N,Integer}, d::Union{Distribution,Nothing}) where {N}
        length(basisl) == N || throw_cannot_mlmc()
        new{N}(basisl, nsl, d)
    end
end
function ECutoffML(model::Model, Ecl::Vector{<:Real}, nsl::Vector{T}, d::Union{Distribution,Nothing}; kws...) where {T<:Integer}
    @assert issorted(Ecl)
	basisl = [PlaneWaveBasis(model; Ecut=ec, kws...) for ec in Ecl]
    ECutoffML(basisl, tuple(nsl...), d)
end
ECutoffML(model::Model, Ecl, nsl; kws...) = ECutoffML(model, Ecl, nsl, Uniform(0, 2pi); kws...)
ECutoffCT(model::Model, Ecl; kws...) = ECutoffML(model, Ecl, Int.(zero.(Ecl)), nothing; kws...)

# TODO: optimal ECutoffML
