abstract type SDFTMethod end

# Monte Carlo SDFT (One level)
struct MC <: SDFTMethod 
	ns::Integer
    d::Union{Distribution,Nothing}
end
MC(ns::Integer) = MC(ns::Integer, Uniform(0, 2pi))
CT() = MC(0, nothing)

count_nl(::MC) = 1

random_orbital(T, dof, M::MC) = random_orbital(T, dof, M.ns, M.d)

# Multilevel Monte Carlo SDFT 
abstract type MLMC{N} <: SDFTMethod end
include("mlmc.jl")

@noinline function throw_cannot_mlmc()
    error("Hierarchy inconsistency.")
end

count_nl(::MLMC{N}) where {N} = N

function random_orbital(T, dof, ML::MLMC{N}, l::Integer) where {N}
	@assert l <= N 
	random_orbital(T, dof, ML.nsl[l], ML.d)
end

random_orbital(T, dof, ns, d::Nothing) = Matrix{T}(I, dof, dof)

random_orbital(T, dof, ns, d::Uniform) = T.(exp.(im .* rand(d, dof, ns))) ./ sqrt(ns)
