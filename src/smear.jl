abstract type SmearFunction end

"""
Gaussian smearing
"""
struct Gaussian{T<:Real} <: SmearFunction
    μ::T
    σ::T
end
Gaussian(σ::Real) = Gaussian(0.0, σ)

# g(x) = e^{-(x-μ)^2/2σ^2} / σ√2π 
evalf(x, GF::Gaussian) = exp(-(x - GF.μ)^2 / (2GF.σ^2)) / (sqrt(2pi) * GF.σ)

"""
Fermi dirac smearing
"""
struct FermiDirac{T<:Real} <: SmearFunction
    μ::T
    β::T
    function FermiDirac(μ::T, β::T) where {T}
        if β == Inf
            error("Fermi dirac smearing only supports non-zero temperature.")
        end
        new{T}(μ, β)
    end
end

function FermiDirac(μ::T1, β::T2) where {T1, T2}
    T = promote_type(T1, T2)
    FermiDirac(T(μ), T(β))
end

# f_{β,μ}(x) = 1/(1+exp(β*(x-μ)))
evalf(x, FD::FermiDirac) = 1.0 / (1.0 + exp(FD.β * (x - FD.μ)))
