
abstract type DosFunction end
"""
Fermi-Dirac distribution
μ : chemical potential
β : inverse temperature
"""
struct FermiDirac
    μ::Float64
    β::Float64
end

function evaluateDos(FD::FermiDirac, x)
    μ = FD.μ
    β = FD.β
    return 1.0 / (1.0 + exp(β * (x - μ)))
end

function evaluateDos(FD::FermiDirac, x::Vector)
    μ = FD.μ
    β = FD.β
    y = @. 1.0 / (1.0 + exp(β * (x - μ)))
    return y
end

"""
Chebyshev polynomial
M :: the M-th Chebyshev polynomial
"""
struct ChebyshevP
    M::Int64
    coef::Array{Float64,1}
end

#use Jackson Damping
function ChebyshevCoef(M::Int64, FD::FermiDirac)
    # f^{1/2}(x) = \sum_{n = 0}^{M}a_n*T_n(x) 
    f(x) = evaluateDos(FD, x)

    Npt = 2 * M # Half of the number of integration points
    pt = collect(range(0, 2pi - pi / Npt, length=2Npt))
    fv = @. sqrt(f(cos(pt)))
    coefft = real.(fft(fv)) ./ (2Npt)
    coef = 2 .* ones(M + 1)
    coef[1] -= 1.0
    @. coef = coef * coefft[1:M+1]
	
	# Jackson Damping
	aM = pi / (M + 2)
	g(m) = ((1 - m / (M + 2))sin(aM)cos(m * aM) + (1 / (M + 2))cos(aM)sin(m * aM)) / sin(aM)
	@. coef = coef * g(0:M)	

    return ChebyshevP(M, coef)
end
ChebyshevP(M::Int64, FD::FermiDirac) = ChebyshevCoef(M, FD)


function evaluateDos(FD::FermiDirac, x)
    μ = FD.μ
    β = FD.β
    return 1.0 / (1.0 + exp(β * (x - μ)))
end


