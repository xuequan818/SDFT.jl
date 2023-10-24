
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

function evaluateEnergy(FD::FermiDirac, x::Vector)
    μ = FD.μ
    β = FD.β
    y = @. 1.0 / (1.0 + exp(β * (x - μ)))
    
    E = sum(y .* x)
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
    if FD.β > 1e2
        aM = pi / (M + 2)
        g(m) = ((1 - m / (M + 2))sin(aM)cos(m * aM) + (1 / (M + 2))cos(aM)sin(m * aM)) / sin(aM)
        @. coef = coef * g(0:M)	
    end

    return ChebyshevP(M, coef)
end
ChebyshevP(M::Int64, FD::FermiDirac) = ChebyshevCoef(M, FD)


function evaluateDos(FD::FermiDirac, x)
    μ = FD.μ
    β = FD.β
    return 1.0 / (1.0 + exp(β * (x - μ)))
end

# Compute the total energy of a given density
function Etot(ρ::Array{Float64,4}, basis)
    ham = Hamiltonian(basis; ρ=ρ)
    eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, 30; ψguess=nothing)
    occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)
    ψ = eigres.X
    eigenvalues = eigres.λ
    energies, _ = energy_hamiltonian(basis, ψ, occupation; ρ=ρ, eigenvalues=eigenvalues, εF=εF)

    E = energies.total
end
