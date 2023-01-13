# generate rho by Chebyshev method and Stochastic method

using Distributions
using KrylovKit
using FFTW

"""Generate density by Chebyshev method
M : The order of the Chebyshev polynomial
"""
function rhoGenChebP(ham, model, εF::Float64, M::Int64)
    β = 1 / model.temperature
    filled_occ = DFTK.filled_occupation(model)

    occ_ChebP = []
    ψ = []
    for k = 1 :  length(ham.basis.kpoints)
        push!(occ_ChebP, Vector{Float64}) 
        npw = size(ham.blocks[k], 1)
        occ_ChebP[k] = filled_occ * ones(npw)
        # 1(spinless) or 2(nonespin)
        H = Matrix(ham.blocks[k])
        Emin, Umin = eigsolve(H, 1, :SR);
        Emax, Umax = eigsolve(H, 1, :LR);
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2
        H = (H - E1 * I) ./ E2   # scale H
        FD2 = FermiDirac((εF - E1) / E2, β * E2)
    
        ChebP = ChebyshevP(M, FD2)
        cf = ChebP.coef

        u2 = 2.0 .* (H * H) - I
        z = cf[3] .* u2 + cf[2] .* H + cf[1] .* I
        u0 = copy(H)
        u1 = copy(u2)
        for k = 4 : M + 1
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. z += cf[k] * u2
    
            u0 = u1
            u1 = copy(u2)
        end
        push!(ψ, z) 
    end
    return ψ, occ_ChebP, DFTK.compute_density(ham.basis, ψ, occ_ChebP)
    #compute_density_sdft(ham.basis, ψ, occ_ChebP)
end 

"""Generate density by Stochastic method
M : The order of the Chebyshev polynomial
Ns : The number of Stochastic vectors
"""

function rhoGenStoc(ham, model, εF::Float64, M::Int, Ns::Int64) 
    β = 1 / model.temperature
    filled_occ = DFTK.filled_occupation(model)

    occ_sdft = []
    ψ = []
    for k = 1 : length(ham.basis.kpoints)
        push!(occ_sdft, Vector{Float64}) 
        npw = size(ham.blocks[k], 1)
        occ_sdft[k] = filled_occ * ones(Ns)  # We consider spin_polarization =:none, the filled occupation is 2, if spin_polarization =:spinless, the filled occupation is 1.
        H = Matrix(ham.blocks[k])
        Emin, Umin = eigsolve(H, 1, :SR);
        Emax, Umax = eigsolve(H, 1, :LR);
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2
        H = (H - E1 * I) ./ E2   # scale H
        FD2 = FermiDirac((εF - E1) / E2, β * E2)
        
        ChebP = ChebyshevP(M, FD2)
        cf = ChebP.coef

        d = Uniform(0, 2pi)
        u0 = exp.(im .* rand(d, npw, Ns))
        u1 = H * u0
        z = cf[1] .* u0 + cf[2] .* u1
        for m = 3 : M + 1
            u2 = H * u1
            @. u2 = 2.0 * u2 - u0
            @. z += cf[m] * u2
        
            u0 = u1
            u1 = u2
        end
        push!(ψ, z) 
    end      
    return ψ, occ_sdft, DFTK.compute_density(ham.basis, ψ, occ_sdft)./ Ns
end

struct ChebyshevP
    M::Int64
    coef::Array{Float64,1}
end

struct FermiDirac 
    μ::Float64
    β::Float64
end

function ChebyshevCoef(M::Int64, FD::FermiDirac)
    # f^{1/2}(x) = \sum_{n = 0}^{M}a_n*T_n(x) 
    f(x) = evaluateDos(FD, x)

    Npt = 2 * M # Half of the number of integration points
    pt = collect(range(0, 2pi - pi / Npt, length=2Npt))
    fv = @. sqrt(f(cos(pt)))
    coefft = real.(fft(fv)) ./ (2Npt)
    coef = 2 .* ones(M + 1)
    coef[1] -= 1.0
    @. coef = coef * coefft[1 : M+1]

    return ChebyshevP(M, coef)
end
ChebyshevP(M::Int64, FD::FermiDirac) = ChebyshevCoef(M, FD)

function evaluateDos(FD::FermiDirac, x)
    μ = FD.μ
    β = FD.β
    return 1.0 / (1.0 + exp(β * (x - μ)))
end

#= 
TODO: use the FermiDirac stuct in DFTK 
; smearing = basis.model.smearing 
occupation(S::FermiDirac, x) = 1 / (1 + exp(x))
Smearing.occupation.(smearing, (εk .- εF) .* inverse_temperature)
=#

# here, all DFTK. no nedded finally
# If have error 
using MPI
mpi_sum!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr,   +, comm)