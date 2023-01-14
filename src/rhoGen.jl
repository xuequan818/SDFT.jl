# generate rho by Chebyshev method and Stochastic method

using Distributions
using KrylovKit
using FFTW

abstract type StocDensity end

"""Generate density by Chebyshev method
M : The order of the Chebyshev polynomial
"""
struct rhoCheb <: StocDensity
    M::Int64

end

function rhoGen(ham::Hamiltonian, model::Model, εF::Float64, rhoG::rhoCheb)
    M = rhoG.M
    β = 1 / model.temperature
    filled_occ = DFTK.filled_occupation(model)
    kpoints = ham.basis.kpoints
    occ_ChebP = Vector{Any}(undef, length(kpoints))
    ψ = Vector{Any}(undef, length(kpoints))

    for k = 1:length(kpoints)
        hamk = ham.blocks[k]
        npw = length(G_vectors(ham.basis, kpoints[k]))
        occ_ChebP[k] = filled_occ .* ones(npw)

        # find the bound of eigenvalues
        H = Matrix(hamk)
        Emin, Umin = eigsolve(H, 1, :SR)
        Emax, Umax = eigsolve(H, 1, :LR)
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2

        FD_s = FermiDirac((εF - E1) / E2, β * E2)
        ChebP = ChebyshevP(M, FD_s)
        cf = ChebP.coef

        u0 = Matrix{ComplexF64}(I, npw, npw)
        u1 = copy(u0)
        u2 = copy(u0)

        # In the test stage, we don't use Matrix Free
        # mul!(u1, hamk, u0)
        mul!(u1, H, u0)
        @. u1 = (u1 - E1 * u0) / E2
        z = cf[1] .* u0 + cf[2] .* u1
        for l = 3:M+1
            #mul!(u2, hamk, u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
            @. z += cf[l] * u2

            u0 = u1
            u1 = copy(u2)
        end
        ψ[k] = z
    end
    return ψ, occ_ChebP, DFTK.compute_density(ham.basis, ψ, occ_ChebP)
    #compute_density_sdft(ham.basis, ψ, occ_ChebP)
end

"""Generate density by Stochastic method
M : The order of the Chebyshev polynomial
Ns : The number of Stochastic vectors
"""
struct rhoStoc <: StocDensity
    M::Int64
    Ns::Int64
end

function rhoGen(ham::Hamiltonian, model::Model, εF::Float64, rhoG::rhoStoc)
    M = rhoG.M
    Ns = rhoG.Ns
    β = 1 / model.temperature
    filled_occ = DFTK.filled_occupation(model)
    kpoints = ham.basis.kpoints
    occ_sdft = Vector{Any}(undef, length(kpoints))
    ψ = Vector{Any}(undef, length(kpoints))

    for k = 1:length(kpoints)
        hamk = ham.blocks[k]
        npw = length(G_vectors(ham.basis, kpoints[k]))
        occ_sdft[k] = filled_occ * ones(Ns)  # We consider spin_polarization =:none, the filled occupation is 2, if spin_polarization =:spinless, the filled occupation is 1.

        # find the bound of eigenvalues
        H = Matrix(ham.blocks[k])
        Emin, Umin = eigsolve(H, 1, :SR)
        Emax, Umax = eigsolve(H, 1, :LR)
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2

        FD_s = FermiDirac((εF - E1) / E2, β * E2)
        ChebP = ChebyshevP(M, FD_s)
        cf = ChebP.coef

        d = Uniform(0, 2pi)
        u0 = exp.(im .* rand(d, npw, Ns))
        u1 = copy(u0)
        u2 = copy(u0)

        # In the test stage, we don't use Matrix Free
        # mul!(u1, hamk, u0)
        mul!(u1, H, u0)
        @. u1 = (u1 - E1 * u0) / E2
        z = cf[1] .* u0 + cf[2] .* u1
        for l = 3:M+1
            #mul!(u2, hamk, u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
            @. z += cf[l] * u2

            u0 = u1
            u1 = copy(u2)
        end
        ψ[k] = z
    end
    return ψ ./ Ns, occ_sdft, DFTK.compute_density(ham.basis, ψ, occ_sdft) ./ Ns
end
