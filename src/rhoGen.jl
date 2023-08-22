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

        # In the test stage, we don't use Matrix Free
        u1 = zero(u0)
        #mul!(u1, hamk, u0)
        mul!(u1, H, u0)
        @. u1 = (u1 - E1 * u0) / E2
        z = cf[1] .* u0 + cf[2] .* u1
        for l = 3:M+1
            u2 = zero(u0)
            #mul!(u2,hamk,u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
            @. z += cf[l] * u2

            @. u0 = u1
            @. u1 = u2
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

        d = Uniform(0, 2pi)
        u0 = exp.(im .* rand(d, npw, Ns))

        # In the test stage, we don't use Matrix Free
        u1 = zero(u0)
        #mul!(u1, hamk, u0)
        mul!(u1, H, u0)
        @. u1 = (u1 - E1 * u0) / E2
        z = cf[1] .* u0 + cf[2] .* u1 # z = P(H)*I
        for l = 3:M+1
            u2 = zero(u0)
            #mul!(u2,hamk,u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
            @. z += cf[l] * u2

            @. u0 = u1
            @. u1 = u2
        end
        ψ[k] = z
    end
    return ψ ./ Ns, occ_sdft, DFTK.compute_density(ham.basis, ψ, occ_sdft) ./ Ns
end

"""Generate density by tempering Stochastic method
nL : The number of levels
β : temperature
M : The order of the Chebyshev polynomial
Ns : The number of Stochastic vectors
"""
struct rhoTStoc <: StocDensity
    nL::Int64
    β::Vector{Float64}
    M::Vector{Int64}
    Ns::Vector{Int64}
end

function rhoGen(ham::Hamiltonian, model::Model, εF::Float64, rhoG::rhoTStoc)
    M = rhoG.M
    Ns = rhoG.Ns
    β = rhoG.β
    nL = rhoG.nL

    tCP = Vector{Any}(undef, nL)
    filled_occ = DFTK.filled_occupation(model)
    kpoints = ham.basis.kpoints
    occ_sdft = Matrix{Any}(undef, length(kpoints), nL)
    ψ = Matrix{Any}(undef, length(kpoints), 2nL - 1)
    d = Uniform(0, 2pi)

    β_s = zeros(nL)
    for k = 1:length(kpoints)
        hamk = ham.blocks[k]
        npw = length(G_vectors(ham.basis, kpoints[k]))

        # find the bound of eigenvalues
        H = Matrix(hamk)
        Emin, Umin = eigsolve(H, 1, :SR)
        Emax, Umax = eigsolve(H, 1, :LR)
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2

        @. β_s = β * E2
        for l = 1:nL
            FD = FermiDirac((εF - E1) / E2, β_s[l])
            tCP[l] = ChebyshevP(M[l], FD)
            occ_sdft[k, l] = filled_occ * (l < nL ? ones(Ns[l]) : ones(npw))
            #occ_sdft[k, l] = filled_occ * ones(Ns[l])
        end

        for l = 1:nL-1
            cf1 = tCP[l].coef
            cf2 = tCP[l+1].coef
            M1 = tCP[l].M
            M2 = tCP[l+1].M
            u0 = exp.(im .* rand(d, npw, Ns[l]))
            # In the test stage, we don't use Matrix Free
            u1 = zero(u0)
            #mul!(u1, hamk, u0)
            mul!(u1, H, u0)
            @. u1 = (u1 - E1 * u0) / E2
            z1 = cf1[1] .* u0 + cf1[2] .* u1
            z2 = cf2[1] .* u0 + cf2[2] .* u1
            for m = 3:M2+1
                u2 = zero(u0)
                #mul!(u2,hamk,u1)
                mul!(u2, H, u1)
                @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
                @. z1 += cf1[m] * u2
                @. z2 += cf2[m] * u2

                @. u0 = u1
                @. u1 = u2
            end
            ψ[k, 2l] = z2 ./ sqrt(Ns[l])

            for m = M2+2:M1+1
                u2 = zero(u0)
                #mul!(u2,hamk,u1)
                mul!(u2, H, u1)
                @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
                @. z1 += cf1[m] * u2

                @. u0 = u1
                @. u1 = u2
            end
            ψ[k, 2l-1] = z1 ./ sqrt(Ns[l])
        end

        cf3 = tCP[nL].coef
        M3 = tCP[nL].M
        #u0 = exp.(im .* rand(d, npw, Ns[nL]))
        u0 = Matrix{ComplexF64}(I, npw, npw)
        # In the test stage, we don't use Matrix Free
        u1 = zero(u0)
        #mul!(u1, hamk, u0)
        mul!(u1, H, u0)
        @. u1 = (u1 - E1 * u0) / E2
        z = cf3[1] .* u0 + cf3[2] .* u1
        for m = 3:M3+1
            u2 = zero(u0)
            #mul!(u2,hamk,u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
            @. z += cf3[m] * u2

            @. u0 = u1
            @. u1 = u2
        end
        ψ[k, end] = z
    end

    ρ = DFTK.compute_density(ham.basis, ψ[:, end], occ_sdft[:, nL]) #./ Ns[nL]
    for l = 1:nL-1
        ρ1 = DFTK.compute_density(ham.basis, ψ[:, 2l-1], occ_sdft[:, l]) 
        ρ2 = DFTK.compute_density(ham.basis, ψ[:, 2l], occ_sdft[:, l]) 
        ρ += (ρ1 - ρ2)
    end

    return ψ[:, 1] ./ Ns[1], occ_sdft, ρ
end

"""Generate density by energy cutoff Stochastic method
M : The order of the Chebyshev polynomial
nL : The number of levels
cut : cutoff
Ns : The number of Stochastic vectors
"""
struct rhoCutStoc <: StocDensity
    M::Int64
    nL::Int64
    cut::Vector{Float64}
    Ns::Vector{Int64}
    hamL::Vector{Hamiltonian}
    same_ind::Matrix{Vector{Int64}}
    full_ind::Matrix{Vector{Int64}}
    dofL::Matrix{Int64}
end

function HamCut(M::Int64, cut::Vector{Float64}, Ns::Vector{Int64}, model::Model; kgrid=kgrid_from_minimal_spacing(model, 2π * 0.022))

    nL = length(cut)
    hamL = Vector{Hamiltonian}(undef, nL)

    for i = 1:nL
        basis = PlaneWaveBasis(model; Ecut=cut[i], kgrid)
        ρin = guess_density(basis)
        hamL[i] = Hamiltonian(basis; ρ=ρin)
    end

    lk = length(hamL[1].basis.kpoints)
    same_ind = Matrix{Vector{Int64}}(undef, lk, nL - 1)
    full_ind = Matrix{Vector{Int64}}(undef, lk, nL)
    dofL = ones(Int64, lk, nL)

    for k = 1:lk
        Gv = G_vectors(hamL[1].basis, hamL[1].basis.kpoints[k])
        full_ind[k,1] = collect(1:length(Gv))
        for i = 1:nL-1
            Gv1 = G_vectors(hamL[i].basis, hamL[i].basis.kpoints[k])
            Gv2 = G_vectors(hamL[i+1].basis, hamL[i+1].basis.kpoints[k])
            dofL[k, i] = length(Gv1)
            if i == nL - 1
                dofL[k, i+1] = length(Gv2)
            end

            same_ind[k, i] = map(x -> findall(Gv2[x], Gv1, Levenshtein(); min_score=1.0)[1], 1:length(Gv2))
            full_ind[k, i+1] = map(x -> findall(Gv2[x], Gv, Levenshtein(); min_score=1.0)[1], 1:length(Gv2))
        end
    end

    return rhoCutStoc(M, nL, cut, Ns, hamL, same_ind, full_ind, dofL)
end

rhoCutStoc(M::Int64, cut::Vector{Float64}, Ns::Vector{Int64}, model::Model; kgrid=kgrid_from_minimal_spacing(model, 2π * 0.022)) = HamCut(M, cut, Ns, model; kgrid=kgrid)

function rhoGen(ham::Hamiltonian, model::Model, εF::Float64, rhoG::rhoCutStoc)
    M = rhoG.M
    β = 1 / model.temperature
    Ns = rhoG.Ns
    nL = rhoG.nL
    hamL = rhoG.hamL
    same_ind = rhoG.same_ind
    full_ind = rhoG.full_ind
    dofL = rhoG.dofL

    filled_occ = DFTK.filled_occupation(model)
    kpoints = ham.basis.kpoints
    occ_sdft = Matrix{Any}(undef, length(kpoints), nL)
    ψ = Matrix{Any}(undef, length(kpoints), 2nL - 1)
    d = Uniform(0, 2pi)

    for k = 1:length(kpoints)

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

        for l = 1:nL-1
            occ_sdft[k, l] = ones(Ns[l])
        end
        occ_sdft[k, nL] = ones(dofL[k, nL])
        @. occ_sdft = filled_occ * occ_sdft

        for l = 1:nL-1
            z_full1 = zeros(ComplexF64, dofL[k, 1], Ns[l])
            #z_full1 = zeros(ComplexF64, dofL[k, l], dofL[k, l])
            z_full2 = copy(z_full1)
            H_l1 = Matrix(hamL[l].blocks[k])
            H_l2 = Matrix(hamL[l+1].blocks[k])
            ind = same_ind[k, l]
            u0_l1 = exp.(im .* rand(d, dofL[k, l], Ns[l]))
            #u0_l1 = Matrix{ComplexF64}(I, dofL[k, l], dofL[k, l])
            u0_l2 = copy(u0_l1[ind,:])
            # In the test stage, we don't use Matrix Free
            u1_l1 = zero(u0_l1)
            u1_l2 = zero(u0_l2)
            #mul!(u1, hamk, u0)
            mul!(u1_l1, H_l1, u0_l1)
            mul!(u1_l2, H_l2, u0_l2)
            @. u1_l1 = (u1_l1 - E1 * u0_l1) / E2
            @. u1_l2 = (u1_l2 - E1 * u0_l2) / E2
            z_l1 = cf[1] .* u0_l1 + cf[2] .* u1_l1
            z_l2 = cf[1] .* u0_l2 + cf[2] .* u1_l2
            for m = 3:M+1
                u2_l1 = zero(u0_l1)
                u2_l2 = zero(u0_l2)
                #mul!(u2,hamk,u1)
                mul!(u2_l1, H_l1, u1_l1)
                mul!(u2_l2, H_l2, u1_l2)
                @. u2_l1 = 2.0 * ((u2_l1 - E1 * u1_l1) / E2) - u0_l1
                @. u2_l2 = 2.0 * ((u2_l2 - E1 * u1_l2) / E2) - u0_l2
                @. z_l1 += cf[m] * u2_l1
                @. z_l2 += cf[m] * u2_l2

                @. u0_l1 = u1_l1
                @. u1_l1 = u2_l1
                @. u0_l2 = u1_l2
                @. u1_l2 = u2_l2
            end
            z_full1[full_ind[k, l], :] = z_l1 ./ sqrt(Ns[l])
            ψ[k, 2l-1] = z_full1
            z_full2[full_ind[k, l+1], :] = z_l2 ./ sqrt(Ns[l])
            ψ[k, 2l] = z_full2
        end

        z_full = zeros(ComplexF64, dofL[k, 1], dofL[k, nL])
        H = hamL[nL].blocks[k]
        #u0 = exp.(im .* rand(d, dofL[k,nl], Ns[nL]))
        u0 = Matrix{ComplexF64}(I, dofL[k, nL], dofL[k, nL])
        # In the test stage, we don't use Matrix Free
        u1 = zero(u0)
        #mul!(u1, hamk, u0)
        mul!(u1, H, u0)
        @. u1 = (u1 - E1 * u0) / E2
        z = cf[1] .* u0 + cf[2] .* u1
        for m = 3:M+1
            u2 = zero(u0)
            #mul!(u2,hamk,u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * ((u2 - E1 * u1) / E2) - u0
            @. z += cf[m] * u2

            @. u0 = u1
            @. u1 = u2
        end
        z_full[full_ind[k, nL], :] = z
        ψ[k, end] = z_full
    end

    ρ = DFTK.compute_density(ham.basis, ψ[:, end], occ_sdft[:, nL]) #./ Ns[nL]
    for l = 1:nL-1
        ρ1 = DFTK.compute_density(ham.basis, ψ[:, 2l-1], occ_sdft[:, l])
        ρ2 = DFTK.compute_density(ham.basis, ψ[:, 2l], occ_sdft[:, l])
        ρ += (ρ1 - ρ2)
    end
    
    return ψ[:, 1] ./ Ns[1], occ_sdft, ρ
end

